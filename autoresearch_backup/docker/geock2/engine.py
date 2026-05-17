"""
engine.py — GEOCK/DNBAP Docking Engine
Main entry point. Wires all 5 stages into one clean call.

Usage:
    from geock.engine import GEOCKEngine
    from geock.config import DockingConfig

    cfg = DockingConfig(
        receptor_path = "receptor.pdbqt",
        ligand_path   = "ligand.pdbqt",
        box_center    = (12.5, -3.2, 8.1),   # from fpocket
    )

    engine = GEOCKEngine.from_weights("weights/")
    result = engine.dock(cfg, n_torsions=7)

    for pose in result.top_poses[:5]:
        print(f"Rank {pose.rank}: {pose.ensemble_score:.3f} kcal/mol")

Pre-training (run once before any docking):
    python -m geock.pretrain.pretrain --pdbind_dir /data/PDBbind --output_dir weights/
"""

import torch
import time
import os
from dataclasses import dataclass
from typing import Optional, Callable, List

from geock.config import DockingConfig, POSE_DIM
from geock.core.som      import BindingModeSOM
from geock.core.hopfield  import HopfieldBindingMemory
from geock.core.vae       import PoseVAE
from geock.core.scoring   import EnsembleScorer, ScoredPose

from geock.pipeline.stage1_sampling import run as stage1_run, Stage1Result
from geock.pipeline.stage2_cluster  import run as stage2_run, Stage2Result
from geock.pipeline.stage3_generate import run as stage3_run, Stage3Result
from geock.pipeline.stage4_refine   import run as stage4_run, Stage4Result
from geock.pipeline.stage5_score    import run as stage5_run, Stage5Result


@dataclass
class DockingResult:
    """Full result from one docking run."""
    top_poses:    List[ScoredPose]  # ranked list, rank-1 is best
    best_score:   float             # ensemble score of top pose (kcal/mol)
    best_vina:    float             # Vina score of top pose

    # Stage diagnostics (for paper reporting)
    stage1: Stage1Result
    stage2: Stage2Result
    stage3: Stage3Result
    stage4: Stage4Result
    stage5: Stage5Result

    total_runtime_s: float

    def to_delta_g(self) -> float:
        """
        Calibrate ensemble score to ΔG using linear regression.
        Fit from 15 CASF compounds:
        Score range: [-0.418, -0.075] → ΔG [-12.1, -7.8]
        """
        slope = 8.8552
        intercept = -8.2906
        return round(slope * self.best_score + intercept, 2)

    def summary(self) -> str:
        lines = [
            "=" * 55,
            "  GEOCK/DNBAP Docking Result",
            "=" * 55,
            f"  Top-1 ensemble score : {self.best_score:.3f} kcal/mol",
            f"  Top-1 Vina score     : {self.best_vina:.3f} kcal/mol",
            f"  Calibrated ΔG        : {self.to_delta_g():.2f} kcal/mol",
            f"  Total runtime        : {self.total_runtime_s:.2f}s",
            f"  Poses sampled (S1)   : {self.stage1.n_poses}",
            f"  Novel poses (S3)     : {self.stage3.n_novel} "
                f"({self.stage3.novelty_rate:.0%} novelty rate)",
            f"  SOM QE after finetune: {self.stage2.quantization_error:.4f}",
            f"  Active SOM neurons   : {self.stage2.n_active_neurons}/64",
            "-" * 55,
            "  Top-5 poses:",
        ]
        for p in self.top_poses[:5]:
            lines.append(
                f"    Rank {p.rank:2d} | ensemble={p.ensemble_score:7.3f} | "
                f"vina={p.vina_score:7.3f} | "
                f"hopfield_sim={p.hopfield_bonus:.3f}"
            )
        lines.append("=" * 55)
        return "\n".join(lines)


class GEOCKEngine:
    """
    GEOCK/DNBAP Neural-Hybrid Docking Engine.

    Loads pre-trained weights once. Docks any number of ligands.
    All neural weights are frozen at inference — only SOM is fine-tuned
    per pocket (10 epochs, < 1s).
    """

    def __init__(
        self,
        som:      BindingModeSOM,
        vae:      PoseVAE,
        hopfield: HopfieldBindingMemory,
        scorer:   EnsembleScorer,
    ):
        self.som      = som
        self.vae      = vae
        self.hopfield = hopfield
        self.scorer   = scorer

        # Freeze VAE and Hopfield (never retrained at inference — BUG 3 fix)
        for p in self.vae.parameters():
            p.requires_grad_(False)

    @classmethod
    def from_weights(cls, weights_dir: str) -> "GEOCKEngine":
        """
        Load pre-trained weights from directory.
        Raises FileNotFoundError with helpful message if weights missing.
        """
        paths = {
            "som"     : os.path.join(weights_dir, "som_weights.pt"),
            "vae"     : os.path.join(weights_dir, "vae_weights.pt"),
            "hopfield": os.path.join(weights_dir, "hopfield_memories.pt"),
            "scorer"  : os.path.join(weights_dir, "scorer_weights.pt"),
        }

        missing = [k for k, v in paths.items() if not os.path.exists(v)]
        if missing:
            raise FileNotFoundError(
                f"Missing pre-trained weight files: {missing}\n"
                f"Run: python -m geock.pretrain.pretrain "
                f"--pdbind_dir /path/to/PDBbind --output_dir {weights_dir}"
            )

        som      = BindingModeSOM.load(paths["som"])
        vae      = PoseVAE.load(paths["vae"])
        hopfield = HopfieldBindingMemory.load(paths["hopfield"])
        scorer   = EnsembleScorer.load(paths["scorer"])

        print(f"[GEOCK] Weights loaded from {weights_dir}")
        return cls(som, vae, hopfield, scorer)

    @classmethod
    def untrained(cls) -> "GEOCKEngine":
        """
        Build engine with fresh (untrained) weights.
        Used for development / testing before PDBbind pre-training.
        Performance will be poor — this is expected.
        """
        print("[GEOCK] WARNING: Using untrained weights. "
              "Run PDBbind pre-training for real performance.")
        return cls(
            som      = BindingModeSOM(),
            vae      = PoseVAE(),
            hopfield = HopfieldBindingMemory(),
            scorer   = EnsembleScorer(),
        )

    def dock(
        self,
        config:     DockingConfig,
        n_torsions: int,
        score_fn:   Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        seed:       int = 42,
    ) -> DockingResult:
        """
        Run the full 5-stage docking pipeline.

        Args:
            config     : DockingConfig — must have box_center set
            n_torsions : number of rotatable bonds in ligand
            score_fn   : optional real Vina scorer:
                         (poses: Tensor[N,24]) → scores Tensor[N]
                         If None, stub scores drive the search.
            seed       : random seed for reproducibility

        Returns:
            DockingResult
        """
        config.validate()   # hard error if anything is wrong

        t_total = time.time()

        if config.verbose:
            print("\n" + "=" * 55)
            print("  GEOCK/DNBAP Docking Engine")
            print("=" * 55)

        # ── Stage 1: MC Sampling (Rust) ───────────────────────
        s1 = stage1_run(config, n_torsions, score_fn=score_fn, seed=seed)

        # ── Stage 2: SOM Clustering ───────────────────────────
        # Deep-copy SOM weights before fine-tuning so original is preserved
        import copy
        pocket_som = copy.deepcopy(self.som)
        s2 = stage2_run(s1, config, pocket_som)

        # ── Stage 3: VAE Generation + Hopfield Filter ─────────
        s3 = stage3_run(s2, config, self.vae, self.hopfield)

        # ── Stage 4: SOM-Neighborhood MC Refinement (Rust) ────
        s4 = stage4_run(s3, s2, config, pocket_som, seed=seed + 1)

        # ── Stage 5: Ensemble Scoring ─────────────────────────
        s5 = stage5_run(s4, s1, config, self.scorer, score_fn=score_fn)

        total_runtime = time.time() - t_total

        result = DockingResult(
            top_poses       = s5.top_poses,
            best_score      = s5.best_score,
            best_vina       = s5.top_poses[0].vina_score if s5.top_poses else float("inf"),
            stage1          = s1,
            stage2          = s2,
            stage3          = s3,
            stage4          = s4,
            stage5          = s5,
            total_runtime_s = total_runtime,
        )

        if config.verbose:
            print(result.summary())

        return result


# ------------------------------------------------------------------
# Quick end-to-end smoke test
# ------------------------------------------------------------------

if __name__ == "__main__":
    print("=== GEOCKEngine End-to-End Smoke Test ===\n")
    import tempfile, os

    torch.manual_seed(42)

    # Build config
    cfg = DockingConfig()
    with tempfile.NamedTemporaryFile(suffix=".pdbqt", delete=False) as f: cfg.receptor_path = f.name
    with tempfile.NamedTemporaryFile(suffix=".pdbqt", delete=False) as f: cfg.ligand_path = f.name
    cfg.box_center = (5.0, -2.0, 8.0)
    cfg.stage1.n_poses       = 100    # small for smoke test
    cfg.stage3.n_vae_samples = 20
    cfg.stage4.mc_steps      = 50
    cfg.som.finetune_epochs  = 3

    # Pre-populate Hopfield with random patterns so it's not empty
    engine = GEOCKEngine.untrained()
    engine.hopfield.store(torch.randn(50, POSE_DIM))

    # Test 1: full pipeline runs
    result = engine.dock(cfg, n_torsions=5, seed=42)
    assert len(result.top_poses) > 0, "No poses returned"
    assert result.best_score < float("inf"), "Best score is inf"
    print(f"\nPASS: Full pipeline completed in {result.total_runtime_s:.2f}s")

    # Test 2: poses are ranked
    scores = [p.ensemble_score for p in result.top_poses]
    assert all(scores[i] <= scores[i+1] for i in range(len(scores)-1)), \
        "Top poses not sorted"
    print("PASS: output poses are ranked correctly")

    # Test 3: result has all stage data
    assert result.stage1.n_poses == 100
    assert result.stage2.quantization_error > 0
    assert result.stage3.n_generated == 20
    print("PASS: all stage results accessible")

    # Test 4: missing box_center caught before any compute runs
    cfg_bad = DockingConfig()
    cfg_bad.receptor_path = cfg.receptor_path
    cfg_bad.ligand_path   = cfg.ligand_path
    try:
        engine.dock(cfg_bad, n_torsions=5)
        raise AssertionError("Should have raised")
    except ValueError as e:
        assert "box_center" in str(e)
        print("PASS: missing box_center caught before any compute")

    # Test 5: reproducible with same seed
    r2 = engine.dock(cfg, n_torsions=5, seed=42)
    assert result.best_score == r2.best_score, "Not reproducible"
    print("PASS: same seed → same best score")

    os.unlink(cfg.receptor_path)
    os.unlink(cfg.ligand_path)
    print("\n=== ALL TESTS PASSED ===")
    print("\nRun full pipeline summary:")
    print(result.summary())
