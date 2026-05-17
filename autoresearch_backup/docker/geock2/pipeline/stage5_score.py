"""
pipeline/stage5_score.py — Stage 5: Ensemble Scoring + Final Ranking

Combines three signal sources into one final ranked list:
  1. Vina score         (from Stage 1 score_fn, or stub if unavailable)
  2. Learned correction (trained on PDBbind to fix Vina systematic errors)
  3. Hopfield similarity bonus (rewards poses resembling known crystal modes)

Output: top-20 ranked ScoredPose objects ready for output / paper results.
"""

import torch
import time
from dataclasses import dataclass
from typing import Optional, List

from geock.config import DockingConfig, POSE_DIM
from geock.core.scoring import EnsembleScorer, ScoredPose
from geock.pipeline.stage1_sampling import Stage1Result
from geock.pipeline.stage4_refine import Stage4Result


@dataclass
class Stage5Result:
    top_poses:       List[ScoredPose]    # top-K ranked poses
    all_scores:      torch.Tensor        # [K] ensemble scores
    vina_scores:     torch.Tensor        # [K] raw Vina (or stub)
    hopfield_sims:   torch.Tensor        # [K]
    best_score:      float               # ensemble score of rank-1 pose
    runtime_s:       float


def run(
    stage4:  Stage4Result,
    stage1:  Stage1Result,
    config:  DockingConfig,
    scorer:  EnsembleScorer,
    score_fn = None,     # optional: real Vina scorer callable
) -> Stage5Result:
    """
    Run Stage 5: ensemble scoring and final ranking.

    Args:
        stage4  : refined poses from Stage 4
        stage1  : original MC results (for Vina score fallback)
        config  : DockingConfig
        scorer  : pre-trained EnsembleScorer
        score_fn: optional callable for real Vina scoring of Stage4 poses

    Returns:
        Stage5Result with ranked ScoredPose list
    """
    poses        = stage4.refined_poses    # [K, 24]
    hopfield_sims = stage4.hopfield_sims   # [K]
    K            = len(poses)

    if config.verbose:
        print(f"[Stage5] Ensemble scoring {K} refined poses")

    t0 = time.time()

    # Get Vina scores: real if score_fn available, else stub
    if score_fn is not None:
        if config.verbose:
            print("[Stage5] Running real Vina scoring on refined poses...")
        vina_scores = score_fn(poses)   # [K]
        assert vina_scores.shape == (K,)
    else:
        # Use refined stub scores as Vina proxy
        vina_scores = stage4.refined_scores
        if config.verbose:
            print("[Stage5] No score_fn provided — using stub scores as Vina proxy")

    # Ensemble scoring
    top_k   = config.stage5.top_k
    results = scorer.score_and_package(
        pose_vectors  = poses,
        vina_scores   = vina_scores,
        hopfield_sims = hopfield_sims,
        top_k         = min(top_k, K),
    )

    all_scores = torch.tensor([r.ensemble_score for r in results])
    runtime    = time.time() - t0

    if config.verbose:
        print(
            f"[Stage5] Done in {runtime:.2f}s | "
            f"top-1 ensemble={results[0].ensemble_score:.3f} | "
            f"top-1 vina={results[0].vina_score:.3f}"
        )
        if scorer.is_trained:
            print(f"[Stage5] {scorer.weight_summary()}")

    return Stage5Result(
        top_poses     = results,
        all_scores    = all_scores,
        vina_scores   = vina_scores,
        hopfield_sims = hopfield_sims,
        best_score    = results[0].ensemble_score if results else float("inf"),
        runtime_s     = runtime,
    )


if __name__ == "__main__":
    print("=== Stage5 Unit Tests ===\n")
    import tempfile, os
    from geock.pipeline.stage4_refine import Stage4Result
    from geock.pipeline.stage1_sampling import Stage1Result

    torch.manual_seed(42)

    K = 20
    mock_stage4 = Stage4Result(
        refined_poses  = torch.randn(K, POSE_DIM),
        refined_scores = -torch.rand(K) * 10,
        hopfield_sims  = torch.rand(K),
        runtime_s = 0.1, used_rust = True,
    )
    mock_stage1 = Stage1Result(
        pose_vectors = torch.randn(200, POSE_DIM),
        stub_scores  = -torch.rand(200) * 10,
        vina_scores  = None,
        n_poses=200, runtime_s=0.5, used_rust=True,
    )

    cfg = DockingConfig()
    with tempfile.NamedTemporaryFile(suffix=".pdbqt", delete=False) as f: cfg.receptor_path = f.name
    with tempfile.NamedTemporaryFile(suffix=".pdbqt", delete=False) as f: cfg.ligand_path = f.name
    cfg.box_center = (0.0, 0.0, 0.0)

    scorer = EnsembleScorer()

    # Test 1: runs end to end
    result = run(mock_stage4, mock_stage1, cfg, scorer)
    assert len(result.top_poses) == min(cfg.stage5.top_k, K)
    assert result.all_scores.shape[0] == min(cfg.stage5.top_k, K)
    print(f"PASS: Stage5 returns {len(result.top_poses)} ranked poses | "
          f"best={result.best_score:.3f}")

    # Test 2: poses are sorted (best first = lowest score)
    scores = [p.ensemble_score for p in result.top_poses]
    assert all(scores[i] <= scores[i+1] for i in range(len(scores)-1)), \
        "Poses not sorted by score"
    print("PASS: poses sorted ascending by ensemble score")

    # Test 3: ranks are 1-indexed sequentially
    assert result.top_poses[0].rank == 1
    assert result.top_poses[-1].rank == len(result.top_poses)
    print("PASS: ranks are sequential 1-indexed")

    # Test 4: score_fn callback overrides stub scores
    def mock_vina(poses): return -torch.rand(len(poses)) * 15
    result2 = run(mock_stage4, mock_stage1, cfg, scorer, score_fn=mock_vina)
    assert result2.vina_scores.min() < -5, "Mock Vina scores not applied"
    print("PASS: score_fn callback overrides stub scores")

    os.unlink(cfg.receptor_path)
    os.unlink(cfg.ligand_path)
    print("\n=== ALL TESTS PASSED ===")
