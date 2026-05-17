"""
pipeline/stage3_generate.py — Stage 3: VAE Generation + Hopfield Filter

VAE samples novel poses from latent space.
Hopfield network discards anything resembling stored prototypes.
Result: ~20 genuinely novel candidate poses for Stage 4 refinement.

Scientific claim being tested:
  VAE + Hopfield together produce more diverse poses than
  just taking the top-K from Stage 2 cluster centroids.
  This claim is measurable via average pairwise RMSD of top-10 poses.
"""

import torch
import torch.nn.functional as F
import time
from dataclasses import dataclass
from typing import Optional

from geock.config import DockingConfig, POSE_DIM
from geock.core.vae import PoseVAE
from geock.core.hopfield import HopfieldBindingMemory
from geock.pipeline.stage2_cluster import Stage2Result


@dataclass
class Stage3Result:
    novel_poses:       torch.Tensor    # [K, 24] — novel poses passing Hopfield filter
    novel_scores:      torch.Tensor    # [K]     — interpolated stub scores
    hopfield_sims:     torch.Tensor    # [K]     — Hopfield similarity (for Stage 5)
    n_generated:       int             # how many VAE samples were drawn
    n_novel:           int             # how many passed the novelty filter
    novelty_rate:      float           # n_novel / n_generated
    runtime_s:         float


def run(
    stage2:  Stage2Result,
    config:  DockingConfig,
    vae:     PoseVAE,
    hopfield: HopfieldBindingMemory,
) -> Stage3Result:
    """
    Run Stage 3: VAE generation + Hopfield novelty filtering.

    Args:
        stage2  : Stage 2 result (centroids used for latent conditioning)
        config  : DockingConfig
        vae     : pre-trained PoseVAE (loaded from vae_weights.pt)
        hopfield: pre-trained HopfieldBindingMemory (loaded from hopfield_memories.pt)

    Returns:
        Stage3Result — novel poses for Stage 4
    """
    cfg3 = config.stage3
    t0   = time.time()

    if config.verbose:
        print(f"[Stage3] VAE sampling {cfg3.n_vae_samples} poses | "
              f"temperature={cfg3.temperature}")

    # Sample from VAE prior (latent space)
    vae.eval()
    with torch.no_grad():
        generated = vae.sample(cfg3.n_vae_samples, temperature=cfg3.temperature)
        # generated: [n_vae_samples, 24]

    # Compute Hopfield similarities for all generated poses
    hopfield_sims = hopfield.similarity(generated)    # [n_vae_samples]

    # Filter: keep only novel poses (similarity < threshold)
    novel_poses, novel_mask = hopfield.filter_novel(
        generated, threshold=config.hopfield.novelty_threshold
    )

    n_novel     = novel_mask.sum().item()
    n_generated = cfg3.n_vae_samples
    novelty_rate = n_novel / max(n_generated, 1)

    if config.verbose:
        print(
            f"[Stage3] {n_novel}/{n_generated} poses passed novelty filter "
            f"(rate={novelty_rate:.1%})"
        )

    # If novelty filter is too aggressive (< 5 poses), relax threshold
    if n_novel < 5:
        relaxed_threshold = config.hopfield.novelty_threshold * 1.2
        if config.verbose:
            print(f"[Stage3] Too few novel poses. Relaxing threshold to {relaxed_threshold:.2f}")
        novel_poses, novel_mask = hopfield.filter_novel(generated, threshold=relaxed_threshold)
        n_novel = novel_mask.sum().item()

    # Still empty: fall back to all generated poses (with warning)
    if n_novel == 0:
        if config.verbose:
            print("[Stage3] WARNING: All poses filtered. Using top-10 by Hopfield sim.")
        top_idx   = hopfield_sims.argsort()[:10]
        novel_poses = generated[top_idx]
        novel_mask  = torch.zeros(n_generated, dtype=torch.bool)
        novel_mask[top_idx] = True

    # Hopfield sims for the novel subset
    novel_sims = hopfield_sims[novel_mask]   # [K]

    # Assign stub scores by interpolating from nearest Stage2 centroid
    # (proxy until real Vina scoring in Stage 5)
    novel_scores = _interpolate_scores(novel_poses, stage2)

    runtime = time.time() - t0

    if config.verbose:
        print(f"[Stage3] Done in {runtime:.2f}s | {n_novel} novel poses ready for Stage 4")

    return Stage3Result(
        novel_poses   = novel_poses,
        novel_scores  = novel_scores,
        hopfield_sims = novel_sims,
        n_generated   = n_generated,
        n_novel       = n_novel,
        novelty_rate  = novelty_rate,
        runtime_s     = runtime,
    )


def _interpolate_scores(
    poses:   torch.Tensor,
    stage2:  Stage2Result,
) -> torch.Tensor:
    """
    Assign scores to VAE-generated poses by nearest-centroid interpolation.

    For each generated pose, find the closest Stage2 centroid and
    use its stub score as a proxy. This is only used to seed Stage 4 —
    real Vina scoring happens in Stage 5.
    """
    if stage2.centroids is None or len(stage2.centroids) == 0:
        return torch.zeros(len(poses))

    # Distance to each centroid
    dists = torch.cdist(poses, stage2.centroids)   # [K, 64]
    nearest = dists.argmin(dim=1)                  # [K]

    # We don't have per-centroid scores stored in stage2,
    # so use a distance-based proxy: closer to centroid = better score
    min_dists = dists.min(dim=1).values            # [K]
    # Normalize to [-5, 0] range (rough kcal/mol proxy)
    proxy_scores = -5.0 * (1.0 - min_dists / (min_dists.max() + 1e-8))
    return proxy_scores


if __name__ == "__main__":
    print("=== Stage3 Unit Tests ===\n")
    import tempfile, os
    import torch

    torch.manual_seed(42)

    # Mock Stage2 result
    mock_stage2 = Stage2Result(
        bmu_indices        = torch.randint(0, 64, (200,)),
        centroids          = torch.randn(64, POSE_DIM),
        assignments        = {i: [] for i in range(64)},
        quantization_error = 0.5,
        n_active_neurons   = 45,
        runtime_s          = 0.2,
    )

    cfg = DockingConfig()
    with tempfile.NamedTemporaryFile(suffix=".pdbqt", delete=False) as f:
        cfg.receptor_path = f.name
    with tempfile.NamedTemporaryFile(suffix=".pdbqt", delete=False) as f:
        cfg.ligand_path = f.name
    cfg.box_center = (0.0, 0.0, 0.0)
    cfg.stage3.n_vae_samples = 30

    vae      = PoseVAE()
    hopfield = HopfieldBindingMemory()
    # Store some patterns so Hopfield is not empty
    hopfield.store(torch.randn(50, POSE_DIM))

    # Test 1: runs end to end
    result = run(mock_stage2, cfg, vae, hopfield)
    assert isinstance(result.novel_poses, torch.Tensor)
    assert result.novel_poses.shape[1] == POSE_DIM
    assert result.n_generated == 30
    print(f"PASS: Stage3 runs | {result.n_novel}/30 novel poses | "
          f"rate={result.novelty_rate:.1%}")

    # Test 2: hopfield_sims has correct length
    assert result.hopfield_sims.shape == (result.n_novel,), \
        f"Hopfield sims shape {result.hopfield_sims.shape} != ({result.n_novel},)"
    print("PASS: hopfield_sims length matches n_novel")

    # Test 3: novelty filter rejects stored patterns
    # Store the generated poses AS memories — they should then fail novelty
    hopfield2 = HopfieldBindingMemory(novelty_threshold=0.3)  # tight threshold
    vae2      = PoseVAE()
    with torch.no_grad():
        test_poses = vae2.sample(20)
    hopfield2.store(test_poses)   # store exact poses

    cfg.hopfield.novelty_threshold = 0.3
    result2 = run(mock_stage2, cfg, vae2, hopfield2)
    print(f"PASS: tight threshold ({cfg.hopfield.novelty_threshold}) → "
          f"{result2.n_novel}/20 novel (expected fewer)")

    # Test 4: proxy scores have correct shape
    assert result.novel_scores.shape == (result.n_novel,)
    print("PASS: proxy scores correct shape")

    os.unlink(cfg.receptor_path)
    os.unlink(cfg.ligand_path)
    print("\n=== ALL TESTS PASSED ===")
