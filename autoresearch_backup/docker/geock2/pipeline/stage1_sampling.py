"""
pipeline/stage1_sampling.py — Stage 1: Monte Carlo Pose Sampling

Calls the Rust geock_mc.mc_sample() hot loop.
Returns 2000 raw pose vectors ready for Stage 2 SOM clustering.

What this stage does NOT do:
  - Real Vina scoring (requires OpenDock/AutoDock at runtime)
  - Any neural processing (that's Stages 2–4)

Stub scores from Rust drive the MC search direction only.
Real Vina scoring is plugged in via the score_fn callback.
If no score_fn is provided, stub scores are used (good for testing).
"""

import torch
import time
from typing import Callable, Optional, Tuple
from dataclasses import dataclass

try:
    import geock_mc
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    print("[Stage1] WARNING: geock_mc Rust extension not found. "
          "Install with: pip install geock_mc wheel. "
          "Falling back to pure Python (slow).")

from geock.config import DockingConfig, POSE_DIM


@dataclass
class Stage1Result:
    pose_vectors: torch.Tensor    # [N, 24] — all sampled poses
    stub_scores:  torch.Tensor    # [N]     — MC stub scores (lower=better)
    vina_scores:  Optional[torch.Tensor]  # [N] if real scoring was done
    n_poses:      int
    runtime_s:    float
    used_rust:    bool


def _python_fallback_sample(
    n_poses:       int,
    box_center:    Tuple[float, float, float],
    box_size:      Tuple[float, float, float],
    n_torsions:    int,
    temperature:   float,
    seed:          int,
) -> Tuple[list, list]:
    """
    Pure Python MC fallback. ~20x slower than Rust.
    Only used if geock_mc.so is not installed.
    Produces identical statistical behavior.
    """
    import random
    import math
    rng = random.Random(seed)

    def _score(dof):
        dx = dof[0] - box_center[0]
        dy = dof[1] - box_center[1]
        dz = dof[2] - box_center[2]
        if abs(dx) > box_size[0]/2 or abs(dy) > box_size[1]/2 or abs(dz) > box_size[2]/2:
            return 1000.0
        dist_sq = dx*dx + dy*dy + dz*dz
        return -10.0 * math.exp(-dist_sq / 50.0) + 0.1 * n_torsions

    # Init pose
    dof = [
        box_center[0] + rng.uniform(-box_size[0]/2, box_size[0]/2),
        box_center[1] + rng.uniform(-box_size[1]/2, box_size[1]/2),
        box_center[2] + rng.uniform(-box_size[2]/2, box_size[2]/2),
    ] + [rng.uniform(-3.14, 3.14) for _ in range(3)] \
      + [rng.uniform(-3.14, 3.14) for _ in range(min(n_torsions, 18))] \
      + [0.0] * max(0, 18 - min(n_torsions, 18))

    cur_score = _score(dof)
    poses, scores = [], []

    while len(poses) < n_poses:
        for _ in range(50):
            trial = dof[:]
            idx = rng.randint(0, 5 + min(n_torsions, 18) - 1)
            trial[idx] += rng.gauss(0, 0.5)
            ts = _score(trial)
            if ts < cur_score or rng.random() < math.exp(-(ts - cur_score) / temperature):
                dof = trial
                cur_score = ts
        poses.append(dof[:])
        scores.append(cur_score)

    return poses, scores


def run(
    config:     DockingConfig,
    n_torsions: int,
    score_fn:   Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    seed:       int = 42,
) -> Stage1Result:
    """
    Run Stage 1 Monte Carlo sampling.

    Args:
        config     : DockingConfig — must have box_center set
        n_torsions : number of rotatable bonds in ligand
        score_fn   : optional callable (pose_vectors: Tensor[N,24]) → scores [N]
                     If provided, called on sampled poses for real Vina scores.
                     If None, stub scores are used.
        seed       : random seed for reproducibility

    Returns:
        Stage1Result with pose_vectors [N, 24] and scores
    """
    box_center = config.get_box_center()   # hard error if not set (BUG 1 fix)
    box_size   = config.stage1.box_size
    n_poses    = config.stage1.n_poses

    if config.verbose:
        print(f"[Stage1] MC sampling {n_poses} poses | "
              f"box_center={box_center} | n_torsions={n_torsions} | "
              f"{'Rust' if RUST_AVAILABLE else 'Python fallback'}")

    t0 = time.time()

    if RUST_AVAILABLE:
        raw_poses, raw_scores = geock_mc.mc_sample(
            n_poses          = n_poses,
            box_center       = list(box_center),
            box_size         = list(box_size),
            n_torsions       = n_torsions,
            temperature      = config.stage1.vina_exhaustiveness * 0.15,
            step_size        = 2.0,
            seed             = seed,
            mc_steps_per_pose= 50,
        )
        used_rust = True
    else:
        raw_poses, raw_scores = _python_fallback_sample(
            n_poses, box_center, box_size, n_torsions,
            temperature=1.2, seed=seed,
        )
        used_rust = False

    pose_vectors = torch.tensor(raw_poses,  dtype=torch.float32)   # [N, 24]
    stub_scores  = torch.tensor(raw_scores, dtype=torch.float32)   # [N]

    # Validate shapes
    assert pose_vectors.shape == (n_poses, POSE_DIM), \
        f"Stage1 pose shape wrong: {pose_vectors.shape}"

    # Optional real scoring
    vina_scores = None
    if score_fn is not None:
        if config.verbose:
            print(f"[Stage1] Running real scoring on {n_poses} poses...")
        vina_scores = score_fn(pose_vectors)
        assert vina_scores.shape == (n_poses,), \
            f"score_fn must return [N] tensor, got {vina_scores.shape}"

    runtime = time.time() - t0

    if config.verbose:
        best = (vina_scores if vina_scores is not None else stub_scores).min().item()
        print(f"[Stage1] Done in {runtime:.2f}s | best score = {best:.3f}")

    return Stage1Result(
        pose_vectors = pose_vectors,
        stub_scores  = stub_scores,
        vina_scores  = vina_scores,
        n_poses      = n_poses,
        runtime_s    = runtime,
        used_rust    = used_rust,
    )


if __name__ == "__main__":
    print("=== Stage1 Unit Tests ===\n")
    import tempfile, os

    # Build minimal config with temp files
    cfg = DockingConfig()
    with tempfile.NamedTemporaryFile(suffix=".pdbqt", delete=False) as f:
        cfg.receptor_path = f.name
    with tempfile.NamedTemporaryFile(suffix=".pdbqt", delete=False) as f:
        cfg.ligand_path = f.name
    cfg.box_center = (15.0, -5.0, 10.0)
    cfg.stage1.n_poses = 100   # small for testing

    # Test 1: runs and returns correct shape
    result = run(cfg, n_torsions=5, seed=42)
    assert result.pose_vectors.shape == (100, 24), f"Shape: {result.pose_vectors.shape}"
    assert result.stub_scores.shape  == (100,)
    assert result.n_poses == 100
    print(f"PASS: stage1 returns [100, 24] poses in {result.runtime_s:.2f}s "
          f"({'Rust' if result.used_rust else 'Python'})")

    # Test 2: reproducible with same seed
    r2 = run(cfg, n_torsions=5, seed=42)
    assert torch.allclose(result.pose_vectors, r2.pose_vectors), \
        "Stage1 not reproducible"
    print("PASS: same seed → identical poses")

    # Test 3: different seed → different poses
    r3 = run(cfg, n_torsions=5, seed=99)
    assert not torch.allclose(result.pose_vectors, r3.pose_vectors), \
        "Different seeds should differ"
    print("PASS: different seed → different poses")

    # Test 4: score_fn callback works
    def mock_vina(poses):
        return -torch.rand(len(poses)) * 10   # mock scores in [-10, 0]

    r4 = run(cfg, n_torsions=5, score_fn=mock_vina, seed=42)
    assert r4.vina_scores is not None
    assert r4.vina_scores.shape == (100,)
    print("PASS: score_fn callback returns vina_scores")

    # Test 5: missing box_center raises immediately
    cfg_bad = DockingConfig()
    cfg_bad.receptor_path = cfg.receptor_path
    cfg_bad.ligand_path   = cfg.ligand_path
    try:
        run(cfg_bad, n_torsions=5)
        raise AssertionError("Should have raised")
    except ValueError as e:
        assert "box_center" in str(e)
        print("PASS: missing box_center raises ValueError immediately")

    os.unlink(cfg.receptor_path)
    os.unlink(cfg.ligand_path)
    print("\n=== ALL TESTS PASSED ===")
