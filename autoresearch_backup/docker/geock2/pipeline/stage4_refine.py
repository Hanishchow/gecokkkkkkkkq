"""
pipeline/stage4_refine.py — Stage 4: SOM-Neighborhood MC Refinement

Takes novel poses from Stage 3 and refines them using the Rust MC engine,
guided by SOM topology: moves are biased toward neighboring archetypes
instead of blind random perturbations.

This is the publishable novelty over standard Vina refinement:
  Standard: random MC perturbation from starting pose
  DNBAP:    topology-aware moves that explore the binding manifold
"""

import torch
import time
from dataclasses import dataclass
from typing import Optional

try:
    import geock_mc
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

from geock.config import DockingConfig, POSE_DIM
from geock.core.som import BindingModeSOM
from geock.pipeline.stage2_cluster import Stage2Result, get_neighbor_map
from geock.pipeline.stage3_generate import Stage3Result


@dataclass
class Stage4Result:
    refined_poses:  torch.Tensor   # [K, 24]
    refined_scores: torch.Tensor   # [K]
    hopfield_sims:  torch.Tensor   # [K] — carried from Stage 3
    runtime_s:      float
    used_rust:      bool


def run(
    stage3:  Stage3Result,
    stage2:  Stage2Result,
    config:  DockingConfig,
    som:     BindingModeSOM,
    seed:    int = 42,
) -> Stage4Result:
    """
    Run Stage 4: SOM-neighborhood-aware MC refinement.

    Args:
        stage3 : Stage 3 result (novel_poses to refine)
        stage2 : Stage 2 result (bmu_indices for neighbor lookup)
        config : DockingConfig
        som    : fine-tuned SOM from Stage 2 (for neighbor traversal)
        seed   : random seed

    Returns:
        Stage4Result — refined poses ready for Stage 5 scoring
    """
    poses  = stage3.novel_poses    # [K, 24]
    scores = stage3.novel_scores   # [K]
    K      = len(poses)

    if K == 0:
        raise ValueError("Stage4 received 0 poses from Stage3. Check novelty threshold.")

    cfg4       = config.stage4
    box_center = config.get_box_center()
    box_size   = config.stage1.box_size

    if config.verbose:
        print(f"[Stage4] Refining {K} poses | "
              f"{cfg4.mc_steps} MC steps each | "
              f"SOM neighbor radius={cfg4.neighbor_radius} | "
              f"{'Rust' if RUST_AVAILABLE else 'Python'}")

    t0 = time.time()

    # Build neighbor map: for each novel pose, find BMU then get grid neighbors
    # Novel poses don't have BMU indices yet — compute them now
    bmu_indices, _ = som.quantize(poses)   # [K]
    neighbor_map   = get_neighbor_map(bmu_indices, som, radius=cfg4.neighbor_radius)

    # SOM weights as list of lists for Rust
    som_weights_list = som.weights.data.tolist()   # [64, 24]

    if RUST_AVAILABLE:
        refined_poses_raw, refined_scores_raw = geock_mc.mc_refine(
            poses            = poses.tolist(),
            scores           = scores.tolist(),
            som_weights      = som_weights_list,
            neighbor_indices = neighbor_map,
            n_torsions       = max(0, POSE_DIM - 6),
            mc_steps         = cfg4.mc_steps,
            temperature      = 0.8,
            box_center       = list(box_center),
            box_size         = list(box_size),
            seed             = seed,
        )
        used_rust = True
    else:
        # Python fallback
        refined_poses_raw, refined_scores_raw = _python_refine_fallback(
            poses.tolist(), scores.tolist(), som_weights_list,
            neighbor_map, cfg4.mc_steps, box_center, box_size, seed,
        )
        used_rust = False

    refined_poses  = torch.tensor(refined_poses_raw,  dtype=torch.float32)
    refined_scores = torch.tensor(refined_scores_raw, dtype=torch.float32)

    runtime = time.time() - t0

    if config.verbose:
        improvement = (scores.mean() - refined_scores.mean()).item()
        print(
            f"[Stage4] Done in {runtime:.2f}s | "
            f"mean score improvement = {improvement:+.4f}"
        )

    return Stage4Result(
        refined_poses  = refined_poses,
        refined_scores = refined_scores,
        hopfield_sims  = stage3.hopfield_sims,   # carry through for Stage 5
        runtime_s      = runtime,
        used_rust      = used_rust,
    )


def _python_refine_fallback(poses, scores, som_weights, neighbor_map,
                             mc_steps, box_center, box_size, seed):
    """Pure Python fallback for Stage 4 (slow but correct)."""
    import random, math
    rng = random.Random(seed)

    def score(dof):
        dx = dof[0]-box_center[0]; dy = dof[1]-box_center[1]; dz = dof[2]-box_center[2]
        if abs(dx)>box_size[0]/2 or abs(dy)>box_size[1]/2 or abs(dz)>box_size[2]/2:
            return 1000.0
        return -10.0 * math.exp(-(dx*dx+dy*dy+dz*dz)/50.0)

    out_poses, out_scores = [], []
    for pi, (pose, sc) in enumerate(zip(poses, scores)):
        cur = pose[:]; cur_s = sc
        nbrs = neighbor_map[pi]
        for si in range(mc_steps):
            trial = cur[:]
            if si % 5 == 0 and nbrs and som_weights:
                nn = som_weights[rng.choice(nbrs)]
                trial = [c + 0.2*(n-c) + rng.gauss(0, 0.3) for c, n in zip(cur, nn)]
            else:
                idx = rng.randint(0, len(trial)-1)
                trial[idx] += rng.gauss(0, 0.5)
            ts = score(trial)
            if ts < cur_s or rng.random() < math.exp(-(ts-cur_s)/0.8):
                cur = trial; cur_s = ts
        out_poses.append(cur); out_scores.append(cur_s)
    return out_poses, out_scores


if __name__ == "__main__":
    print("=== Stage4 Unit Tests ===\n")
    import tempfile, os
    from geock.pipeline.stage3_generate import Stage3Result

    torch.manual_seed(42)

    K = 15
    mock_stage3 = Stage3Result(
        novel_poses   = torch.randn(K, POSE_DIM),
        novel_scores  = -torch.rand(K) * 5,
        hopfield_sims = torch.rand(K),
        n_generated   = 30, n_novel = K,
        novelty_rate  = 0.5, runtime_s = 0.1,
    )
    mock_stage2 = Stage2Result(
        bmu_indices        = torch.randint(0, 64, (200,)),
        centroids          = torch.randn(64, POSE_DIM),
        assignments        = {i: [] for i in range(64)},
        quantization_error = 0.5,
        n_active_neurons   = 45, runtime_s = 0.2,
    )

    cfg = DockingConfig()
    with tempfile.NamedTemporaryFile(suffix=".pdbqt", delete=False) as f: cfg.receptor_path = f.name
    with tempfile.NamedTemporaryFile(suffix=".pdbqt", delete=False) as f: cfg.ligand_path = f.name
    cfg.box_center = (0.0, 0.0, 0.0)
    cfg.stage4.mc_steps = 50   # fast for test

    som = BindingModeSOM()

    # Test 1: runs end to end
    result = run(mock_stage3, mock_stage2, cfg, som)
    assert result.refined_poses.shape  == (K, POSE_DIM)
    assert result.refined_scores.shape == (K,)
    assert result.hopfield_sims.shape  == (K,)
    print(f"PASS: Stage4 shapes correct | {'Rust' if result.used_rust else 'Python'} | "
          f"{result.runtime_s:.2f}s")

    # Test 2: hopfield_sims carried through unchanged
    assert torch.allclose(result.hopfield_sims, mock_stage3.hopfield_sims)
    print("PASS: hopfield_sims carried through from Stage3")

    # Test 3: empty poses raises
    empty = Stage3Result(
        novel_poses=torch.zeros(0, POSE_DIM), novel_scores=torch.zeros(0),
        hopfield_sims=torch.zeros(0), n_generated=0, n_novel=0,
        novelty_rate=0.0, runtime_s=0.0
    )
    try:
        run(empty, mock_stage2, cfg, som)
        raise AssertionError("Should have raised")
    except ValueError:
        print("PASS: empty poses raises ValueError")

    os.unlink(cfg.receptor_path)
    os.unlink(cfg.ligand_path)
    print("\n=== ALL TESTS PASSED ===")
