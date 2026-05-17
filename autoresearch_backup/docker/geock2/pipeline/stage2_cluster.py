"""
pipeline/stage2_cluster.py — Stage 2: SOM Topology Clustering

Replaces K-means with BindingModeSOM.
Maps 2000 raw MC poses to 64 binding archetypes on a topology-preserving 2D grid.

Output:
  - 64 cluster centroids (neuron weights) for Stage 3 VAE conditioning
  - Per-pose BMU assignments for Stage 4 neighborhood-aware refinement
  - SOM fine-tuned 10 epochs on this specific pocket (pocket adaptation)
"""

import torch
import time
from dataclasses import dataclass, field
from typing import Optional

from geock.config import DockingConfig, POSE_DIM
from geock.core.som import BindingModeSOM
from geock.pipeline.stage1_sampling import Stage1Result


@dataclass
class Stage2Result:
    bmu_indices:   torch.Tensor    # [N] — which neuron won for each pose
    centroids:     torch.Tensor    # [64, 24] — SOM neuron weights after fine-tune
    assignments:   dict            # neuron_idx → list of pose indices
    quantization_error: float      # final QE after fine-tuning (honest metric)
    n_active_neurons: int          # neurons with ≥1 pose assigned
    runtime_s:     float


def run(
    stage1:  Stage1Result,
    config:  DockingConfig,
    som:     BindingModeSOM,
) -> Stage2Result:
    """
    Run Stage 2 SOM clustering on Stage 1 poses.

    Args:
        stage1: output of Stage 1 (pose_vectors [N, 24])
        config: DockingConfig
        som   : pre-trained BindingModeSOM (loaded from som_weights.pt)
                Fine-tuned here for pocket adaptation.

    Returns:
        Stage2Result
    """
    poses = stage1.pose_vectors    # [N, 24]
    N     = len(poses)

    if config.verbose:
        print(f"[Stage2] SOM clustering {N} poses → {som.n} archetypes")

    t0 = time.time()

    # Fine-tune pre-trained SOM on this pocket (10 epochs)
    # This adapts universal binding archetypes to the current pocket shape
    finetune_epochs = config.som.finetune_epochs
    finetune_batch  = config.som.finetune_batch

    if config.verbose:
        print(f"[Stage2] Fine-tuning SOM {finetune_epochs} epochs on {N} pocket poses")

    som.fit(poses, epochs=finetune_epochs, batch_size=finetune_batch)

    # Assign all poses to their BMU
    bmu_indices, _ = som.quantize(poses)                    # [N]
    assignments    = som.get_cluster_assignments(poses)     # dict
    centroids      = som.weights.data.clone()               # [64, 24]
    qe             = som.quantization_error(poses)

    # Count active neurons (those with at least one pose)
    n_active = sum(1 for v in assignments.values() if len(v) > 0)

    runtime = time.time() - t0

    if config.verbose:
        print(
            f"[Stage2] Done in {runtime:.2f}s | QE={qe:.4f} | "
            f"active neurons={n_active}/{som.n}"
        )

    return Stage2Result(
        bmu_indices        = bmu_indices,
        centroids          = centroids,
        assignments        = assignments,
        quantization_error = qe,
        n_active_neurons   = n_active,
        runtime_s          = runtime,
    )


def get_neighbor_map(
    bmu_indices: torch.Tensor,
    som:         BindingModeSOM,
    radius:      int = 1,
) -> list:
    """
    For each pose, return flat indices of its BMU's grid neighbors.
    Used by Stage 4 Rust MC refinement for topology-aware moves.

    Returns:
        List of lists: [[neighbor_idx, ...], ...] — one list per pose
    """
    neighbor_map = []
    for bmu_idx in bmu_indices.tolist():
        nbrs = som.get_neighbors(bmu_idx, radius=radius).tolist()
        neighbor_map.append(nbrs)
    return neighbor_map


if __name__ == "__main__":
    print("=== Stage2 Unit Tests ===\n")
    import tempfile, os
    from geock.pipeline.stage1_sampling import Stage1Result

    torch.manual_seed(42)

    # Build mock Stage1 result
    n_poses = 200
    mock_stage1 = Stage1Result(
        pose_vectors = torch.randn(n_poses, POSE_DIM),
        stub_scores  = -torch.rand(n_poses) * 10,
        vina_scores  = None,
        n_poses      = n_poses,
        runtime_s    = 0.1,
        used_rust    = True,
    )

    cfg = DockingConfig()
    with tempfile.NamedTemporaryFile(suffix=".pdbqt", delete=False) as f:
        cfg.receptor_path = f.name
    with tempfile.NamedTemporaryFile(suffix=".pdbqt", delete=False) as f:
        cfg.ligand_path = f.name
    cfg.box_center = (0.0, 0.0, 0.0)
    cfg.som.finetune_epochs = 5   # fast for test

    som = BindingModeSOM()

    # Test 1: runs and returns correct shapes
    result = run(mock_stage1, cfg, som)
    assert result.bmu_indices.shape == (n_poses,), f"BMU shape: {result.bmu_indices.shape}"
    assert result.centroids.shape   == (64, POSE_DIM)
    assert isinstance(result.assignments, dict)
    print(f"PASS: Stage2 shapes correct | QE={result.quantization_error:.4f} | "
          f"active={result.n_active_neurons}/64")

    # Test 2: all poses are assigned
    total = sum(len(v) for v in result.assignments.values())
    assert total == n_poses, f"Only {total}/{n_poses} poses assigned"
    print("PASS: all poses have a BMU assignment")

    # Test 3: BMU indices are in valid range
    assert result.bmu_indices.min() >= 0
    assert result.bmu_indices.max() < 64
    print("PASS: BMU indices in [0, 63]")

    # Test 4: neighbor map
    nbr_map = get_neighbor_map(result.bmu_indices[:5], som, radius=1)
    assert len(nbr_map) == 5
    for nbrs in nbr_map:
        assert all(0 <= n < 64 for n in nbrs), "Neighbor out of range"
    print(f"PASS: neighbor map — first pose has {len(nbr_map[0])} neighbors")

    os.unlink(cfg.receptor_path)
    os.unlink(cfg.ligand_path)
    print("\n=== ALL TESTS PASSED ===")
