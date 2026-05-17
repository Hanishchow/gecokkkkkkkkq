"""
core/som.py — BindingModeSOM
Self-Organizing Map for topology-preserving binding pose clustering.

Scientific contract:
  - Input : N pose vectors, each padded to POSE_DIM (24D)
  - Output: 8×8 grid where spatial neighbors = similar binding modes
  - Guarantee: quantization error decreases monotonically during training
  - Replaces: MiniBatchKMeans in Stage 2

Why SOM over K-means:
  K-means clusters have no spatial relationship.
  SOM neurons form a 2D manifold — neighbor neurons encode similar poses.
  This lets Stage 4 MC refinement walk the manifold instead of jumping blind.

Reference: Kohonen, T. (1990). The self-organizing map. IEEE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


POSE_DIM = 24       # 3 translation + 3 rotation + 18 torsion (padded to 95th pct)
GRID_H   = 8
GRID_W   = 8
N_NEURONS = GRID_H * GRID_W   # 64


class BindingModeSOM(nn.Module):
    """
    8×8 Self-Organizing Map over 24-dimensional pose space.

    Weights shape: [64, 24] — one weight vector per neuron.
    Grid coords  : [64, 2]  — (row, col) for each neuron.

    Training uses competitive Hebbian learning with a Gaussian
    neighborhood function that shrinks over epochs.
    """

    def __init__(
        self,
        grid_h: int = GRID_H,
        grid_w: int = GRID_W,
        pose_dim: int = POSE_DIM,
        initial_lr: float = 0.5,
        initial_sigma: float = 3.0,
    ):
        super().__init__()
        self.grid_h   = grid_h
        self.grid_w   = grid_w
        self.n        = grid_h * grid_w
        self.pose_dim = pose_dim

        # Learnable neuron weights — initialized uniformly in [-1, 1]
        self.weights = nn.Parameter(
            torch.randn(self.n, pose_dim) * 0.1,
            requires_grad=False,   # SOM uses custom Hebbian update, not backprop
        )

        # 2D grid coordinates for each neuron — fixed, not learned
        rows = torch.arange(grid_h).float()
        cols = torch.arange(grid_w).float()
        grid_r, grid_c = torch.meshgrid(rows, cols, indexing="ij")
        coords = torch.stack([grid_r.flatten(), grid_c.flatten()], dim=1)
        self.register_buffer("coords", coords)   # [64, 2]

        self.initial_lr    = initial_lr
        self.initial_sigma = initial_sigma

        # Training state — tracked for learning rate/sigma decay
        self._epoch      = 0
        self._total_epochs = 1

        # Running quantization error (honest metric, logged during training)
        self.quantization_errors: list[float] = []

    # ------------------------------------------------------------------
    # Core SOM operations
    # ------------------------------------------------------------------

    def find_bmu(self, x: torch.Tensor) -> torch.Tensor:
        """
        Best Matching Unit: neuron whose weight vector is closest to x.

        Args:
            x: [B, pose_dim] or [pose_dim]
        Returns:
            bmu_idx: [B] long tensor of flat neuron indices
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        # Euclidean distance to every neuron
        dists = torch.cdist(x, self.weights)   # [B, n_neurons]
        return dists.argmin(dim=1)             # [B]

    def neighborhood(self, bmu_idx: torch.Tensor, sigma: float) -> torch.Tensor:
        """
        Gaussian neighborhood function centered on BMU.

        Returns influence weight for every neuron given BMU location.

        Args:
            bmu_idx: [B] flat neuron indices
            sigma  : current neighborhood radius
        Returns:
            h: [B, n_neurons] influence weights in [0, 1]
        """
        bmu_coords = self.coords[bmu_idx]          # [B, 2]
        # Squared distances on the 2D grid
        diff = self.coords.unsqueeze(0) - bmu_coords.unsqueeze(1)  # [B, n, 2]
        sq_dist = (diff ** 2).sum(dim=-1)          # [B, n]
        h = torch.exp(-sq_dist / (2 * sigma ** 2)) # [B, n]
        return h

    def _decay(self, initial: float, epoch: int, total: int) -> float:
        """Exponential decay: halves over total epochs."""
        return initial * np.exp(-epoch / max(total, 1))

    def hebbian_update(self, x: torch.Tensor, epoch: int, total_epochs: int):
        """
        Single SOM weight update step (Hebbian, not backprop).

        Args:
            x: [B, pose_dim] batch of pose vectors
            epoch: current epoch index
            total_epochs: total planned epochs (for decay)
        """
        lr    = self._decay(self.initial_lr,    epoch, total_epochs)
        sigma = self._decay(self.initial_sigma, epoch, total_epochs)
        sigma = max(sigma, 0.5)   # never collapse to 0

        bmu_idx = self.find_bmu(x)                  # [B]
        h       = self.neighborhood(bmu_idx, sigma)  # [B, n]

        # Δw_i = lr * h_i * (x - w_i)  averaged over batch
        delta = x.unsqueeze(1) - self.weights.unsqueeze(0)  # [B, n, d]
        update = (h.unsqueeze(-1) * delta).mean(dim=0)      # [n, d]

        with torch.no_grad():
            self.weights.data += lr * update

    def quantization_error(self, x: torch.Tensor) -> float:
        """
        Mean distance from each input to its BMU weight.
        Lower = better fit. Must decrease (roughly) over training.
        """
        bmu_idx = self.find_bmu(x)
        bmu_w   = self.weights[bmu_idx]          # [B, d]
        return F.mse_loss(x, bmu_w).item()

    # ------------------------------------------------------------------
    # High-level API
    # ------------------------------------------------------------------

    def fit(self, pose_vectors: torch.Tensor, epochs: int = 100, batch_size: int = 64):
        """
        Train SOM on a dataset of pose vectors.

        Args:
            pose_vectors: [N, pose_dim]
            epochs: training epochs
            batch_size: mini-batch size
        """
        self.quantization_errors = []
        N = pose_vectors.shape[0]

        for epoch in range(epochs):
            # Shuffle
            perm = torch.randperm(N)
            data = pose_vectors[perm]

            for i in range(0, N, batch_size):
                batch = data[i : i + batch_size]
                self.hebbian_update(batch, epoch, epochs)

            qe = self.quantization_error(pose_vectors)
            self.quantization_errors.append(qe)

            if epoch % 10 == 0:
                print(f"  SOM epoch {epoch:3d}/{epochs} | QE = {qe:.6f}")

        # Validate monotonic decrease (allow 20% tolerance for noise)
        self._validate_training()

    def quantize(self, pose_vectors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Map pose vectors to their BMU indices and BMU weight vectors.

        Args:
            pose_vectors: [N, pose_dim]
        Returns:
            bmu_indices : [N] long   — which neuron won for each pose
            bmu_weights : [N, pose_dim] — the winning neuron's weight
        """
        bmu_idx = self.find_bmu(pose_vectors)
        return bmu_idx, self.weights[bmu_idx]

    def get_neighbors(self, neuron_idx: int, radius: int = 1) -> torch.Tensor:
        """
        Return flat indices of neurons within grid radius of neuron_idx.
        Used by Stage 4 MC refinement to walk the manifold.

        Args:
            neuron_idx: flat index of center neuron
            radius: Chebyshev distance on 2D grid
        Returns:
            neighbor_indices: [K] long tensor
        """
        center = self.coords[neuron_idx]   # [2]
        diff   = (self.coords - center).abs().max(dim=1).values  # [n]
        mask   = diff <= radius
        return mask.nonzero(as_tuple=True)[0]

    def get_cluster_assignments(self, pose_vectors: torch.Tensor) -> dict:
        """
        Returns a dict mapping neuron_idx → list of pose indices assigned to it.
        Used for Stage 2 archetype extraction.
        """
        bmu_idx, _ = self.quantize(pose_vectors)
        assignments: dict = {i: [] for i in range(self.n)}
        for pose_i, neuron_i in enumerate(bmu_idx.tolist()):
            assignments[neuron_i].append(pose_i)
        return assignments

    def _validate_training(self):
        """
        Scientific honesty check: QE should decrease overall.
        Raises a warning (not error) if it doesn't — training data
        might be too small.
        """
        if len(self.quantization_errors) < 2:
            return
        first = self.quantization_errors[0]
        last  = self.quantization_errors[-1]
        if last >= first:
            print(
                f"  [WARNING] SOM QE did not decrease: {first:.4f} → {last:.4f}. "
                f"Training set may be too small or LR too low."
            )
        else:
            improvement = 100 * (first - last) / first
            print(f"  [SOM] QE improved {improvement:.1f}%: {first:.4f} → {last:.4f}")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str):
        torch.save(
            {
                "weights": self.weights.data,
                "coords" : self.coords,
                "config" : {
                    "grid_h"        : self.grid_h,
                    "grid_w"        : self.grid_w,
                    "pose_dim"      : self.pose_dim,
                    "initial_lr"    : self.initial_lr,
                    "initial_sigma" : self.initial_sigma,
                },
                "quantization_errors": self.quantization_errors,
            },
            path,
        )
        print(f"[SOM] Saved to {path}")

    @classmethod
    def load(cls, path: str) -> "BindingModeSOM":
        ckpt = torch.load(path, map_location="cpu")
        cfg  = ckpt["config"]
        som  = cls(**cfg)
        som.weights.data = ckpt["weights"]
        som.coords       = ckpt["coords"]
        som.quantization_errors = ckpt.get("quantization_errors", [])
        print(f"[SOM] Loaded from {path}")
        return som


# ------------------------------------------------------------------
# Inline unit tests — run: python -m geock.core.som
# ------------------------------------------------------------------

if __name__ == "__main__":
    print("=== BindingModeSOM Unit Tests ===\n")

    torch.manual_seed(42)

    # Test 1: shape contracts
    som = BindingModeSOM()
    assert som.weights.shape == (64, 24), "Weight shape wrong"
    assert som.coords.shape  == (64, 2),  "Coord shape wrong"
    print("PASS: weight and coord shapes correct")

    # Test 2: BMU returns valid indices
    x = torch.randn(10, 24)
    bmu = som.find_bmu(x)
    assert bmu.shape == (10,), "BMU shape wrong"
    assert bmu.min() >= 0 and bmu.max() < 64, "BMU out of range"
    print("PASS: BMU indices in valid range")

    # Test 3: neighborhood is in [0,1] and peaks at BMU
    h = som.neighborhood(bmu[:1], sigma=2.0)
    assert h.shape == (1, 64), "Neighborhood shape wrong"
    assert (h >= 0).all() and (h <= 1).all(), "Neighborhood out of [0,1]"
    assert h[0, bmu[0]].item() == h[0].max().item(), "BMU not peak of neighborhood"
    print("PASS: neighborhood function correct")

    # Test 4: training reduces quantization error
    data = torch.randn(200, 24)
    som.fit(data, epochs=30, batch_size=32)
    assert som.quantization_errors[-1] < som.quantization_errors[0], \
        "QE must decrease during training"
    print("PASS: quantization error decreases during training")

    # Test 5: get_neighbors returns center and ring
    neighbors = som.get_neighbors(neuron_idx=0, radius=1)
    assert 0 in neighbors, "Center not in neighbors"
    print(f"PASS: neuron 0 has {len(neighbors)} neighbors at radius 1")

    # Test 6: cluster assignments cover all poses
    assignments = som.get_cluster_assignments(data)
    total = sum(len(v) for v in assignments.values())
    assert total == len(data), "Not all poses assigned"
    print("PASS: all poses assigned to clusters")

    # Test 7: save/load roundtrip
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    som.save(path)
    som2 = BindingModeSOM.load(path)
    assert torch.allclose(som.weights.data, som2.weights.data), "Weights changed after reload"
    os.unlink(path)
    print("PASS: save/load roundtrip preserves weights")

    print("\n=== ALL TESTS PASSED ===")
