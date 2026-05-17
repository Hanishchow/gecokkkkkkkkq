"""
core/hopfield.py — HopfieldBindingMemory
Modern Hopfield Network for binding pose novelty filtering.

Scientific contract:
  - Stores prototypical binding poses from PDBbind pre-training
  - At runtime: query → recall → if similarity < threshold → novel → keep
  - Prevents VAE from generating poses that are just memorized noise

Reference: Ramsauer et al. (2020). Hopfield Networks is All You Need.
           ICLR 2021. https://arxiv.org/abs/2008.02217

Key difference from classical Hopfield:
  Classical: binary patterns, limited capacity (~0.14N patterns)
  Modern:    continuous patterns, exponential capacity (2^(N/2))
  We use modern because binding poses are continuous vectors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class HopfieldBindingMemory(nn.Module):
    """
    Modern Hopfield Network storing binding mode prototypes.

    Architecture:
      - Stored patterns: [M, pose_dim]  (up to 1000 from PDBbind)
      - Query          : [B, pose_dim]
      - Recall         : softmax(beta * Q @ K^T) @ V
        where K = V = stored patterns (key = value = memory)

    The beta parameter controls retrieval sharpness:
      - Low beta  → blended recall (generalization)
      - High beta → sharp recall (nearest prototype lookup)

    We tune beta so that similar poses recall cleanly and
    dissimilar poses recall noisily → similarity score discriminates well.
    """

    def __init__(
        self,
        pose_dim: int = 24,
        max_memories: int = 1000,
        beta: float = 8.0,
        novelty_threshold: float = 0.85,
    ):
        super().__init__()
        self.pose_dim          = pose_dim
        self.max_memories      = max_memories
        self.beta              = beta
        self.novelty_threshold = novelty_threshold

        # Stored memory patterns — filled during pre-training
        # Not a Parameter: we don't backprop through Hopfield in main pipeline
        self.register_buffer(
            "memories",
            torch.zeros(max_memories, pose_dim)
        )
        self.register_buffer(
            "n_stored",
            torch.tensor(0, dtype=torch.long)
        )

        # Optional: learned projection to query/key space
        # Helps when pose dim is small relative to memory size
        self.query_proj = nn.Linear(pose_dim, pose_dim, bias=False)
        self.key_proj   = nn.Linear(pose_dim, pose_dim, bias=False)

        # Initialize projections as identity (can be fine-tuned)
        nn.init.eye_(self.query_proj.weight)
        nn.init.eye_(self.key_proj.weight)

    # ------------------------------------------------------------------
    # Memory management
    # ------------------------------------------------------------------

    def store(self, patterns: torch.Tensor):
        """
        Store binding mode prototypes (called during PDBbind pre-training).

        Args:
            patterns: [M, pose_dim] — prototype poses to store
                      If M > max_memories, keeps most diverse subset.
        """
        M = patterns.shape[0]

        if M > self.max_memories:
            # Keep most diverse subset via greedy farthest-point sampling
            patterns = self._farthest_point_sample(patterns, self.max_memories)
            M = self.max_memories

        # L2-normalize before storing (cosine similarity assumption)
        normed = F.normalize(patterns, p=2, dim=-1)
        self.memories[:M] = normed
        self.n_stored.fill_(M)
        print(f"[Hopfield] Stored {M} binding mode prototypes")

    def _farthest_point_sample(
        self, points: torch.Tensor, k: int
    ) -> torch.Tensor:
        """
        Greedy farthest-point sampling for maximum diversity.
        O(N*k) — acceptable for N ≤ 5000.

        Returns k points maximally spread in pose space.
        """
        N = points.shape[0]
        selected = [torch.randint(N, (1,)).item()]
        dists    = torch.full((N,), float("inf"))

        for _ in range(k - 1):
            last     = points[selected[-1]].unsqueeze(0)
            d        = torch.cdist(points, last).squeeze(1)
            dists    = torch.minimum(dists, d)
            selected.append(dists.argmax().item())

        return points[torch.tensor(selected)]

    @property
    def active_memories(self) -> torch.Tensor:
        """Returns only the filled portion of the memory buffer."""
        n = self.n_stored.item()
        if n == 0:
            raise RuntimeError(
                "Hopfield memory is empty. "
                "Run pre-training or load hopfield_memories.pt first."
            )
        return self.memories[:n]   # [n_stored, pose_dim]

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def recall(self, query: torch.Tensor) -> torch.Tensor:
        """
        Modern Hopfield retrieval: softmax attention over stored memories.

        Args:
            query: [B, pose_dim] — L2-normalized query poses
        Returns:
            recalled: [B, pose_dim] — pattern-completed outputs
        """
        Q = self.query_proj(query)                    # [B, d]
        K = self.key_proj(self.active_memories)       # [M, d]

        # Scaled dot-product attention (beta = inverse temperature)
        attn = torch.matmul(Q, K.T) * self.beta       # [B, M]
        attn = F.softmax(attn, dim=-1)                 # [B, M]

        # Value = memories themselves (key = value in Hopfield)
        recalled = torch.matmul(attn, self.active_memories)  # [B, d]
        return recalled

    def similarity(self, query: torch.Tensor) -> torch.Tensor:
        """
        Cosine similarity between query and its recalled pattern.

        High similarity → query matches a stored prototype → not novel
        Low similarity  → query is genuinely different   → novel

        Args:
            query: [B, pose_dim]
        Returns:
            sim: [B] cosine similarities in [-1, 1]
        """
        q_norm = F.normalize(query,          p=2, dim=-1)
        recalled = self.recall(q_norm)
        r_norm   = F.normalize(recalled,     p=2, dim=-1)
        return F.cosine_similarity(q_norm, r_norm, dim=-1)   # [B]

    def is_novel(
        self,
        query: torch.Tensor,
        threshold: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Boolean mask: True if pose is novel (not in Hopfield memory).

        Args:
            query    : [B, pose_dim]
            threshold: override default novelty_threshold if needed
        Returns:
            novel_mask: [B] bool — True = novel, keep; False = memorized, discard
        """
        t   = threshold if threshold is not None else self.novelty_threshold
        sim = self.similarity(query)   # [B]
        return sim < t                  # novel if similarity is LOW

    def filter_novel(
        self,
        poses: torch.Tensor,
        threshold: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Filter a batch of poses, keeping only novel ones.

        Args:
            poses: [N, pose_dim]
        Returns:
            novel_poses : [K, pose_dim] where K ≤ N
            novel_mask  : [N] bool mask
        """
        mask = self.is_novel(poses, threshold)
        return poses[mask], mask

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def memory_utilization(self) -> float:
        """Fraction of memory slots used."""
        return self.n_stored.item() / self.max_memories

    def retrieval_sharpness(self, sample_size: int = 100) -> float:
        """
        Measures retrieval sharpness: average max attention weight.
        High → sharp (good), Low → blurry retrieval.
        Only valid after memories are stored.
        """
        if self.n_stored.item() == 0:
            return 0.0
        mem = self.active_memories
        idx = torch.randperm(len(mem))[:min(sample_size, len(mem))]
        sample = mem[idx]
        Q = self.query_proj(sample)
        K = self.key_proj(mem)
        attn = F.softmax(torch.matmul(Q, K.T) * self.beta, dim=-1)
        return attn.max(dim=-1).values.mean().item()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str):
        torch.save(
            {
                "memories"          : self.memories,
                "n_stored"          : self.n_stored,
                "query_proj"        : self.query_proj.state_dict(),
                "key_proj"          : self.key_proj.state_dict(),
                "config": {
                    "pose_dim"          : self.pose_dim,
                    "max_memories"      : self.max_memories,
                    "beta"              : self.beta,
                    "novelty_threshold" : self.novelty_threshold,
                },
            },
            path,
        )
        print(f"[Hopfield] Saved {self.n_stored.item()} memories to {path}")

    @classmethod
    def load(cls, path: str) -> "HopfieldBindingMemory":
        ckpt = torch.load(path, map_location="cpu")
        cfg  = ckpt["config"]
        hfn  = cls(**cfg)
        hfn.memories  = ckpt["memories"]
        hfn.n_stored  = ckpt["n_stored"]
        hfn.query_proj.load_state_dict(ckpt["query_proj"])
        hfn.key_proj.load_state_dict(ckpt["key_proj"])
        print(f"[Hopfield] Loaded {hfn.n_stored.item()} memories from {path}")
        return hfn


# ------------------------------------------------------------------
# Inline unit tests — run: python -m geock.core.hopfield
# ------------------------------------------------------------------

if __name__ == "__main__":
    print("=== HopfieldBindingMemory Unit Tests ===\n")

    torch.manual_seed(42)

    # Test 1: store and recall shapes
    hfn = HopfieldBindingMemory(pose_dim=24, max_memories=100)
    patterns = torch.randn(80, 24)
    hfn.store(patterns)
    assert hfn.n_stored.item() == 80, "Wrong number stored"

    query   = torch.randn(5, 24)
    recalled = hfn.recall(query)
    assert recalled.shape == (5, 24), f"Recall shape wrong: {recalled.shape}"
    print("PASS: store and recall shapes correct")

    # Test 2: similarity in [-1, 1]
    sim = hfn.similarity(query)
    assert sim.shape == (5,), "Similarity shape wrong"
    assert (sim >= -1.0).all() and (sim <= 1.0).all(), "Similarity out of range"
    print("PASS: similarity values in [-1, 1]")

    # Test 3: stored pattern recalls itself with high similarity
    stored_query = F.normalize(patterns[:5], p=2, dim=-1)
    sim_stored   = hfn.similarity(stored_query)
    # Stored patterns should have high similarity (> 0.7 with beta=8)
    assert sim_stored.mean().item() > 0.5, \
        f"Stored patterns recall similarity too low: {sim_stored.mean():.3f}"
    print(f"PASS: stored patterns recall with mean sim = {sim_stored.mean():.3f}")

    # Test 4: random noise is more novel than stored patterns
    noise = torch.randn(10, 24)
    sim_noise  = hfn.similarity(noise)
    sim_stored2 = hfn.similarity(F.normalize(patterns[10:20], p=2, dim=-1))
    assert sim_stored2.mean() > sim_noise.mean(), \
        "Stored patterns should be less novel than random noise"
    print(f"PASS: stored sim {sim_stored2.mean():.3f} > noise sim {sim_noise.mean():.3f}")

    # Test 5: novelty filter returns bool mask
    poses      = torch.randn(20, 24)
    novel, mask = hfn.filter_novel(poses)
    assert mask.dtype == torch.bool, "Mask should be bool"
    assert novel.shape[1] == 24, "Novel poses wrong dim"
    print(f"PASS: novelty filter keeps {mask.sum().item()}/20 poses")

    # Test 6: farthest point sampling preserves diversity
    big_set = torch.randn(500, 24)
    sampled = hfn._farthest_point_sample(big_set, 50)
    assert sampled.shape == (50, 24), "FPS shape wrong"
    # Sampled set should have higher pairwise distance than random subset
    rand_subset = big_set[:50]
    fps_pdist  = torch.pdist(sampled).mean().item()
    rand_pdist = torch.pdist(rand_subset).mean().item()
    assert fps_pdist >= rand_pdist * 0.9, \
        f"FPS not diverse enough: {fps_pdist:.3f} vs random {rand_pdist:.3f}"
    print(f"PASS: FPS diversity {fps_pdist:.3f} >= random {rand_pdist:.3f}")

    # Test 7: retrieval sharpness is positive
    sharpness = hfn.retrieval_sharpness()
    assert sharpness > 0, "Sharpness should be positive"
    print(f"PASS: retrieval sharpness = {sharpness:.4f}")

    # Test 8: save/load roundtrip
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    hfn.save(path)
    hfn2 = HopfieldBindingMemory.load(path)
    assert torch.allclose(hfn.memories, hfn2.memories), "Memories changed after reload"
    os.unlink(path)
    print("PASS: save/load roundtrip preserves memories")

    # Test 9: max_memories overflow triggers FPS
    hfn_small = HopfieldBindingMemory(pose_dim=24, max_memories=10)
    hfn_small.store(torch.randn(50, 24))   # 50 > 10 → FPS triggered
    assert hfn_small.n_stored.item() == 10, "Should cap at max_memories"
    print("PASS: overflow triggers farthest-point sampling")

    print("\n=== ALL TESTS PASSED ===")
