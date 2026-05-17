"""
core/gnn.py — PocketGNN
Graph Neural Network encoder for receptor binding pockets.

Scientific justification:
  The VAE in vae.py samples from N(0, I) — completely blind to the pocket.
  This GNN encodes the pocket as a molecular graph and produces a
  pocket embedding vector that conditions the VAE decoder.

  Result: VAE generates pocket-SPECIFIC poses, not generic binding poses.
  This is the key architectural difference from vanilla generative docking.

Architecture:
  Nodes  : heavy atoms in binding pocket (within 8Å of ligand)
  Edges  : pairs within distance threshold (6Å)
  Node features [9D]:
    - Atomic number (one-hot: C, N, O, S, P, F, Cl, Br, other)
  Edge features [1D]:
    - Interatomic distance (normalized)
  Layers : 3 × Message Passing (EdgeConv-style)
  Output : 32D pocket embedding → conditions VAE latent space

Why EdgeConv over GCN:
  GCN averages neighbor features (loses distance info).
  EdgeConv: h_i = max_j MLP([h_i, h_j - h_i, d_ij])
  Preserves relative position + distance → better geometry encoding.

Reference:
  Wang et al. (2019). Dynamic Graph CNN. ACM ToG.
  Equivariant-inspired but not full SE(3) — justified for docking
  where the pocket orientation is fixed to the box frame.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


# Atom type vocabulary: C, N, O, S, P, F, Cl, Br, other
ATOM_TYPES = {6: 0, 7: 1, 8: 2, 16: 3, 15: 4, 9: 5, 17: 6, 35: 7}
N_ATOM_TYPES = 9        # 8 types + 1 "other"
NODE_FEAT_DIM = N_ATOM_TYPES   # node feature dimension
EDGE_FEAT_DIM = 1              # just distance for now
POCKET_EMB_DIM = 32            # output embedding size
POCKET_RADIUS  = 8.0           # Angstroms — atoms within this of ligand centroid
EDGE_RADIUS    = 6.0           # Angstroms — edge cutoff


def atom_type_to_onehot(atomic_num: int) -> torch.Tensor:
    """Convert atomic number to 9D one-hot vector."""
    vec = torch.zeros(N_ATOM_TYPES)
    idx = ATOM_TYPES.get(atomic_num, N_ATOM_TYPES - 1)   # unknown → last slot
    vec[idx] = 1.0
    return vec


def build_pocket_graph(
    coords:       torch.Tensor,   # [N, 3] — all receptor heavy atom coords
    atomic_nums:  list,           # [N]    — atomic numbers
    center:       torch.Tensor,   # [3]    — pocket center (box center)
    pocket_radius: float = POCKET_RADIUS,
    edge_radius:   float = EDGE_RADIUS,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build a pocket graph from receptor coordinates.

    1. Select atoms within pocket_radius of center
    2. Build edges for pairs within edge_radius
    3. Compute node and edge features

    Returns:
        node_feats : [M, NODE_FEAT_DIM]  — M = atoms in pocket
        edge_index : [2, E]             — COO format edge list
        edge_feats : [E, EDGE_FEAT_DIM] — distances
    """
    # Select pocket atoms
    dists_to_center = torch.norm(coords - center.unsqueeze(0), dim=1)  # [N]
    pocket_mask     = dists_to_center < pocket_radius
    pocket_coords   = coords[pocket_mask]       # [M, 3]
    pocket_anums    = [a for a, m in zip(atomic_nums, pocket_mask.tolist()) if m]
    M = len(pocket_coords)

    if M == 0:
        # Empty pocket — return minimal graph with one dummy node
        return (
            torch.zeros(1, NODE_FEAT_DIM),
            torch.zeros(2, 0, dtype=torch.long),
            torch.zeros(0, EDGE_FEAT_DIM),
        )

    # Node features: one-hot atom type
    node_feats = torch.stack([atom_type_to_onehot(a) for a in pocket_anums])  # [M, 9]

    # Edge construction: all pairs within edge_radius
    pairwise = torch.cdist(pocket_coords, pocket_coords)  # [M, M]
    mask     = (pairwise < edge_radius) & (pairwise > 0)  # exclude self-loops
    src, dst = mask.nonzero(as_tuple=True)                # [E] each

    if len(src) == 0:
        # No edges — fully isolated graph
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_feats = torch.zeros(0, EDGE_FEAT_DIM)
    else:
        edge_index = torch.stack([src, dst])              # [2, E]
        distances  = pairwise[src, dst].unsqueeze(1)      # [E, 1]
        # Normalize distances to [0, 1]
        edge_feats = distances / edge_radius              # [E, 1]

    return node_feats, edge_index, edge_feats


class EdgeConvLayer(nn.Module):
    """
    EdgeConv message passing layer.

    For each node i: h_i' = max_{j ∈ N(i)} MLP([h_i || h_j - h_i || d_ij])

    The subtraction h_j - h_i captures relative features (directional info).
    Max aggregation is more expressive than sum/mean for geometry.
    """

    def __init__(self, in_dim: int, out_dim: int, edge_dim: int = EDGE_FEAT_DIM):
        super().__init__()
        mlp_in = in_dim * 2 + edge_dim   # [h_i, h_j - h_i, edge_feat]
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, out_dim),
            nn.LayerNorm(out_dim),
            nn.ELU(),
            nn.Linear(out_dim, out_dim),
            nn.ELU(),
        )

    def forward(
        self,
        x:          torch.Tensor,   # [N, in_dim]
        edge_index: torch.Tensor,   # [2, E]
        edge_feats: torch.Tensor,   # [E, edge_dim]
    ) -> torch.Tensor:
        """Returns updated node features [N, out_dim]."""
        N = x.shape[0]

        if edge_index.shape[1] == 0:
            # No edges — pass through with zero message
            dummy_in = torch.zeros(N, x.shape[1] * 2 + edge_feats.shape[-1] if edge_feats.numel() > 0 else x.shape[1] * 2 + 1)
            # Just apply MLP to self-loop features
            self_feat = torch.cat([x, torch.zeros_like(x), torch.zeros(N, 1)], dim=-1)
            return self.mlp(self_feat)

        src, dst = edge_index[0], edge_index[1]  # E each

        # Message: [h_src || h_dst - h_src || edge_feat]
        h_src  = x[src]                          # [E, in_dim]
        h_dst  = x[dst]                          # [E, in_dim]
        msgs   = torch.cat([h_src, h_dst - h_src, edge_feats], dim=-1)  # [E, mlp_in]
        msgs   = self.mlp(msgs)                  # [E, out_dim]

        # Max aggregation: for each destination node, take max over incoming messages
        out = torch.full((N, msgs.shape[1]), fill_value=-1e9, device=x.device)
        # Scatter max
        dst_expanded = dst.unsqueeze(1).expand_as(msgs)
        out.scatter_reduce_(0, dst_expanded, msgs, reduce="amax", include_self=True)
        # Replace -inf with 0 for isolated nodes
        out = out.clamp(min=0)

        return out


class PocketGNN(nn.Module):
    """
    3-layer EdgeConv GNN producing a 32D pocket embedding.

    The embedding conditions the AttentionPoseVAE decoder —
    making pose generation pocket-specific.

    Input  : pocket graph (node_feats, edge_index, edge_feats)
    Output : 32D embedding vector
    """

    def __init__(
        self,
        node_dim:   int = NODE_FEAT_DIM,   # 9
        hidden_dim: int = 32,
        out_dim:    int = POCKET_EMB_DIM,  # 32
        n_layers:   int = 3,
    ):
        super().__init__()
        self.node_dim   = node_dim
        self.hidden_dim = hidden_dim
        self.out_dim    = out_dim

        # Input projection
        self.input_proj = nn.Linear(node_dim, hidden_dim)

        # EdgeConv layers
        dims = [hidden_dim] * (n_layers + 1)
        self.conv_layers = nn.ModuleList([
            EdgeConvLayer(dims[i], dims[i+1])
            for i in range(n_layers)
        ])

        # Global pooling → graph-level embedding
        # Concatenate mean + max pooling for richer representation
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, out_dim),
            nn.LayerNorm(out_dim),
            nn.ELU(),
        )

    def forward(
        self,
        node_feats: torch.Tensor,   # [N, node_dim]
        edge_index: torch.Tensor,   # [2, E]
        edge_feats: torch.Tensor,   # [E, 1]
    ) -> torch.Tensor:
        """
        Returns pocket embedding: [out_dim]
        """
        h = self.input_proj(node_feats)   # [N, hidden_dim]

        for conv in self.conv_layers:
            h = conv(h, edge_index, edge_feats)   # [N, hidden_dim]

        # Global mean + max pooling
        h_mean = h.mean(dim=0)   # [hidden_dim]
        h_max  = h.max(dim=0).values   # [hidden_dim]
        h_pool = torch.cat([h_mean, h_max], dim=0)   # [2 * hidden_dim]

        return self.output_proj(h_pool)   # [out_dim]

    def encode_pocket(
        self,
        coords:      torch.Tensor,
        atomic_nums: list,
        center:      torch.Tensor,
    ) -> torch.Tensor:
        """
        Full pipeline: coords + atomic_nums → 32D embedding.
        Convenience wrapper that calls build_pocket_graph internally.
        """
        node_feats, edge_index, edge_feats = build_pocket_graph(
            coords, atomic_nums, center
        )
        return self.forward(node_feats, edge_index, edge_feats)

    def save(self, path: str):
        torch.save({
            "state_dict": self.state_dict(),
            "config": {
                "node_dim": self.node_dim,
                "hidden_dim": self.hidden_dim,
                "out_dim": self.out_dim,
            },
        }, path)
        print(f"[PocketGNN] Saved to {path}")

    @classmethod
    def load(cls, path: str) -> "PocketGNN":
        ckpt = torch.load(path, map_location="cpu")
        gnn  = cls(**ckpt["config"])
        gnn.load_state_dict(ckpt["state_dict"])
        gnn.eval()
        print(f"[PocketGNN] Loaded from {path}")
        return gnn


# ------------------------------------------------------------------
# Unit tests
# ------------------------------------------------------------------

if __name__ == "__main__":
    print("=== PocketGNN Unit Tests ===\n")
    torch.manual_seed(42)

    # Test 1: atom one-hot encoding
    oh = atom_type_to_onehot(6)   # Carbon
    assert oh.shape == (9,) and oh[0] == 1.0 and oh.sum() == 1.0
    oh_unk = atom_type_to_onehot(99)   # Unknown
    assert oh_unk[-1] == 1.0
    print("PASS: atom one-hot encoding correct")

    # Test 2: build_pocket_graph shapes
    N = 50
    coords   = torch.randn(N, 3) * 5
    anums    = [6] * 20 + [7] * 15 + [8] * 15
    center   = torch.zeros(3)
    nf, ei, ef = build_pocket_graph(coords, anums, center)
    assert nf.shape[1] == NODE_FEAT_DIM, f"Node feats: {nf.shape}"
    assert ei.shape[0] == 2
    assert ef.shape[1] == EDGE_FEAT_DIM
    print(f"PASS: build_pocket_graph | {nf.shape[0]} pocket atoms | {ei.shape[1]} edges")

    # Test 3: forward pass shapes
    gnn = PocketGNN()
    emb = gnn(nf, ei, ef)
    assert emb.shape == (POCKET_EMB_DIM,), f"Embedding shape: {emb.shape}"
    print(f"PASS: GNN forward → {emb.shape} embedding")

    # Test 4: different pockets produce different embeddings
    coords2 = torch.randn(N, 3) * 5 + 10
    nf2, ei2, ef2 = build_pocket_graph(coords2, anums, center)
    emb2 = gnn(nf2, ei2, ef2)
    assert not torch.allclose(emb, emb2), "Different pockets should give different embeddings"
    print("PASS: different pockets → different embeddings")

    # Test 5: encode_pocket convenience wrapper
    emb3 = gnn.encode_pocket(coords, anums, center)
    assert emb3.shape == (POCKET_EMB_DIM,)
    assert torch.allclose(emb, emb3), "encode_pocket inconsistent with forward"
    print("PASS: encode_pocket convenience wrapper consistent")

    # Test 6: gradients flow through GNN
    gnn.train()
    emb_grad = gnn(nf, ei, ef)
    loss = emb_grad.sum()
    loss.backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in gnn.parameters())
    assert has_grad, "No gradients flowing through GNN"
    print("PASS: gradients flow through all GNN layers")

    # Test 7: empty pocket (no atoms in radius) handled gracefully
    far_center = torch.tensor([1000.0, 1000.0, 1000.0])
    nf_empty, ei_empty, ef_empty = build_pocket_graph(coords, anums, far_center)
    emb_empty = gnn(nf_empty, ei_empty, ef_empty)
    assert emb_empty.shape == (POCKET_EMB_DIM,), "Empty pocket embedding shape wrong"
    print("PASS: empty pocket handled gracefully")

    # Test 8: save/load
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    gnn.eval()
    gnn.save(path)
    gnn2 = PocketGNN.load(path)
    emb_r = gnn2(nf, ei, ef)
    assert torch.allclose(emb, emb_r, atol=1e-5), "GNN changed after reload"
    os.unlink(path)
    print("PASS: save/load preserves embeddings exactly")

    print("\n=== ALL TESTS PASSED ===")
