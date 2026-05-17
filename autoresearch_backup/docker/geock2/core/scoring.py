"""
core/scoring.py — EnsembleScorer
Stage 5: Combine Vina score + learned correction term + Hopfield similarity.

Scientific contract:
  Final score = w_vina * vina_score
              + w_learned * learned_correction(pose_vector)
              + w_hopfield * hopfield_similarity_bonus

  All weights are learned on PDBbind (correlation to experimental ΔG).
  If weights file is missing, falls back to Vina-only (w_vina=1, rest=0).

Why this is publishable:
  The learned correction term captures systematic Vina errors
  (e.g., Vina penalizes buried polar contacts that are genuinely favorable).
  Hopfield similarity bonus rewards poses resembling known crystal modes.

  This is a re-scoring approach, not a black-box replacement.
  It's interpretable, ablatable, and defensible at peer review.

Architecture of learned correction:
  Input: [pose_vector (24D) | vina_score (1D)] → 25D total
  Network: 25 → 32 → 16 → 1  (correction delta in kcal/mol units)
  Loss: MSE to experimental binding affinity (PDBbind -log Kd/Ki)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
from dataclasses import dataclass


@dataclass
class ScoredPose:
    """Container for a fully scored pose."""
    pose_vector:      torch.Tensor   # [24] — pose DOF vector
    vina_score:       float          # raw Vina score (kcal/mol)
    learned_score:    float          # learned correction term
    hopfield_bonus:   float          # Hopfield similarity bonus
    ensemble_score:   float          # final combined score
    rank:             int            # rank in final sorted list


class LearnedCorrectionNet(nn.Module):
    """
    Small MLP that corrects Vina systematic errors.

    Input : [pose_vector | vina_score] = 25D
    Output: scalar correction in kcal/mol units

    Designed to be small (< 1k parameters) — we have limited PDBbind labels.
    Larger networks would overfit.
    """

    def __init__(self, pose_dim: int = 24, hidden: int = 32):
        super().__init__()
        input_dim = pose_dim + 1   # pose + vina_score

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 16),
            nn.ELU(),
            nn.Linear(16, 1),
        )

        # Initialize output layer near zero — correction starts neutral
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, pose_vector: torch.Tensor, vina_score: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pose_vector: [B, 24]
            vina_score : [B] or [B, 1]
        Returns:
            correction: [B] scalar corrections
        """
        if vina_score.dim() == 1:
            vina_score = vina_score.unsqueeze(-1)
        x = torch.cat([pose_vector, vina_score], dim=-1)   # [B, 25]
        return self.net(x).squeeze(-1)                      # [B]


class EnsembleScorer(nn.Module):
    """
    Combines three score sources into one final ranking score.

    Weights (w_vina, w_learned, w_hopfield) are learned by ridge regression
    on PDBbind validation set. If not trained, falls back to Vina only.

    Score convention: LOWER = BETTER (following Vina convention, kcal/mol)
    Hopfield bonus: negative (more similar to known good binder → lower score)
    """

    def __init__(self, pose_dim: int = 24):
        super().__init__()
        self.pose_dim = pose_dim
        self.correction_net = LearnedCorrectionNet(pose_dim=pose_dim)

        # Ensemble weights — learned, not hand-tuned
        # Initialize: full weight on Vina, zero on others (safe fallback)
        self.register_parameter(
            "log_w_vina",
            nn.Parameter(torch.tensor(0.0))    # exp(0) = 1.0
        )
        self.register_parameter(
            "log_w_learned",
            nn.Parameter(torch.tensor(-10.0))  # ≈ 0, small initial influence
        )
        self.register_parameter(
            "log_w_hopfield",
            nn.Parameter(torch.tensor(-10.0))
        )

        # Training state
        self.is_trained = False
        self.train_losses: list[float] = []

    @property
    def w_vina(self)    -> float: return torch.exp(self.log_w_vina).item()
    @property
    def w_learned(self) -> float: return torch.exp(self.log_w_learned).item()
    @property
    def w_hopfield(self)-> float: return torch.exp(self.log_w_hopfield).item()

    def forward(
        self,
        pose_vectors:      torch.Tensor,
        vina_scores:       torch.Tensor,
        hopfield_sims:     Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute ensemble scores for a batch of poses.

        Args:
            pose_vectors : [B, 24]
            vina_scores  : [B]  (kcal/mol, lower = better)
            hopfield_sims: [B]  cosine similarity in [0,1] (optional)
        Returns:
            ensemble_scores: [B]  (lower = better)
        """
        w_v = torch.exp(self.log_w_vina)
        w_l = torch.exp(self.log_w_learned)
        w_h = torch.exp(self.log_w_hopfield)

        correction = self.correction_net(pose_vectors, vina_scores)   # [B]

        score = w_v * vina_scores + w_l * correction

        if hopfield_sims is not None:
            # Bonus: higher similarity → lower (better) score
            # Multiply by -1 because lower is better
            score = score - w_h * hopfield_sims

        return score

    def rank_poses(
        self,
        pose_vectors:      torch.Tensor,
        vina_scores:       torch.Tensor,
        hopfield_sims:     Optional[torch.Tensor] = None,
        top_k:             int = 20,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Rank poses and return top-k.

        Returns:
            ranked_indices: [top_k] indices into original batch
            ranked_scores : [top_k] ensemble scores
        """
        with torch.no_grad():
            scores = self.forward(pose_vectors, vina_scores, hopfield_sims)
        sorted_idx = scores.argsort()          # ascending: best first
        top_idx    = sorted_idx[:top_k]
        return top_idx, scores[top_idx]

    def score_and_package(
        self,
        pose_vectors:  torch.Tensor,
        vina_scores:   torch.Tensor,
        hopfield_sims: Optional[torch.Tensor] = None,
        top_k:         int = 20,
    ) -> List[ScoredPose]:
        """
        Full scoring + packaging into ScoredPose objects for output.
        """
        with torch.no_grad():
            correction   = self.correction_net(pose_vectors, vina_scores)
            ensemble     = self.forward(pose_vectors, vina_scores, hopfield_sims)

        sorted_idx = ensemble.argsort()
        results    = []

        for rank, idx in enumerate([int(x) for x in sorted_idx[:top_k].tolist()]):
            h_bonus = hopfield_sims[idx].item() if hopfield_sims is not None else 0.0
            results.append(
                ScoredPose(
                    pose_vector    = pose_vectors[idx].detach(),
                    vina_score     = vina_scores[idx].item(),
                    learned_score  = correction[idx].item(),
                    hopfield_bonus = h_bonus,
                    ensemble_score = ensemble[idx].item(),
                    rank           = rank + 1,
                )
            )

        return results

    # ------------------------------------------------------------------
    # Training (called from pretrain/pretrain.py on PDBbind)
    # ------------------------------------------------------------------

    def fit(
        self,
        pose_vectors:    torch.Tensor,
        vina_scores:     torch.Tensor,
        experimental_dg: torch.Tensor,
        hopfield_sims:   Optional[torch.Tensor] = None,
        epochs:          int   = 200,
        lr:              float = 1e-3,
        val_split:       float = 0.1,
    ):
        """
        Train correction network + weights to predict experimental ΔG.

        Args:
            pose_vectors    : [N, 24]
            vina_scores     : [N]   — Vina docking scores
            experimental_dg : [N]   — experimental binding affinities (kcal/mol)
            hopfield_sims   : [N]   — Hopfield similarity scores (optional)
            epochs          : training epochs
            lr              : learning rate
            val_split       : fraction held out for validation
        """
        N     = len(pose_vectors)
        n_val = max(1, int(N * val_split))
        perm  = torch.randperm(N)

        val_idx   = perm[:n_val]
        train_idx = perm[n_val:]

        def _split(t):
            if t is None: return None, None
            return t[train_idx], t[val_idx]

        pv_tr,  pv_va  = _split(pose_vectors)
        vs_tr,  vs_va  = _split(vina_scores)
        dg_tr,  dg_va  = _split(experimental_dg)
        hs_tr,  hs_va  = _split(hopfield_sims)

        optim = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4)

        best_val_loss = float("inf")
        best_state    = None
        self.train_losses = []

        self.train()

        for epoch in range(epochs):
            optim.zero_grad()
            pred    = self.forward(pv_tr, vs_tr, hs_tr)
            loss    = F.mse_loss(pred, dg_tr)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            optim.step()
            self.train_losses.append(loss.item())

            if epoch % 20 == 0:
                self.eval()
                with torch.no_grad():
                    v_pred = self.forward(pv_va, vs_va, hs_va)
                    v_loss = F.mse_loss(v_pred, dg_va).item()
                self.train()

                if v_loss < best_val_loss:
                    best_val_loss = v_loss
                    best_state    = {k: v.clone() for k, v in self.state_dict().items()}

                print(
                    f"  Scorer epoch {epoch:3d}/{epochs} | "
                    f"train={loss.item():.4f} | val={v_loss:.4f} | "
                    f"w_vina={self.w_vina:.3f} w_learned={self.w_learned:.3f}"
                )

        if best_state:
            self.load_state_dict(best_state)

        self.is_trained = True
        self.eval()
        print(f"[Scorer] Training complete. Best val loss = {best_val_loss:.4f}")

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def weight_summary(self) -> str:
        return (
            f"EnsembleScorer weights | "
            f"Vina: {self.w_vina:.4f} | "
            f"Learned: {self.w_learned:.4f} | "
            f"Hopfield: {self.w_hopfield:.4f}"
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str):
        torch.save(
            {
                "state_dict" : self.state_dict(),
                "is_trained" : self.is_trained,
                "train_losses": self.train_losses,
                "config"     : {"pose_dim": self.pose_dim},
            },
            path,
        )
        print(f"[Scorer] Saved to {path}")

    @classmethod
    def load(cls, path: str) -> "EnsembleScorer":
        ckpt = torch.load(path, map_location="cpu")
        obj  = cls(**ckpt["config"])
        obj.load_state_dict(ckpt["state_dict"])
        obj.is_trained  = ckpt.get("is_trained", False)
        obj.train_losses = ckpt.get("train_losses", [])
        obj.eval()
        print(f"[Scorer] Loaded from {path}")
        return obj


# ------------------------------------------------------------------
# Inline unit tests
# ------------------------------------------------------------------

if __name__ == "__main__":
    print("=== EnsembleScorer Unit Tests ===\n")
    torch.manual_seed(42)

    B = 20
    scorer      = EnsembleScorer()
    poses       = torch.randn(B, 24)
    vina        = torch.rand(B) * (-15) - 2   # realistic: -2 to -17 kcal/mol
    hopfield    = torch.rand(B)

    # Test 1: forward shape
    scores = scorer(poses, vina, hopfield)
    assert scores.shape == (B,), f"Score shape: {scores.shape}"
    print("PASS: forward() returns [B] scores")

    # Test 2: rank_poses returns top-k
    idx, sc = scorer.rank_poses(poses, vina, hopfield, top_k=5)
    assert len(idx) == 5, "rank_poses should return 5"
    assert (sc[:-1] <= sc[1:]).all(), "Scores not sorted ascending"
    print("PASS: rank_poses returns sorted top-k")

    # Test 3: Vina fallback (w_hopfield ≈ 0 initially)
    scores_no_hop = scorer(poses, vina, None)
    assert scores_no_hop.shape == (B,), "Fallback shape wrong"
    print("PASS: works without hopfield_sims")

    # Test 4: ScoredPose packaging
    results = scorer.score_and_package(poses, vina, hopfield, top_k=10)
    assert len(results) == 10, "Should return 10 poses"
    assert results[0].rank == 1, "First pose rank should be 1"
    assert results[0].ensemble_score <= results[-1].ensemble_score, \
        "Poses not sorted by score"
    print("PASS: score_and_package returns sorted ScoredPose list")

    # Test 5: training on synthetic data
    N   = 300
    pv  = torch.randn(N, 24)
    vs  = torch.rand(N) * -15
    dg  = vs + torch.randn(N) * 0.5   # synthetic: noisy version of vina
    hs  = torch.rand(N)
    scorer2 = EnsembleScorer()
    scorer2.fit(pv, vs, dg, hs, epochs=50)
    assert scorer2.is_trained, "Scorer should be marked trained"
    print(f"PASS: training completed | {scorer2.weight_summary()}")

    # Test 6: save/load
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    scorer2.save(path)
    scorer3 = EnsembleScorer.load(path)
    s1 = scorer2(pv[:5], vs[:5])
    s2 = scorer3(pv[:5], vs[:5])
    assert torch.allclose(s1, s2, atol=1e-5), "Scores differ after reload"
    os.unlink(path)
    print("PASS: save/load preserves scores exactly")

    print("\n=== ALL TESTS PASSED ===")
