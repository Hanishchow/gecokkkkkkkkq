"""
core/attention_vae.py — AttentionPoseVAE
Pocket-conditioned β-VAE with torsion self-attention encoder.

Two upgrades over the plain PoseVAE in vae.py:

UPGRADE 1 — Torsion self-attention encoder:
  Plain VAE encoder: Linear(24→64) — treats all DOF as independent.
  AttentionVAE encoder: self-attention over the 18 torsion dimensions.
  Torsion angles are NOT independent: rotating bond i affects the
  accessible range of bond i+1 (steric clash chains). Attention
  captures these pairwise dependencies.

  Scientific test: AttentionVAE reconstruction MSE < PlainVAE on
  molecules with >8 rotatable bonds (where dependencies matter most).

UPGRADE 2 — Pocket conditioning:
  Plain VAE: samples from unconditional prior N(0, I).
  AttentionVAE: decoder is conditioned on PocketGNN embedding.
  p(z | pocket) = N(μ_pocket, σ_pocket) — pocket-specific prior.

  Scientific test: pocket-conditioned samples have lower Vina score
  vs unconditioned samples on the same pocket (when real scoring added).

Architecture:
  Encoder:
    rigid DOF [6D] → Linear → h_rigid [16D]
    torsions [18D] → reshape [18, 1] → TransformerEncoder → pool → h_torsion [16D]
    cat([h_rigid, h_torsion]) [32D] → fc_mu, fc_logvar → μ, logσ² [latent_dim]

  Decoder (conditioned on pocket embedding):
    z [latent_dim] + pocket_emb [32D] → Linear → 32D → 64D → 24D

  Pocket prior network:
    pocket_emb [32D] → μ_prior [latent_dim], logσ²_prior [latent_dim]
    Used for KL regularization toward pocket-specific prior.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional

from geock.config import POSE_DIM, LATENT_DIM
from geock.core.gnn import POCKET_EMB_DIM

N_RIGID    = 6
N_TORSIONS = POSE_DIM - N_RIGID   # 18


class TorsionAttentionEncoder(nn.Module):
    """
    Self-attention over torsion angle dimensions.

    Treats each of the 18 torsion angles as a "token" with a 1D value.
    A learned positional embedding gives each torsion bond an identity.
    Multi-head attention captures pairwise torsion dependencies.

    Why 2 heads: 18 torsion tokens, head_dim = 8 each → captures
    local (adjacent) and global (distal) interactions simultaneously.
    """

    def __init__(
        self,
        n_torsions:  int = N_TORSIONS,   # 18
        d_model:     int = 16,
        n_heads:     int = 2,
        out_dim:     int = 16,
    ):
        super().__init__()
        self.n_torsions = n_torsions
        self.d_model    = d_model

        # Project each scalar torsion value to d_model
        self.value_proj = nn.Linear(1, d_model)

        # Learnable positional embedding: bond position → embedding
        self.pos_embed = nn.Embedding(n_torsions, d_model)

        # Transformer encoder layer
        self.attn = nn.TransformerEncoderLayer(
            d_model   = d_model,
            nhead     = n_heads,
            dim_feedforward = d_model * 4,
            dropout   = 0.1,
            batch_first = True,    # [B, seq, d_model]
            norm_first  = True,    # pre-LN for stability
        )

        # Pool attention output to fixed vector
        self.pool_proj = nn.Linear(d_model, out_dim)

    def forward(self, torsions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            torsions: [B, N_TORSIONS] — torsion angles
        Returns:
            h: [B, out_dim]
        """
        B = torsions.shape[0]

        # [B, 18, 1] → project to [B, 18, d_model]
        x = self.value_proj(torsions.unsqueeze(-1))

        # Add positional embedding
        pos = torch.arange(self.n_torsions, device=torsions.device)
        x   = x + self.pos_embed(pos).unsqueeze(0)   # [B, 18, d_model]

        # Self-attention
        x = self.attn(x)   # [B, 18, d_model]

        # Mean pooling over torsion sequence → [B, d_model]
        x = x.mean(dim=1)

        return self.pool_proj(x)   # [B, out_dim]


class AttentionEncoder(nn.Module):
    """
    Encodes pose vector to (μ, logσ²) using:
      - MLP for rigid DOF (translation + rotation)
      - Self-attention for torsion DOF
      - Fusion of both streams
    """

    def __init__(
        self,
        pose_dim:   int = POSE_DIM,
        latent_dim: int = LATENT_DIM,
        rigid_hidden: int = 16,
        torsion_out:  int = 16,
    ):
        super().__init__()
        self.pose_dim   = pose_dim
        self.latent_dim = latent_dim

        # Rigid DOF stream
        self.rigid_net = nn.Sequential(
            nn.Linear(N_RIGID, rigid_hidden),
            nn.LayerNorm(rigid_hidden),
            nn.ELU(),
        )

        # Torsion stream (attention)
        self.torsion_attn = TorsionAttentionEncoder(
            n_torsions = N_TORSIONS,
            d_model    = 16,
            n_heads    = 2,
            out_dim    = torsion_out,
        )

        # Fusion
        fused_dim = rigid_hidden + torsion_out
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.ELU(),
        )

        self.fc_mu     = nn.Linear(fused_dim, latent_dim)
        self.fc_logvar = nn.Linear(fused_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, pose_dim]
        Returns:
            mu, logvar: [B, latent_dim] each
        """
        rigid    = x[:, :N_RIGID]           # [B, 6]
        torsions = x[:, N_RIGID:]           # [B, 18]

        h_rigid    = self.rigid_net(rigid)          # [B, 16]
        h_torsions = self.torsion_attn(torsions)    # [B, 16]

        h      = torch.cat([h_rigid, h_torsions], dim=-1)   # [B, 32]
        h      = self.fusion(h)

        mu     = self.fc_mu(h)
        logvar = torch.clamp(self.fc_logvar(h), min=-10, max=10)
        return mu, logvar


class PocketConditionedDecoder(nn.Module):
    """
    Decodes latent z conditioned on pocket GNN embedding.

    Architecture:
      cat([z, pocket_emb]) → 32+latent_dim → 64 → pose_dim

    If no pocket embedding provided (e.g., at test time with no receptor),
    falls back to zero pocket embedding (equivalent to unconditioned decoding).
    """

    def __init__(
        self,
        latent_dim:  int = LATENT_DIM,
        pocket_dim:  int = POCKET_EMB_DIM,
        pose_dim:    int = POSE_DIM,
        hidden:      int = 64,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.pocket_dim = pocket_dim

        self.net = nn.Sequential(
            nn.Linear(latent_dim + pocket_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden),
            nn.ELU(),
            nn.Linear(hidden, pose_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        z:           torch.Tensor,            # [B, latent_dim]
        pocket_emb:  Optional[torch.Tensor],  # [32] or [B, 32] or None
    ) -> torch.Tensor:
        B = z.shape[0]

        if pocket_emb is None:
            pocket_emb = torch.zeros(B, self.pocket_dim, device=z.device)
        elif pocket_emb.dim() == 1:
            pocket_emb = pocket_emb.unsqueeze(0).expand(B, -1)

        inp = torch.cat([z, pocket_emb], dim=-1)   # [B, latent+pocket]
        return self.net(inp)


class PocketPriorNet(nn.Module):
    """
    Pocket-specific prior: given pocket embedding, predict μ_prior and σ_prior.
    The KL term regularizes toward this pocket-specific prior instead of N(0,I).

    This makes the latent space pocket-aware: different pockets
    occupy different regions of latent space.
    """

    def __init__(self, pocket_dim: int = POCKET_EMB_DIM, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(pocket_dim, latent_dim * 2),
            nn.ELU(),
        )
        self.fc_mu     = nn.Linear(latent_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(latent_dim * 2, latent_dim)
        # Init so prior starts near N(0,I)
        nn.init.zeros_(self.fc_mu.weight); nn.init.zeros_(self.fc_mu.bias)
        nn.init.zeros_(self.fc_logvar.weight); nn.init.zeros_(self.fc_logvar.bias)

    def forward(self, pocket_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """pocket_emb: [32] or [B, 32] → μ_prior, logvar_prior"""
        if pocket_emb.dim() == 1:
            pocket_emb = pocket_emb.unsqueeze(0)
        h = self.net(pocket_emb)
        return self.fc_mu(h), torch.clamp(self.fc_logvar(h), -5, 5)


class AttentionPoseVAE(nn.Module):
    """
    Full pocket-conditioned β-VAE with torsion self-attention.

    Replaces PoseVAE in the pipeline when:
      1. A PocketGNN embedding is available (real docking with receptor)
      2. Ligands have many rotatable bonds (where attention helps most)

    Both can still fall back to PoseVAE for:
      - Testing without receptor
      - Ablation study (paper Table: "no GNN conditioning")

    Training:
      Loss = MSE recon + β * KL(q(z|x, pocket) || p(z|pocket))
      where p(z|pocket) is the pocket-specific prior.
    """

    def __init__(
        self,
        pose_dim:   int   = POSE_DIM,
        latent_dim: int   = LATENT_DIM,
        pocket_dim: int   = POCKET_EMB_DIM,
        beta:       float = 2.0,
    ):
        super().__init__()
        self.pose_dim   = pose_dim
        self.latent_dim = latent_dim
        self.pocket_dim = pocket_dim
        self.beta       = beta

        self.encoder     = AttentionEncoder(pose_dim, latent_dim)
        self.decoder     = PocketConditionedDecoder(latent_dim, pocket_dim, pose_dim)
        self.prior_net   = PocketPriorNet(pocket_dim, latent_dim)

        # Normalisation stats (set at pre-training time)
        self.register_buffer("pose_mean", torch.zeros(pose_dim))
        self.register_buffer("pose_std",  torch.ones(pose_dim))
        self._normalised = False

        # Training metrics
        self.train_losses: list = []
        self.val_losses:   list = []

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not torch.is_grad_enabled():
            return mu
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def forward(
        self,
        x:          torch.Tensor,
        pocket_emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns: (recon, mu, logvar, mu_prior, logvar_prior)
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z, pocket_emb)

        # Pocket prior
        if pocket_emb is not None:
            mu_prior, logvar_prior = self.prior_net(pocket_emb)
            # Broadcast to batch size
            B = mu.shape[0]
            if mu_prior.shape[0] == 1:
                mu_prior     = mu_prior.expand(B, -1)
                logvar_prior = logvar_prior.expand(B, -1)
        else:
            mu_prior     = torch.zeros_like(mu)
            logvar_prior = torch.zeros_like(logvar)

        return recon, mu, logvar, mu_prior, logvar_prior

    def loss(
        self,
        x:            torch.Tensor,
        recon:        torch.Tensor,
        mu:           torch.Tensor,
        logvar:       torch.Tensor,
        mu_prior:     torch.Tensor,
        logvar_prior: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        β-VAE loss with pocket-specific prior.

        KL(q(z|x,pocket) || p(z|pocket)):
          = ½ Σ [logσ_p² - logσ_q² + (σ_q² + (μ_q - μ_p)²)/σ_p² - 1]
        """
        recon_loss = F.mse_loss(recon, x, reduction="mean")

        # KL toward pocket-specific prior (not always N(0,I))
        var_prior  = logvar_prior.exp()
        var_q      = logvar.exp()
        kl = 0.5 * torch.mean(
            logvar_prior - logvar
            + (var_q + (mu - mu_prior).pow(2)) / var_prior.clamp(min=1e-8)
            - 1.0
        )
        total = recon_loss + self.beta * kl
        return total, recon_loss, kl

    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------

    def set_normalisation(self, mean: torch.Tensor, std: torch.Tensor):
        self.pose_mean.copy_(mean)
        self.pose_std.copy_(std.clamp(min=1e-8))
        self._normalised = True

    def normalise(self, x):
        return (x - self.pose_mean) / self.pose_std if self._normalised else x

    def denormalise(self, x):
        return x * self.pose_std + self.pose_mean if self._normalised else x

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        n:           int,
        pocket_emb:  Optional[torch.Tensor] = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Sample n poses, optionally conditioned on pocket_emb.

        If pocket_emb provided: sample from pocket-specific prior.
        If None: sample from N(0, I).
        """
        if pocket_emb is not None:
            mu_p, logvar_p = self.prior_net(pocket_emb)
            mu_p     = mu_p.expand(n, -1)
            std_p    = torch.exp(0.5 * logvar_p).expand(n, -1) * temperature
            z        = mu_p + torch.randn_like(mu_p) * std_p
        else:
            z = torch.randn(n, self.latent_dim) * temperature

        recon = self.decoder(z, pocket_emb)
        return self.denormalise(recon)

    @torch.no_grad()
    def encode_to_latent(self, x: torch.Tensor) -> torch.Tensor:
        mu, _ = self.encoder(self.normalise(x))
        return mu

    def reconstruction_error(self, x: torch.Tensor, pocket_emb=None) -> float:
        x_n  = self.normalise(x)
        mu, logvar = self.encoder(x_n)
        z    = mu   # deterministic encode
        recon = self.decoder(z, pocket_emb)
        return F.mse_loss(recon, x_n).item()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        train_data:   torch.Tensor,
        val_data:     Optional[torch.Tensor] = None,
        pocket_embs:  Optional[torch.Tensor] = None,   # [N, 32] or None
        epochs:       int   = 150,
        batch_size:   int   = 256,
        lr:           float = 1e-3,
        patience:     int   = 20,
    ):
        mean = train_data.mean(0); std = train_data.std(0)
        self.set_normalisation(mean, std)
        train_norm = self.normalise(train_data)
        val_norm   = self.normalise(val_data) if val_data is not None else None

        optim = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)

        best_val, best_state, patience_count = float("inf"), None, 0
        N = len(train_norm)

        self.train()
        for epoch in range(epochs):
            perm  = torch.randperm(N)
            eloss = 0.0; nb = 0
            for i in range(0, N, batch_size):
                idx   = perm[i:i+batch_size]
                batch = train_norm[idx]
                pemb  = pocket_embs[idx] if pocket_embs is not None else None
                optim.zero_grad()
                recon, mu, logvar, mu_p, logvar_p = self.forward(batch, pemb)
                total, _, _ = self.loss(batch, recon, mu, logvar, mu_p, logvar_p)
                total.backward()
                nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optim.step()
                eloss += total.item(); nb += 1
            sched.step()
            self.train_losses.append(eloss / nb)

            if val_norm is not None:
                self.eval()
                with torch.no_grad():
                    vr, vm, vl, vmp, vlp = self.forward(val_norm)
                    vt, _, _ = self.loss(val_norm, vr, vm, vl, vmp, vlp)
                    vt = vt.item()
                self.val_losses.append(vt)
                self.train()
                if vt < best_val:
                    best_val = vt
                    best_state = {k: v.clone() for k, v in self.state_dict().items()}
                    patience_count = 0
                else:
                    patience_count += 1
                if epoch % 10 == 0:
                    print(f"  AttentionVAE epoch {epoch}/{epochs} | "
                          f"train={self.train_losses[-1]:.4f} | val={vt:.4f}")
                if patience_count >= patience:
                    print(f"  [AttentionVAE] Early stop at epoch {epoch}")
                    break
            else:
                if epoch % 10 == 0:
                    print(f"  AttentionVAE epoch {epoch}/{epochs} | "
                          f"train={self.train_losses[-1]:.4f}")

        if best_state:
            self.load_state_dict(best_state)
        self.eval()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str):
        torch.save({
            "state_dict": self.state_dict(),
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "config": {
                "pose_dim": self.pose_dim, "latent_dim": self.latent_dim,
                "pocket_dim": self.pocket_dim, "beta": self.beta,
            },
        }, path)
        print(f"[AttentionPoseVAE] Saved to {path}")

    @classmethod
    def load(cls, path: str) -> "AttentionPoseVAE":
        ckpt = torch.load(path, map_location="cpu")
        vae  = cls(**ckpt["config"])
        vae.load_state_dict(ckpt["state_dict"])
        vae.train_losses = ckpt.get("train_losses", [])
        vae.val_losses   = ckpt.get("val_losses",   [])
        vae.eval()
        print(f"[AttentionPoseVAE] Loaded from {path}")
        return vae


# ------------------------------------------------------------------
# Unit tests
# ------------------------------------------------------------------

if __name__ == "__main__":
    print("=== AttentionPoseVAE Unit Tests ===\n")
    torch.manual_seed(42)

    B = 8
    vae  = AttentionPoseVAE()
    x    = torch.randn(B, POSE_DIM)
    pemb = torch.randn(POCKET_EMB_DIM)

    # Test 1: forward pass shapes (with pocket)
    recon, mu, logvar, mu_p, logvar_p = vae(x, pemb)
    assert recon.shape   == (B, POSE_DIM)
    assert mu.shape      == (B, LATENT_DIM)
    assert logvar.shape  == (B, LATENT_DIM)
    assert mu_p.shape    == (B, LATENT_DIM)
    print("PASS: forward shapes with pocket embedding")

    # Test 2: forward pass shapes (without pocket)
    recon2, mu2, logvar2, mu_p2, logvar_p2 = vae(x, None)
    assert recon2.shape == (B, POSE_DIM)
    print("PASS: forward shapes without pocket embedding")

    # Test 3: loss is finite and positive
    total, rl, kl = vae.loss(x, recon, mu, logvar, mu_p, logvar_p)
    assert torch.isfinite(total) and total.item() > 0
    print(f"PASS: loss={total.item():.4f} (recon={rl.item():.4f}, kl={kl.item():.4f})")

    # Test 4: unconditional vs conditioned sampling differ
    vae.eval()
    s_uncond = vae.sample(20, pocket_emb=None)
    s_cond   = vae.sample(20, pocket_emb=pemb)
    assert s_uncond.shape == s_cond.shape == (20, POSE_DIM)
    assert not torch.allclose(s_uncond, s_cond, atol=1e-3), \
        "Conditioned and unconditioned samples should differ"
    print("PASS: pocket conditioning changes sample distribution")

    # Test 5: attention encoder captures torsion structure
    # Two poses with same rigid DOF but different torsions → different latents
    x1 = torch.zeros(1, POSE_DIM); x1[0, 6:] = torch.rand(N_TORSIONS)
    x2 = torch.zeros(1, POSE_DIM); x2[0, 6:] = torch.rand(N_TORSIONS)
    mu1, _ = vae.encoder(x1)
    mu2, _ = vae.encoder(x2)
    assert not torch.allclose(mu1, mu2), "Different torsions → same latent (attention broken)"
    print("PASS: torsion attention distinguishes different torsion patterns")

    # Test 6: training runs
    data = torch.randn(200, POSE_DIM)
    vae2 = AttentionPoseVAE()
    vae2.fit(data, epochs=10, batch_size=32)
    assert len(vae2.train_losses) > 0
    print(f"PASS: training completes | final loss={vae2.train_losses[-1]:.4f}")

    # Test 7: save/load
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f: path = f.name
    vae2.save(path)
    vae3 = AttentionPoseVAE.load(path)
    for (n1,p1),(n2,p2) in zip(vae2.named_parameters(), vae3.named_parameters()):
        assert torch.allclose(p1, p2), f"Param {n1} differs"
    os.unlink(path)
    print("PASS: save/load preserves all parameters")

    print("\n=== ALL TESTS PASSED ===")
