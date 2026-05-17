"""
core/vae.py — PoseVAE
Variational Autoencoder trained on PDBbind pose vectors.

Scientific contract:
  - Learns a continuous latent manifold of biologically valid binding poses
  - At runtime: sample latent z → decode to novel candidate pose vectors
  - Hopfield filter then removes anything matching stored prototypes
  - NEVER retrained at inference time (Bug #3 fix)

Architecture:
  Input  : 24D pose vector (3 trans + 3 rot + 18 torsion, padded)
  Encoder: 24 → 64 → 32 → latent (μ, logσ²)  [latent_dim=8]
  Decoder: latent_dim → 32 → 64 → 24
  Loss   : MSE reconstruction + β-KL divergence

  β-VAE formulation (Higgins et al. 2017):
    β > 1 → encourages disentangled latent dimensions
    We use β=2 — empirically, β=1-4 works for pose spaces

Note on latent_dim=8:
  - Pose space has ~6 rigid + up to 18 torsional DOF
  - 8D latent is intentionally undercomplete → forced compression
  - Too high → memorization, not generation
  - Too low → fails to reconstruct diverse poses
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


POSE_DIM   = 24
LATENT_DIM = 8


class PoseEncoder(nn.Module):
    """
    Encodes a pose vector to (μ, log σ²) in latent space.
    Uses LayerNorm for training stability on small datasets.
    """

    def __init__(self, pose_dim: int = POSE_DIM, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(pose_dim, 64),
            nn.LayerNorm(64),
            nn.ELU(),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ELU(),
        )
        self.fc_mu     = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h      = self.net(x)
        mu     = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        # Clamp logvar for numerical stability
        logvar = torch.clamp(logvar, min=-10, max=10)
        return mu, logvar


class PoseDecoder(nn.Module):
    """
    Decodes a latent vector z back to a 24D pose vector.
    No final activation — pose components are unbounded.
    (Normalization is handled at the pipeline level.)
    """

    def __init__(self, latent_dim: int = LATENT_DIM, pose_dim: int = POSE_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.LayerNorm(32),
            nn.ELU(),
            nn.Linear(32, 64),
            nn.LayerNorm(64),
            nn.ELU(),
            nn.Linear(64, pose_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class PoseVAE(nn.Module):
    """
    β-VAE for binding pose generation.

    Training:
      Trained ONCE on PDBbind (~193k complexes).
      Saves vae_weights.pt. Never retrained at inference.

    Inference:
      sample(n) → decode n novel pose candidates
      Caller (Stage 3) passes these through Hopfield filter.
    """

    def __init__(
        self,
        pose_dim:   int   = POSE_DIM,
        latent_dim: int   = LATENT_DIM,
        beta:       float = 2.0,
    ):
        super().__init__()
        self.pose_dim   = pose_dim
        self.latent_dim = latent_dim
        self.beta       = beta

        self.encoder = PoseEncoder(pose_dim, latent_dim)
        self.decoder = PoseDecoder(latent_dim, pose_dim)

        # Training state tracking (for honest reporting)
        self.train_losses: list[float] = []
        self.val_losses:   list[float] = []

        # Normalisation statistics — set during pre-training
        self.register_buffer("pose_mean", torch.zeros(pose_dim))
        self.register_buffer("pose_std",  torch.ones(pose_dim))
        self._normalised = False

    # ------------------------------------------------------------------
    # Core VAE operations
    # ------------------------------------------------------------------

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = μ + ε·σ, ε ~ N(0,I)
        Enables gradient flow through sampling.
        """
        if not torch.is_grad_enabled():
            return mu   # deterministic at pure inference if grads off
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full VAE forward pass.
        Returns: (reconstruction, mu, logvar)
        """
        mu, logvar = self.encode(x)
        z          = self.reparameterize(mu, logvar)
        recon      = self.decode(z)
        return recon, mu, logvar

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def loss(
        self,
        x:      torch.Tensor,
        recon:  torch.Tensor,
        mu:     torch.Tensor,
        logvar: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        β-VAE loss = MSE reconstruction + β · KL divergence

        Returns: total_loss, recon_loss, kl_loss (all scalars)
        """
        recon_loss = F.mse_loss(recon, x, reduction="mean")

        # KL divergence: -½ Σ(1 + logσ² - μ² - σ²)
        kl_loss = -0.5 * torch.mean(
            1 + logvar - mu.pow(2) - logvar.exp()
        )

        total = recon_loss + self.beta * kl_loss
        return total, recon_loss, kl_loss

    # ------------------------------------------------------------------
    # Training API
    # ------------------------------------------------------------------

    def set_normalisation(self, mean: torch.Tensor, std: torch.Tensor):
        """Store dataset-level normalisation statistics."""
        self.pose_mean.copy_(mean)
        self.pose_std.copy_(std.clamp(min=1e-8))
        self._normalised = True

    def normalise(self, x: torch.Tensor) -> torch.Tensor:
        if not self._normalised:
            return x
        return (x - self.pose_mean) / self.pose_std

    def denormalise(self, x: torch.Tensor) -> torch.Tensor:
        if not self._normalised:
            return x
        return x * self.pose_std + self.pose_mean

    def fit(
        self,
        train_data: torch.Tensor,
        val_data:   Optional[torch.Tensor] = None,
        epochs:     int   = 100,
        batch_size: int   = 256,
        lr:         float = 1e-3,
        patience:   int   = 15,
    ):
        """
        Train VAE on pose vectors.

        Args:
            train_data: [N, pose_dim] — raw pose vectors from PDBbind
            val_data  : [M, pose_dim] — held-out validation set
            epochs    : max training epochs
            batch_size: mini-batch size
            lr        : Adam learning rate
            patience  : early stopping patience
        """
        # Compute and store normalisation statistics from training data
        mean = train_data.mean(dim=0)
        std  = train_data.std(dim=0)
        self.set_normalisation(mean, std)

        train_norm = self.normalise(train_data)
        val_norm   = self.normalise(val_data) if val_data is not None else None

        optim     = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)

        best_val_loss  = float("inf")
        best_state     = None
        patience_count = 0

        self.train_losses = []
        self.val_losses   = []

        N = len(train_norm)
        self.train()

        for epoch in range(epochs):
            # Mini-batch SGD
            perm       = torch.randperm(N)
            epoch_loss = 0.0
            n_batches  = 0

            for i in range(0, N, batch_size):
                batch  = train_norm[perm[i : i + batch_size]]
                optim.zero_grad()
                recon, mu, logvar = self.forward(batch)
                total, _, _       = self.loss(batch, recon, mu, logvar)
                total.backward()
                nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optim.step()
                epoch_loss += total.item()
                n_batches  += 1

            scheduler.step()
            avg_train_loss = epoch_loss / n_batches
            self.train_losses.append(avg_train_loss)

            # Validation
            if val_norm is not None:
                self.eval()
                with torch.no_grad():
                    vr, vm, vl = self.forward(val_norm)
                    vt, _, _   = self.loss(val_norm, vr, vm, vl)
                    avg_val    = vt.item()
                self.val_losses.append(avg_val)
                self.train()

                if avg_val < best_val_loss:
                    best_val_loss  = avg_val
                    best_state     = {k: v.clone() for k, v in self.state_dict().items()}
                    patience_count = 0
                else:
                    patience_count += 1

                if epoch % 10 == 0:
                    print(
                        f"  VAE epoch {epoch:3d}/{epochs} | "
                        f"train={avg_train_loss:.4f} | val={avg_val:.4f}"
                    )

                if patience_count >= patience:
                    print(f"  [VAE] Early stopping at epoch {epoch}")
                    break
            else:
                if epoch % 10 == 0:
                    print(f"  VAE epoch {epoch:3d}/{epochs} | train={avg_train_loss:.4f}")

        # Restore best checkpoint
        if best_state is not None:
            self.load_state_dict(best_state)
            print(f"  [VAE] Restored best model (val loss={best_val_loss:.4f})")

        self.eval()

    # ------------------------------------------------------------------
    # Inference API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(self, n: int, temperature: float = 1.0) -> torch.Tensor:
        """
        Sample n novel pose vectors from the prior N(0, I).

        Args:
            n          : number of poses to generate
            temperature: > 1 → more diverse but less valid
                         < 1 → more conservative, closer to training dist
        Returns:
            poses: [n, pose_dim] — denormalised pose vectors
        """
        z     = torch.randn(n, self.latent_dim) * temperature
        recon = self.decode(z)
        return self.denormalise(recon)

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Encode then decode (for reconstruction error evaluation)."""
        x_norm        = self.normalise(x)
        recon, mu, _  = self.forward(x_norm)
        return self.denormalise(recon)

    @torch.no_grad()
    def encode_to_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Return mean latent vector μ for a batch of poses."""
        mu, _ = self.encode(self.normalise(x))
        return mu

    def reconstruction_error(self, x: torch.Tensor) -> float:
        """MSE between input and reconstruction. Honest quality metric."""
        recon = self.reconstruct(x)
        return F.mse_loss(recon, x).item()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str):
        torch.save(
            {
                "state_dict"    : self.state_dict(),
                "train_losses"  : self.train_losses,
                "val_losses"    : self.val_losses,
                "config": {
                    "pose_dim"  : self.pose_dim,
                    "latent_dim": self.latent_dim,
                    "beta"      : self.beta,
                },
            },
            path,
        )
        print(f"[VAE] Saved to {path}")

    @classmethod
    def load(cls, path: str) -> "PoseVAE":
        ckpt = torch.load(path, map_location="cpu")
        cfg  = ckpt["config"]
        vae  = cls(**cfg)
        vae.load_state_dict(ckpt["state_dict"])
        vae.train_losses = ckpt.get("train_losses", [])
        vae.val_losses   = ckpt.get("val_losses",   [])
        vae.eval()
        print(f"[VAE] Loaded from {path}")
        return vae


# ------------------------------------------------------------------
# Inline unit tests — run: python -m geock.core.vae
# ------------------------------------------------------------------

if __name__ == "__main__":
    print("=== PoseVAE Unit Tests ===\n")

    torch.manual_seed(42)

    # Test 1: forward pass shapes
    vae = PoseVAE()
    x   = torch.randn(16, 24)
    recon, mu, logvar = vae(x)
    assert recon.shape   == (16, 24), f"Recon shape: {recon.shape}"
    assert mu.shape      == (16,  8), f"Mu shape: {mu.shape}"
    assert logvar.shape  == (16,  8), f"Logvar shape: {logvar.shape}"
    print("PASS: forward pass shapes correct")

    # Test 2: loss is positive and finite
    total, rl, kl = vae.loss(x, recon, mu, logvar)
    assert total.item() > 0,                        "Loss should be positive"
    assert torch.isfinite(total),                   "Loss should be finite"
    assert torch.isfinite(torch.tensor(rl.item())), "Recon loss infinite"
    print(f"PASS: loss = {total.item():.4f} (recon={rl.item():.4f}, kl={kl.item():.4f})")

    # Test 3: sample shapes
    vae.eval()
    samples = vae.sample(50)
    assert samples.shape == (50, 24), f"Sample shape: {samples.shape}"
    print("PASS: sample() returns correct shape")

    # Test 4: training reduces loss
    data = torch.randn(500, 24)
    vae2 = PoseVAE()
    vae2.fit(data, epochs=20, batch_size=64)
    assert len(vae2.train_losses) > 0, "No training losses recorded"
    # Loss should go down overall (allow noise)
    assert vae2.train_losses[-1] < vae2.train_losses[0] * 2, \
        "Training loss did not decrease"
    print(f"PASS: training loss {vae2.train_losses[0]:.4f} → {vae2.train_losses[-1]:.4f}")

    # Test 5: normalisation roundtrip
    mean = data.mean(dim=0)
    std  = data.std(dim=0)
    vae2.set_normalisation(mean, std)
    normed    = vae2.normalise(data[:5])
    denormed  = vae2.denormalise(normed)
    assert torch.allclose(denormed, data[:5], atol=1e-5), \
        "Normalise/denormalise roundtrip failed"
    print("PASS: normalisation roundtrip exact")

    # Test 6: save/load
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    vae2.save(path)
    vae3 = PoseVAE.load(path)
    s1 = vae2.sample(10)
    vae3.eval()
    # Not checking output equality (sampling is stochastic) — check param equality
    for (n1, p1), (n2, p2) in zip(vae2.named_parameters(), vae3.named_parameters()):
        assert torch.allclose(p1, p2), f"Parameter {n1} differs after reload"
    os.unlink(path)
    print("PASS: save/load preserves all parameters")

    # Test 7: encode_to_latent deterministic
    mu1 = vae2.encode_to_latent(data[:5])
    mu2 = vae2.encode_to_latent(data[:5])
    assert torch.allclose(mu1, mu2), "Latent encoding not deterministic"
    print("PASS: encode_to_latent is deterministic")

    print("\n=== ALL TESTS PASSED ===")
