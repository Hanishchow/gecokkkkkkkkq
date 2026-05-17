"""
config.py — Central configuration for GEOCK/DNBAP docking engine.

All tuneable parameters live here. Never scatter magic numbers in pipeline files.
Defaults are scientifically motivated — see comments for justification.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple
import os


# ------------------------------------------------------------------
# Pose representation
# ------------------------------------------------------------------

POSE_DIM       = 24     # 3 translation + 3 rotation + 18 torsion (padded to 95th pct)
MAX_TORSIONS   = 18     # Torsion dimensions (POSE_DIM - 6)
LATENT_DIM     = 8      # VAE latent space dimensionality


# ------------------------------------------------------------------
# Weight file paths (set via environment or config object)
# ------------------------------------------------------------------

DEFAULT_WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights")

SOM_WEIGHTS_PATH       = os.path.join(DEFAULT_WEIGHTS_DIR, "som_weights.pt")
VAE_WEIGHTS_PATH       = os.path.join(DEFAULT_WEIGHTS_DIR, "vae_weights.pt")
HOPFIELD_WEIGHTS_PATH  = os.path.join(DEFAULT_WEIGHTS_DIR, "hopfield_memories.pt")
SCORER_WEIGHTS_PATH    = os.path.join(DEFAULT_WEIGHTS_DIR, "scorer_weights.pt")


@dataclass
class SOMConfig:
    """Self-Organizing Map configuration."""
    grid_h:           int   = 8       # 8×8 = 64 neurons
    grid_w:           int   = 8
    pose_dim:         int   = POSE_DIM
    initial_lr:       float = 0.5
    initial_sigma:    float = 3.0
    # Pre-training
    pretrain_epochs:  int   = 100
    pretrain_batch:   int   = 256
    # Per-job fine-tuning
    finetune_epochs:  int   = 10
    finetune_batch:   int   = 32


@dataclass
class HopfieldConfig:
    """Modern Hopfield Network configuration."""
    pose_dim:          int   = POSE_DIM
    max_memories:      int   = 1000    # Top-1000 diverse binding modes from PDBbind
    beta:              float = 8.0     # Retrieval sharpness
    novelty_threshold: float = 0.85   # < threshold → novel → keep


@dataclass
class VAEConfig:
    """Pose VAE configuration."""
    pose_dim:    int   = POSE_DIM
    latent_dim:  int   = LATENT_DIM
    beta:        float = 2.0          # β-VAE: β=2 encourages mild disentanglement
    # Training
    epochs:      int   = 150
    batch_size:  int   = 512
    lr:          float = 1e-3
    patience:    int   = 20           # early stopping
    val_split:   float = 0.1


@dataclass
class ScorerConfig:
    """Ensemble scorer configuration."""
    pose_dim:    int   = POSE_DIM
    epochs:      int   = 200
    lr:          float = 1e-3


@dataclass
class Stage1Config:
    """Monte Carlo sampling configuration."""
    n_poses:          int   = 2000    # raw poses from MC search
    vina_exhaustiveness: int = 8      # Vina exhaustiveness parameter
    # Box defaults — ALWAYS override with receptor pocket coords
    box_size:         Tuple[float,float,float] = (20.0, 20.0, 20.0)  # Angstroms


@dataclass
class Stage2Config:
    """SOM clustering configuration."""
    n_poses_in:  int = 2000   # from Stage 1
    n_clusters:  int = 64     # = SOM grid size (8×8)


@dataclass
class Stage3Config:
    """VAE + Hopfield generation configuration."""
    n_vae_samples:   int   = 50    # VAE generates 50 candidates
    target_novel:    int   = 20    # keep ~20 after Hopfield filter
    temperature:     float = 1.0   # VAE sampling temperature


@dataclass
class Stage4Config:
    """SOM-neighborhood MC refinement configuration."""
    n_candidates:     int = 20    # top poses from Stage 3
    mc_steps:         int = 500   # MC steps per candidate
    neighbor_radius:  int = 1     # SOM neighbor radius for guided moves


@dataclass
class Stage5Config:
    """Ensemble scoring configuration."""
    top_k: int = 20   # return top-20 ranked poses


@dataclass
class DockingConfig:
    """
    Master configuration for a single docking run.

    Critical fields:
      box_center: MUST be provided for virtual screening.
                  Use fpocket / DoGSiteScorer to find it.
                  Raises ValueError if missing.

      receptor_path: Path to receptor .pdbqt file
      ligand_path  : Path to ligand .pdbqt/.sdf/.mol2 file
    """
    # Required
    receptor_path: str = ""
    ligand_path:   str = ""

    # Box center — receptor pocket coords (NOT ligand centroid)
    # Must be set explicitly. No default — forces user to think about it.
    box_center: Optional[Tuple[float, float, float]] = None

    # Sub-configs
    stage1: Stage1Config = field(default_factory=Stage1Config)
    stage2: Stage2Config = field(default_factory=Stage2Config)
    stage3: Stage3Config = field(default_factory=Stage3Config)
    stage4: Stage4Config = field(default_factory=Stage4Config)
    stage5: Stage5Config = field(default_factory=Stage5Config)

    som:      SOMConfig      = field(default_factory=SOMConfig)
    hopfield: HopfieldConfig = field(default_factory=HopfieldConfig)
    vae:      VAEConfig      = field(default_factory=VAEConfig)
    scorer:   ScorerConfig   = field(default_factory=ScorerConfig)

    # Weight paths
    weights_dir:      str = DEFAULT_WEIGHTS_DIR
    som_weights:      str = SOM_WEIGHTS_PATH
    vae_weights:      str = VAE_WEIGHTS_PATH
    hopfield_weights: str = HOPFIELD_WEIGHTS_PATH
    scorer_weights:   str = SCORER_WEIGHTS_PATH

    # Runtime flags
    verbose: bool = True
    device:  str  = "cpu"   # "cuda" if available

    def validate(self):
        """
        Hard validation before any docking run starts.
        Raises ValueError immediately on invalid config.
        """
        errors = []

        if not self.receptor_path:
            errors.append("receptor_path is required")
        elif not os.path.exists(self.receptor_path):
            errors.append(f"receptor_path not found: {self.receptor_path}")

        if not self.ligand_path:
            errors.append("ligand_path is required")
        elif not os.path.exists(self.ligand_path):
            errors.append(f"ligand_path not found: {self.ligand_path}")

        if self.box_center is None:
            errors.append(
                "box_center is required. "
                "Run fpocket or DoGSiteScorer on your receptor to find pocket coordinates. "
                "Do NOT use the ligand centroid for virtual screening."
            )
        elif len(self.box_center) != 3:
            errors.append("box_center must be (x, y, z) — 3 floats")

        if errors:
            raise ValueError(
                "DockingConfig validation failed:\n" +
                "\n".join(f"  • {e}" for e in errors)
            )

    def get_box_center(self) -> Tuple[float, float, float]:
        """
        Return validated box center. Hard error if not set.
        This is the fix for BUG 1 — no silent ligand centroid fallback.
        """
        if self.box_center is None:
            raise ValueError(
                "_get_box_center() called without box_center set. "
                "This is a critical bug. Set config.box_center from "
                "fpocket / DoGSiteScorer output before calling docking."
            )
        return tuple(self.box_center)


@dataclass
class PretrainConfig:
    """Configuration for the one-time PDBbind pre-training run."""
    pdbind_dir:    str = ""        # path to PDBbind refined set
    output_dir:    str = DEFAULT_WEIGHTS_DIR
    device:        str = "cpu"
    val_fraction:  float = 0.1

    som:     SOMConfig     = field(default_factory=SOMConfig)
    vae:     VAEConfig     = field(default_factory=VAEConfig)
    hopfield: HopfieldConfig = field(default_factory=HopfieldConfig)
    scorer:  ScorerConfig  = field(default_factory=ScorerConfig)


# ------------------------------------------------------------------
# Quick self-test
# ------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Config Unit Tests ===\n")

    # Test 1: default config builds without error
    cfg = DockingConfig()
    print(f"PASS: DockingConfig built | device={cfg.device}")

    # Test 2: validation catches missing fields
    try:
        cfg.validate()
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "receptor_path" in str(e)
        assert "box_center" in str(e)
        print("PASS: validate() catches missing receptor_path and box_center")

    # Test 3: validate passes with correct fields
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".pdbqt", delete=False) as f1:
        rec_path = f1.name
    with tempfile.NamedTemporaryFile(suffix=".pdbqt", delete=False) as f2:
        lig_path = f2.name

    cfg.receptor_path = rec_path
    cfg.ligand_path   = lig_path
    cfg.box_center    = (12.5, -3.2, 8.1)
    cfg.validate()   # should not raise
    print("PASS: validate() passes with valid config")

    os.unlink(rec_path)
    os.unlink(lig_path)

    # Test 4: get_box_center returns tuple
    bc = cfg.get_box_center()
    assert len(bc) == 3, "Box center should be 3-tuple"
    print(f"PASS: get_box_center() = {bc}")

    # Test 5: None box_center raises
    cfg2 = DockingConfig()
    try:
        cfg2.get_box_center()
        raise AssertionError("Should have raised")
    except ValueError:
        print("PASS: get_box_center() raises on None")

    print("\n=== ALL TESTS PASSED ===")
