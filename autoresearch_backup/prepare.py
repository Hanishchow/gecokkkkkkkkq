"""
prepare.py — GEOCK AutoResearch (Simplified)
============================================
Uses 20 compounds (10 train, 5 val, 5 test) for fast experiments.

KNOWN ISSUES (do not modify without fixing all):
  1. Sequential split (first 10 train, 11-15 val, 16-20 test) does not
     account for protein family diversity. Should be random split with
     stratification by family if family info is available.
  2. bio_engine.run_all_compounds() IntegrationFilter was previously fit on
     ALL compounds (including val/test) — DATA LEAKAGE. This is fixed in
     bio_engine.py (train_mask parameter). Ensure train_mask is used.
  3. X_filt was set to X_raw (workaround) — confusing naming.
     IntegrationFilter now only fits on training compounds when train_mask provided.
  4. extract_fast.py (features_110.pkl) uses biological_features_simple which
     generates different features than bio_engine.py biological_features.
     SMILES-based features (drug_likeness) work correctly. VQE and 3D
     conformer features are skipped for speed.
"""

import os
import sys
import json
import pickle
import warnings
import numpy as np

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_DIR   = "/mnt/c/Users/yakka/Downloads/geock_110_data"
CACHE_DIR  = os.path.expanduser("~/.cache/geock_autoresearch")
CACHE_FILE = os.path.join(CACHE_DIR, "features_v2.pkl")

# ── Fixed splits ───────────────────────────────────────────────────────────
TRAIN_N    = 10    # training compounds (first 10)
VAL_N      = 5     # validation compounds (11-15)
TEST_N     = 5     # held-out test (16-20)
TOTAL_N    = TRAIN_N + VAL_N + TEST_N  # 20 total

# ── Metric ─────────────────────────────────────────────────────────────────
BASELINE_R = 0.70   # Adjusted for smaller dataset

os.makedirs(CACHE_DIR, exist_ok=True)

# ── RDKit ──────────────────────────────────────────────────────────────────
try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
    RDKIT_OK = True
except ImportError:
    RDKIT_OK = False


def get_ecfp(smiles: str, fp_size: int = 512) -> np.ndarray:
    """ECFP4 fingerprint."""
    if not RDKIT_OK or not smiles:
        return np.zeros(fp_size)
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(fp_size)
        fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=fp_size)
        return np.array(fpgen.GetFingerprintAsNumPy(mol), dtype=np.float32)
    except:
        return np.zeros(fp_size, dtype=np.float32)


def compute_features_simple(compounds: list) -> dict:
    """Compute features using bio_engine with simpler settings."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from bio_engine import run_all_compounds
    
    # Run with use_quantum=False for speed (quantum is slow)
    X_filt, X_raw, y_raw, pdb_ids = run_all_compounds(
        compounds, DATA_DIR, use_quantum=False, verbose=True
    )
    
    # Fix NaNs
    X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)
    
    return {
        'X_raw': X_raw,
        'X_filt': X_filt,
        'y_raw': y_raw,
        'pdb_ids': pdb_ids
    }


def load_features(force_recompute: bool = False) -> dict:
    """Load or compute features."""
    if os.path.exists(CACHE_FILE) and not force_recompute:
        print(f"Loading cached features from {CACHE_FILE}")
        with open(CACHE_FILE, "rb") as f:
            return pickle.load(f)

    print("Computing features (this runs once)...")
    
    with open(os.path.join(DATA_DIR, "compounds.json")) as f:
        compounds_all = json.load(f)
    
    compounds = compounds_all[:TOTAL_N]
    
    # Compute features
    result = compute_features_simple(compounds)
    X_raw = result['X_raw']
    y_raw = result['y_raw']
    pdb_ids = result['pdb_ids']
    
    # Convert ΔG → pKd
    y_pkd = (-y_raw / 1.364).astype(np.float32)
    
    # ECFP4
    matched = {c["pdb_id"]: c for c in compounds}
    smiles_list = [matched.get(pid, {}).get("smiles", "") for pid in pdb_ids]
    X_ecfp = np.stack([get_ecfp(s) for s in smiles_list])
    
    data = {
        "X_raw": X_raw,
        "X_filt": X_raw,  # Use X_raw as filtered
        "X_ecfp": X_ecfp,
        "y_pkd": y_pkd,
        "pdb_ids": pdb_ids,
        "smiles_list": smiles_list,
        "n_train": TRAIN_N,
        "n_val": VAL_N,
        "n_test": TEST_N,
    }

    with open(CACHE_FILE, "wb") as f:
        pickle.dump(data, f)

    print(f"Cached to {CACHE_FILE}")
    print(f"  X_raw shape: {X_raw.shape}, X_ecfp shape: {X_ecfp.shape}")

    return data


def get_splits(data: dict) -> tuple:
    """Return train/val/test splits."""
    n_tr = data["n_train"]
    
    X_r = data["X_raw"]
    X_e = data["X_ecfp"]
    y = data["y_pkd"]
    
    X_all = np.hstack([X_r, X_e])
    
    X_train = X_all[:n_tr]
    y_train = y[:n_tr]
    
    X_val = X_all[n_tr:n_tr + VAL_N]
    y_val = y[n_tr:n_tr + VAL_N]
    
    X_test = X_all[n_tr + VAL_N:]
    y_test = y[n_tr + VAL_N:]
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def evaluate_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pearson R on validation set."""
    from scipy.stats import pearsonr
    if len(y_true) < 2:
        return 0.0
    r, _ = pearsonr(y_true, y_pred)
    return float(r) if not np.isnan(r) else 0.0


def evaluate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAE in pKd units."""
    return float(np.mean(np.abs(y_true - y_pred)))


# ── Feature names ─────────────────────────────────────────────────────────
FEATURE_NAMES = [
    "E1_vinardo_gauss1", "E1_vinardo_repulsion", "E1_vinardo_hydrophobic",
    "E1_vinardo_hbond", "E1_vinardo_torsion", "E1_vinardo_affinity",
    "E2_chem_pi_pi", "E2_chem_cation_pi", "E2_chem_salt_bridge",
    "E2_chem_halogen_bond", "E2_chem_metal_coord", "E2_chem_burial",
    "E2_chem_shape", "E2_chem_lipophilic",
    "E3_quantum_vqe",
    "E4_bio_drug_likeness", "E4_bio_ligand_efficiency",
    "E4_bio_pocket_druggability", "E4_bio_resolution_weight",
    "E4_bio_family_hydrophobic", "E4_bio_family_hbond",
    "E4_bio_pocket_polarity", "E4_bio_size_penalty", "E4_bio_pharmacophore",
    *[f"ecfp4_{i}" for i in range(512)],
]

if __name__ == "__main__":
    print("="*60)
    print("GEOCK AutoResearch — Data Prep (20 compounds)")
    print("="*60)
    
    data = load_features()
    X_train, y_train, X_val, y_val, X_test, y_test = get_splits(data)
    
    print(f"\nSplits: train={len(y_train)}, val={len(y_val)}, test={len(y_test)}")
    print(f"Features: {X_train.shape[1]} (24 physics + 512 ECFP)")
    print(f"Baseline R: {BASELINE_R}")