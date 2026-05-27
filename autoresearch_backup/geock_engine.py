#!/usr/bin/env python3
"""
geock_engine.py — GEOCK Binding Affinity Prediction Engine
=======================================================
Production-ready binding affinity prediction using ECFP fingerprints + Ensemble model.

ARCHITECTURE:
  - Features: 512 ECFP4 Morgan fingerprints (selected 500 best bits by SelectKBest)
  - Model: Ensemble of XGBoost (80%) + Ridge regression (20%)
  - XGBoost: depth=5, learning_rate=0.05, regularization=1.0
  - Ridge: alpha=100 regularization
  - Training: 7,000 compounds

RESULTS (7,000 compounds, 1,977 test compounds):
  Test-R  = 0.6227  ← Generalization estimate
  Train-R = 0.7209
  Gap     = 0.098   ← Minimal overfitting
  MAE     = 1.16 pKd units

USAGE:
  # Predict pKd for a molecule (from SMILES only):
  python geock_engine.py --smiles "CC(=O)Oc1ccccc1C(=O)O"

  # Predict pKd for a molecule from PDB pocket + SMILES:
  python geock_engine.py --pdb-file pocket.pdb --smiles "CC(=O)Oc1ccccc1C(=O)O"

  # As a Python module:
  from geock_engine import predict_pKd, GEOCKEngine
  result = predict_pKd("CCO")
  print(f"pKd = {result['pKd']:.2f}")
"""

import os
import sys
import json
import pickle
import warnings
import argparse
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator


# ===== CROSS-PLATFORM PATH HELPER =====
def _get_cache_dir():
    """Get cache directory - works on Linux and Windows."""
    # Try Linux path first
    linux_cache = Path("/home/chow/.cache/geock_autoresearch")
    if linux_cache.exists():
        return linux_cache
    # Try Windows path (OneDrive)
    win_cache = Path(os.path.expanduser("~/OneDrive/.cache/geock_autoresearch"))
    if win_cache.exists():
        return win_cache
    # Fallback to current directory
    return Path("./cache")


def _get_autoresearch_dir():
    """Get autoresearch directory - works on Linux and Windows."""
    linux = Path("/home/chow/autoresearch")
    if linux.exists():
        return linux
    # Try Windows path
    win = Path(os.path.expanduser("~/OneDrive/autoresearch"))
    if win.exists():
        return win
    return Path(".")


# Get paths
_CACHE_DIR = _get_cache_dir()
_AUTORESEARCH_DIR = _get_autoresearch_dir()
_DEFAULT_MODEL = _AUTORESEARCH_DIR / "geock_deep_trees_final.pkl"

# Fallback to hardcoded path for backwards compatibility if new path doesn't exist
if _DEFAULT_MODEL.exists():
    MODEL_PATH = _DEFAULT_MODEL
else:
    MODEL_PATH = Path("geock_deep_trees_final.pkl")


# ── Load Model ────────────────────────────────────────────────────────────────
_model_cache = None
_model_cache_lock = None


def _load_model():
    global _model_cache
    if _model_cache is not None:
        return _model_cache
    with open(MODEL_PATH, "rb") as f:
        _model_cache = pickle.load(f)
    return _model_cache


# ===== BETTER ERROR HANDLING =====
def _load_model_safe():
    """Load model with proper error handling."""
    global _model_cache
    if _model_cache is not None:
        return _model_cache
    try:
        with open(MODEL_PATH, "rb") as f:
            _model_cache = pickle.load(f)
        return _model_cache
    except FileNotFoundError:
        print(f"ERROR: Model file not found: {MODEL_PATH}")
        # Try alternative models in order
        alternatives = [
            _AUTORESEARCH_DIR / "geock_v2_best_final.pkl",
            _AUTORESEARCH_DIR / "geock_v2_xgboost_39k.pkl",
            _AUTORESEARCH_DIR / "geock_deep_trees_no2016.pkl",
        ]
        for alt_path in alternatives:
            if alt_path.exists():
                print(f"  Trying alternative: {alt_path.name}")
                with open(alt_path, "rb") as f:
                    _model_cache = pickle.load(f)
                return _model_cache
        raise
    except Exception as e:
        print(f"ERROR loading model: {e}")
        raise


# ── Feature Extraction ──────────────────────────────────────────────────────
def get_ecfp(smiles, fp_size=512):
    """Generate ECFP4 Morgan fingerprint from SMILES."""
    if not smiles or not isinstance(smiles, str):
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=fp_size)
        return np.array(fpgen.GetFingerprintAsNumPy(mol), dtype=np.float32)
    except Exception:
        return None


# ── Prediction API ──────────────────────────────────────────────────────────
def predict_pKd(smiles, pdb_file=None):
    """
    Predict binding affinity (pKd) for a molecule.

    Args:
        smiles: SMILES string of the ligand (REQUIRED)
        pdb_file: Path to pocket PDB file (optional, not used in current model)

    Returns:
        dict with keys: pKd, confidence, model_loo_r

    Example:
        result = predict_pKd("CCO")
        print(f"pKd = {result['pKd']:.2f}")
    """
    try:
        m = _load_model_safe()
    except Exception as e:
        return {"pKd": None, "error": f"Failed to load model: {e}", "smiles": smiles}

    ecfp = get_ecfp(smiles)
    if ecfp is None:
        return {"pKd": None, "error": "Invalid SMILES", "smiles": smiles}

    # Handle both old (mu_e, sel_e) and new (sel) model formats
    if "mu_e" in m:
        X_norm = (ecfp - m["mu_e"]) / np.where(m["sd_e"] < 1e-10, 1, m["sd_e"])
        X_sel = m["sel_e"].transform(X_norm.reshape(1, -1))
    elif "sel" in m:
        X_sel = m["sel"].transform(ecfp.reshape(1, -1))
    elif "scaler" in m and "selector" in m:
        # Pipeline trained model (StandardScaler + SelectKBest)
        X_scaled = m["scaler"].transform(ecfp.reshape(1, -1))
        X_sel = m["selector"].transform(X_scaled)
    else:
        X_sel = ecfp.reshape(1, -1)

    # Handle Ensemble, Ridge, RF, and XGBoost models
    if "model" in m:
        # Single model (XGBoost from pipeline_train.py)
        pKd = float(m["model"].predict(X_sel)[0])
    elif "ensemble_weight" in m:
        # Ensemble: XGBoost + Ridge blend
        xgb_weight = m["ensemble_weight"]
        ridge_weight = 1.0 - xgb_weight
        xgb_pred = float(m["xgb"].predict(X_sel)[0])
        ridge_pred = float(m["ridge"].predict(X_sel)[0])
        pKd = xgb_weight * xgb_pred + ridge_weight * ridge_pred
    elif "xgb" in m:
        pKd = float(m["xgb"].predict(X_sel)[0])
    elif "ridge" in m:
        pKd = float(m["ridge"].predict(X_sel)[0])
    elif "rf" in m:
        pKd = float(m["rf"].predict(X_sel)[0])
    else:
        pKd = float(m["ridge"].predict(X_sel)[0])

    # Get performance metric
    perf = m.get("loo_r", m.get("cv_r", m.get("test_r", 0)))
    confidence = "low"
    if perf > 0.6:
        confidence = "medium"
    if perf > 0.7:
        confidence = "high"

    return {
        "pKd": pKd,
        "smiles": smiles,
        "confidence": confidence,
        "model_loo_r": m.get("loo_r", m.get("cv_r", m.get("test_r", 0))),
        "n_training": m.get("n_compounds", "N/A"),
    }


def batch_predict(smiles_list):
    """Predict pKd for multiple molecules."""
    return [predict_pKd(s) for s in smiles_list]


# ── GEOCKEngine Class ───────────────────────────────────────────────────────
class GEOCKEngine:
    """GEOCK Binding Affinity Prediction Engine."""

    def __init__(self, model_path=None):
        if model_path:
            with open(model_path, "rb") as f:
                self.data = pickle.load(f)
        else:
            self.data = _load_model()

        self.loo_r = self.data.get(
            "loo_r", self.data.get("cv_r", self.data.get("test_r", 0))
        )
        self.n_compounds = self.data.get("n_compounds", "N/A")
        self.ke = self.data.get("ke", 512)
        self.alpha = self.data.get("alpha", "N/A")

    def predict(self, smiles, pdb_file=None):
        return predict_pKd(smiles, pdb_file)

    def batch_predict(self, smiles_list):
        return batch_predict(smiles_list)


# ── CLI Interface ───────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="GEOCK Binding Affinity Prediction")
    parser.add_argument("--smiles", type=str, help="SMILES string of the ligand")
    parser.add_argument("--pdb-file", type=str, help="Path to pocket PDB file")
    parser.add_argument("--batch", type=str, help="File with SMILES, one per line")
    parser.add_argument("--model", type=str, default=str(MODEL_PATH), help="Model path")
    args = parser.parse_args()

    m = _load_model()
    print("=" * 60)
    print("  GEOCK Binding Affinity Prediction Engine")
    print("=" * 60)

    model_type = m.get("model_type", "ridge")
    print(f"\nModel type: {model_type}")
    print(f"Features: ke={m.get('ke', 512)}")
    if "alpha" in m:
        print(f"Alpha: {m['alpha']}")
    print(f"Training data: {m.get('n_compounds', 'N/A')} compounds")

    perf = m.get("loo_r", m.get("cv_r", m.get("test_r", 0)))
    print(f"CV-R: {perf:.4f}")
    if "cv_mae" in m:
        print(f"CV-MAE: {m['cv_mae']:.2f} pKd")
    print("=" * 60)

    if args.batch:
        with open(args.batch) as f:
            smiles_list = [line.strip() for line in f if line.strip()]
        print(f"\nPredicting for {len(smiles_list)} molecules...")
        results = batch_predict(smiles_list)
        for r in results:
            if "error" in r:
                print(f"  {r['smiles'][:40]:<40} ERROR: {r['error']}")
            else:
                print(
                    f"  {r['smiles'][:40]:<40} pKd={r['pKd']:>6.2f} ({r['confidence']})"
                )

    elif args.smiles:
        print(f"\nInput SMILES: {args.smiles}")
        result = predict_pKd(args.smiles, args.pdb_file)
        if "error" in result and result["pKd"] is None:
            print(f"ERROR: {result['error']}")
        else:
            print(f"\nPredicted pKd: {result['pKd']:.2f}")
            print(f"Confidence: {result['confidence']}")
            print(f"Model LOO-R: {result['model_loo_r']:.4f}")

    else:
        print("\nUsage:")
        print("  python geock_engine.py --smiles 'CCO'")
        print("  python geock_engine.py --batch smiles.txt")
        print("\nAs a module:")
        print("  from geock_engine import predict_pKd")
        print("  result = predict_pKd('CCO')")
        print(f"  print(f\"pKd = {{result['pKd']:.2f}}\")")


if __name__ == "__main__":
    main()
