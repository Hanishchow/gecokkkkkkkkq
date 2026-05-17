#!/usr/bin/env python3
"""
Train GEOCK v2 - Use all 2048 bits + physics, tune more aggressively
"""

import pickle
import numpy as np
import pandas as pd
import os
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski
import xgboost as xgb
import warnings

warnings.filterwarnings("ignore")
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")


# ===== CROSS-PLATFORM PATH HELPER =====
def _get_cache_dir():
    """Get cache directory - works on Linux and Windows."""
    linux_cache = Path("/home/chow/.cache/geock_autoresearch")
    if linux_cache.exists():
        return linux_cache
    win_cache = Path(os.path.expanduser("~/OneDrive/.cache/geock_autoresearch"))
    if win_cache.exists():
        return win_cache
    return Path("./cache")


def _get_autoresearch_dir():
    """Get autoresearch directory - works on Linux and Windows."""
    linux = Path("/home/chow/autoresearch")
    if linux.exists():
        return linux
    win = Path(os.path.expanduser("~/OneDrive/autoresearch"))
    if win.exists():
        return win
    return Path(".")


cache_dir = _get_cache_dir()
work_dir = _get_autoresearch_dir()

print("=" * 60)
print("GEOCK v2 - Full Features + Tuning")
print("=" * 60)
print(f"Cache dir: {cache_dir}")

# Load LP-PDBBind
print("\n[1/4] Loading data...")
lp_df = pd.read_csv(cache_dir / "LP_PDBBind.csv")
print(f"  LP-PDBBind: {len(lp_df)} samples")

# Compute features
print("\n[2/4] Computing features...")


def compute_features(smiles):
    if not isinstance(smiles, str):
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Full Morgan fingerprint (2048 bits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    fp_arr = np.array(fp, dtype=np.float32)

    # Extended physics features
    try:
        phys = np.array(
            [
                Descriptors.MolWt(mol),
                Descriptors.MolLogP(mol),
                Descriptors.TPSA(mol),
                Descriptors.NumRotatableBonds(mol),
                Lipinski.NumHDonors(mol),
                Lipinski.NumHAcceptors(mol),
                Lipinski.NumAromaticRings(mol),
                Lipinski.NumHeteroatoms(mol),
                Lipinski.NumHeavyAtoms(mol),
                Descriptors.FractionCSP3(mol),
                Lipinski.RingCount(mol),
                Descriptors.BertzCT(mol),
                Descriptors.Chi0(mol),
                Descriptors.Chi1(mol),
                Descriptors.Kappa1(mol),
                Descriptors.Kappa2(mol),
                Descriptors.LabuteASA(mol),
                Descriptors.PEOE_VSA1(mol),
                Descriptors.PEOE_VSA2(mol),
                Descriptors.PEOE_VSA3(mol),
            ],
            dtype=np.float32,
        )
    except:
        phys = np.zeros(21, dtype=np.float32)

    return np.concatenate([fp_arr, phys])


# Process all data
data_list = []
for i, row in lp_df.iterrows():
    if i % 3000 == 0:
        print(f"  Processing: {i}/{len(lp_df)}")

    features = compute_features(row["smiles"])
    if features is not None:
        data_list.append({"affinity": row["value"], "features": features})

print(f"  Valid: {len(data_list)} samples")

# Build arrays
X = np.array([d["features"] for d in data_list], dtype=np.float32)
y = np.array([d["affinity"] for d in data_list], dtype=np.float32)

print(f"  Features shape: {X.shape}")

# Standardize physics part only (fp is already binary)
print("\n[3/4] Training...")

# Standardize all features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Try multiple XGBoost configurations
configs = [
    {
        "max_depth": 12,
        "learning_rate": 0.03,
        "reg_alpha": 1.0,
        "reg_lambda": 5.0,
        "n_estimators": 400,
        "min_child_weight": 3,
    },
    {
        "max_depth": 15,
        "learning_rate": 0.02,
        "reg_alpha": 0.5,
        "reg_lambda": 3.0,
        "n_estimators": 500,
        "min_child_weight": 2,
    },
    {
        "max_depth": 10,
        "learning_rate": 0.05,
        "reg_alpha": 0.3,
        "reg_lambda": 2.0,
        "n_estimators": 300,
        "min_child_weight": 5,
    },
    {
        "max_depth": 8,
        "learning_rate": 0.1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "n_estimators": 200,
        "min_child_weight": 3,
    },
]

best_r2 = 0
best_model = None
best_config = None

for cfg in configs:
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_scores = []

    for fold, (tr_idx, vl_idx) in enumerate(kf.split(X_scaled)):
        model = xgb.XGBRegressor(
            **cfg,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42 + fold,
            verbosity=0,
            n_jobs=-1,
        )
        model.fit(X_scaled[tr_idx], y[tr_idx])

        pred = model.predict(X_scaled[vl_idx])
        r, _ = pearsonr(y[vl_idx], pred)
        fold_scores.append(r**2)

    cv_r2 = np.mean(fold_scores)
    cv_std = np.std(fold_scores)
    print(
        f"  {cfg['max_depth']}/{cfg['learning_rate']}/{cfg['n_estimators']}: CV R² = {cv_r2:.4f} ± {cv_std:.4f}"
    )

    if cv_r2 > best_r2:
        best_r2 = cv_r2
        best_config = cfg

print(f"\nBest config: {best_config}")
print(f"Best CV R²: {best_r2:.4f}")

# Train final model
print("\n[4/4] Training final model...")
final_model = xgb.XGBRegressor(
    **best_config,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    random_state=42,
    verbosity=0,
    n_jobs=-1,
)
final_model.fit(X_scaled, y)

# Save
model_data = {
    "model": final_model,
    "scaler": scaler,
    "config": best_config,
    "cv_r2": best_r2,
    "n_features": X.shape[1],
    "n_samples": len(y),
    "date": pd.Timestamp.now().isoformat(),
}

with open("WORK_DIR / geock_v2_full.pkl", "wb") as f:
    pickle.dump(model_data, f)

print(f"\nSaved: geock_v2_full.pkl")
print(f"Samples: {len(y)}, Features: {X.shape[1]}")
print(f"CV R²: {best_r2:.4f}")
