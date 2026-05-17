#!/usr/bin/env python3
"""
GEOCK v2 - Quick training with best settings found
"""

import pickle
import numpy as np
import pandas as pd
import os
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from rdkit import Chem
from rdkit.Chem import AllChem
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
print("GEOCK v2 - Quick Training")
print("=" * 60)
print(f"Cache dir: {cache_dir}")

# Load data
print("\n[1/4] Loading data...")
lp_df = pd.read_csv(cache_dir / "LP_PDBBind.csv")
print(f"  LP-PDBBind: {len(lp_df)} samples")

# Compute fingerprints
print("\n[2/4] Computing fingerprints...")


def smiles_to_fp(smiles):
    if not isinstance(smiles, str):
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    return np.array(fp, dtype=np.float32)


data_list = []
for i, row in lp_df.iterrows():
    if i % 3000 == 0:
        print(f"  {i}/{len(lp_df)}")
    fp = smiles_to_fp(row["smiles"])
    if fp is not None:
        data_list.append({"affinity": row["value"], "fp": fp})

print(f"  Valid: {len(data_list)}")

# Build features
X = np.array([d["fp"] for d in data_list], dtype=np.float32)
y = np.array([d["affinity"] for d in data_list], dtype=np.float32)

# Select 1024 features (best from comparison)
print("\n[3/4] Feature selection & training...")
selector = SelectKBest(f_regression, k=1024)
X_sel = selector.fit_transform(X, y)
print(f"  Features: {X_sel.shape}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_sel)

# Train with best config found: max_depth=12, lr=0.03, n_estimators=300
config = {
    "max_depth": 12,
    "learning_rate": 0.03,
    "reg_alpha": 1.0,
    "reg_lambda": 5.0,
    "n_estimators": 300,
    "min_child_weight": 3,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_scores = []

for fold, (tr_idx, vl_idx) in enumerate(kf.split(X_scaled)):
    model = xgb.XGBRegressor(**config, random_state=42 + fold, verbosity=0, n_jobs=-1)
    model.fit(X_scaled[tr_idx], y[tr_idx])
    pred = model.predict(X_scaled[vl_idx])
    r, _ = pearsonr(y[vl_idx], pred)
    fold_scores.append(r**2)
    print(f"  Fold {fold + 1}: R² = {fold_scores[-1]:.4f}")

cv_r2 = np.mean(fold_scores)
print(f"\n  CV R²: {cv_r2:.4f} ± {np.std(fold_scores):.4f}")

# Train final
print("\n[4/4] Final model...")
final = xgb.XGBRegressor(**config, random_state=42, verbosity=0, n_jobs=-1)
final.fit(X_scaled, y)

# Save
model_data = {
    "model": final,
    "scaler": scaler,
    "selector": selector,
    "config": config,
    "cv_r2": cv_r2,
    "fold_scores": fold_scores,
    "n_features": X_sel.shape[1],
    "n_samples": len(y),
    "date": pd.Timestamp.now().isoformat(),
}

with open("WORK_DIR / geock_v2_quick.pkl", "wb") as f:
    pickle.dump(model_data, f)

print(f"\n✓ Saved: geock_v2_quick.pkl")
print(f"  CV R² = {cv_r2:.4f}")
