#!/usr/bin/env python3
"""
GEOCK v2 - Quick Test with 1024-bit FP
"""

import pickle
import numpy as np
import pandas as pd
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

print("=" * 60)
print("GEOCK v2 - Quick 1024-bit FP Test")
print("=" * 60)

# Import path helpers
try:
    from geock_paths import get_cache_dir, get_work_dir

    cache_dir = get_cache_dir()
    work_dir = get_work_dir()
except ImportError:
    cache_dir = Path("/home/chow/.cache/geock_autoresearch")
    work_dir = Path("/home/chow/autoresearch")

# Load data
print("\n[1] Loading...")
with open(cache_dir / "lp_new_features_8k_no2016.pkl", "rb") as f:
    data = pickle.load(f)
print(f"  {len(data)} samples")

# Generate 1024-bit fingerprints
print("\n[2] Generating 1024-bit ECFP4...")
ecfp_list = []
for d in data:
    smiles = d.get("smiles", "")
    if smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
            ecfp_list.append(np.array(fp))
        else:
            ecfp_list.append(np.zeros(1024))
    else:
        ecfp_list.append(np.zeros(1024))

X = np.array(ecfp_list, dtype=np.float32)
y = np.array([d["affinity"] for d in data], dtype=np.float32)
print(f"  Shape: {X.shape}")

# Quick 3-fold CV
print("\n[3] 3-fold CV...")
cfg = {
    "max_depth": 10,
    "learning_rate": 0.03,
    "reg_alpha": 0.5,
    "reg_lambda": 2.5,
    "n_estimators": 300,
}

kf = KFold(n_splits=3, shuffle=True, random_state=42)
fold_scores = []

for fold, (tr_idx, vl_idx) in enumerate(kf.split(X)):
    fold_selector = SelectKBest(f_regression, k=500)
    X_tr = fold_selector.fit_transform(X[tr_idx], y[tr_idx])
    X_vl = fold_selector.transform(X[vl_idx])

    fold_scaler = StandardScaler()
    X_tr_s = fold_scaler.fit_transform(X_tr)
    X_vl_s = fold_scaler.transform(X_vl)

    model = xgb.XGBRegressor(
        **cfg, subsample=0.8, colsample_bytree=0.8, verbosity=0, n_jobs=-1
    )
    model.fit(X_tr_s, y[tr_idx])

    pred = model.predict(X_vl_s)
    r, _ = pearsonr(y[vl_idx], pred)
    r2 = r**2
    fold_scores.append(r2)
    print(f"    Fold {fold + 1}: R² = {r2:.4f}")

cv_r2 = np.mean(fold_scores)
print(f"\n  CV R² = {cv_r2:.4f} ± {np.std(fold_scores):.4f}")
print(f"  CV R = {np.sqrt(cv_r2):.4f}")

# Save result
output_path = work_dir / "geock_v2_1024_test.pkl"
with open(output_path, "wb") as f:
    pickle.dump({"cv_r2": cv_r2, "fold_scores": fold_scores, "fp_size": 1024}, f)

print(f"\n{'=' * 60}")
print(f"1024-bit result: R² = {cv_r2:.4f}")
print(f"vs 512-bit (prev): R² = 0.5956")
print(f"Improvement: {cv_r2 - 0.5956:.4f}")
print(f"{'=' * 60}")
