#!/usr/bin/env python3
"""
GEOCK v2 - Large Fingerprints + Combined Features
Try 1024-bit fingerprints + physics features
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

print("=" * 70)
print("GEOCK v2 - LARGER FINGERPRINTS + PHYSICS")
print("=" * 70)

# Import path helpers
try:
    from geock_paths import get_cache_dir, get_work_dir

    cache_dir = get_cache_dir()
    work_dir = get_work_dir()
except ImportError:
    cache_dir = Path("/home/chow/.cache/geock_autoresearch")
    work_dir = Path("/home/chow/autoresearch")

# Load base data
print("\n[1/5] Loading data...")
with open(cache_dir / "lp_new_features_8k_no2016.pkl", "rb") as f:
    data = pickle.load(f)
print(f"  Base data: {len(data)} samples")

# Load physics features if available
physics_data = None
try:
    with open(cache_dir / "physics_features_8k.pkl", "rb") as f:
        physics_data = pickle.load(f)
    print(f"  Physics features: {len(physics_data)} samples")
except:
    print("  No physics features found")

# Generate larger ECFP fingerprints (1024 bits)
print("\n[2/5] Generating 1024-bit ECFP4 fingerprints...")
ecfp_list = []
for i, d in enumerate(data):
    smiles = d.get("smiles", "")
    if smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
            ecfp_list.append(np.array(fp))
        else:
            ecfp_list.append(np.zeros(1024, dtype=np.float32))
    else:
        ecfp_list.append(np.zeros(1024, dtype=np.float32))
    if (i + 1) % 5000 == 0:
        print(f"    Processed {i + 1}/{len(data)}")

X_ecfp = np.array(ecfp_list, dtype=np.float32)
print(f"  ECFP shape: {X_ecfp.shape}")

# Combine with physics features if available
if physics_data and len(physics_data) == len(data):
    print("\n[3/5] Combining with physics features...")
    X_physics = np.array(
        [p.get("features", np.zeros(20)) for p in physics_data], dtype=np.float32
    )
    X = np.hstack([X_ecfp, X_physics])
    print(f"  Combined shape: {X.shape}")
else:
    X = X_ecfp
    print(f"  Using ECFP only: {X.shape}")

y = np.array([d["affinity"] for d in data], dtype=np.float32)
print(f"  Target range: {y.min():.2f} - {y.max():.2f}")

# Best configs to try
print("\n[4/5] Testing configurations...")

configs = [
    # Config A: Original best with more features
    {
        "max_depth": 12,
        "learning_rate": 0.02,
        "reg_alpha": 0.6,
        "reg_lambda": 3.0,
        "n_estimators": 600,
    },
    # Config B: Deeper
    {
        "max_depth": 16,
        "learning_rate": 0.015,
        "reg_alpha": 0.8,
        "reg_lambda": 4.0,
        "n_estimators": 700,
    },
    # Config C: More trees, lower LR
    {
        "max_depth": 10,
        "learning_rate": 0.01,
        "reg_alpha": 1.0,
        "reg_lambda": 5.0,
        "n_estimators": 900,
    },
]

kf = KFold(n_splits=5, shuffle=True, random_state=42)
best_result = None

for cfg in configs:
    fold_scores = []

    for fold, (tr_idx, vl_idx) in enumerate(kf.split(X)):
        # Feature selection INSIDE fold
        fold_selector = SelectKBest(f_regression, k=min(700, X.shape[1]))
        X_tr_sel = fold_selector.fit_transform(X[tr_idx], y[tr_idx])
        X_vl_sel = fold_selector.transform(X[vl_idx])

        # Scale INSIDE fold
        fold_scaler = StandardScaler()
        X_tr_s = fold_scaler.fit_transform(X_tr_sel)
        X_vl_s = fold_scaler.transform(X_vl_sel)

        model = xgb.XGBRegressor(
            **cfg,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            random_state=42 + fold,
            verbosity=0,
            n_jobs=-1,
        )
        model.fit(X_tr_s, y[tr_idx])

        pred = model.predict(X_vl_s)
        r, _ = pearsonr(y[vl_idx], pred)
        fold_scores.append(r**2)

    cv_r2 = np.mean(fold_scores)
    print(
        f"  {cfg['max_depth']}d/{cfg['n_estimators']}e/lr{cfg['learning_rate']}: CV R² = {cv_r2:.4f}"
    )

    if best_result is None or cv_r2 > best_result["cv_r2"]:
        best_result = {"config": cfg, "cv_r2": cv_r2, "fold_scores": fold_scores}

# Final model
print("\n[5/5] Training final model with best config...")
best_cfg = best_result["config"]

# Feature selection on all data
selector = SelectKBest(f_regression, k=min(700, X.shape[1]))
X_sel = selector.fit_transform(X, y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_sel)

final_model = xgb.XGBRegressor(
    **best_cfg,
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
    "selector": selector,
    "config": best_cfg,
    "cv_r2": best_result["cv_r2"],
    "cv_r": np.sqrt(best_result["cv_r2"]),
    "cv_std": np.std(best_result["fold_scores"]),
    "fold_scores": best_result["fold_scores"],
    "n_features": X_sel.shape[1],
    "n_samples": len(y),
    "fp_type": "ECFP4_1024",
    "date": pd.Timestamp.now().isoformat(),
}

output_path = work_dir / "geock_v2_large_fp.pkl"
with open(output_path, "wb") as f:
    pickle.dump(model_data, f)

cv_r2 = best_result["cv_r2"]
print(f"\n{'=' * 70}")
print(f"RESULT: CV R² = {cv_r2:.4f} (R = {np.sqrt(cv_r2):.4f})")
print(f"Fingerprints: 1024-bit ECFP4")
print(f"Features: {X_sel.shape[1]} (after selection)")
print(f"Samples: {len(y)}")
print(f"Saved: {output_path}")
print(f"{'=' * 70}")
print(f"\n  vs Original: R² = 0.7118 (39,109 samples)")
print(f"  vs Previous: R² = 0.5956 (23,782 samples)")
print(f"  Improvement: {cv_r2 - 0.5956:.4f}")
