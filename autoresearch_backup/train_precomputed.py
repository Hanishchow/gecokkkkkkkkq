#!/usr/bin/env python3
"""
GEOCK v2 - Use Pre-computed Features from lp_new_features_8k
This data already has 23,782 samples with 512-bit ECFP + 24 physics features
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import warnings

warnings.filterwarnings("ignore")

print("=" * 70)
print("GEOCK v2 - USING PRE-COMPUTED FEATURES")
print("=" * 70)

# Import path helpers
try:
    from geock_paths import get_cache_dir, get_work_dir

    cache_dir = get_cache_dir()
    work_dir = get_work_dir()
except ImportError:
    cache_dir = Path("/home/chow/.cache/geock_autoresearch")
    work_dir = Path("/home/chow/autoresearch")

# ==================== LOAD PRE-COMPUTED DATA ====================
print("\n[1/5] Loading pre-computed features...")

with open(cache_dir / "lp_new_features_8k_no2016.pkl", "rb") as f:
    data = pickle.load(f)

print(f"  Loaded {len(data)} samples")

# ==================== BUILD FEATURES ====================
print("\n[2/5] Building feature matrix...")

X_ecfp = np.array([d["ecfp"] for d in data], dtype=np.float32)
y = np.array([d["affinity"] for d in data], dtype=np.float32)

print(f"  ECFP shape: {X_ecfp.shape}")
print(f"  Affinity: min={y.min():.2f}, max={y.max():.2f}, mean={y.mean():.2f}")

# Use only ECFP (physics features appear to be mostly zeros)
X = X_ecfp
print(f"  Using: {X.shape}")

# ==================== FEATURE SELECTION ====================
print("\n[3/5] Feature selection (SelectKBest k=500)...")

selector = SelectKBest(f_regression, k=500)
X_selected = selector.fit_transform(X, y)
print(f"  Selected: {X_selected.shape}")

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

# ==================== TRAIN ====================
print("\n[4/5] Training XGBoost...")

config = {
    "n_estimators": 200,
    "max_depth": 10,
    "learning_rate": 0.05,
    "reg_alpha": 0.5,
    "reg_lambda": 2.0,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_scores = []

for fold, (tr_idx, vl_idx) in enumerate(kf.split(X_scaled)):
    model = xgb.XGBRegressor(**config, random_state=42 + fold, verbosity=0, n_jobs=-1)
    model.fit(X_scaled[tr_idx], y[tr_idx])

    pred = model.predict(X_scaled[vl_idx])
    r, _ = pearsonr(y[vl_idx], pred)
    fold_scores.append(r**2)
    print(f"    Fold {fold + 1}: R² = {fold_scores[-1]:.4f}")

cv_r2 = np.mean(fold_scores)
print(f"\n  CV R²: {cv_r2:.4f} ± {np.std(fold_scores):.4f}")

# Final model
print("\n[5/5] Training final model...")
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
    "n_features": X_selected.shape[1],
    "n_samples": len(y),
    "date": pd.Timestamp.now().isoformat(),
}

output_path = work_dir / "geock_v2_precomputed.pkl"
with open(output_path, "wb") as f:
    pickle.dump(model_data, f)

print(f"\n{'=' * 70}")
print(f"RESULT: CV R² = {cv_r2:.4f}")
print(f"Saved: {output_path}")
print(f"{'=' * 70}")
