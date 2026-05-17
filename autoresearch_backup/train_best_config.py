#!/usr/bin/env python3
"""
GEOCK v2 - Aggressive Training with Best Config
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
print("GEOCK v2 - BEST CONFIG TRAINING")
print("=" * 70)

# Import path helpers
try:
    from geock_paths import get_cache_dir, get_work_dir

    cache_dir = get_cache_dir()
    work_dir = get_work_dir()
except ImportError:
    cache_dir = Path("/home/chow/.cache/geock_autoresearch")
    work_dir = Path("/home/chow/autoresearch")

# Load data
print("\n[1/5] Loading data...")
with open(cache_dir / "lp_new_features_8k_no2016.pkl", "rb") as f:
    data = pickle.load(f)
print(f"  {len(data)} samples")

# Build features
print("\n[2/5] Building features...")
X = np.array([d["ecfp"] for d in data], dtype=np.float32)
y = np.array([d["affinity"] for d in data], dtype=np.float32)
print(f"  Shape: {X.shape}")

# Feature selection
print("\n[3/5] Feature selection...")
selector = SelectKBest(f_regression, k=500)
X_sel = selector.fit_transform(X, y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_sel)
print(f"  Selected: {X_scaled.shape}")

# Best config from previous run
config = {
    "max_depth": 14,
    "learning_rate": 0.025,
    "reg_alpha": 0.5,
    "reg_lambda": 2.5,
    "n_estimators": 500,
    "min_child_weight": 3,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}

print(f"\n[4/5] Training with best config: {config}")

kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_scores = []
models = []

for fold, (tr_idx, vl_idx) in enumerate(kf.split(X_scaled)):
    model = xgb.XGBRegressor(**config, random_state=42 + fold, verbosity=0, n_jobs=-1)
    model.fit(X_scaled[tr_idx], y[tr_idx])

    pred = model.predict(X_scaled[vl_idx])
    r, _ = pearsonr(y[vl_idx], pred)
    r2 = r**2
    fold_scores.append(r2)
    models.append(model)
    print(f"    Fold {fold + 1}: R² = {r2:.4f}")

cv_r2 = np.mean(fold_scores)
cv_r = np.sqrt(cv_r2)
print(f"\n  CV R: {cv_r:.4f}")
print(f"  CV R²: {cv_r2:.4f} ± {np.std(fold_scores):.4f}")

# Final model
print("\n[5/5] Training final model...")
final_model = xgb.XGBRegressor(**config, random_state=42, verbosity=0, n_jobs=-1)
final_model.fit(X_scaled, y)

# Save
model_data = {
    "model": final_model,
    "models": models,
    "scaler": scaler,
    "selector": selector,
    "config": config,
    "cv_r": cv_r,
    "cv_r2": cv_r2,
    "cv_std": np.std(fold_scores),
    "fold_scores": fold_scores,
    "n_features": X_sel.shape[1],
    "n_samples": len(y),
    "date": pd.Timestamp.now().isoformat(),
}

output_path = work_dir / "geock_v2_best.pkl"
with open(output_path, "wb") as f:
    pickle.dump(model_data, f)

print(f"\n{'=' * 70}")
print(f"RESULT: CV R = {cv_r:.4f}, CV R² = {cv_r2:.4f}")
print(f"Samples: {len(y)}")
print(f"Saved: {output_path}")
print(f"{'=' * 70}")

# Compare with original
print(f"\n  COMPARISON:")
print(f"    Original: R² = 0.7118 (39,109 samples)")
print(f"    Current:  R² = {cv_r2:.4f} ({len(y)} samples)")
print(f"    Gap:      {0.7118 - cv_r2:.4f}")
