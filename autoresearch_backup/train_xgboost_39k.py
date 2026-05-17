#!/usr/bin/env python3
"""XGBoost on full 39K merged dataset."""

import pickle
import numpy as np
import pandas as pd
import os
import json
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import warnings

warnings.filterwarnings("ignore")


# Cross-platform paths
def _get_cache_dir():
    linux = Path("/home/chow/.cache/geock_autoresearch")
    if linux.exists():
        return linux
    win = Path(os.path.expanduser("~/OneDrive/.cache/geock_autoresearch"))
    if win.exists():
        return win
    return Path("./cache")


def _get_work_dir():
    linux = Path("/home/chow/autoresearch")
    if linux.exists():
        return linux
    win = Path(os.path.expanduser("~/OneDrive/autoresearch"))
    if win.exists():
        return win
    return Path(".")


cache_dir = _get_cache_dir()
work_dir = _get_work_dir()

print("=" * 60)
print("GEOCK v2 - XGBOOST on 39K (Merged) Data")
print("=" * 60)
print(f"Cache dir: {cache_dir}")
print(f"Work dir: {work_dir}")

# Load merged data
with open(cache_dir / "merged_39k.pkl", "rb") as f:
    data = pickle.load(f)

X = np.array([d["ecfp"] for d in data], dtype=np.float32)
y = np.array([d["affinity"] for d in data], dtype=np.float32)
print(f"\nLoaded: {len(data)} samples, {X.shape[1]} features")

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5-Fold CV with feature selection INSIDE each fold
print("\n[1] 5-Fold CV with XGBoost (39K samples)...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_scores = []

for fold, (tr_idx, vl_idx) in enumerate(kf.split(X_scaled)):
    print(f"\n{'=' * 60}")
    print(f"FOLD {fold + 1}/5")
    print(f"{'=' * 60}")

    # Feature selection INSIDE fold
    selector = SelectKBest(f_regression, k=400)
    X_tr = selector.fit_transform(X_scaled[tr_idx], y[tr_idx])
    X_vl = selector.transform(X_scaled[vl_idx])

    # XGBoost params (optimized)
    params = {
        "objective": "reg:squarederror",
        "max_depth": 12,
        "n_estimators": 500,
        "learning_rate": 0.01,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "gamma": 0.1,
        "random_state": 42,
    }

    model = xgb.XGBRegressor(**params)
    model.fit(X_tr, y[tr_idx], eval_set=[(X_vl, y[vl_idx])], verbose=False)

    preds = model.predict(X_vl)
    r, _ = pearsonr(y[vl_idx], preds)
    r2 = r**2
    fold_scores.append(r2)
    print(f"Fold {fold + 1} R² = {r2:.4f}, R = {r:.4f}")

cv_r2 = np.mean(fold_scores)
cv_r = np.sqrt(cv_r2)
print(f"\n{'=' * 60}")
print(f"XGBOOST 39K RESULTS:")
print(f"{'=' * 60}")
print(f"  CV R² = {cv_r2:.4f} ± {np.std(fold_scores):.4f}")
print(f"  CV R  = {cv_r:.4f}")
print(f"  Fold scores: {[f'{s:.4f}' for s in fold_scores]}")
print(f"\n  Target (original): R² = 0.7118")
print(f"  Difference: {cv_r2 - 0.7118:+.4f}")
print(f"{'=' * 60}")

# Train final model on ALL data
print("\n[2] Training FINAL XGBoost on ALL 39K samples...")
selector = SelectKBest(f_regression, k=400)
X_final = selector.fit_transform(X_scaled, y)

final_model = xgb.XGBRegressor(
    objective="reg:squarederror",
    max_depth=12,
    n_estimators=500,
    learning_rate=0.01,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.1,
    random_state=42,
)
final_model.fit(X_final, y, verbose=False)

# Save
model_data = {
    "model": final_model,
    "scaler": scaler,
    "selector": selector,
    "cv_r2": cv_r2,
    "cv_r": cv_r,
    "fold_scores": fold_scores,
    "n_samples": len(y),
    "date": pd.Timestamp.now().isoformat(),
}

output_path = work_dir / "geock_v2_xgboost_39k.pkl"
with open(output_path, "wb") as f:
    pickle.dump(model_data, f)

print(f"Saved: {output_path}")
print(f"Final CV R² = {cv_r2:.4f}")
print(f"{'=' * 60}")
