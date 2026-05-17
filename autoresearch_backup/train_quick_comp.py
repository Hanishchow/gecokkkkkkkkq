#!/usr/bin/env python3
"""
GEOCK v2 - Quick Model Comparison (3 models only)
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import warnings

warnings.filterwarnings("ignore")

print("=" * 60)
print("GEOCK v2 - QUICK MODEL COMPARISON")
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

X = np.array([d["ecfp"] for d in data], dtype=np.float32)
y = np.array([d["affinity"] for d in data], dtype=np.float32)
print(f"  {len(data)} samples")

# Quick test: XGBoost vs RF with fixed k=500
print("\n[2] Quick comparison (k=500, 3-fold)...")
kf = KFold(n_splits=3, shuffle=True, random_state=42)

configs = [
    (
        "XGB",
        xgb.XGBRegressor(
            max_depth=12,
            learning_rate=0.02,
            n_estimators=400,
            reg_alpha=0.6,
            reg_lambda=3.0,
            subsample=0.8,
            colsample_bytree=0.8,
            verbosity=0,
            n_jobs=-1,
        ),
    ),
    (
        "RF",
        RandomForestRegressor(
            n_estimators=300,
            max_depth=15,
            min_samples_leaf=3,
            n_jobs=-1,
            random_state=42,
        ),
    ),
]

best_result = None

for name, model_template in configs:
    fold_scores = []
    for fold, (tr_idx, vl_idx) in enumerate(kf.split(X)):
        selector = SelectKBest(f_regression, k=500)
        X_tr = selector.fit_transform(X[tr_idx], y[tr_idx])
        X_vl = selector.transform(X[vl_idx])

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_vl_s = scaler.transform(X_vl)

        import copy

        model = copy.deepcopy(model_template)
        if hasattr(model, "random_state"):
            model.random_state = 42 + fold
        model.fit(X_tr_s, y[tr_idx])

        pred = model.predict(X_vl_s)
        r, _ = pearsonr(y[vl_idx], pred)
        fold_scores.append(r**2)

    cv_r2 = np.mean(fold_scores)
    print(f"  {name}: CV R² = {cv_r2:.4f}")

    if best_result is None or cv_r2 > best_result["cv_r2"]:
        best_result = {
            "name": name,
            "model": model_template,
            "cv_r2": cv_r2,
            "fold_scores": fold_scores,
        }

# Full 5-fold
print(f"\n[3] 5-fold CV with {best_result['name']}...")
kf5 = KFold(n_splits=5, shuffle=True, random_state=42)
fold_scores = []

for fold, (tr_idx, vl_idx) in enumerate(kf5.split(X)):
    selector = SelectKBest(f_regression, k=500)
    X_tr = selector.fit_transform(X[tr_idx], y[tr_idx])
    X_vl = selector.transform(X[vl_idx])

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_vl_s = scaler.transform(X_vl)

    import copy

    model = copy.deepcopy(best_result["model"])
    if hasattr(model, "random_state"):
        model.random_state = 42 + fold
    model.fit(X_tr_s, y[tr_idx])

    pred = model.predict(X_vl_s)
    r, _ = pearsonr(y[vl_idx], pred)
    fold_scores.append(r**2)
    print(f"    Fold {fold + 1}: R² = {fold_scores[-1]:.4f}")

cv_r2 = np.mean(fold_scores)
cv_r = np.sqrt(cv_r2)
print(f"\n  CV R² = {cv_r2:.4f}, CV R = {cv_r:.4f}")

# Save
selector_f = SelectKBest(f_regression, k=500)
X_sel = selector_f.fit_transform(X, y)
scaler_f = StandardScaler()
X_scaled = scaler_f.fit_transform(X_sel)

import copy

final = copy.deepcopy(best_result["model"])
final.random_state = 42
final.fit(X_scaled, y)

model_data = {
    "model": final,
    "scaler": scaler_f,
    "selector": selector_f,
    "model_type": best_result["name"],
    "cv_r2": cv_r2,
    "cv_r": cv_r,
    "fold_scores": fold_scores,
    "n_samples": len(y),
    "date": pd.Timestamp.now().isoformat(),
}

output_path = work_dir / "geock_v2_model_comp.pkl"
with open(output_path, "wb") as f:
    pickle.dump(model_data, f)

print(f"\n{'=' * 60}")
print(f"RESULT: {best_result['name']} CV R² = {cv_r2:.4f}")
print(f"Previous: 0.5956, Diff: {cv_r2 - 0.5956:.4f}")
print(f"{'=' * 60}")
