#!/usr/bin/env python3
"""
GEOCK v2 - Aggressive Training with Multiple Configs
Try to match/exceed original 0.84 R²
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
print("GEOCK v2 - AGGRESSIVE MULTI-CONFIG TRAINING")
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

# Try multiple XGBoost configs
print("\n[4/5] Testing multiple configurations...")

configs = [
    # Original config
    {
        "name": "original",
        "max_depth": 10,
        "learning_rate": 0.05,
        "reg_alpha": 0.5,
        "reg_lambda": 2.0,
        "n_estimators": 200,
    },
    # Deeper
    {
        "name": "deep",
        "max_depth": 12,
        "learning_rate": 0.03,
        "reg_alpha": 0.5,
        "reg_lambda": 2.0,
        "n_estimators": 300,
    },
    # Very deep
    {
        "name": "very_deep",
        "max_depth": 15,
        "learning_rate": 0.02,
        "reg_alpha": 1.0,
        "reg_lambda": 3.0,
        "n_estimators": 400,
    },
    # Wider
    {
        "name": "wide",
        "max_depth": 8,
        "learning_rate": 0.08,
        "reg_alpha": 0.3,
        "reg_lambda": 1.0,
        "n_estimators": 250,
    },
    # More regularized
    {
        "name": "reg",
        "max_depth": 10,
        "learning_rate": 0.05,
        "reg_alpha": 1.0,
        "reg_lambda": 5.0,
        "n_estimators": 300,
    },
    # Aggressive
    {
        "name": "agg",
        "max_depth": 14,
        "learning_rate": 0.025,
        "reg_alpha": 0.5,
        "reg_lambda": 2.5,
        "n_estimators": 500,
    },
]

results = []
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for cfg in configs:
    fold_scores = []
    for fold, (tr_idx, vl_idx) in enumerate(kf.split(X_scaled)):
        model = xgb.XGBRegressor(
            **cfg,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            random_state=42 + fold,
            verbosity=0,
            n_jobs=-1,
        )
        model.fit(X_scaled[tr_idx], y[tr_idx])
        pred = model.predict(X_scaled[vl_idx])
        r, _ = pearsonr(y[vl_idx], pred)
        fold_scores.append(r**2)

    cv_r2 = np.mean(fold_scores)
    results.append({"config": cfg, "cv_r2": cv_r2, "fold_scores": fold_scores})
    print(f"  {cfg['name']}: R² = {cv_r2:.4f}")

# Find best
best = max(results, key=lambda x: x["cv_r2"])
print(f"\n  BEST: {best['config']['name']} with R² = {best['cv_r2']:.4f}")

# Ensemble top 3 configs
print("\n[5/5] Creating ensemble...")
results.sort(key=lambda x: x["cv_r2"], reverse=True)
top3 = results[:3]

# Get predictions from each config
ens_preds = []
for cfg_data in top3:
    cfg = cfg_data["config"]
    preds = []
    for fold, (tr_idx, vl_idx) in enumerate(kf.split(X_scaled)):
        model = xgb.XGBRegressor(
            **cfg,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            random_state=42 + fold,
            verbosity=0,
            n_jobs=-1,
        )
        model.fit(X_scaled[tr_idx], y[tr_idx])
        preds.append(model.predict(X_scaled[vl_idx]))
    ens_preds.append(np.mean(preds, axis=0))

# Average ensemble
ens_pred = np.mean(ens_preds, axis=0)
r_ens, _ = pearsonr(y[kf.split(X_scaled).__iter__().__next__()[1]], ens_pred)
# Actually compute properly
cv_preds = []
cv_trues = []
for fold, (tr_idx, vl_idx) in enumerate(kf.split(X_scaled)):
    cv_trues.extend(y[vl_idx])
cv_trues = np.array(cv_trues)

# Recompute ensemble CV properly
ens_fold_scores = []
for fold, (tr_idx, vl_idx) in enumerate(kf.split(X_scaled)):
    fold_preds = []
    for cfg_data in top3:
        cfg = cfg_data["config"]
        model = xgb.XGBRegressor(
            **cfg,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            random_state=42 + fold,
            verbosity=0,
            n_jobs=-1,
        )
        model.fit(X_scaled[tr_idx], y[tr_idx])
        fold_preds.append(model.predict(X_scaled[vl_idx]))
    ens_fold_pred = np.mean(fold_preds, axis=0)
    r, _ = pearsonr(y[vl_idx], ens_fold_pred)
    ens_fold_scores.append(r**2)

ens_cv_r2 = np.mean(ens_fold_scores)
print(f"  Ensemble (top 3): R² = {ens_cv_r2:.4f}")

# Determine final best
if ens_cv_r2 > best["cv_r2"]:
    final_r2 = ens_cv_r2
    final_type = "ensemble"
else:
    final_r2 = best["cv_r2"]
    final_type = best["config"]["name"]

# Train final model(s)
print("\nTraining final model(s)...")
final_models = []
for cfg_data in top3[:3]:
    cfg = cfg_data["config"]
    m = xgb.XGBRegressor(
        **cfg,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        random_state=42,
        verbosity=0,
        n_jobs=-1,
    )
    m.fit(X_scaled, y)
    final_models.append(m)

# Save
model_data = {
    "models": final_models,
    "scaler": scaler,
    "selector": selector,
    "configs": [r["config"] for r in top3[:3]],
    "cv_r2": final_r2,
    "cv_type": final_type,
    "n_features": X_sel.shape[1],
    "n_samples": len(y),
    "date": pd.Timestamp.now().isoformat(),
}

output_path = work_dir / "geock_v2_aggressive.pkl"
with open(output_path, "wb") as f:
    pickle.dump(model_data, f)

print(f"\n{'=' * 70}")
print(f"FINAL RESULT: CV R² = {final_r2:.4f} ({final_type})")
print(f"Saved: {output_path}")
print(f"{'=' * 70}")
