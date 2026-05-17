#!/usr/bin/env python3
"""
GEOCK v2 - Ensemble Training (XGBoost + Random Forest)
Try to improve further with model ensemble
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
import warnings

warnings.filterwarnings("ignore")

print("=" * 70)
print("GEOCK v2 - ENSEMBLE TRAINING")
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

# Different model types
print("\n[4/5] Training ensemble of models...")

model_types = {
    "xgb_deep": lambda: xgb.XGBRegressor(
        max_depth=14,
        learning_rate=0.025,
        reg_alpha=0.5,
        reg_lambda=2.5,
        n_estimators=500,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        verbosity=0,
        n_jobs=-1,
    ),
    "xgb_wide": lambda: xgb.XGBRegressor(
        max_depth=8,
        learning_rate=0.05,
        reg_alpha=0.3,
        reg_lambda=1.5,
        n_estimators=400,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        verbosity=0,
        n_jobs=-1,
    ),
    "rf": lambda: RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
    ),
    "gbm": lambda: GradientBoostingRegressor(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_split=5,
        random_state=42,
    ),
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)
all_preds = {}
all_scores = {}

for model_name, model_fn in model_types.items():
    print(f"  Training {model_name}...")
    fold_scores = []
    fold_preds = np.zeros(len(y))

    for fold, (tr_idx, vl_idx) in enumerate(kf.split(X_scaled)):
        model = model_fn()
        model.fit(X_scaled[tr_idx], y[tr_idx])
        pred = model.predict(X_scaled[vl_idx])
        fold_preds[vl_idx] = pred
        r, _ = pearsonr(y[vl_idx], pred)
        fold_scores.append(r**2)

    cv_r2 = np.mean(fold_scores)
    all_preds[model_name] = fold_preds
    all_scores[model_name] = fold_scores
    print(f"    {model_name}: CV R² = {cv_r2:.4f}")

# Simple average ensemble
print("\n  Creating simple average ensemble...")
ens_pred = np.mean([all_preds[k] for k in all_preds], axis=0)
ens_scores = []
for fold, (tr_idx, vl_idx) in enumerate(kf.split(X_scaled)):
    r, _ = pearsonr(y[vl_idx], ens_pred[vl_idx])
    ens_scores.append(r**2)

ens_r2 = np.mean(ens_scores)
print(f"    Ensemble (avg): CV R² = {ens_r2:.4f}")

# Weighted ensemble (based on individual scores)
weights = {k: np.mean(v) for k, v in all_scores.items()}
total_w = sum(weights.values())
weights = {k: v / total_w for k, v in weights.items()}
print(f"    Weights: {weights}")

weighted_pred = sum(all_preds[k] * w for k, w in weights.items())
weighted_scores = []
for fold, (tr_idx, vl_idx) in enumerate(kf.split(X_scaled)):
    r, _ = pearsonr(y[vl_idx], weighted_pred[vl_idx])
    weighted_scores.append(r**2)

weighted_r2 = np.mean(weighted_scores)
print(f"    Ensemble (weighted): CV R² = {weighted_r2:.4f}")

# Find best
best_single = max(all_scores.items(), key=lambda x: np.mean(x[1]))
best_single_r2 = np.mean(best_single[1])
print(f"\n  Best single: {best_single[0]} with R² = {best_single_r2:.4f}")

# Determine final
final_options = [
    ("xgb_deep", best_single_r2),
    ("ensemble_avg", ens_r2),
    ("ensemble_weighted", weighted_r2),
]
final_choice = max(final_options, key=lambda x: x[1])
print(f"  FINAL CHOICE: {final_choice[0]} with R² = {final_choice[1]:.4f}")

# Save final model
print("\n[5/5] Training final models...")

final_models = {}
for model_name, model_fn in model_types.items():
    final_models[model_name] = model_fn()
    final_models[model_name].fit(X_scaled, y)

model_data = {
    "models": final_models,
    "scaler": scaler,
    "selector": selector,
    "cv_r2": final_choice[1],
    "cv_type": final_choice[0],
    "weights": weights if "weighted" in final_choice[0] else None,
    "all_scores": {k: np.mean(v) for k, v in all_scores.items()},
    "n_features": X_sel.shape[1],
    "n_samples": len(y),
    "date": pd.Timestamp.now().isoformat(),
}

output_path = work_dir / "geock_v2_ensemble.pkl"
with open(output_path, "wb") as f:
    pickle.dump(model_data, f)

print(f"\n{'=' * 70}")
print(f"FINAL RESULT: CV R² = {final_choice[1]:.4f}")
print(f"Type: {final_choice[0]}")
print(f"Saved: {output_path}")
print(f"{'=' * 70}")
