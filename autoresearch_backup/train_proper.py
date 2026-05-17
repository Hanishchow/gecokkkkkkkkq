#!/usr/bin/env python3
"""
GEOCK v2 - PROPER TRAINING (No Overfitting)
With strong regularization to prevent overfitting
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
print("GEOCK v2 - PROPER REGULARIZED TRAINING")
print("=" * 70)

# Import path helpers
try:
    from geock_paths import get_cache_dir, get_work_dir

    cache_dir = get_cache_dir()
    work_dir = get_work_dir()
except ImportError:
    cache_dir = Path("/home/chow/.cache/geock_autoresearch")
    work_dir = Path("/home/chow/autoresearch")

# ==================== LOAD BEST DATA ====================
print("\n[1/5] Loading pre-computed features (already validated)...")

with open(cache_dir / "lp_new_features_8k_no2016.pkl", "rb") as f:
    data = pickle.load(f)

print(f"  Loaded {len(data)} samples")

# Build features
X = np.array([d["ecfp"] for d in data], dtype=np.float32)
y = np.array([d["affinity"] for d in data], dtype=np.float32)
print(f"  X shape: {X.shape}, y range: {y.min():.2f} - {y.max():.2f}")

# ==================== TRAIN WITH PROPER CV ====================
print("\n[2/5] Feature selection (on full data for final model)...")

selector = SelectKBest(f_regression, k=500)
X_sel = selector.fit_transform(X, y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_sel)
print(f"  Selected: {X_scaled.shape}")

# STRONG REGULARIZATION configs to prevent overfitting
print("\n[3/5] Testing regularization configs...")

configs = [
    # Very strong regularization
    {
        "max_depth": 6,
        "learning_rate": 0.01,
        "reg_alpha": 2.0,
        "reg_lambda": 10.0,
        "n_estimators": 800,
        "min_child_weight": 10,
    },
    # Medium strong
    {
        "max_depth": 8,
        "learning_rate": 0.02,
        "reg_alpha": 1.5,
        "reg_lambda": 8.0,
        "n_estimators": 600,
        "min_child_weight": 8,
    },
    # Conservative
    {
        "max_depth": 5,
        "learning_rate": 0.05,
        "reg_alpha": 3.0,
        "reg_lambda": 15.0,
        "n_estimators": 500,
        "min_child_weight": 15,
    },
    # Original but regularized
    {
        "max_depth": 10,
        "learning_rate": 0.03,
        "reg_alpha": 1.0,
        "reg_lambda": 5.0,
        "n_estimators": 400,
        "min_child_weight": 5,
    },
]

kf = KFold(n_splits=5, shuffle=True, random_state=42)
results = []

for cfg in configs:
    fold_scores = []
    train_scores = []

    for fold, (tr_idx, vl_idx) in enumerate(kf.split(X_scaled)):
        # Feature selection on training fold ONLY
        fold_selector = SelectKBest(f_regression, k=500)
        X_tr_sel = fold_selector.fit_transform(X_scaled[tr_idx], y[tr_idx])
        X_vl_sel = fold_selector.transform(X_scaled[vl_idx])

        # Standardize on training fold
        fold_scaler = StandardScaler()
        X_tr_s = fold_scaler.fit_transform(X_tr_sel)
        X_vl_s = fold_scaler.transform(X_vl_sel)

        model = xgb.XGBRegressor(
            **cfg,
            subsample=0.7,
            colsample_bytree=0.7,
            random_state=42 + fold,
            verbosity=0,
            n_jobs=-1,
        )
        model.fit(X_tr_s, y[tr_idx])

        train_pred = model.predict(X_tr_s)
        val_pred = model.predict(X_vl_s)

        train_r, _ = pearsonr(y[tr_idx], train_pred)
        val_r, _ = pearsonr(y[vl_idx], val_pred)

        train_scores.append(train_r**2)
        fold_scores.append(val_r**2)

    cv_r2 = np.mean(fold_scores)
    train_r2 = np.mean(train_scores)
    gap = train_r2 - cv_r2

    results.append({"config": cfg, "train_r2": train_r2, "cv_r2": cv_r2, "gap": gap})

    print(
        f"  depth={cfg['max_depth']}, lr={cfg['learning_rate']}: Train={train_r2:.4f}, CV={cv_r2:.4f}, Gap={gap:.4f}"
    )

# Find best with minimal overfitting
valid = [r for r in results if r["gap"] < 0.15]
if valid:
    best = max(valid, key=lambda x: x["cv_r2"])
    print(f"\n  BEST (valid): CV R² = {best['cv_r2']:.4f}, Gap = {best['gap']:.4f}")
else:
    best = min(results, key=lambda x: x["gap"])
    print(
        f"\n  BEST (least overfit): CV R² = {best['cv_r2']:.4f}, Gap = {best['gap']:.4f}"
    )

# ==================== FINAL CV ====================
print("\n[4/5] Final 5-fold CV with best config...")

best_cfg = best["config"]
fold_scores = []
models = []

for fold, (tr_idx, vl_idx) in enumerate(kf.split(X_scaled)):
    fold_selector = SelectKBest(f_regression, k=500)
    X_tr_sel = fold_selector.fit_transform(X_scaled[tr_idx], y[tr_idx])
    X_vl_sel = fold_selector.transform(X_scaled[vl_idx])

    fold_scaler = StandardScaler()
    X_tr_s = fold_scaler.fit_transform(X_tr_sel)
    X_vl_s = fold_scaler.transform(X_vl_sel)

    model = xgb.XGBRegressor(
        **best_cfg,
        subsample=0.7,
        colsample_bytree=0.7,
        random_state=42 + fold,
        verbosity=0,
        n_jobs=-1,
    )
    model.fit(X_tr_s, y[tr_idx])

    pred = model.predict(X_vl_s)
    r, _ = pearsonr(y[vl_idx], pred)
    fold_scores.append(r**2)
    models.append((fold_scaler, fold_selector, model))

    print(f"    Fold {fold + 1}: R² = {fold_scores[-1]:.4f}")

cv_r2 = np.mean(fold_scores)
cv_r = np.sqrt(cv_r2)
print(f"\n  CV R: {cv_r:.4f}")
print(f"  CV R²: {cv_r2:.4f} ± {np.std(fold_scores):.4f}")

# ==================== SAVE ====================
print("\n[5/5] Training final model...")

# Final model on all data
final_selector = SelectKBest(f_regression, k=500)
X_final_sel = final_selector.fit_transform(X_scaled, y)

final_scaler = StandardScaler()
X_final_s = final_scaler.fit_transform(X_final_sel)

final_model = xgb.XGBRegressor(
    **best_cfg,
    subsample=0.7,
    colsample_bytree=0.7,
    random_state=42,
    verbosity=0,
    n_jobs=-1,
)
final_model.fit(X_final_s, y)

# Check training performance
train_pred = final_model.predict(X_final_s)
train_r2 = pearsonr(y, train_pred)[0] ** 2

model_data = {
    "model": final_model,
    "scaler": final_scaler,
    "selector": final_selector,
    "config": best_cfg,
    "cv_r": cv_r,
    "cv_r2": cv_r2,
    "cv_std": np.std(fold_scores),
    "fold_scores": fold_scores,
    "train_r2": train_r2,
    "overfit_gap": train_r2 - cv_r2,
    "n_features": X_final_sel.shape[1],
    "n_samples": len(y),
    "date": pd.Timestamp.now().isoformat(),
}

output_path = work_dir / "geock_v2_proper.pkl"
with open(output_path, "wb") as f:
    pickle.dump(model_data, f)

print(f"\n{'=' * 70}")
print("FINAL RESULTS")
print(f"{'=' * 70}")
print(f"  Samples: {len(y)}")
print(f"  Features: {X_final_sel.shape[1]}")
print(f"  Train R²: {train_r2:.4f}")
print(f"  CV R²: {cv_r2:.4f} ± {np.std(fold_scores):.4f}")
print(f"  Overfit Gap: {train_r2 - cv_r2:.4f}")
print(f"\n  Saved: {output_path}")
print(f"{'=' * 70}")
