#!/usr/bin/env python3
"""
GEOCK v2 - XGBoost + Neural Network Ensemble
Simpler approach: combine XGBoost with a small NN
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import warnings

warnings.filterwarnings("ignore")

print("=" * 60)
print("GEOCK v2 - XGB + NN ENSEMBLE")
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
print("\n[1] Loading data...")
with open(cache_dir / "lp_new_features_8k_no2016.pkl", "rb") as f:
    data = pickle.load(f)

X = np.array([d["ecfp"] for d in data], dtype=np.float32)
y = np.array([d["affinity"] for d in data], dtype=np.float32)
print(f"  {len(data)} samples, {X.shape[1]} features")

# Models
print("\n[2] Testing ensemble...")
configs = [
    (
        "XGBoost",
        xgb.XGBRegressor(
            max_depth=16,
            learning_rate=0.02,
            n_estimators=600,
            reg_alpha=0.6,
            reg_lambda=3.0,
            subsample=0.8,
            colsample_bytree=0.8,
            verbosity=0,
            n_jobs=-1,
            random_state=42,
        ),
    ),
    (
        "NeuralNet",
        MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            max_iter=200,
            early_stopping=True,
            n_iter_no_change=20,
            random_state=42,
        ),
    ),
]

# Test single models first
kf = KFold(n_splits=3, shuffle=True, random_state=42)
single_results = []

for name, model in configs:
    fold_scores = []
    for fold, (tr_idx, vl_idx) in enumerate(kf.split(X)):
        selector = SelectKBest(f_regression, k=500)
        X_tr = selector.fit_transform(X[tr_idx], y[tr_idx])
        X_vl = selector.transform(X[vl_idx])

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_vl_s = scaler.transform(X_vl)

        try:
            model.fit(X_tr_s, y[tr_idx])
            pred = model.predict(X_vl_s)
            r, _ = pearsonr(y[vl_idx], pred)
            fold_scores.append(r**2)
        except Exception as e:
            print(f"    {name} error: {e}")
            fold_scores.append(0.0)

    cv_r2 = np.mean(fold_scores)
    single_results.append({"name": name, "cv_r2": cv_r2})
    print(f"  {name}: R² = {cv_r2:.4f}")

# Ensemble: XGB + NN (average predictions)
print("\n[3] Testing ensemble (XGB + NN)...")
ensemble_fold_scores = []

for fold, (tr_idx, vl_idx) in enumerate(kf.split(X)):
    selector = SelectKBest(f_regression, k=500)
    X_tr = selector.fit_transform(X[tr_idx], y[tr_idx])
    X_vl = selector.transform(X[vl_idx])

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_vl_s = scaler.transform(X_vl)

    # Train XGB
    xgb_model = xgb.XGBRegressor(
        max_depth=16,
        learning_rate=0.02,
        n_estimators=600,
        reg_alpha=0.6,
        reg_lambda=3.0,
        subsample=0.8,
        colsample_bytree=0.8,
        verbosity=0,
        n_jobs=-1,
        random_state=42 + fold,
    )
    xgb_model.fit(X_tr_s, y[tr_idx])
    xgb_pred = xgb_model.predict(X_vl_s)

    # Train NN
    nn_model = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        max_iter=200,
        early_stopping=True,
        n_iter_no_change=20,
        random_state=42 + fold,
    )
    nn_model.fit(X_tr_s, y[tr_idx])
    nn_pred = nn_model.predict(X_vl_s)

    # Average ensemble
    final_pred = (xgb_pred + nn_pred) / 2.0
    r, _ = pearsonr(y[vl_idx], final_pred)
    ensemble_fold_scores.append(r**2)
    print(f"  Fold {fold + 1}: R² = {ensemble_fold_scores[-1]:.4f}")

cv_r2 = np.mean(ensemble_fold_scores)
print(f"\n  Ensemble R² = {cv_r2:.4f}")
print(f"  vs XGBoost alone: 0.5956")
print(f"  Improvement: {cv_r2 - 0.5956:.4f}")

# Full 5-fold CV with ensemble
print("\n[4] Full 5-fold CV with ensemble...")
kf5 = KFold(n_splits=5, shuffle=True, random_state=42)
fold_scores = []

for fold, (tr_idx, vl_idx) in enumerate(kf5.split(X)):
    selector = SelectKBest(f_regression, k=500)
    X_tr = selector.fit_transform(X[tr_idx], y[tr_idx])
    X_vl = selector.transform(X[vl_idx])

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_vl_s = scaler.transform(X_vl)

    xgb_model = xgb.XGBRegressor(
        max_depth=16,
        learning_rate=0.02,
        n_estimators=600,
        reg_alpha=0.6,
        reg_lambda=3.0,
        subsample=0.8,
        colsample_bytree=0.8,
        verbosity=0,
        n_jobs=-1,
        random_state=42 + fold,
    )
    xgb_model.fit(X_tr_s, y[tr_idx])
    xgb_pred = xgb_model.predict(X_vl_s)

    nn_model = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        max_iter=200,
        early_stopping=True,
        n_iter_no_change=20,
        random_state=42 + fold,
    )
    nn_model.fit(X_tr_s, y[tr_idx])
    nn_pred = nn_model.predict(X_vl_s)

    final_pred = (xgb_pred + nn_pred) / 2.0
    r, _ = pearsonr(y[vl_idx], final_pred)
    fold_scores.append(r**2)
    print(f"    Fold {fold + 1}: R² = {fold_scores[-1]:.4f}")

cv_r2 = np.mean(fold_scores)
cv_r = np.sqrt(cv_r2)
print(f"\n  ENSEMBLE 5-fold: R² = {cv_r2:.4f}, R = {cv_r:.4f}")
print(f"  vs XGBoost alone: 0.5956")
print(f"  Improvement: {cv_r2 - 0.5956:.4f}")

# Save ensemble
print("\n[5] Training final ensemble...")
# Train on all data
selector_f = SelectKBest(f_regression, k=500)
X_sel = selector_f.fit_transform(X, y)
scaler_f = StandardScaler()
X_scaled = scaler_f.fit_transform(X_sel)

xgb_final = xgb.XGBRegressor(
    max_depth=16,
    learning_rate=0.02,
    n_estimators=600,
    reg_alpha=0.6,
    reg_lambda=3.0,
    subsample=0.8,
    colsample_bytree=0.8,
    verbosity=0,
    n_jobs=-1,
    random_state=42,
)
xgb_final.fit(X_scaled, y)

nn_final = MLPRegressor(
    hidden_layer_sizes=(128, 64, 32),
    activation="relu",
    max_iter=200,
    early_stopping=True,
    n_iter_no_change=20,
    random_state=42,
)
nn_final.fit(X_scaled, y)

model_data = {
    "xgb_model": xgb_final,
    "nn_model": nn_final,
    "scaler": scaler_f,
    "selector": selector_f,
    "model_type": "ensemble_xgb_nn",
    "cv_r2": cv_r2,
    "cv_r": cv_r,
    "fold_scores": fold_scores,
    "n_samples": len(y),
    "date": pd.Timestamp.now().isoformat(),
}

output_path = work_dir / "geock_v2_ensemble_nn.pkl"
with open(output_path, "wb") as f:
    pickle.dump(model_data, f)

print(f"\n{'=' * 60}")
print(f"FINAL RESULT: R² = {cv_r2:.4f}, R = {cv_r:.4f}")
print(f"Saved: {output_path}")
print(f"{'=' * 60}")
