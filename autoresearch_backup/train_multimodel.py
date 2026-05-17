#!/usr/bin/env python3
"""
GEOCK v2 - Multi-Model Comparison
Compare XGBoost, Random Forest, and ensemble
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

print("=" * 60)
print("GEOCK v2 - MULTI-MODEL COMPARISON")
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
print(f"  {len(data)} samples, {X.shape[1]} features")

# Models to try
models = {
    "XGBoost": xgb.XGBRegressor(
        max_depth=12,
        learning_rate=0.02,
        n_estimators=500,
        reg_alpha=0.6,
        reg_lambda=3.0,
        subsample=0.8,
        colsample_bytree=0.8,
        verbosity=0,
        n_jobs=-1,
    ),
    "RandomForest": RandomForestRegressor(
        n_estimators=500, max_depth=15, min_samples_leaf=3, n_jobs=-1, random_state=42
    ),
    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    ),
}

# Test different feature counts
k_values = [300, 400, 500, 600]

print("\n[2] Testing models with different feature counts...")
results = []

kf = KFold(n_splits=3, shuffle=True, random_state=42)

for k in k_values:
    for model_name, model_template in models.items():
        fold_scores = []

        for fold, (tr_idx, vl_idx) in enumerate(kf.split(X)):
            # Feature selection inside fold
            selector = SelectKBest(f_regression, k=k)
            X_tr = selector.fit_transform(X[tr_idx], y[tr_idx])
            X_vl = selector.transform(X[vl_idx])

            # Scale inside fold
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_vl_s = scaler.transform(X_vl)

            # Clone and train
            import copy

            model = copy.deepcopy(model_template)
            if hasattr(model, "random_state"):
                model.random_state = 42 + fold
            model.fit(X_tr_s, y[tr_idx])

            pred = model.predict(X_vl_s)
            r, _ = pearsonr(y[vl_idx], pred)
            fold_scores.append(r**2)

        cv_r2 = np.mean(fold_scores)
        results.append(
            {"model": model_name, "k": k, "cv_r2": cv_r2, "std": np.std(fold_scores)}
        )
        print(f"  {model_name} k={k}: R² = {cv_r2:.4f} ± {np.std(fold_scores):.4f}")

# Find best
best = max(results, key=lambda x: x["cv_r2"])
print(f"\n  BEST: {best['model']} with k={best['k']}, R² = {best['cv_r2']:.4f}")

# Full 5-fold CV with best
print(f"\n[3] Full 5-fold CV with best config...")
best_k = best["k"]
best_model_name = best["model"]

# Re-create best model
if best_model_name == "XGBoost":
    best_model = xgb.XGBRegressor(
        max_depth=12,
        learning_rate=0.02,
        n_estimators=600,
        reg_alpha=0.6,
        reg_lambda=3.0,
        subsample=0.8,
        colsample_bytree=0.8,
        verbosity=0,
        n_jobs=-1,
    )
elif best_model_name == "RandomForest":
    best_model = RandomForestRegressor(
        n_estimators=600, max_depth=15, min_samples_leaf=3, n_jobs=-1, random_state=42
    )
else:
    best_model = GradientBoostingRegressor(
        n_estimators=400,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )

kf5 = KFold(n_splits=5, shuffle=True, random_state=42)
fold_scores = []
models_list = []

for fold, (tr_idx, vl_idx) in enumerate(kf5.split(X)):
    selector = SelectKBest(f_regression, k=best_k)
    X_tr = selector.fit_transform(X[tr_idx], y[tr_idx])
    X_vl = selector.transform(X[vl_idx])

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_vl_s = scaler.transform(X_vl)

    import copy

    model = copy.deepcopy(best_model)
    if hasattr(model, "random_state"):
        model.random_state = 42 + fold
    model.fit(X_tr_s, y[tr_idx])

    pred = model.predict(X_vl_s)
    r, _ = pearsonr(y[vl_idx], pred)
    r2 = r**2
    fold_scores.append(r2)
    models_list.append((scaler, selector, model))
    print(f"    Fold {fold + 1}: R² = {r2:.4f}")

cv_r2 = np.mean(fold_scores)
cv_r = np.sqrt(cv_r2)
print(f"\n  CV R² = {cv_r2:.4f} ± {np.std(fold_scores):.4f}")
print(f"  CV R = {cv_r:.4f}")

# Save best single model
print("\n[4] Training final model...")
selector_final = SelectKBest(f_regression, k=best_k)
X_sel = selector_final.fit_transform(X, y)

scaler_final = StandardScaler()
X_scaled = scaler_final.fit_transform(X_sel)

final_model = copy.deepcopy(best_model)
final_model.random_state = 42
final_model.fit(X_scaled, y)

model_data = {
    "model": final_model,
    "scaler": scaler_final,
    "selector": selector_final,
    "model_type": best_model_name,
    "cv_r2": cv_r2,
    "cv_r": cv_r,
    "cv_std": np.std(fold_scores),
    "fold_scores": fold_scores,
    "k_features": best_k,
    "n_samples": len(y),
    "date": pd.Timestamp.now().isoformat(),
}

output_path = work_dir / "geock_v2_best_model.pkl"
with open(output_path, "wb") as f:
    pickle.dump(model_data, f)

print(f"\n{'=' * 60}")
print(f"RESULT: {best_model_name} with k={best_k}")
print(f"  CV R² = {cv_r2:.4f} (R = {cv_r:.4f})")
print(f"  Previous best: R² = 0.5956")
print(f"  Improvement: {cv_r2 - 0.5956:.4f}")
print(f"  Saved: {output_path}")
print(f"{'=' * 60}")
