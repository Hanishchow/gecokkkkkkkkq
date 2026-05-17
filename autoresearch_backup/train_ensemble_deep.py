#!/usr/bin/env python3
"""
GEOCK v2 - XGBoost + Deep Forest Ensemble
Combines XGBoost with multi-layer decision trees for better representations
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
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
import warnings

warnings.filterwarnings("ignore")

print("=" * 60)
print("GEOCK v2 - XGB + DEEP FOREST ENSEMBLE")
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


# Deep Forest (layered trees)
class DeepForestLayer:
    def __init__(self, n_trees=100, max_depth=8, n_layers=3):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.n_layers = n_layers
        self.layers = []
        self.scalers = []

    def fit(self, X, y):
        self.layers = []
        self.scalers = []
        current_X = X

        for layer_idx in range(self.n_layers):
            # Train trees
            forest = RandomForestRegressor(
                n_estimators=self.n_trees,
                max_depth=self.max_depth,
                min_samples_leaf=5,
                n_jobs=-1,
                random_state=42 + layer_idx,
            )
            forest.fit(current_X, y)
            self.layers.append(forest)

            # Get predictions (out-of-bag or predictions)
            preds = (
                forest.oob_prediction_
                if hasattr(forest, "oob_prediction_")
                else forest.predict(current_X)
            )

            # Combine original features with predictions
            # Add residual as new feature
            residual = y - preds
            new_feature = residual.reshape(-1, 1)

            # Update for next layer
            current_X = np.hstack([current_X, new_feature])

            # Scale
            scaler = StandardScaler()
            current_X = scaler.fit_transform(current_X)
            self.scalers.append(scaler)

        return self

    def predict(self, X, n_layers=None):
        if n_layers is None:
            n_layers = self.n_layers

        current_X = X
        for i in range(n_layers):
            if i >= len(self.layers):
                break
            preds = self.layers[i].predict(current_X)
            new_feature = (
                (y_true - preds).reshape(-1, 1) if False else preds.reshape(-1, 1)
            )
            current_X = np.hstack([current_X, new_feature])
            if i < len(self.scalers):
                current_X = self.scalers[i].transform(current_X)

        # Use last layer's prediction
        return self.layers[-1].predict(X)


# Deep Forest wrapper for sklearn
class DeepForestRegressor:
    def __init__(self, n_trees=100, max_depth=8, n_layers=3):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.n_layers = n_layers
        self.model = None

    def fit(self, X, y):
        self.model = DeepForestLayer(self.n_trees, self.max_depth, self.n_layers)
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)


# XGBoost config
xgb_config = {
    "max_depth": 16,
    "learning_rate": 0.02,
    "n_estimators": 600,
    "reg_alpha": 0.6,
    "reg_lambda": 3.0,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "verbosity": 0,
    "n_jobs": -1,
}

# Test configurations
print("\n[2] Testing models...")
configs = [
    ("XGBoost", xgb.XGBRegressor(**xgb_config, random_state=42)),
    ("DeepForest(3L)", DeepForestRegressor(n_trees=100, max_depth=8, n_layers=3)),
    ("DeepForest(5L)", DeepForestRegressor(n_trees=100, max_depth=10, n_layers=5)),
]

kf = KFold(n_splits=3, shuffle=True, random_state=42)
results = []

for name, model in configs:
    fold_scores = []
    for fold, (tr_idx, vl_idx) in enumerate(kf.split(X)):
        # Feature selection inside fold
        selector = SelectKBest(f_regression, k=500)
        X_tr = selector.fit_transform(X[tr_idx], y[tr_idx])
        X_vl = selector.transform(X[vl_idx])

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_vl_s = scaler.transform(X_vl)

        model.fit(X_tr_s, y[tr_idx])
        pred = model.predict(X_vl_s)
        r, _ = pearsonr(y[vl_idx], pred)
        fold_scores.append(r**2)

    cv_r2 = np.mean(fold_scores)
    results.append({"name": name, "cv_r2": cv_r2, "std": np.std(fold_scores)})
    print(f"  {name}: R² = {cv_r2:.4f} ± {np.std(fold_scores):.4f}")

# Sort by performance
results.sort(key=lambda x: x["cv_r2"], reverse=True)
print(f"\n  BEST: {results[0]['name']} with R² = {results[0]['cv_r2']:.4f}")

# Ensemble (average top 2)
print("\n[3] Ensemble (XGB + Best DeepForest)...")
ensemble_configs = [
    (
        "XGB+DF3",
        [
            xgb.XGBRegressor(**xgb_config, random_state=42),
            DeepForestRegressor(n_trees=100, max_depth=8, n_layers=3),
        ],
    ),
]

for ens_name, models in ensemble_configs:
    fold_scores = []
    for fold, (tr_idx, vl_idx) in enumerate(kf.split(X)):
        selector = SelectKBest(f_regression, k=500)
        X_tr = selector.fit_transform(X[tr_idx], y[tr_idx])
        X_vl = selector.transform(X[vl_idx])

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_vl_s = scaler.transform(X_vl)

        preds_all = []
        for model in models:
            model.fit(X_tr_s, y[tr_idx])
            preds_all.append(model.predict(X_vl_s))

        # Average predictions
        final_pred = np.mean(preds_all, axis=0)
        r, _ = pearsonr(y[vl_idx], final_pred)
        fold_scores.append(r**2)

    cv_r2 = np.mean(fold_scores)
    print(f"  {ens_name}: R² = {cv_r2:.4f}")

# Full 5-fold CV with best single model
print("\n[4] Full 5-fold CV with best model...")
best_name = results[0]["name"]
if best_name == "XGBoost":
    best_model = xgb.XGBRegressor(**xgb_config, random_state=42)
else:
    best_model = DeepForestRegressor(n_trees=100, max_depth=8, n_layers=3)

kf5 = KFold(n_splits=5, shuffle=True, random_state=42)
fold_scores = []

for fold, (tr_idx, vl_idx) in enumerate(kf5.split(X)):
    selector = SelectKBest(f_regression, k=500)
    X_tr = selector.fit_transform(X[tr_idx], y[tr_idx])
    X_vl = selector.transform(X[vl_idx])

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_vl_s = scaler.transform(X_vl)

    best_model.fit(X_tr_s, y[tr_idx])
    pred = best_model.predict(X_vl_s)
    r, _ = pearsonr(y[vl_idx], pred)
    fold_scores.append(r**2)
    print(f"    Fold {fold + 1}: R² = {fold_scores[-1]:.4f}")

cv_r2 = np.mean(fold_scores)
cv_r = np.sqrt(cv_r2)
print(f"\n  {best_name} 5-fold: R² = {cv_r2:.4f}, R = {cv_r:.4f}")
print(f"  Previous XGB best: R² = 0.5956")
print(f"  Improvement: {cv_r2 - 0.5956:.4f}")

# Save best model
print("\n[5] Saving best model...")
selector_f = SelectKBest(f_regression, k=500)
X_sel = selector_f.fit_transform(X, y)
scaler_f = StandardScaler()
X_scaled = scaler_f.fit_transform(X_sel)

final_model = xgb.XGBRegressor(**xgb_config, random_state=42)
final_model.fit(X_scaled, y)

model_data = {
    "model": final_model,
    "scaler": scaler_f,
    "selector": selector_f,
    "model_type": best_name,
    "cv_r2": cv_r2,
    "cv_r": cv_r,
    "fold_scores": fold_scores,
    "n_samples": len(y),
    "date": pd.Timestamp.now().isoformat(),
}

output_path = work_dir / "geock_v2_ensemble.pkl"
with open(output_path, "wb") as f:
    pickle.dump(model_data, f)

print(f"\n{'=' * 60}")
print(f"FINAL RESULT: {best_name}")
print(f"  CV R² = {cv_r2:.4f}")
print(f"  CV R = {cv_r:.4f}")
print(f"{'=' * 60}")
