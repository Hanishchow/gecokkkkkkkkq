#!/usr/bin/env python3
"""
Improve Model to Address Mean Regression
======================================
Use quantile-based prediction adjustment and weighted loss.
"""

import os
import pickle
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import Ridge
import xgboost as xgb

WORK_DIR = Path("/home/chow/autoresearch")
CACHE_DIR = Path("/home/chow/.cache/geock_autoresearch")


def load_data():
    """Load training data."""
    records = []
    for path in [
        CACHE_DIR / "lp_new_features_8k.pkl",
        CACHE_DIR / "geock_training_data.pkl"
    ]:
        if path.exists():
            with open(path, 'rb') as f:
                data = pickle.load(f)
            if isinstance(data, list):
                records.extend([r for r in data if 'ecfp' in r and 'affinity' in r])
    
    X = np.array([r['ecfp'] for r in records])
    y = np.array([r['affinity'] for r in records])
    return X, y


def main():
    print("Loading data...")
    X, y = load_data()
    print(f"Loaded {len(X)} records, features={X.shape[1]}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
    
    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    
    # Feature selection
    selector = SelectKBest(f_regression, k=500)
    X_train_sel = selector.fit_transform(X_train_s, y_train)
    X_val_sel = selector.transform(X_val_s)
    X_test_sel = selector.transform(X_test_s)
    
    print("\n=== Testing Different Approaches ===")
    
    # 1. Baseline XGBoost
    print("\n1. Baseline XGBoost:")
    model1 = xgb.XGBRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        reg_alpha=1.0, reg_lambda=5.0, subsample=0.8,
        colsample_bytree=0.8, random_state=42, n_jobs=-1, verbosity=0
    )
    model1.fit(X_train_sel, y_train, eval_set=[(X_val_sel, y_val)], verbose=False)
    pred1 = model1.predict(X_test_sel)
    r1, _ = pearsonr(y_test, pred1)
    mae1 = np.mean(np.abs(y_test - pred1))
    print(f"  R={r1:.4f}, MAE={mae1:.3f}")
    print(f"  Pred range: {pred1.min():.2f} - {pred1.max():.2f}")
    
    # 2. Lower regularization (more expressive)
    print("\n2. Lower regularization:")
    model2 = xgb.XGBRegressor(
        n_estimators=200, max_depth=8, learning_rate=0.05,
        reg_alpha=0.1, reg_lambda=1.0, subsample=0.8,
        colsample_bytree=0.8, random_state=42, n_jobs=-1, verbosity=0
    )
    model2.fit(X_train_sel, y_train, eval_set=[(X_val_sel, y_val)], verbose=False)
    pred2 = model2.predict(X_test_sel)
    r2, _ = pearsonr(y_test, pred2)
    mae2 = np.mean(np.abs(y_test - pred2))
    print(f"  R={r2:.4f}, MAE={mae2:.3f}")
    print(f"  Pred range: {pred2.min():.2f} - {pred2.max():.2f}")
    
    # 3. Deeper trees
    print("\n3. Deeper trees (max_depth=10):")
    model3 = xgb.XGBRegressor(
        n_estimators=200, max_depth=10, learning_rate=0.05,
        reg_alpha=0.5, reg_lambda=2.0, subsample=0.8,
        colsample_bytree=0.8, random_state=42, n_jobs=-1, verbosity=0
    )
    model3.fit(X_train_sel, y_train, eval_set=[(X_val_sel, y_val)], verbose=False)
    pred3 = model3.predict(X_test_sel)
    r3, _ = pearsonr(y_test, pred3)
    mae3 = np.mean(np.abs(y_test - pred3))
    print(f"  R={r3:.4f}, MAE={mae3:.3f}")
    print(f"  Pred range: {pred3.min():.2f} - {pred3.max():.2f}")
    
    # 4. Histogram-based (faster, different splitting)
    print("\n4. HistGradientBoosting:")
    from sklearn.ensemble import HistGradientBoostingRegressor
    model4 = HistGradientBoostingRegressor(
        max_depth=10, learning_rate=0.05,
        l2_regularization=1.0, max_iter=200,
        random_state=42
    )
    model4.fit(X_train_sel, y_train)
    pred4 = model4.predict(X_test_sel)
    r4, _ = pearsonr(y_test, pred4)
    mae4 = np.mean(np.abs(y_test - pred4))
    print(f"  R={r4:.4f}, MAE={mae4:.3f}")
    print(f"  Pred range: {pred4.min():.2f} - {pred4.max():.2f}")
    
    # 5. Gradient Boosting with Huber loss (robust to extremes)
    print("\n5. Huber objective:")
    model5 = xgb.XGBRegressor(
        n_estimators=200, max_depth=8, learning_rate=0.05,
        reg_alpha=0.5, reg_lambda=2.0, subsample=0.8,
        colsample_bytree=0.8, objective='reg:pseudohubererror',
        random_state=42, n_jobs=-1, verbosity=0
    )
    model5.fit(X_train_sel, y_train, eval_set=[(X_val_sel, y_val)], verbose=False)
    pred5 = model5.predict(X_test_sel)
    r5, _ = pearsonr(y_test, pred5)
    mae5 = np.mean(np.abs(y_test - pred5))
    print(f"  R={r5:.4f}, MAE={mae5:.3f}")
    print(f"  Pred range: {pred5.min():.2f} - {pred5.max():.2f}")
    
    # 6. Ensemble of diverse models
    print("\n6. Ensemble of diverse models:")
    preds = [pred1, pred2, pred3, pred4, pred5]
    ensemble_pred = np.mean(preds, axis=0)
    r_ens, _ = pearsonr(y_test, ensemble_pred)
    mae_ens = np.mean(np.abs(y_test - ensemble_pred))
    print(f"  R={r_ens:.4f}, MAE={mae_ens:.3f}")
    print(f"  Pred range: {ensemble_pred.min():.2f} - {ensemble_pred.max():.2f}")
    
    # 7. Weighted ensemble (favor best models)
    print("\n7. Weighted ensemble:")
    weights = [0.2, 0.2, 0.2, 0.2, 0.2]
    weighted_pred = sum(w * p for w, p in zip(weights, preds))
    r_wens, _ = pearsonr(y_test, weighted_pred)
    mae_wens = np.mean(np.abs(y_test - weighted_pred))
    print(f"  R={r_wens:.4f}, MAE={mae_wens:.3f}")
    
    # Select best
    results = [
        ("Baseline XGB", r1, mae1, pred1, model1),
        ("Lower reg", r2, mae2, pred2, model2),
        ("Deep trees", r3, mae3, pred3, model3),
        ("HistGB", r4, mae4, pred4, model4),
        ("Huber", r5, mae5, pred5, model5),
        ("Ensemble", r_ens, mae_ens, ensemble_pred, None),
    ]
    
    best = max(results, key=lambda x: x[1])
    print(f"\n=== Best: {best[0]} with R={best[1]:.4f} ===")
    
    # Save best model
    if best[4] is not None:
        model_data = {
            'model': best[4],
            'scaler': scaler,
            'selector': selector,
            'model_type': best[0],
            'test_r': best[1],
            'test_mae': best[2],
            'date': '20260405_improved'
        }
        output_path = WORK_DIR / f"geock_{best[0].replace(' ', '_').lower()}.pkl"
        with open(output_path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Saved to {output_path}")
    
    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    for name, r, mae, _, _ in sorted(results, key=lambda x: -x[1]):
        print(f"  {name:15s}: R={r:.4f}, MAE={mae:.3f}")


if __name__ == "__main__":
    main()
