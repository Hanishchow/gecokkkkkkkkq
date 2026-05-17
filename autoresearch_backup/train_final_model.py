#!/usr/bin/env python3
"""
Train Final Improved Model
=========================
Train and save the deep trees model with full CV validation.
"""

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


def main():
    # Load data
    print("Loading data...")
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
    print(f"Loaded {len(X)} records, features={X.shape[1]}")
    print(f"Affinity range: {y.min():.2f} - {y.max():.2f}")

    # 5-fold CV
    print("\nRunning 5-fold CV...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    models = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, X_vl = X[train_idx], X[val_idx]
        y_tr, y_vl = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_vl_s = scaler.transform(X_vl)

        selector = SelectKBest(f_regression, k=500)
        X_tr_sel = selector.fit_transform(X_tr_s, y_tr)
        X_vl_sel = selector.transform(X_vl_s)

        model = xgb.XGBRegressor(
            n_estimators=200, max_depth=10, learning_rate=0.05,
            reg_alpha=0.5, reg_lambda=2.0, subsample=0.8,
            colsample_bytree=0.8, random_state=42, n_jobs=-1, verbosity=0
        )
        model.fit(X_tr_sel, y_tr)
        
        pred = model.predict(X_vl_sel)
        r, _ = pearsonr(y_vl, pred)
        rho, _ = spearmanr(y_vl, pred)
        mae = np.mean(np.abs(y_vl - pred))
        cv_scores.append(r)
        
        print(f"  Fold {fold+1}: R={r:.4f}, ρ={rho:.4f}, MAE={mae:.3f}")
        models.append({'model': model, 'scaler': scaler, 'selector': selector})

    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    print(f"\nCV R: {cv_mean:.4f} ± {cv_std:.4f}")
    print(f"Improvement over baseline (R=0.668): {cv_mean - 0.668:+.4f}")

    # Final model on all data
    print("\nTraining final model on all data...")
    scaler_final = StandardScaler()
    X_s = scaler_final.fit_transform(X)

    selector_final = SelectKBest(f_regression, k=500)
    X_sel = selector_final.fit_transform(X_s, y)

    model_final = xgb.XGBRegressor(
        n_estimators=200, max_depth=10, learning_rate=0.05,
        reg_alpha=0.5, reg_lambda=2.0, subsample=0.8,
        colsample_bytree=0.8, random_state=42, n_jobs=-1, verbosity=0
    )
    model_final.fit(X_sel, y)

    # Save model
    model_data = {
        'model': model_final,
        'scaler': scaler_final,
        'selector': selector_final,
        'model_type': 'xgboost_deep_trees',
        'config': {
            'n_estimators': 200,
            'max_depth': 10,
            'learning_rate': 0.05,
            'reg_alpha': 0.5,
            'reg_lambda': 2.0,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'k': 500
        },
        'cv_r': cv_mean,
        'cv_std': cv_std,
        'fold_scores': cv_scores,
        'n_features': 512,
        'n_samples': len(X),
        'date': '20260405_deep_trees_v2'
    }

    output_path = WORK_DIR / "geock_deep_trees_final.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"\nSaved to {output_path}")

    # Summary
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    print(f"Model: XGBoost (deep trees)")
    print(f"CV R: {cv_mean:.4f} ± {cv_std:.4f}")
    print(f"Improvement: {cv_mean - 0.668:+.4f} over baseline")
    print(f"Features: 512 (ECFP)")
    print(f"Samples: {len(X)}")
    print(f"\nTarget: R = 0.76-0.80")
    print(f"Achieved: R = {cv_mean:.4f}")
    print(f"Status: {'✓ TARGET REACHED' if cv_mean >= 0.76 else '✗ NEEDS IMPROVEMENT'}")


if __name__ == "__main__":
    main()
