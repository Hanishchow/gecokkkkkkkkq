#!/usr/bin/env python3
"""
Train with Original ECFP Features
===============================
Match the original model's data and training setup.
"""

import os
import pickle
import json
import logging
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import Ridge
import xgboost as xgb

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

WORK_DIR = Path("/home/chow/autoresearch")
CACHE_DIR = Path("/home/chow/.cache/geock_autoresearch")


def load_data():
    """Load training data matching original model."""
    all_records = []
    
    # Load from multiple sources
    for path in [
        CACHE_DIR / "lp_new_features_8k.pkl",
        CACHE_DIR / "geock_training_data.pkl"
    ]:
        if path.exists():
            with open(path, 'rb') as f:
                data = pickle.load(f)
            if isinstance(data, list):
                all_records.extend([r for r in data if 'ecfp' in r and 'affinity' in r])
    
    log.info(f"Total records: {len(all_records)}")
    
    X = np.array([r['ecfp'] for r in all_records])
    y = np.array([r['affinity'] for r in all_records])
    
    log.info(f"After loading: {len(X)} records")
    
    return X, y


def main():
    log.info("Loading original ECFP features...")
    X, y = load_data()
    
    log.info(f"Feature matrix: {X.shape}")
    log.info(f"Affinity range: {y.min():.2f} - {y.max():.2f}")
    
    # Configuration matching original
    k = 500
    ensemble_weight = 0.8
    
    # Split like original
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
    
    log.info(f"Split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    
    # Feature selection
    selector = SelectKBest(f_regression, k=min(k, X.shape[1]))
    X_train_sel = selector.fit_transform(X_train_s, y_train)
    X_val_sel = selector.transform(X_val_s)
    X_test_sel = selector.transform(X_test_s)
    
    # Train Ridge
    log.info("Training Ridge...")
    ridge = Ridge(alpha=100.0)
    ridge.fit(X_train_sel, y_train)
    
    ridge_train_pred = ridge.predict(X_train_sel)
    ridge_val_pred = ridge.predict(X_val_sel)
    ridge_test_pred = ridge.predict(X_test_sel)
    
    ridge_train_r, _ = pearsonr(y_train, ridge_train_pred)
    ridge_val_r, _ = pearsonr(y_val, ridge_val_pred)
    ridge_test_r, _ = pearsonr(y_test, ridge_test_pred)
    
    log.info(f"Ridge - Train R: {ridge_train_r:.4f}, Val R: {ridge_val_r:.4f}, Test R: {ridge_test_r:.4f}")
    
    # Train XGBoost
    log.info("Training XGBoost...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        reg_alpha=1.0, reg_lambda=5.0, subsample=0.8,
        colsample_bytree=0.8, random_state=42, n_jobs=-1, verbosity=0
    )
    xgb_model.fit(X_train_sel, y_train, eval_set=[(X_val_sel, y_val)], verbose=False)
    
    xgb_train_pred = xgb_model.predict(X_train_sel)
    xgb_val_pred = xgb_model.predict(X_val_sel)
    xgb_test_pred = xgb_model.predict(X_test_sel)
    
    xgb_train_r, _ = pearsonr(y_train, xgb_train_pred)
    xgb_val_r, _ = pearsonr(y_val, xgb_val_pred)
    xgb_test_r, _ = pearsonr(y_test, xgb_test_pred)
    
    log.info(f"XGBoost - Train R: {xgb_train_r:.4f}, Val R: {xgb_val_r:.4f}, Test R: {xgb_test_r:.4f}")
    
    # Ensemble
    log.info("Creating ensemble...")
    train_pred = ensemble_weight * xgb_train_pred + (1 - ensemble_weight) * ridge_train_pred
    val_pred = ensemble_weight * xgb_val_pred + (1 - ensemble_weight) * ridge_val_pred
    test_pred = ensemble_weight * xgb_test_pred + (1 - ensemble_weight) * ridge_test_pred
    
    ens_train_r, _ = pearsonr(y_train, train_pred)
    ens_val_r, _ = pearsonr(y_val, val_pred)
    ens_test_r, _ = pearsonr(y_test, test_pred)
    ens_test_mae = np.mean(np.abs(y_test - test_pred))
    ens_spearman, _ = spearmanr(y_test, test_pred)
    
    log.info(f"Ensemble - Train R: {ens_train_r:.4f}, Val R: {ens_val_r:.4f}, Test R: {ens_test_r:.4f}")
    log.info(f"Ensemble MAE: {ens_test_mae:.3f}, Spearman: {ens_spearman:.4f}")
    log.info(f"Gap: {ens_train_r - ens_test_r:.4f}")
    
    # 5-fold CV
    log.info("\nRunning 5-fold CV...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, X_vl = X[train_idx], X[val_idx]
        y_tr, y_vl = y[train_idx], y[val_idx]
        
        scaler_f = StandardScaler()
        X_tr_s = scaler_f.fit_transform(X_tr)
        X_vl_s = scaler_f.transform(X_vl)
        
        selector_f = SelectKBest(f_regression, k=min(k, X.shape[1]))
        X_tr_sel = selector_f.fit_transform(X_tr_s, y_tr)
        X_vl_sel = selector_f.transform(X_vl_s)
        
        xgb_f = xgb.XGBRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            reg_alpha=1.0, reg_lambda=5.0, subsample=0.8,
            colsample_bytree=0.8, random_state=42, n_jobs=-1, verbosity=0
        )
        xgb_f.fit(X_tr_sel, y_tr)
        
        ridge_f = Ridge(alpha=100.0)
        ridge_f.fit(X_tr_sel, y_tr)
        
        xgb_pred = xgb_f.predict(X_vl_sel)
        ridge_pred = ridge_f.predict(X_vl_sel)
        pred = ensemble_weight * xgb_pred + (1 - ensemble_weight) * ridge_pred
        
        r, _ = pearsonr(y_vl, pred)
        cv_scores.append(r)
        log.info(f"  Fold {fold+1}: R={r:.4f}")
    
    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    log.info(f"CV R: {cv_mean:.4f} ± {cv_std:.4f}")
    
    # Save model
    model_data = {
        'xgb_model': xgb_model,
        'ridge_model': ridge,
        'scaler': scaler,
        'selector': selector,
        'ensemble_weight': ensemble_weight,
        'model_type': 'ensemble_ecfp',
        'config': {
            'k': k, 'alpha': 100.0, 'ensemble_weight': ensemble_weight,
            'n_features': X.shape[1], 'n_samples': len(X)
        },
        'cv_r': cv_mean,
        'cv_std': cv_std,
        'test_r': ens_test_r,
        'test_mae': ens_test_mae,
        'spearman_r': ens_spearman,
        'train_r': ens_train_r,
        'gap': ens_train_r - ens_test_r,
        'date': '20260405_ecfp_baseline'
    }
    
    output_path = WORK_DIR / "geock_ecfp_baseline.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump(model_data, f)
    log.info(f"Saved to {output_path}")
    
    # Summary
    log.info("\n" + "="*50)
    log.info("RESULTS SUMMARY (Original ECFP Features)")
    log.info("="*50)
    log.info(f"Features: {X.shape[1]} (ECFP)")
    log.info(f"Samples: {len(X)}")
    log.info(f"Test R (Pearson): {ens_test_r:.4f}")
    log.info(f"Test R (Spearman): {ens_spearman:.4f}")
    log.info(f"CV R: {cv_mean:.4f} ± {cv_std:.4f}")
    log.info(f"MAE: {ens_test_mae:.3f}")


if __name__ == "__main__":
    main()
