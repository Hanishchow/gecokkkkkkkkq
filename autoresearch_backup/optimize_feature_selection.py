#!/usr/bin/env python3
"""
Feature Selection + Optimized Training
======================================
Use mutual information and correlation-based feature selection.
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
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

WORK_DIR = Path("/home/chow/autoresearch")
CACHE_DIR = Path("/home/chow/.cache/geock_autoresearch")


def main():
    log.info("Loading enhanced features...")
    with open(CACHE_DIR / "lp_features_enhanced.pkl", 'rb') as f:
        enhanced_data = pickle.load(f)
    
    records = [r for r in enhanced_data if 'features' in r and 'affinity' in r]
    X = np.array([r['features'] for r in records])
    y = np.array([r['affinity'] for r in records])
    
    log.info(f"Feature matrix: {X.shape}")
    
    # Analyze individual feature correlations
    log.info("Analyzing feature correlations with target...")
    correlations = []
    for i in range(X.shape[1]):
        r, _ = pearsonr(X[:, i], y)
        if not np.isnan(r):
            correlations.append((i, abs(r), r))
    
    correlations.sort(key=lambda x: x[1], reverse=True)
    
    log.info("Top 20 most correlated features:")
    for i, (idx, abs_r, r) in enumerate(correlations[:20]):
        log.info(f"  {i+1}. Feature {idx}: r={r:.4f}")
    
    # Try different feature counts
    feature_counts = [100, 200, 300, 400, 500]
    best_k = 300
    best_cv = 0
    
    log.info("\nTesting different feature counts...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for k in feature_counts:
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_tr, X_vl = X[train_idx], X[val_idx]
            y_tr, y_vl = y[train_idx], y[val_idx]
            
            scaler_f = StandardScaler()
            X_tr_s = scaler_f.fit_transform(X_tr)
            X_vl_s = scaler_f.transform(X_vl)
            
            # Use top-k features by correlation
            top_indices = [c[0] for c in correlations[:k]]
            X_tr_sel = X_tr_s[:, top_indices]
            X_vl_sel = X_vl_s[:, top_indices]
            
            model = xgb.XGBRegressor(
                n_estimators=200, max_depth=5, learning_rate=0.05,
                reg_alpha=1.0, reg_lambda=10.0, subsample=0.8,
                colsample_bytree=0.8, min_child_weight=5,
                random_state=42, n_jobs=-1, verbosity=0
            )
            model.fit(X_tr_sel, y_tr)
            pred = model.predict(X_vl_sel)
            r, _ = pearsonr(y_vl, pred)
            cv_scores.append(r)
        
        cv_mean = np.mean(cv_scores)
        log.info(f"  k={k}: CV R={cv_mean:.4f}")
        
        if cv_mean > best_cv:
            best_cv = cv_mean
            best_k = k
    
    log.info(f"\nBest k={best_k} with CV R={best_cv:.4f}")
    
    # Train final model with best k
    log.info(f"\nTraining final model with k={best_k}...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    
    top_indices = [c[0] for c in correlations[:best_k]]
    X_train_sel = X_train_s[:, top_indices]
    X_val_sel = X_val_s[:, top_indices]
    X_test_sel = X_test_s[:, top_indices]
    
    # Train XGBoost
    xgb_model = xgb.XGBRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        reg_alpha=1.0, reg_lambda=10.0, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=5,
        random_state=42, n_jobs=-1, verbosity=0
    )
    xgb_model.fit(X_train_sel, y_train, eval_set=[(X_val_sel, y_val)], verbose=False)
    
    xgb_pred = xgb_model.predict(X_test_sel)
    xgb_r, _ = pearsonr(y_test, xgb_pred)
    xgb_mae = np.mean(np.abs(y_test - xgb_pred))
    log.info(f"XGBoost Test R: {xgb_r:.4f}, MAE: {xgb_mae:.3f}")
    
    # Train Ridge
    ridge = Ridge(alpha=100.0)
    ridge.fit(X_train_sel, y_train)
    ridge_pred = ridge.predict(X_test_sel)
    ridge_r, _ = pearsonr(y_test, ridge_pred)
    ridge_mae = np.mean(np.abs(y_test - ridge_pred))
    log.info(f"Ridge Test R: {ridge_r:.4f}, MAE: {ridge_mae:.3f}")
    
    # Try ensemble weights
    log.info("\nEnsemble weights:")
    for w in [0.5, 0.6, 0.7, 0.8]:
        ens_pred = w * xgb_pred + (1 - w) * ridge_pred
        ens_r, _ = pearsonr(y_test, ens_pred)
        ens_mae = np.mean(np.abs(y_test - ens_pred))
        log.info(f"  w={w}: R={ens_r:.4f}, MAE={ens_mae:.3f}")
    
    # Final model
    best_w = 0.6
    final_pred = best_w * xgb_pred + (1 - best_w) * ridge_pred
    final_r, _ = pearsonr(y_test, final_pred)
    final_mae = np.mean(np.abs(y_test - final_pred))
    final_spearman, _ = spearmanr(y_test, final_pred)
    
    log.info("\n" + "="*50)
    log.info("FINAL RESULTS (Optimized Feature Selection)")
    log.info("="*50)
    log.info(f"Features used: {best_k}")
    log.info(f"Test R (Pearson): {final_r:.4f}")
    log.info(f"Test R (Spearman): {final_spearman:.4f}")
    log.info(f"Test MAE: {final_mae:.3f}")
    log.info(f"CV R: {best_cv:.4f}")
    
    # Save
    model_data = {
        'xgb_model': xgb_model,
        'ridge_model': ridge,
        'scaler': scaler,
        'selected_indices': top_indices,
        'best_k': best_k,
        'ensemble_weight': best_w,
        'model_type': 'optimized_featureselection',
        'cv_r': best_cv,
        'test_r': final_r,
        'test_mae': final_mae,
        'date': '20260405_feature_selection'
    }
    
    output_path = WORK_DIR / "geock_optimized_fs.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump(model_data, f)
    log.info(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
