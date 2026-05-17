#!/usr/bin/env python3
"""
Optimize Enhanced Features Model
===============================
Reduce overfitting and improve generalization.
"""

import os
import sys
import pickle
import json
import logging
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

WORK_DIR = Path("/home/chow/autoresearch")
CACHE_DIR = Path("/home/chow/.cache/geock_autoresearch")


def main():
    log.info("Loading enhanced features...")
    with open(CACHE_DIR / "lp_features_enhanced.pkl", 'rb') as f:
        enhanced_data = pickle.load(f)
    
    log.info(f"Enhanced features: {len(enhanced_data)} records")
    
    # Prepare data
    records = [r for r in enhanced_data if 'features' in r and 'affinity' in r]
    X = np.array([r['features'] for r in records])
    y = np.array([r['affinity'] for r in records])
    
    log.info(f"Feature matrix: {X.shape}")
    log.info(f"Affinity range: {y.min():.2f} - {y.max():.2f}")
    
    # Test multiple configurations
    configs = [
        {'name': 'Strong Regularization', 'k': 400, 'reg_lambda': 20.0, 'reg_alpha': 5.0, 'min_child_weight': 10},
        {'name': 'Moderate Features', 'k': 500, 'reg_lambda': 15.0, 'reg_alpha': 2.0, 'min_child_weight': 8},
        {'name': 'Fewer Features', 'k': 300, 'reg_lambda': 10.0, 'reg_alpha': 1.0, 'min_child_weight': 5},
        {'name': 'Balanced', 'k': 400, 'reg_lambda': 12.0, 'reg_alpha': 2.0, 'min_child_weight': 7},
    ]
    
    best_config = None
    best_score = 0
    best_model_data = None
    
    for config in configs:
        log.info(f"\nTesting: {config['name']}")
        log.info(f"  k={config['k']}, reg_lambda={config['reg_lambda']}, reg_alpha={config['reg_alpha']}")
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
        
        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Feature selection
        selector = SelectKBest(f_regression, k=min(config['k'], X.shape[1]))
        X_train_sel = selector.fit_transform(X_train_scaled, y_train)
        X_val_sel = selector.transform(X_val_scaled)
        X_test_sel = selector.transform(X_test_scaled)
        
        # Train XGBoost with early stopping
        model = xgb.XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.03,
            reg_alpha=config['reg_alpha'],
            reg_lambda=config['reg_lambda'],
            subsample=0.8,
            colsample_bytree=0.7,
            min_child_weight=config['min_child_weight'],
            gamma=0.1,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
            early_stopping_rounds=50
        )
        
        model.fit(
            X_train_sel, y_train,
            eval_set=[(X_val_sel, y_val)],
            verbose=False
        )
        
        train_pred = model.predict(X_train_sel)
        val_pred = model.predict(X_val_sel)
        test_pred = model.predict(X_test_sel)
        
        train_r, _ = pearsonr(y_train, train_pred)
        val_r, _ = pearsonr(y_val, val_pred)
        test_r, _ = pearsonr(y_test, test_pred)
        test_mae = np.mean(np.abs(y_test - test_pred))
        gap = train_r - test_r
        
        log.info(f"  Train R: {train_r:.4f}, Val R: {val_r:.4f}, Test R: {test_r:.4f}")
        log.info(f"  MAE: {test_mae:.3f}, Gap: {gap:.4f}")
        
        if val_r > best_score:
            best_score = val_r
            best_config = config
            best_model_data = {
                'model': model,
                'scaler': scaler,
                'selector': selector,
                'config': config,
                'val_r': val_r,
                'test_r': test_r,
                'test_mae': test_mae
            }
    
    log.info(f"\n{'='*50}")
    log.info(f"Best config: {best_config['name']}")
    log.info(f"Validation R: {best_score:.4f}")
    
    # Run full CV with best config
    log.info("\nRunning 5-fold CV with best config...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, X_vl = X[train_idx], X[val_idx]
        y_tr, y_vl = y[train_idx], y[val_idx]
        
        scaler_fold = StandardScaler()
        X_tr_s = scaler_fold.fit_transform(X_tr)
        X_vl_s = scaler_fold.transform(X_vl)
        
        selector_fold = SelectKBest(f_regression, k=min(best_config['k'], X.shape[1]))
        X_tr_sel = selector_fold.fit_transform(X_tr_s, y_tr)
        X_vl_sel = selector_fold.transform(X_vl_s)
        
        model_fold = xgb.XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.03,
            reg_alpha=best_config['reg_alpha'],
            reg_lambda=best_config['reg_lambda'],
            subsample=0.8,
            colsample_bytree=0.7,
            min_child_weight=best_config['min_child_weight'],
            gamma=0.1,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
            early_stopping_rounds=50
        )
        
        model_fold.fit(X_tr_sel, y_tr, eval_set=[(X_vl_sel, y_vl)], verbose=False)
        pred = model_fold.predict(X_vl_sel)
        r, _ = pearsonr(y_vl, pred)
        cv_scores.append(r)
        log.info(f"  Fold {fold+1}: R={r:.4f}")
    
    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    log.info(f"CV R: {cv_mean:.4f} ± {cv_std:.4f}")
    
    # Final training with all data
    log.info("\nTraining final model...")
    scaler_final = StandardScaler()
    X_scaled = scaler_final.fit_transform(X)
    
    selector_final = SelectKBest(f_regression, k=min(best_config['k'], X.shape[1]))
    X_sel = selector_final.fit_transform(X_scaled, y)
    
    model_final = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.03,
        reg_alpha=best_config['reg_alpha'],
        reg_lambda=best_config['reg_lambda'],
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=best_config['min_child_weight'],
        gamma=0.1,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    model_final.fit(X_sel, y)
    
    # Save model
    model_data = {
        'model': model_final,
        'scaler': scaler_final,
        'selector': selector_final,
        'model_type': 'xgboost_optimized',
        'config': best_config,
        'cv_r': cv_mean,
        'cv_std': cv_std,
        'val_r': best_model_data['val_r'],
        'test_r': best_model_data['test_r'],
        'test_mae': best_model_data['test_mae'],
        'n_features': best_config['k'],
        'n_samples': len(X),
        'date': '20260405_optimized'
    }
    
    output_path = WORK_DIR / "geock_optimized_v2.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump(model_data, f)
    log.info(f"Saved to {output_path}")
    
    # Compare
    log.info("\n" + "="*50)
    log.info("RESULTS SUMMARY")
    log.info("="*50)
    baselines = {
        'ECFP only (baseline)': 0.668,
        'Enhanced features (original)': 0.7049,
        'Optimized XGBoost': cv_mean
    }
    for name, r in baselines.items():
        log.info(f"  {name}: R={r:.4f}")


if __name__ == "__main__":
    main()
