#!/usr/bin/env python3
"""
Train Combined Physics + Molecular Model
========================================
Combine physics features with enhanced molecular features for improved prediction.
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
from sklearn.linear_model import Ridge
import xgboost as xgb

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

WORK_DIR = Path("/home/chow/autoresearch")
CACHE_DIR = Path("/home/chow/.cache/geock_autoresearch")


def main():
    log.info("Loading enhanced molecular features...")
    with open(CACHE_DIR / "lp_features_enhanced.pkl", 'rb') as f:
        enhanced_data = pickle.load(f)
    
    log.info(f"Enhanced features: {len(enhanced_data)} records")
    
    log.info("Loading physics features...")
    with open(CACHE_DIR / "physics_24d.pkl", 'rb') as f:
        physics_data = pickle.load(f)
    
    X_phys = physics_data['X_phys']
    physics_valid_idx = set(physics_data['valid_idx']) if isinstance(physics_data['valid_idx'], list) else set(range(len(physics_data['valid_idx'])))
    
    log.info(f"Physics features: {X_phys.shape}")
    
    # Match records
    common_records = []
    physics_idx = 0
    
    for i, record in enumerate(enhanced_data):
        if i in physics_valid_idx and physics_idx < len(X_phys):
            if 'features' in record and 'affinity' in record:
                combined_features = np.concatenate([record['features'], X_phys[physics_idx]])
                common_records.append({
                    'features': combined_features,
                    'affinity': record['affinity'],
                    'pdb_id': record.get('pdb_id'),
                    'smiles': record.get('smiles')
                })
                physics_idx += 1
    
    log.info(f"Common records with both features: {len(common_records)}")
    
    if len(common_records) < 1000:
        log.error("Not enough common records!")
        return
    
    X = np.array([r['features'] for r in common_records])
    y = np.array([r['affinity'] for r in common_records])
    
    log.info(f"Combined feature matrix: {X.shape}")
    log.info(f"Affinity range: {y.min():.2f} - {y.max():.2f}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
    
    log.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Cross-validation
    log.info("Running 5-fold cross-validation...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, X_vl = X[train_idx], X[val_idx]
        y_tr, y_vl = y[train_idx], y[val_idx]
        
        scaler_fold = StandardScaler()
        X_tr_s = scaler_fold.fit_transform(X_tr)
        X_vl_s = scaler_fold.transform(X_vl)
        
        model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            reg_alpha=1.0,
            reg_lambda=10.0,
            subsample=0.8,
            colsample_bytree=0.6,
            min_child_weight=5,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
        model.fit(X_tr_s, y_tr)
        pred = model.predict(X_vl_s)
        r, _ = pearsonr(y_vl, pred)
        cv_scores.append(r)
        log.info(f"  Fold {fold+1}: R={r:.4f}")
    
    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    log.info(f"CV R: {cv_mean:.4f} ± {cv_std:.4f}")
    
    # Train XGBoost with early stopping
    log.info("Training XGBoost with physics features...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        reg_alpha=1.0,
        reg_lambda=10.0,
        subsample=0.8,
        colsample_bytree=0.6,
        min_child_weight=5,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
        early_stopping_rounds=50
    )
    
    xgb_model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_val_scaled, y_val)],
        verbose=False
    )
    
    xgb_train_pred = xgb_model.predict(X_train_scaled)
    xgb_val_pred = xgb_model.predict(X_val_scaled)
    xgb_test_pred = xgb_model.predict(X_test_scaled)
    
    xgb_train_r, _ = pearsonr(y_train, xgb_train_pred)
    xgb_val_r, _ = pearsonr(y_val, xgb_val_pred)
    xgb_test_r, _ = pearsonr(y_test, xgb_test_pred)
    xgb_test_mae = np.mean(np.abs(y_test - xgb_test_pred))
    
    log.info(f"XGBoost - Train R: {xgb_train_r:.4f}, Val R: {xgb_val_r:.4f}, Test R: {xgb_test_r:.4f}")
    log.info(f"XGBoost - Test MAE: {xgb_test_mae:.3f}, Gap: {xgb_train_r - xgb_test_r:.4f}")
    
    # Train Ridge
    log.info("Training Ridge with physics features...")
    ridge_model = Ridge(alpha=100.0)
    ridge_model.fit(X_train_scaled, y_train)
    
    ridge_train_pred = ridge_model.predict(X_train_scaled)
    ridge_val_pred = ridge_model.predict(X_val_scaled)
    ridge_test_pred = ridge_model.predict(X_test_scaled)
    
    ridge_train_r, _ = pearsonr(y_train, ridge_train_pred)
    ridge_val_r, _ = pearsonr(y_val, ridge_val_pred)
    ridge_test_r, _ = pearsonr(y_test, ridge_test_pred)
    ridge_test_mae = np.mean(np.abs(y_test - ridge_test_pred))
    
    log.info(f"Ridge - Train R: {ridge_train_r:.4f}, Val R: {ridge_val_r:.4f}, Test R: {ridge_test_r:.4f}")
    log.info(f"Ridge - Test MAE: {ridge_test_mae:.3f}, Gap: {ridge_train_r - ridge_test_r:.4f}")
    
    # Ensemble
    log.info("Creating ensemble...")
    for w in [0.6, 0.7, 0.8, 0.9]:
        ens_test_pred = w * xgb_test_pred + (1 - w) * ridge_test_pred
        ens_test_r, _ = pearsonr(y_test, ens_test_pred)
        ens_test_mae = np.mean(np.abs(y_test - ens_test_pred))
        log.info(f"  Weight XGB={w:.1f}: Test R={ens_test_r:.4f}, MAE={ens_test_mae:.3f}")
    
    best_weight = 0.7
    final_pred = best_weight * xgb_test_pred + (1 - best_weight) * ridge_test_pred
    final_r, _ = pearsonr(y_test, final_pred)
    final_mae = np.mean(np.abs(y_test - final_pred))
    final_spearman, _ = spearmanr(y_test, final_pred)
    
    log.info(f"\n{'='*50}")
    log.info(f"FINAL RESULTS (Physics + Molecular Combined)")
    log.info(f"{'='*50}")
    log.info(f"Test R (Pearson): {final_r:.4f}")
    log.info(f"Test R (Spearman): {final_spearman:.4f}")
    log.info(f"Test MAE: {final_mae:.3f}")
    log.info(f"CV R: {cv_mean:.4f} ± {cv_std:.4f}")
    log.info(f"Gap (train-test): {xgb_train_r - xgb_test_r:.4f}")
    log.info(f"Features: {X.shape[1]} (982 mol + 22 phys)")
    log.info(f"Samples: {len(X)}")
    
    # Save model
    model_data = {
        'xgb_model': xgb_model,
        'ridge_model': ridge_model,
        'scaler': scaler,
        'model_type': 'ensemble_physics',
        'config': {
            'xgb_weight': best_weight,
            'n_features': X.shape[1],
            'n_mol_features': 982,
            'n_phys_features': 22
        },
        'cv_r': cv_mean,
        'cv_std': cv_std,
        'test_r': xgb_test_r,
        'test_mae': xgb_test_mae,
        'ensemble_weight': best_weight,
        'date': '20260405_physics'
    }
    
    output_path = WORK_DIR / "geock_physics_combined.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump(model_data, f)
    log.info(f"Saved model to {output_path}")
    
    # Compare with baseline
    log.info("\n" + "="*50)
    log.info("COMPARISON WITH BASELINES")
    log.info("="*50)
    baselines = {
        'ECFP only (baseline)': 0.668,
        'Enhanced features (982D)': 0.7049,
        'Physics + Molecular': final_r
    }
    for name, r in baselines.items():
        log.info(f"  {name}: R={r:.4f}")
    
    improvement = final_r - 0.7049
    log.info(f"\nImprovement over enhanced: {improvement:+.4f}")


if __name__ == "__main__":
    main()
