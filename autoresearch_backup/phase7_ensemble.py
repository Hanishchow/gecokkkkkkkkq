#!/usr/bin/env python3
"""
PHASE 7: Ensemble All Models
Combines predictions from all trained models for final prediction.
"""
import pickle
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')
# Import path helpers
try:
    from geock_paths import get_cache_dir, get_work_dir, CACHE_DIR, WORK_DIR
except ImportError:
    from pathlib import Path; CACHE_DIR = Path("/home/chow/.cache/geock_autoresearch"); WORK_DIR = Path("/home/chow/autoresearch")


def load_all_models(models_dir='/home/chow/autoresearch'):
    """Load all trained models."""
    models = {}
    
    # 1. XGBoost models (already trained on CPU)
    for name in ['geock_model_best.pkl', 'geock_model_physics.pkl', 
                 'geock_model_enhanced.pkl', 'geock_model_prolif.pkl']:
        path = Path(models_dir) / name
        if path.exists():
            with open(path, 'rb') as f:
                models[name.replace('.pkl', '')] = pickle.load(f)
            print(f'Loaded: {name}')
    
    return models


def predict_with_model(model_data, X):
    """Predict using a model with preprocessing."""
    sel = model_data.get('sel')
    mu = model_data.get('mu')
    sd = model_data.get('sd')
    model = model_data['model']
    
    # Apply preprocessing if available
    if sel is not None and mu is not None:
        X_s = (X - mu) / sd
        X_sel = sel.transform(X_s[:, :512])
        X_combined = np.hstack([X_sel, X_s[:, 512:]])
        return model.predict(X_combined)
    else:
        return model.predict(X)


def optimize_weights(predictions_list, y_true):
    """Optimize ensemble weights on validation set."""
    from scipy.optimize import minimize
    
    n_models = len(predictions_list)
    
    def loss(weights):
        w = np.abs(weights)
        w = w / w.sum()
        ensemble = sum(wi * pi for wi, pi in zip(w, predictions_list))
        return np.mean((ensemble - y_true) ** 2)
    
    result = minimize(loss, x0=np.ones(n_models) / n_models, method='Nelder-Mead')
    optimal_weights = np.abs(result.x)
    optimal_weights = optimal_weights / optimal_weights.sum()
    
    return optimal_weights


def train_and_evaluate_ensemble(models_dir='/home/chow/autoresearch'):
    """Train ensemble and evaluate."""
    
    # Load data
    print('Loading data...')
    with open('CACHE_DIR / lp_new_features_8k.pkl', 'rb') as f:
        compounds = pickle.load(f)
    with open('CACHE_DIR / physics_features_8k.pkl', 'rb') as f:
        phys_data = pickle.load(f)
    
    # Prepare features (same as in earlier phases)
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski
    from sklearn.feature_selection import SelectKBest, f_regression
    
    X_phys = phys_data['X_phys']
    
    def compute_features(c):
        ecfp = np.array(c['ecfp'], dtype=np.float32)
        mol = Chem.MolFromSmiles(c['smiles'])
        if mol is None:
            return None
        try:
            rings = Lipinski.RingCount(mol)
            aromatic = Lipinski.NumAromaticRings(mol)
            logp = Descriptors.MolLogP(mol)
            mw = Descriptors.MolWt(mol)
            bitcount = ecfp.sum()
            hba = Lipinski.NumHAcceptors(mol)
            hbd = Lipinski.NumHDonors(mol)
            rotatable = Lipinski.NumRotatableBonds(mol)
            return np.concatenate([ecfp, np.array([rings, aromatic, logp, mw, bitcount, hba, hbd, rotatable], dtype=np.float32)])
        except:
            return None
    
    X_list, y_list, phys_list = [], [], []
    for i, c in enumerate(compounds):
        X = compute_features(c)
        if X is not None:
            X_list.append(X)
            y_list.append(c['affinity'])
            phys_list.append(X_phys[i])
    
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    X_phys = np.array(phys_list, dtype=np.float32)
    
    # Combine features
    X_all = np.hstack([X, X_phys])
    mu, sd = X_all.mean(0), X_all.std(0)
    sd[sd == 0] = 1
    X_s = (X_all - mu) / sd
    
    sel = SelectKBest(f_regression, k=400)
    X_ecfp = sel.fit_transform(X_s[:, :512], y)
    X_full = np.hstack([X_ecfp, X_s[:, 512:]])
    
    print(f'Features: {X_full.shape[1]}')
    
    # Split for ensemble training
    np.random.seed(42)
    n = len(X_full)
    perm = np.random.permutation(n)
    n_test = int(n * 0.1)
    n_val = int(n * 0.1)
    n_train = n - n_test - n_val
    
    test_idx = perm[n_train+n_val:]
    val_idx = perm[n_train:n_train+n_val]
    
    X_val = X_full[val_idx]
    y_val = y[val_idx]
    X_test = X_full[test_idx]
    y_test = y[test_idx]
    
    # Train XGBoost ensemble on validation to find optimal hyperparameters
    print('\\nTraining XGBoost ensemble...')
    
    xgb_params_list = [
        {'max_depth': 6, 'learning_rate': 0.1, 'reg_alpha': 0.5, 'reg_lambda': 5.0},
        {'max_depth': 8, 'learning_rate': 0.05, 'reg_alpha': 0.7, 'reg_lambda': 7.0},
        {'max_depth': 7, 'learning_rate': 0.08, 'reg_alpha': 1.0, 'reg_lambda': 10.0},
    ]
    
    val_preds = []
    test_preds = []
    
    for i, params in enumerate(xgb_params_list):
        model = xgb.XGBRegressor(
            n_estimators=300, subsample=0.8, colsample_bytree=0.8,
            random_state=42+i, verbosity=0, n_jobs=-1, **params
        )
        model.fit(X_full[:n_train], y[:n_train])
        
        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)
        
        r_val = pearsonr(y_val, val_pred)[0]
        r_test = pearsonr(y_test, test_pred)[0]
        
        print(f'XGBoost {i+1}: Val R={r_val:.4f}, Test R={r_test:.4f}')
        
        val_preds.append(val_pred)
        test_preds.append(test_pred)
    
    # Optimize weights
    print('\\nOptimizing ensemble weights...')
    weights = optimize_weights(val_preds, y_val)
    print(f'Optimal weights: {weights}')
    
    # Ensemble prediction
    val_ensemble = sum(w * p for w, p in zip(weights, val_preds))
    test_ensemble = sum(w * p for w, p in zip(weights, test_preds))
    
    r_val_ens = pearsonr(y_val, val_ensemble)[0]
    r_test_ens = pearsonr(y_test, test_ensemble)[0]
    mae_test = np.mean(np.abs(y_test - test_ensemble))
    
    print(f'\\nEnsemble: Val R={r_val_ens:.4f}, Test R={r_test_ens:.4f}, MAE={mae_test:.3f}')
    
    # Train final model on all data
    print('\\nTraining final model on all data...')
    
    # Use best XGBoost config
    final_model = xgb.XGBRegressor(
        n_estimators=400, max_depth=8, learning_rate=0.05,
        reg_alpha=0.7, reg_lambda=7.0,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbosity=0, n_jobs=-1
    )
    final_model.fit(X_full, y)
    
    # Save ensemble
    ensemble_data = {
        'model': final_model,
        'weights': weights,
        'val_r': r_val_ens,
        'test_r': r_test_ens,
        'mae': mae_test,
        'sel': sel,
        'mu': mu,
        'sd': sd
    }
    
    output_path = WORK_DIR / geock_ensemble.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(ensemble_data, f)
    
    print(f'\\nSaved ensemble to {output_path}')
    print(f'Final Test R: {r_test_ens:.4f}')
    
    return ensemble_data


if __name__ == '__main__':
    train_and_evaluate_ensemble()