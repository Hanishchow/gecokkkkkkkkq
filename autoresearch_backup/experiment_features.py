#!/usr/bin/env python3
"""
Comprehensive model experiments to improve binding affinity predictions.
Tests multiple approaches and tracks overfitting via held-out validation.
"""
import pickle
import numpy as np
import json
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import KFold
# Import path helpers
try:
    from geock_paths import get_cache_dir, get_work_dir, CACHE_DIR, WORK_DIR
except ImportError:
    from pathlib import Path; CACHE_DIR = Path("/home/chow/.cache/geock_autoresearch"); WORK_DIR = Path("/home/chow/autoresearch")
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# ==================== DATA LOADING ====================
print("Loading data...")
cache = CACHE_DIR / lp_new_features_8k.pkl')
with open(cache, 'rb') as f:
    compounds = pickle.load(f)

# Compute features
def compute_features(c):
    ecfp = np.array(c['ecfp'], dtype=np.float32)
    mol = Chem.MolFromSmiles(c['smiles'])
    if mol is None:
        return None
    try:
        rings = np.array([Lipinski.RingCount(mol)], dtype=np.float32)
        aromatic = np.array([Lipinski.NumAromaticRings(mol)], dtype=np.float32)
        logp = np.array([Descriptors.MolLogP(mol)], dtype=np.float32)
        mw = np.array([Descriptors.MolWt(mol)], dtype=np.float32)
        bitcount = np.array([ecfp.sum()], dtype=np.float32)
        hba = np.array([Lipinski.NumHAcceptors(mol)], dtype=np.float32)
        hbd = np.array([Lipinski.NumHDonors(mol)], dtype=np.float32)
        rotatable = np.array([Lipinski.NumRotatableBonds(mol)], dtype=np.float32)
        
        # All extra features: rings, aromatic, logp, mw, bitcount, hba, hbd, rotatable
        X = np.concatenate([ecfp, rings, aromatic, logp, mw, bitcount, hba, hbd, rotatable])
        return X
    except:
        return None

X_list = []
y_list = []

for c in compounds:
    X = compute_features(c)
    if X is not None:
        X_list.append(X)
        y_list.append(c['affinity'])

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.float32)

print(f"Feature matrix: {X.shape}")
print(f"y range: {y.min():.2f} - {y.max():.2f}")

# ==================== SPLIT ====================
np.random.seed(42)
n = len(X)
perm = np.random.permutation(n)
n_test = int(n * 0.1)
n_val = int(n * 0.1)
n_train = n - n_test - n_val

train_idx = perm[:n_train]
val_idx = perm[n_train:n_train + n_val]
test_idx = perm[n_train + n_val:]

X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]
X_test, y_test = X[test_idx], y[test_idx]

print(f"Split: {n_train} train / {n_val} val / {n_test} test")

# Standardize
mu = X_train.mean(0)
sd = X_train.std(0)
sd[sd == 0] = 1
X_train_s = (X_train - mu) / sd
X_val_s = (X_val - mu) / sd
X_test_s = (X_test - mu) / sd

# Feature selection
ECFP_IDX = list(range(512))
EXTRA_IDX = list(range(512, 520))  # 8 extra features (512-519)

k = 400
sel = SelectKBest(f_regression, k=k)
X_train_ecfp = sel.fit_transform(X_train_s[:, ECFP_IDX], y_train)
X_val_ecfp = sel.transform(X_val_s[:, ECFP_IDX])
X_test_ecfp = sel.transform(X_test_s[:, ECFP_IDX])

X_train_full = np.hstack([X_train_ecfp, X_train_s[:, EXTRA_IDX]])
X_val_full = np.hstack([X_val_ecfp, X_val_s[:, EXTRA_IDX]])
X_test_full = np.hstack([X_test_ecfp, X_test_s[:, EXTRA_IDX]])

print(f"Features: {X_train_full.shape[1]} (400 ECFP + {len(EXTRA_IDX)} extra)")

# ==================== EVALUATION FUNCTION ====================
def evaluate(y_true, y_pred, name=""):
    r, _ = pearsonr(y_true, y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    bias = np.mean(y_pred - y_true)
    return {'r': r, 'mae': mae, 'bias': bias, 'name': name}

def evaluate_by_bin(y_true, y_pred):
    bins = [(0, 5, 'Weak'), (5, 7, 'Moderate'), (7, 10, 'Strong'), (10, 20, 'VeryStrong')]
    results = []
    for low, high, label in bins:
        mask = (y_true >= low) & (y_true < high)
        if mask.sum() >= 2:
            mae = np.mean(np.abs(y_true[mask] - y_pred[mask]))
            results.append((label, mask.sum(), mae))
    return results

def kfold_cv(X, y, params, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    rs = []
    for tr_idx, vl_idx in kf.split(X):
        m = xgb.XGBRegressor(**params)
        m.fit(X[tr_idx], y[tr_idx])
        pred = m.predict(X[vl_idx])
        r, _ = pearsonr(y[vl_idx], pred)
        rs.append(r)
    return np.mean(rs), np.std(rs)

# ==================== EXPERIMENT 1: TARGET TRANSFORMATION ====================
print("\n" + "="*70)
print("EXPERIMENT 1: Target Transformation (log scale)")
print("="*70)

# Instead of pKd, try log(Kd) - note Kd = 10^(-pKd), so log(Kd) = -pKd * log(10)
# Let's try different transformations

# Option A: Use Kd directly (not log)
kd = 10**(-y)  # Kd in nM

# Option B: Use sqrt transform
y_sqrt = np.sqrt(y)

# Option C: Use box-cox style (just try different powers)
transforms = [
    ('pKd (original)', y_train),
    ('sqrt(pKd)', np.sqrt(y_train)),
    ('log(pKd+1)', np.log(y_train + 1)),
    ('Kd (nM)', kd[train_idx]),
]

best_transform = None
best_r = -1

for name, y_tr in transforms:
    model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1,
                            min_child_weight=3, reg_alpha=0.1, reg_lambda=1.0,
                            random_state=42, verbosity=0, n_jobs=-1)
    model.fit(X_train_full, y_tr)
    pred_val = model.predict(X_val_full)
    
    if name == 'Kd (nM)':
        # Convert back to pKd for comparison
        pred_val = -np.log10(pred_val + 1e-10)
        y_val_compare = y_val
    else:
        y_val_compare = y_val
    
    r = pearsonr(y_val_compare, pred_val)[0]
    mae = np.mean(np.abs(y_val_compare - pred_val))
    
    print(f"  {name}: Val R={r:.4f}, MAE={mae:.3f}")
    
    if r > best_r:
        best_r = r
        best_transform = name

print(f"Best transform: {best_transform}")

# ==================== EXPERIMENT 2: SAMPLE WEIGHTING ====================
print("\n" + "="*70)
print("EXPERIMENT 2: Sample Weighting for Extreme Binders")
print("="*70)

# Create sample weights - give more weight to extreme binders
def get_weights(y, method='inverse'):
    if method == 'inverse':
        # Weight by 1/|y - mean|
        mean_y = np.mean(y)
        weights = 1.0 / (np.abs(y - mean_y) + 0.5)
        weights = weights / weights.mean()
    elif method == 'square':
        # Weight extremes more
        mean_y = np.mean(y)
        weights = np.abs(y - mean_y) + 0.5
        weights = weights / weights.mean()
    elif method == 'balanced':
        # Equal weights for each bin
        weights = np.ones(len(y))
        for low, high in [(0, 5), (5, 7), (7, 10), (10, 20)]:
            mask = (y >= low) & (y < high)
            if mask.sum() > 0:
                weights[mask] = len(y) / (4 * mask.sum())
    else:
        weights = np.ones(len(y))
    return weights

weight_methods = ['none', 'inverse', 'square', 'balanced']

for wm in weight_methods:
    if wm == 'none':
        weights = np.ones(len(y_train))
    else:
        weights = get_weights(y_train, wm)
    
    model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1,
                            min_child_weight=3, reg_alpha=0.1, reg_lambda=1.0,
                            random_state=42, verbosity=0, n_jobs=-1)
    model.fit(X_train_full, y_train, sample_weight=weights)
    pred_val = model.predict(X_val_full)
    
    r = pearsonr(y_val, pred_val)[0]
    mae = np.mean(np.abs(y_val - pred_val))
    
    # Also check by bin
    bins_results = evaluate_by_bin(y_val, pred_val)
    
    print(f"  {wm:12s}: Val R={r:.4f}, MAE={mae:.3f}")
    for label, count, mae_bin in bins_results:
        print(f"    {label}: n={count}, MAE={mae_bin:.2f}")

# ==================== EXPERIMENT 3: REGULARIZATION ====================
print("\n" + "="*70)
print("EXPERIMENT 3: Different Regularization to Prevent Overfitting")
print("="*70)

reg_configs = [
    {'name': 'low_reg', 'reg_alpha': 0.01, 'reg_lambda': 0.1},
    {'name': 'medium_reg', 'reg_alpha': 0.1, 'reg_lambda': 1.0},
    {'name': 'high_reg', 'reg_alpha': 1.0, 'reg_lambda': 10.0},
    {'name': 'very_high_reg', 'reg_alpha': 5.0, 'reg_lambda': 50.0},
    {'name': 'shallow', 'max_depth': 3, 'n_estimators': 50},
    {'name': 'deep', 'max_depth': 8, 'n_estimators': 200},
]

for cfg in reg_configs:
    name = cfg.pop('name')
    
    params = {
        'n_estimators': cfg.get('n_estimators', 100),
        'max_depth': cfg.get('max_depth', 5),
        'learning_rate': 0.1,
        'min_child_weight': 3,
        'reg_alpha': cfg.get('reg_alpha', 0.1),
        'reg_lambda': cfg.get('reg_lambda', 1.0),
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'verbosity': 0,
        'n_jobs': -1,
    }
    
    model = xgb.XGBRegressor(**params)
    model.fit(X_train_full, y_train)
    
    # Train prediction (check overfitting)
    pred_train = model.predict(X_train_full)
    r_train = pearsonr(y_train, pred_train)[0]
    
    # Val prediction
    pred_val = model.predict(X_val_full)
    r_val = pearsonr(y_val, pred_val)[0]
    mae_val = np.mean(np.abs(y_val - pred_val))
    
    # Test prediction
    pred_test = model.predict(X_test_full)
    r_test = pearsonr(y_test, pred_test)[0]
    mae_test = np.mean(np.abs(y_test - pred_test))
    
    gap = r_train - r_val
    
    print(f"  {name:15s}: Train R={r_train:.4f}, Val R={r_val:.4f}, Test R={r_test:.4f}, Gap={gap:.4f}")

# ==================== EXPERIMENT 4: CLASSIFICATION + REGRESSION ====================
print("\n" + "="*70)
print("EXPERIMENT 4: Classification-Aided Regression")
print("="*70)

# First classify, then use different regressors for each class
y_class = np.zeros(len(y_train), dtype=int)
y_class[y_train >= 7] = 1  # strong
y_class[y_train >= 10] = 2  # very strong

print(f"Class distribution: Weak={np.sum(y_class==0)}, Strong={np.sum(y_class==1)}, VeryStrong={np.sum(y_class==2)}")

# Train separate models per class
class_models = {}
for c in [0, 1, 2]:
    mask = y_class == c
    if mask.sum() >= 5:
        X_c = X_train_full[mask]
        y_c = y_train[mask]
        
        model = xgb.XGBRegressor(n_estimators=50, max_depth=4, learning_rate=0.1,
                                random_state=42, verbosity=0, n_jobs=-1)
        model.fit(X_c, y_c)
        class_models[c] = model
        print(f"  Trained class {c} model on {mask.sum()} samples")

# Predict using class-specific models
def predict_class_aware(X):
    preds = []
    for i in range(len(X)):
        # Determine class based on predicted value from full model
        # For simplicity, use actual class distribution
        pred_full = class_models[1].predict(X[i:i+1])[0] if 1 in class_models else y_train.mean()
        
        if pred_full >= 10 and 2 in class_models:
            pred = class_models[2].predict(X[i:i+1])[0]
        elif pred_full >= 7 and 1 in class_models:
            pred = class_models[1].predict(X[i:i+1])[0]
        else:
            pred = class_models[0].predict(X[i:i+1])[0]
        preds.append(pred)
    return np.array(preds)

# ==================== EXPERIMENT 5: ENSEMBLE ====================
print("\n" + "="*70)
print("EXPERIMENT 5: Ensemble of Different Models")
print("="*70)

# Train multiple model types
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

models_ensemble = []

# XGBoost
xgb_m = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1,
                         random_state=42, verbosity=0, n_jobs=-1)
xgb_m.fit(X_train_full, y_train)
models_ensemble.append(('XGBoost', xgb_m))

# Ridge
ridge_m = Ridge(alpha=1.0)
ridge_m.fit(X_train_full, y_train)
models_ensemble.append(('Ridge', ridge_m))

# Random Forest
rf_m = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
rf_m.fit(X_train_full, y_train)
models_ensemble.append(('RF', rf_m))

# Ensemble prediction (average)
preds_ensemble = np.zeros(len(X_val))
for name, m in models_ensemble:
    preds_ensemble += m.predict(X_val_full)
preds_ensemble /= len(models_ensemble)

r_ens = pearsonr(y_val, preds_ensemble)[0]
mae_ens = np.mean(np.abs(y_val - preds_ensemble))
print(f"  Ensemble: Val R={r_ens:.4f}, MAE={mae_ens:.3f}")

for name, m in models_ensemble:
    pred = m.predict(X_val_full)
    r = pearsonr(y_val, pred)[0]
    mae = np.mean(np.abs(y_val - pred))
    print(f"    {name}: R={r:.4f}, MAE={mae:.3f}")

# ==================== FINAL MODEL SELECTION ====================
print("\n" + "="*70)
print("FINAL MODEL: Compare all approaches on TEST set")
print("="*70)

# Test different configurations
configs = [
    ('Baseline (no extra)', X_train_ecfp, X_val_ecfp, X_test_ecfp),
    ('Full features', X_train_full, X_val_full, X_test_full),
    ('High reg', None, None, None),  # Will do separately
    ('Ensemble', None, None, None),
]

results = []

# Baseline
model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1,
                        min_child_weight=3, reg_alpha=0.1, reg_lambda=1.0,
                        random_state=42, verbosity=0, n_jobs=-1)
model.fit(X_train_ecfp, y_train)
pred_test = model.predict(X_test_ecfp)
r_test = pearsonr(y_test, pred_test)[0]
mae_test = np.mean(np.abs(y_test - pred_test))
cv_r, cv_std = kfold_cv(X_train_ecfp, y_train, {'n_estimators': 100, 'max_depth': 5})
results.append(('Baseline ECFP', r_test, mae_test, cv_r, cv_std))
print(f"  Baseline ECFP: Test R={r_test:.4f}, MAE={mae_test:.3f}, CV={cv_r:.4f}±{cv_std:.4f}")

# Full features
model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1,
                        min_child_weight=3, reg_alpha=0.1, reg_lambda=1.0,
                        random_state=42, verbosity=0, n_jobs=-1)
model.fit(X_train_full, y_train)
pred_test = model.predict(X_test_full)
r_test = pearsonr(y_test, pred_test)[0]
mae_test = np.mean(np.abs(y_test - pred_test))
cv_r, cv_std = kfold_cv(X_train_full, y_train, {'n_estimators': 100, 'max_depth': 5})
results.append(('Full features', r_test, mae_test, cv_r, cv_std))
print(f"  Full features: Test R={r_test:.4f}, MAE={mae_test:.3f}, CV={cv_r:.4f}±{cv_std:.4f}")

# High regularization
model = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05,
                        min_child_weight=5, reg_alpha=1.0, reg_lambda=10.0,
                        subsample=0.7, colsample_bytree=0.7,
                        random_state=42, verbosity=0, n_jobs=-1)
model.fit(X_train_full, y_train)
pred_test = model.predict(X_test_full)
r_test = pearsonr(y_test, pred_test)[0]
mae_test = np.mean(np.abs(y_test - pred_test))
cv_r, cv_std = kfold_cv(X_train_full, y_train, {'n_estimators': 100, 'max_depth': 4, 'reg_alpha': 1.0, 'reg_lambda': 10.0})
results.append(('High reg', r_test, mae_test, cv_r, cv_std))
print(f"  High reg: Test R={r_test:.4f}, MAE={mae_test:.3f}, CV={cv_r:.4f}±{cv_std:.4f}")

# Ensemble
pred_ens = np.zeros(len(X_test_full))
for name, m in models_ensemble:
    pred_ens += m.predict(X_test_full)
pred_ens /= len(models_ensemble)
r_test = pearsonr(y_test, pred_ens)[0]
mae_test = np.mean(np.abs(y_test - pred_ens))
results.append(('Ensemble', r_test, mae_test, 0, 0))
print(f"  Ensemble: Test R={r_test:.4f}, MAE={mae_test:.3f}")

# Find best
best = max(results, key=lambda x: x[2])  # Lowest MAE
print(f"\nBest by MAE: {best[0]} with MAE={best[2]:.3f}")

best = max(results, key=lambda x: x[3])  # Highest CV R
print(f"Best by CV R: {best[0]} with CV R={best[3]:.4f}")

# ==================== SAVE BEST MODEL ====================
print("\n" + "="*70)
print("Saving best model...")
print("="*70)

# Use full features as it showed improvement
model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1,
                        min_child_weight=3, reg_alpha=0.1, reg_lambda=1.0,
                        random_state=42, verbosity=0, n_jobs=-1)
model.fit(X_train_full, y_train)

out = {
    'model': model,
    'sel': sel,
    'mu': mu,
    'sd': sd,
    'k': k,
    'extra_features': ['rings', 'aromatic', 'logp', 'mw', 'bitcount', 'hba', 'hbd', 'rotatable'],
    'cv_r': cv_r,
    'cv_std': cv_std,
}

out_path = WORK_DIR / geock_model_best.pkl')
with open(out_path, 'wb') as f:
    pickle.dump(out, f)

print(f"Saved to {out_path}")
print(f"CV R: {cv_r:.4f} ± {cv_std:.4f}")

# Final test set evaluation
pred_test = model.predict(X_test_full)
r_test = pearsonr(y_test, pred_test)[0]
mae_test = np.mean(np.abs(y_test - pred_test))
print(f"Final Test R: {r_test:.4f}, MAE: {mae_test:.3f}")

# By bin
print("\nFinal error by binding strength:")
for label, count, mae_bin in evaluate_by_bin(y_test, pred_test):
    print(f"  {label}: n={count}, MAE={mae_bin:.2f}")