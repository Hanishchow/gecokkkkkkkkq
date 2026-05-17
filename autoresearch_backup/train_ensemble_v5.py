#!/usr/bin/env python3
"""
Keep ALL features (no selection) to replicate best result.
"""
import pickle
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')
# Import path helpers
try:
    from geock_paths import get_cache_dir, get_work_dir, CACHE_DIR, WORK_DIR
except ImportError:
    from pathlib import Path; CACHE_DIR = Path("/home/chow/.cache/geock_autoresearch"); WORK_DIR = Path("/home/chow/autoresearch")

# ==================== LOAD DATA ====================
print("Loading data...")
cache = CACHE_DIR / lp_new_features_8k.pkl')
with open(cache, 'rb') as f:
    compounds = pickle.load(f)

# Load physics features
with open('CACHE_DIR / physics_features_8k.pkl', 'rb') as f:
    phys_data = pickle.load(f)
X_phys = phys_data['X_phys']

# Load interaction features
X_interactions = np.load('WORK_DIR / X_interactions.npy')
with open('WORK_DIR / interaction_pdb_ids.pkl', 'rb') as f:
    interaction_pdb_ids = pickle.load(f)

# Create mapping for interaction features
interaction_idx_map = {pdb: i for i, pdb in enumerate(interaction_pdb_ids)}

# ==================== COMPUTE FEATURES ====================
print("\nComputing features...")

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski

X_list = []
y_list = []

for i, c in enumerate(compounds):
    ecfp = np.array(c['ecfp'], dtype=np.float32)
    mol = Chem.MolFromSmiles(c['smiles'])
    if mol is None:
        continue
    
    pdb_id = c['pdb_id']
    
    # Interaction features
    if pdb_id in interaction_idx_map:
        int_idx = interaction_idx_map[pdb_id]
        int_features = X_interactions[int_idx]
    else:
        int_features = np.zeros(20, dtype=np.float32)
    
    try:
        mol_features = [
            Lipinski.RingCount(mol),
            Lipinski.NumAromaticRings(mol),
            Descriptors.MolLogP(mol),
            Descriptors.MolWt(mol),
            ecfp.sum(),
            Lipinski.NumHAcceptors(mol),
            Lipinski.NumHDonors(mol),
            Lipinski.NumRotatableBonds(mol),
        ]
        mol_features = np.array(mol_features, dtype=np.float32)
        
        # Combined: ECFP + mol + physics + interactions (ALL features)
        X = np.concatenate([ecfp, mol_features, X_phys[i], int_features])
        X_list.append(X)
        y_list.append(c['affinity'])
    except:
        continue

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.float32)

print(f"Total features (ALL): {X.shape}")

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

X_train = X[train_idx]
X_val = X[val_idx]
X_test = X[test_idx]
y_train = y[train_idx]
y_val = y[val_idx]
y_test = y[test_idx]

print(f"Split: {n_train} train / {n_val} val / {n_test} test")

# Standardize
mu = X_train.mean(0)
sd = X_train.std(0)
sd[sd == 0] = 1
X_train_s = (X_train - mu) / sd
X_val_s = (X_val - mu) / sd
X_test_s = (X_test - mu) / sd

# ==================== TRAIN ENSEMBLE ====================
print("\n" + "="*70)
print("Training ensemble (ALL features)")
print("="*70)

xgb_configs = [
    {'max_depth': 6, 'learning_rate': 0.1, 'reg_alpha': 0.5, 'reg_lambda': 5.0},
    {'max_depth': 8, 'learning_rate': 0.05, 'reg_alpha': 0.7, 'reg_lambda': 7.0},
    {'max_depth': 7, 'learning_rate': 0.08, 'reg_alpha': 0.5, 'reg_lambda': 5.0},
    {'max_depth': 10, 'learning_rate': 0.03, 'reg_alpha': 1.0, 'reg_lambda': 10.0},
    {'max_depth': 5, 'learning_rate': 0.1, 'reg_alpha': 0.1, 'reg_lambda': 1.0},
]

models = []
val_preds = []
test_preds = []

for i, cfg in enumerate(xgb_configs):
    model = xgb.XGBRegressor(
        n_estimators=300,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        random_state=42 + i,
        verbosity=0,
        n_jobs=-1,
        **cfg
    )
    model.fit(X_train_s, y_train)
    
    pred_val = model.predict(X_val_s)
    pred_test = model.predict(X_test_s)
    
    r_val = pearsonr(y_val, pred_val)[0]
    r_test = pearsonr(y_test, pred_test)[0]
    
    print(f"Model {i+1}: Val R={r_val:.4f}, Test R={r_test:.4f}")
    
    models.append(model)
    val_preds.append(pred_val)
    test_preds.append(pred_test)

# Optimize weights
from scipy.optimize import minimize

def optimize_weights(preds_list, y_true):
    n = len(preds_list)
    def loss(w):
        w = np.abs(w)
        w = w / w.sum()
        ensemble = sum(wi * pi for wi, pi in zip(w, preds_list))
        return np.mean((ensemble - y_true) ** 2)
    result = minimize(loss, x0=np.ones(n) / n, method='Nelder-Mead')
    w = np.abs(result.x)
    return w / w.sum()

weights = optimize_weights(val_preds, y_val)
print(f"\nWeights: {weights}")

val_ensemble = sum(w * p for w, p in zip(weights, val_preds))
test_ensemble = sum(w * p for w, p in zip(weights, test_preds))

r_ens_val = pearsonr(y_val, val_ensemble)[0]
r_ens_test = pearsonr(y_test, test_ensemble)[0]
mae_ens = np.mean(np.abs(y_test - test_ensemble))

print(f"Ensemble: Val R={r_ens_val:.4f}, Test R={r_ens_test:.4f}, MAE={mae_ens:.3f}")

# ==================== CROSS-VALIDATION ====================
print("\n" + "="*70)
print("5-Fold Cross-Validation")
print("="*70)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_rs = []

for fold, (tr_idx, vl_idx) in enumerate(kf.split(X_train_s)):
    X_tr = X_train_s[tr_idx]
    y_tr = y_train[tr_idx]
    X_vl = X_train_s[vl_idx]
    y_vl = y_train[vl_idx]
    
    fold_preds = []
    for cfg in xgb_configs:
        m = xgb.XGBRegressor(
            n_estimators=200,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
            n_jobs=-1,
            **cfg
        )
        m.fit(X_tr, y_tr)
        fold_preds.append(m.predict(X_vl))
    
    fold_ensemble = sum(w * p for w, p in zip(weights, fold_preds))
    r = pearsonr(y_vl, fold_ensemble)[0]
    cv_rs.append(r)
    print(f"Fold {fold+1}: R = {r:.4f}")

cv_mean = np.mean(cv_rs)
cv_std = np.std(cv_rs)
print(f"\nCV R: {cv_mean:.4f} ± {cv_std:.4f}")

# ==================== ERROR BY BINDING STRENGTH ====================
print("\nError by binding strength:")
bins = [(0, 5, 'Weak'), (5, 7, 'Moderate'), (7, 10, 'Strong'), (10, 20, 'VeryStrong')]
for low, high, label in bins:
    mask = (y_test >= low) & (y_test < high)
    if mask.sum() >= 2:
        mae = np.mean(np.abs(y_test[mask] - test_ensemble[mask]))
        bias = np.mean(test_ensemble[mask] - y_test[mask])
        print(f"  {label}: n={mask.sum()}, MAE={mae:.2f}, Bias={bias:+.2f}")

# ==================== SAVE ====================
output = {
    'models': models,
    'weights': weights,
    'configs': xgb_configs,
    'mu': mu,
    'sd': sd,
    'cv_r': cv_mean,
    'cv_std': cv_std,
    'val_r': r_ens_val,
    'test_r': r_ens_test,
    'mae': mae_ens,
}

output_path = WORK_DIR / geock_ensemble_v5.pkl')
with open(output_path, 'wb') as f:
    pickle.dump(output, f)

print(f"\nSaved to {output_path}")
print(f"CV R: {cv_mean:.4f} ± {cv_std:.4f}")
print(f"Val R: {r_ens_val:.4f}")
print(f"Test R: {r_ens_test:.4f}")