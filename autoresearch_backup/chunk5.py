#!/usr/bin/env python3
"""
Chunk 5: Use the CORRECT 24D physics features that original model used
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

print("="*70)
print("CHUNK 5: Using 24D Physics Features (Original)")
print("="*70)

# Load data
cache = CACHE_DIR / lp_new_features_8k.pkl')
with open(cache, 'rb') as f:
    compounds = pickle.load(f)

# Load the CORRECT 24D physics features
with open('CACHE_DIR / physics_24d.pkl', 'rb') as f:
    phys_24d = pickle.load(f)
    print(f"Keys in physics_24d: {phys_24d.keys()}")
    X_phys_24d = phys_24d['X_phys']
    print(f"Physics features shape: {X_phys_24d.shape}")

# Load interaction features
X_int = np.load('WORK_DIR / X_interactions.npy')
with open('WORK_DIR / interaction_pdb_ids.pkl', 'rb') as f:
    int_pdb_ids = pickle.load(f)
int_map = {pdb: i for i, pdb in enumerate(int_pdb_ids)}

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski

# Build features - using 24D physics
X_list, y_list = [], []

for i, c in enumerate(compounds):
    ecfp = np.array(c['ecfp'], dtype=np.float32)
    mol = Chem.MolFromSmiles(c['smiles'])
    if mol is None:
        continue
    
    pdb_id = c['pdb_id']
    int_feat = X_int[int_map[pdb_id]] if pdb_id in int_map else np.zeros(20, dtype=np.float32)
    
    mol_feat = np.array([
        Lipinski.RingCount(mol),
        Lipinski.NumAromaticRings(mol),
        Descriptors.MolLogP(mol),
        Descriptors.MolWt(mol),
        ecfp.sum(),
        Lipinski.NumHAcceptors(mol),
        Lipinski.NumHDonors(mol),
        Lipinski.NumRotatableBonds(mol),
    ], dtype=np.float32)
    
    # Use 24D physics + 20D interaction = 44D extra features
    # Total: 512 ECFP + 8 mol + 24 phys + 20 int = 564
    X = np.concatenate([ecfp, mol_feat, X_phys_24d[i], int_feat])
    X_list.append(X)
    y_list.append(c['affinity'])

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.float32)
print(f"Features (with 24D physics): {X.shape}")

# Split
np.random.seed(42)
n = len(X)
perm = np.random.permutation(n)
n_test = int(n * 0.1)
n_val = int(n * 0.1)
n_train = n - n_test - n_val

idx_tr = perm[:n_train]
idx_vl = perm[n_train:n_train+n_val]
idx_te = perm[n_train+n_val:]

X_tr, y_tr = X[idx_tr], y[idx_tr]
X_vl, y_vl = X[idx_vl], y[idx_vl]
X_te, y_te = X[idx_te], y[idx_te]

mu, sd = X_tr.mean(0), X_tr.std(0)
sd[sd == 0] = 1
X_tr_s = (X_tr - mu) / sd
X_vl_s = (X_vl - mu) / sd
X_te_s = (X_te - mu) / sd

print(f"Split: {n_train}/{n_val}/{n_test}")

# Same configs as original ensemble
configs = [
    {'max_depth': 6, 'learning_rate': 0.1, 'reg_alpha': 0.5, 'reg_lambda': 5.0},
    {'max_depth': 8, 'learning_rate': 0.05, 'reg_alpha': 0.7, 'reg_lambda': 7.0},
    {'max_depth': 7, 'learning_rate': 0.08, 'reg_alpha': 0.5, 'reg_lambda': 5.0},
    {'max_depth': 10, 'learning_rate': 0.03, 'reg_alpha': 1.0, 'reg_lambda': 10.0},
    {'max_depth': 5, 'learning_rate': 0.1, 'reg_alpha': 0.1, 'reg_lambda': 1.0},
]

print("\n--- Training Models ---")
models, preds_vl, preds_te = [], [], []

for i, cfg in enumerate(configs):
    m = xgb.XGBRegressor(n_estimators=300, subsample=0.8, colsample_bytree=0.8,
                         min_child_weight=3, random_state=42+i, verbosity=0, n_jobs=-1, **cfg)
    m.fit(X_tr_s, y_tr)
    
    p_vl = m.predict(X_vl_s)
    p_te = m.predict(X_te_s)
    
    r_vl = pearsonr(y_vl, p_vl)[0]
    r_te = pearsonr(y_te, p_te)[0]
    print(f"Model {i+1}: Val R={r_vl:.4f}, Test R={r_te:.4f}")
    
    models.append(m)
    preds_vl.append(p_vl)
    preds_te.append(p_te)

# Optimize weights
from scipy.optimize import minimize

def opt_w(preds, y):
    n = len(preds)
    def loss(w):
        w = np.abs(w) / np.abs(w).sum()
        return np.mean((sum(wi*pi for wi,pi in zip(w,preds)) - y)**2)
    result = minimize(loss, np.ones(n)/n, method='Nelder-Mead')
    return np.abs(result.x) / np.abs(result.x).sum()

weights = opt_w(preds_vl, y_vl)
print(f"\nWeights: {weights.round(3)}")

ens_vl = sum(w*p for w,p in zip(weights, preds_vl))
ens_te = sum(w*p for w,p in zip(weights, preds_te))

r_ens_vl = pearsonr(y_vl, ens_vl)[0]
r_ens_te = pearsonr(y_te, ens_te)[0]
mae = np.mean(np.abs(y_te - ens_te))
print(f"Ensemble: Val R={r_ens_vl:.4f}, Test R={r_ens_te:.4f}, MAE={mae:.3f}")

# 5-Fold CV
print("\n--- 5-Fold CV ---")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_rs = []

for fold, (tr_idx, vl_idx) in enumerate(kf.split(X_tr_s)):
    fold_preds = []
    for cfg in configs:
        m = xgb.XGBRegressor(n_estimators=200, subsample=0.8, colsample_bytree=0.8,
                             random_state=42, verbosity=0, n_jobs=-1, **cfg)
        m.fit(X_tr_s[tr_idx], y_tr[tr_idx])
        fold_preds.append(m.predict(X_tr_s[vl_idx]))
    
    fold_ens = sum(w*p for w,p in zip(weights, fold_preds))
    cv_rs.append(pearsonr(y_tr[vl_idx], fold_ens)[0])
    print(f"  Fold {fold+1}: {cv_rs[-1]:.4f}")

cv_mean = np.mean(cv_rs)
cv_std = np.std(cv_rs)
print(f"\nCV R: {cv_mean:.4f} ± {cv_std:.4f}")

# Error by strength
print("\n--- Error by strength ---")
for lo, hi, lb in [(0,5,'Weak'),(5,7,'Mod'),(7,10,'Strong'),(10,20,'VStrong')]:
    m = (y_te >= lo) & (y_te < hi)
    if m.sum() >= 2:
        print(f"  {lb}: n={m.sum()}, MAE={np.mean(np.abs(y_te[m]-ens_te[m])):.2f}, Bias={np.mean(ens_te[m]-y_te[m]):+.2f}")

# Save
output = {'models': models, 'weights': weights, 'configs': configs,
          'mu': mu, 'sd': sd, 'cv_r': cv_mean, 'cv_std': cv_std,
          'val_r': r_ens_vl, 'test_r': r_ens_te, 'mae': mae}

with open('WORK_DIR / geock_best_24d.pkl', 'wb') as f:
    pickle.dump(output, f)

print(f"\n✓ Saved: CV R={cv_mean:.4f}, Val={r_ens_vl:.4f}, Test={r_ens_te:.4f}")