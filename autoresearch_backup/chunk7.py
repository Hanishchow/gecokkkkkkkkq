#!/usr/bin/env python3
"""
Chunk 7: Try smaller, more regularized models with bagging
"""
import pickle
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from sklearn.ensemble import BaggingRegressor
import xgboost as xgb
import warnings
# Import path helpers
try:
    from geock_paths import get_cache_dir, get_work_dir, CACHE_DIR, WORK_DIR
except ImportError:
    from pathlib import Path; CACHE_DIR = Path("/home/chow/.cache/geock_autoresearch"); WORK_DIR = Path("/home/chow/autoresearch")
warnings.filterwarnings('ignore')

print("="*70)
print("CHUNK 7: Bagging + Smaller Trees")
print("="*70)

# Load data
cache = CACHE_DIR / lp_new_features_8k.pkl')
with open(cache, 'rb') as f:
    compounds = pickle.load(f)

with open('CACHE_DIR / physics_features_8k.pkl', 'rb') as f:
    phys_data = pickle.load(f)
X_phys = phys_data['X_phys']

X_int = np.load('WORK_DIR / X_interactions.npy')
with open('WORK_DIR / interaction_pdb_ids.pkl', 'rb') as f:
    int_pdb_ids = pickle.load(f)
int_map = {pdb: i for i, pdb in enumerate(int_pdb_ids)}

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski

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
    
    X = np.concatenate([ecfp, mol_feat, X_phys[i], int_feat])
    X_list.append(X)
    y_list.append(c['affinity'])

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.float32)
print(f"Features: {X.shape}")

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

# Strategy 1: More trees with very small learning rate
print("\n--- XGB with many trees, tiny LR ---")
xgb_many = xgb.XGBRegressor(
    n_estimators=1000,
    max_depth=5,
    learning_rate=0.01,
    reg_alpha=1.0,
    reg_lambda=10.0,
    subsample=0.7,
    colsample_bytree=0.7,
    min_child_weight=5,
    random_state=42,
    verbosity=0,
    n_jobs=-1
)
xgb_many.fit(X_tr_s, y_tr)
r_many_vl = pearsonr(y_vl, xgb_many.predict(X_vl_s))[0]
r_many_te = pearsonr(y_te, xgb_many.predict(X_te_s))[0]
print(f"Many trees: Val={r_many_vl:.4f}, Test={r_many_te:.4f}")

# Strategy 2: Very shallow trees with bagging
print("\n--- Shallow XGB Bagging ---")
xgb_shallow = xgb.XGBRegressor(
    n_estimators=500,
    max_depth=3,
    learning_rate=0.05,
    reg_alpha=0.5,
    reg_lambda=5.0,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=0,
    n_jobs=-1
)
xgb_shallow.fit(X_tr_s, y_tr)
r_shallow_vl = pearsonr(y_vl, xgb_shallow.predict(X_vl_s))[0]
r_shallow_te = pearsonr(y_te, xgb_shallow.predict(X_te_s))[0]
print(f"Shallow: Val={r_shallow_vl:.4f}, Test={r_shallow_te:.4f}")

# Strategy 3: Different subsample/colsample
print("\n--- XGB with different sampling ---")
xgb_sample = xgb.XGBRegressor(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    reg_alpha=0.5,
    reg_lambda=5.0,
    subsample=0.6,
    colsample_bytree=0.6,
    random_state=42,
    verbosity=0,
    n_jobs=-1
)
xgb_sample.fit(X_tr_s, y_tr)
r_sample_vl = pearsonr(y_vl, xgb_sample.predict(X_vl_s))[0]
r_sample_te = pearsonr(y_te, xgb_sample.predict(X_te_s))[0]
print(f"Sample: Val={r_sample_vl:.4f}, Test={r_sample_te:.4f}")

# Strategy 4: Higher min_child_weight
print("\n--- XGB with higher min_child_weight ---")
xgb_mcw = xgb.XGBRegressor(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    reg_alpha=0.5,
    reg_lambda=5.0,
    min_child_weight=10,
    random_state=42,
    verbosity=0,
    n_jobs=-1
)
xgb_mcw.fit(X_tr_s, y_tr)
r_mcw_vl = pearsonr(y_vl, xgb_mcw.predict(X_vl_s))[0]
r_mcw_te = pearsonr(y_te, xgb_mcw.predict(X_te_s))[0]
print(f"min_child_weight=10: Val={r_mcw_vl:.4f}, Test={r_mcw_te:.4f}")

# Ensemble all
print("\n--- Ensemble ---")
all_preds_vl = [
    xgb_many.predict(X_vl_s),
    xgb_shallow.predict(X_vl_s),
    xgb_sample.predict(X_vl_s),
    xgb_mcw.predict(X_vl_s),
]
all_preds_te = [
    xgb_many.predict(X_te_s),
    xgb_shallow.predict(X_te_s),
    xgb_sample.predict(X_te_s),
    xgb_mcw.predict(X_te_s),
]

ens_vl = np.mean(all_preds_vl, axis=0)
ens_te = np.mean(all_preds_te, axis=0)

r_ens_vl = pearsonr(y_vl, ens_vl)[0]
r_ens_te = pearsonr(y_te, ens_te)[0]
print(f"Ensemble: Val={r_ens_vl:.4f}, Test={r_ens_te:.4f}")

# 5-Fold CV
print("\n--- 5-Fold CV ---")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_rs = []

for fold, (tr_idx, vl_idx) in enumerate(kf.split(X_tr_s)):
    fold_preds = []
    # Use best performing configs
    configs = [
        {'n_estimators': 500, 'max_depth': 5, 'learning_rate': 0.03, 'reg_alpha': 0.5, 'reg_lambda': 5.0, 'subsample': 0.8, 'colsample_bytree': 0.8},
        {'n_estimators': 400, 'max_depth': 4, 'learning_rate': 0.05, 'reg_alpha': 0.5, 'reg_lambda': 5.0},
        {'n_estimators': 400, 'max_depth': 6, 'learning_rate': 0.05, 'reg_alpha': 0.3, 'reg_lambda': 3.0, 'subsample': 0.7, 'colsample_bytree': 0.7},
    ]
    for cfg in configs:
        m = xgb.XGBRegressor(**cfg, random_state=42, verbosity=0, n_jobs=-1)
        m.fit(X_tr_s[tr_idx], y_tr[tr_idx])
        fold_preds.append(m.predict(X_tr_s[vl_idx]))
    
    fold_ens = np.mean(fold_preds, axis=0)
    cv_rs.append(pearsonr(y_tr[vl_idx], fold_ens)[0])
    print(f"  Fold {fold+1}: {cv_rs[-1]:.4f}")

cv_mean = np.mean(cv_rs)
cv_std = np.std(cv_rs)
print(f"\nCV R: {cv_mean:.4f} ± {cv_std:.4f}")

# Save
with open('WORK_DIR / chunk7_results.pkl', 'wb') as f:
    pickle.dump({
        'cv_r': cv_mean,
        'cv_std': cv_std,
        'val_r': r_ens_vl,
        'test_r': r_ens_te,
    }, f)

print(f"\n✓ Chunk 7: CV R={cv_mean:.4f}")