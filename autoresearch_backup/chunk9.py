#!/usr/bin/env python3
"""
Chunk 9: Try different random seeds + more iterations + cross-validation ensemble
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
print("CHUNK 9: Multi-Seed Ensemble")
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

# Standardize
mu, sd = X.mean(0), X.std(0)
sd[sd == 0] = 1
X_s = (X - mu) / sd

# Use same split as before
np.random.seed(42)
n = len(X)
perm = np.random.permutation(n)
n_test = int(n * 0.1)
n_val = int(n * 0.1)
n_train = n - n_test - n_val

idx_tr = perm[:n_train]
idx_vl = perm[n_train:n_train+n_val]
idx_te = perm[n_train+n_val:]

X_tr_s = X_s[idx_tr]
X_vl_s = X_s[idx_vl]
X_te_s = X_s[idx_te]
y_tr, y_vl, y_te = y[idx_tr], y[idx_vl], y[idx_te]

print(f"Split: {n_train}/{n_val}/{n_test}")

# Train with multiple random seeds
print("\n--- Multi-seed XGBoost ---")
seeds = [42, 123, 456, 789, 1000]
all_preds_vl = []
all_preds_te = []

base_cfg = {
    'n_estimators': 400,
    'max_depth': 8,
    'learning_rate': 0.03,
    'reg_alpha': 0.7,
    'reg_lambda': 7.0,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'verbosity': 0,
    'n_jobs': -1,
}

for seed in seeds:
    cfg = base_cfg.copy()
    cfg['random_state'] = seed
    m = xgb.XGBRegressor(**cfg)
    m.fit(X_tr_s, y_tr)
    
    r_vl = pearsonr(y_vl, m.predict(X_vl_s))[0]
    print(f"Seed {seed}: Val R={r_vl:.4f}")
    
    all_preds_vl.append(m.predict(X_vl_s))
    all_preds_te.append(m.predict(X_te_s))

# Ensemble multi-seed
ens_vl = np.mean(all_preds_vl, axis=0)
ens_te = np.mean(all_preds_te, axis=0)

r_ens_vl = pearsonr(y_vl, ens_vl)[0]
r_ens_te = pearsonr(y_te, ens_te)[0]
print(f"\nMulti-seed ensemble: Val R={r_ens_vl:.4f}, Test R={r_ens_te:.4f}")

# Now do proper CV with seed ensemble
print("\n--- CV with seed ensemble ---")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_rs = []

for fold, (tr_idx, vl_idx) in enumerate(kf.split(X_s)):
    fold_preds = []
    for seed in seeds:
        cfg = base_cfg.copy()
        cfg['random_state'] = seed
        m = xgb.XGBRegressor(**cfg)
        m.fit(X_s[tr_idx], y[tr_idx])
        fold_preds.append(m.predict(X_s[vl_idx]))
    
    fold_ens = np.mean(fold_preds, axis=0)
    cv_rs.append(pearsonr(y[vl_idx], fold_ens)[0])
    print(f"  Fold {fold+1}: {cv_rs[-1]:.4f}")

cv_mean = np.mean(cv_rs)
cv_std = np.std(cv_rs)
print(f"\nCV R: {cv_mean:.4f} ± {cv_std:.4f}")

# Also try mixing best configs with different seeds
print("\n--- Config + seed matrix ---")
configs = [
    {'max_depth': 8, 'learning_rate': 0.03, 'reg_alpha': 0.7, 'reg_lambda': 7.0},
    {'max_depth': 10, 'learning_rate': 0.03, 'reg_alpha': 1.0, 'reg_lambda': 10.0},
    {'max_depth': 7, 'learning_rate': 0.05, 'reg_alpha': 0.5, 'reg_lambda': 5.0},
]

matrix_preds_vl = []
matrix_preds_te = []

for cfg in configs:
    for seed in [42, 123]:
        m = xgb.XGBRegressor(n_estimators=300, subsample=0.8, colsample_bytree=0.8,
                             min_child_weight=3, random_state=seed, verbosity=0, 
                             n_jobs=-1, **cfg)
        m.fit(X_tr_s, y_tr)
        
        matrix_preds_vl.append(m.predict(X_vl_s))
        matrix_preds_te.append(m.predict(X_te_s))

matrix_ens_vl = np.mean(matrix_preds_vl, axis=0)
matrix_ens_te = np.mean(matrix_preds_te, axis=0)

r_matrix_vl = pearsonr(y_vl, matrix_ens_vl)[0]
r_matrix_te = pearsonr(y_te, matrix_ens_te)[0]
print(f"Config-seed matrix: Val R={r_matrix_vl:.4f}, Test R={r_matrix_te:.4f}")

# Pick best
if r_matrix_vl > r_ens_vl:
    final_vl, final_te = matrix_ens_vl, matrix_ens_te
    final_r = r_matrix_vl
    final_te_r = r_matrix_te
    best_method = "Matrix"
else:
    final_vl, final_te = ens_vl, ens_te
    final_r = r_ens_vl
    final_te_r = r_ens_te
    best_method = "Multi-seed"

print(f"\nBest: {best_method} with Val R={final_r:.4f}")

# Save
with open('WORK_DIR / chunk9_results.pkl', 'wb') as f:
    pickle.dump({
        'cv_r': cv_mean,
        'cv_std': cv_std,
        'val_r': final_r,
        'test_r': final_te_r,
        'method': best_method,
    }, f)

print(f"\n✓ Chunk 9: CV R={cv_mean:.4f}")