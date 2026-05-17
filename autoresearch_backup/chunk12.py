#!/usr/bin/env python3
"""
Continue training - Multi-seed + different feature combinations
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
print("CHUNK 12: Extended Multi-Seed + Feature Variations")
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

# Build features
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

mu, sd = X.mean(0), X.std(0)
sd[sd == 0] = 1
X_s = (X - mu) / sd

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

# Extended seeds
print("\n--- Extended multi-seed ensemble ---")
seeds = list(range(20, 70, 5))  # 10 seeds: 20, 25, 30, ..., 65

best_cfg = {
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

all_preds_vl = []
all_preds_te = []

for seed in seeds:
    cfg = best_cfg.copy()
    cfg['random_state'] = seed
    m = xgb.XGBRegressor(**cfg)
    m.fit(X_tr_s, y_tr)
    all_preds_vl.append(m.predict(X_vl_s))
    all_preds_te.append(m.predict(X_te_s))

ens_vl = np.mean(all_preds_vl, axis=0)
ens_te = np.mean(all_preds_te, axis=0)

r_ens_vl = pearsonr(y_vl, ens_vl)[0]
r_ens_te = pearsonr(y_te, ens_te)[0]
print(f"10-seed ensemble: Val R={r_ens_vl:.4f}, Test R={r_ens_te:.4f}")

# Try different config variations
print("\n--- Config variations ---")
configs = [
    {'max_depth': 9, 'learning_rate': 0.025, 'reg_alpha': 0.8, 'reg_lambda': 8.0},
    {'max_depth': 7, 'learning_rate': 0.04, 'reg_alpha': 0.6, 'reg_lambda': 6.0},
    {'max_depth': 8, 'learning_rate': 0.035, 'reg_alpha': 0.5, 'reg_lambda': 5.0},
]

all_var_preds = []
for cfg in configs:
    cfg_full = best_cfg.copy()
    cfg_full.update(cfg)
    preds = []
    for seed in [42, 123, 456]:
        cfg_full['random_state'] = seed
        m = xgb.XGBRegressor(**cfg_full)
        m.fit(X_tr_s, y_tr)
        preds.append(m.predict(X_vl_s))
    all_var_preds.append(np.mean(preds, axis=0))
    r = pearsonr(y_vl, all_var_preds[-1])[0]
    print(f"  {cfg}: Val R={r:.4f}")

# Combine best ensemble + variations
combined_vl = np.mean(all_var_preds + [ens_vl], axis=0)
combined_te = np.mean([np.mean([xgb.XGBRegressor(**{**best_cfg, 'random_state': s}).fit(X_tr_s, y_tr).predict(X_te_s) for s in [42,123,456]], axis=0) for _ in configs], axis=0)

r_comb = pearsonr(y_vl, combined_vl)[0]
print(f"Combined: Val R={r_comb:.4f}")

# CV
print("\n--- 5-Fold CV ---")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_rs = []

for fold, (tr_idx, vl_idx) in enumerate(kf.split(X_s)):
    fold_preds = []
    for seed in seeds[:5]:  # Use 5 seeds for CV
        cfg = best_cfg.copy()
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

# Save
with open('WORK_DIR / chunk12_results.pkl', 'wb') as f:
    pickle.dump({
        'cv_r': cv_mean, 
        'cv_std': cv_std, 
        'val_r': r_ens_vl, 
        'test_r': r_ens_te
    }, f)

print(f"\n✓ Chunk 12: CV R={cv_mean:.4f}")