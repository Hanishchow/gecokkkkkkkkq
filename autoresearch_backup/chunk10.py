#!/usr/bin/env python3
"""
Chunk 10: Final optimization - more seeds + best configuration
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
print("CHUNK 10: Final Optimization")
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

# More seeds with tuned params
print("\n--- Extended multi-seed ensemble ---")
seeds = [42, 123, 456, 789, 1000, 2024, 3333, 5555]

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

all_preds = []
for seed in seeds:
    cfg = base_cfg.copy()
    cfg['random_state'] = seed
    m = xgb.XGBRegressor(**cfg)
    m.fit(X_tr_s, y_tr)
    all_preds.append(m.predict(X_vl_s))

ens_vl = np.mean(all_preds, axis=0)
r_ens = pearsonr(y_vl, ens_vl)[0]
print(f"8-seed ensemble: Val R={r_ens:.4f}")

# Test predictions
all_test = []
for seed in seeds:
    cfg = base_cfg.copy()
    cfg['random_state'] = seed
    m = xgb.XGBRegressor(**cfg)
    m.fit(X_tr_s, y_tr)
    all_test.append(m.predict(X_te_s))

ens_te = np.mean(all_test, axis=0)
r_test = pearsonr(y_te, ens_te)[0]
mae = np.mean(np.abs(y_te - ens_te))
print(f"Test R={r_test:.4f}, MAE={mae:.3f}")

# Proper CV
print("\n--- 5-Fold CV ---")
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

# Also try a few more tuning variations
print("\n--- Trying variations ---")
variations = [
    {'max_depth': 9, 'learning_rate': 0.025},
    {'max_depth': 7, 'learning_rate': 0.04},
    {'reg_alpha': 1.0, 'reg_lambda': 12.0},
]

best_r = r_ens
best_var = None

for var in variations:
    cfg = base_cfg.copy()
    cfg.update(var)
    
    m = xgb.XGBRegressor(**cfg)
    m.fit(X_tr_s, y_tr)
    r = pearsonr(y_vl, m.predict(X_vl_s))[0]
    print(f"  {var}: Val R={r:.4f}")
    
    if r > best_r:
        best_r = r
        best_var = var

# Final model with best config
if best_var:
    final_cfg = base_cfg.copy()
    final_cfg.update(best_var)
else:
    final_cfg = base_cfg

print(f"\nUsing config: {final_cfg}")

# Save final model
final_model = xgb.XGBRegressor(**final_cfg)
final_model.fit(X_tr_s, y_tr)

output = {
    'model': final_model,
    'mu': mu,
    'sd': sd,
    'cv_r': cv_mean,
    'cv_std': cv_std,
    'val_r': best_r if best_var else r_ens,
    'test_r': r_test,
    'mae': mae,
    'seeds': seeds,
    'config': final_cfg,
}

with open('WORK_DIR / geock_final_best.pkl', 'wb') as f:
    pickle.dump(output, f)

print(f"\n✓ FINAL: CV R={cv_mean:.4f} ± {cv_std:.4f}")
print(f"  Val R={best_r if best_var else r_ens:.4f}, Test R={r_test:.4f}")