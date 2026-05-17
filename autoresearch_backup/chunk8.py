#!/usr/bin/env python3
"""
Chunk 8: Final push - use the best configurations found + try blending
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
print("CHUNK 8: Best Configs Blend + Final Push")
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

# Best configs from chunk 4
configs = [
    {'max_depth': 8, 'learning_rate': 0.03, 'reg_alpha': 0.7, 'reg_lambda': 7.0, 'n_estimators': 500},
    {'max_depth': 10, 'learning_rate': 0.03, 'reg_alpha': 1.0, 'reg_lambda': 10.0, 'n_estimators': 500},
    {'max_depth': 7, 'learning_rate': 0.05, 'reg_alpha': 0.5, 'reg_lambda': 5.0, 'n_estimators': 400},
    {'max_depth': 6, 'learning_rate': 0.1, 'reg_alpha': 0.5, 'reg_lambda': 5.0, 'n_estimators': 300},
    {'max_depth': 9, 'learning_rate': 0.04, 'reg_alpha': 0.8, 'reg_lambda': 8.0, 'n_estimators': 400},
]

print("\n--- Training best configs ---")
models = []
preds_vl = []
preds_te = []

for i, cfg in enumerate(configs):
    m = xgb.XGBRegressor(
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        random_state=42+i,
        verbosity=0,
        n_jobs=-1,
        **cfg
    )
    m.fit(X_tr_s, y_tr)
    
    p_vl = m.predict(X_vl_s)
    p_te = m.predict(X_te_s)
    
    r_vl = pearsonr(y_vl, p_vl)[0]
    print(f"Config {i+1}: Val R={r_vl:.4f}")
    
    models.append(m)
    preds_vl.append(p_vl)
    preds_te.append(p_te)

# Try different blending strategies
print("\n--- Blending strategies ---")

# Simple average
avg_vl = np.mean(preds_vl, axis=0)
avg_te = np.mean(preds_te, axis=0)
r_avg = pearsonr(y_vl, avg_vl)[0]
print(f"Simple avg: Val R={r_avg:.4f}")

# Weighted by validation performance
weights = np.array([pearsonr(y_vl, p)[0] for p in preds_vl])
weights = weights / weights.sum()
weighted_vl = sum(w*p for w,p in zip(weights, preds_vl))
weighted_te = sum(w*p for w,p in zip(weights, preds_te))
r_weighted = pearsonr(y_vl, weighted_vl)[0]
print(f"Weighted avg: Val R={r_weighted:.4f}")

# Use scipy optimize
from scipy.optimize import minimize
def opt_w(preds, y):
    n = len(preds)
    def loss(w):
        w = np.abs(w) / np.abs(w).sum()
        return np.mean((sum(wi*pi for wi,pi in zip(w,preds)) - y)**2)
    result = minimize(loss, np.ones(n)/n, method='Nelder-Mead')
    return np.abs(result.x) / np.abs(result.x).sum()

opt_weights = opt_w(preds_vl, y_vl)
opt_vl = sum(w*p for w,p in zip(opt_weights, preds_vl))
opt_te = sum(w*p for w,p in zip(opt_weights, preds_te))
r_opt = pearsonr(y_vl, opt_vl)[0]
print(f"Optimized: Val R={r_opt:.4f}")

# Use best 3 only
best3_idx = np.argsort([pearsonr(y_vl, p)[0] for p in preds_vl])[-3:]
best3_vl = np.mean([preds_vl[i] for i in best3_idx], axis=0)
best3_te = np.mean([preds_te[i] for i in best3_idx], axis=0)
r_best3 = pearsonr(y_vl, best3_vl)[0]
print(f"Best 3 avg: Val R={r_best3:.4f}")

# Pick best strategy
best_r = max(r_avg, r_weighted, r_opt, r_best3)
if best_r == r_opt:
    final_vl, final_te = opt_vl, opt_te
    best_name = "Optimized"
elif best_r == r_weighted:
    final_vl, final_te = weighted_vl, weighted_te
    best_name = "Weighted"
elif best_r == r_best3:
    final_vl, final_te = best3_vl, best3_te
    best_name = "Best3"
else:
    final_vl, final_te = avg_vl, avg_te
    best_name = "Simple"

r_final_te = pearsonr(y_te, final_te)[0]
mae = np.mean(np.abs(y_te - final_te))
print(f"\nBest strategy: {best_name}, Test R={r_final_te:.4f}")

# 5-Fold CV
print("\n--- 5-Fold CV ---")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_rs = []

for fold, (tr_idx, vl_idx) in enumerate(kf.split(X_tr_s)):
    fold_preds = []
    for cfg in configs:
        m = xgb.XGBRegressor(subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                             random_state=42, verbosity=0, n_jobs=-1, **cfg)
        m.fit(X_tr_s[tr_idx], y_tr[tr_idx])
        fold_preds.append(m.predict(X_tr_s[vl_idx]))
    
    fold_ens = np.mean(fold_preds, axis=0)
    cv_rs.append(pearsonr(y_tr[vl_idx], fold_ens)[0])
    print(f"  Fold {fold+1}: {cv_rs[-1]:.4f}")

cv_mean = np.mean(cv_rs)
cv_std = np.std(cv_rs)
print(f"\nCV R: {cv_mean:.4f} ± {cv_std:.4f}")

# Save best model
best_model = models[np.argmax([pearsonr(y_vl, p)[0] for p in preds_vl])]
output = {
    'model': best_model,
    'mu': mu,
    'sd': sd,
    'cv_r': cv_mean,
    'cv_std': cv_std,
    'val_r': best_r,
    'test_r': r_final_te,
    'mae': mae,
    'strategy': best_name,
}

with open('WORK_DIR / geock_best_final.pkl', 'wb') as f:
    pickle.dump(output, f)

print(f"\n✓ Saved best model: CV R={cv_mean:.4f}")