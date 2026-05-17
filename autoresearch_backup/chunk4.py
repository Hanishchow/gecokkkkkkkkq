#!/usr/bin/env python3
"""
Chunk 4: Extensive hyperparameter tuning + stacking
"""
import pickle
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
import xgboost as xgb
import warnings
# Import path helpers
try:
    from geock_paths import get_cache_dir, get_work_dir, CACHE_DIR, WORK_DIR
except ImportError:
    from pathlib import Path; CACHE_DIR = Path("/home/chow/.cache/geock_autoresearch"); WORK_DIR = Path("/home/chow/autoresearch")
warnings.filterwarnings('ignore')

print("="*70)
print("CHUNK 4: Hyperparameter Grid + Stacking")
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

# Grid search on key parameters
print("\n--- Grid Search ---")
best_r = 0
best_params = None
best_model = None

configs = [
    # depth, lr, alpha, lambda, n_est
    (5, 0.1, 0.1, 1.0, 200),
    (5, 0.1, 0.5, 5.0, 200),
    (6, 0.1, 0.1, 1.0, 200),
    (6, 0.1, 0.5, 5.0, 200),
    (6, 0.05, 0.5, 5.0, 300),
    (7, 0.08, 0.3, 3.0, 250),
    (7, 0.05, 0.5, 5.0, 400),
    (8, 0.03, 0.7, 7.0, 500),
    (8, 0.05, 0.5, 5.0, 400),
    (10, 0.03, 1.0, 10.0, 500),
]

for depth, lr, alpha, lam, n_est in configs:
    m = xgb.XGBRegressor(
        n_estimators=n_est,
        max_depth=depth,
        learning_rate=lr,
        reg_alpha=alpha,
        reg_lambda=lam,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        random_state=42,
        verbosity=0,
        n_jobs=-1
    )
    m.fit(X_tr_s, y_tr)
    r = pearsonr(y_vl, m.predict(X_vl_s))[0]
    r_te = pearsonr(y_te, m.predict(X_te_s))[0]
    print(f"d={depth}, lr={lr}, a={alpha}, l={lam}, n={n_est} -> Val={r:.4f}, Test={r_te:.4f}")
    
    if r > best_r:
        best_r = r
        best_params = (depth, lr, alpha, lam, n_est)
        best_model = m

print(f"\nBest: {best_params} with Val R={best_r:.4f}")

# Stacking: Use multiple models as features for meta-learner
print("\n--- Stacking ---")

# Train base models
base_models = []
for i, (depth, lr, alpha, lam, n_est) in enumerate(configs[:5]):
    m = xgb.XGBRegressor(
        n_estimators=n_est,
        max_depth=depth,
        learning_rate=lr,
        reg_alpha=alpha,
        reg_lambda=lam,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        random_state=42+i,
        verbosity=0,
        n_jobs=-1
    )
    m.fit(X_tr_s, y_tr)
    base_models.append(m)
    print(f"Base model {i+1}: Val R={pearsonr(y_vl, m.predict(X_vl_s))[0]:.4f}")

# Get OOF predictions for stacking
print("\nGenerating OOF predictions...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros((n_train, len(base_models)))

for fold, (tr_idx, vl_idx) in enumerate(kf.split(X_tr_s)):
    for i, (depth, lr, alpha, lam, n_est) in enumerate(configs[:5]):
        m = xgb.XGBRegressor(
            n_estimators=n_est,
            max_depth=depth,
            learning_rate=lr,
            reg_alpha=alpha,
            reg_lambda=lam,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            random_state=42+i,
            verbosity=0,
            n_jobs=-1
        )
        m.fit(X_tr_s[tr_idx], y_tr[tr_idx])
        oof_preds[vl_idx, i] = m.predict(X_tr_s[vl_idx])

# Meta-learner (Ridge)
meta = Ridge(alpha=1.0)
meta.fit(oof_preds, y_tr)

# Predict on validation
meta_vl = meta.predict(np.column_stack([m.predict(X_vl_s) for m in base_models]))
r_meta_vl = pearsonr(y_vl, meta_vl)[0]
print(f"Meta-learner: Val R={r_meta_vl:.4f}")

# Also try simple average of best models
print("\n--- Best Model Ensemble ---")
# Use top 3 from grid
top3 = sorted(enumerate(configs[:5]), key=lambda x: 
              pearsonr(y_vl, base_models[configs.index(x[1])].predict(X_vl_s))[0], reverse=True)[:3]

ens_vl = np.mean([base_models[i].predict(X_vl_s) for i,_ in top3], axis=0)
ens_te = np.mean([base_models[i].predict(X_te_s) for i,_ in top3], axis=0)

r_ens_vl = pearsonr(y_vl, ens_vl)[0]
r_ens_te = pearsonr(y_te, ens_te)[0]
print(f"Top3 Ensemble: Val R={r_ens_vl:.4f}, Test R={r_ens_te:.4f}")

# Full CV on best single model + ensemble
print("\n--- 5-Fold CV ---")
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# CV on best single model
cv_rs = []
best_cfg = best_params
for fold, (tr_idx, vl_idx) in enumerate(kf.split(X_tr_s)):
    m = xgb.XGBRegressor(
        n_estimators=best_cfg[4],
        max_depth=best_cfg[0],
        learning_rate=best_cfg[1],
        reg_alpha=best_cfg[2],
        reg_lambda=best_cfg[3],
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        random_state=42,
        verbosity=0,
        n_jobs=-1
    )
    m.fit(X_tr_s[tr_idx], y_tr[tr_idx])
    cv_rs.append(pearsonr(y_tr[vl_idx], m.predict(X_tr_s[vl_idx]))[0])
    print(f"  Fold {fold+1}: {cv_rs[-1]:.4f}")

cv_mean = np.mean(cv_rs)
cv_std = np.std(cv_rs)
print(f"\nBest Single CV: {cv_mean:.4f} ± {cv_std:.4f}")

# Save
chunk4_results = {
    'best_params': best_params,
    'best_val_r': best_r,
    'best_test_r': pearsonr(y_te, best_model.predict(X_te_s))[0],
    'meta_val_r': r_meta_vl,
    'ens_val_r': r_ens_vl,
    'ens_test_r': r_ens_te,
    'cv_r': cv_mean,
    'cv_std': cv_std,
}

with open('WORK_DIR / chunk4_results.pkl', 'wb') as f:
    pickle.dump(chunk4_results, f)

print("\n✓ Chunk 4 complete")