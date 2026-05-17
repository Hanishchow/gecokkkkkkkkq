#!/usr/bin/env python3
"""
Chunk 6: Try BOTH physics feature sets combined + more ensemble diversity
"""
import pickle
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import warnings
# Import path helpers
try:
    from geock_paths import get_cache_dir, get_work_dir, CACHE_DIR, WORK_DIR
except ImportError:
    from pathlib import Path; CACHE_DIR = Path("/home/chow/.cache/geock_autoresearch"); WORK_DIR = Path("/home/chow/autoresearch")
warnings.filterwarnings('ignore')

print("="*70)
print("CHUNK 6: Combined Physics + RF + Extra Diversity")
print("="*70)

# Load data
cache = CACHE_DIR / lp_new_features_8k.pkl')
with open(cache, 'rb') as f:
    compounds = pickle.load(f)

# Load BOTH physics features
with open('CACHE_DIR / physics_features_8k.pkl', 'rb') as f:
    phys_data = pickle.load(f)
X_phys_20 = phys_data['X_phys']  # 20D

with open('CACHE_DIR / physics_24d.pkl', 'rb') as f:
    phys_24d = pickle.load(f)
X_phys_22 = phys_24d['X_phys']  # 22D

print(f"Physics 20D: {X_phys_20.shape}, Physics 22D: {X_phys_22.shape}")

# Load interaction features
X_int = np.load('WORK_DIR / X_interactions.npy')
with open('WORK_DIR / interaction_pdb_ids.pkl', 'rb') as f:
    int_pdb_ids = pickle.load(f)
int_map = {pdb: i for i, pdb in enumerate(int_pdb_ids)}

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski

# Build features - COMBINE both physics feature sets
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
    
    # Combine both physics + interactions: 20D + 22D + 20D = 62D
    X = np.concatenate([ecfp, mol_feat, X_phys_20[i], X_phys_22[i], int_feat])
    X_list.append(X)
    y_list.append(c['affinity'])

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.float32)
print(f"Features (combined physics): {X.shape}")

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

# Extended configs with more diversity
configs = [
    # XGBoost variants
    {'max_depth': 6, 'learning_rate': 0.1, 'reg_alpha': 0.5, 'reg_lambda': 5.0},
    {'max_depth': 8, 'learning_rate': 0.05, 'reg_alpha': 0.7, 'reg_lambda': 7.0},
    {'max_depth': 7, 'learning_rate': 0.08, 'reg_alpha': 0.5, 'reg_lambda': 5.0},
    {'max_depth': 10, 'learning_rate': 0.03, 'reg_alpha': 1.0, 'reg_lambda': 10.0},
    {'max_depth': 5, 'learning_rate': 0.1, 'reg_alpha': 0.1, 'reg_lambda': 1.0},
    # More varied
    {'max_depth': 4, 'learning_rate': 0.15, 'reg_alpha': 0.2, 'reg_lambda': 2.0},
    {'max_depth': 9, 'learning_rate': 0.04, 'reg_alpha': 0.8, 'reg_lambda': 8.0},
]

print("\n--- Training XGBoost Models ---")
xgb_models = []
xgb_preds_vl = []
xgb_preds_te = []

for i, cfg in enumerate(configs[:5]):
    m = xgb.XGBRegressor(n_estimators=300, subsample=0.8, colsample_bytree=0.8,
                         min_child_weight=3, random_state=42+i, verbosity=0, n_jobs=-1, **cfg)
    m.fit(X_tr_s, y_tr)
    
    p_vl = m.predict(X_vl_s)
    p_te = m.predict(X_te_s)
    
    r_vl = pearsonr(y_vl, p_vl)[0]
    print(f"XGB {i+1}: Val R={r_vl:.4f}")
    
    xgb_models.append(m)
    xgb_preds_vl.append(p_vl)
    xgb_preds_te.append(p_te)

# Add Random Forest for diversity
print("\n--- Adding Random Forest ---")
rf = RandomForestRegressor(n_estimators=150, max_depth=10, min_samples_leaf=3,
                           random_state=42, n_jobs=-1)
rf.fit(X_tr_s, y_tr)
rf_vl = pearsonr(y_vl, rf.predict(X_vl_s))[0]
rf_te = pearsonr(y_te, rf.predict(X_te_s))[0]
print(f"RF: Val R={rf_vl:.4f}, Test R={rf_te:.4f}")

# Combine XGB + RF
all_preds_vl = xgb_preds_vl + [rf.predict(X_vl_s)]
all_preds_te = xgb_preds_te + [rf.predict(X_te_s)]

# Optimize weights
from scipy.optimize import minimize

def opt_w(preds, y):
    n = len(preds)
    def loss(w):
        w = np.abs(w) / np.abs(w).sum()
        return np.mean((sum(wi*pi for wi,pi in zip(w,preds)) - y)**2)
    result = minimize(loss, np.ones(n)/n, method='Nelder-Mead')
    return np.abs(result.x) / np.abs(result.x).sum()

weights = opt_w(all_preds_vl, y_vl)
print(f"\nWeights: {weights.round(3)}")

ens_vl = sum(w*p for w,p in zip(weights, all_preds_vl))
ens_te = sum(w*p for w,p in zip(weights, all_preds_te))

r_ens_vl = pearsonr(y_vl, ens_vl)[0]
r_ens_te = pearsonr(y_te, ens_te)[0]
mae = np.mean(np.abs(y_te - ens_te))
print(f"Ensemble (XGB+RF): Val R={r_ens_vl:.4f}, Test R={r_ens_te:.4f}, MAE={mae:.3f}")

# 5-Fold CV
print("\n--- 5-Fold CV ---")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_rs = []

for fold, (tr_idx, vl_idx) in enumerate(kf.split(X_tr_s)):
    fold_preds = []
    # XGB models
    for cfg in configs[:5]:
        m = xgb.XGBRegressor(n_estimators=200, subsample=0.8, colsample_bytree=0.8,
                             random_state=42, verbosity=0, n_jobs=-1, **cfg)
        m.fit(X_tr_s[tr_idx], y_tr[tr_idx])
        fold_preds.append(m.predict(X_tr_s[vl_idx]))
    # RF
    rf = RandomForestRegressor(n_estimators=150, max_depth=10, min_samples_leaf=3,
                               random_state=42, n_jobs=-1)
    rf.fit(X_tr_s[tr_idx], y_tr[tr_idx])
    fold_preds.append(rf.predict(X_tr_s[vl_idx]))
    
    fold_ens = sum(w*p for w,p in zip(weights, fold_preds))
    cv_rs.append(pearsonr(y_tr[vl_idx], fold_ens)[0])
    print(f"  Fold {fold+1}: {cv_rs[-1]:.4f}")

cv_mean = np.mean(cv_rs)
cv_std = np.std(cv_rs)
print(f"\nCV R: {cv_mean:.4f} ± {cv_std:.4f}")

# Save
chunk6_results = {
    'val_r': r_ens_vl,
    'test_r': r_ens_te,
    'cv_r': cv_mean,
    'cv_std': cv_std,
    'mae': mae,
    'weights': weights,
}

with open('WORK_DIR / chunk6_results.pkl', 'wb') as f:
    pickle.dump(chunk6_results, f)

print(f"\n✓ Chunk 6: CV R={cv_mean:.4f}, Val={r_ens_vl:.4f}, Test={r_ens_te:.4f}")