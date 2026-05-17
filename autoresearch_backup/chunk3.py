#!/usr/bin/env python3
"""
Chunk 3: Neural Network, Ridge, and focus on what works
"""
import pickle
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
# Import path helpers
try:
    from geock_paths import get_cache_dir, get_work_dir, CACHE_DIR, WORK_DIR
except ImportError:
    from pathlib import Path; CACHE_DIR = Path("/home/chow/.cache/geock_autoresearch"); WORK_DIR = Path("/home/chow/autoresearch")
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("CHUNK 3: NN, Ridge, Focus on ECFP + physics only")
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

# Build multiple feature sets
X_ecfp_list, X_mol_list, X_full_list = [], [], []
y_list = []

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
    
    X_ecfp_list.append(ecfp)
    X_mol_list.append(np.concatenate([mol_feat, X_phys[i], int_feat]))
    X_full_list.append(np.concatenate([ecfp, mol_feat, X_phys[i], int_feat]))
    y_list.append(c['affinity'])

X_ecfp = np.array(X_ecfp_list, dtype=np.float32)
X_mol = np.array(X_mol_list, dtype=np.float32)
X_full = np.array(X_full_list, dtype=np.float32)
y = np.array(y_list, dtype=np.float32)

print(f"ECFP: {X_ecfp.shape}, Mol+Phys+Int: {X_mol.shape}, Full: {X_full.shape}")

# Split
np.random.seed(42)
n = len(X_full)
perm = np.random.permutation(n)
n_test = int(n * 0.1)
n_val = int(n * 0.1)
n_train = n - n_test - n_val

idx_tr = perm[:n_train]
idx_vl = perm[n_train:n_train+n_val]
idx_te = perm[n_train+n_val:]

# Standardize each feature set
scaler_ecfp = StandardScaler()
scaler_mol = StandardScaler()
scaler_full = StandardScaler()

X_tr_ecfp = scaler_ecfp.fit_transform(X_ecfp[idx_tr])
X_vl_ecfp = scaler_ecfp.transform(X_ecfp[idx_vl])
X_te_ecfp = scaler_ecfp.transform(X_ecfp[idx_te])

X_tr_mol = scaler_mol.fit_transform(X_mol[idx_tr])
X_vl_mol = scaler_mol.transform(X_mol[idx_vl])
X_te_mol = scaler_mol.transform(X_mol[idx_te])

X_tr_full = scaler_full.fit_transform(X_full[idx_tr])
X_vl_full = scaler_full.transform(X_full[idx_vl])
X_te_full = scaler_full.transform(X_full[idx_te])

y_tr, y_vl, y_te = y[idx_tr], y[idx_vl], y[idx_te]

print(f"Split: {n_train}/{n_val}/{n_test}")

# Test 1: Ridge on Mol+Phys+Int only (like traditional ML)
print("\n--- Ridge on Mol+Phys+Int ---")
for alpha in [0.1, 1.0, 10.0, 100.0]:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_tr_mol, y_tr)
    r_vl = pearsonr(y_vl, ridge.predict(X_vl_mol))[0]
    r_te = pearsonr(y_te, ridge.predict(X_te_mol))[0]
    print(f"  alpha={alpha}: Val={r_vl:.4f}, Test={r_te:.4f}")

# Test 2: MLP on full
print("\n--- MLP on Full ---")
mlp = MLPRegressor(hidden_layer_sizes=(256, 128, 64), max_iter=200, 
                   early_stopping=True, random_state=42)
mlp.fit(X_tr_full, y_tr)
mlp_vl = pearsonr(y_vl, mlp.predict(X_vl_full))[0]
mlp_te = pearsonr(y_te, mlp.predict(X_te_full))[0]
print(f"MLP: Val={mlp_vl:.4f}, Test={mlp_te:.4f}")

# Test 3: XGBoost on ECFP only (baseline)
print("\n--- XGBoost on ECFP only ---")
xgb_ecfp = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1,
                            reg_alpha=0.1, reg_lambda=1.0, random_state=42, 
                            verbosity=0, n_jobs=-1)
xgb_ecfp.fit(X_tr_ecfp, y_tr)
xgb_ecfp_vl = pearsonr(y_vl, xgb_ecfp.predict(X_vl_ecfp))[0]
xgb_ecfp_te = pearsonr(y_te, xgb_ecfp.predict(X_te_ecfp))[0]
print(f"XGB ECFP: Val={xgb_ecfp_vl:.4f}, Test={xgb_ecfp_te:.4f}")

# Test 4: XGBoost on Mol+Phys+Int (no ECFP)
print("\n--- XGBoost on Mol+Phys+Int ---")
xgb_mol = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1,
                           reg_alpha=0.5, reg_lambda=5.0, random_state=42,
                           verbosity=0, n_jobs=-1)
xgb_mol.fit(X_tr_mol, y_tr)
xgb_mol_vl = pearsonr(y_vl, xgb_mol.predict(X_vl_mol))[0]
xgb_mol_te = pearsonr(y_te, xgb_mol.predict(X_te_mol))[0]
print(f"XGB Mol: Val={xgb_mol_vl:.4f}, Test={xgb_mol_te:.4f}")

# Test 5: XGBoost on full
print("\n--- XGBoost on Full ---")
xgb_full = xgb.XGBRegressor(n_estimators=300, max_depth=7, learning_rate=0.05,
                            reg_alpha=0.5, reg_lambda=5.0, random_state=42,
                            verbosity=0, n_jobs=-1)
xgb_full.fit(X_tr_full, y_tr)
xgb_full_vl = pearsonr(y_vl, xgb_full.predict(X_vl_full))[0]
xgb_full_te = pearsonr(y_te, xgb_full.predict(X_te_full))[0]
print(f"XGB Full: Val={xgb_full_vl:.4f}, Test={xgb_full_te:.4f}")

# Ensemble: Best combinations
print("\n--- Hybrid Ensemble ---")
# Combine: XGB on ECFP + XGB on Mol
preds_vl = [xgb_ecfp.predict(X_vl_ecfp), xgb_mol.predict(X_vl_mol), xgb_full.predict(X_vl_full)]
preds_te = [xgb_ecfp.predict(X_te_ecfp), xgb_mol.predict(X_te_mol), xgb_full.predict(X_te_full)]

# Weighted ensemble
from scipy.optimize import minimize

def opt_w(preds, y):
    n = len(preds)
    def loss(w):
        w = np.abs(w) / np.abs(w).sum()
        return np.mean((sum(wi*pi for wi,pi in zip(w,preds)) - y)**2)
    result = minimize(loss, np.ones(n)/n, method='Nelder-Mead')
    return np.abs(result.x) / np.abs(result.x).sum()

weights = opt_w(preds_vl, y_vl)
print(f"Weights: {weights.round(3)}")

ens_vl = sum(w*p for w,p in zip(weights, preds_vl))
ens_te = sum(w*p for w,p in zip(weights, preds_te))

r_ens_vl = pearsonr(y_vl, ens_vl)[0]
r_ens_te = pearsonr(y_te, ens_te)[0]
print(f"Hybrid Ensemble: Val={r_ens_vl:.4f}, Test={r_ens_te:.4f}")

# 5-fold CV
print("\n--- 5-Fold CV ---")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_rs = []

# Use XGB Full for CV
xgb_params = {'n_estimators': 300, 'max_depth': 7, 'learning_rate': 0.05,
              'reg_alpha': 0.5, 'reg_lambda': 5.0, 'random_state': 42, 
              'verbosity': 0, 'n_jobs': -1}

for fold, (tr_idx, vl_idx) in enumerate(kf.split(X_tr_full)):
    m = xgb.XGBRegressor(**xgb_params)
    m.fit(X_tr_full[tr_idx], y_tr[tr_idx])
    cv_rs.append(pearsonr(y_tr[vl_idx], m.predict(X_tr_full[vl_idx]))[0])
    print(f"  Fold {fold+1}: {cv_rs[-1]:.4f}")

cv_mean = np.mean(cv_rs)
cv_std = np.std(cv_rs)
print(f"\nXGB Full CV: {cv_mean:.4f} ± {cv_std:.4f}")

# Save results
chunk3_results = {
    'xgb_ecfp': {'val_r': xgb_ecfp_vl, 'test_r': xgb_ecfp_te},
    'xgb_mol': {'val_r': xgb_mol_vl, 'test_r': xgb_mol_te},
    'xgb_full': {'val_r': xgb_full_vl, 'test_r': xgb_full_te},
    'hybrid': {'val_r': r_ens_vl, 'test_r': r_ens_te},
    'cv_r': cv_mean,
    'cv_std': cv_std,
}

with open('WORK_DIR / chunk3_results.pkl', 'wb') as f:
    pickle.dump(chunk3_results, f)

print("\n✓ Chunk 3 complete")