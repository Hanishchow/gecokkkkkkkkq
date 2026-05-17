#!/usr/bin/env python3
"""
Chunk 2: CatBoost, Random Forest, and feature engineering
"""
import pickle
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
import warnings
# Import path helpers
try:
    from geock_paths import get_cache_dir, get_work_dir, CACHE_DIR, WORK_DIR
except ImportError:
    from pathlib import Path; CACHE_DIR = Path("/home/chow/.cache/geock_autoresearch"); WORK_DIR = Path("/home/chow/autoresearch")
warnings.filterwarnings('ignore')

print("="*70)
print("CHUNK 2: RF, GBM, CatBoost, Feature Engineering")
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
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors

# Build features WITH additional engineered features
X_list, y_list = [], []

for i, c in enumerate(compounds):
    ecfp = np.array(c['ecfp'], dtype=np.float32)
    mol = Chem.MolFromSmiles(c['smiles'])
    if mol is None:
        continue
    
    pdb_id = c['pdb_id']
    int_feat = X_int[int_map[pdb_id]] if pdb_id in int_map else np.zeros(20, dtype=np.float32)
    
    # Basic mol features
    rings = Lipinski.RingCount(mol)
    aromatic = Lipinski.NumAromaticRings(mol)
    logp = Descriptors.MolLogP(mol)
    mw = Descriptors.MolWt(mol)
    bitcount = ecfp.sum()
    hba = Lipinski.NumHAcceptors(mol)
    hbd = Lipinski.NumHDonors(mol)
    rot = Lipinski.NumRotatableBonds(mol)
    
    # Additional features
    try:
        tpsa = Descriptors.TPSA(mol)
    except:
        tpsa = 0
    try:
        frac_sp3 = Descriptors.FractionCSP3(mol)
    except:
        frac_sp3 = 0
    try:
        hetero = Descriptors.NumHeteroatoms(mol)
    except:
        hetero = 0
    try:
        heavy = Descriptors.NumHeavyAtoms(mol)
    except:
        heavy = 0
    try:
        num_rot = rdMolDescriptors.CalcNumRotatableBonds(mol)
    except:
        num_rot = rot
    try:
        num_aliphatic = rdMolDescriptors.CalcNumAliphaticRings(mol)
    except:
        num_aliphatic = 0
    try:
        num_saturated = rdMolDescriptors.CalcNumSaturatedRings(mol)
    except:
        num_saturated = 0
    try:
        labuteASA = Descriptors.LabuteASA(mol)
    except:
        labuteASA = 0
    try:
        peoe = Descriptors.PEOE_VSA1(mol)  # Just get one for now
    except:
        peoe = 0
    
    mol_feat = np.array([rings, aromatic, logp, mw, bitcount, hba, hbd, rot,
                         tpsa, frac_sp3, hetero, heavy, num_rot, num_aliphatic,
                         num_saturated, labuteASA], dtype=np.float32)
    
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

# Standardize
mu, sd = X_tr.mean(0), X_tr.std(0)
sd[sd == 0] = 1
X_tr_s = (X_tr - mu) / sd
X_vl_s = (X_vl - mu) / sd
X_te_s = (X_te - mu) / sd

print(f"Split: {n_train}/{n_val}/{n_test}")

# Test CatBoost
try:
    from catboost import CatBoostRegressor
    HAS_CB = True
    print("\n--- CatBoost ---")
    cb = CatBoostRegressor(iterations=300, depth=6, learning_rate=0.1,
                           l2_leaf_reg=5.0, random_seed=42, verbose=0)
    cb.fit(X_tr_s, y_tr)
    cb_vl = pearsonr(y_vl, cb.predict(X_vl_s))[0]
    cb_te = pearsonr(y_te, cb.predict(X_te_s))[0]
    print(f"CatBoost: Val={cb_vl:.4f}, Test={cb_te:.4f}")
except:
    HAS_CB = False
    print("CatBoost not available")

# Random Forest
print("\n--- Random Forest ---")
rf = RandomForestRegressor(n_estimators=200, max_depth=12, min_samples_leaf=3,
                           random_state=42, n_jobs=-1)
rf.fit(X_tr_s, y_tr)
rf_vl = pearsonr(y_vl, rf.predict(X_vl_s))[0]
rf_te = pearsonr(y_te, rf.predict(X_te_s))[0]
print(f"RF: Val={rf_vl:.4f}, Test={rf_te:.4f}")

# Gradient Boosting
print("\n--- GradientBoosting ---")
gbm = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.1,
                                 subsample=0.8, random_state=42)
gbm.fit(X_tr_s, y_tr)
gbm_vl = pearsonr(y_vl, gbm.predict(X_vl_s))[0]
gbm_te = pearsonr(y_te, gbm.predict(X_te_s))[0]
print(f"GBM: Val={gbm_vl:.4f}, Test={gbm_te:.4f}")

# XGBoost variants
print("\n--- XGBoost ---")
xgb1 = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1,
                        reg_alpha=0.5, reg_lambda=5.0, subsample=0.8,
                        colsample_bytree=0.8, random_state=42, verbosity=0, n_jobs=-1)
xgb1.fit(X_tr_s, y_tr)
xgb1_vl = pearsonr(y_vl, xgb1.predict(X_vl_s))[0]
xgb1_te = pearsonr(y_te, xgb1.predict(X_te_s))[0]
print(f"XGB1: Val={xgb1_vl:.4f}, Test={xgb1_te:.4f}")

xgb2 = xgb.XGBRegressor(n_estimators=400, max_depth=7, learning_rate=0.05,
                        reg_alpha=0.7, reg_lambda=7.0, subsample=0.8,
                        colsample_bytree=0.8, random_state=42, verbosity=0, n_jobs=-1)
xgb2.fit(X_tr_s, y_tr)
xgb2_vl = pearsonr(y_vl, xgb2.predict(X_vl_s))[0]
xgb2_te = pearsonr(y_te, xgb2.predict(X_te_s))[0]
print(f"XGB2: Val={xgb2_vl:.4f}, Test={xgb2_te:.4f}")

# Ensemble of ALL models
print("\n--- Full Ensemble ---")
all_preds_vl = [xgb1.predict(X_vl_s), xgb2.predict(X_vl_s), rf.predict(X_vl_s), gbm.predict(X_vl_s)]
all_preds_te = [xgb1.predict(X_te_s), xgb2.predict(X_te_s), rf.predict(X_te_s), gbm.predict(X_te_s)]

if HAS_CB:
    all_preds_vl.append(cb.predict(X_vl_s))
    all_preds_te.append(cb.predict(X_te_s))

# Simple average
ens_vl = np.mean(all_preds_vl, axis=0)
ens_te = np.mean(all_preds_te, axis=0)

r_ens_vl = pearsonr(y_vl, ens_vl)[0]
r_ens_te = pearsonr(y_te, ens_te)[0]
print(f"Full Ensemble: Val R={r_ens_vl:.4f}, Test R={r_ens_te:.4f}")

# 5-fold CV on ensemble
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_rs = []
models_to_cv = [xgb1, xgb2, rf, gbm]
if HAS_CB:
    models_to_cv.append(cb)

for fold, (tr_idx, vl_idx) in enumerate(kf.split(X_tr_s)):
    fold_preds = []
    for m in models_to_cv:
        if hasattr(m, 'get_params'):
            new_m = m.__class__(**m.get_params())
        else:
            continue
        new_m.fit(X_tr_s[tr_idx], y_tr[tr_idx])
        fold_preds.append(new_m.predict(X_tr_s[vl_idx]))
    fold_ens = np.mean(fold_preds, axis=0)
    cv_rs.append(pearsonr(y_tr[vl_idx], fold_ens)[0])
    print(f"  Fold {fold+1}: {cv_rs[-1]:.4f}")

cv_mean = np.mean(cv_rs)
cv_std = np.std(cv_rs)
print(f"\nFull Ensemble CV: {cv_mean:.4f} ± {cv_std:.4f}")

# Save
chunk2_results = {
    'xgb1': {'val_r': xgb1_vl, 'test_r': xgb1_te},
    'xgb2': {'val_r': xgb2_vl, 'test_r': xgb2_te},
    'rf': {'val_r': rf_vl, 'test_r': rf_te},
    'gbm': {'val_r': gbm_vl, 'test_r': gbm_te},
    'ensemble': {'val_r': r_ens_vl, 'test_r': r_ens_te, 'cv_r': cv_mean, 'cv_std': cv_std},
    'has_cb': HAS_CB,
}
if HAS_CB:
    chunk2_results['catboost'] = {'val_r': cb_vl, 'test_r': cb_te}

with open('WORK_DIR / chunk2_results.pkl', 'wb') as f:
    pickle.dump(chunk2_results, f)

print("\n✓ Chunk 2 complete")
print(f"Best: CV R = {cv_mean:.4f}")