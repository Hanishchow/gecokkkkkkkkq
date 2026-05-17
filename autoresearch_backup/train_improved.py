#!/usr/bin/env python3
"""Train improved model with extra molecular features."""
import pickle
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import KFold
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Import path helpers
try:
    from geock_paths import get_cache_dir, get_work_dir, CACHE_DIR, WORK_DIR
except ImportError:
    from pathlib import Path; CACHE_DIR = Path("/home/chow/.cache/geock_autoresearch"); WORK_DIR = Path("/home/chow/autoresearch")
# Load data
cache = CACHE_DIR / lp_new_features_8k.pkl')
with open(cache, 'rb') as f:
    compounds = pickle.load(f)

print('Computing molecular features...')

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski

X_list = []
y_list = []
pdb_ids = []

for c in compounds:
    ecfp = np.array(c['ecfp'], dtype=np.float32)
    
    mol = Chem.MolFromSmiles(c['smiles'])
    if mol is None:
        continue
    
    try:
        rings = np.array([Lipinski.RingCount(mol)], dtype=np.float32)
        aromatic = np.array([Lipinski.NumAromaticRings(mol)], dtype=np.float32)
        logp = np.array([Descriptors.MolLogP(mol)], dtype=np.float32)
        mw = np.array([Descriptors.MolWt(mol)], dtype=np.float32)
        bitcount = np.array([ecfp.sum()], dtype=np.float32)
        
        X = np.concatenate([ecfp, rings, aromatic, logp, mw, bitcount])
        X_list.append(X)
        y_list.append(c['affinity'])
        pdb_ids.append(c['pdb_id'])
    except:
        continue

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.float32)

print(f'Feature matrix: {X.shape}')

# Split
np.random.seed(42)
n = len(X)
perm = np.random.permutation(n)
n_test = int(n * 0.1)
n_val = int(n * 0.1)
n_train = n - n_test - n_val

train_idx = perm[:n_train]
val_idx = perm[n_train:n_train + n_val]
test_idx = perm[n_train + n_val:]

X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]
X_test, y_test = X[test_idx], y[test_idx]

# Standardize
mu = X_train.mean(0)
sd = X_train.std(0)
sd[sd == 0] = 1
X_train_s = (X_train - mu) / sd
X_val_s = (X_val - mu) / sd
X_test_s = (X_test - mu) / sd

# Feature selection
ECFP_IDX = list(range(512))
EXTRA_IDX = list(range(512, 517))

k = 400
sel = SelectKBest(f_regression, k=k)
X_train_ecfp = sel.fit_transform(X_train_s[:, ECFP_IDX], y_train)
X_val_ecfp = sel.transform(X_val_s[:, ECFP_IDX])
X_test_ecfp = sel.transform(X_test_s[:, ECFP_IDX])

# Full features
X_train_full = np.hstack([X_train_ecfp, X_train_s[:, EXTRA_IDX]])
X_val_full = np.hstack([X_val_ecfp, X_val_s[:, EXTRA_IDX]])
X_test_full = np.hstack([X_test_ecfp, X_test_s[:, EXTRA_IDX]])

# XGBoost
xgb_params = {
    'n_estimators': 100,
    'max_depth': 5,
    'learning_rate': 0.1,
    'min_child_weight': 3,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'verbosity': 0,
    'n_jobs': -1,
}

print('Training final model...')
model = xgb.XGBRegressor(**xgb_params)
model.fit(X_train_full, y_train)

val_r = pearsonr(y_val, model.predict(X_val_full))[0]
test_r = pearsonr(y_test, model.predict(X_test_full))[0]

# 5-fold CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_r = []
for tr_idx, vl_idx in kf.split(X_train_full):
    m = xgb.XGBRegressor(**xgb_params)
    m.fit(X_train_full[tr_idx], y_train[tr_idx])
    cv_r.append(pearsonr(y_train[vl_idx], m.predict(X_train_full[vl_idx]))[0])

print()
print(f'Val R: {val_r:.4f}')
print(f'Test R: {test_r:.4f}')
print(f'5-Fold CV R: {np.mean(cv_r):.4f} ± {np.std(cv_r):.4f}')

# Save model
out = {
    'model': model,
    'sel': sel,
    'mu': mu,
    'sd': sd,
    'k': k,
    'extra_features': ['rings', 'aromatic_rings', 'logp', 'mw', 'bitcount'],
    'val_r': val_r,
    'test_r': test_r,
    'cv_r': np.mean(cv_r),
    'cv_std': np.std(cv_r),
}

out_path = WORK_DIR / geock_model_improved.pkl')
with open(out_path, 'wb') as f:
    pickle.dump(out, f)

print(f'Saved to {out_path}')