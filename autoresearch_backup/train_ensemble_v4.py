#!/usr/bin/env python3
"""
Replicate best ensemble (CV R = 0.736) by using physics + interaction features.
"""
import pickle
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import KFold
import xgboost as xgb
import warnings
# Import path helpers
try:
    from geock_paths import get_cache_dir, get_work_dir, CACHE_DIR, WORK_DIR
except ImportError:
    from pathlib import Path; CACHE_DIR = Path("/home/chow/.cache/geock_autoresearch"); WORK_DIR = Path("/home/chow/autoresearch")
warnings.filterwarnings('ignore')

# ==================== LOAD DATA ====================
print("Loading data...")
cache = CACHE_DIR / lp_new_features_8k.pkl')
with open(cache, 'rb') as f:
    compounds = pickle.load(f)

# Load physics features
with open('CACHE_DIR / physics_features_8k.pkl', 'rb') as f:
    phys_data = pickle.load(f)
X_phys = phys_data['X_phys']
print(f"Physics features: {X_phys.shape}")

# Load interaction features
X_interactions = np.load('WORK_DIR / X_interactions.npy')
with open('WORK_DIR / interaction_pdb_ids.pkl', 'rb') as f:
    interaction_pdb_ids = pickle.load(f)

print(f"Interaction features: {X_interactions.shape}")

# Create mapping for interaction features
interaction_idx_map = {pdb: i for i, pdb in enumerate(interaction_pdb_ids)}

# ==================== COMPUTE FEATURES ====================
print("\nComputing features...")

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski

def compute_features(c):
    ecfp = np.array(c['ecfp'], dtype=np.float32)
    mol = Chem.MolFromSmiles(c['smiles'])
    if mol is None:
        return None, None
    
    pdb_id = c['pdb_id']
    
    # Interaction features
    if pdb_id in interaction_idx_map:
        int_idx = interaction_idx_map[pdb_id]
        int_features = X_interactions[int_idx]
    else:
        int_features = np.zeros(20, dtype=np.float32)
    
    try:
        mol_features = [
            Lipinski.RingCount(mol),
            Lipinski.NumAromaticRings(mol),
            Descriptors.MolLogP(mol),
            Descriptors.MolWt(mol),
            ecfp.sum(),
            Lipinski.NumHAcceptors(mol),
            Lipinski.NumHDonors(mol),
            Lipinski.NumRotatableBonds(mol),
        ]
        mol_features = np.array(mol_features, dtype=np.float32)
        
        X = np.concatenate([ecfp, mol_features])
        return X, int_features
    except:
        return None, None

X_list = []
int_list = []
y_list = []

for i, c in enumerate(compounds):
    X, int_f = compute_features(c)
    if X is not None:
        X_list.append(X)
        int_list.append(int_f)
        y_list.append(c['affinity'])

X = np.array(X_list, dtype=np.float32)
X_int = np.array(int_list, dtype=np.float32)
y = np.array(y_list, dtype=np.float32)

print(f"ECFP+mol features: {X.shape}")
print(f"Interaction features: {X_int.shape}")

# Combine all features
X_all = np.hstack([X, X_int])
print(f"Combined features: {X_all.shape}")

# ==================== SPLIT ====================
np.random.seed(42)
n = len(X_all)
perm = np.random.permutation(n)
n_test = int(n * 0.1)
n_val = int(n * 0.1)
n_train = n - n_test - n_val

train_idx = perm[:n_train]
val_idx = perm[n_train:n_train + n_val]
test_idx = perm[n_train + n_val:]

X_train = X_all[train_idx]
X_val = X_all[val_idx]
X_test = X_all[test_idx]
y_train = y[train_idx]
y_val = y[val_idx]
y_test = y[test_idx]

print(f"Split: {n_train} train / {n_val} val / {n_test} test")

# Standardize
mu = X_train.mean(0)
sd = X_train.std(0)
sd[sd == 0] = 1
X_train_s = (X_train - mu) / sd
X_val_s = (X_val - mu) / sd
X_test_s = (X_test - mu) / sd

# Feature selection on ECFP part only
ECFP_IDX = list(range(512))
MOL_IDX = list(range(512, 520))  # 8 mol features
INT_IDX = list(range(520, 540))  # 20 interaction features

# Test different k
for k in [400, 450]:
    sel = SelectKBest(f_regression, k=k)
    X_train_ecfp = sel.fit_transform(X_train_s[:, ECFP_IDX], y_train)
    
    X_train_full = np.hstack([X_train_ecfp, X_train_s[:, MOL_IDX], X_train_s[:, INT_IDX]])
    
    m = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1,
                         random_state=42, verbosity=0, n_jobs=-1)
    m.fit(X_train_full, y_train)
    pred = m.predict(X_val_s[:, :X_train_full.shape[1]])
    r = pearsonr(y_val, pred)[0]
    print(f"k={k}: Val R = {r:.4f}")

# Use k=450
k = 450
sel = SelectKBest(f_regression, k=k)
X_train_ecfp = sel.fit_transform(X_train_s[:, ECFP_IDX], y_train)
X_val_ecfp = sel.transform(X_val_s[:, ECFP_IDX])
X_test_ecfp = sel.transform(X_test_s[:, ECFP_IDX])

X_train_full = np.hstack([X_train_ecfp, X_train_s[:, MOL_IDX], X_train_s[:, INT_IDX]])
X_val_full = np.hstack([X_val_ecfp, X_val_s[:, MOL_IDX], X_val_s[:, INT_IDX]])
X_test_full = np.hstack([X_test_ecfp, X_test_s[:, MOL_IDX], X_test_s[:, INT_IDX]])

print(f"Features: {X_train_full.shape[1]} ({k} ECFP + 8 mol + 20 int)")

# ==================== TRAIN ENSEMBLE ====================
print("\n" + "="*70)
print("Training ensemble")
print("="*70)

# Same configs as best ensemble
xgb_configs = [
    {'max_depth': 6, 'learning_rate': 0.1, 'reg_alpha': 0.5, 'reg_lambda': 5.0, 'name': 'shallow'},
    {'max_depth': 8, 'learning_rate': 0.05, 'reg_alpha': 0.7, 'reg_lambda': 7.0, 'name': 'deep'},
    {'max_depth': 7, 'learning_rate': 0.08, 'reg_alpha': 0.5, 'reg_lambda': 5.0, 'name': 'balanced'},
    {'max_depth': 10, 'learning_rate': 0.03, 'reg_alpha': 1.0, 'reg_lambda': 10.0, 'name': 'very_deep'},
    {'max_depth': 5, 'learning_rate': 0.1, 'reg_alpha': 0.1, 'reg_lambda': 1.0, 'name': 'low_reg'},
]

models = []
val_preds = []
test_preds = []

for i, cfg in enumerate(xgb_configs):
    name = cfg.pop('name')
    model = xgb.XGBRegressor(
        n_estimators=300,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        random_state=42 + i,
        verbosity=0,
        n_jobs=-1,
        **cfg
    )
    model.fit(X_train_full, y_train)
    
    pred_val = model.predict(X_val_full)
    pred_test = model.predict(X_test_full)
    
    r_val = pearsonr(y_val, pred_val)[0]
    r_test = pearsonr(y_test, pred_test)[0]
    
    print(f"{name}: Val R={r_val:.4f}, Test R={r_test:.4f}")
    
    cfg['name'] = name  # restore
    models.append(model)
    val_preds.append(pred_val)
    test_preds.append(pred_test)

# Optimize weights
from scipy.optimize import minimize

def optimize_weights(preds_list, y_true):
    n = len(preds_list)
    def loss(w):
        w = np.abs(w)
        w = w / w.sum()
        ensemble = sum(wi * pi for wi, pi in zip(w, preds_list))
        return np.mean((ensemble - y_true) ** 2)
    result = minimize(loss, x0=np.ones(n) / n, method='Nelder-Mead')
    w = np.abs(result.x)
    return w / w.sum()

weights = optimize_weights(val_preds, y_val)
print(f"\nOptimized weights: {weights}")

# Weighted ensemble
val_ensemble = sum(w * p for w, p in zip(weights, val_preds))
test_ensemble = sum(w * p for w, p in zip(weights, test_preds))

r_ens_val = pearsonr(y_val, val_ensemble)[0]
r_ens_test = pearsonr(y_test, test_ensemble)[0]
mae_ens = np.mean(np.abs(y_test - test_ensemble))

print(f"Ensemble: Val R={r_ens_val:.4f}, Test R={r_ens_test:.4f}, MAE={mae_ens:.3f}")

# ==================== CROSS-VALIDATION ====================
print("\n" + "="*70)
print("5-Fold Cross-Validation")
print("="*70)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_rs = []

for fold, (tr_idx, vl_idx) in enumerate(kf.split(X_train_full)):
    X_tr = X_train_full[tr_idx]
    y_tr = y_train[tr_idx]
    X_vl = X_train_full[vl_idx]
    y_vl = y_train[vl_idx]
    
    # Train ensemble
    fold_preds = []
    for cfg in xgb_configs:
        name = cfg['name']
        m = xgb.XGBRegressor(
            n_estimators=200,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
            n_jobs=-1,
            max_depth=cfg['max_depth'],
            learning_rate=cfg['learning_rate'],
            reg_alpha=cfg['reg_alpha'],
            reg_lambda=cfg['reg_lambda'],
        )
        m.fit(X_tr, y_tr)
        fold_preds.append(m.predict(X_vl))
    
    # Weighted ensemble
    fold_ensemble = sum(w * p for w, p in zip(weights, fold_preds))
    r = pearsonr(y_vl, fold_ensemble)[0]
    cv_rs.append(r)
    print(f"Fold {fold+1}: R = {r:.4f}")

cv_mean = np.mean(cv_rs)
cv_std = np.std(cv_rs)
print(f"\nCV R: {cv_mean:.4f} ± {cv_std:.4f}")

# ==================== ERROR BY BINDING STRENGTH ====================
print("\nError by binding strength:")
bins = [(0, 5, 'Weak'), (5, 7, 'Moderate'), (7, 10, 'Strong'), (10, 20, 'VeryStrong')]
for low, high, label in bins:
    mask = (y_test >= low) & (y_test < high)
    if mask.sum() >= 2:
        mae = np.mean(np.abs(y_test[mask] - test_ensemble[mask]))
        bias = np.mean(test_ensemble[mask] - y_test[mask])
        print(f"  {label}: n={mask.sum()}, MAE={mae:.2f}, Bias={bias:+.2f}")

# ==================== SAVE ====================
output = {
    'models': models,
    'weights': weights,
    'configs': xgb_configs,
    'sel': sel,
    'mu': mu,
    'sd': sd,
    'k': k,
    'cv_r': cv_mean,
    'cv_std': cv_std,
    'val_r': r_ens_val,
    'test_r': r_ens_test,
    'mae': mae_ens,
}

output_path = WORK_DIR / geock_ensemble_v4.pkl')
with open(output_path, 'wb') as f:
    pickle.dump(output, f)

print(f"\nSaved to {output_path}")
print(f"CV R: {cv_mean:.4f} ± {cv_std:.4f}")
print(f"Val R: {r_ens_val:.4f}")
print(f"Test R: {r_ens_test:.4f}")