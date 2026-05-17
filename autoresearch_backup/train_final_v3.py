#!/usr/bin/env python3
"""
Final improved model using ProLIF interaction features + enhanced molecular features.
Target: Improve from CV R = 0.736
"""
import pickle
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import KFold
import xgboost as xgb
# Import path helpers
try:
    from geock_paths import get_cache_dir, get_work_dir, CACHE_DIR, WORK_DIR
except ImportError:
    from pathlib import Path; CACHE_DIR = Path("/home/chow/.cache/geock_autoresearch"); WORK_DIR = Path("/home/chow/autoresearch")
import warnings
warnings.filterwarnings('ignore')

# ==================== LOAD DATA ====================
print("Loading data...")
cache = CACHE_DIR / lp_new_features_8k.pkl')
with open(cache, 'rb') as f:
    compounds = pickle.load(f)

# Load interaction features
X_interactions = np.load('WORK_DIR / X_interactions.npy')
with open('WORK_DIR / interaction_pdb_ids.pkl', 'rb') as f:
    interaction_pdb_ids = pickle.load(f)

print(f"Compounds: {len(compounds)}")
print(f"Interaction features: {X_interactions.shape}")

# Create PDB ID to index mapping for interaction features
interaction_pdb_set = set(interaction_pdb_ids)
interaction_idx_map = {pdb: i for i, pdb in enumerate(interaction_pdb_ids)}

# ==================== COMPUTE FEATURES ====================
print("\nComputing features with interaction features...")

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors

def compute_features_with_interactions(c):
    """Compute features including ProLIF interaction features."""
    ecfp = np.array(c['ecfp'], dtype=np.float32)
    mol = Chem.MolFromSmiles(c['smiles'])
    if mol is None:
        return None
    
    pdb_id = c['pdb_id']
    
    # Get interaction features if available
    if pdb_id in interaction_idx_map:
        int_idx = interaction_idx_map[pdb_id]
        int_features = X_interactions[int_idx]
    else:
        int_features = np.zeros(20, dtype=np.float32)
    
    try:
        # Basic molecular features
        mol_features = [
            Lipinski.RingCount(mol),
            Lipinski.NumAromaticRings(mol),
            Descriptors.MolLogP(mol),
            Descriptors.MolWt(mol),
            ecfp.sum(),  # bitcount
            Lipinski.NumHAcceptors(mol),
            Lipinski.NumHDonors(mol),
            Lipinski.NumRotatableBonds(mol),
        ]
        
        # Enhanced features
        try:
            mol_features.append(Descriptors.TPSA(mol))
        except:
            mol_features.append(0)
        
        try:
            mol_features.append(Descriptors.FractionCSP3(mol))
        except:
            mol_features.append(0)
        
        try:
            mol_features.append(Descriptors.NumHeteroatoms(mol))
        except:
            mol_features.append(0)
        
        try:
            mol_features.append(Descriptors.NumHeavyAtoms(mol))
        except:
            mol_features.append(0)
        
        mol_features = np.array(mol_features, dtype=np.float32)
        
        # Combine: ECFP + molecular features + interaction features
        X = np.concatenate([ecfp, mol_features, int_features])
        return X
    except:
        return None

X_list = []
y_list = []

for c in compounds:
    X = compute_features_with_interactions(c)
    if X is not None:
        X_list.append(X)
        y_list.append(c['affinity'])

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.float32)

print(f"Feature matrix: {X.shape}")
print(f"y range: {y.min():.2f} - {y.max():.2f}")

# ==================== SPLIT ====================
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

print(f"Split: {n_train} train / {n_val} val / {n_test} test")

# Standardize
mu = X_train.mean(0)
sd = X_train.std(0)
sd[sd == 0] = 1
X_train_s = (X_train - mu) / sd
X_val_s = (X_val - mu) / sd
X_test_s = (X_test - mu) / sd

# Feature selection
ECFP_IDX = list(range(512))
MOL_EXTRA_IDX = list(range(512, 520))  # 8 basic + 4 enhanced = 12
INT_IDX = list(range(520, 540))  # 20 interaction features

# Try different k values for feature selection
best_k = 400
best_r = 0

for k in [400, 450, 500]:
    sel = SelectKBest(f_regression, k=k)
    X_train_ecfp = sel.fit_transform(X_train_s[:, ECFP_IDX], y_train)
    
    # Use molecular extra + interaction features
    X_train_full = np.hstack([X_train_ecfp, X_train_s[:, MOL_EXTRA_IDX], X_train_s[:, INT_IDX]])
    
    m = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1,
                         random_state=42, verbosity=0, n_jobs=-1)
    m.fit(X_train_full, y_train)
    pred = m.predict(X_val_s[:, :X_train_full.shape[1]])
    r = pearsonr(y_val, pred)[0]
    print(f"k={k}: Val R = {r:.4f}")
    
    if r > best_r:
        best_r = r
        best_k = k

# Use best k
sel = SelectKBest(f_regression, k=best_k)
X_train_ecfp = sel.fit_transform(X_train_s[:, ECFP_IDX], y_train)
X_val_ecfp = sel.transform(X_val_s[:, ECFP_IDX])
X_test_ecfp = sel.transform(X_test_s[:, ECFP_IDX])

# Full features: selected ECFP + molecular extras + interaction
X_train_full = np.hstack([X_train_ecfp, X_train_s[:, MOL_EXTRA_IDX], X_train_s[:, INT_IDX]])
X_val_full = np.hstack([X_val_ecfp, X_val_s[:, MOL_EXTRA_IDX], X_val_s[:, INT_IDX]])
X_test_full = np.hstack([X_test_ecfp, X_test_s[:, MOL_EXTRA_IDX], X_test_s[:, INT_IDX]])

print(f"Features: {X_train_full.shape[1]} ({best_k} ECFP + 12 mol + 20 interaction)")

# ==================== ENSEMBLE TRAINING ====================
print("\n" + "="*70)
print("Training XGBoost ensemble")
print("="*70)

# Different XGBoost configs
xgb_configs = [
    {'max_depth': 5, 'learning_rate': 0.1, 'reg_alpha': 0.1, 'reg_lambda': 1.0},
    {'max_depth': 6, 'learning_rate': 0.08, 'reg_alpha': 0.3, 'reg_lambda': 3.0},
    {'max_depth': 7, 'learning_rate': 0.05, 'reg_alpha': 0.5, 'reg_lambda': 5.0},
    {'max_depth': 8, 'learning_rate': 0.03, 'reg_alpha': 0.7, 'reg_lambda': 7.0},
    {'max_depth': 4, 'learning_rate': 0.1, 'reg_alpha': 1.0, 'reg_lambda': 10.0},
]

models = []
val_preds = []
test_preds = []

for i, cfg in enumerate(xgb_configs):
    model = xgb.XGBRegressor(
        n_estimators=200,
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
    
    print(f"XGBoost {i+1}: Val R={r_val:.4f}, Test R={r_test:.4f}")
    
    models.append(model)
    val_preds.append(pred_val)
    test_preds.append(pred_test)

# Ensemble
val_ensemble = np.mean(val_preds, axis=0)
test_ensemble = np.mean(test_preds, axis=0)

r_ens_val = pearsonr(y_val, val_ensemble)[0]
r_ens_test = pearsonr(y_test, test_ensemble)[0]
mae_ens = np.mean(np.abs(y_test - test_ensemble))

print(f"\nEnsemble: Val R={r_ens_val:.4f}, Test R={r_ens_test:.4f}, MAE={mae_ens:.3f}")

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
    
    # Train ensemble of 3 models
    fold_preds = []
    for model_cfg in xgb_configs[:3]:
        m = xgb.XGBRegressor(
            n_estimators=150,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
            n_jobs=-1,
            **model_cfg
        )
        m.fit(X_tr, y_tr)
        fold_preds.append(m.predict(X_vl))
    
    fold_ensemble = np.mean(fold_preds, axis=0)
    r = pearsonr(y_vl, fold_ensemble)[0]
    cv_rs.append(r)
    print(f"Fold {fold+1}: R = {r:.4f}")

cv_mean = np.mean(cv_rs)
cv_std = np.std(cv_rs)
print(f"\nCV R: {cv_mean:.4f} ± {cv_std:.4f}")

# ==================== ERROR BY BINDING STRENGTH ====================
print("\n" + "="*70)
print("Error by binding strength")
print("="*70)

bins = [(0, 5, 'Weak'), (5, 7, 'Moderate'), (7, 10, 'Strong'), (10, 20, 'VeryStrong')]
for low, high, label in bins:
    mask = (y_test >= low) & (y_test < high)
    if mask.sum() >= 2:
        mae = np.mean(np.abs(y_test[mask] - test_ensemble[mask]))
        bias = np.mean(test_ensemble[mask] - y_test[mask])
        print(f"  {label}: n={mask.sum()}, MAE={mae:.2f}, Bias={bias:+.2f}")

# ==================== SAVE ====================
print("\n" + "="*70)
print("Saving model...")
print("="*70)

output = {
    'models': models,
    'configs': xgb_configs,
    'sel': sel,
    'mu': mu,
    'sd': sd,
    'k': best_k,
    'cv_r': cv_mean,
    'cv_std': cv_std,
    'val_r': r_ens_val,
    'test_r': r_ens_test,
    'mae': mae_ens,
}

output_path = WORK_DIR / geock_model_final_v3.pkl')
with open(output_path, 'wb') as f:
    pickle.dump(output, f)

print(f"Saved to {output_path}")
print(f"CV R: {cv_mean:.4f} ± {cv_std:.4f}")
print(f"Val R: {r_ens_val:.4f}")
print(f"Test R: {r_ens_test:.4f}")