#!/usr/bin/env python3
"""
Chunk 1: Try LightGBM and different feature combinations
"""

import pickle
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
import xgboost as xgb
import warnings

warnings.filterwarnings("ignore")

print("=" * 70)
print("CHUNK 1: LightGBM + Feature Combinations")
print("=" * 70)

# Load data
try:
    from geock_paths import get_cache_dir

    cache_dir = get_cache_dir()
except ImportError:
    cache_dir = Path("/home/chow/.cache/geock_autoresearch")

cache = cache_dir / "lp_new_features_8k.pkl"
with open(cache, "rb") as f:
    compounds = pickle.load(f)

with open("CACHE_DIR / physics_features_8k.pkl", "rb") as f:
    phys_data = pickle.load(f)
X_phys = phys_data["X_phys"]

X_int = np.load("WORK_DIR / X_interactions.npy")
with open("WORK_DIR / interaction_pdb_ids.pkl", "rb") as f:
    int_pdb_ids = pickle.load(f)

int_map = {pdb: i for i, pdb in enumerate(int_pdb_ids)}

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski

# Try LightGBM
try:
    import lightgbm as lgb

    HAS_LGB = True
    print("LightGBM available")
except:
    HAS_LGB = False
    print("LightGBM not available")

# Build features
X_list, y_list = [], []

for i, c in enumerate(compounds):
    ecfp = np.array(c["ecfp"], dtype=np.float32)
    mol = Chem.MolFromSmiles(c["smiles"])
    if mol is None:
        continue

    pdb_id = c["pdb_id"]
    int_feat = (
        X_int[int_map[pdb_id]] if pdb_id in int_map else np.zeros(20, dtype=np.float32)
    )

    mol_feat = np.array(
        [
            Lipinski.RingCount(mol),
            Lipinski.NumAromaticRings(mol),
            Descriptors.MolLogP(mol),
            Descriptors.MolWt(mol),
            ecfp.sum(),
            Lipinski.NumHAcceptors(mol),
            Lipinski.NumHDonors(mol),
            Lipinski.NumRotatableBonds(mol),
        ],
        dtype=np.float32,
    )

    X = np.concatenate([ecfp, mol_feat, X_phys[i], int_feat])
    X_list.append(X)
    y_list.append(c["affinity"])

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
idx_vl = perm[n_train : n_train + n_val]
idx_te = perm[n_train + n_val :]

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

# Test LightGBM
if HAS_LGB:
    print("\n--- LightGBM ---")
    lgb_params = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "reg_alpha": 0.5,
        "reg_lambda": 5.0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "verbosity": -1,
        "n_jobs": -1,
    }
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(X_tr_s, y_tr)

    p_vl = lgb_model.predict(X_vl_s)
    p_te = lgb_model.predict(X_te_s)

    r_vl = pearsonr(y_vl, p_vl)[0]
    r_te = pearsonr(y_te, p_te)[0]
    print(f"LightGBM: Val R={r_vl:.4f}, Test R={r_te:.4f}")

    # 5-fold CV
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_rs = []
    for tr_idx, vl_idx in kf.split(X_tr_s):
        m = lgb.LGBMRegressor(**lgb_params)
        m.fit(X_tr_s[tr_idx], y_tr[tr_idx])
        cv_rs.append(pearsonr(y_tr[vl_idx], m.predict(X_tr_s[vl_idx]))[0])
    print(f"LightGBM CV: {np.mean(cv_rs):.4f} ± {np.std(cv_rs):.4f}")

# Test XGBoost with different settings
print("\n--- XGBoost variants ---")

# XGBoost 1: Standard
xgb1 = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    reg_alpha=0.5,
    reg_lambda=5.0,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=0,
    n_jobs=-1,
)
xgb1.fit(X_tr_s, y_tr)
r1_vl = pearsonr(y_vl, xgb1.predict(X_vl_s))[0]
r1_te = pearsonr(y_te, xgb1.predict(X_te_s))[0]
print(f"XGB1: Val={r1_vl:.4f}, Test={r1_te:.4f}")

# XGBoost 2: Higher regularization
xgb2 = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    reg_alpha=1.0,
    reg_lambda=10.0,
    subsample=0.7,
    colsample_bytree=0.7,
    min_child_weight=5,
    random_state=42,
    verbosity=0,
    n_jobs=-1,
)
xgb2.fit(X_tr_s, y_tr)
r2_vl = pearsonr(y_vl, xgb2.predict(X_vl_s))[0]
r2_te = pearsonr(y_te, xgb2.predict(X_te_s))[0]
print(f"XGB2: Val={r2_vl:.4f}, Test={r2_te:.4f}")

# XGBoost 3: More trees, lower LR
xgb3 = xgb.XGBRegressor(
    n_estimators=500,
    max_depth=7,
    learning_rate=0.03,
    reg_alpha=0.7,
    reg_lambda=7.0,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=0,
    n_jobs=-1,
)
xgb3.fit(X_tr_s, y_tr)
r3_vl = pearsonr(y_vl, xgb3.predict(X_vl_s))[0]
r3_te = pearsonr(y_te, xgb3.predict(X_te_s))[0]
print(f"XGB3: Val={r3_vl:.4f}, Test={r3_te:.4f}")

# Ensemble all
print("\n--- Ensemble ---")
all_preds_vl = [xgb1.predict(X_vl_s), xgb2.predict(X_vl_s), xgb3.predict(X_vl_s)]
all_preds_te = [xgb1.predict(X_te_s), xgb2.predict(X_te_s), xgb3.predict(X_te_s)]

if HAS_LGB:
    all_preds_vl.append(lgb_model.predict(X_vl_s))
    all_preds_te.append(lgb_model.predict(X_te_s))

# Simple average
ens_vl = np.mean(all_preds_vl, axis=0)
ens_te = np.mean(all_preds_te, axis=0)

r_ens_vl = pearsonr(y_vl, ens_vl)[0]
r_ens_te = pearsonr(y_te, ens_te)[0]
print(f"Ensemble: Val R={r_ens_vl:.4f}, Test R={r_ens_te:.4f}")

# CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_rs = []
for fold, (tr_idx, vl_idx) in enumerate(kf.split(X_tr_s)):
    fold_preds = []
    for model in [xgb1, xgb2, xgb3]:
        m = model.__class__(**model.get_params())
        m.fit(X_tr_s[tr_idx], y_tr[tr_idx])
        fold_preds.append(m.predict(X_tr_s[vl_idx]))
    if HAS_LGB:
        m = lgb.LGBMRegressor(**lgb_params)
        m.fit(X_tr_s[tr_idx], y_tr[tr_idx])
        fold_preds.append(m.predict(X_tr_s[vl_idx]))
    fold_ens = np.mean(fold_preds, axis=0)
    cv_rs.append(pearsonr(y_tr[vl_idx], fold_ens)[0])
    print(f"  Fold {fold + 1}: {cv_rs[-1]:.4f}")

cv_mean = np.mean(cv_rs)
cv_std = np.std(cv_rs)
print(f"\nEnsemble CV: {cv_mean:.4f} ± {cv_std:.4f}")

# Save results
chunk1_results = {
    "xgb1": {"val_r": r1_vl, "test_r": r1_te},
    "xgb2": {"val_r": r2_vl, "test_r": r2_te},
    "xgb3": {"val_r": r3_vl, "test_r": r3_te},
    "ensemble": {
        "val_r": r_ens_vl,
        "test_r": r_ens_te,
        "cv_r": cv_mean,
        "cv_std": cv_std,
    },
    "has_lgb": HAS_LGB,
}
if HAS_LGB:
    chunk1_results["lgb"] = {
        "val_r": pearsonr(y_vl, lgb_model.predict(X_vl_s))[0],
        "test_r": pearsonr(y_te, lgb_model.predict(X_te_s))[0],
    }

with open("WORK_DIR / chunk1_results.pkl", "wb") as f:
    pickle.dump(chunk1_results, f)

print("\n✓ Chunk 1 complete")
print(f"Best: CV R = {cv_mean:.4f}")
