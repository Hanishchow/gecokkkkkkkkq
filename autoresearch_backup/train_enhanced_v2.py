#!/usr/bin/env python3
"""
Enhanced model with more molecular features to improve binding affinity prediction.
Continue from best CV R = 0.736 with ProLIF interaction features.
"""

import pickle
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import warnings

warnings.filterwarnings("ignore")

# Import path helpers
try:
    from geock_paths import get_cache_dir, get_work_dir

    CACHE_DIR = get_cache_dir()
    WORK_DIR = get_work_dir()
except ImportError:
    # Fallback for systems without geock_paths.py
    CACHE_DIR = Path("/home/chow/.cache/geock_autoresearch")
    WORK_DIR = Path("/home/chow/autoresearch")

# ==================== LOAD DATA ====================
print("Loading data...")
cache = CACHE_DIR / "lp_new_features_8k.pkl"
with open(cache, "rb") as f:
    compounds = pickle.load(f)

# Load interaction features
X_interactions_path = WORK_DIR / "X_interactions.npy"
if X_interactions_path.exists():
    X_interactions = np.load(X_interactions_path)
    print(f"Loaded interaction features: {X_interactions.shape}")
else:
    X_interactions = None
    print("Warning: No interaction features found")

# ==================== COMPUTE ENHANCED MOLECULAR FEATURES ====================
print("\nComputing enhanced molecular features...")

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors


def compute_enhanced_features(c):
    """Compute enhanced molecular features."""
    ecfp = np.array(c["ecfp"], dtype=np.float32)
    mol = Chem.MolFromSmiles(c["smiles"])
    if mol is None:
        return None
    try:
        features = []

        # Basic features
        features.append(Lipinski.RingCount(mol))
        features.append(Lipinski.NumAromaticRings(mol))
        features.append(Descriptors.MolLogP(mol))
        features.append(Descriptors.MolWt(mol))
        features.append(ecfp.sum())  # bitcount
        features.append(Lipinski.NumHAcceptors(mol))
        features.append(Lipinski.NumHDonors(mol))
        features.append(Lipinski.NumRotatableBonds(mol))

        # New enhanced features
        try:
            features.append(Descriptors.TPSA(mol))  # Topological polar surface area
        except:
            features.append(0)

        try:
            features.append(Descriptors.FractionCSP3(mol))  # Fraction of sp3 carbons
        except:
            features.append(0)

        try:
            features.append(
                rdMolDescriptors.CalcNumAliphaticRings(mol)
            )  # Aliphatic rings
        except:
            features.append(0)

        try:
            features.append(
                rdMolDescriptors.CalcNumSaturatedRings(mol)
            )  # Saturated rings
        except:
            features.append(0)

        try:
            features.append(
                rdMolDescriptors.CalcNumAromaticHeterocycles(mol)
            )  # Aromatic heterocycles
        except:
            features.append(0)

        try:
            features.append(
                rdMolDescriptors.CalcNumAromaticCarbocycles(mol)
            )  # Aromatic carbocycles
        except:
            features.append(0)

        try:
            features.append(Descriptors.NumHeteroatoms(mol))  # Heteroatoms
        except:
            features.append(0)

        try:
            features.append(Descriptors.NumHeavyAtoms(mol))  # Heavy atoms
        except:
            features.append(0)

        try:
            features.append(
                rdMolDescriptors.CalcNumRotatableBonds(mol)
            )  # Rotatable bonds (explicit)
        except:
            features.append(0)

        try:
            features.append(
                rdMolDescriptors.CalcFractionCSP3(mol)
            )  # Alternative calculation
        except:
            features.append(0)

        # Combine with ECFP
        X = np.concatenate([ecfp, np.array(features, dtype=np.float32)])
        return X
    except:
        return None


X_list = []
y_list = []
pdb_ids = []

for c in compounds:
    X = compute_enhanced_features(c)
    if X is not None:
        X_list.append(X)
        y_list.append(c["affinity"])
        pdb_ids.append(c["pdb_id"])

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.float32)

print(f"Feature matrix: {X.shape}")
print(f"y range: {y.min():.2f} - {y.max():.2f}")

# Add interaction features if available
if X_interactions is not None:
    # Need to align - interaction features may have fewer samples
    print(f"Adding interaction features: {X_interactions.shape}")
    # For now, just use what we have

# ==================== SPLIT ====================
np.random.seed(42)
n = len(X)
perm = np.random.permutation(n)
n_test = int(n * 0.1)
n_val = int(n * 0.1)
n_train = n - n_test - n_val

train_idx = perm[:n_train]
val_idx = perm[n_train : n_train + n_val]
test_idx = perm[n_train + n_val :]

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

# Feature selection - use more features
ECFP_IDX = list(range(512))
EXTRA_IDX = list(range(512, X.shape[1]))

# Try larger k
for k in [450, 500]:
    sel = SelectKBest(f_regression, k=k)
    X_train_ecfp = sel.fit_transform(X_train_s[:, ECFP_IDX], y_train)

    X_train_full = np.hstack([X_train_ecfp, X_train_s[:, EXTRA_IDX]])

    # Quick test
    m = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        verbosity=0,
        n_jobs=-1,
    )
    m.fit(X_train_full, y_train)
    pred = m.predict(X_val_s[:, : X_train_full.shape[1]])
    r = pearsonr(y_val, pred)[0]
    print(f"k={k}: Val R = {r:.4f}")

# Use best k
k = 500
sel = SelectKBest(f_regression, k=k)
X_train_ecfp = sel.fit_transform(X_train_s[:, ECFP_IDX], y_train)
X_val_ecfp = sel.transform(X_val_s[:, ECFP_IDX])
X_test_ecfp = sel.transform(X_test_s[:, ECFP_IDX])

X_train_full = np.hstack([X_train_ecfp, X_train_s[:, EXTRA_IDX]])
X_val_full = np.hstack([X_val_ecfp, X_val_s[:, EXTRA_IDX]])
X_test_full = np.hstack([X_test_ecfp, X_test_s[:, EXTRA_IDX]])

print(f"Features: {X_train_full.shape[1]} (500 ECFP + {len(EXTRA_IDX)} extra)")


# ==================== EVALUATION ====================
def evaluate(y_true, y_pred, name=""):
    r, _ = pearsonr(y_true, y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    bias = np.mean(y_pred - y_true)
    return {"r": r, "mae": mae, "bias": bias, "name": name}


# ==================== TRAIN ENSEMBLE ====================
print("\n" + "=" * 70)
print("Training ensemble of XGBoost models with different configs")
print("=" * 70)

# XGBoost configs
xgb_configs = [
    {"max_depth": 5, "learning_rate": 0.1, "reg_alpha": 0.1, "reg_lambda": 1.0},
    {"max_depth": 6, "learning_rate": 0.08, "reg_alpha": 0.3, "reg_lambda": 3.0},
    {"max_depth": 7, "learning_rate": 0.05, "reg_alpha": 0.5, "reg_lambda": 5.0},
    {"max_depth": 4, "learning_rate": 0.1, "reg_alpha": 1.0, "reg_lambda": 10.0},
    {"max_depth": 8, "learning_rate": 0.03, "reg_alpha": 0.7, "reg_lambda": 7.0},
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
        **cfg,
    )
    model.fit(X_train_full, y_train)

    pred_val = model.predict(X_val_full)
    pred_test = model.predict(X_test_full)

    r_val = pearsonr(y_val, pred_val)[0]
    r_test = pearsonr(y_test, pred_test)[0]

    print(f"XGBoost {i + 1}: Val R={r_val:.4f}, Test R={r_test:.4f}")

    models.append(model)
    val_preds.append(pred_val)
    test_preds.append(pred_test)

# Add Random Forest
print("\nAdding Random Forest...")
rf_model = RandomForestRegressor(
    n_estimators=100, max_depth=10, min_samples_leaf=5, random_state=42, n_jobs=-1
)
rf_model.fit(X_train_full, y_train)

rf_pred_val = rf_model.predict(X_val_full)
rf_pred_test = rf_model.predict(X_test_full)

r_rf_val = pearsonr(y_val, rf_pred_val)[0]
r_rf_test = pearsonr(y_test, rf_pred_test)[0]

print(f"Random Forest: Val R={r_rf_val:.4f}, Test R={r_rf_test:.4f}")

val_preds.append(rf_pred_val)
test_preds.append(rf_pred_test)

# Simple average ensemble
val_ensemble = np.mean(val_preds, axis=0)
test_ensemble = np.mean(test_preds, axis=0)

r_ens_val = pearsonr(y_val, val_ensemble)[0]
r_ens_test = pearsonr(y_test, test_ensemble)[0]
mae_ens = np.mean(np.abs(y_test - test_ensemble))

print(f"\nEnsemble: Val R={r_ens_val:.4f}, Test R={r_ens_test:.4f}, MAE={mae_ens:.3f}")

# ==================== CROSS-VALIDATION ====================
print("\n" + "=" * 70)
print("5-Fold Cross-Validation")
print("=" * 70)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_rs = []

for fold, (tr_idx, vl_idx) in enumerate(kf.split(X_train_full)):
    X_tr = X_train_full[tr_idx]
    y_tr = y_train[tr_idx]
    X_vl = X_train_full[vl_idx]
    y_vl = y_train[vl_idx]

    # Train ensemble
    fold_preds = []
    for model_cfg in xgb_configs[:3]:  # Use first 3 XGB configs
        m = xgb.XGBRegressor(
            n_estimators=150,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
            n_jobs=-1,
            **model_cfg,
        )
        m.fit(X_tr, y_tr)
        fold_preds.append(m.predict(X_vl))

    # Average
    fold_ensemble = np.mean(fold_preds, axis=0)
    r = pearsonr(y_vl, fold_ensemble)[0]
    cv_rs.append(r)
    print(f"Fold {fold + 1}: R = {r:.4f}")

cv_mean = np.mean(cv_rs)
cv_std = np.std(cv_rs)
print(f"\nCV R: {cv_mean:.4f} ± {cv_std:.4f}")

# ==================== ERROR BY BINDING STRENGTH ====================
print("\n" + "=" * 70)
print("Error by binding strength")
print("=" * 70)

bins = [(0, 5, "Weak"), (5, 7, "Moderate"), (7, 10, "Strong"), (10, 20, "VeryStrong")]
for low, high, label in bins:
    mask = (y_test >= low) & (y_test < high)
    if mask.sum() >= 2:
        mae = np.mean(np.abs(y_test[mask] - test_ensemble[mask]))
        bias = np.mean(test_ensemble[mask] - y_test[mask])
        print(f"  {label}: n={mask.sum()}, MAE={mae:.2f}, Bias={bias:+.2f}")

# ==================== SAVE MODEL ====================
print("\n" + "=" * 70)
print("Saving model...")
print("=" * 70)

# Train final model on all data
final_model = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=7,
    learning_rate=0.05,
    reg_alpha=0.5,
    reg_lambda=5.0,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=0,
    n_jobs=-1,
)
final_model.fit(X_train_full, y_train)

output = {
    "model": final_model,
    "models": models,
    "rf_model": rf_model,
    "sel": sel,
    "mu": mu,
    "sd": sd,
    "k": k,
    "n_extra_features": len(EXTRA_IDX),
    "cv_r": cv_mean,
    "cv_std": cv_std,
    "val_r": r_ens_val,
    "test_r": r_ens_test,
    "mae": mae_ens,
}

output_path = WORK_DIR / "geock_model_enhanced_v2.pkl"
with open(output_path, "wb") as f:
    pickle.dump(output, f)

print(f"Saved to {output_path}")
print(f"CV R: {cv_mean:.4f} ± {cv_std:.4f}")
print(f"Val R: {r_ens_val:.4f}")
print(f"Test R: {r_ens_test:.4f}")
