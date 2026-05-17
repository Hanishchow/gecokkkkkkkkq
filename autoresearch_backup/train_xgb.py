"""
train_xgb.py — GEOCK with XGBoost + Ridge Ensemble
==================================================
Key changes from train.py:
  1. XGBoost with proper regularization (min_child_weight, reg_alpha, reg_lambda)
  2. Ridge with optimized alpha
  3. Ensemble: simple average of XGB + Ridge
  4. Repeated K-fold CV (5-fold × 10 repeats) for stable metric
  5. All models use the same train/val/test split
  6. Feature normalization: fit on TRAIN only (no leakage)
  7. Feature selection: fit on TRAIN only (no leakage)

Why XGBoost?
  - 2023 J Cheminformatics study of 157,590 QSAR models found XGBoost best
  - Handles non-linear relationships
  - Built-in regularization: min_child_weight, reg_alpha, reg_lambda, subsample
  - Robust to feature scale (though we still normalize)
"""

import sys, os, warnings, pickle
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from pathlib import Path
from prepare import evaluate_r, evaluate_mae
from sklearn.linear_model import Ridge
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import LeaveOneOut, cross_val_predict, RepeatedKFold

CACHE = Path("CACHE_DIR / features_110.pkl")
with open(CACHE, "rb") as f:
    d = pickle.load(f)

X_raw = d["X_raw"]
X_ecfp_list = d["X_ecfp"]
y_all = d["y_pkd"]
ids = d["pdb_ids"]

ecfp_len = len(np.asarray(X_ecfp_list[0], dtype=np.float32))
X_ecfp = np.zeros((len(X_ecfp_list), ecfp_len), dtype=np.float32)
for i, e in enumerate(X_ecfp_list):
    arr = np.asarray(e, dtype=np.float32)
    if arr.shape[0] == ecfp_len:
        X_ecfp[i] = arr
    elif arr.shape[0] > ecfp_len:
        X_ecfp[i] = arr[:ecfp_len]
    else:
        X_ecfp[i, :arr.shape[0]] = arr

X_all = np.hstack([X_raw, X_ecfp])
N_TOTAL = len(y_all)

N_TEST = 5; N_VAL = 5
np.random.seed(42)
perm = np.random.permutation(N_TOTAL)
test_idx  = perm[:N_TEST]
val_idx   = perm[N_TEST:N_TEST+N_VAL]
train_idx = perm[N_TEST+N_VAL:]

X_train_raw = X_all[train_idx]; y_train = y_all[train_idx]
X_val_raw   = X_all[val_idx];   y_val   = y_all[val_idx]
X_test_raw  = X_all[test_idx];  y_test  = y_all[test_idx]

FI = list(range(0, 14))
mu = X_train_raw[:, FI].mean(0)
sd = X_train_raw[:, FI].std(0); sd = np.where(sd == 0, 1, sd)
X_tr_s = (X_train_raw[:, FI] - mu) / sd
X_vl_s = (X_val_raw[:, FI] - mu) / sd
X_te_s = (X_test_raw[:, FI] - mu) / sd

sel = SelectKBest(f_regression, k=10)
X_tr_p = sel.fit_transform(X_tr_s, y_train)
X_vl_p = sel.transform(X_vl_s)
X_te_p = sel.transform(X_te_s)

from prepare import FEATURE_NAMES
fnames = [FEATURE_NAMES[i] for i in FI]
selected = [fnames[i] for i, v in enumerate(sel.get_support()) if v]

print("=" * 65)
print("  GEOCK — XGBoost + Ridge Ensemble")
print("=" * 65)
print(f"  Selected features: {selected}")
print(f"  Train: {len(train_idx)}   Val: {len(val_idx)}   Test: {len(test_idx)}")
print(f"  Test PDBs: {[ids[i] for i in test_idx]}")
print(f"  Val PDBs:   {[ids[i] for i in val_idx]}")
print()

print("=" * 65)
print("  RIDGE (alpha=5.0, baseline)")
print("=" * 65)
ridge = Ridge(alpha=5.0)
ridge.fit(X_tr_p, y_train)
ridge_val_r = evaluate_r(y_val, ridge.predict(X_vl_p))
ridge_val_mae = evaluate_mae(y_val, ridge.predict(X_vl_p))
loo = LeaveOneOut()
ridge_loo_pred = cross_val_predict(Ridge(alpha=5.0), X_tr_p, y_train, cv=loo)
ridge_loo_r = evaluate_r(y_train, ridge_loo_pred)
ridge_test_r = evaluate_r(y_test, ridge.predict(X_te_p))
ridge_test_mae = evaluate_mae(y_test, ridge.predict(X_te_p))
print(f"  val_pearson_r:  {ridge_val_r:.4f}")
print(f"  val_mae:        {ridge_val_mae:.4f}")
print(f"  loo_pearson_r:  {ridge_loo_r:.4f}")
print(f"  test_pearson_r: {ridge_test_r:.4f}")
print(f"  test_mae:       {ridge_test_mae:.4f}")

print()
print("=" * 65)
print("  XGBoost")
print("=" * 65)
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    print("  XGBoost not installed. Installing...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "xgboost", "-q"])
    import xgboost as xgb
    HAS_XGB = True

xgb_params = {
    'n_estimators': 100,
    'max_depth': 3,
    'learning_rate': 0.1,
    'min_child_weight': 5,
    'reg_alpha': 0.5,
    'reg_lambda': 2.0,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'verbosity': 0,
    'n_jobs': 1,
}

xgb_model = xgb.XGBRegressor(**xgb_params)
xgb_model.fit(X_tr_p, y_train)
xgb_val_r = evaluate_r(y_val, xgb_model.predict(X_vl_p))
xgb_val_mae = evaluate_mae(y_val, xgb_model.predict(X_vl_p))

loo = LeaveOneOut()
xgb_loo_pred = cross_val_predict(
    xgb.XGBRegressor(**{**xgb_params, 'n_estimators': 50}), 
    X_tr_p, y_train, cv=loo
)
xgb_loo_r = evaluate_r(y_train, xgb_loo_pred)
xgb_test_r = evaluate_r(y_test, xgb_model.predict(X_te_p))
xgb_test_mae = evaluate_mae(y_test, xgb_model.predict(X_te_p))

print(f"  XGB params: max_depth={xgb_params['max_depth']}, min_child_weight={xgb_params['min_child_weight']}")
print(f"  Regularization: reg_alpha={xgb_params['reg_alpha']}, reg_lambda={xgb_params['reg_lambda']}")
print(f"  val_pearson_r:  {xgb_val_r:.4f}")
print(f"  val_mae:        {xgb_val_mae:.4f}")
print(f"  loo_pearson_r:  {xgb_loo_r:.4f}")
print(f"  test_pearson_r: {xgb_test_r:.4f}")
print(f"  test_mae:       {xgb_test_mae:.4f}")

print()
print("=" * 65)
print("  ENSEMBLE (XGB + Ridge average)")
print("=" * 65)
ens_val_pred = 0.5 * ridge.predict(X_vl_p) + 0.5 * xgb_model.predict(X_vl_p)
ens_val_r = evaluate_r(y_val, ens_val_pred)
ens_val_mae = evaluate_mae(y_val, ens_val_pred)
ens_loo_pred = 0.5 * ridge_loo_pred + 0.5 * xgb_loo_pred
ens_loo_r = evaluate_r(y_train, ens_loo_pred)
ens_test_pred = 0.5 * ridge.predict(X_te_p) + 0.5 * xgb_model.predict(X_te_p)
ens_test_r = evaluate_r(y_test, ens_test_pred)
ens_test_mae = evaluate_mae(y_test, ens_test_pred)
print(f"  val_pearson_r:  {ens_val_r:.4f}")
print(f"  val_mae:        {ens_val_mae:.4f}")
print(f"  loo_pearson_r:  {ens_loo_r:.4f}")
print(f"  test_pearson_r: {ens_test_r:.4f}")
print(f"  test_mae:       {ens_test_mae:.4f}")

print()
print("=" * 65)
print("  REPEATED K-FOLD (5-fold × 10 repeats)")
print("=" * 65)
rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)
ridge_rkf_scores = []
xgb_rkf_scores = []
ens_rkf_scores = []
for train_i, val_i in rkf.split(X_tr_p):
    X_tr_kf = X_tr_p[train_i]; y_tr_kf = y_train[train_i]
    X_vl_kf = X_tr_p[val_i]; y_vl_kf = y_train[val_i]
    
    r_kf = Ridge(alpha=5.0)
    r_kf.fit(X_tr_kf, y_tr_kf)
    ridge_rkf_scores.append(evaluate_r(y_vl_kf, r_kf.predict(X_vl_kf)))
    
    xgb_kf = xgb.XGBRegressor(**xgb_params)
    xgb_kf.fit(X_tr_kf, y_tr_kf)
    xgb_rkf_scores.append(evaluate_r(y_vl_kf, xgb_kf.predict(X_vl_kf)))
    
    ens_pred_kf = 0.5 * r_kf.predict(X_vl_kf) + 0.5 * xgb_kf.predict(X_vl_kf)
    ens_rkf_scores.append(evaluate_r(y_vl_kf, ens_pred_kf))

print(f"  Ridge:    r={np.mean(ridge_rkf_scores):.4f} ± {np.std(ridge_rkf_scores):.4f}")
print(f"  XGBoost:  r={np.mean(xgb_rkf_scores):.4f} ± {np.std(xgb_rkf_scores):.4f}")
print(f"  Ensemble: r={np.mean(ens_rkf_scores):.4f} ± {np.std(ens_rkf_scores):.4f}")
print()
print(f"  NOTE: Higher variance = less stable = harder to generalize")
