#!/usr/bin/env python3
"""
train_with_bitcount.py — Add bit count feature to XGBoost model

Simplified version: use K-fold instead of LOO-CV for speed.
"""
import pickle
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import KFold
# Import path helpers
try:
    from geock_paths import get_cache_dir, get_work_dir, CACHE_DIR, WORK_DIR
except ImportError:
    from pathlib import Path; CACHE_DIR = Path("/home/chow/.cache/geock_autoresearch"); WORK_DIR = Path("/home/chow/autoresearch")

print("=" * 70)
print("  TRAINING XGBOOST WITH BIT COUNT FEATURE")
print("=" * 70)

# Load data
cache = CACHE_DIR / lp_new_features_8k.pkl')
with open(cache, 'rb') as f:
    compounds = pickle.load(f)

print(f"  Loaded {len(compounds)} compounds")

# Build feature matrix: ECFP + bit count
X_list = []
y_list = []
bit_counts = []

for c in compounds:
    ecfp = np.array(c['ecfp'], dtype=np.float32)
    bc = np.array([ecfp.sum()], dtype=np.float32)
    X = np.concatenate([ecfp, bc])
    X_list.append(X)
    y_list.append(c['affinity'])
    bit_counts.append(ecfp.sum())

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.float32)
bit_counts = np.array(bit_counts, dtype=np.float32)

print(f"  Feature matrix: {X.shape}")
print(f"  y range: {y.min():.2f} — {y.max():.2f} pKd")
print(f"  Bit count range: {bit_counts.min():.0f} — {bit_counts.max():.0f}")

r_bc, _ = pearsonr(bit_counts, y)
print(f"  Bit count ↔ Affinity correlation: r = {r_bc:.4f}")

ECFP_IDX = list(range(512))
BC_IDX = [512]

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

print(f"\n  Split: {n_train} train / {n_val} val / {n_test} test")

# Standardize
mu_train = X_train.mean(0)
sd_train = X_train.std(0)
sd_train[sd_train == 0] = 1
X_train_s = (X_train - mu_train) / sd_train
X_val_s = (X_val - mu_train) / sd_train
X_test_s = (X_test - mu_train) / sd_train

# Feature selection
k = 400
sel = SelectKBest(f_regression, k=k)
X_train_ecfp = sel.fit_transform(X_train_s[:, ECFP_IDX], y_train)
X_val_ecfp = sel.transform(X_val_s[:, ECFP_IDX])
X_test_ecfp = sel.transform(X_test_s[:, ECFP_IDX])

print(f"  Selected {k} ECFP features")

# With bit count
X_train_full = np.hstack([X_train_ecfp, X_train_s[:, BC_IDX]])
X_val_full = np.hstack([X_val_ecfp, X_val_s[:, BC_IDX]])
X_test_full = np.hstack([X_test_ecfp, X_test_s[:, BC_IDX]])

# Without bit count
X_train_no_bc = X_train_ecfp
X_val_no_bc = X_val_ecfp
X_test_no_bc = X_test_ecfp

print(f"  Full features: {k} ECFP + 1 bit count = {X_train_full.shape[1]}")

# XGBoost
try:
    import xgboost as xgb
except ImportError:
    import subprocess
    subprocess.run(['pip', 'install', 'xgboost', '-q'])
    import xgboost as xgb

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

print("\n" + "=" * 70)
print("  XGBOOST MODEL WITH BIT COUNT")
print("=" * 70)

model_bc = xgb.XGBRegressor(**xgb_params)
model_bc.fit(X_train_full, y_train)

val_r_bc = pearsonr(y_val, model_bc.predict(X_val_full))[0]
test_r_bc = pearsonr(y_test, model_bc.predict(X_test_full))[0]

# 5-fold CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_r_bc = []
for tr_idx, vl_idx in kf.split(X_train_full):
    m = xgb.XGBRegressor(**xgb_params)
    m.fit(X_train_full[tr_idx], y_train[tr_idx])
    cv_r_bc.append(pearsonr(y_train[vl_idx], m.predict(X_train_full[vl_idx]))[0])
cv_r_bc = np.mean(cv_r_bc)

print(f"  Val R:   {val_r_bc:.4f}")
print(f"  Test R:  {test_r_bc:.4f}")
print(f"  5-Fold CV R: {cv_r_bc:.4f}")

print("\n" + "=" * 70)
print("  WITHOUT BIT COUNT")
print("=" * 70)

model_no_bc = xgb.XGBRegressor(**xgb_params)
model_no_bc.fit(X_train_no_bc, y_train)

val_r_no_bc = pearsonr(y_val, model_no_bc.predict(X_val_no_bc))[0]
test_r_no_bc = pearsonr(y_test, model_no_bc.predict(X_test_no_bc))[0]

cv_r_no_bc = []
for tr_idx, vl_idx in kf.split(X_train_no_bc):
    m = xgb.XGBRegressor(**xgb_params)
    m.fit(X_train_no_bc[tr_idx], y_train[tr_idx])
    cv_r_no_bc.append(pearsonr(y_train[vl_idx], m.predict(X_train_no_bc[vl_idx]))[0])
cv_r_no_bc = np.mean(cv_r_no_bc)

print(f"  Val R:   {val_r_no_bc:.4f}")
print(f"  Test R:  {test_r_no_bc:.4f}")
print(f"  5-Fold CV R: {cv_r_no_bc:.4f}")

print("\n" + "=" * 70)
print("  RESULTS SUMMARY")
print("=" * 70)
print(f"  {'Metric':<10} {'With Bit Count':>15} {'Without':>15} {'Delta':>12}")
print(f"  {'-'*52}")
print(f"  {'Val R':<10} {val_r_bc:>15.4f} {val_r_no_bc:>15.4f} {val_r_bc - val_r_no_bc:>+12.4f}")
print(f"  {'Test R':<10} {test_r_bc:>15.4f} {test_r_no_bc:>15.4f} {test_r_bc - test_r_no_bc:>+12.4f}")
print(f"  {'5-Fold CV':<10} {cv_r_bc:>15.4f} {cv_r_no_bc:>15.4f} {cv_r_bc - cv_r_no_bc:>+12.4f}")

# Save model
out = {
    'model_bc': model_bc,
    'model_no_bc': model_no_bc,
    'sel': sel,
    'mu': mu_train,
    'sd': sd_train,
    'k': k,
    'with_bit_count': True,
    'cv_r_bc': cv_r_bc,
    'cv_r_no_bc': cv_r_no_bc,
    'val_r_bc': val_r_bc,
    'val_r_no_bc': val_r_no_bc,
    'test_r_bc': test_r_bc,
    'test_r_no_bc': test_r_no_bc,
}

out_path = WORK_DIR / geock_model_bitcount.pkl')
with open(out_path, 'wb') as f:
    pickle.dump(out, f)

print(f"\n  Saved to {out_path}")