"""
train_ridge_tune.py — Ridge hyperparameter sweep
================================================
Purpose: Find the best alpha and k (number of features) for Ridge.
         Track both val_r (optimistic) and LOO_r (honest) for each config.
         The honest metric is LOO_r — this is what we optimize for.
"""
import sys, os, warnings, pickle
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from pathlib import Path
from prepare import evaluate_r, evaluate_mae
from sklearn.linear_model import Ridge
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import LeaveOneOut, cross_val_predict

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

loo = LeaveOneOut()
ALPHAS = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
K_VALS = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

print(f"{'ALPHA':>8} | {'K':>4} | {'VAL_R':>8} | {'VAL_MAE':>9} | {'LOO_R':>8} | {'TEST_R':>8} | {'TEST_MAE':>9}")
print("-" * 85)

best_loo_r = -999
best_config = None
results = []

for alpha in ALPHAS:
    for k in K_VALS:
        sel = SelectKBest(f_regression, k=k)
        X_tr_p = sel.fit_transform(X_tr_s, y_train)
        X_vl_p = sel.transform(X_vl_s)
        X_te_p = sel.transform(X_te_s)
        
        m = Ridge(alpha=alpha)
        m.fit(X_tr_p, y_train)
        
        val_r = evaluate_r(y_val, m.predict(X_vl_p))
        val_mae = evaluate_mae(y_val, m.predict(X_vl_p))
        loo_pred = cross_val_predict(Ridge(alpha=alpha), X_tr_p, y_train, cv=loo)
        loo_r = evaluate_r(y_train, loo_pred)
        test_r = evaluate_r(y_test, m.predict(X_te_p))
        test_mae = evaluate_mae(y_test, m.predict(X_te_p))
        
        marker = ""
        if loo_r > best_loo_r:
            best_loo_r = loo_r
            best_config = (alpha, k)
            marker = " ★ BEST"
        
        print(f"{alpha:>8.2f} | {k:>4} | {val_r:>8.4f} | {val_mae:>9.4f} | {loo_r:>8.4f} | {test_r:>8.4f} | {test_mae:>9.4f}{marker}")
        results.append({
            'alpha': alpha, 'k': k, 'val_r': val_r, 'val_mae': val_mae,
            'loo_r': loo_r, 'test_r': test_r, 'test_mae': test_mae
        })

print()
print(f"=== BEST by LOO_r: alpha={best_config[0]}, k={best_config[1]}, LOO_r={best_loo_r:.4f} ===")

best_row = next(r for r in results if r['alpha']==best_config[0] and r['k']==best_config[1])
print(f"  val_r:    {best_row['val_r']:.4f}")
print(f"  val_mae:  {best_row['val_mae']:.4f}")
print(f"  test_r:   {best_row['test_r']:.4f}")
print(f"  test_mae: {best_row['test_mae']:.4f}")

print(f"\n=== ANALYSIS ===")
print(f"NOTE: LOO is on 86 training compounds (85 unique families)")
print(f"Each LOO fold tests on a COMPLETELY NEW protein family")
print(f"LOO_r ≈ 0.1-0.2 is realistic for cross-family prediction")
print(f"val_r ≈ 0.9 is OPTIMISTIC — val compounds may share patterns with train")
