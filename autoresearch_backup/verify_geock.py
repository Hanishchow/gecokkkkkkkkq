"""
verify_geock.py — GEOCK Reality Check
=======================================
Tests whether the GradientBoosting CV R=0.823 is real or overfitted.

Uses leave-one-out cross-validation on the 10 training compounds
to estimate generalisation — more reliable for n=10 than 5-fold CV.

Then also reports performance on the 5 held-out TEST compounds
(which are completely unseen during any training or selection).

Run with: python verify_geock.py
"""

import sys, os, warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare import load_features, get_splits, evaluate_r, evaluate_mae, FEATURE_NAMES
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneOut, cross_val_predict
import numpy as np

# ── Config ──────────────────────────────────────────────────────────────────
FEATURE_SUBSET = slice(0, 24)   # physics only (no ECFP)
K_FEATURES     = 2
LASSO_ALPHA    = 0.015

# ── Load data ────────────────────────────────────────────────────────────────
data = load_features()
X_train, y_train, X_val, y_val, X_test, y_test = get_splits(data)

X_tr = X_train[:, FEATURE_SUBSET]
X_te = X_test[:, FEATURE_SUBSET]
X_vl = X_val[:, FEATURE_SUBSET]

# Normalise all splits using train stats (avoid leakage)
all_scaler = StandardScaler()
X_tr_s = all_scaler.fit_transform(X_tr)
X_vl_s = all_scaler.transform(X_vl)
X_te_s = all_scaler.transform(X_te)

# ── LASSO baseline ───────────────────────────────────────────────────────────
lasso = Lasso(alpha=LASSO_ALPHA, max_iter=5000)
lasso_selector = SelectKBest(f_regression, k=K_FEATURES)
X_tr_sel = lasso_selector.fit_transform(X_tr_s, y_train)
X_vl_sel = lasso_selector.transform(X_vl_s)
X_te_sel = lasso_selector.transform(X_te_s)

lasso.fit(X_tr_sel, y_train)
lasso_r_val,  lasso_mae_val  = evaluate_r(y_val,  lasso.predict(X_vl_sel)), evaluate_mae(y_val,  lasso.predict(X_vl_sel))
lasso_r_test, lasso_mae_test = evaluate_r(y_test, lasso.predict(X_te_sel)), evaluate_mae(y_test, lasso.predict(X_te_sel))

# ── LASSO LOO-CV (honest generalisation estimate on n=10) ───────────────────
loo = LeaveOneOut()
lasso_loo_preds = cross_val_predict(
    Lasso(alpha=LASSO_ALPHA, max_iter=5000),
    SelectKBest(f_regression, k=K_FEATURES).fit_transform(X_tr_s, y_train),
    y_train, cv=loo
)
lasso_r_loo = evaluate_r(y_train, lasso_loo_preds)

# ── GB pipeline ──────────────────────────────────────────────────────────────
gb_pipeline = Pipeline([
    ("selector", SelectKBest(f_regression, k=K_FEATURES)),
    ("model", GradientBoostingRegressor(
        n_estimators=50, max_depth=2, learning_rate=0.1,
        loss='absolute_error', random_state=42)),
])

gb_pipeline.fit(X_tr_s, y_train)
y_pred_val = gb_pipeline.predict(X_vl_s)
y_pred_test = gb_pipeline.predict(X_te_s)

gb_r_val,  gb_mae_val  = evaluate_r(y_val,  y_pred_val),  evaluate_mae(y_val,  y_pred_val)
gb_r_test, gb_mae_test = evaluate_r(y_test, y_pred_test), evaluate_mae(y_test, y_pred_test)

# ── GB LOO-CV ───────────────────────────────────────────────────────────────
gb_loo_preds = cross_val_predict(
    Pipeline([
        ("selector", SelectKBest(f_regression, k=K_FEATURES)),
        ("model", GradientBoostingRegressor(
            n_estimators=50, max_depth=2, learning_rate=0.1,
            loss='absolute_error', random_state=42)),
    ]),
    X_tr_s, y_train, cv=loo
)
gb_r_loo = evaluate_r(y_train, gb_loo_preds)

# ── Print results ────────────────────────────────────────────────────────────
print("=" * 62)
print("  GEOCK REALITY CHECK — RESULTS")
print("=" * 62)

print(f"\n  Dataset:  {len(y_train)} train / {len(y_val)} val / {len(y_test)} test  compounds")
print(f"  Features: {K_FEATURES} best from {FEATURE_SUBSET.stop} physics features")

print("\n" + "-" * 62)
print(f"  {'Model':<20}  {'LOO-CV R':>9}  {'Val R':>9}  {'Test R':>9}")
print("-" * 62)
print(f"  {'Lasso':<20}  {lasso_r_loo:>9.3f}  {lasso_r_val:>9.3f}  {lasso_r_test:>9.3f}")
print(f"  {'GradientBoosting':<20}  {gb_r_loo:>9.3f}  {gb_r_val:>9.3f}  {gb_r_test:>9.3f}")
print("-" * 62)

print(f"\n  MAE on held-out test set:")
print(f"    Lasso           : {lasso_mae_test:.3f}")
print(f"    GradientBoosting: {gb_mae_test:.3f}")

print(f"\n  Per-compound TEST predictions (GB):")
print(f"  {'Compound':>8}  {'True pKd':>9}  {'GB Pred':>9}  {'Error':>9}")
print(f"  {'--------':>8}  {'---------':>9}  {'-------':>9}  {'-----':>9}")
for i, (pid, true, pred) in enumerate(zip(data["pdb_ids"][-5:], y_test, y_pred_test)):
    print(f"  {pid:>8}  {true:>9.3f}  {pred:>9.3f}  {pred-true:>+9.3f}")

# ── Verdict ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("  VERDICT")
print("=" * 62)

test_r = gb_r_test
loo_r  = gb_r_loo

gap_cv_test = gb_r_val - gb_r_test   # CV inflation (val vs test)
gap_cv_loo  = gb_r_val - gb_r_loo    # CV inflation (val vs LOO)

print(f"\n  GB 5-fold CV R (from train.py): ~0.823")
print(f"  GB LOO-CV R  (honest):           {loo_r:.3f}")
print(f"  GB Val R      (from train.py):   {gb_r_val:.3f}")
print(f"  GB Test R     (completely new):  {test_r:.3f}")
print(f"  CV inflation vs LOO:             {gap_cv_loo:+.3f}")
print(f"  CV inflation vs Test:           {gap_cv_test:+.3f}")

if test_r > 0.75:
    print(f"\n  [REAL]   Test R={test_r:.3f} — GB generalises well.")
    print("           Publishable result.")
elif test_r > 0.55:
    print(f"\n  [PARTIAL] Test R={test_r:.3f} — Real signal, inflated CV.")
    print(f"            Honest estimate is LOO R={loo_r:.3f}.")
    print("            Use this number in the paper, not 0.823.")
else:
    print(f"\n  [OVERFIT] Test R={test_r:.3f} — GB memorised training set.")
    print("            Lasso (R={:.3f}) is more trustworthy.".format(lasso_r_test))

print("\n  Does GB beat Lasso on unseen data?")
delta = gb_r_test - lasso_r_test
print(f"    GB - Lasso = {delta:+.3f}  ({'GB wins' if delta > 0 else 'Lasso wins'})")
print("=" * 62)
