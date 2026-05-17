"""
augment_train.py — GEOCK Path 2: Smart Augmentation Pipeline
==========================================================
Uses cached features directly (~3 seconds total).
Noise augmentation to expand n=10 training set.
Runs all feature subsets + models, LOO-CV for model selection.
"""

import pickle, numpy as np, warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, BayesianRidge, ElasticNet
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import LeaveOneOut, cross_val_predict

WORKSPACE = Path("/home/chow/autoresearch")
import sys
sys.path.insert(0, str(WORKSPACE))

from prepare import get_splits, FEATURE_NAMES

FEATURE_NAMES_SHORT = FEATURE_NAMES[:24]

AUG_COPIES = 5
AUG_NOISE  = 0.05
N_BOOTSTRAP = 10

FEATURE_SUBSETS = {
    "E4_only":      list(range(15, 24)),
    "E2_E4":        list(range(6, 13)) + list(range(15, 24)),
    "E1_E2_E4":     list(range(0, 13)) + list(range(15, 24)),
    "all_physics":  list(range(0, 24)),
    "E4_top3":      [15, 17, 21],
}

MODELS = [
    ("Lasso_0.005",    Lasso,         {"alpha": 0.005, "max_iter": 5000}),
    ("Lasso_0.01",     Lasso,         {"alpha": 0.01,  "max_iter": 5000}),
    ("Lasso_0.015",    Lasso,         {"alpha": 0.015, "max_iter": 5000}),
    ("Lasso_0.03",     Lasso,         {"alpha": 0.03,  "max_iter": 5000}),
    ("ElasticNet_0.5", ElasticNet,    {"l1_ratio": 0.5, "alpha": 0.01, "max_iter": 5000}),
    ("ElasticNet_0.3", ElasticNet,    {"l1_ratio": 0.3, "alpha": 0.01, "max_iter": 5000}),
    ("BayesianRidge",  BayesianRidge, {}),
]


def add_noise(X, sigma=0.05, n_copies=5):
    X_aug = [X]
    for _ in range(n_copies):
        X_aug.append(X + np.random.randn(*X.shape) * sigma)
    return np.vstack(X_aug)


def run_exp(X_tr, y_tr, X_vl, y_vl, X_te, y_te,
            fidx, k, mcls, mparams, name):
    X_tr_f = X_tr[:, fidx]
    X_vl_f = X_vl[:, fidx]
    X_te_f = X_te[:, fidx]

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr_f)
    X_vl_s = scaler.transform(X_vl_f)
    X_te_s = scaler.transform(X_te_f)

    if k is not None and k < X_tr_s.shape[1]:
        sel = SelectKBest(f_regression, k=k)
        X_tr_s = sel.fit_transform(X_tr_s, y_tr)
        X_vl_s = sel.transform(X_vl_s)
        X_te_s = sel.transform(X_te_s)

    model = mcls(**mparams)
    model.fit(X_tr_s, y_tr)

    pred_vl = model.predict(X_vl_s)
    pred_te = model.predict(X_te_s)
    val_r   = pearsonr(y_vl, pred_vl)[0]
    test_r  = pearsonr(y_te, pred_te)[0]
    val_mae = np.mean(np.abs(y_vl - pred_vl))

    loo = LeaveOneOut()
    try:
        lp = cross_val_predict(mcls(**mparams), X_tr_s, y_tr, cv=loo)
        loo_r = pearsonr(y_tr, lp)[0]
    except:
        loo_r = float("nan")

    return dict(name=name, k=k, val_r=val_r, loo_r=loo_r,
                test_r=test_r, val_mae=val_mae)


def main():
    print("=" * 70)
    print("  GEOCK PATH 2 — Smart Augmentation Pipeline")
    print("=" * 70)

    data = pickle.load(open("CACHE_DIR / features_v2.pkl", "rb"))
    X_train, y_train, X_val, y_val, X_test, y_test = get_splits(data)
    print(f"\nData: {len(y_train)} train / {len(y_val)} val / {len(y_test)} test")
    print(f"Features: {X_train.shape[1]} (24 physics + 512 ECFP)")

    all_results = []

    # ── Baseline: no augmentation ──────────────────────────────────────────
    print("\n[Baseline]")
    for sname, fidx in FEATURE_SUBSETS.items():
        for k in [2, 3, min(4, len(fidx))]:
            for mname, mcls, mparams in MODELS:
                r = run_exp(X_train, y_train, X_val, y_val, X_test, y_test,
                            fidx, k, mcls, mparams, f"{sname} {mname} k={k}")
                all_results.append(r)

    # ── Noise augmentation ──────────────────────────────────────────────────
    print(f"[Augmented x{AUG_COPIES+1}]")
    for sname, fidx in FEATURE_SUBSETS.items():
        X_ph = X_train[:, fidx]
        X_aug = add_noise(X_ph, sigma=AUG_NOISE, n_copies=AUG_COPIES)
        y_aug = np.concatenate([y_train] * (AUG_COPIES + 1))

        X_vl_f = X_val[:, fidx]
        X_te_f = X_test[:, fidx]
        local_fidx = list(range(len(fidx)))

        for k in [2, 3, min(4, len(fidx))]:
            for mname, mcls, mparams in MODELS:
                r = run_exp(X_aug, y_aug, X_vl_f, y_val, X_te_f, y_test,
                            local_fidx, k, mcls, mparams,
                            f"{sname} {mname} k={k} [AUG]")
                all_results.append(r)

    # ── Bootstrap stability ───────────────────────────────────────────────────
    print(f"\n[Bootstrap coef stability (k=3, n={N_BOOTSTRAP})]")
    np.random.seed(42)
    for sname, fidx in FEATURE_SUBSETS.items():
        stds = []
        for _ in range(N_BOOTSTRAP):
            idx = np.random.randint(0, len(X_train), len(X_train))
            Xb = X_train[idx][:, fidx]
            yb = y_train[idx]
            sc = StandardScaler()
            Xb_s = sc.fit_transform(Xb)
            if 3 < Xb_s.shape[1]:
                sl = SelectKBest(f_regression, k=3)
                Xb_s = sl.fit_transform(Xb_s, yb)
            m = Lasso(alpha=0.015, max_iter=5000)
            m.fit(Xb_s, yb)
            stds.append(np.std(m.coef_))
        print(f"  {sname:15s}: {np.mean(stds):.4f} avg coef std")

    # ── Results table ────────────────────────────────────────────────────────
    all_results.sort(key=lambda r: r["val_r"], reverse=True)

    print("\n" + "=" * 80)
    print("  ALL RESULTS (top 30 by Val R)")
    print("=" * 80)
    print(f"\n{'#':>3}  {'Model':<46}  {'k':>3}  {'Val':>7}  {'LOO':>7}  {'Test':>7}  {'MAE':>6}")
    print("-" * 80)
    for i, r in enumerate(all_results[:30]):
        loo_s = f"{r['loo_r']:.3f}" if r["loo_r"] == r["loo_r"] else "  nan"
        flag = "★" if r["val_r"] > 0.70 else ("+" if r["val_r"] > 0.66 else " ")
        aug = "[AUG]" if "[AUG]" in r["name"] else "     "
        print(f"{i+1:>3}  {r['name']:<46}  {r['k']:>3}  {r['val_r']:>7.3f}  {loo_s:>7}  {r['test_r']:>7.3f}  {r['val_mae']:>6.3f}  {aug} {flag}")

    # ── Best by LOO ─────────────────────────────────────────────────────────
    valid = [r for r in all_results if r["loo_r"] == r["loo_r"]]
    valid.sort(key=lambda r: r["loo_r"], reverse=True)

    print("\n" + "=" * 80)
    print("  TOP 10 BY LOO-CV (honest)")
    print("=" * 80)
    for r in valid[:10]:
        flag = "★" if r["val_r"] > 0.70 else ("+" if r["val_r"] > 0.66 else " ")
        print(f"  LOO={r['loo_r']:.3f}  Val={r['val_r']:.3f}  Test={r['test_r']:.3f}  {r['name']}  {flag}")

    best = valid[0]
    print(f"\n  BEST: {best['name']}")
    print(f"    Val R:  {best['val_r']:.3f}")
    print(f"    LOO-CV: {best['loo_r']:.3f}")
    print(f"    Test R: {best['test_r']:.3f}")
    print(f"    Val MAE: {best['val_mae']:.3f}")

    # ── Augmentation effect ──────────────────────────────────────────────────
    no  = [r for r in all_results if "[AUG]" not in r["name"]]
    yes = [r for r in all_results if "[AUG]" in r["name"]]
    no_loo  = np.mean([r["loo_r"] for r in no  if r["loo_r"] == r["loo_r"]])
    yes_loo = np.mean([r["loo_r"] for r in yes if r["loo_r"] == r["loo_r"]])
    print(f"\n  Augmentation delta LOO: {yes_loo - no_loo:+.3f}")
    print(f"  ({'helps' if yes_loo > no_loo else 'hurts'} — "
          f"{'use it' if abs(yes_loo - no_loo) > 0.02 else 'marginal'})")


if __name__ == "__main__":
    main()
