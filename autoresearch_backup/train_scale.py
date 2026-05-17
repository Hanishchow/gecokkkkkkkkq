"""
train_scale.py — GEOCK Path 2: Lean Scale Pipeline
===================================================
Uses cached 20-compound data, tries all E4 subsets,
Lasso + BayesianRidge ensemble, honest LOO-CV reporting.

Ready to scale to PDBbind/CASF when data arrives:
  python train_scale.py --data /path/to/pdbbind_features.pkl
"""

import pickle, numpy as np, json, warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, BayesianRidge, ElasticNet
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.ensemble import GradientBoostingRegressor

sys_path = Path(__file__).parent
import sys
sys.path.insert(0, str(sys_path))

from prepare import load_features, get_splits, FEATURE_NAMES, evaluate_r

# E4 feature names (indices 15-23)
E4_IDX  = list(range(15, 24))
E4_NAMES = FEATURE_NAMES[15:24]

# Data paths
GEOCK_CACHE  = Path("CACHE_DIR / features_v2.pkl")
PDBBIND_CACHE = Path("/mnt/c/Users/yakka/Downloads/geock_pdbbind_data/features_pdbbind_v2.pkl")
CASF_CACHE  = Path("/mnt/c/Users/yakka/Downloads/geock_casf_data/features_casf.pkl")


def load_data():
    """Load whichever dataset is available."""
    if CASF_CACHE.exists():
        print(f"[DATA] CASF-2016: {CASF_CACHE}")
        with open(CASF_CACHE, "rb") as f:
            d = pickle.load(f)
        return d, "CASF-2016"
    elif PDBBIND_CACHE.exists():
        print(f"[DATA] PDBbind: {PDBBIND_CACHE}")
        with open(PDBBIND_CACHE, "rb") as f:
            d = pickle.load(f)
        return d, "PDBbind"
    else:
        print(f"[DATA] GEOCK cache: {GEOCK_CACHE}")
        d = pickle.load(open(GEOCK_CACHE, "rb"))
        return d, "GEOCK-20"


def make_splits(data):
    """Return train/val/test from whichever format."""
    if "X_train" in data:
        # PDBbind/CASF format
        return (data["X_train"], data["y_train"],
                data["X_val"],   data["y_val"],
                data["X_test"],  data["y_test"])
    else:
        # GEOCK format
        return get_splits(data)


def train_eval(X_tr, y_tr, X_vl, y_vl, X_te, y_te,
              feature_idx, k, model_cls, model_params, name):
    """Train one model, return metrics."""
    X_tr_f = X_tr[:, feature_idx]
    X_vl_f = X_vl[:, feature_idx]
    X_te_f = X_te[:, feature_idx]

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr_f)
    X_vl_s = scaler.transform(X_vl_f)
    X_te_s = scaler.transform(X_te_f)

    if k is not None and k < X_tr_s.shape[1]:
        sel = SelectKBest(f_regression, k=k)
        X_tr_s = sel.fit_transform(X_tr_s, y_tr)
        X_vl_s = sel.transform(X_vl_s)
        X_te_s = sel.transform(X_te_s)
        feat_names = [FEATURE_NAMES[feature_idx[i]] for i, s in enumerate(sel.get_support()) if s]
    else:
        feat_names = [FEATURE_NAMES[feature_idx[i]] for i in range(len(feature_idx))]

    model = model_cls(**model_params)
    model.fit(X_tr_s, y_tr)

    pred_tr = model.predict(X_tr_s)
    pred_vl = model.predict(X_vl_s)
    pred_te = model.predict(X_te_s)

    train_r = pearsonr(y_tr, pred_tr)[0]
    val_r   = pearsonr(y_vl, pred_vl)[0]
    test_r  = pearsonr(y_te, pred_te)[0]
    val_mae = np.mean(np.abs(y_vl - pred_vl))

    # LOO-CV on train
    try:
        loo = LeaveOneOut()
        loo_preds = cross_val_predict(
            model_cls(**model_params), X_tr_s, y_tr, cv=loo
        )
        loo_r = pearsonr(y_tr, loo_preds)[0]
    except:
        loo_r = float("nan")

    return {
        "name": name,
        "k": k or len(feat_names),
        "val_r": val_r, "val_mae": val_mae,
        "loo_r": loo_r, "test_r": test_r,
        "train_r": train_r,
        "features": feat_names,
        "model": model_cls.__name__,
    }


def ensemble_predict(X_tr, y_tr, X_vl, y_te, feature_idx, k, models):
    """Simple average ensemble of multiple models."""
    X_tr_f = X_tr[:, feature_idx]
    X_vl_f = X_vl[:, feature_idx]
    X_te_f = X_te[:, feature_idx]

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr_f)
    X_vl_s = scaler.transform(X_vl_f)
    X_te_s = scaler.transform(X_te_f)

    if k < X_tr_s.shape[1]:
        sel = SelectKBest(f_regression, k=k)
        X_tr_s = sel.fit_transform(X_tr_s, y_tr)
        X_vl_s = sel.transform(X_vl_s)
        X_te_s = sel.transform(X_te_s)

    preds_vl, preds_te = [], []
    for m_cls, m_params in models:
        m = m_cls(**m_params)
        m.fit(X_tr_s, y_tr)
        preds_vl.append(m.predict(X_vl_s))
        preds_te.append(m.predict(X_te_s))

    avg_vl = np.mean(preds_vl, axis=0)
    avg_te = np.mean(preds_te, axis=0)

    return pearsonr(y_vl, avg_vl)[0], pearsonr(y_te, avg_te)[0]


def main():
    data, source = load_data()
    X_train, y_train, X_val, y_val, X_test, y_test = make_splits(data)
    n_tr, n_vl, n_te = len(y_train), len(y_val), len(y_test)

    print(f"\n{'='*70}")
    print(f"  GEOCK PATH 2 — Lean Scale Pipeline ({source})")
    print(f"{'='*70}")
    print(f"  Splits: {n_tr} train / {n_vl} val / {n_te} test")
    print(f"  Features: {X_train.shape[1]} total (24 physics + 512 ECFP)")
    print()

    results = []

    # ── All feature subsets to try ─────────────────────────────────────────────
    subsets = {
        "E4_pocket":     (E4_IDX, [2, 3, 4]),
        "E2+E4":         ([i for i in range(6, 13)] + E4_IDX, [2, 3, 4]),
        "E1+E2+E4":      ([i for i in range(0, 13)] + E4_IDX, [2, 3, 4]),
        "all_physics":   (list(range(0, 24)), [2, 3, 4, 5, 6]),
        "E4+pocket_vol": ([i for i in range(6, 13)] + [15,16,17,18,19,20,21,22,23], [2, 3, 4]),
    }

    # ── Models ────────────────────────────────────────────────────────────────
    models = [
        ("Lasso", Lasso, {"alpha": 0.015, "max_iter": 5000}),
        ("Lasso", Lasso, {"alpha": 0.01,  "max_iter": 5000}),
        ("Lasso", Lasso, {"alpha": 0.005, "max_iter": 5000}),
        ("BayesianRidge", BayesianRidge, {}),
        ("ElasticNet", ElasticNet, {"l1_ratio": 0.3, "alpha": 0.01, "max_iter": 5000}),
    ]

    print("  Running experiments...")
    for subset_name, (fidx, k_vals) in subsets.items():
        for k in k_vals:
            for mname, mcls, mparams in models:
                r = train_eval(
                    X_train, y_train, X_val, y_val, X_test, y_test,
                    fidx, k, mcls, mparams,
                    f"{subset_name} {mname} k={k}"
                )
                results.append(r)

    # ── Sort by LOO-CV R ───────────────────────────────────────────────────
    results.sort(key=lambda r: (r["loo_r"] if r["loo_r"] == r["loo_r"] else -999), reverse=True)

    print(f"\n  TOP 15 by LOO-CV R:")
    print(f"  {'Subset':<20} {'Model':<15} {'k':>3}  {'Train':>7} {'LOO':>7} {'Val':>7} {'Test':>7}  Features")
    print(f"  {'-'*20} {'-'*15} {'-'*3}  {'-'*7} {'-'*7} {'-'*7} {'-'*7}  ---------")
    for r in results[:15]:
        flag = "★" if r["loo_r"] > 0.70 else ("+" if r["loo_r"] > 0.60 else " ")
        loo_str = f"{r['loo_r']:.3f}" if r["loo_r"] == r["loo_r"] else "  nan"
        feats_short = ", ".join(f.split("_")[-1] for f in r["features"][:2])
        print(f"  {r['name']:<50} {r['loo_r']:>7.3f} {r['val_r']:>7.3f} {r['test_r']:>7.3f} {r['val_mae']:>6.3f}  {feats_short}  {flag}")

    # ── Best ────────────────────────────────────────────────────────────────
    best = results[0]
    print(f"\n  BEST: {best['name']}")
    print(f"    Train R: {best['train_r']:.3f}")
    print(f"    LOO-CV:  {best['loo_r']:.3f}")
    print(f"    Val R:   {best['val_r']:.3f}")
    print(f"    Test R:  {best['test_r']:.3f}")
    print(f"    Val MAE: {best['val_mae']:.3f}")
    print(f"    Features: {best['features']}")

    # ── Write train.py with best ───────────────────────────────────────────
    feat_str = ", ".join(f"FEATURE_NAMES[{E4_IDX[0]}:{E4_IDX[-1]+1}]" if "E4" in best["name"] and best["k"] == 9
                          else str(best["features"][0]) if len(best["features"]) == 1
                          else str(best["features"]) for _ in [1])

    print(f"\n  NOTE: For Path 2 with PDBbind data, re-run after download.")
    print(f"  Download CASF-2016: https://doi.org/10.6084/m9.figshare.12368363")
    print(f"  Then run: python train_scale.py")


if __name__ == "__main__":
    main()
