"""
batch_train.py — GEOCK AutoResearch Batch Experiments
====================================================
Runs all experiments from program.md systematically.
Writes results to results.tsv.
"""

import sys, os, warnings, subprocess
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare import load_features, get_splits, evaluate_r, evaluate_mae, FEATURE_NAMES
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, ElasticNet, BayesianRidge
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import numpy as np

data = load_features()
X_train, y_train, X_val, y_val, X_test, y_test = get_splits(data)

X_tr = X_train[:, :24]
X_vl = X_val[:, :24]
X_te = X_test[:, :24]

scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_vl_s = scaler.transform(X_vl)
X_te_s = scaler.transform(X_te)

ALL_PHYSICS = list(range(24))
FEATURE_NAMES_SHORT = FEATURE_NAMES[:24]

def get_selected(selector, subset=ALL_PHYSICS):
    support = selector.get_support()
    return [FEATURE_NAMES_SHORT[subset[i]] for i, s in enumerate(support) if s]

def run_exp(name, model, X_tr_s, X_vl_s, X_te_s, y_train, y_val, y_test,
            feature_indices=None, selector=None, notes="", k=None, alpha=None,
            n_features=None):
    if feature_indices is not None:
        Xtr = X_tr_s[:, feature_indices]
        Xvl = X_vl_s[:, feature_indices]
        Xte = X_te_s[:, feature_indices]
    else:
        Xtr, Xvl, Xte = X_tr_s, X_vl_s, X_te_s

    if selector is not None:
        Xtr_sel = selector.fit_transform(Xtr, y_train)
        Xvl_sel = selector.transform(Xvl)
        Xte_sel = selector.transform(Xte)
    else:
        Xtr_sel, Xvl_sel, Xte_sel = Xtr, Xvl, Xte

    n_sel = Xtr_sel.shape[1]
    selected_names = ""
    if hasattr(selector, 'get_support'):
        sf = selector.get_support()
        fi = feature_indices if feature_indices is not None else list(range(X_tr_s.shape[1]))
        try:
            selected_names = ", ".join([FEATURE_NAMES_SHORT[fi[i]] for i, s in enumerate(sf) if s])
        except (IndexError, KeyError):
            selected_names = f"{n_sel} features (indices: {[fi[i] for i, s in enumerate(sf) if s][:5]}...)"

    model.fit(Xtr_sel, y_train)
    pred_val  = model.predict(Xvl_sel)
    pred_test = model.predict(Xte_sel)

    val_r   = evaluate_r(y_val, pred_val)
    val_mae = evaluate_mae(y_val, pred_val)
    test_r  = evaluate_r(y_test, pred_test)
    test_mae = evaluate_mae(y_test, pred_test)

    loo = LeaveOneOut()
    loo_r = float('nan')
    try:
        if isinstance(model, (Lasso, ElasticNet, BayesianRidge)):
            loo_preds = cross_val_predict(type(model)(**model.get_params()), Xtr_sel, y_train, cv=loo)
            loo_r = evaluate_r(y_train, loo_preds)
        elif isinstance(model, SVR):
            loo_preds = cross_val_predict(SVR(**model.get_params()), Xtr_sel, y_train, cv=loo)
            loo_r = evaluate_r(y_train, loo_preds)
        else:
            loo_r = val_r
    except Exception:
        loo_r = val_r

    return {
        "name": name,
        "val_r": val_r,
        "val_mae": val_mae,
        "test_r": test_r,
        "test_mae": test_mae,
        "loo_r": loo_r,
        "n_sel": n_sel,
        "selected": selected_names,
        "notes": notes,
    }

def try_clone_and_run(name, **kwargs):
    import random
    tag = name.replace(" ", "_").replace("=", "_").replace(".", "_")[:40]
    branch = f"autoresearch/exp_{tag}_{random.randint(100,999)}"
    try:
        subprocess.run(["git", "checkout", "-b", branch], capture_output=True, check=True)
    except:
        pass
    result = run_exp(name, **kwargs)
    try:
        subprocess.run(["git", "checkout", "-"], capture_output=True)
        subprocess.run(["git", "branch", "-D", branch], capture_output=True)
    except:
        pass
    return result

def build_results_table(results):
    print("\n" + "=" * 100)
    print("  GEOCK AUTORESEARCH — BATCH EXPERIMENT RESULTS")
    print("=" * 100)
    print(f"\n{'#':>3}  {'Experiment':<40} {'Val R':>7} {'LOO R':>7} {'Test R':>7} {'MAE':>6}  {'k':>3}  {'Notes'}")
    print("-" * 100)
    for i, r in enumerate(results):
        flag = ""
        if r["val_r"] > 0.70: flag = "★"
        elif r["val_r"] > 0.66: flag = "+"
        sel_short = r["selected"][:30] if r["selected"] else ""
        print(f"{i+1:>3}  {r['name']:<40} {r['val_r']:>7.3f} {r['loo_r']:>7.3f} {r['test_r']:>7.3f} {r['val_mae']:>6.3f}  {r['n_sel']:>3}  {sel_short}  {flag}")
    print("-" * 100)
    best = max(results, key=lambda r: r["val_r"])
    print(f"\nBest: {best['name']}  Val R={best['val_r']:.3f}  LOO={best['loo_r']:.3f}  Test={best['test_r']:.3f}")
    print(f"Features: {best['selected']}")
    return results.index(best)

print("=" * 80)
print("  GEOCK BATCH EXPERIMENTS")
print("=" * 80)
print(f"\nDataset: {len(y_train)} train / {len(y_val)} val / {len(y_test)} test")
print(f"Baseline (Lasso k=2 alpha=0.015): Val R≈0.660")

results = []

# ══════════════════════════════════════════════════════════════
# ROUND 1 — Feature Selection
# ══════════════════════════════════════════════════════════════
print("\n[ROUND 1] Feature Selection...")

selector_k2 = SelectKBest(f_regression, k=2)
r = run_exp(
    "Lasso k=2 baseline",
    Lasso(alpha=0.015, max_iter=5000),
    X_tr_s, X_vl_s, X_te_s, y_train, y_val, y_test,
    feature_indices=ALL_PHYSICS, selector=SelectKBest(f_regression, k=2),
    notes="baseline"
)
results.append(r)

selector_k3 = SelectKBest(f_regression, k=3)
r = run_exp(
    "Lasso k=3",
    Lasso(alpha=0.015, max_iter=5000),
    X_tr_s, X_vl_s, X_te_s, y_train, y_val, y_test,
    feature_indices=ALL_PHYSICS, selector=SelectKBest(f_regression, k=3),
    notes="k=3"
)
results.append(r)

selector_k4 = SelectKBest(f_regression, k=4)
r = run_exp(
    "Lasso k=4",
    Lasso(alpha=0.015, max_iter=5000),
    X_tr_s, X_vl_s, X_te_s, y_train, y_val, y_test,
    feature_indices=ALL_PHYSICS, selector=SelectKBest(f_regression, k=4),
    notes="k=4"
)
results.append(r)

selector_k5 = SelectKBest(f_regression, k=5)
r = run_exp(
    "Lasso k=5",
    Lasso(alpha=0.015, max_iter=5000),
    X_tr_s, X_vl_s, X_te_s, y_train, y_val, y_test,
    feature_indices=ALL_PHYSICS, selector=SelectKBest(f_regression, k=5),
    notes="k=5"
)
results.append(r)

selector_k6 = SelectKBest(f_regression, k=6)
r = run_exp(
    "Lasso k=6",
    Lasso(alpha=0.015, max_iter=5000),
    X_tr_s, X_vl_s, X_te_s, y_train, y_val, y_test,
    feature_indices=ALL_PHYSICS, selector=SelectKBest(f_regression, k=6),
    notes="k=6"
)
results.append(r)

# ══════════════════════════════════════════════════════════════
# Test quantum feature specifically (E3_quantum_vqe = index 13)
# ══════════════════════════════════════════════════════════════
vqe_idx = 13
for k in [2, 3]:
    combos = []
    others = [i for i in ALL_PHYSICS if i != vqe_idx]
    from itertools import combinations
    for combo in combinations(others, k-1):
        feat_idx = list(combo) + [vqe_idx]
        r = run_exp(
            f"Lasso k={k} + VQE",
            Lasso(alpha=0.015, max_iter=5000),
            X_tr_s, X_vl_s, X_te_s, y_train, y_val, y_test,
            feature_indices=sorted(feat_idx), selector=None,
            notes=f"k={k}+VQE",
            n_features=k
        )
        results.append(r)
        combos.append((r["val_r"], feat_idx, r))
    best_combo = max(combos, key=lambda x: x[0])

print(f"  Best k=2+VQE: Val R={best_combo[0]:.3f}")

# ══════════════════════════════════════════════════════════════
# ROUND 2 — Model Changes
# ══════════════════════════════════════════════════════════════
print("\n[ROUND 2] Model Changes...")

for k in [2, 3]:
    for kernel in ["rbf", "linear"]:
        r = run_exp(
            f"SVR {kernel} k={k}",
            SVR(kernel=kernel, C=1.0, epsilon=0.1),
            X_tr_s, X_vl_s, X_te_s, y_train, y_val, y_test,
            feature_indices=ALL_PHYSICS, selector=SelectKBest(f_regression, k=k),
            notes=f"model=SVR"
        )
        results.append(r)

for k in [2, 3]:
    r = run_exp(
        f"GaussianProcess k={k}",
        GaussianProcessRegressor(kernel=ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)),
        X_tr_s, X_vl_s, X_te_s, y_train, y_val, y_test,
        feature_indices=ALL_PHYSICS, selector=SelectKBest(f_regression, k=k),
        notes="model=GP"
    )
    results.append(r)

for k in [2, 3]:
    r = run_exp(
        f"BayesianRidge k={k}",
        BayesianRidge(),
        X_tr_s, X_vl_s, X_te_s, y_train, y_val, y_test,
        feature_indices=ALL_PHYSICS, selector=SelectKBest(f_regression, k=k),
        notes="model=BayesianRidge"
    )
    results.append(r)

# ══════════════════════════════════════════════════════════════
# ROUND 3 — Feature Engineering
# ══════════════════════════════════════════════════════════════
print("\n[ROUND 3] Feature Engineering...")

X_tr_log = X_tr.copy()
X_vl_log = X_vl.copy()
X_te_log = X_te.copy()
X_tr_log[:, 13] = np.log(np.abs(X_tr[:, 13]) + 1)
X_vl_log[:, 13] = np.log(np.abs(X_vl[:, 13]) + 1)
X_te_log[:, 13] = np.log(np.abs(X_te[:, 13]) + 1)

scaler3 = StandardScaler()
X_tr_log_s = scaler3.fit_transform(X_tr_log)
X_vl_log_s = scaler3.transform(X_vl_log)
X_te_log_s = scaler3.transform(X_te_log)

for k in [2, 3]:
    r = run_exp(
        f"Lasso k={k} log(VQE)",
        Lasso(alpha=0.015, max_iter=5000),
        X_tr_log_s, X_vl_log_s, X_te_log_s, y_train, y_val, y_test,
        feature_indices=ALL_PHYSICS, selector=SelectKBest(f_regression, k=k),
        notes="log(VQE)"
    )
    results.append(r)

X_tr_inter = np.hstack([X_tr, X_tr[:, 12:13] * X_tr[:, 22:23]])
X_vl_inter = np.hstack([X_vl, X_vl[:, 12:13] * X_vl[:, 22:23]])
X_te_inter = np.hstack([X_te, X_te[:, 12:13] * X_te[:, 22:23]])

scaler_inter = StandardScaler()
X_tr_int_s = scaler_inter.fit_transform(X_tr_inter)
X_vl_int_s = scaler_inter.transform(X_vl_inter)
X_te_int_s = scaler_inter.transform(X_te_inter)

for k in [2, 3]:
    r = run_exp(
        f"Lasso k={k} interaction",
        Lasso(alpha=0.015, max_iter=5000),
        X_tr_int_s, X_vl_int_s, X_te_int_s, y_train, y_val, y_test,
        feature_indices=ALL_PHYSICS, selector=SelectKBest(f_regression, k=k),
        notes="interaction_terms"
    )
    results.append(r)

# PCA on ECFP
for n_pca in [3, 5]:
    X_ecfp_tr = X_train[:, 24:]
    X_ecfp_vl = X_val[:, 24:]
    X_ecfp_te = X_test[:, 24:]

    pca = PCA(n_components=n_pca)
    pca_scaler = StandardScaler()
    X_pca_tr = pca_scaler.fit_transform(X_ecfp_tr)
    X_pca_vl = pca_scaler.transform(X_ecfp_vl)
    X_pca_te = pca_scaler.transform(X_ecfp_te)

    X_combined_tr = np.hstack([X_tr_s, X_pca_tr])
    X_combined_vl = np.hstack([X_vl_s, X_pca_vl])
    X_combined_te = np.hstack([X_te_s, X_pca_te])

    for k in [2, 3]:
        r = run_exp(
            f"Lasso k={k} PCA{n_pca}",
            Lasso(alpha=0.015, max_iter=5000),
            X_combined_tr, X_combined_vl, X_combined_te, y_train, y_val, y_test,
            feature_indices=list(range(24)) + list(range(24, 24 + n_pca)),
            selector=SelectKBest(f_regression, k=k),
            notes=f"PCA_ecfp={n_pca}"
        )
        results.append(r)

# ══════════════════════════════════════════════════════════════
# ROUND 4 — Hyperparameter Tuning
# ══════════════════════════════════════════════════════════════
print("\n[ROUND 4] Hyperparameter Tuning...")

for alpha in [0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1]:
    r = run_exp(
        f"Lasso alpha={alpha} k=2",
        Lasso(alpha=alpha, max_iter=5000),
        X_tr_s, X_vl_s, X_te_s, y_train, y_val, y_test,
        feature_indices=ALL_PHYSICS, selector=SelectKBest(f_regression, k=2),
        notes=f"alpha_tuning"
    )
    results.append(r)

for l1 in [0.1, 0.3, 0.5, 0.7, 0.9]:
    r = run_exp(
        f"ElasticNet l1={l1} k=2",
        ElasticNet(l1_ratio=l1, alpha=0.015, max_iter=5000),
        X_tr_s, X_vl_s, X_te_s, y_train, y_val, y_test,
        feature_indices=ALL_PHYSICS, selector=SelectKBest(f_regression, k=2),
        notes=f"ElasticNet"
    )
    results.append(r)

# ══════════════════════════════════════════════════════════════
# ROUND 5 — Feature Subset Exploration
# ══════════════════════════════════════════════════════════════
print("\n[ROUND 5] Feature Subsets...")

subsets = {
    "physics_only": list(range(24)),
    "no_quantum": [i for i in range(24) if i != 13],
    "E1_only": list(range(0, 6)),
    "E2_only": list(range(6, 13)),
    "E4_only": list(range(15, 24)),
    "E1+E2": list(range(0, 13)),
    "E2+E4": list(range(6, 13)) + list(range(15, 24)),
}

for subset_name, indices in subsets.items():
    if len(indices) < 2:
        continue
    for k in [2, min(3, len(indices))]:
        if k > len(indices):
            continue
        scaler_sub = StandardScaler()
        Xtr_sub = scaler_sub.fit_transform(X_train[:, indices])
        Xvl_sub = scaler_sub.transform(X_val[:, indices])
        Xte_sub = scaler_sub.transform(X_test[:, indices])

        r = run_exp(
            f"Lasso {subset_name} k={k}",
            Lasso(alpha=0.015, max_iter=5000),
            Xtr_sub, Xvl_sub, Xte_sub, y_train, y_val, y_test,
            feature_indices=list(range(len(indices))), selector=SelectKBest(f_regression, k=k),
            notes=f"subset={subset_name}"
        )
        results.append(r)

# ══════════════════════════════════════════════════════════════
# RESULTS TABLE
# ══════════════════════════════════════════════════════════════
best_idx = build_results_table(results)

# ══════════════════════════════════════════════════════════════
# WRITE RESULTS.TSV
# ══════════════════════════════════════════════════════════════
with open("results.tsv", "w") as f:
    f.write("experiment_id\tdescription\tval_pearson_r\tval_mae\tloo_r\ttest_r\tn_features\tselected\tnotes\n")
    for i, r in enumerate(results):
        eid = f"exp_{i+1:03d}"
        desc = r["name"]
        f.write(f"{eid}\t{desc}\t{r['val_r']:.6f}\t{r['val_mae']:.6f}\t{r['loo_r']:.6f}\t{r['test_r']:.6f}\t{r['n_sel']}\t{r['selected']}\t{r['notes']}\n")

print(f"\n[WROTE] results.tsv ({len(results)} experiments)")

# ══════════════════════════════════════════════════════════════
# UPDATE TRAIN.PY with best
# ══════════════════════════════════════════════════════════════
best = results[best_idx]
train_content = f'''"""
train.py — GEOCK AutoResearch
Experiment: {best['name']}
Selected: {best['selected']}
"""

import sys, os, warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare import load_features, get_splits, evaluate_r, evaluate_mae, FEATURE_NAMES
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectKBest, f_regression
import numpy as np

data = load_features()
X_train, y_train, X_val, y_val, X_test, y_test = get_splits(data)

FEATURE_SUBSET = slice(0, 24)
K_FEATURES = {best['n_sel']}
LASSO_ALPHA = 0.015

X_tr = X_train[:, FEATURE_SUBSET]
X_vl = X_val[:, FEATURE_SUBSET]

scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_vl_s = scaler.transform(X_vl)

selector = SelectKBest(f_regression, k=K_FEATURES)
X_tr_sel = selector.fit_transform(X_tr_s, y_train)
X_vl_sel = selector.transform(X_vl_s)

model = Lasso(alpha=LASSO_ALPHA, max_iter=5000)
model.fit(X_tr_sel, y_train)

y_pred_val = model.predict(X_vl_sel)
y_pred_train = model.predict(X_tr_sel)

val_r   = evaluate_r(y_val, y_pred_val)
val_mae = evaluate_mae(y_val, y_pred_val)
train_r = evaluate_r(y_train, y_pred_train)

support = selector.get_support()
feat_slice = list(range(*FEATURE_SUBSET.indices(536)))
selected = [FEATURE_NAMES[feat_slice[i]] for i, s in enumerate(support) if s]
print(f"Selected: {selected}")

print(f"train_pearson_r: {train_r:.6f}")
print(f"val_pearson_r: {val_r:.6f}")
print(f"val_mae: {val_mae:.6f}")
'''

with open("train.py", "w") as f:
    f.write(train_content)

print(f"\n[UPDATED] train.py with best experiment")
print(f"   Best: {best['name']}  Val R={best['val_r']:.3f}  LOO={best['loo_r']:.3f}  Test={best['test_r']:.3f}")
