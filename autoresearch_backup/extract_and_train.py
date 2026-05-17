#!/usr/bin/env python3
"""
extract_and_train.py — GEOCK AutoResearch
Uses fast-extracted features (96 compounds, 536D).
"""
import os, sys, json, pickle, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/chow/autoresearch")
import numpy as np
from pathlib import Path

CACHE = Path("CACHE_DIR / features_110.pkl")
from prepare import evaluate_r, evaluate_mae, FEATURE_NAMES
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import LeaveOneOut, cross_val_predict

print("="*65)
print("  LOADING FEATURES")
print("="*65)
with open(CACHE, "rb") as f:
    d = pickle.load(f)

X_raw  = d["X_raw"]   # (96, 24) physics
X_ecfp_list = d["X_ecfp"]  # list of arrays
y_all  = d["y_pkd"]
ids    = d["pdb_ids"]

# Handle ECFP — stored as object dtype array of numpy arrays
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

X_all = np.hstack([X_raw, X_ecfp])  # (96, 536)
N_TOTAL = len(y_all)
print(f"  {N_TOTAL} compounds, {X_all.shape[1]}D features")
print(f"  y range: {y_all.min():.2f} – {y_all.max():.2f} pKd")

# Feature subsets
PHYS  = list(range(0, 24))   # all physics (E1+E2+E3)
E1    = list(range(0, 6))    # E1 vinardo only
E4    = list(range(15, 24))  # E4 bio only
E1E4  = list(range(0, 6)) + list(range(15, 24))  # E1 + E4
E1E2  = list(range(0, 14))   # E1 + E2

def get_scaler(X, idx):
    mu = X[:, idx].mean(0)
    sd = X[:, idx].std(0)
    sd = np.where(sd == 0, 1, sd)
    return mu, sd

def run_exp(fi, model_type, alpha, k_sel):
    try:
        mu, sd = get_scaler(X_train, fi)
        Xtr = (X_train[:, fi] - mu) / sd
        Xvl = (X_val[:, fi] - mu) / sd
        Xte = (X_test[:, fi] - mu) / sd

        if k_sel and k_sel < len(fi):
            sel = SelectKBest(f_regression, k=k_sel)
            Xtr_p = sel.fit_transform(Xtr, y_train)
            Xvl_p = sel.transform(Xvl)
            Xte_p = sel.transform(Xte)
        else:
            Xtr_p = Xtr; Xvl_p = Xvl; Xte_p = Xte; sel = None

        if model_type == 'lasso':
            m = Lasso(alpha=alpha, max_iter=5000)
        elif model_type == 'ridge':
            m = Ridge(alpha=alpha)
        elif model_type == 'elasticnet':
            m = ElasticNet(alpha=alpha, l1_ratio=0.5, max_iter=5000)

        m.fit(Xtr_p, y_train)
        vr  = evaluate_r(y_val, m.predict(Xvl_p))
        vm  = evaluate_mae(y_val, m.predict(Xvl_p))
        tr  = evaluate_r(y_test, m.predict(Xte_p))
        tm  = evaluate_mae(y_test, m.predict(Xte_p))

        loo = LeaveOneOut()
        lp = cross_val_predict(
            Lasso(alpha=alpha, max_iter=5000) if model_type == 'lasso' else
            Ridge(alpha=alpha) if model_type == 'ridge' else
            ElasticNet(alpha=alpha, l1_ratio=0.5, max_iter=5000),
            Xtr_p, y_train, cv=loo)
        lr = evaluate_r(y_train, lp)
        return vr, vm, lr, tr, tm, sel
    except Exception as e:
        return None

# ── Split ─────────────────────────────────────────────────────────────
N_TEST = 5; N_VAL = 5
np.random.seed(42)
perm = np.random.permutation(N_TOTAL)
test_idx  = perm[:N_TEST]
val_idx   = perm[N_TEST:N_TEST+N_VAL]
train_idx = perm[N_TEST+N_VAL:]

X_train = X_all[train_idx]; y_train = y_all[train_idx]
X_val   = X_all[val_idx];  y_val   = y_all[val_idx]
X_test  = X_all[test_idx]; y_test  = y_all[test_idx]

print(f"\n  Train: {len(train_idx)}   Val: {len(val_idx)}   Test: {len(test_idx)}")
print(f"  Test PDBs: {[ids[i] for i in test_idx]}")
print(f"  Val PDBs:   {[ids[i] for i in val_idx]}")

# ── Model search ───────────────────────────────────────────────────
print("\n" + "="*65)
print("  MODEL SEARCH")
print("="*65)

configs = []
for fi_name, fi in [('PHYS', PHYS), ('E1', E1), ('E4', E4), ('E1E4', E1E4), ('E1E2', E1E2)]:
    for mt in ['lasso', 'ridge', 'elasticnet']:
        for a in [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]:
            for k in [3, 5, 7, 10, 15, 20, 'all']:
                configs.append((fi_name, fi, mt, a, k))

print(f"  Running {len(configs)} configurations...")
results = []
for i, (fi_name, fi, mt, a, k) in enumerate(configs):
    res = run_exp(fi, mt, a, k)
    if res:
        vr, vm, lr, tr, tm, sel = res
        results.append({'fi': fi_name, 'mt': mt, 'a': a, 'k': k,
                        'vr': vr, 'vm': vm, 'lr': lr, 'tr': tr, 'tm': tm})
    if (i+1) % 200 == 0:
        print(f"    [{i+1}/{len(configs)}] done...")

results.sort(key=lambda x: x['vr'], reverse=True)

print("\n  TOP 20 BY VAL-R:")
for r in results[:20]:
    print(f"  {r['fi']:6s} {r['mt']:10s} a={r['a']:6.4f} k={str(r['k']):4s}  "
          f"val={r['vr']:.4f} mae={r['vm']:.4f} loo={r['lr']:.4f} test={r['tr']:.4f}")

print("\n  TOP 20 BY LOO-R (honest):")
rl = sorted(results, key=lambda x: x['lr'], reverse=True)
for r in rl[:20]:
    print(f"  {r['fi']:6s} {r['mt']:10s} a={r['a']:6.4f} k={str(r['k']):4s}  "
          f"val={r['vr']:.4f} mae={r['vm']:.4f} loo={r['lr']:.4f} test={r['tr']:.4f}")

print("\n  TOP 20 BY TEST-R (external):")
rt2 = sorted(results, key=lambda x: x['tr'], reverse=True)
for r in rt2[:20]:
    print(f"  {r['fi']:6s} {r['mt']:10s} a={r['a']:6.4f} k={str(r['k']):4s}  "
          f"val={r['vr']:.4f} mae={r['vm']:.4f} loo={r['lr']:.4f} test={r['tr']:.4f}")

# ── Final model: best LOO that also has decent val ─────────────────
print("\n" + "="*65)
print("  FINAL MODEL SELECTION")
print("="*65)

# Pick best LOO with positive val
candidates = [r for r in rl if r['lr'] > 0.05 and r['vr'] > 0.0]
if candidates:
    best = candidates[0]
else:
    best = results[0]

fi_name = best['fi']
fi = {'PHYS': PHYS, 'E1': E1, 'E4': E4, 'E1E4': E1E4, 'E1E2': E1E2}[fi_name]
k = best['k']
mt = best['mt']
a  = best['a']

mu, sd = get_scaler(X_train, fi)
X_tr_s = (X_train[:, fi] - mu) / sd
X_vl_s = (X_val[:, fi] - mu) / sd
X_te_s = (X_test[:, fi] - mu) / sd

if k and k < len(fi):
    sel = SelectKBest(f_regression, k=k)
    X_tr_p = sel.fit_transform(X_tr_s, y_train)
    X_vl_p = sel.transform(X_vl_s)
    X_te_p = sel.transform(X_te_s)
    fnames = [FEATURE_NAMES[i] if i < len(FEATURE_NAMES) else f"ecfp{i-24}" for i in fi]
    selected = [fnames[i] for i, v in enumerate(sel.get_support()) if v]
else:
    X_tr_p = X_tr_s; X_vl_p = X_vl_s; X_te_p = X_te_s; sel = None
    selected = ["all"]

if mt == 'lasso':
    m = Lasso(alpha=a, max_iter=5000)
elif mt == 'ridge':
    m = Ridge(alpha=a)
else:
    m = ElasticNet(alpha=a, l1_ratio=0.5, max_iter=5000)
m.fit(X_tr_p, y_train)

vr = evaluate_r(y_val, m.predict(X_vl_p))
vm = evaluate_mae(y_val, m.predict(X_vl_p))
tr = evaluate_r(y_test, m.predict(X_te_p))
tm = evaluate_mae(y_test, m.predict(X_te_p))

loo = LeaveOneOut()
lp = cross_val_predict(
    Lasso(alpha=a, max_iter=5000) if mt == 'lasso' else
    Ridge(alpha=a) if mt == 'ridge' else
    ElasticNet(alpha=a, l1_ratio=0.5, max_iter=5000),
    X_tr_p, y_train, cv=loo)
lr = evaluate_r(y_train, lp)

print(f"\n  Model:      {mt} α={a}, k={k}, {fi_name} features")
print(f"  Selected:   {selected}")
print(f"  Train:      {len(train_idx)}   Val: {len(val_idx)}   Test: {len(test_idx)}")
print(f"  Test PDBs:  {[ids[i] for i in test_idx]}")
print(f"  Val PDBs:    {[ids[i] for i in val_idx]}")
print()
print(f"  val_pearson_r: {vr:.6f}")
print(f"  val_mae: {vm:.6f}")
print(f"  loo_pearson_r: {lr:.6f}")
print(f"  test_pearson_r: {tr:.6f}")
print(f"  test_mae: {tm:.6f}")
print()
print("="*65)
print("  VERDICT")
print("="*65)
print(f"  Dataset:     {N_TOTAL} compounds (96 with valid pocket+smiles)")
print(f"  Features:    {len(fi)}D → {k if k else len(fi)}D (SelectKBest)")
print(f"  Model:       {mt}")
print(f"  Split:       {len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} test")
