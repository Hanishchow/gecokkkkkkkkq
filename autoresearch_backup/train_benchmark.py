#!/usr/bin/env python3
"""
train_benchmark.py — Train on benchmark data (n=115+)
=====================================================
Uses features_benchmark.pkl (115 compounds from benchmark_pdbs/)
with the existing GEOCK-20 data to train and evaluate.
"""

import pickle, numpy as np, warnings, sys
warnings.filterwarnings("ignore")
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, Ridge, ElasticNet, BayesianRidge
from sklearn.model_selection import KFold, LeaveOneOut, cross_val_predict
from sklearn.feature_selection import SelectKBest, f_regression

# ── Load benchmark data ──────────────────────────────────────────────────────
CACHE_BENCH  = "CACHE_DIR / features_benchmark.pkl"
CACHE_V2     = "CACHE_DIR / features_v2.pkl"

bench = pickle.load(open(CACHE_BENCH, "rb"))
X_bench  = bench["X_raw"]     # (115, 24)
y_bench  = bench["y_pkd"]       # (115,)
pdb_ids  = bench["pdb_ids"]

# ── Load existing data to check overlap ────────────────────────────────────
existing = pickle.load(open(CACHE_V2, "rb"))
existing_pdbs = set(existing["pdb_ids"])

# Remove overlap
overlap = set(pdb_ids) & existing_pdbs
print(f"Benchmark: {len(pdb_ids)} compounds")
print(f"Existing GEOCK: {len(existing_pdbs)} compounds")
print(f"Overlap: {len(overlap)}")
if overlap:
    mask = [p not in overlap for p in pdb_ids]
    X_bench = X_bench[mask]
    y_bench = y_bench[mask]
    pdb_ids  = [p for p, m in zip(pdb_ids, mask) if m]
print(f"After removing overlap: {len(pdb_ids)} compounds")

# Replace NaN/Inf
X_bench = np.nan_to_num(X_bench, nan=0.0, posinf=0.0, neginf=0.0)

print(f"\nBenchmark pKd range: {y_bench.min():.2f} to {y_bench.max():.2f}")
print(f"Mean pKd: {y_bench.mean():.2f}")

# ── Feature names ────────────────────────────────────────────────────────────
FEATURE_NAMES = [
    "E1_vinardo_gauss1","E1_vinardo_repulsion","E1_vinardo_hydrophobic",
    "E1_vinardo_hbond","E1_vinardo_torsion","E1_vinardo_affinity",
    "E2_chem_pi_pi","E2_chem_cation_pi","E2_chem_salt_bridge",
    "E2_chem_halogen_bond","E2_chem_metal_coord","E2_chem_burial",
    "E2_chem_shape","E2_chem_lipophilic",
    "E3_quantum_vqe",
    "E4_bio_drug_likeness","E4_bio_ligand_efficiency",
    "E4_bio_pocket_druggability","E4_bio_resolution_weight",
    "E4_bio_family_hydrophobic","E4_bio_family_hbond",
    "E4_bio_pocket_polarity","E4_bio_size_penalty","E4_bio_pharmacophore",
]

# ── Feature subsets ─────────────────────────────────────────────────────────
SUBSETS = {
    "all_phys":   list(range(0, 24)),
    "E1_E2":      list(range(0, 14)),
    "E4_only":    list(range(15, 24)),
    "E4_top3":    [15, 16, 17, 18, 19, 20, 21, 22, 23],  # all E4, let SelectKBest pick
}

# ── Model configs ─────────────────────────────────────────────────────────────
MODELS = [
    ("Lasso_0.01",    Lasso,         {"alpha": 0.01, "max_iter": 5000}),
    ("Lasso_0.05",    Lasso,         {"alpha": 0.05, "max_iter": 5000}),
    ("Lasso_0.1",     Lasso,         {"alpha": 0.1,  "max_iter": 5000}),
    ("Ridge_1.0",     Ridge,         {"alpha": 1.0}),
    ("ElasticNet",     ElasticNet,    {"l1_ratio": 0.5, "alpha": 0.01, "max_iter": 5000}),
    ("BayesianRidge",  BayesianRidge, {}),
]

# ── Cross-validation ────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print(f"  BENCHMARK RESULTS (n={len(pdb_ids)}, LOO-CV + 5-fold CV)")
print("=" * 70)
print(f"\n{'Subset':<15}  {'Model':<18}  {'k':>3}  {'LOO':>7}  {'5-fold':>8}  {'Test':>7}  Notes")
print("-" * 80)

all_results = []

for sname, fidx in SUBSETS.items():
    Xf = X_bench[:, fidx]
    fnames = [FEATURE_NAMES[i] for i in fidx]
    
    for mname, mcls, mparams in MODELS:
        for k in [2, 3, 5]:
            if k >= len(fidx):
                continue
            
            try:
                sc = StandardScaler()
                Xf_s = sc.fit_transform(Xf)
                
                sel = SelectKBest(f_regression, k=min(k, len(fidx)))
                X_sel = sel.fit_transform(Xf_s, y_bench)
                
                m = mcls(**mparams)
                
                # LOO-CV
                loo = LeaveOneOut()
                lp = cross_val_predict(mcls(**mparams), X_sel, y_bench, cv=loo)
                loo_r = pearsonr(y_bench, lp)[0]
                
                # 5-fold CV
                kf = KFold(n_splits=5, shuffle=True, random_state=42)
                kp = cross_val_predict(mcls(**mparams), X_sel, y_bench, cv=kf)
                kf_r = pearsonr(y_bench, kp)[0]
                
                # Fit on all
                m.fit(X_sel, y_bench)
                pred_all = m.predict(X_sel)
                train_r = pearsonr(y_bench, pred_all)[0]
                
                selected_idx = sel.get_support().nonzero()[0]
                selected = [fnames[i] for i in selected_idx]
                
                result = dict(
                    subset=sname, model=mname, k=k,
                    loo_r=loo_r, kf_r=kf_r, train_r=train_r,
                    selected=selected
                )
                all_results.append(result)
                
                flag = "★" if loo_r > 0.5 else ("+" if loo_r > 0.3 else " ")
                print(f"  {sname:<15}  {mname:<18}  {k:>3}  {loo_r:>7.3f}  {kf_r:>8.3f}  {train_r:>7.3f}  {flag} {', '.join(selected[:3])}")
                
            except Exception as e:
                pass

# Sort by LOO
all_results.sort(key=lambda r: r["loo_r"], reverse=True)
best = all_results[0]

print("\n" + "=" * 70)
print(f"  BEST by LOO-CV")
print("=" * 70)
print(f"  Config:    {best['subset']} / {best['model']} / k={best['k']}")
print(f"  Features:  {best['selected']}")
print(f"  LOO-CV R:  {best['loo_r']:.4f}")
print(f"  5-fold R:  {best['kf_r']:.4f}")
print(f"  Train R:   {best['train_r']:.4f}")

# ── With VQE features ──────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  NOTE: E3_quantum_vqe was skipped (use_quantum=False)")
print("  Re-run extract_benchmark.py with use_quantum=True to include VQE")
print("=" * 70)
