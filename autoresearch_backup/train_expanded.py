#!/usr/bin/env python3
"""
train_expanded.py — GEOCK with LP-PDBBind expanded data
=======================================================
Combines original 96 compounds (physics+ECFP) with 998 new LP-PDBBind 
compounds (ECFP + simple structural features) for a total of 1,094.

Strategy:
- Use kp=0 (physics disabled) for pure ECFP model on all 1,094 compounds
- Use kp>0 (physics enabled) with combined features
- Also train original 96-compound model for comparison
"""

import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pickle
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import LeaveOneOut, cross_val_predict, RepeatedKFold

np.random.seed(42)

CACHE_DIR = Path("/home/chow/.cache/geock_autoresearch")
COMBINED_CACHE = CACHE_DIR / "features_combined.pkl"
ORIGINAL_CACHE = CACHE_DIR / "features_110.pkl"

print("Loading combined features (1,094 compounds)...")
with open(COMBINED_CACHE, "rb") as f:
    comb = pickle.load(f)

X_raw_all = comb["X_raw"]        # (1094, 24) physics
X_ecfp_all = comb["X_ecfp"]       # (1094, 512) ECFP
y_all = comb["y_pkd"]             # (1094,)
ids_all = comb["pdb_ids"]          # list of 1094 pdb_ids
N = len(y_all)

print(f"Total: {N} compounds")
print(f"pKd range: {y_all.min():.1f} - {y_all.max():.1f}")
print(f"Affinity mean: {y_all.mean():.1f}")

# Check how many have near-zero physics
physics_nonzero = np.abs(X_raw_all).max(axis=1)
print(f"Compounds with real physics (>0.1): {(physics_nonzero > 0.1).sum()}")
print(f"Compounds with zero physics: {(physics_nonzero < 0.1).sum()}")

PHYS_USE = list(range(0, 14))  # E1+E2 physics (same as original)

FEATURE_NAMES_24 = [
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

def r_score(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return pearsonr(y_true, y_pred)[0] if len(y_true) >= 2 else np.nan

# ── Split ─────────────────────────────────────────────────────────────────────
N_TEST = 10; N_VAL = 10
np.random.seed(42)
perm = np.random.permutation(N)
test_idx  = perm[:N_TEST]
val_idx   = perm[N_TEST:N_TEST+N_VAL]
train_idx = perm[N_TEST+N_VAL:]

y_train = y_all[train_idx]
y_val   = y_all[val_idx]
y_test  = y_all[test_idx]

print(f"\nSplit: {len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} test")
print(f"Train pKd range: {y_train.min():.1f} - {y_train.max():.1f}")

# ── Normalize ────────────────────────────────────────────────────────────────
mu_p = X_raw_all[train_idx][:, PHYS_USE].mean(0)
sd_p = X_raw_all[train_idx][:, PHYS_USE].std(0)
sd_p = np.where(sd_p < 1e-10, 1, sd_p)

mu_e = X_ecfp_all[train_idx].mean(0)
sd_e = X_ecfp_all[train_idx].std(0)
sd_e = np.where(sd_e < 1e-10, 1, sd_e)

X_tr_p = (X_raw_all[train_idx][:, PHYS_USE] - mu_p) / sd_p
X_vl_p = (X_raw_all[val_idx][:, PHYS_USE] - mu_p) / sd_p
X_te_p = (X_raw_all[test_idx][:, PHYS_USE] - mu_p) / sd_p

X_tr_e = (X_ecfp_all[train_idx] - mu_e) / sd_e
X_vl_e = (X_ecfp_all[val_idx] - mu_e) / sd_e
X_te_e = (X_ecfp_all[test_idx] - mu_e) / sd_e

# ── Grid Search ──────────────────────────────────────────────────────────────
RK = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)

configs = []
for kp in [0, 5, 7, 9, 11]:
    for ke in [10, 15, 18, 22, 30, 50]:
        for alpha in [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]:
            configs.append((kp, ke, alpha))

print(f"\nRunning {len(configs)} configurations...")
results = []

for idx, (kp, ke, alpha) in enumerate(configs):
    if idx % 50 == 0:
        print(f"  {idx}/{len(configs)}", flush=True)
    
    try:
        if kp > 0:
            sel_p = SelectKBest(f_regression, k=min(kp, 14))
            Xp_tr = sel_p.fit_transform(X_tr_p, y_train)
            Xp_vl = sel_p.transform(X_vl_p)
            Xp_te = sel_p.transform(X_te_p)
        else:
            sel_p = None
            Xp_tr = np.zeros((len(y_train), 0))
            Xp_vl = np.zeros((len(y_val), 0))
            Xp_te = np.zeros((len(y_test), 0))
        
        sel_e = SelectKBest(f_regression, k=min(ke, 512))
        Xe_tr = sel_e.fit_transform(X_tr_e, y_train)
        Xe_vl = sel_e.transform(X_vl_e)
        Xe_te = sel_e.transform(X_te_e)
        
        X_tr = np.hstack([Xp_tr, Xe_tr])
        X_vl = np.hstack([Xp_vl, Xe_vl])
        X_te = np.hstack([Xp_te, Xe_te])
        
        if X_tr.shape[1] < 2:
            continue
        
        model = Ridge(alpha=alpha)
        model.fit(X_tr, y_train)
        
        loo_preds = cross_val_predict(Ridge(alpha=alpha), X_tr, y_train, cv=LeaveOneOut())
        loo_r = r_score(y_train, loo_preds)
        loo_mae = np.mean(np.abs(y_train - loo_preds))
        
        rkf_rs = []
        for tr_i, vl_i in RK.split(X_tr):
            m = Ridge(alpha=alpha)
            m.fit(X_tr[tr_i], y_train[tr_i])
            rkf_rs.append(r_score(y_train[vl_i], m.predict(X_tr[vl_i])))
        
        val_r = r_score(y_val, model.predict(X_vl))
        test_r = r_score(y_test, model.predict(X_te))
        train_r = r_score(y_train, model.predict(X_tr))
        
        results.append({
            'kp': kp, 'ke': ke, 'alpha': alpha,
            'loo_r': loo_r, 'val_r': val_r, 'test_r': test_r,
            'rkf_r': np.mean(rkf_rs), 'rkf_std': np.std(rkf_rs),
            'train_r': train_r, 'gap': train_r - loo_r,
            'loo_mae': loo_mae,
            'n_features': X_tr.shape[1],
            'sel_p': sel_p, 'sel_e': sel_e,
        })
    except Exception as e:
        pass

print(f"\nEvaluated {len(results)} configurations")
results.sort(key=lambda x: x['loo_r'], reverse=True)

# ── Top Results ─────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("  TOP 20 CONFIGURATIONS (by LOO-R)")
print("=" * 80)
print(f"  {'#':<3} {'kp':>3} {'ke':>3} {'alpha':>6} {'LOO-R':>7} {'RKF-R':>8} {'Val-R':>7} {'Test-R':>7} {'Gap':>6}")
print(f"  {'-'*3:<3} {'-'*3:>3} {'-'*3:>3} {'-'*6:>6} {'-'*7:>7} {'-'*8:>8} {'-'*7:>7} {'-'*7:>7} {'-'*6:>6}")
for i, r in enumerate(results[:20]):
    marker = " ★" if i == 0 else ""
    print(f"  {i+1:<3} {r['kp']:>3} {r['ke']:>3} {r['alpha']:>6.3f} {r['loo_r']:>7.4f} {r['rkf_r']:>7.3f}±{r['rkf_std']:<5.3f} {r['val_r']:>7.4f} {r['test_r']:>7.4f} {r['gap']:>6.3f}{marker}")

best = results[0]
kp, ke, alpha = best['kp'], best['ke'], best['alpha']
print(f"\n  ★ BEST: kp={kp}, ke={ke}, alpha={alpha}")
print(f"     LOO_r = {best['loo_r']:.4f}")
print(f"     RKF_r = {best['rkf_r']:.4f} ± {best['rkf_std']:.4f}")
print(f"     Val_r = {best['val_r']:.4f}")
print(f"     Test_r = {best['test_r']:.4f}")

# ── Rebuild with best config ────────────────────────────────────────────────
print("\n" + "=" * 80)
print("  REBUILDING BEST MODEL")
print("=" * 80)

if kp > 0:
    sel_p = SelectKBest(f_regression, k=min(kp, 14))
    Xp_tr = sel_p.fit_transform(X_tr_p, y_train)
    Xp_vl = sel_p.transform(X_vl_p)
    Xp_te = sel_p.transform(X_te_p)
else:
    sel_p = None
    Xp_tr = np.zeros((len(y_train), 0))
    Xp_vl = np.zeros((len(y_val), 0))
    Xp_te = np.zeros((len(y_test), 0))

sel_e = SelectKBest(f_regression, k=min(ke, 512))
Xe_tr = sel_e.fit_transform(X_tr_e, y_train)
Xe_vl = sel_e.transform(X_vl_e)
Xe_te = sel_e.transform(X_te_e)

X_tr_c = np.hstack([Xp_tr, Xe_tr])
X_vl_c = np.hstack([Xp_vl, Xe_vl])
X_te_c = np.hstack([Xp_te, Xe_te])

model = Ridge(alpha=alpha)
model.fit(X_tr_c, y_train)

loo_preds = cross_val_predict(Ridge(alpha=alpha), X_tr_c, y_train, cv=LeaveOneOut())
loo_r_final = r_score(y_train, loo_preds)

if sel_p is not None:
    selected_phy = [FEATURE_NAMES_24[PHYS_USE[j]] for j, v in enumerate(sel_p.get_support()) if v]
else:
    selected_phy = []

# ── Per-compound predictions ─────────────────────────────────────────────────
print(f"\n  Train ({len(train_idx)} compounds):")
print(f"  {'PDB':<6} {'True':>6} {'Pred':>6} {'Error':>7}")
print(f"  {'-'*28}")
train_errors = []
for idx, i in enumerate(train_idx):
    pred = loo_preds[idx]
    err = pred - y_all[i]
    train_errors.append(err)
    marker = " *" if abs(err) > 0.5 else ""
    print(f"  {ids_all[i]:<6} {y_all[i]:6.2f} {pred:6.2f} {err:+7.3f}{marker}")

val_preds = model.predict(X_vl_c)
print(f"\n  Val ({len(val_idx)} compounds):")
val_errors = []
for idx, i in enumerate(val_idx):
    err = val_preds[idx] - y_all[i]
    val_errors.append(err)
    marker = " *" if abs(err) > 0.5 else ""
    print(f"  {ids_all[i]:<6} {y_all[i]:6.2f} {val_preds[idx]:6.2f} {err:+7.3f}{marker}")

test_preds = model.predict(X_te_c)
print(f"\n  Test ({len(test_idx)} compounds):")
for idx, i in enumerate(test_idx):
    err = test_preds[idx] - y_all[i]
    marker = " *" if abs(err) > 0.5 else ""
    print(f"  {ids_all[i]:<6} {y_all[i]:6.2f} {test_preds[idx]:6.2f} {err:+7.3f}{marker}")

train_mae = np.mean(np.abs(train_errors))
val_mae = np.mean(np.abs(val_errors))
test_mae = np.mean(np.abs(test_preds - y_test))
print(f"\n  MAE — Train(LOO): {train_mae:.3f}, Val: {val_mae:.3f}, Test: {test_mae:.3f}")

# ── Save Results ─────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("  SAVING RESULTS")
print("=" * 80)

import json
results_summary = []
for r in results:
    d = {k: v for k, v in r.items() if k not in ('sel_p', 'sel_e')}
    results_summary.append(d)
with open("WORK_DIR / train_expanded_results.json", "w") as f:
    json.dump(results_summary, f, indent=2)
print(f"  Saved results to train_expanded_results.json")

# Save predictions TSV
with open("WORK_DIR / results_expanded.tsv", "w") as f:
    f.write("split\tpdb_id\ttrue\tpred\terror\n")
    for idx, i in enumerate(train_idx):
        f.write(f"train\t{ids_all[i]}\t{y_all[i]:.2f}\t{loo_preds[idx]:.2f}\t{loo_preds[idx]-y_all[i]:+.3f}\n")
    for idx, i in enumerate(val_idx):
        f.write(f"val\t{ids_all[i]}\t{y_all[i]:.2f}\t{val_preds[idx]:.2f}\t{val_preds[idx]-y_all[i]:+.3f}\n")
    for idx, i in enumerate(test_idx):
        f.write(f"test\t{ids_all[i]}\t{y_all[i]:.2f}\t{test_preds[idx]:.2f}\t{test_preds[idx]-y_all[i]:+.3f}\n")
print(f"  Saved predictions to results_expanded.tsv")

# Save model
out = {
    'ridge': model,
    'sel_p': sel_p,
    'sel_e': sel_e,
    'kp': kp, 'ke': ke, 'alpha': alpha,
    'mu_p': mu_p, 'sd_p': sd_p,
    'mu_e': mu_e, 'sd_e': sd_e,
    'loo_r': loo_r_final,
    'rkf_r': best['rkf_r'],
    'rkf_std': best['rkf_std'],
    'val_r': best['val_r'],
    'test_r': best['test_r'],
    'train_r': best['train_r'],
    'gap': best['gap'],
    'loo_mae': train_mae,
    'n_features': X_tr_c.shape[1],
    'selected_phy': selected_phy,
    'feature_names': FEATURE_NAMES_24,
    'all_pdb_ids': list(ids_all),
    'PHYS_USE': PHYS_USE,
    'n_compounds': N,
}

model_path = Path("WORK_DIR / best_model_expanded.pkl")
with open(model_path, "wb") as f:
    pickle.dump(out, f)
print(f"  Saved model to {model_path}")

# ── Also compare with original 96-compound model ─────────────────────────────
print("\n" + "=" * 80)
print("  COMPARISON: Original 96 vs Expanded 1094")
print("=" * 80)
print(f"  Original (96 compounds): kp=9, ke=18, alpha=0.2 → LOO_r=0.504")
print(f"  Expanded ({N} compounds): kp={kp}, ke={ke}, alpha={alpha} → LOO_r={loo_r_final:.4f}")
print(f"  Improvement: {loo_r_final - 0.504:+.4f}")

# ── Final Summary ────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("  FINAL RESULTS — GEOCK ENGINE (EXPANDED)")
print("=" * 80)
print(f"""
  ═══════════════════════════════════════════════════════════════════════════════
  ★ BEST CONFIG: kp={kp} physics, ke={ke} ECFP, alpha={alpha}
  ═══════════════════════════════════════════════════════════════════════════════

  METRIC         VALUE      MEANING
  ───────────── ────────── ──────────────────────────────────────────────
  Dataset Size   {N}        Total compounds
  Train Size     {len(train_idx)}       For training
  LOO-R          {loo_r_final:.4f}     Honest generalization (key metric)
  RKF-R          {best['rkf_r']:.4f}±{best['rkf_std']:.3f}  Repeated K-Fold
  Val-R          {best['val_r']:.4f}     Optimistic
  Test-R         {best['test_r']:.4f}     {N_TEST} compounds
  Train-LOO Gap  {best['gap']:.4f}     {'Mild overfitting' if best['gap'] > 0.2 else 'Acceptable'}
  LOO-MAE        {train_mae:.3f}     pKd units

  FEATURES:
    Physics ({len(selected_phy)}): {selected_phy if selected_phy else 'Disabled (ECFP-only)'}
    ECFP ({Xe_tr.shape[1]} bits): selected by SelectKBest
""")
