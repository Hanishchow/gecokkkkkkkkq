#!/usr/bin/env python3
"""
train_augmented.py — GEOCK with SMILES augmentation
===================================================
Implements SMILES augmentation for training data:
- RDKit random SMILES generation (doRandom=True)
- Generate 3-5 augmented SMILES per compound
- Each augmented SMILES gets the same pKd label

Evaluates with LOO-CV and compares with baseline model (LOO-R = 0.614)
"""

import sys, os, warnings, pickle
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.linear_model import Ridge
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import LeaveOneOut, cross_val_predict, RepeatedKFold

np.random.seed(42)

CACHE_DIR = Path("/home/chow/.cache/geock_autoresearch")
DATA_PATH = CACHE_DIR / "lp_all_features.pkl"
OUTPUT_MODEL = Path("WORK_DIR / geock_model_augmented.pkl")
OUTPUT_RESULTS = Path("WORK_DIR / results_augmented.tsv")

AUGMENTATIONS_PER_COMPOUND = 4  # Generate 4 augmented SMILES per compound
FP_SIZE = 512
RADIUS = 2

def get_ecfp(smiles, fp_size=FP_SIZE, radius=RADIUS):
    """Generate ECFP fingerprint from SMILES."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fp_size)
        return np.array(fpgen.GetFingerprintAsNumPy(mol), dtype=np.float32)
    except:
        return None

def augment_smiles(smiles, n_augment=AUGMENTATIONS_PER_COMPOUND):
    """Generate augmented SMILES using RDKit's random SMILES generation."""
    augmented = []
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []
        
        for _ in range(n_augment):
            random_smiles = Chem.MolToSmiles(mol, doRandom=True)
            if random_smiles and random_smiles != smiles:
                augmented.append(random_smiles)
    except:
        pass
    
    return augmented[:n_augment]

def r_score(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return pearsonr(y_true, y_pred)[0] if len(y_true) >= 2 else np.nan

print("=" * 70)
print("  GEOCK SMILES AUGMENTATION TRAINING")
print("=" * 70)

print("\n[1/6] Loading original features...")
with open(DATA_PATH, "rb") as f:
    data = pickle.load(f)

print(f"  Loaded {len(data)} compounds")
print(f"  Sample: {data[0]['pdb_id']}, pKd={data[0]['affinity']:.2f}")

print(f"\n[2/6] Generating augmented SMILES ({AUGMENTATIONS_PER_COMPOUND} per compound)...")
augmented_data = []

for i, item in enumerate(data):
    original_smiles = item['smiles']
    original_ecfp = item['ecfp']
    affinity = item['affinity']
    pdb_id = item['pdb_id']
    
    augmented_data.append({
        'pdb_id': pdb_id,
        'smiles': original_smiles,
        'ecfp': original_ecfp,
        'affinity': affinity,
        'is_original': True
    })
    
    aug_smiles = augment_smiles(original_smiles, AUGMENTATIONS_PER_COMPOUND)
    
    for aug_smi in aug_smiles:
        aug_ecfp = get_ecfp(aug_smi)
        if aug_ecfp is not None:
            augmented_data.append({
                'pdb_id': pdb_id,
                'smiles': aug_smi,
                'ecfp': aug_ecfp,
                'affinity': affinity,
                'is_original': False
            })
    
    if (i + 1) % 1000 == 0:
        print(f"  Processed {i+1}/{len(data)} compounds...")

print(f"  Generated {len(augmented_data) - len(data)} augmented samples")
print(f"  Total samples: {len(augmented_data)}")

print("\n[3/6] Preparing feature matrices...")

X = np.array([item['ecfp'] for item in augmented_data], dtype=np.float32)
y = np.array([item['affinity'] for item in augmented_data], dtype=np.float32)

print(f"  X shape: {X.shape}")
print(f"  y shape: {y.shape}")
print(f"  pKd range: {y.min():.1f} - {y.max():.1f}")
print(f"  pKd mean: {y.mean():.2f}")

print("\n[4/6] Training with grid search (ke, alpha)...")

mu_e = X.mean(0)
sd_e = X.std(0)
sd_e = np.where(sd_e < 1e-10, 1, sd_e)

X_norm = (X - mu_e) / sd_e

RK = RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)

configs = []
for ke in [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]:
    for alpha in [10, 50, 100, 150, 200, 300, 500]:
        configs.append((ke, alpha))

print(f"  Testing {len(configs)} configurations...")
results = []

for idx, (ke, alpha) in enumerate(configs):
    try:
        sel_e = SelectKBest(f_regression, k=min(ke, FP_SIZE))
        X_sel = sel_e.fit_transform(X_norm, y)
        
        if X_sel.shape[1] < 2:
            continue
        
        loo_preds = cross_val_predict(Ridge(alpha=alpha), X_sel, y, cv=LeaveOneOut())
        loo_r = r_score(y, loo_preds)
        loo_mae = np.mean(np.abs(y - loo_preds))
        
        rkf_rs = []
        for tr_i, vl_i in RK.split(X_sel):
            m = Ridge(alpha=alpha)
            m.fit(X_sel[tr_i], y[tr_i])
            rkf_rs.append(r_score(y[vl_i], m.predict(X_sel[vl_i])))
        
        model = Ridge(alpha=alpha)
        model.fit(X_sel, y)
        train_r = r_score(y, model.predict(X_sel))
        
        results.append({
            'ke': ke,
            'alpha': alpha,
            'loo_r': loo_r,
            'rkf_r': np.mean(rkf_rs),
            'rkf_std': np.std(rkf_rs),
            'train_r': train_r,
            'gap': train_r - loo_r,
            'loo_mae': loo_mae,
            'n_features': X_sel.shape[1],
            'sel_e': sel_e,
            'model': model,
        })
    except Exception as e:
        pass
    
    if (idx + 1) % 20 == 0:
        print(f"    Tested {idx+1}/{len(configs)} configurations...")

print(f"  Evaluated {len(results)} configurations")

results.sort(key=lambda x: x['loo_r'], reverse=True)

print("\n" + "=" * 70)
print("  TOP 15 CONFIGURATIONS (by LOO-R)")
print("=" * 70)
print(f"  {'#':<3} {'ke':>4} {'alpha':>6} {'LOO-R':>8} {'RKF-R':>10} {'Train-R':>8} {'Gap':>7} {'MAE':>6}")
print(f"  {'-'*60}")
for i, r in enumerate(results[:15]):
    marker = " ★" if i == 0 else ""
    print(f"  {i+1:<3} {r['ke']:>4} {r['alpha']:>6.0f} {r['loo_r']:>8.4f} {r['rkf_r']:>7.3f}±{r['rkf_std']:<4.3f} {r['train_r']:>8.4f} {r['gap']:>7.3f} {r['loo_mae']:>6.3f}{marker}")

best = results[0]
ke, alpha = best['ke'], best['alpha']

print(f"\n  ★ BEST: ke={ke}, alpha={alpha}")
print(f"     LOO-R  = {best['loo_r']:.4f}")
print(f"     RKF-R  = {best['rkf_r']:.4f} ± {best['rkf_std']:.4f}")
print(f"     Train-R = {best['train_r']:.4f}")
print(f"     Gap    = {best['gap']:.4f}")
print(f"     MAE    = {best['loo_mae']:.3f} pKd")

baseline_loo_r = 0.614
improvement = best['loo_r'] - baseline_loo_r
print(f"\n  Comparison with baseline (ke=250, alpha=100):")
print(f"    Baseline LOO-R: {baseline_loo_r:.4f}")
print(f"    Augmented LOO-R: {best['loo_r']:.4f}")
print(f"    Improvement: {improvement:+.4f}")

print("\n[5/6] Saving model...")
model_data = {
    'ridge': best['model'],
    'sel_e': best['sel_e'],
    'mu_e': mu_e,
    'sd_e': sd_e,
    'ke': ke,
    'alpha': alpha,
    'loo_r': best['loo_r'],
    'rkf_r': best['rkf_r'],
    'rkf_std': best['rkf_std'],
    'train_r': best['train_r'],
    'gap': best['gap'],
    'loo_mae': best['loo_mae'],
    'n_compounds': len(data),
    'n_augmented': len(augmented_data),
    'n_features': best['n_features'],
}

with open(OUTPUT_MODEL, "wb") as f:
    pickle.dump(model_data, f)
print(f"  Saved to: {OUTPUT_MODEL}")

print("\n[6/6] Saving results...")

header = "ke\talpha\tLOO-R\tRKF-R\tRKF-STD\tTrain-R\tGap\tMAE\tN_Features\tN_Original\tN_Augmented"
rows = []
for r in results[:20]:
    row = f"{r['ke']}\t{r['alpha']}\t{r['loo_r']:.4f}\t{r['rkf_r']:.4f}\t{r['rkf_std']:.4f}\t{r['train_r']:.4f}\t{r['gap']:.4f}\t{r['loo_mae']:.3f}\t{r['n_features']}\t{len(data)}\t{len(augmented_data)}"
    rows.append(row)

with open(OUTPUT_RESULTS, "w") as f:
    f.write(header + "\n")
    f.write("\n".join(rows) + "\n")

print(f"  Saved to: {OUTPUT_RESULTS}")

print("\n" + "=" * 70)
print("  TRAINING COMPLETE")
print("=" * 70)
print(f"\n  Final Results:")
print(f"    LOO-R    = {best['loo_r']:.4f}")
print(f"    RKF-R    = {best['rkf_r']:.4f} ± {best['rkf_std']:.4f}")
print(f"    Train-R  = {best['train_r']:.4f}")
print(f"    Gap      = {best['gap']:.4f}")
print(f"    MAE      = {best['loo_mae']:.3f} pKd")
print(f"\n  Baseline (no augmentation): LOO-R = {baseline_loo_r:.4f}")
print(f"  Improvement: {improvement:+.4f}")
print("=" * 70)
