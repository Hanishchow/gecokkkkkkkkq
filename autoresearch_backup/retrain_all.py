#!/usr/bin/env python3
"""Extract features from ALL downloaded PDB files and retrain."""
import os, pickle, time
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.linear_model import Ridge
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import LeaveOneOut, cross_val_predict, RepeatedKFold
from scipy.stats import pearsonr
import json

CACHE_DIR = '/home/chow/.cache/geock_autoresearch'
LP_CSV = f'{CACHE_DIR}/LP_PDBBind.csv'
PDB_DIR = f'{CACHE_DIR}/lp_pdb_files'

def get_ecfp(smiles, fp_size=512):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=fp_size)
        return np.array(fpgen.GetFingerprintAsNumPy(mol), dtype=np.float32)
    except:
        return None

def process_one(item):
    pdb_id, row = item
    pdb_id = pdb_id.lower().strip()
    pdb_path = f'{PDB_DIR}/{pdb_id}.pdb'
    if not os.path.exists(pdb_path):
        return None
    smiles = row['smiles']
    if not isinstance(smiles, str) or pd.isna(smiles):
        return None
    ecfp = get_ecfp(smiles)
    if ecfp is None:
        return None
    value = row['value']
    if not isinstance(value, (int, float)) or pd.isna(value):
        return None
    return {'pdb_id': pdb_id, 'smiles': smiles, 'affinity': float(value), 'ecfp': ecfp}

print("Loading LP-PDBBind CSV...")
lp_df = pd.read_csv(LP_CSV, index_col=0)
lp_df.index = lp_df.index.str.lower()

print(f"Total LP-PDBBind entries: {len(lp_df)}")

existing_pdb_ids = [f.replace('.pdb', '') for f in os.listdir(PDB_DIR) if f.endswith('.pdb')]
existing_pdb_ids = set(existing_pdb_ids)
print(f"Downloaded PDB files: {len(existing_pdb_ids)}")

matched = lp_df[lp_df.index.str.lower().isin(existing_pdb_ids)]
matched = matched[matched['smiles'].notna() & matched['value'].notna()]
print(f"Matched with SMILES + affinity: {len(matched)}")

# High-quality filter (CL1, non-covalent, has SMILES)
hq = matched[(matched['CL1'] == True) & (matched['covalent'] == False)]
print(f"High-quality (CL1, non-covalent): {len(hq)}")

# All matched (less strict)
all_matched = matched[matched['covalent'] == False]
print(f"Non-covalent matched: {len(all_matched)}")

print("\nExtracting ECFP features in parallel...")
t0 = time.time()

# Process all matched (non-covalent with SMILES)
items = [(pid, row) for pid, row in all_matched.iterrows()]
N_WORKERS = 30

results = []
with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
    futures = {ex.submit(process_one, item): item for item in items}
    for i, future in enumerate(as_completed(futures)):
        r = future.result()
        if r is not None:
            results.append(r)
        if (i+1) % 200 == 0:
            print(f"  {i+1}/{len(items)} processed, {len(results)} valid")

print(f"\nValid compounds: {len(results)} in {time.time()-t0:.0f}s")

# Load existing features for comparison/merge
existing_path = f'{CACHE_DIR}/features_combined.pkl'
if os.path.exists(existing_path):
    with open(existing_path, 'rb') as f:
        existing = pickle.load(f)
    existing_ecfp = existing['X_ecfp']
    existing_y = existing['y_pkd']
    existing_ids = existing['pdb_ids']
    print(f"Existing features: {len(existing_ids)}")
else:
    existing = None

# Save new features
with open(f'{CACHE_DIR}/lp_all_features.pkl', 'wb') as f:
    pickle.dump(results, f)
print(f"Saved to {CACHE_DIR}/lp_all_features.pkl")

# Train models
print("\n" + "="*60)
print("TRAINING MODELS")
print("="*60)

def train_model(X, y, name, alphas=[0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0], ks=[50, 100, 150, 200]):
    mu = X.mean(0)
    sd = X.std(0)
    sd = np.where(sd < 1e-10, 1, sd)
    X_n = (X - mu) / sd
    
    best_r, best_k, best_a = 0, None, None
    for k in ks:
        if k >= X.shape[1]:
            continue
        sel = SelectKBest(f_regression, k=k)
        X_s = sel.fit_transform(X_n, y)
        for a in alphas:
            loo_preds = cross_val_predict(Ridge(alpha=a), X_s, y, cv=LeaveOneOut())
            loo_r = pearsonr(y, loo_preds)[0]
            if loo_r > best_r:
                best_r, best_k, best_a = loo_r, k, a
    
    sel = SelectKBest(f_regression, k=best_k)
    X_s = sel.fit_transform(X_n, y)
    ridge = Ridge(alpha=best_a)
    ridge.fit(X_s, y)
    train_r = pearsonr(y, ridge.predict(X_s))[0]
    
    rkf = RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)
    rkf_preds = cross_val_predict(Ridge(alpha=best_a), X_s, y, cv=rkf)
    rkf_r = pearsonr(y, rkf_preds)[0]
    
    gap = train_r - rkf_r
    
    print(f"\n{name}")
    print(f"  N={len(y)}, ke={best_k}, alpha={best_a}")
    print(f"  LOO-R={best_r:.4f}, RKF-R={rkf_r:.4f}, Train-R={train_r:.4f}, Gap={gap:.4f}")
    
    return {
        'ridge': ridge,
        'sel': sel,
        'mu': mu,
        'sd': sd,
        'ke': best_k,
        'alpha': best_a,
        'loo_r': best_r,
        'rkf_r': rkf_r,
        'train_r': train_r,
        'gap': gap,
        'n_compounds': len(y),
        'ecfp_len': X.shape[1],
    }

X_all = np.array([r['ecfp'] for r in results], dtype=np.float32)
y_all = np.array([r['affinity'] for r in results])

# Model 1: All available data (ECFP only)
m1 = train_model(X_all, y_all, "All available data (ECFP only)")

# Save predictions
preds = m1['ridge'].predict(m1['sel'].transform((X_all - m1['mu']) / np.where(m1['sd'] < 1e-10, 1, m1['sd'])))
with open(f'{CACHE_DIR}/lp_all_predictions.pkl', 'wb') as f:
    pickle.dump({'smiles': [r['smiles'] for r in results], 'pdb_id': [r['pdb_id'] for r in results], 
                 'actual': y_all, 'predicted': preds}, f)

# Save the new model
import json
model_path = 'WORK_DIR / geock_model_all.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(m1, f)
print(f"\nSaved model to {model_path}")

# Save TSV
tsv_path = 'WORK_DIR / results_all.tsv'
with open(tsv_path, 'w') as f:
    f.write("pdb_id\tsmiles\tactual_pKd\tpredicted_pKd\terror\n")
    for i, r in enumerate(results):
        err = abs(y_all[i] - preds[i])
        f.write(f"{r['pdb_id']}\t{r['smiles']}\t{y_all[i]:.3f}\t{preds[i]:.3f}\t{err:.3f}\n")
print(f"Saved predictions to {tsv_path}")

print("\n" + "="*60)
print("DONE")
print("="*60)
