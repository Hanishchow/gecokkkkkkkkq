#!/usr/bin/env python3
"""Continuous download → extract → retrain pipeline for GEOCK."""
import os, pickle, time, sys
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from scipy.stats import pearsonr

CACHE_DIR = '/home/chow/.cache/geock_autoresearch'
LP_CSV = f'{CACHE_DIR}/LP_PDBBind.csv'
PDB_DIR = f'{CACHE_DIR}/lp_pdb_files'
AUTORESEARCH = '/home/chow/autoresearch'
N_WORKERS = 20

os.makedirs(PDB_DIR, exist_ok=True)

def get_ecfp(smiles, fp_size=512):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=fp_size)
        return np.array(fpgen.GetFingerprintAsNumPy(mol), dtype=np.float32)
    except:
        return None

def download_pdb(pdb_id):
    pdb_id = pdb_id.lower().strip()
    out_path = f'{PDB_DIR}/{pdb_id}.pdb'
    if os.path.exists(out_path):
        return pdb_id, True, None
    try:
        import requests
        r = requests.get(f'https://files.rcsb.org/download/{pdb_id.upper()}.pdb', timeout=15)
        if r.status_code == 200 and 'HEADER' in r.text:
            with open(out_path, 'w') as f:
                f.write(r.text)
            return pdb_id, True, None
        return pdb_id, False, f'status={r.status_code}'
    except Exception as e:
        return pdb_id, False, str(e)

def extract_features(pdb_id, smiles, affinity):
    pdb_path = f'{PDB_DIR}/{pdb_id}.pdb'
    if not os.path.exists(pdb_path):
        return None
    ecfp = get_ecfp(smiles)
    if ecfp is None:
        return None
    return {'pdb_id': pdb_id, 'smiles': smiles, 'affinity': float(affinity), 'ecfp': ecfp}

def download_batch(pdb_ids_smiles, batch_name=''):
    downloaded = failed = 0
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {ex.submit(download_pdb, pid): (pid, smi, aff) for pid, smi, aff in pdb_ids_smiles}
        for future in as_completed(futures):
            pid, ok, err = future.result()
            if ok: downloaded += 1
            else: failed += 1
            if (downloaded + failed) % 500 == 0:
                print(f'  {batch_name}: {downloaded} ok, {failed} failed')
    print(f'  {batch_name}: downloaded={downloaded}, failed={failed}')
    return downloaded

def extract_batch(pdb_ids_smiles, batch_name=''):
    processed = 0
    results = []
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {ex.submit(extract_features, pid, smi, aff): pid for pid, smi, aff in pdb_ids_smiles}
        for future in as_completed(futures):
            r = future.result()
            if r is not None:
                processed += 1
                results.append(r)
            if processed % 200 == 0:
                print(f'  {batch_name}: {processed} processed')
    print(f'  {batch_name}: extracted={processed}')
    return results

def train_and_save(features, label=''):
    if len(features) < 50:
        print(f'  Too few features ({len(features)}), skipping training')
        return None
    
    X = np.array([f['ecfp'] for f in features], dtype=np.float32)
    y = np.array([f['affinity'] for f in features])
    pdb_ids = [f['pdb_id'] for f in features]
    smiles_list = [f['smiles'] for f in features]
    
    mu = X.mean(0)
    sd = X.std(0)
    sd = np.where(sd < 1e-10, 1, sd)
    X_n = (X - mu) / sd
    
    # Fast RidgeCV for best ke + alpha
    print(f'  Training on {len(y)} compounds...')
    best_r, best_ke, best_alpha = 0, None, None
    
    for ke in [100, 150, 200, 250]:
        if ke >= X.shape[1]: continue
        sel = SelectKBest(f_regression, k=ke)
        X_s = sel.fit_transform(X_n, y)
        ridgecv = RidgeCV(alphas=[0.5, 1, 5, 10, 50, 100], cv=5, scoring='r2')
        ridgecv.fit(X_s, y)
        r2 = ridgecv.best_score_
        if r2 > best_r:
            best_r, best_ke, best_alpha = r2, ke, ridgecv.alpha_
    
    # Full LOO-CV
    sel = SelectKBest(f_regression, k=best_ke)
    X_s = sel.fit_transform(X_n, y)
    ridge = Ridge(alpha=best_alpha)
    ridge.fit(X_s, y)
    
    loo_preds = cross_val_predict(ridge, X_s, y, cv=LeaveOneOut())
    loo_r = pearsonr(y, loo_preds)[0]
    loo_mae = np.mean(np.abs(y - loo_preds))
    train_r = pearsonr(y, ridge.predict(X_s))[0]
    gap = train_r - loo_r
    
    print(f'  Best: ke={best_ke}, alpha={best_alpha}, LOO-R={loo_r:.4f}, R2_cv={best_r:.4f}')
    
    # Save model
    model = {
        'ridge': ridge, 'sel': sel, 'mu': mu, 'sd': sd,
        'sel_e': sel, 'mu_e': mu, 'sd_e': sd,
        'ke': best_ke, 'alpha': best_alpha,
        'loo_r': float(loo_r), 'rkf_r': float(best_r),
        'rkf_std': 0.0, 'train_r': float(train_r),
        'gap': float(gap), 'loo_mae': float(loo_mae),
        'n_compounds': len(y), 'ecfp_len': X.shape[1],
        'pdb_ids': pdb_ids, 'smiles': smiles_list,
    }
    model_path = f'{AUTORESEARCH}/geock_model_all.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f'  Model saved: {model_path}')
    
    # Save TSV
    tsv_path = f'{AUTORESEARCH}/results_all.tsv'
    with open(tsv_path, 'w') as f:
        f.write('pdb_id\tsmiles\tactual_pKd\tpredicted_pKd\terror\n')
        for i in range(len(y)):
            f.write(f'{pdb_ids[i]}\t{smiles_list[i]}\t{y[i]:.3f}\t{loo_preds[i]:.3f}\t{abs(y[i]-loo_preds[i]):.3f}\n')
    print(f'  Predictions saved: {tsv_path}')
    
    return model

def main():
    print("Loading LP-PDBBind CSV...")
    lp_df = pd.read_csv(LP_CSV, index_col=0)
    lp_df.index = lp_df.index.str.lower()
    
    # High quality filter
    hq = lp_df[(lp_df['CL1'] == True) & (lp_df['new_split'] == 'train') & (lp_df['covalent'] == False)]
    hq = hq[hq['smiles'].notna() & hq['value'].notna()]
    
    existing_pdb = set([f.replace('.pdb', '') for f in os.listdir(PDB_DIR) if f.endswith('.pdb')])
    to_download = [pid for pid in hq.index if pid.lower() not in existing_pdb]
    
    print(f'PDB files: {len(existing_pdb)} downloaded, {len(to_download)} to download')
    print(f'High quality LP-PDBBind total: {len(hq)}')
    
    BATCH = 500
    round_num = 1
    
    while to_download:
        batch_ids = to_download[:BATCH]
        batch_df = hq[hq.index.isin(batch_ids)]
        pdb_smiles = [(pid.lower(), row['smiles'], row['value']) for pid, row in batch_df.iterrows()]
        
        print(f'\n=== Round {round_num}: Downloading {len(pdb_smiles)} PDBs ===')
        t0 = time.time()
        
        download_batch(pdb_smiles, f'R{round_num}')
        time.sleep(2)
        features_new = extract_batch(pdb_smiles, f'R{round_num}')
        print(f'  Time: {time.time()-t0:.0f}s')
        
        if features_new:
            # Load existing features
            all_path = f'{CACHE_DIR}/lp_all_features.pkl'
            if os.path.exists(all_path):
                with open(all_path, 'rb') as f:
                    all_features = pickle.load(f)
                existing_ids = {f['pdb_id'] for f in all_features}
                new_only = [f for f in features_new if f['pdb_id'] not in existing_ids]
                all_features.extend(new_only)
                print(f'  Merged: {len(all_features)} total ({len(new_only)} new)')
            else:
                all_features = features_new
            
            with open(all_path, 'wb') as f:
                pickle.dump(all_features, f)
            
            # Train
            print(f'\n=== Training ===')
            train_and_save(all_features, f'round_{round_num}')
        
        # Update remaining
        existing_pdb = set([f.replace('.pdb', '') for f in os.listdir(PDB_DIR) if f.endswith('.pdb')])
        to_download = [pid for pid in hq.index if pid.lower() not in existing_pdb]
        print(f'\nRemaining: {len(to_download)} PDBs to download')
        
        round_num += 1
        if len(to_download) == 0:
            print('\n=== ALL DONE ===')
            break
        
        print(f'Pausing 5s before next batch...')
        time.sleep(5)

if __name__ == '__main__':
    main()
