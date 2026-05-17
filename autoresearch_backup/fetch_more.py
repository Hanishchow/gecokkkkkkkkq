#!/usr/bin/env python3
"""Download more LP-PDBBind PDBs and generate features. Continues from previous batch."""
import os, pickle, time, json
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, rdFingerprintGenerator
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

CACHE_DIR = '/home/chow/.cache/geock_autoresearch'
LP_CSV = f'{CACHE_DIR}/LP_PDBBind.csv'
PDB_DIR = f'{CACHE_DIR}/lp_pdb_files'
os.makedirs(PDB_DIR, exist_ok=True)

N_WORKERS = 20
BATCH_SIZE = 1000

def get_ecfp(smiles, fp_size=512):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=fp_size)
    return np.array(fpgen.GetFingerprintAsNumPy(mol), dtype=np.float32)

def download_pdb(pdb_id):
    pdb_id = pdb_id.lower().strip()
    out_path = f'{PDB_DIR}/{pdb_id}.pdb'
    if os.path.exists(out_path):
        return pdb_id, True, None, 0
    try:
        r = requests.get(f'https://files.rcsb.org/download/{pdb_id.upper()}.pdb', timeout=15)
        if r.status_code == 200 and 'HEADER' in r.text:
            with open(out_path, 'w') as f:
                f.write(r.text)
            return pdb_id, True, None, len(r.text)
        else:
            return pdb_id, False, f'status={r.status_code}', 0
    except Exception as e:
        return pdb_id, False, str(e), 0

def process_compound(pdb_id, smiles, affinity):
    pdb_id = pdb_id.lower().strip()
    pdb_path = f'{PDB_DIR}/{pdb_id}.pdb'
    
    if not os.path.exists(pdb_path):
        return None
    
    ecfp = get_ecfp(smiles)
    if ecfp is None:
        return None
    
    # Simple structural features from PDB
    try:
        with open(pdb_path) as f:
            content = f.read()
        lines = content.split('\n')
        protein_atoms = [l for l in lines if l.startswith('ATOM')]
        hetatm = [l for l in lines if l.startswith('HETATM')]
        res_types = set()
        for l in protein_atoms:
            if len(l) > 17:
                res_types.add(l[17:20].strip())
        
        features = np.array([
            len(protein_atoms),      # n_protein_atoms
            len(res_types),          # n_residue_types
            len(hetatm),             # n_ligand_atoms
        ], dtype=np.float32)
    except:
        features = np.zeros(3, dtype=np.float32)
    
    return {
        'pdb_id': pdb_id,
        'smiles': smiles,
        'affinity': float(affinity),
        'ecfp': ecfp,
        'physics_simple': features,
    }

def download_and_process_batch(pdb_ids_smiles, batch_name=''):
    """Download PDBs and extract features."""
    downloaded = 0
    processed = 0
    failed = 0
    
    # First pass: download PDBs
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {ex.submit(download_pdb, pid): (pid, smi, aff) 
                   for pid, smi, aff in pdb_ids_smiles}
        results = {}
        for future in as_completed(futures):
            pid, ok, err, size = future.result()
            if ok:
                downloaded += 1
            else:
                failed += 1
            if (downloaded + failed) % 200 == 0:
                print(f'  {batch_name} Downloaded: {downloaded}, Failed: {failed}')
    
    # Second pass: process features
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {ex.submit(process_compound, pid, smi, aff): pid 
                   for pid, smi, aff in pdb_ids_smiles}
        compound_results = []
        for future in as_completed(futures):
            r = future.result()
            if r is not None:
                processed += 1
                compound_results.append(r)
            if processed % 100 == 0:
                print(f'  {batch_name} Processed: {processed}')
    
    print(f'  {batch_name} Final: {downloaded} downloaded, {processed} processed, {failed} failed')
    return compound_results

# Load LP-PDBBind
lp_df = pd.read_csv(LP_CSV, index_col=0)
lp_df.index = lp_df.index.str.lower()

# Load existing pdb files
existing_pdb_files = set([f.replace('.pdb', '') for f in os.listdir(PDB_DIR) if f.endswith('.pdb')])
print(f'Existing PDB files: {len(existing_pdb_files)}')

# Load previous results
prev_features_path = f'{CACHE_DIR}/lp_new_features.pkl'
if os.path.exists(prev_features_path):
    with open(prev_features_path, 'rb') as f:
        prev_features = pickle.load(f)
    prev_pdb_ids = set([r['pdb_id'] for r in prev_features])
    print(f'Previous features: {len(prev_features)} compounds')
else:
    prev_features = []
    prev_pdb_ids = set()

# Filter high quality training data
high_q = lp_df[(lp_df['CL1'] == True) & (lp_df['new_split'] == 'train') & (lp_df['covalent'] == False)]
high_q = high_q[high_q['smiles'].notna() & high_q['value'].notna()]

# Get PDB IDs we don't have yet
already_have = existing_pdb_files | prev_pdb_ids
new_pdb_ids = [pid.lower() for pid in high_q.index if pid.lower() not in already_have]
print(f'New PDB IDs to download: {len(new_pdb_ids)}')

# Take first 3000 for this batch
target_ids = new_pdb_ids[:3000]
target_df = high_q[high_q.index.str.lower().isin(set(target_ids))]
pdb_ids_smiles = [(pid, row['smiles'], row['value']) 
                   for pid, row in target_df.iterrows()]

print(f'\nProcessing batch of {len(pdb_ids_smiles)} compounds...')
t0 = time.time()
results = download_and_process_batch(pdb_ids_smiles, f'Batch1')
print(f'Time: {time.time()-t0:.0f}s')

if results:
    # Merge with previous
    all_features = prev_features + results
    print(f'\nTotal features: {len(all_features)} compounds')
    
    # Save
    with open(f'{CACHE_DIR}/lp_new_features_all.pkl', 'wb') as f:
        pickle.dump(all_features, f)
    
    # Quick training check
    if len(all_features) > 100:
        print('\nQuick training check...')
        X = np.array([r['ecfp'] for r in all_features], dtype=np.float32)
        y = np.array([r['affinity'] for r in all_features])
        
        from sklearn.linear_model import Ridge
        from sklearn.feature_selection import SelectKBest, f_regression
        from sklearn.model_selection import LeaveOneOut, cross_val_predict
        from scipy.stats import pearsonr
        
        mu = X.mean(0)
        sd = X.std(0)
        sd = np.where(sd < 1e-10, 1, sd)
        X_n = (X - mu) / sd
        
        sel = SelectKBest(f_regression, k=min(100, X.shape[1]))
        X_s = sel.fit_transform(X_n, y)
        
        loo_preds = cross_val_predict(Ridge(alpha=10), X_s, y, cv=LeaveOneOut())
        loo_r = pearsonr(y, loo_preds)[0]
        print(f'LOO-R with {len(all_features)} compounds: {loo_r:.4f}')

print('\nDone!')
