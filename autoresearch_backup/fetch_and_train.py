#!/usr/bin/env python3
"""Download PDBs from LP-PDBBind and generate features in parallel."""
import os, pickle, time, json
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

DATA_DIR = '/mnt/c/Users/yakka/Downloads/geock_110_data'
CACHE_DIR = '/home/chow/.cache/geock_autoresearch'
LP_CSV = f'{CACHE_DIR}/LP_PDBBind.csv'
PDB_DIR = f'{CACHE_DIR}/lp_pdb_files'
os.makedirs(PDB_DIR, exist_ok=True)

N_WORKERS = 20
BATCH_SIZE = 500

def get_ecfp(smiles, radius=2, n_bits=512):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits))

def download_pdb(pdb_id):
    pdb_id = pdb_id.lower().strip()
    out_path = f'{PDB_DIR}/{pdb_id}.pdb'
    if os.path.exists(out_path):
        return pdb_id, True, None
    try:
        r = requests.get(f'https://files.rcsb.org/download/{pdb_id.upper()}.pdb', timeout=15)
        if r.status_code == 200 and 'HEADER' in r.text:
            with open(out_path, 'w') as f:
                f.write(r.text)
            return pdb_id, True, None
        else:
            return pdb_id, False, f'status={r.status_code}'
    except Exception as e:
        return pdb_id, False, str(e)

def parse_pdb_ligands(pdb_path):
    """Extract ligand info from PDB file HETATM records."""
    ligands = []
    hetatm_lines = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith('HETATM') or line.startswith('ATOM'):
                hetatm_lines.append(line)
            elif line.startswith('HET '):
                parts = line.split()
                if len(parts) >= 2:
                    ligands.append(parts[1])
    return ligands, hetatm_lines

def simple_physics_from_pdb(pdb_path, ligand_sdf=None):
    """Compute simple physics-like features from PDB file."""
    try:
        with open(pdb_path) as f:
            content = f.read()
        
        lines = content.split('\n')
        
        # Extract protein atoms (ATOM records)
        protein_atoms = [l for l in lines if l.startswith('ATOM')]
        n_protein_atoms = len(protein_atoms)
        
        # Count residue types
        res_types = set()
        for l in protein_atoms:
            if len(l) > 17:
                res_types.add(l[17:20].strip())
        n_residues = len(res_types)
        
        # Extract ligand HETATM
        hetatm = [l for l in lines if l.startswith('HETATM')]
        n_ligand_atoms = len(hetatm)
        
        # Simple features
        features = [
            n_protein_atoms,                    # 0
            n_residues,                          # 1
            n_ligand_atoms,                      # 2
            0, 0, 0,                            # placeholders for E1 Vinardo (3-5)
            0, 0, 0, 0, 0, 0,                  # placeholders for E2 Chemistry (6-11)
            0, 0, 0,                            # placeholders for E3 VQE (12-14)
            0, 0, 0, 0, 0, 0, 0, 0, 0,       # placeholders for E4 Bio (15-23)
        ]
        return np.array(features, dtype=np.float32)
    except Exception as e:
        return None

def process_compound(pdb_id):
    """Download PDB and compute features for one compound."""
    pdb_id = pdb_id.lower().strip()
    pdb_path = f'{PDB_DIR}/{pdb_id}.pdb'
    
    # Download if needed
    if not os.path.exists(pdb_path):
        pid, ok, err = download_pdb(pdb_id)
        if not ok:
            return None
    
    # Get SMILES and affinity from LP_PDBBind
    if pdb_id not in lp_df.index.str.lower().tolist():
        return None
    
    row = lp_df[lp_df.index.str.lower() == pdb_id].iloc[0]
    smiles = row['smiles']
    affinity = row['value']
    if pd.isna(smiles) or pd.isna(affinity):
        return None
    
    # Generate ECFP
    ecfp = get_ecfp(smiles)
    if ecfp is None:
        return None
    
    # Simple physics from PDB
    physics = simple_physics_from_pdb(pdb_path)
    if physics is None:
        return None
    
    return {
        'pdb_id': pdb_id,
        'smiles': smiles,
        'affinity': float(affinity),
        'ecfp': ecfp,
        'physics': physics,
        'pdb_path': pdb_path,
    }

def download_batch(pdb_ids, desc=''):
    """Download PDB files in parallel."""
    results = {}
    done = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {ex.submit(download_pdb, pid): pid for pid in pdb_ids}
        for future in as_completed(futures):
            pid, ok, err = future.result()
            if ok:
                done += 1
            else:
                failed += 1
            if (done + failed) % 100 == 0:
                print(f'  {desc} Downloaded: {done}, Failed: {failed}')
    print(f'  {desc} Done: {done} ok, {failed} failed')
    return done, failed

def process_batch(compounds, desc=''):
    """Process compounds in parallel."""
    results = []
    n_none = 0
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {ex.submit(process_compound, cid): cid for cid in compounds}
        for i, future in enumerate(as_completed(futures)):
            r = future.result()
            if r is not None:
                results.append(r)
            else:
                n_none += 1
            if (i+1) % 100 == 0:
                print(f'  {desc} Processed: {i+1}/{len(compounds)}, Got: {len(results)}')
    print(f'  {desc} Final: {len(results)} results from {len(compounds)} IDs')
    return results

# Load LP-PDBBind
print('Loading LP-PDBBind data...')
lp_df = pd.read_csv(LP_CSV, index_col=0)
lp_df.index = lp_df.index.str.lower()

# Load existing cache
print('Loading existing features cache...')
with open(f'{CACHE_DIR}/features_110.pkl', 'rb') as f:
    existing = pickle.load(f)
existing_pdb_ids = set(existing['pdb_ids'])

# Filter high quality training data
high_q = lp_df[(lp_df['CL1'] == True) & (lp_df['new_split'] == 'train') & (lp_df['covalent'] == False)]
high_q = high_q[high_q['smiles'].notna() & high_q['value'].notna()]
new_pdb_ids = [pid.lower() for pid in high_q.index if pid.lower() not in existing_pdb_ids]
print(f'New PDB IDs to process: {len(new_pdb_ids)}')

# Process in batches
all_new = []
total_downloaded = 0

for batch_start in range(0, min(len(new_pdb_ids), 1000), BATCH_SIZE):
    batch_end = batch_start + BATCH_SIZE
    batch_ids = new_pdb_ids[batch_start:batch_end]
    batch_desc = f'Batch {batch_start//BATCH_SIZE + 1}'
    print(f'\n{batch_desc}: Downloading {len(batch_ids)} PDBs...')
    
    t0 = time.time()
    done, failed = download_batch(batch_ids, batch_desc)
    total_downloaded += done
    print(f'  Download time: {time.time()-t0:.1f}s')
    
    print(f'  Processing features...')
    t1 = time.time()
    results = process_batch(batch_ids, batch_desc)
    print(f'  Process time: {time.time()-t1:.1f}s')
    
    all_new.extend(results)
    print(f'  Batch results: {len(results)}/{len(batch_ids)}')
    
    # Save intermediate results
    if all_new:
        print(f'  Saving intermediate: {len(all_new)} compounds...')
        with open(f'{CACHE_DIR}/lp_new_features_partial.pkl', 'wb') as f:
            pickle.dump(all_new, f)

print(f'\nTotal downloaded PDBs: {total_downloaded}')
print(f'Total new compounds with features: {len(all_new)}')

# Save final results
if all_new:
    print('Saving final new features...')
    with open(f'{CACHE_DIR}/lp_new_features.pkl', 'wb') as f:
        pickle.dump(all_new, f)
    
    # Merge with existing
    print('Merging with existing features...')
    new_X_raw = np.array([r['physics'] for r in all_new])
    new_y = np.array([r['affinity'] for r in all_new])
    new_pdb_ids_out = [r['pdb_id'] for r in all_new]
    new_X_ecfp = np.array([r['ecfp'] for r in all_new], dtype=np.float32)
    
    combined_X_raw = np.vstack([existing['X_raw'], new_X_raw])
    combined_y = np.concatenate([existing['y_pkd'], new_y])
    combined_pdb_ids = existing['pdb_ids'] + new_pdb_ids_out
    combined_X_ecfp = np.vstack([existing['X_ecfp'].astype(np.float32), new_X_ecfp])
    
    combined_n = len(combined_pdb_ids)
    print(f'Combined: {combined_n} compounds')
    
    with open(f'{CACHE_DIR}/features_combined.pkl', 'wb') as f:
        pickle.dump({
            'X_raw': combined_X_raw,
            'y_pkd': combined_y,
            'pdb_ids': combined_pdb_ids,
            'X_ecfp': combined_X_ecfp,
            'n_compounds': combined_n,
        }, f)
    print('Saved combined features to features_combined.pkl')
else:
    print('No new compounds processed!')

print('\nDone!')
