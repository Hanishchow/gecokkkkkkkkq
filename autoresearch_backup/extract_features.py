#!/usr/bin/env python3
"""
Quick feature extraction for GEOCK AutoResearch
Run in stages to avoid timeout
"""
import sys
import os
import json
import time
import pickle
import numpy as np

sys.path.insert(0, '/home/chow/autoresearch')
from bio_engine import run_all_compounds

DATA_DIR = '/mnt/c/Users/yakka/Downloads/geock_110_data'
CACHE_FILE = 'CACHE_DIR / features_v2.pkl'

STAGE = int(sys.argv[1]) if len(sys.argv) > 1 else 1

with open(os.path.join(DATA_DIR, 'compounds.json')) as f:
    compounds = json.load(f)

print(f"Stage {STAGE}: Starting...")

if STAGE == 1:
    # First 30 compounds (training set)
    n = 30
    print(f"Computing features for {n} compounds...")
    start = time.time()
    X_filt, X_raw, y_raw, pdb_ids = run_all_compounds(
        compounds[:n], DATA_DIR, use_quantum=True, verbose=True
    )
    print(f"Done in {time.time() - start:.1f}s")
    print(f"X_raw shape: {X_raw.shape}")
    
    # Save intermediate
    with open('/tmp/geock_stage1.pkl', 'wb') as f:
        pickle.dump({'X_raw': X_raw, 'y_raw': y_raw, 'pdb_ids': pdb_ids}, f)
    print("Saved to /tmp/geock_stage1.pkl")

elif STAGE == 2:
    # Compounds 31-50 (validation set)
    n_start, n_end = 30, 50
    print(f"Computing features for compounds {n_start}-{n_end}...")
    start = time.time()
    X_filt, X_raw, y_raw, pdb_ids = run_all_compounds(
        compounds[n_start:n_end], DATA_DIR, use_quantum=True, verbose=True
    )
    print(f"Done in {time.time() - start:.1f}s")
    
    with open('/tmp/geock_stage2.pkl', 'wb') as f:
        pickle.dump({'X_raw': X_raw, 'y_raw': y_raw, 'pdb_ids': pdb_ids}, f)
    print("Saved to /tmp/geock_stage2.pkl")

elif STAGE == 3:
    # Compounds 51-70 (test set) + ECFP + save final cache
    n_start, n_end = 50, 70
    print(f"Computing features for compounds {n_start}-{n_end}...")
    start = time.time()
    X_filt, X_raw, y_raw, pdb_ids = run_all_compounds(
        compounds[n_start:n_end], DATA_DIR, use_quantum=True, verbose=True
    )
    print(f"Done in {time.time() - start:.1f}s")
    
    # Load previous stages
    with open('/tmp/geock_stage1.pkl', 'rb') as f:
        s1 = pickle.load(f)
    with open('/tmp/geock_stage2.pkl', 'rb') as f:
        s2 = pickle.load(f)
    
    # Combine all
    X_raw = np.vstack([s1['X_raw'], s2['X_raw'], X_raw])
    y_raw = np.concatenate([s1['y_raw'], s2['y_raw'], y_raw])
    pdb_ids = s1['pdb_ids'] + s2['pdb_ids'] + pdb_ids
    
    # Convert to pKd
    y_pkd = (-y_raw / 1.364).astype(np.float32)
    
    # ECFP4 fingerprints
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    
    matched = {c['pdb_id']: c for c in compounds[:70]}
    smiles_list = [matched.get(pid, {}).get('smiles', '') for pid in pdb_ids]
    
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=512)
    X_ecfp = []
    for smi in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                X_ecfp.append(np.array(fpgen.GetFingerprintAsNumPy(mol), dtype=np.float32))
            else:
                X_ecfp.append(np.zeros(512, dtype=np.float32))
        except:
            X_ecfp.append(np.zeros(512, dtype=np.float32))
    X_ecfp = np.stack(X_ecfp)
    
    # Save final cache
    data = {
        'X_raw': X_raw,
        'X_filt': np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0),
        'X_ecfp': X_ecfp,
        'y_pkd': y_pkd,
        'pdb_ids': pdb_ids,
        'smiles_list': smiles_list,
        'n_train': 30,
        'n_val': 20,
        'n_test': 20,
    }
    
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Saved to {CACHE_FILE}")
    print(f"Final shapes: X_raw={X_raw.shape}, X_ecfp={X_ecfp.shape}")

print("Stage complete!")