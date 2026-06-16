"""Compute comprehensive fingerprints for 24K dataset + combine with pocket"""
import pickle, numpy as np
from pathlib import Path
from collections import Counter
from rdkit import Chem
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator
import warnings; warnings.filterwarnings('ignore')

BACKUP = Path(r'C:\Users\yakka\Downloads\Hanish_Brain_Vault\autoresearch_backup')

# Load 24K dataset
nf = pickle.load(open(BACKUP / 'lp_new_features_8k.pkl', 'rb'))
print(f'Loaded {len(nf)} entries', flush=True)

# Load pocket features
pocket = pickle.load(open(BACKUP / 'pocket_features_full.pkl', 'rb'))
print(f'Pocket features: {len(pocket)}', flush=True)

# Load existing Phase 2 data for comparison
X19 = np.load(BACKUP / 'phase2_X.npy')
y19 = np.load(BACKUP / 'phase2_y.npy')
pdb19 = pickle.load(open(BACKUP / 'phase2_pdb_ids.pkl', 'rb'))
print(f'Phase 2: {len(X19)} entries, {len(set(pdb19))} unique PDBs', flush=True)

# Check overlap between new dataset and Phase 2 by PDB ID
new_pdbs = set(e['pdb_id'] for e in nf)
old_pdbs = set(pdb19)
overlap = len(new_pdbs & old_pdbs)
only_new = len(new_pdbs - old_pdbs)
print(f'PDB overlap: {overlap}, Only in new: {only_new}', flush=True)

# Compute features
print('\nComputing 2D fingerprints from SMILES...', flush=True)
fgen2 = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=512)
fgen3 = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=256)

X_all = np.zeros((len(nf), 1056), dtype=np.float32)  # 982 + 24 physics + 50 pocket
y_all = np.zeros(len(nf), dtype=np.float32)
pocket_count = 0
rdkit_errors = 0
smiles_errors = 0

for i, entry in enumerate(nf):
    y_all[i] = entry['affinity']
    smiles = entry['smiles']
    pid = entry['pdb_id']
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            smiles_errors += 1
            # Use ECFP from file + zeros for MACCS/FCFP/RDKit
            X_all[i, :512] = entry['ecfp']
            X_all[i, 512:679] = 0  # MACCS = 0
            X_all[i, 679:935] = 0  # FCFP = 0
            X_all[i, 935:982] = 0  # RDKit = 0
        else:
            # ECFP4
            X_all[i, :512] = np.array(fgen2.GetFingerprintAsNumPy(mol), dtype=np.float32)
            # MACCS
            X_all[i, 512:679] = np.array(MACCSkeys.GenMACCSKeys(mol), dtype=np.float32)
            # FCFP4
            X_all[i, 679:935] = np.array(fgen3.GetFingerprintAsNumPy(mol), dtype=np.float32)
            # RDKit descriptors (47, may be zeros if fails)
            try:
                from rdkit.Chem import Descriptors
                desc = []
                for d in Descriptors.descList[:47]:
                    desc.append(d[1](mol))
                X_all[i, 935:982] = np.array(desc, dtype=np.float32)
            except:
                rdkit_errors += 1
                X_all[i, 935:982] = 0
    except Exception as e:
        smiles_errors += 1
        X_all[i, :512] = entry['ecfp']
        X_all[i, 512:982] = 0
    
    # Physics features (24-dim, from dataset)
    X_all[i, 982:1006] = entry['physics']
    
    # Pocket features
    if pid in pocket:
        X_all[i, 1006:] = pocket[pid]
        pocket_count += 1
    
    if (i+1) % 5000 == 0:
        print(f'  Processed {i+1}/{len(nf)}...', flush=True)

print(f'Done. SMILES errors: {smiles_errors}, RDKit errors: {rdkit_errors}', flush=True)
print(f'With pocket features: {pocket_count}/{len(nf)}', flush=True)
print(f'X shape: {X_all.shape}, y shape: {y_all.shape}', flush=True)

# ECFP-only uniqueness
ecfp_only = X_all[:, :512]
ecfp_hashes = [hash(row.tobytes()) for row in ecfp_only]
print(f'Unique ECFP (ECFP4-only): {len(set(ecfp_hashes))} / {len(X_all)}', flush=True)

# Full feature uniqueness
full_hashes = [hash(row.tobytes()) for row in X_all]
print(f'Unique 1056-dim vectors: {len(set(full_hashes))} / {len(X_all)}', flush=True)

# Save
np.save(BACKUP / 'all24k_X.npy', X_all)
np.save(BACKUP / 'all24k_y.npy', y_all)
pdb_list = [e['pdb_id'] for e in nf]
pickle.dump(pdb_list, open(BACKUP / 'all24k_pdb_ids.pkl', 'wb'))

# Also save a reduced version matching Phase 2 dim (982, no physics)
X_982 = X_all[:, :982]
np.save(BACKUP / 'all24k_X_982dim.npy', X_982)

print(f'\nSaved: all24k_X.npy ({X_all.shape})', flush=True)
print(f'Saved: all24k_X_982dim.npy ({X_982.shape})', flush=True)
print(f'Saved: all24k_y.npy ({y_all.shape})', flush=True)
print(f'Saved: all24k_pdb_ids.pkl ({len(pdb_list)} entries)', flush=True)
