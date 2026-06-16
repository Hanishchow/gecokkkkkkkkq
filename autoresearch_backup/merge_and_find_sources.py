"""Merge Phase 2 + 24K + ChEMBL data, find overlaps, identify new data sources"""
import pickle, numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from rdkit import Chem
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator
import warnings; warnings.filterwarnings('ignore')

BACKUP = Path(r'C:\Users\yakka\Downloads\Hanish_Brain_Vault\autoresearch_backup')

# ===== Load all datasets =====
# 1. Phase 2 (19K, 982-dim)
X19 = np.load(BACKUP / 'phase2_X.npy')
y19 = np.load(BACKUP / 'phase2_y.npy')
pdb19 = pickle.load(open(BACKUP / 'phase2_pdb_ids.pkl', 'rb'))
ecfp19 = [hash(row.tobytes()) for row in X19[:, :512]]
print(f'Phase 2: {len(X19)} entries, {len(set(ecfp19))} unique ECFP')

# 2. 24K dataset (computed from lp_new_features_8k)
X24 = np.load(BACKUP / 'all24k_X.npy')
y24 = np.load(BACKUP / 'all24k_y.npy')
pdb24 = pickle.load(open(BACKUP / 'all24k_pdb_ids.pkl', 'rb'))
ecfp24 = [hash(row.tobytes()) for row in X24[:, :512]]
print(f'24K data: {len(X24)} entries, {len(set(ecfp24))} unique ECFP')

# Overlap between 19K and 24K at ECFP level
set19 = set(ecfp19)
set24 = set(ecfp24)
print(f'  Overlap: {len(set19 & set24)}, Only in Phase2: {len(set19 - set24)}, Only in 24K: {len(set24 - set19)}')

# 3. ChEMBL data
# Need to get from WSL or check if available locally
import subprocess
result = subprocess.run(['wsl', 'python3', '-c', '''
import pickle, os
base = "/home/chow/.cache/geock_autoresearch"
for fname in ["chembl_v2.pkl", "chembl_more.pkl"]:
    data = pickle.load(open(os.path.join(base, fname), "rb"))
    keys = list(data[0].keys()) if data and isinstance(data[0], dict) else []
    print(f"{fname}: {len(data)} entries, keys={keys}")
    ecfps = [hash(e["ecfp"].tobytes()) for e in data if "ecfp" in e]
    smiles = [e["smiles"] for e in data if "smiles" in e]
    print(f"  Unique ECFP: {len(set(ecfps))}, With SMILES: {len(smiles)}")
'''], capture_output=True, text=True, timeout=30)
print('\n=== ChEMBL data ===')
print(result.stdout)
if result.stderr:
    print('ERR:', result.stderr[:500])

# 4. Check if there are MORE data sources
result2 = subprocess.run(['wsl', 'python3', '-c', '''
import pickle, os
base = "/home/chow/.cache/geock_autoresearch"
# Check for any other pickle files with training data
import glob
for f in sorted(os.listdir(base)):
    if f.endswith('.pkl') or f.endswith('.npy'):
        size = os.path.getsize(os.path.join(base, f))
        if size > 500000:  # >500KB
            print(f"{f}: {size/1e6:.1f}MB")
'''], capture_output=True, text=True, timeout=30)
print('\n=== Large files on WSL ===')
print(result2.stdout)
if result2.stderr:
    print('ERR:', result2.stderr[:500])

print('\n=== SUMMARY: What we have ===')
print(f'Phase 2 (ECFP+MACCS+FCFP+RDKit):          19,087 entries, {len(set19)} unique')
print(f'lp_new_features_8k (SMILES available):     24,067 entries, {len(set24)} unique')
print(f'  -> But ECFP overlap with Phase 2:        {len(set19 & set24)} of {len(set24)}')
print(f'  -> Truly new molecules:                  {len(set24 - set19)}')
print(f'Combined unique molecules:                 {len(set19 | set24)}')
print(f'With pocket features:                      18,832')

print('\n=== WHERE TO GET MORE DATA ===')
print('To improve beyond R=0.731, we need NEW molecules (chemical diversity):')
print()
print('1. PDBbind v2020 Refined Set (non-core)')
print('   - Location: http://www.pdbbind.org.cn/')
print('   - ~5,000 high-quality complexes explicitly excluding CASF core sets')
print('   - Would add ~3-4K new molecules to our 15K')
print()
print('2. PDBbind v2020 General Set')
print('   - Location: http://www.pdbbind.org.cn/')
print('   - ~19,000 complexes (includes refined + core + general)')
print('   - BUT: need to carefully exclude CASF entries')
print()
print('3. BindingDB')
print('   - Location: https://www.bindingdb.org/')
print('   - ~2.6M measurements, ~1.4M compounds, ~7K protein targets')
print('   - Has SMILES + Ki/Kd values')
print('   - Best source for truly new chemical space')
print()
print('4. ChEMBL (we have 4K already - need to see if they add anything)')
print('   - Location: https://www.ebi.ac.uk/chembl/')
print('   - ~2M compounds with bioactivity data')
print('   - Has SMILES, can compute fingerprints')
print()
print('5. PDBbind v2021+ (newer structures, 2020-2026)')
print('   - Newer PDB entries not in our training set')
print('   - Would provide temporal extrapolation test')
print()
print('6. DUD-E / DEKOIS (decoy sets for virtual screening)')
print('   - Not directly useful for affinity prediction')
print()
print('RECOMMENDATION: BindingDB or ChEMBL download for maximum new chemical diversity.')
