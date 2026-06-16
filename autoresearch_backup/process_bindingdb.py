"""Process BindingDB: compute 2D fingerprints in chunks, filter, dedup"""
import pyarrow.parquet as pq
import numpy as np
import pickle
import time
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
from rdkit.Avalon import pyAvalonTools

base = r'C:\Users\yakka\Downloads\Hanish_Brain_Vault\autoresearch_backup'

def compute_982_fp(smiles):
    """Compute ECFP4(512) + MACCS(170) + FCFP4(300) = 982-dim fingerprint"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, 4, nBits=512)
    maccs = MACCSkeys.GenMACCSKeys(mol)
    fcfp = AllChem.GetMorganFingerprintAsBitVect(mol, 4, nBits=300, useFeatures=True)
    fp = np.zeros(982, dtype=np.uint8)
    fp[:512] = np.array(ecfp)
    fp[512:679] = np.array(maccs)  # 167 bits
    fp[679:979] = np.array(fcfp)   # 300 bits
    return fp

def process_chunk(chunk_df, chunk_idx):
    fps, y_vals, smiles_list = [], [], []
    for i, row in chunk_df.iterrows():
        smi = row['ligand']
        ic50 = row['ic50']
        fp = compute_982_fp(smi)
        if fp is not None and not np.isnan(ic50):
            fps.append(fp)
            y_vals.append(ic50)
            smiles_list.append(smi)
    X = np.array(fps, dtype=np.uint8)
    y = np.array(y_vals, dtype=np.float64)
    print(f"  Chunk {chunk_idx}: {len(fps)} valid / {len(chunk_df)} total")
    return X, y, smiles_list

# Load Phase 2 fingerprints for dedup
print("Loading Phase 2 data for dedup reference...")
X19 = np.load(base + '/phase2_X.npy')
y19 = np.load(base + '/phase2_y.npy')
# ECFP-only (first 512 bits) for hashing
p2_ecfp = X19[:, :512]
p2_hashes = set(hash(row.tobytes()) for row in p2_ecfp)
print(f"Phase 2: {len(X19)} entries, {len(p2_hashes)} unique ECFP hashes")

# Process both train parquet files
chunksize = 50000
files = [
    base + '/binddb_train-00000-of-00002.parquet',
    base + '/binddb_train-00001-of-00002.parquet',
]

all_X, all_y = [], []
new_mols = 0
dup_mols = 0
chunk_idx = 0

for fpath in files:
    pf = pq.ParquetFile(fpath)
    total = pf.metadata.num_rows
    print(f"\nProcessing {fpath} ({total} rows)...")
    for batch in pf.iter_batches(batch_size=chunksize):
        df = batch.to_pandas()
        X_chunk, y_chunk, smiles = process_chunk(df, chunk_idx)
        chunk_idx += 1
        # Check for new molecules
        for i in range(len(X_chunk)):
            h = hash(X_chunk[i, :512].tobytes())
            if h in p2_hashes:
                dup_mols += 1
            else:
                new_mols += 1
                all_X.append(X_chunk[i])
                all_y.append(y_chunk[i])
                p2_hashes.add(h)  # dedup within BindingDB too
        print(f"  Running: {new_mols} new, {dup_mols} dup")

X_all = np.array(all_X, dtype=np.uint8)
y_all = np.array(all_y, dtype=np.float64)
print(f"\n=== BindingDB Processing Complete ===")
print(f"Total valid fingerprints: {len(all_X)}")
print(f"New molecules (not in Phase 2): {new_mols}")
print(f"Duplicates skipped: {dup_mols}")
print(f"X shape: {X_all.shape}")
print(f"y stats: mean={y_all.mean():.3f}, std={y_all.std():.3f}, min={y_all.min():.3f}, max={y_all.max():.3f}")

# Save
np.save(base + '/bindingdb_X.npy', X_all)
np.save(base + '/bindingdb_y.npy', y_all)
print("\nSaved to bindingdb_X.npy and bindingdb_y.npy")
