"""Quick test: process 10K BindingDB entries"""
import pyarrow.parquet as pq
import numpy as np
import time
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys

base = r'C:\Users\yakka\Downloads\Hanish_Brain_Vault\autoresearch_backup'

def compute_982_fp(smiles):
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
    # last 3 bits (979-981) are padding = 0
    return fp

fpath = base + '/binddb_train-00000-of-00002.parquet'
pf = pq.ParquetFile(fpath)

# Test with 10K
print("Processing 10K test...")
t0 = time.time()
batch = next(pf.iter_batches(batch_size=10000))
df = batch.to_pandas()
print(f"Loaded {len(df)} rows in {time.time()-t0:.1f}s")

valid = 0
t1 = time.time()
for i, row in df.iterrows():
    fp = compute_982_fp(row['ligand'])
    if fp is not None:
        valid += 1
    if (i+1) % 1000 == 0:
        elapsed = time.time() - t1
        rate = (i+1) / elapsed if elapsed > 0 else 0
        print(f"  {i+1}/{len(df)}: {valid} valid, {rate:.0f} mol/s")

total_time = time.time() - t1
print(f"\nTest complete: {valid}/{len(df)} valid molecules")
print(f"Total time: {total_time:.1f}s, Rate: {len(df)/total_time:.0f} mol/s")

# Estimate full processing time
rate = len(df) / total_time
total_mols = 972285
est_hours = total_mols / rate / 3600
print(f"Rate: {rate:.0f} mol/s")
print(f"Estimated full processing: {total_mols/rate:.0f}s = {est_hours:.1f} hours")
