"""Process BindingDB: compute 2D fingerprints in parallel with multiprocessing"""
import pyarrow.parquet as pq
import numpy as np
import pandas as pd
import pickle
import time
import multiprocessing as mp
from functools import partial
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
    fp[512:679] = np.array(maccs)
    fp[679:979] = np.array(fcfp)
    return fp

def process_smiles_list(smiles_list, ic50_list):
    """Process a list of SMILES and return valid fingerprints and values"""
    fps, ys, valid_smi = [], [], []
    for smi, ic50 in zip(smiles_list, ic50_list):
        if np.isnan(ic50):
            continue
        fp = compute_982_fp(smi)
        if fp is not None:
            fps.append(fp)
            ys.append(ic50)
            valid_smi.append(smi)
    return np.array(fps, dtype=np.uint8), np.array(ys, dtype=np.float64), valid_smi

def load_phase2_hashes():
    X19 = np.load(base + '/phase2_X.npy')
    p2_ecfp = X19[:, :512]
    p2_hashes = set(hash(row.tobytes()) for row in p2_ecfp)
    print(f"Phase 2: {len(X19)} entries, {len(p2_hashes)} unique ECFP")
    return p2_hashes

def process_chunk_parallel(args):
    chunk_df, chunk_id = args
    smiles_list = chunk_df['ligand'].tolist()
    ic50_list = chunk_df['ic50'].tolist()
    fps, ys, _ = process_smiles_list(smiles_list, ic50_list)
    print(f"  Chunk {chunk_id}: {len(fps)} valid / {len(chunk_df)} total")
    return fps, ys

if __name__ == '__main__':
    mp.freeze_support()
    t0 = time.time()

    # Load Phase 2 hashes for dedup
    print("Loading Phase 2 data...")
    X19 = np.load(base + '/phase2_X.npy')
    p2_hashes = set(hash(row.tobytes()) for row in X19[:, :512])
    print(f"Phase 2: {len(X19)} entries, {len(p2_hashes)} unique ECFP")

    files = [
        base + '/binddb_train-00000-of-00002.parquet',
        base + '/binddb_train-00001-of-00002.parquet',
    ]

    # Read all data first (fast with pyarrow)
    print("Reading parquet files...")
    all_dfs = []
    for fpath in files:
        t1 = time.time()
        df = pq.read_table(fpath).to_pandas()
        print(f"  {fpath}: {len(df)} rows in {time.time()-t1:.1f}s")
        all_dfs.append(df)

    df_all = pd.concat(all_dfs, ignore_index=True)
    print(f"Total: {len(df_all)} entries")

    # Process in parallel chunks
    n_cores = mp.cpu_count()
    chunk_size = max(10000, len(df_all) // (n_cores * 4))
    chunks = [(df_all.iloc[i:i+chunk_size], i//chunk_size) 
              for i in range(0, len(df_all), chunk_size)]
    print(f"Processing {len(chunks)} chunks with {n_cores} cores...")

    t1 = time.time()
    with mp.Pool(n_cores) as pool:
        results = pool.map(process_chunk_parallel, chunks)

    all_fps = []
    all_ys = []
    for fps, ys in results:
        all_fps.append(fps)
        all_ys.append(ys)

    X_all = np.concatenate(all_fps)
    y_all = np.concatenate(all_ys)
    print(f"\nProcessed {len(X_all)} valid fingerprints in {time.time()-t1:.0f}s")
    print(f"Rate: {len(X_all)/(time.time()-t1):.0f} mol/s")

    # Dedup vs Phase 2
    new_mask = np.ones(len(X_all), dtype=bool)
    unique_hashes = set()
    for i in range(len(X_all)):
        h = hash(X_all[i, :512].tobytes())
        if h in p2_hashes or h in unique_hashes:
            new_mask[i] = False
        else:
            unique_hashes.add(h)

    X_new = X_all[new_mask]
    y_new = y_all[new_mask]
    dup_count = len(X_all) - len(X_new)
    print(f"\nDeduplication:")
    print(f"  Total processed: {len(X_all)}")
    print(f"  Duplicates (in Phase 2 or BindingDB): {dup_count}")
    print(f"  New molecules: {len(X_new)}")
    print(f"  y stats: mean={y_new.mean():.3f}, std={y_new.std():.3f}")

    # Save
    np.save(base + '/bindingdb_X.npy', X_all)
    np.save(base + '/bindingdb_y.npy', y_all)
    np.save(base + '/bindingdb_X_new.npy', X_new)
    np.save(base + '/bindingdb_y_new.npy', y_new)
    print(f"\nSaved: bindingdb_X.npy ({X_all.shape}), bindingdb_y.npy ({y_all.shape})")
    print(f"Saved: bindingdb_X_new.npy ({X_new.shape}), bindingdb_y_new.npy ({y_new.shape})")
    print(f"Total time: {(time.time()-t0)/60:.1f} min")
