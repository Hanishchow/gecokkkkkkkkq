#!/usr/bin/env python3
"""
Download remaining PDB files for training.
"""

import requests
import pickle
from pathlib import Path
import time
import sys

PDB_DIR = Path("CACHE_DIR / lp_pdb_files")

# Load missing PDB IDs
with open('CACHE_DIR / geock_training_data_no2016.pkl', 'rb') as f:
    train_list = pickle.load(f)

with open('training_pocket_features.pkl', 'rb') as f:
    pocket_data = pickle.load(f)

existing = set(pocket_data['pdb_ids'])
missing = [t['pdb_id'] for t in train_list if t['pdb_id'] not in existing]

print(f"Missing: {len(missing)}")
print(f"Will download up to 2000...")

# Download with rate limiting
SUCCESS = 0
FAILED = []
MAX_DOWNLOAD = 2000

for i, pdb_id in enumerate(missing[:MAX_DOWNLOAD]):
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            path = PDB_DIR / f"{pdb_id}.pdb"
            path.write_bytes(r.content)
            SUCCESS += 1
    except Exception as e:
        FAILED.append(pdb_id)
    
    # Progress
    if (i + 1) % 100 == 0:
        print(f"{i+1}/{MAX_DOWNLOAD}: {SUCCESS} OK, {len(FAILED)} failed")
    
    # Rate limit (0.2s = 5/sec max)
    time.sleep(0.2)

print(f"\n=== SUMMARY ===")
print(f"Downloaded: {SUCCESS} / {MAX_DOWNLOAD}")
print(f"Failed: {len(FAILED)}")

# Save failed list
if FAILED:
    with open('pdb_download_failed.pkl', 'wb') as f:
        pickle.dump(FAILED, f)
    print(f"Failed list saved to pdb_download_failed.pkl")