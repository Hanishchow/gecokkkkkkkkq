#!/usr/bin/env python3
"""
Download PDB files for training data from RCSB PDB.
"""

import requests
import os
from pathlib import Path
from tqdm import tqdm
import time

PDB_DIR = Path("CACHE_DIR / lp_pdb_files")
OUTPUT_DIR = PDB_DIR.parent / "pdb_downloads"
OUTPUT_DIR.mkdir(exist_ok=True)

def download_pdb(pdb_id, output_dir):
    """Download PDB file from RCSB."""
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            path = output_dir / f"{pdb_id}.pdb"
            path.write_bytes(r.content)
            return True
    except Exception as e:
        pass
    return False

# Get missing PDB IDs
import pickle
with open('CACHE_DIR / geock_training_data_no2016.pkl', 'rb') as f:
    train_list = pickle.load(f)

with open('training_pocket_features.pkl', 'rb') as f:
    pocket_data = pickle.load(f)

existing = set(pocket_data['pdb_ids'])
missing = [t['pdb_id'] for t in train_list if t['pdb_id'] not in existing]

print(f"Need to download: {len(missing)} PDB files")
print(f"Output directory: {OUTPUT_DIR}")

# Download in batches with rate limiting
BATCH_SIZE = 100
SUCCESS = 0
FAILED = []

for i, pdb_id in enumerate(missing[:500]):  # Try first 500
    if download_pdb(pdb_id, OUTPUT_DIR):
        SUCCESS += 1
    else:
        FAILED.append(pdb_id)
    
    # Rate limit
    if i % 10 == 0:
        time.sleep(0.5)
    
    if (i + 1) % 50 == 0:
        print(f"Progress: {i+1}/{min(500, len(missing))}, Success: {SUCCESS}")

print(f"\nDownloaded: {SUCCESS} / {min(500, len(missing))}")
print(f"Failed: {len(FAILED)}")

# Show some failed
if FAILED:
    print(f"Some failed: {FAILED[:10]}")