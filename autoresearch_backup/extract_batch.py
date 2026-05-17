#!/usr/bin/env python3
"""Extract pocket features in small batch."""
import pickle, sys
from pathlib import Path
sys.path.insert(0, '/home/chow/autoresearch')
from feature_extractor import protein_pocket_features

PDB_DIR = Path("CACHE_DIR / lp_pdb_files")
files = list(PDB_DIR.glob("*.pdb"))[:5000]
print(f"Processing {len(files)} files...")

feats = {}
for i, f in enumerate(files):
    feats[f.stem] = protein_pocket_features(str(f))
    if (i+1) % 1000 == 0:
        print(f"{i+1} done")

with open('pocket_batch1.pkl', 'wb') as f:
    pickle.dump(feats, f)
print(f"Saved {len(feats)} features")