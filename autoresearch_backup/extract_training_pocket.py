#!/usr/bin/env python3
"""
Extract pocket features for PDBBind training data.
"""

import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.insert(0, '/home/chow/autoresearch')
from feature_extractor import protein_pocket_features

CACHE_DIR = Path("/home/chow/.cache/geock_autoresearch")
PDB_DIR = CACHE_DIR / "lp_pdb_files"

def extract_training_pocket_features():
    print("Loading training data...")
    with open(CACHE_DIR / "geock_training_data_no2016.pkl", 'rb') as f:
        train_list = pickle.load(f)
    
    print(f"Total: {len(train_list)} entries")
    
    # Extract pocket features for entries with PDB files
    print("\nExtracting pocket features...")
    pocket_features = {}
    missing = 0
    
    for t in tqdm(train_list):
        pdb_id = t['pdb_id']
        pdb_path = t.get('pdb_path')
        
        if pdb_path and Path(pdb_path).exists():
            feats = protein_pocket_features(str(pdb_path))
            pocket_features[pdb_id] = feats
        else:
            missing += 1
    
    print(f"\nExtracted: {len(pocket_features)}")
    print(f"Missing PDB: {missing}")
    
    # Save
    output = {
        'pocket_features': pocket_features,
        'pdb_ids': list(pocket_features.keys())
    }
    
    with open('training_pocket_features.pkl', 'wb') as f:
        pickle.dump(output, f)
    
    print(f"Saved to training_pocket_features.pkl")
    
    return pocket_features

if __name__ == "__main__":
    extract_training_pocket_features()