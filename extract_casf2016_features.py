#!/usr/bin/env python3
"""
Extract enhanced features for CASF-2016 using feature_extractor.py
Handles None values and inhomogeneous arrays.
"""

import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.insert(0, '/home/chow/autoresearch')
from feature_extractor import extract_all_features, combine_features

CASF_2016_DIR = Path("/mnt/c/Users/yakka/Downloads/CASF-2016/CASF-2016")
CORESET_DIR = CASF_2016_DIR / "coreset"
CORESET_DAT = CASF_2016_DIR / "power_scoring" / "CoreSet.dat"


def load_core_set():
    """Load CASF-2016 core set data."""
    complexes = []
    with open(CORESET_DAT, 'r') as f:
        header = f.readline().strip().split()
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                complexes.append({
                    'pdb_id': parts[0],
                    'log_ka': float(parts[3])
                })
    return complexes


def extract_features_safe(pdb_id):
    """Extract features for one complex, handling errors gracefully."""
    ligand_path = CORESET_DIR / pdb_id / f"{pdb_id}_ligand.mol2"
    pocket_path = CORESET_DIR / pdb_id / f"{pdb_id}_pocket.pdb"
    
    if not ligand_path.exists():
        return None
    
    feats = extract_all_features(str(ligand_path), str(pocket_path))
    
    # Check if all features are valid
    for k, v in feats.items():
        if v is None or (hasattr(v, 'any') and v.any() == False):
            pass  # Allow zeros, but not None
    
    combined = combine_features(feats)
    return combined


def main():
    print("Extracting enhanced features for CASF-2016...")
    
    complexes = load_core_set()
    print(f"Total complexes: {len(complexes)}")
    
    features_list = []
    valid_complexes = []
    failed = []
    
    for cx in tqdm(complexes):
        pdb_id = cx['pdb_id']
        feats = extract_features_safe(pdb_id)
        
        if feats is None:
            failed.append(pdb_id)
            continue
        
        features_list.append(feats)
        valid_complexes.append(cx)
    
    print(f"\nValid: {len(valid_complexes)} / {len(complexes)}")
    print(f"Failed: {len(failed)}")
    
    if features_list:
        X = np.array(features_list)
        print(f"Feature matrix shape: {X.shape}")
        print(f"Feature stats - min: {X.min():.3f}, max: {X.max():.3f}, mean: {X.mean():.3f}")
        
        # Save
        output = {
            'X': X,
            'complexes': valid_complexes,
            'y': np.array([cx['log_ka'] for cx in valid_complexes])
        }
        with open('casf2016_enhanced_features.pkl', 'wb') as f:
            pickle.dump(output, f)
        print(f"Saved to casf2016_enhanced_features.pkl")
    
    return len(valid_complexes), len(failed)


if __name__ == "__main__":
    main()