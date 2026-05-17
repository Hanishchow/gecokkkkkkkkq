#!/usr/bin/env python3
"""
GEOCK Docking - Score protein-ligand complexes
"""

import sys
import pickle
import numpy as np
import argparse
from pathlib import Path

# Colors
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
CYAN = '\033[96m'
BOLD = '\033[1m'
RESET = '\033[0m'

def load_hybrid_model():
    print(f"{GREEN}Loading hybrid model...{RESET}", end=" ")
    
    # Try multiple paths for Docker compatibility
    possible_paths = [
        Path('/app/models/geock_model_hybrid.pkl'),
        Path('./models/geock_model_hybrid.pkl'),
        Path('../models/geock_model_hybrid.pkl'),
    ]
    
    model_path = None
    for p in possible_paths:
        if p.exists():
            model_path = p
            break
    
    if model_path is None:
        print(f"{YELLOW}not found{RESET}")
        return None, "Model not found"
    
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        print(f"{GREEN}✓{RESET}")
        return model_data, None
    except Exception as e:
        print(f"{YELLOW}✗{RESET}")
        return None, str(e)

def compute_physics(pdb_path):
    """Compute physics features from PDB file."""
    import sys
    # Try multiple paths for Docker compatibility
    possible_paths = ['/app/geock2', './geock2', '../geock2']
    for p in possible_paths:
        if Path(p).exists():
            sys.path.insert(0, p)
            break
    
    from patch_parse import parse_pocket_and_ligand
    from score_compound import _compute_physics_features
    
    try:
        rec_coords, rec_types, lig_coords, lig_types, _, _ = parse_pocket_and_ligand(pdb_path, cutoff=10.0)
        center = rec_coords.mean(axis=0)
        phys = _compute_physics_features(lig_coords, lig_types, rec_coords, rec_types, center)
        return phys, None
    except Exception as e:
        return None, str(e)

def predict(smiles, pdb_path, model_data):
    """Predict affinity with PDB physics features."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from sklearn.feature_selection import SelectKBest, f_regression
    
    # ECFP from SMILES
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, "Invalid SMILES"
    
    ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=512)
    ecfp_arr = np.array(ecfp).reshape(1, -1)
    
    # Physics from PDB
    phys, err = compute_physics(pdb_path)
    if err:
        return None, f"PDB error: {err}"
    phys = phys.reshape(1, -1)
    
    # Transform features
    sel_ecfp = model_data['sel_ecfp']
    sel_phys = model_data['sel_phys']
    
    X_ecfp = sel_ecfp.transform(ecfp_arr)
    X_phys = sel_phys.transform(phys)
    
    # Combine (400 + 50)
    X = np.hstack([X_ecfp, X_phys])
    
    # Predict
    model = model_data['xgb']
    pred = model.predict(X)[0]
    
    return pred, None

def main():
    parser = argparse.ArgumentParser(description='GEOCK Docking - Score protein-ligand')
    parser.add_argument('pdb', help='PDB file path')
    parser.add_argument('smiles', help='Ligand SMILES')
    parser.add_argument('--no-banner', action='store_true', help='Skip banner')
    args = parser.parse_args()
    
    if not args.no_banner:
        print(f"""
{BLUE}{BOLD}
    ╔═══════════════════════════════════════════════════════╗
    ║   {CYAN}GEOCK{RESET} {BLUE}Docking - Protein-Ligand Scoring{RESET}       ║
    ║   {YELLOW}Hybrid ECFP + Physics Features{RESET}                     ║
    ╚═══════════════════════════════════════════════════════╝
{RESET}
""")
    
    # Load model
    model_data, err = load_hybrid_model()
    if err:
        print(f"{YELLOW}Error: {err}{RESET}")
        sys.exit(1)
    
    # Check PDB exists
    if not Path(args.pdb).exists():
        print(f"{YELLOW}Error: PDB file not found: {args.pdb}{RESET}")
        sys.exit(1)
    
    # Predict
    print(f"\n{CYAN}PDB:{RESET} {args.pdb}")
    print(f"{CYAN}Ligand:{RESET} {args.smiles}")
    
    pred, err = predict(args.smiles, args.pdb, model_data)
    
    if err:
        print(f"\n{YELLOW}Error: {err}{RESET}")
        sys.exit(1)
    
    # Output
    kd_nm = 10**(-pred) * 1e9
    if kd_nm >= 1000:
        kd_str = f"{kd_nm/1000:.2f} μM"
    elif kd_nm >= 1:
        kd_str = f"{kd_nm:.2f} nM"
    else:
        kd_str = f"{kd_nm*1000:.2f} pM"
    
    print(f"\n{BOLD}Predicted pKd:{RESET} {GREEN}{pred:.2f}{RESET}")
    print(f"{BOLD}Estimated Kd:{RESET}  {kd_str}")
    
    # Interpretation
    if pred >= 9:
        strength = "Very Strong"
    elif pred >= 7:
        strength = "Strong"
    elif pred >= 5:
        strength = "Moderate"
    elif pred >= 3:
        strength = "Weak"
    else:
        strength = "Very Weak"
    
    print(f"{BOLD}Binding:{RESET} {strength}")

if __name__ == "__main__":
    main()
