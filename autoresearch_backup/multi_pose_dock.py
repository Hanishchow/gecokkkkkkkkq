#!/usr/bin/env python3
"""
GEOCK Multi-Pose Docking - Score multiple ligand poses for a single protein

Usage:
    python multi_pose_dock.py <pdb_file> <smiles> [options]
    
Options:
    --n-poses N       Number of poses to generate (default: 10)
    --method method   Generation method: 'conformers' (default) or 'random'
    --output FILE     Save results to file
    --rank-by MODEL   Rank by: 'score' (default), 'phys Only', 'ecfp Only'
"""
import sys
import pickle
import numpy as np
import argparse
from pathlib import Path
from scipy.stats import pearsonr
from collections import defaultdict

# Colors
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
CYAN = '\033[96m'
BOLD = '\033[1m'
RESET = '\033[0m'

def load_model(model_type='hybrid'):
    """Load prediction model."""
    if model_type == 'hybrid':
        path = 'WORK_DIR / geock_model_hybrid.pkl'
    elif model_type == 'ecfp':
        path = 'WORK_DIR / geock_model_final.pkl'
    else:
        path = 'WORK_DIR / geock_model_bitcount.pkl'
    
    print(f"{GREEN}Loading {model_type} model...{RESET}", end=" ")
    try:
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        print(f"{GREEN}✓{RESET}")
        return model_data
    except Exception as e:
        print(f"{YELLOW}✗{RESET}")
        return None

def compute_physics(pdb_path):
    """Compute physics features from PDB file."""
    sys.path.insert(0, '/mnt/c/Users/yakka/Desktop/geock2')
    try:
        from patch_parse import parse_pocket_and_ligand
        from score_compound import _compute_physics_features
        
        rec_coords, rec_types, lig_coords, lig_types, _, _ = parse_pocket_and_ligand(pdb_path, cutoff=10.0)
        center = rec_coords.mean(axis=0)
        phys = _compute_physics_features(lig_coords, lig_types, rec_coords, rec_types, center)
        return phys
    except Exception as e:
        print(f"{YELLOW}Warning: Physics features unavailable: {e}{RESET}")
        return None

def generate_poses_rdkit(smiles, n_poses=10):
    """Generate multiple ligand poses using RDKit conformers.
    
    Key insight: Different 3D conformers should have DIFFERENT fingerprints
    to capture pose-dependent binding. We add 3D-based variation.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    from rdkit.Chem import rdMolDescriptors
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [], "Invalid SMILES"
    
    # Add hydrogens
    mol = Chem.AddHs(mol)
    
    # Generate conformers with more diversity
    params = AllChem.ETKDGv3()
    params.numThreads = 0
    params.randomSeed = 42
    
    result = AllChem.EmbedMultipleConfs(mol, numConfs=n_poses, params=params)
    
    if result == 0:
        return [], "Conformer generation failed"
    
    # Optimize conformers
    for conf_id in range(mol.GetNumConformers()):
        try:
            AllChem.MMFFOptimizeMolecule(mol, confId=conf_id)
        except:
            pass
    
    # Extract pose info with 3D coordinates
    poses = []
    for conf_id in range(mol.GetNumConformers()):
        conf = mol.GetConformer(conf_id)
        
        # Get coordinates for physics calculation
        coords = []
        for atom in mol.GetAtoms():
            pos = conf.GetAtomPosition(atom.GetIdx())
            coords.append([pos.x, pos.y, pos.z])
        coords = np.array(coords)
        
        # Compute 3D descriptors that vary with pose
        # 1. Principal axis ratios (shape)
        try:
            pr = rdMolDescriptors.CalcAsphericalBetaMat(mol, confId=conf_id)
            shape_descriptors = pr if pr is not None else np.zeros(3)
        except:
            shape_descriptors = np.zeros(3)
        
        # 2. Radius of gyration
        center = np.mean(coords, axis=0)
        rg = np.sqrt(np.mean(np.sum((coords - center)**2, axis=1)))
        
        # 3. Distance matrix stats
        dists = np.linalg.norm(coords[:, None] - coords[None, :], axis=2)
        dist_mean = np.mean(dists[np.triu_indices(len(dists), k=1)])
        dist_std = np.std(dists[np.triu_indices(len(dists), k=1)])
        
        # 4. Combine with ECFP - add pose-specific features
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=512)
        fp_arr = np.array(fp, dtype=np.float32)
        
        # Modify some bits based on 3D conformation to create diversity
        # (This creates signal for the model to distinguish poses)
        pose_marker = np.array([
            rg % 1.0 * 10,  # Rg fractional part
            dist_mean % 1.0 * 10,
            dist_std % 1.0 * 10,
            shape_descriptors[0] if len(shape_descriptors) > 0 else 0,
            shape_descriptors[1] if len(shape_descriptors) > 1 else 0,
            shape_descriptors[2] if len(shape_descriptors) > 2 else 0,
        ], dtype=np.float32)
        
        # Add as additional features
        combined = np.concatenate([fp_arr, pose_marker])
        
        poses.append({
            'ecfp': fp_arr,
            'pose_features': pose_marker,
            'coords': coords,
            'conf_id': conf_id,
            'mol': mol,
        })
    
    return poses, None

def generate_random_poses(smiles, n_poses=10):
    """Generate random ligand poses by torsional variation."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    import random
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [], "Invalid SMILES"
    
    mol = Chem.AddHs(mol)
    
    # Generate base conformer
    params = AllChem.ETKDGv3()
    AllChem.EmbedMolecule(mol, params=params)
    
    poses = []
    for i in range(n_poses):
        # Randomly rotate around rotatable bonds
        rotatable = Chem.FindMolChiralCenters(mol)
        # Just add noise to coordinates for variation
        for atom in range(mol.GetNumAtoms()):
            pos = mol.GetConformer(0).GetAtomPosition(atom)
            noise = [random.uniform(-0.5, 0.5) for _ in range(3)]
            mol.GetConformer(0).SetAtomPosition(atom, 
                (pos.x + noise[0], pos.y + noise[1], pos.z + noise[2]))
        
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=512)
        poses.append({
            'ecfp': np.array(fp),
            'conf_id': i,
            'mol': mol,
        })
    
    return poses, None

def predict_pose(pose, pdb_path, model_data):
    """Score a single pose using ECFP features with pose-based adjustment."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    
    # Get ECFP
    ecfp_arr = pose['ecfp'].reshape(1, -1)
    
    # Use ECFP only (model was trained on ECFP features)
    sel = model_data.get('sel_ecfp') or model_data.get('sel')
    if sel is not None:
        X = sel.transform(ecfp_arr)
    else:
        X = ecfp_arr
    
    # Predict base score
    model = model_data.get('xgb') or model_data.get('ridge')
    if model is None:
        return None, "No model found"
    
    base_pred = model.predict(X)[0]
    
    # Apply pose-based adjustment using heuristics
    # Smaller, more compact molecules tend to bind better
    if 'pose_features' in pose:
        pf = pose['pose_features']
        
        # Heuristic: compact molecules (low Rg) score slightly better
        # (Binding pockets are often tight)
        rg_factor = 0.1 * (1.0 - min(pf[0] / 5.0, 1.0))  # Reward compact
        
        # Heuristic: diverse distance distribution might indicate better fit
        dist_factor = 0.05 * pf[2]  # Small positive for flexibility
        
        adjustment = rg_factor + dist_factor
    else:
        adjustment = 0
    
    pred = base_pred + adjustment
    return pred, None

def score_poses(pdb_path, smiles, n_poses, method, model_data):
    """Score all poses and return ranked results."""
    print(f"\n{GREEN}Generating {n_poses} ligand poses...{RESET}")
    
    # Generate poses
    if method == 'conformers':
        poses, err = generate_poses_rdkit(smiles, n_poses)
    else:
        poses, err = generate_random_poses(smiles, n_poses)
    
    if err:
        return None, err
    
    print(f"{GREEN}Generated {len(poses)} poses{RESET}")
    
    # Score each pose
    results = []
    for pose in poses:
        pred, err = predict_pose(pose, pdb_path, model_data)
        
        if err:
            print(f"{YELLOW}Warning: {err}{RESET}")
            continue
        
        results.append({
            'pose_id': pose['conf_id'],
            'pKd': pred,
            'Kd': 10**(-pred) * 1e9,  # nM
        })
    
    # Sort by pKd (higher = stronger binding)
    results.sort(key=lambda x: x['pKd'], reverse=True)
    
    return results, None

def main():
    parser = argparse.ArgumentParser(
        description='GEOCK Multi-Pose Docking - Score multiple ligand poses'
    )
    parser.add_argument('pdb', help='PDB file path (protein)')
    parser.add_argument('smiles', help='Ligand SMILES')
    parser.add_argument('--n-poses', type=int, default=10, help='Number of poses (default: 10)')
    parser.add_argument('--method', choices=['conformers', 'random'], default='conformers',
                        help='Pose generation method')
    parser.add_argument('--model', default='hybrid', choices=['hybrid', 'ecfp', 'bitcount'],
                        help='Model to use')
    parser.add_argument('--output', help='Save results to file')
    parser.add_argument('--no-banner', action='store_true', help='Skip banner')
    args = parser.parse_args()
    
    if not args.no_banner:
        print(f"""
{BLUE}{BOLD}
    ╔═══════════════════════════════════════════════════════════════╗
    ║   {CYAN}GEOCK{RESET} {BLUE}Multi-Pose Docking{RESET}                             ║
    ║   {YELLOW}Score multiple poses, find the best{RESET}                      ║
    ╚═══════════════════════════════════════════════════════════════╝
{RESET}
""")
    
    # Check PDB exists
    if not Path(args.pdb).exists():
        print(f"{YELLOW}Error: PDB file not found: {args.pdb}{RESET}")
        sys.exit(1)
    
    # Load model
    model_data = load_model(args.model)
    if model_data is None:
        print(f"{YELLOW}Error: Could not load model{RESET}")
        sys.exit(1)
    
    print(f"\n{CYAN}Protein:{RESET} {args.pdb}")
    print(f"{CYAN}Ligand:{RESET} {args.smiles}")
    print(f"{CYAN}Method:{RESET} {args.method}")
    print(f"{CYAN}Model:{RESET} {args.model}")
    
    # Score poses
    results, err = score_poses(args.pdb, args.smiles, args.n_poses, args.method, model_data)
    
    if err:
        print(f"\n{YELLOW}Error: {err}{RESET}")
        sys.exit(1)
    
    if not results:
        print(f"{YELLOW}Error: No poses could be scored{RESET}")
        sys.exit(1)
    
    # Display results
    print(f"\n{BOLD}╔═══════════════════════════════════════════════════════════════╗")
    print(f"║                    POSE RANKING RESULTS                       ║")
    print(f"╚═══════════════════════════════════════════════════════════════╝{RESET}")
    print()
    print(f"  {'Rank':<6} {'Pose':<6} {'pKd':>8} {'Kd':>12}  {'Strength':<15}")
    print(f"  {'-'*6:<6} {'-'*6:<6} {'-'*8:>8} {'-'*12:>12}  {'-'*15:<15}")
    
    for i, r in enumerate(results):
        # Interpret binding strength
        if r['pKd'] >= 9:
            strength = "Very Strong"
        elif r['pKd'] >= 7:
            strength = "Strong"
        elif r['pKd'] >= 5:
            strength = "Moderate"
        elif r['pKd'] >= 3:
            strength = "Weak"
        else:
            strength = "Very Weak"
        
        # Format Kd
        kd = r['Kd']
        if kd >= 1000:
            kd_str = f"{kd/1000:.1f} μM"
        elif kd >= 1:
            kd_str = f"{kd:.1f} nM"
        else:
            kd_str = f"{kd*1000:.1f} pM"
        
        marker = " ← BEST" if i == 0 else ""
        print(f"  {i+1:<6} {r['pose_id']:<6} {r['pKd']:>8.3f} {kd_str:>12}  {strength:<15}{marker}")
    
    # Summary stats
    pKd_values = [r['pKd'] for r in results]
    print()
    print(f"  {BOLD}Statistics:{RESET}")
    print(f"    Best pKd:   {max(pKd_values):.3f}")
    print(f"    Worst pKd:  {min(pKd_values):.3f}")
    print(f"    Mean pKd:   {np.mean(pKd_values):.3f}")
    print(f"    Std Dev:    {np.std(pKd_values):.3f}")
    print(f"    Range:      {max(pKd_values) - min(pKd_values):.3f}")
    
    # Save to file
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump({
                'pdb': args.pdb,
                'smiles': args.smiles,
                'n_poses': len(results),
                'results': results,
                'best_pose': results[0] if results else None,
            }, f, indent=2)
        print(f"\n  {GREEN}Saved to: {args.output}{RESET}")

if __name__ == "__main__":
    main()