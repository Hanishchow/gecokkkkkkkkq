"""
simple_train.py - Simple linear model baseline for binding affinity
"""

import os
import json
import numpy as np
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_predict, KFold
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))


def compute_physics_features(
    ligand_coords, ligand_types, pocket_coords, pocket_features, center
):
    """Compute physics-based features."""
    features = np.zeros(30)
    
    if len(ligand_coords) == 0 or len(pocket_coords) == 0:
        return features
    
    lig_center = ligand_coords.mean(axis=0)
    dist_lig_center = np.linalg.norm(lig_center - center)
    
    all_dists = np.array([np.linalg.norm(lc - pc) for lc in ligand_coords for pc in pocket_coords])
    
    features[0] = np.exp(-dist_lig_center**2 / 8.0)
    features[1] = np.exp(-dist_lig_center**2 / 32.0)
    features[2] = np.exp(-all_dists.min()**2 / 4.0)
    features[3] = np.exp(-all_dists.mean()**2 / 16.0)
    features[4] = min(0, 2.0 - all_dists.min())**2 if all_dists.min() < 2.0 else 0
    
    for i, d in enumerate([2.0, 4.0, 6.0, 8.0]):
        features[5+i] = np.sum(all_dists < d) / len(all_dists)
    
    features[9] = all_dists.mean()
    features[10] = all_dists.min()
    features[11] = all_dists.max()
    features[12] = all_dists.std()
    
    n_hydro = sum(1 for t in ligand_types if t in ['C', 'S'])
    n_hbond = sum(1 for t in ligand_types if t in ['N', 'O'])
    n_atoms = len(ligand_types)
    features[13] = n_hydro / max(1, n_atoms)
    features[14] = n_hbond / max(1, n_atoms)
    features[15] = n_atoms / 50.0
    features[16] = len(pocket_coords) / 100.0
    features[17] = dist_lig_center
    
    features[18] = np.sin(dist_lig_center / 10.0)
    features[19] = np.cos(dist_lig_center / 10.0)
    
    # Parse pocket types from features (indices 6, 7, 8 are hydrophobic, donor, acceptor)
    pocket_types = []
    for f in pocket_features:
        if len(f) > 8:
            if f[6] > 0.5:
                pocket_types.append('C')
            elif f[7] > 0.5:
                pocket_types.append('N')
            elif f[8] > 0.5:
                pocket_types.append('O')
            else:
                pocket_types.append('C')
        else:
            pocket_types.append('C')
    n_rec_hydro = sum(1 for t in pocket_types if t == 'C')
    n_rec_hbond = sum(1 for t in pocket_types if t in ['N', 'O'])
    features[20] = n_rec_hydro / max(1, len(pocket_types))
    features[21] = n_rec_hbond / max(1, len(pocket_types))
    
    for i, sig in enumerate([2.0, 4.0, 8.0]):
        features[22+i] = np.exp(-all_dists.mean()**2 / sig**2)
    
    features[25] = np.percentile(all_dists, 25)
    features[26] = np.percentile(all_dists, 50)
    features[27] = np.percentile(all_dists, 75)
    
    for i, p in enumerate([10, 25, 50, 75, 90]):
        features[28+i] = np.percentile(all_dists, p) / 10.0
    
    return features


def parse_pocket_atoms(pocket_pdb, center, radius=10.0):
    coords, features = [], []
    ATOMIC_MAP = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'S': 16}
    
    with open(pocket_pdb) as f:
        for line in f:
            if line.startswith(('ATOM', 'HETATM')):
                try:
                    x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                    if np.sqrt((x-center[0])**2 + (y-center[1])**2 + (z-center[2])**2) > radius:
                        continue
                    
                    atom_type = line[76:78].strip()
                    if not atom_type:
                        atom_type = line[12:16].strip()[0]
                    
                    coords.append([x, y, z])
                    atomic_num = ATOMIC_MAP.get(atom_type.upper(), 6)
                    one_hot = np.zeros(100)
                    one_hot[atomic_num] = 1.0
                    
                    feat = np.concatenate([one_hot, [
                        1.0 if atom_type in ['C', 'S'] else 0.0,
                        1.0 if atom_type in ['N', 'O'] else 0.0,
                        1.0 if atom_type in ['N', 'O', 'S'] else 0.0,
                        0.0
                    ]])
                    features.append(feat)
                except:
                    continue
    
    coords = np.array(coords)
    features = np.array(features) if features else np.zeros((0, 105))
    return coords, features


def parse_ligand_sdf(sdf_path):
    coords, atom_types = [], []
    
    with open(sdf_path) as f:
        lines = f.readlines()
    
    if len(lines) < 4:
        return np.array(coords), atom_types
    
    n_atoms = int(lines[2].split()[0])
    
    for i in range(n_atoms):
        line = lines[3 + i]
        parts = line.split()
        if len(parts) >= 4:
            coords.append([float(parts[0]), float(parts[1]), float(parts[2])])
            atom_types.append(parts[3] if len(parts) > 3 else 'C')
    
    return np.array(coords), atom_types


def prepare_training_data(compounds, data_dir, max_pocket_atoms=100):
    samples = []
    
    for compound in compounds:
        pdb_id = compound['pdb_id']
        exp_affinity = compound['experimental_affinity']
        center = np.array([compound['center']['x'], compound['center']['y'], compound['center']['z']])
        
        pocket_path = os.path.join(data_dir, pdb_id, f"{pdb_id}_pocket.pdb")
        sdf_path = os.path.join(data_dir, pdb_id, f"{pdb_id}_ligand.sdf")
        
        if not os.path.exists(pocket_path) or not os.path.exists(sdf_path):
            continue
        
        try:
            pocket_coords, pocket_features = parse_pocket_atoms(pocket_path, center)
            ligand_coords, ligand_types = parse_ligand_sdf(sdf_path)
            
            if len(pocket_coords) < 5 or len(ligand_coords) < 3:
                continue
            
            if len(pocket_coords) > max_pocket_atoms:
                idx = np.random.choice(len(pocket_coords), max_pocket_atoms, replace=False)
                pocket_coords = pocket_coords[idx]
                pocket_features = pocket_features[idx]
            elif len(pocket_coords) < 30:
                # Skip compounds with too few pocket atoms
                print(f"Skipping {pdb_id}: only {len(pocket_coords)} pocket atoms")
                continue
            
            feat = compute_physics_features(ligand_coords, ligand_types, pocket_coords, pocket_features, center)
            
            samples.append({
                'pdb_id': pdb_id,
                'features': feat,
                'affinity': exp_affinity
            })
        except Exception as e:
            print(f"Error {pdb_id}: {e}")
            continue
    
    return samples


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--compounds', default='/mnt/c/Users/yakka/Downloads/geock_110_data/compounds.json')
    parser.add_argument('--data_dir', default='/mnt/c/Users/yakka/Downloads/geock_110_data')
    parser.add_argument('--output_dir', default='/mnt/c/Users/yakka/Downloads/GEOCK_Project')
    parser.add_argument('--n_compounds', type=int, default=30)
    parser.add_argument('--folds', type=int, default=5)
    args = parser.parse_args()
    
    print(f"Loading {args.n_compounds} compounds...")
    with open(args.compounds) as f:
        compounds = json.load(f)[:args.n_compounds]
    
    print("Preparing data...")
    samples = prepare_training_data(compounds, args.data_dir)
    print(f"Prepared {len(samples)} samples")
    
    X = np.array([s['features'] for s in samples])
    y = np.array([s['affinity'] for s in samples])
    
    print(f"X shape: {X.shape}, y range: {y.min():.2f} to {y.max():.2f}")
    
    # Test multiple models
    models = {
        'Ridge': Ridge(alpha=1.0),
        'Ridge(0.1)': Ridge(alpha=0.1),
        'Ridge(10)': Ridge(alpha=10.0),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
        'RF': RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
        'GB': GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42),
    }
    
    results = {}
    kf = KFold(n_splits=args.folds, shuffle=True, random_state=42)
    
    for name, model in models.items():
        preds = cross_val_predict(model, X, y, cv=kf)
        r, _ = pearsonr(preds, y)
        mae = mean_absolute_error(y, preds)
        results[name] = {'r': r, 'mae': mae}
        print(f"{name}: r={r:.3f}, MAE={mae:.3f}")
    
    # Best model
    best = max(results.items(), key=lambda x: x[1]['r'])
    print(f"\nBest: {best[0]} with r={best[1]['r']:.3f}")
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f"simple_results_{args.n_compounds}.json")
    with open(output_file, 'w') as f:
        json.dump({'n_samples': len(samples), 'results': results}, f, indent=2)
    print(f"Saved to {output_file}")


if __name__ == "__main__":
    np.random.seed(42)
    main()
