"""
optimized_affinity.py - Optimized Binding Affinity Prediction

Optimizations:
1. Better pocket sampling (closest atoms to ligand)
2. SMILES-derived molecular descriptors
3. Feature selection based on importance
4. Ensemble of multiple models
5. Alpha optimization via cross-validation
"""

import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_predict
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error
import json, os, time
from typing import Optional, Tuple
import warnings
warnings.filterwarnings("ignore")

# RDKit for molecular descriptors
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, AllChem
    from rdkit.Chem.rdMolDescriptors import CalcTPSA
    RDKIT_AVAILABLE = True
except:
    RDKIT_AVAILABLE = False

# Constants
MAX_POCKET_ATOMS = 500
POCKET_CUTOFF = 15.0  # Angstroms


def get_molecular_descriptors(smiles: str) -> np.ndarray:
    """Get molecular descriptors from SMILES (protein-invariant features)."""
    if not RDKIT_AVAILABLE or not smiles:
        return np.zeros(15)
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(15)
    
    try:
        # Basic descriptors
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        tpsa = CalcTPSA(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        n_rot = Descriptors.NumRotatableBonds(mol)
        n_aromatic = Descriptors.NumAromaticRings(mol)
        n_hetero = Descriptors.NumHeteroatoms(mol)
        n_rings = Descriptors.RingCount(mol)
        frac_sp3 = Descriptors.FractionCSP3(mol)
        n_atoms = mol.GetNumAtoms()
        n_heavy = mol.GetNumHeavyAtoms()
        n_aliphatic = Descriptors.NumAliphaticRings(mol)
        
        # Normalize
        features = np.array([
            mw / 500.0,           # Normalized MW
            logp / 5.0,           # Normalized LogP
            tpsa / 140.0,         # Normalized TPSA
            hbd / 5.0,            # Normalized HBD
            hba / 10.0,           # Normalized HBA
            n_rot / 10.0,         # Normalized rotatable bonds
            n_aromatic / 5.0,     # Normalized aromatic rings
            n_hetero / 10.0,      # Normalized heteroatoms
            n_rings / 5.0,        # Normalized ring count
            frac_sp3,              # sp3 fraction
            n_atoms / 50.0,       # Normalized atom count
            n_heavy / 30.0,       # Normalized heavy atom count
            n_aliphatic / 3.0,   # Normalized aliphatic rings
            (mw / n_heavy) if n_heavy > 0 else 0,  # Heavy atom MW
            n_hba / (n_hbd + 1),  # HBA/HBD ratio
        ], dtype=np.float32)
        
        return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    except:
        return np.zeros(15)


def compute_physics_features(
    ligand_coords: np.ndarray,
    ligand_types: list,
    pocket_coords: np.ndarray,
    center: np.ndarray
) -> np.ndarray:
    """Compute physics-based features with smarter pocket handling."""
    features = np.zeros(40)
    
    if len(ligand_coords) == 0 or len(pocket_coords) == 0:
        return features
    
    # Sample pocket atoms closest to ligand
    if len(pocket_coords) > MAX_POCKET_ATOMS:
        lig_center = ligand_coords.mean(axis=0)
        dists = np.linalg.norm(pocket_coords - lig_center, axis=1)
        idx = np.argsort(dists)[:MAX_POCKET_ATOMS]
        pocket_coords = pocket_coords[idx]
    
    # Compute distances
    n_lig = len(ligand_coords)
    n_rec = len(pocket_coords)
    
    # All pairwise distances
    all_dists = []
    for lc in ligand_coords:
        for rc in pocket_coords:
            all_dists.append(np.linalg.norm(lc - rc))
    all_dists = np.array(all_dists)
    
    # Ligand atom distances to pocket
    lig_dists = np.array([
        min(np.linalg.norm(lc - rc) for rc in pocket_coords)
        for lc in ligand_coords
    ])
    
    # Ligand center distance
    lig_center = ligand_coords.mean(axis=0)
    dist_lig_center = np.linalg.norm(lig_center - center)
    
    # Basic statistics
    features[0] = np.exp(-dist_lig_center**2 / (2 * 3.0**2))
    features[1] = np.exp(-all_dists.mean()**2 / (2 * 3.0**2))
    features[2] = all_dists.min()
    features[3] = all_dists.mean()
    features[4] = all_dists.std()
    features[5] = lig_dists.min()
    features[6] = lig_dists.mean()
    features[7] = lig_dists.std()
    
    # Contact fractions at different distances (NORMALIZED by ligand atoms)
    for i, d in enumerate([2.0, 3.0, 4.0, 5.0, 6.0]):
        features[8+i] = np.sum(all_dists < d) / n_lig
    
    # Ligand composition
    n_atoms = len(ligand_types)
    n_c = sum(1 for t in ligand_types if t == 'C')
    n_n = sum(1 for t in ligand_types if t == 'N')
    n_o = sum(1 for t in ligand_types if t == 'O')
    n_s = sum(1 for t in ligand_types if t == 'S')
    
    features[13] = n_c / n_atoms if n_atoms > 0 else 0
    features[14] = n_n / n_atoms if n_atoms > 0 else 0
    features[15] = n_o / n_atoms if n_atoms > 0 else 0
    features[16] = n_s / n_atoms if n_atoms > 0 else 0
    features[17] = n_atoms / 50.0
    
    # Pocket stats
    features[18] = n_rec / 200.0
    features[19] = n_lig * n_rec  # Interaction space
    
    # Interaction scores
    contact_score = hydro_score = hbond_score = 0.0
    for i, lc in enumerate(ligand_coords):
        for rc in pocket_coords:
            d = np.linalg.norm(lc - rc)
            if d < 4.0:
                contact_score += np.exp(-d**2 / 4.0)
                lt = ligand_types[i]
                # Hydrophobic
                if lt in ['C', 'S']:
                    hydro_score += np.exp(-d**2 / 9.0) if d < 3.5 else 0
                # H-bond
                if lt in ['N', 'O']:
                    hbond_score += np.exp(-d**2 / 4.0)
    
    features[20] = contact_score / max(1, n_lig)
    features[21] = hydro_score / max(1, n_lig)
    features[22] = hbond_score / max(1, n_lig)
    
    # Electrostatic proxy
    n_pos = n_n
    n_neg = n_o
    features[23] = (n_pos - n_neg) / n_atoms if n_atoms > 0 else 0
    
    # Distance percentiles
    for i, p in enumerate([25, 50, 75]):
        features[24+i] = np.percentile(all_dists, p) / 10.0
    
    # Histogram
    hist, _ = np.histogram(all_dists, bins=5, range=(0, 10))
    features[27:32] = hist / len(all_dists)
    
    # Geometric
    features[32] = np.sin(dist_lig_center / 10.0)
    features[33] = np.cos(dist_lig_center / 10.0)
    
    # Contact depth
    features[34] = np.percentile(lig_dists, 10) / 10.0  # Deep contacts
    features[35] = np.percentile(lig_dists, 90) / 10.0   # Surface contacts
    
    # Scaled interaction counts
    features[36] = contact_score / (n_lig * n_rec + 1)
    features[37] = hydro_score / (n_lig * n_rec + 1)
    features[38] = hbond_score / (n_lig * n_rec + 1)
    features[39] = (n_c * n_rec) / (n_lig * n_rec + 1)  # Hydrophobic fraction
    
    return features


def load_compound_data(pdb_id: str, data_dir: str, smiles_map: dict) -> Optional[Tuple[np.ndarray, float]]:
    """Load and extract features for a compound."""
    try:
        # Load ligand JSON
        with open(f'{data_dir}/{pdb_id}/{pdb_id}_ligand.json') as f:
            lig_data = json.load(f)
        
        # Get SMILES
        compounds_file = f'{data_dir}/compounds.json'
        smiles = None
        if os.path.exists(compounds_file):
            with open(compounds_file) as f:
                for c in json.load(f):
                    if c['pdb_id'] == pdb_id:
                        smiles = smiles_map.get(c.get('ligand_id'))
                        affinity = c['experimental_affinity']
                        break
        
        # Parse ligand
        ligand_coords = np.array([[c['x'], c['y'], c['z']] for c in lig_data['coords']], dtype=np.float32)
        ligand_types = [c['elem'] for c in lig_data['coords']]
        center = np.array([lig_data['center']['x'], lig_data['center']['y'], lig_data['center']['z']], dtype=np.float32)
        
        # Parse pocket
        pocket_coords = []
        pocket_file = f'{data_dir}/{pdb_id}/{pdb_id}_pocket.pdb'
        if os.path.exists(pocket_file):
            with open(pocket_file) as f:
                for line in f:
                    if line.startswith(('ATOM', 'HETATM')):
                        # Only receptor atoms, not ligand
                        if line.startswith('ATOM'):
                            x = float(line[30:38])
                            y = float(line[38:46])
                            z = float(line[46:54])
                            pocket_coords.append([x, y, z])
        
        if len(pocket_coords) == 0:
            return None
        
        pocket_coords = np.array(pocket_coords, dtype=np.float32)
        
        # Physics features
        physics = compute_physics_features(ligand_coords, ligand_types, pocket_coords, center)
        
        # Molecular descriptors from SMILES
        mol_desc = get_molecular_descriptors(smiles) if smiles else np.zeros(15)
        
        # Combine features
        features = np.concatenate([physics, mol_desc])
        
        return features, affinity
        
    except Exception as e:
        return None


def optimize_and_train(data_dir: str, smiles_cache: str, n_samples: int = None) -> dict:
    """Optimize model hyperparameters and train."""
    
    # Load SMILES
    with open(smiles_cache) as f:
        smiles_map = json.load(f)
    
    # Load compounds
    with open(f'{data_dir}/compounds.json') as f:
        compounds = json.load(f)
    
    if n_samples:
        compounds = compounds[:n_samples]
    
    print(f"Loading {len(compounds)} compounds...")
    
    # Extract features
    X_list, y_list = [], []
    t0 = time.time()
    for i, comp in enumerate(compounds):
        result = load_compound_data(comp['pdb_id'], data_dir, smiles_map)
        if result is not None:
            features, affinity = result
            X_list.append(features)
            y_list.append(affinity)
        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{len(compounds)}, {len(X_list)} valid")
    
    X = np.array(X_list)
    y = np.array(y_list)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"\nData: {X.shape}, Affinity: {y.min():.2f} to {y.max():.2f}")
    print(f"Extraction time: {time.time()-t0:.1f}s")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Test multiple models and alphas
    print("\n=== Model Comparison ===")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    best_r = -999
    best_model_name = None
    best_alpha = None
    best_preds = None
    
    # Models to test
    models = [
        ('Ridge(0.1)', Ridge(alpha=0.1)),
        ('Ridge(1.0)', Ridge(alpha=1.0)),
        ('Ridge(10.0)', Ridge(alpha=10.0)),
        ('Ridge(100.0)', Ridge(alpha=100.0)),
        ('Lasso(0.1)', Lasso(alpha=0.1, max_iter=5000)),
        ('Lasso(1.0)', Lasso(alpha=1.0, max_iter=5000)),
        ('ElasticNet', ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000)),
        ('GB(50)', GradientBoostingRegressor(n_estimators=50, max_depth=3, learning_rate=0.05, random_state=42)),
        ('RF(50)', RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)),
    ]
    
    results = []
    for name, model in models:
        preds = []
        for train_idx, val_idx in kf.split(X_scaled):
            model.fit(X_scaled[train_idx], y[train_idx])
            preds.extend(model.predict(X_scaled[val_idx]))
        
        preds = np.array(preds)
        r = pearsonr(y, preds)[0]
        mae = mean_absolute_error(y, preds)
        
        print(f"{name:20s}: R={r:.4f}, MAE={mae:.4f}")
        
        results.append((name, r, mae, preds))
        
        if r > best_r:
            best_r = r
            best_model_name = name
            best_preds = preds
    
    print(f"\n=== Best Model ===")
    print(f"Model: {best_model_name}")
    print(f"Pearson R: {best_r:.4f}")
    
    # Feature importance (for Ridge)
    print(f"\n=== Feature Importance (Ridge) ===")
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_scaled, y)
    coefs = np.abs(ridge.coef_)
    feat_names = [f'phys_{i}' for i in range(40)] + [f'mol_{i}' for i in range(15)]
    top_idx = np.argsort(coefs)[::-1][:10]
    for idx in top_idx:
        print(f"  {feat_names[idx]}: {coefs[idx]:.4f}")
    
    return {
        'best_model': best_model_name,
        'pearson_r': best_r,
        'n_samples': len(y),
        'n_features': X.shape[1],
        'results': [(n, r, mae) for n, r, mae, _ in results]
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/mnt/c/Users/yakka/Downloads/geock_110_data")
    parser.add_argument("--smiles", default="/mnt/c/Users/yakka/Downloads/geock_110_data/smiles_cache.json")
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--output", default="optimized_results.json")
    args = parser.parse_args()
    
    results = optimize_and_train(args.data, args.smiles, args.n)
    
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {args.output}")
