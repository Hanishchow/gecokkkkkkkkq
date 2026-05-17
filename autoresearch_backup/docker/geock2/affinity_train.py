"""
affinity_train.py - GEOCK 2.0 Binding Affinity Training (Fixed)

Optimized for small dataset (n=100):
- PLSRegression (best for n<200, high-dim features)
- 533 features: Vina(6) + ECFP4(128) + PhysChem(9) + ProLIF(6)
- LeaveOneOut CV for robust evaluation

Fixed: Proper scaling, normalized features, ProLIF from JSON coords
"""

import json
import numpy as np
import os
import warnings
from typing import Optional

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.rdMolDescriptors import CalcTPSA

from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")

# Constants
ECFP_BITS = 128
ECFP_RADIUS = 2
N_COMPONENTS = 10
TARGET_R = 0.65
STRETCH_R = 0.75
MAX_POCKET_ATOMS = 2000  # Limit for computational efficiency

# VDW radii
_VDW = {"C": 1.9, "A": 1.9, "N": 1.8, "NA": 1.8, "O": 1.7, "OA": 1.7,
        "S": 2.0, "SA": 2.0, "P": 2.1, "F": 1.5, "Cl": 1.8, "Br": 2.0}
_DEFAULT_VDW = 1.8
_HYDROPHOBIC = frozenset({"C", "A", "S", "Cl", "Br", "F"})
_HBOND_ACC = frozenset({"NA", "OA", "SA", "N", "O"})
_HBOND_DON = frozenset({"NA", "OA", "N", "O"})

# Vinardo weights (normalized so sum ≈ 1)
_W = dict(gauss1=-0.045, repulsion=0.800, hydrophobic=-0.030, hbond=-0.600)


def _vdw(t): return _VDW.get(t, _DEFAULT_VDW)

def _surface_dist(r, t1, t2): return r - (_vdw(t1) + _vdw(t2))
def _gauss1(d): return np.exp(-((d / 0.5) ** 2))
def _repulsion(d): return d * d if d < 0 else 0.0
def _hydrophobic(d, good=0.0, bad=2.5):
    if d <= good: return 1.0
    if d >= bad: return 0.0
    return (bad - d) / (bad - good)
def _hbond(d, good=-0.7, bad=0.0):
    if d <= good: return 1.0
    if d >= bad: return 0.0
    return (bad - d) / (bad - good)


def compute_vina_features(rec_coords, rec_types, lig_coords, lig_types, n_pairs):
    """Layer 1: Vina/Vinardo physics (6 features) - NORMALIZED by n_pairs."""
    g1 = rep = hydro = hb = 0.0
    
    for ri, rt in zip(rec_coords, rec_types):
        for li, lt in zip(lig_coords, lig_types):
            r = float(np.linalg.norm(ri - li))
            if r >= 8.0: continue
            d = _surface_dist(r, rt, lt)
            g1 += _gauss1(d)
            rep += _repulsion(d)
            if rt in _HYDROPHOBIC and lt in _HYDROPHOBIC:
                hydro += _hydrophobic(d)
            if (rt in _HBOND_DON and lt in _HBOND_ACC) or (lt in _HBOND_DON and rt in _HBOND_ACC):
                hb += _hbond(d)
    
    # Normalize by number of atom pairs
    n_pairs = max(1, n_pairs)
    
    # Weighted terms
    w_gauss1 = _W["gauss1"] * g1 / n_pairs
    w_repulsion = _W["repulsion"] * rep / n_pairs
    w_hydrophobic = _W["hydrophobic"] * hydro / n_pairs
    w_hbond = _W["hbond"] * hb / n_pairs
    
    # Combined score
    score = w_gauss1 + w_repulsion + w_hydrophobic + w_hbond
    
    return np.array([w_gauss1, w_repulsion, w_hydrophobic, w_hbond, hydro/n_pairs, score], dtype=np.float32)


def compute_rdkit_features(smiles: str, n_atoms: int = 1) -> np.ndarray:
    """Layers 2+3: ECFP4 fingerprint (128D) + Physicochemical (9D)."""
    zeros = np.zeros(ECFP_BITS + 9, dtype=np.float32)
    if not smiles: return zeros
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return zeros
    
    # ECFP4 - normalize by atom count
    try:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=ECFP_RADIUS, nBits=ECFP_BITS)
        ecfp = np.array(fp, dtype=np.float32) / max(1, mol.GetNumHeavyAtoms())
    except:
        ecfp = np.zeros(ECFP_BITS, dtype=np.float32)
    
    # Physicochemical - normalized
    try:
        physchem = np.array([
            Descriptors.MolLogP(mol),
            CalcTPSA(mol) / 200.0,  # Normalize TPSA
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.NumRotatableBonds(mol) / 20.0,
            Descriptors.MolWt(mol) / 1000.0,  # Normalize MW
            Descriptors.NumAromaticRings(mol) / 5.0,
            Descriptors.FractionCSP3(mol),
            Descriptors.NumHeteroatoms(mol) / 10.0,
        ], dtype=np.float32)
    except:
        physchem = np.zeros(9, dtype=np.float32)
    
    return np.concatenate([ecfp, physchem])


def compute_prolif_features_json(lig_coords, lig_types, rec_coords, rec_types):
    """Layer 4: ProLIF-like interaction counts (6D) from JSON coords."""
    n_lig = max(1, len(lig_coords))
    
    counts = np.zeros(6, dtype=np.float32)
    
    # Count interactions
    for i, (lc, lt) in enumerate(zip(lig_coords, lig_types)):
        min_dist = float('inf')
        for j, (rc, rt) in enumerate(zip(rec_coords, rec_types)):
            d = np.linalg.norm(lc - rc)
            if d < min_dist:
                min_dist = d
                closest_rt = rt
        
        if min_dist < 4.5:
            # Hydrophobic (C with C)
            if lt in ['C'] and closest_rt in ['C']:
                counts[2] += 1
            # H-bond donor (N,O with H)
            if lt in ['N', 'O'] and min_dist < 3.5:
                if lt in ['N', 'O']:
                    counts[0] += 1  # HBDonor
                    counts[1] += 1  # HBAcceptor (simplified)
    
    # Normalize by ligand atom count
    counts = counts / n_lig
    
    return counts


def parse_pocket(pdb_path: str, max_atoms: int = MAX_POCKET_ATOMS):
    """Parse pocket PDB to get coords and types, with atom limit."""
    coords, types = [], []
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")): continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                elem = line[76:78].strip().upper() if len(line) > 76 else ""
                if elem in ("H", "D"): continue
                resname = line[17:20].strip()
                if resname in ("HOH", "WAT", "H2O"): continue
                coords.append([x, y, z])
                types.append(elem if elem in _VDW else "C")
            except: continue
    
    if not coords: return np.zeros((1, 3), dtype=np.float32), ["C"]
    
    # Limit to max_atoms (sample evenly if too many)
    if len(coords) > max_atoms:
        indices = np.linspace(0, len(coords)-1, max_atoms, dtype=int)
        coords = [coords[i] for i in indices]
        types = [types[i] for i in indices]
    
    return np.array(coords, dtype=np.float32), types


def count_rotatable_bonds(smiles: str) -> int:
    """Count rotatable bonds from SMILES."""
    if not smiles: return 0
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return 0
    try:
        return Descriptors.NumRotatableBonds(mol)
    except:
        return 0


def extract_features(pdb_id: str, data_dir: str, smiles_map: dict) -> Optional[np.ndarray]:
    """Extract all 149 features for a compound."""
    pocket_pdb = os.path.join(data_dir, pdb_id, f"{pdb_id}_pocket.pdb")
    if not os.path.exists(pocket_pdb): return None
    
    # Get ligand ID
    with open(os.path.join(data_dir, "compounds.json")) as f:
        compounds = json.load(f)
        ligand_id = None
        for c in compounds:
            if c["pdb_id"] == pdb_id:
                ligand_id = c["ligand_id"]
                break
    
    if not ligand_id: return None
    smiles = smiles_map.get(ligand_id)
    if not smiles: return None
    
    # Load ligand JSON
    json_path = os.path.join(data_dir, pdb_id, f"{pdb_id}_ligand.json")
    if not os.path.exists(json_path): return None
    with open(json_path) as f:
        lig_data = json.load(f)
    
    # Parse pocket
    rec_coords, rec_types = parse_pocket(pocket_pdb)
    
    # Ligand coords and types
    lig_coords = np.array([[c['x'], c['y'], c['z']] for c in lig_data['coords']], dtype=np.float32)
    lig_types = [c['elem'] for c in lig_data['coords']]
    n_atoms = len(lig_coords)
    
    # Layer 1: Vina (normalized)
    n_pairs = n_atoms * len(rec_coords)
    vina = compute_vina_features(rec_coords, rec_types, lig_coords, lig_types, n_pairs)
    
    # Layer 2+3: RDKit
    rdkit = compute_rdkit_features(smiles, n_atoms)
    
    # Layer 4: Interaction counts (simplified ProLIF)
    prolif = compute_prolif_features_json(lig_coords, lig_types, rec_coords, rec_types)
    
    return np.concatenate([vina, rdkit, prolif]).astype(np.float32)


def train_and_evaluate(data_dir: str, smiles_cache: str, n_samples: int = None):
    """Train model with LeaveOneOut CV."""
    
    # Load data
    with open(os.path.join(data_dir, "compounds.json")) as f:
        compounds = json.load(f)
    with open(smiles_cache) as f:
        smiles_map = json.load(f)
    
    if n_samples:
        compounds = compounds[:n_samples]
    
    print(f"Loading {len(compounds)} compounds...")
    
    # Extract features
    X_list, y_list, valid_ids = [], [], []
    for i, comp in enumerate(compounds):
        feats = extract_features(comp["pdb_id"], data_dir, smiles_map)
        if feats is not None:
            X_list.append(feats)
            y_list.append(comp["experimental_affinity"])
            valid_ids.append(comp["pdb_id"])
        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{len(compounds)}, {len(X_list)} valid")
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    # Handle NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"\nFeature matrix: {X.shape}")
    print(f"  Vina: features 0-5")
    print(f"  RDKit/ECFP4: features 6-133 (normalized)")
    print(f"  PhysChem: features 134-142")
    print(f"  ProLIF-like: features 143-148")
    print(f"Affinity range: {y.min():.2f} to {y.max():.2f}")
    
    # LeaveOneOut CV
    print(f"\nRunning LeaveOneOut CV with PLSRegression (n_components={N_COMPONENTS})...")
    loo = LeaveOneOut()
    
    preds_list = []
    for train_idx, val_idx in loo.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Scale ALL features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Train PLS
        model = PLSRegression(n_components=min(N_COMPONENTS, len(y_train)-1))
        model.fit(X_train_scaled, y_train)
        pred = float(model.predict(X_val_scaled).ravel()[0])
        preds_list.append(pred)
    
    preds = np.array(preds_list)
    
    # Calculate metrics
    mean_r = pearsonr(preds, y)[0]
    mean_mae = mean_absolute_error(y, preds)
    mean_rmse = np.sqrt(mean_squared_error(y, preds))
    
    print(f"\n{'='*50}")
    print(f"RESULTS (LeaveOneOut CV)")
    print(f"{'='*50}")
    print(f"Compounds: {len(X)}")
    print(f"Pearson R: {mean_r:.4f}")
    print(f"MAE: {mean_mae:.4f} kcal/mol")
    print(f"RMSE: {mean_rmse:.4f} kcal/mol")
    print(f"Target R: {TARGET_R} | Stretch: {STRETCH_R}")
    status = "PASS" if mean_r >= TARGET_R else ("STRETCH!" if mean_r >= STRETCH_R else "FAIL")
    print(f"Status: {status}")
    print(f"{'='*50}")
    
    # Feature importance
    print(f"\nTop 10 most important features (by coefficient magnitude):")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = PLSRegression(n_components=min(N_COMPONENTS, len(y)-1))
    model.fit(X_scaled, y)
    
    coefs = np.abs(model.coef_).mean(axis=0)
    feat_names = ["vina_gauss1", "vina_repulsion", "vina_hydro", "vina_hbond", "vina_contact", "vina_score"] + \
                 [f"ecfp4_{i}" for i in range(ECFP_BITS)] + \
                 ["logP", "TPSA_n", "HBD", "HBA", "RotBonds_n", "MW_n", "AromRings_n", "FracCSP3", "HeteroAtoms_n"] + \
                 ["HBDonor_n", "HBAcceptor_n", "Hydrophobic_n", "PiStack", "Cationic", "Anionic"]
    
    top_idx = np.argsort(coefs)[::-1][:10]
    for idx in top_idx:
        name = feat_names[idx] if idx < len(feat_names) else f"feat_{idx}"
        print(f"  {name}: {coefs[idx]:.4f}")
    
    return {
        "mean_r": float(mean_r),
        "mae": float(mean_mae),
        "rmse": float(mean_rmse),
        "n_samples": len(X),
        "n_features": X.shape[1],
        "status": status
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/mnt/c/Users/yakka/Downloads/geock_110_data")
    parser.add_argument("--smiles", default="/mnt/c/Users/yakka/Downloads/geock_110_data/smiles_cache.json")
    parser.add_argument("--n", type=int, default=None, help="Number of samples (default: all)")
    parser.add_argument("--output", default="cv_results_pls.json")
    args = parser.parse_args()
    
    results = train_and_evaluate(args.data, args.smiles, args.n)
    
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")
