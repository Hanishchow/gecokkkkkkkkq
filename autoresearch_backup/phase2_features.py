#!/usr/bin/env python3
"""
Phase 2: Enhanced Feature Engineering for CASF-2016
- Extract high-quality pocket features from CASF PDBs
- Compute ligand-pocket interaction features
- Build comprehensive feature matrix
"""

import pickle
import numpy as np
from pathlib import Path
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

CASF_DIR = Path("/mnt/c/Users/yakka/Downloads/CASF-2016/CASF-2016")
CORESET_DIR = CASF_DIR / "coreset"

print("="*70)
print("PHASE 2: ENHANCED FEATURE ENGINEERING")
print("="*70)

def parse_pocket_pdb(pdb_path):
    """Parse pocket PDB and extract features"""
    if not pdb_path.exists():
        return None
    
    try:
        lines = open(pdb_path).readlines()
        
        # Extract atoms
        atoms = []
        for line in lines:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    resname = line[17:20].strip()
                    resid = int(line[22:26].strip())
                    atom_name = line[12:16].strip()
                    element = line[76:78].strip() if len(line) > 76 else ''
                    
                    atoms.append({
                        'x': x, 'y': y, 'z': z,
                        'resname': resname,
                        'resid': resid,
                        'atom': atom_name,
                        'element': element
                    })
                except:
                    continue
        
        if not atoms:
            return None
        
        return atoms
    except:
        return None

def compute_pocket_features(atoms):
    """Compute 50-dim pocket features"""
    if atoms is None:
        return np.zeros(50, dtype=np.float32)
    
    # Amino acid composition (20)
    common_aa = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY',
                 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER',
                 'THR', 'TRP', 'TYR', 'VAL']
    
    aa_counts = Counter(a['resname'] for a in atoms)
    total_atoms = len(atoms)
    
    aa_features = [aa_counts.get(aa, 0) / total_atoms for aa in common_aa]
    
    # Charge categories (5)
    positive = aa_counts.get('ARG', 0) + aa_counts.get('LYS', 0) + aa_counts.get('HIS', 0)
    negative = aa_counts.get('ASP', 0) + aa_counts.get('GLU', 0)
    polar = aa_counts.get('SER', 0) + aa_counts.get('THR', 0) + aa_counts.get('ASN', 0) + aa_counts.get('GLN', 0)
    hydrophobic = aa_counts.get('ALA', 0) + aa_counts.get('VAL', 0) + aa_counts.get('LEU', 0) + aa_counts.get('ILE', 0) + aa_counts.get('MET', 0) + aa_counts.get('PHE', 0) + aa_counts.get('TRP', 0)
    aromatic = aa_counts.get('PHE', 0) + aa_counts.get('TYR', 0) + aa_counts.get('TRP', 0)
    
    charge_features = [
        positive / total_atoms,
        negative / total_atoms,
        polar / total_atoms,
        hydrophobic / total_atoms,
        aromatic / total_atoms
    ]
    
    # Structural features (10)
    coords = np.array([[a['x'], a['y'], a['z']] for a in atoms])
    
    center = coords.mean(axis=0)
    distances_from_center = np.sqrt(((coords - center) ** 2).sum(axis=1))
    
    pocket_radius = distances_from_center.max()
    pocket_volume_estimate = (4/3) * np.pi * (pocket_radius ** 3) / 1000
    compactness = len(atoms) / (pocket_volume_estimate + 0.01)
    
    # Atom type counts
    carbon_count = sum(1 for a in atoms if a['element'] == 'C' or a['atom'][0] == 'C')
    nitrogen_count = sum(1 for a in atoms if a['element'] == 'N' or a['atom'][0] == 'N')
    oxygen_count = sum(1 for a in atoms if a['element'] == 'O' or a['atom'][0] == 'O')
    sulfur_count = sum(1 for a in atoms if a['element'] == 'S' or a['atom'][0] == 'S')
    
    struct_features = [
        pocket_radius / 20,  # Normalized
        pocket_volume_estimate,
        compactness / 10,
        len(atoms) / 100,
        carbon_count / total_atoms,
        nitrogen_count / total_atoms,
        oxygen_count / total_atoms,
        sulfur_count / max(total_atoms, 1),
        (positive - negative) / total_atoms,  # Net charge
        aa_counts.get('GLY', 0) / total_atoms  # Flexibility proxy
    ]
    
    # Interaction potential (10)
    # H-bond donors/acceptors in pocket
    hbond_donors = aa_counts.get('SER', 0) + aa_counts.get('THR', 0) + aa_counts.get('TYR', 0)
    hbond_acceptors = aa_counts.get('ASP', 0) + aa_counts.get('GLU', 0) + aa_counts.get('ASN', 0) + aa_counts.get('GLN', 0)
    
    # Aromatic for pi-stacking
    aromatic_residues = aa_counts.get('PHE', 0) + aa_counts.get('TYR', 0) + aa_counts.get('TRP', 0) + aa_counts.get('HIS', 0)
    
    # Charged for ionic interactions
    charged = positive + negative
    
    interaction_features = [
        hbond_donors / total_atoms,
        hbond_acceptors / total_atoms,
        aromatic_residues / total_atoms,
        charged / total_atoms,
        hydrophobic / total_atoms,  # Hydrophobic contact potential
        polar / total_atoms,  # Polar contact potential
        aa_counts.get('CYS', 0) / total_atoms,  # Disulfide potential
        aa_counts.get('PRO', 0) / total_atoms,  # Rigidity indicator
        compactness / 5,  # Shape complementarity proxy
        pocket_radius / 25  # Size match potential
    ]
    
    # Flexibility index (5)
    flexible = aa_counts.get('GLY', 0) + aa_counts.get('ALA', 0)
    rigid = aa_counts.get('PRO', 0)
    
    flex_features = [
        flexible / total_atoms,
        rigid / total_atoms,
        aa_counts.get('GLY', 0) / max(flexible, 1),
        (flexible - rigid) / total_atoms,
        pocket_radius / 15  # Pocket enclosure
    ]
    
    features = aa_features + charge_features + struct_features + interaction_features + flex_features
    
    return np.array(features[:50], dtype=np.float32)

# Load PDB IDs from CASF-2016
print("\n1. Loading CASF-2016 complex list...")
with open('casf2016_enhanced_features.pkl', 'rb') as f:
    test_data = pickle.load(f)

test_pdb_ids = [cx['pdb_id'] for cx in test_data['complexes']]
print(f"   Found {len(test_pdb_ids)} CASF-2016 complexes")

# Extract pocket features
print("\n2. Extracting high-quality pocket features from CASF PDBs...")
pocket_features = {}
failed = []

for pdb_id in test_pdb_ids:
    pocket_path = CORESET_DIR / pdb_id / f"{pdb_id}_pocket.pdb"
    
    atoms = parse_pocket_pdb(pocket_path)
    if atoms:
        feats = compute_pocket_features(atoms)
        pocket_features[pdb_id] = feats
    else:
        failed.append(pdb_id)

print(f"   Extracted: {len(pocket_features)} / {len(test_pdb_ids)}")
print(f"   Failed: {len(failed)}")

if failed:
    print(f"   First 5 failed: {failed[:5]}")

# Build enhanced feature matrix
print("\n3. Building enhanced feature matrix...")

# Load existing features
with open('casf2016_enhanced_features.pkl', 'rb') as f:
    existing = pickle.load(f)

X_enhanced = []
for cx in test_data['complexes']:
    pdb_id = cx['pdb_id']
    
    # Get existing features
    orig = cx['features'] if 'features' in cx else existing['X'][test_pdb_ids.index(pdb_id)]
    
    # Get pocket features
    if pdb_id in pocket_features:
        pocket = pocket_features[pdb_id]
    else:
        pocket = np.zeros(50, dtype=np.float32)
    
    # Combine: ligand (512) + physics (24) + pocket (50) = 586
    combined = np.concatenate([orig[:512], orig[512:536], pocket])
    X_enhanced.append(combined)

X_enhanced = np.array(X_enhanced)
print(f"   Enhanced feature matrix: {X_enhanced.shape}")

# Statistics
pocket_usage = (X_enhanced[:, 536:] > 0).any(axis=1).sum()
print(f"   Samples with pocket features: {pocket_usage} / {len(X_enhanced)}")

# Save enhanced features
enhanced_data = {
    'X': X_enhanced,
    'y': test_data['y'],
    'complexes': test_data['complexes'],
    'pocket_features': pocket_features,
    'feature_dim': 586,
    'features': {
        'ligand': (0, 512),
        'physics': (512, 536),
        'pocket': (536, 586)
    }
}

with open('casf2016_enhanced_v2.pkl', 'wb') as f:
    pickle.dump(enhanced_data, f)

print("\n4. Saved to casf2016_enhanced_v2.pkl")

# Feature statistics
print("\n" + "="*70)
print("FEATURE STATISTICS")
print("="*70)
print(f"Total features: {X_enhanced.shape[1]}")
print(f"  Ligand (ECFP4): 512")
print(f"  Physics: 24")  
print(f"  Pocket: 50")
print(f"\nPocket feature statistics:")
pocket_feats = X_enhanced[:, 536:]
print(f"  Non-zero per sample: {(pocket_feats > 0).any(axis=1).sum()} / {len(pocket_feats)}")
print(f"  Mean pocket sum: {pocket_feats.sum(axis=1).mean():.3f}")
print(f"  Std: {pocket_feats.sum(axis=1).std():.3f}")

print("\nPhase 2 complete. Ready for Phase 3.")