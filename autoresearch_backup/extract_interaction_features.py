#!/usr/bin/env python3
"""
Extract ligand-pocket interaction features from CASF-2016
- Hydrogen bond donors/acceptors
- Hydrophobic contacts
- Electrostatic interactions
- Ring stacking
- Aromatic interactions
"""

import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

CASF_DIR = Path("/mnt/c/Users/yakka/Downloads/CASF-2016/CASF-2016/coreset")

print("="*70)
print("EXTRACTING INTERACTION FEATURES")
print("="*70)

# Element categories
HBOND_DONORS = {'N', 'O', 'OH', 'NH'}  # Simplified
HBOND_ACCEPTORS = {'N', 'O'}
AROMATIC = {'PHE', 'TYR', 'TRP', 'HIS', 'HID', 'HIE'}
HYDROPHOBIC = {'ALA', 'VAL', 'LEU', 'ILE', 'MET', 'PHE', 'TRP'}
POSITIVE = {'ARG', 'LYS', 'HIS'}
NEGATIVE = {'ASP', 'GLU'}

def parse_pdb_atoms(pdb_path):
    """Parse PDB and extract atom info"""
    atoms = []
    try:
        with open(pdb_path) as f:
            for line in f:
                if line.startswith('ATOM') or line.startswith('HETATM'):
                    try:
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        resname = line[17:20].strip()
                        atom_name = line[12:16].strip()
                        element = (line[76:78].strip() or 
                                atom_name[0])
                        
                        atoms.append({
                            'x': x, 'y': y, 'z': z,
                            'resname': resname,
                            'atom': atom_name,
                            'element': element
                        })
                    except:
                        continue
    except:
        return []
    return atoms

def parse_mol2_atoms(mol2_path):
    """Parse mol2 file for ligand atoms"""
    atoms = []
    try:
        with open(mol2_path) as f:
            in_atom = False
            for line in f:
                if line.startswith('@<TRIPOS>ATOM'):
                    in_atom = True
                    continue
                if line.startswith('@<TRIPOS>BOND'):
                    in_atom = False
                if in_atom and line.strip():
                    parts = line.split()
                    if len(parts) >= 5:
                        try:
                            idx = int(parts[0])
                            atom_name = parts[1]
                            x = float(parts[2])
                            y = float(parts[3])
                            z = float(parts[4])
                            element = ''.join([c for c in atom_name if c.isalpha()])
                            
                            atoms.append({
                                'x': x, 'y': y, 'z': z,
                                'atom': atom_name,
                                'element': element
                            })
                        except:
                            continue
    except:
        return []
    return atoms

def compute_distance(a1, a2):
    """Euclidean distance between atoms"""
    return np.sqrt((a1['x']-a2['x'])**2 + 
                 (a1['y']-a2['y'])**2 + 
                 (a1['z']-a2['z'])**2)

def compute_interactions(protein_atoms, ligand_atoms):
    """Compute interaction features between protein and ligand"""
    features = np.zeros(30, dtype=np.float32)
    
    if not protein_atoms or not ligand_atoms:
        return features
    
    # Categorize protein residues
    res_atoms = defaultdict(list)
    for a in protein_atoms:
        res_atoms[a['resname']].append(a)
    
    # H-bond donors in protein (within 4A of ligand)
    hbond_donors = []
    hbond_acceptors = []
    for a in protein_atoms:
        if a['element'] in HBOND_DONORS:
            hbond_donors.append(a)
        if a['element'] in HBOND_ACCEPTORS:
            hbond_acceptors.append(a)
    
    # Check H-bonds (distance < 3.5A)
    hbond_count = 0
    salt_bridges = 0
    for la in ligand_atoms:
        for pa in hbond_donors + hbond_acceptors:
            d = compute_distance(la, pa)
            if d < 3.5:
                hbond_count += 1
            if d < 4.0:
                la_elem = la.get('element', '')
                pa_elem = pa.get('element', '')
                # Salt bridge: charged interactions
                if (la_elem in ['N', 'O'] and pa_elem in ['N', 'O']):
                    salt_bridges += 1
    
    # Hydrophobic contacts (< 4A to hydrophobic residues)
    hydrophobic_contacts = 0
    for resname, ratoms in res_atoms.items():
        if resname in HYDROPHOBIC:
            for la in ligand_atoms:
                for pa in ratoms:
                    if compute_distance(la, pa) < 4.0:
                        hydrophobic_contacts += 1
                        break
    
    # Aromatic interactions
    aromatic_contacts = 0
    pi_stacking = 0
    for resname, ratoms in res_atoms.items():
        if resname in AROMATIC:
            center = np.mean([[a['x'], a['y'], a['z']] for a in ratoms], axis=0)
            for la in ligand_atoms:
                d = compute_distance(la, {'x': center[0], 'y': center[1], 'z': center[2]})
                if d < 6.0:
                    aromatic_contacts += 1
                    if d < 5.0:
                        pi_stacking += 1
    
    # Electrostatics
    positive_charge = sum(1 for r in res_atoms if r in POSITIVE)
    negative_charge = sum(1 for r in res_atoms if r in NEGATIVE)
    
    # Counts
    n_protein = len(protein_atoms)
    n_ligand = len(ligand_atoms)
    
    # Populate features
    features[0] = hbond_count / max(1, n_ligand)
    features[1] = salt_bridges / max(1, n_ligand)
    features[2] = hydrophobic_contacts / max(1, n_protein)
    features[3] = aromatic_contacts
    features[4] = pi_stacking
    features[5] = positive_charge
    features[6] = negative_charge
    features[7] = positive_charge - negative_charge
    features[8] = n_protein / 100
    features[9] = n_ligand / 50
    features[10] = n_protein + n_ligand
    
    # Normalize by protein size
    features[11] = hbond_count / max(1, n_protein) * 100
    features[12] = hydrophobic_contacts / max(1, n_protein) * 100
    
    # Additional features (extended to 30 dims)
    # Contact statistics
    all_dists = []
    for la in ligand_atoms[:20]:  # Sample ligand atoms
        min_dists = []
        for pa in protein_atoms[:50]:  # Sample protein atoms
            d = compute_distance(la, pa)
            min_dists.append(d)
        if min_dists:
            all_dists.append(min(min_dists))
    
    if all_dists:
        features[13] = np.min(all_dists)
        features[14] = np.mean(all_dists)
        features[15] = np.max(all_dists)
        features[16] = np.std(all_dists)
    else:
        features[13:17] = 10.0
    
    # Binding site density
    close_contacts = sum(1 for d in all_dists if d < 4.0)
    features[17] = close_contacts / max(1, len(all_dists))
    
    # Charge complementarity
    ligand_charge = sum(1 for a in ligand_atoms if a.get('element') in ['N', 'O'])
    features[18] = positive_charge * ligand_charge / max(1, n_protein)
    features[19] = negative_charge * ligand_charge / max(1, n_protein)
    
    # Hydrophobic ratio
    features[20] = hydrophobic_contacts / max(1, hbond_count + 1)
    features[21] = aromatic_contacts / max(1, hbond_count + 1)
    
    # Spatial distribution
    if protein_atoms:
        coords = np.array([[a['x'], a['y'], a['z']] for a in protein_atoms])
        center = coords.mean(axis=0)
        spread = np.std(coords, axis=0)
        features[22] = spread[0]
        features[23] = spread[1]
        features[24] = spread[2]
        features[25] = np.sqrt(spread[0]**2 + spread[1]**2 + spread[2]**2)
    
    # Pocket flexibility indicator (residue types)
    unique_res = len(res_atoms)
    features[26] = unique_res
    
    # Aromatic proportion
    n_aromatic = sum(len(atoms) for r, atoms in res_atoms.items() if r in AROMATIC)
    features[27] = n_aromatic / max(1, n_protein)
    
    # Hydrophobic proportion  
    n_hydro = sum(len(atoms) for r, atoms in res_atoms.items() if r in HYDROPHOBIC)
    features[28] = n_hydro / max(1, n_protein)
    
    # Charged proportion
    features[29] = (positive_charge + negative_charge) / max(1, unique_res)
    
    return features

# Process all CASF complexes
interaction_features = {}
processed = 0
failed = 0

for pdb_dir in sorted(CASF_DIR.iterdir()):
    if not pdb_dir.is_dir():
        continue
    
    pdb_id = pdb_dir.name
    
    pocket_pdb = pdb_dir / f"{pdb_id}_pocket.pdb"
    protein_mol2 = pdb_dir / f"{pdb_id}_protein.mol2"
    ligand_mol2 = pdb_dir / f"{pdb_id}_ligand.mol2"
    
    if not pocket_pdb.exists():
        failed += 1
        continue
    
    # Parse atoms
    protein_atoms = parse_pdb_atoms(pocket_pdb)
    if not protein_atoms:
        failed += 1
        continue
    
    if ligand_mol2.exists():
        ligand_atoms = parse_mol2_atoms(ligand_mol2)
    else:
        failed += 1
        continue
    
    if not ligand_atoms:
        failed += 1
        continue
    
    # Compute interactions
    features = compute_interactions(protein_atoms, ligand_atoms)
    interaction_features[pdb_id] = features
    processed += 1
    
    if processed % 50 == 0:
        print(f"   Processed: {processed}")

print(f"\n   Successfully processed: {processed}")
print(f"   Failed: {failed}")

# Save interaction features
output = {
    'interactions': interaction_features,
    'n_features': 30,
    'processed': processed
}

with open('WORK_DIR / casf_interaction_features.pkl', 'wb') as f:
    pickle.dump(output, f)

print(f"\nSaved to casf_interaction_features.pkl")

# Statistics
print("\nInteraction feature statistics:")
for i, name in enumerate(['hbond_count', 'salt_bridges', 'hydrophobic', 'aromatic', 
                        'pi_stacking', 'positive', 'negative', 'charge_diff',
                        'n_protein', 'n_ligand', 'total_atoms']):
    vals = [f[i] for f in interaction_features.values()]
    print(f"  {name}: mean={np.mean(vals):.3f}, std={np.std(vals):.3f}")