"""
GEOCK: Run full pipeline on a PDB file
- Inspect structure
- Extract pocket features (if any)
- Predict binding affinity (if ligand available)
"""
import sys, os, pickle, numpy as np
from pathlib import Path
from collections import Counter

BACKUP = Path(r'C:\Users\yakka\Downloads\Hanish_Brain_Vault\autoresearch_backup')
sys.path.insert(0, str(BACKUP))
from simple_pocket import extract_simple_pocket

pdb_path = Path(r'C:\Users\yakka\Downloads\AF-0000000078834380-model_v1.pdb')
print("=" * 70)
print("GEOCK - PDB Analysis")
print("=" * 70)

# 1. Parse PDB
lines = open(pdb_path).readlines()

# Extract header info
title = ''
for l in lines:
    if l.startswith('TITLE '):
        title += l[10:].strip() + ' '
    elif l.startswith('COMPND') and 'MOLECULE' in l:
        title += l[30:].strip() + ' '
title = title.strip()
print(f"Title: {title}")

# Count atoms
atoms = []
chains = set()
residues = {}
het_atoms = []
for l in lines:
    if l.startswith('ATOM') or l.startswith('HETATM'):
        try:
            x = float(l[30:38]); y = float(l[38:46]); z = float(l[46:54])
            resname = l[17:20].strip()
            resid = int(l[22:26].strip())
            chain = l[21].strip() if l[21].strip() else 'A'
            chains.add(chain)
            key = f"{chain}:{resname}:{resid}"
            atoms.append({'x': x, 'y': y, 'z': z, 'resname': resname, 'resid': resid, 'chain': chain})
            residues[key] = residues.get(key, 0) + 1
            if l.startswith('HETATM'):
                het_atoms.append({'resname': resname, 'resid': resid})
        except:
            pass
    elif l.startswith('TER'):
        pass

# Check for model info
model_type = 'AlphaFold' if 'ALPHAFOLD' in title.upper() else 'Unknown'
if 'AF-' in pdb_path.name:
    model_type = 'AlphaFold'
print(f"Model type: {model_type}")
print(f"Total atoms: {len(atoms)}")
print(f"Residues: {len(residues)}")
print(f"Chains: {', '.join(sorted(chains))}")

# Residue composition
res_names = Counter(k.split(':')[1] for k in residues)
print(f"\nResidue composition (top 10):")
for r, c in res_names.most_common(10):
    print(f"  {r}: {c}")

# Check for hetatms / ligands
het_unique = set((a['resname'], a['resid']) for a in het_atoms)
water_count = sum(1 for r, i in het_unique if r == 'HOH')
non_water = [(r, i) for r, i in het_unique if r != 'HOH']
print(f"\nHETATM residues: {len(het_unique)} total")
print(f"  Water molecules: {water_count}")
if non_water:
    print(f"  Non-water HETATMs: {non_water}")
    lig_present = True
else:
    print(f"  No non-water HETATMs found - protein-only structure")
    lig_present = False

# 2. Extract pocket features
print(f"\n--- Pocket Feature Extraction ---")
pocket_feat = extract_simple_pocket(str(pdb_path), cutoff=10.0)
nonzero = np.count_nonzero(pocket_feat)
print(f"Pocket features dim: {len(pocket_feat)}")
print(f"Non-zero features: {nonzero}")
print(f"Feature vector: {pocket_feat[:20]}...")

# 3. If no ligand, try to identify potential binding pockets by clustering
if not lig_present:
    print(f"\n--- Potential Binding Sites (dummy ligand not available) ---")
    # Find center of mass
    coords = np.array([[a['x'], a['y'], a['z']] for a in atoms])
    centroid = coords.mean(axis=0)
    size = coords.max(axis=0) - coords.min(axis=0)
    print(f"Protein size (XYZ): {size[0]:.1f} x {size[1]:.1f} x {size[2]:.1f} Angstrom")
    print(f"Centroid: ({centroid[0]:.1f}, {centroid[1]:.1f}, {centroid[2]:.1f})")
    print(f"Total mass: ~{len(atoms)} atoms")

# 4. Load model (if available) and explain what's needed
model_path = BACKUP / 'geock_final_best.pkl'
if model_path.exists():
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    print(f"\n--- GEOCK Model Status ---")
    print(f"Model: XGBoost ({model_data['config']})")
    print(f"Trained on: {model_data['n_samples']} complexes")
    print(f"Input features: {model_data['scaler'].n_features_in_}-dim")
    print(f"Selected features: {model_data['selector'].k}")
    
    if lig_present:
        print("\nLigand found! Can predict binding affinity.")
    else:
        print("\nNo ligand found in structure.")
        print("To predict binding affinity, provide a ligand SMILES string.")
        print("Usage: geock_predict.py <PDB> <LIGAND_SMILES>")

print("\n" + "=" * 70)
print("Analysis complete")
print("=" * 70)
