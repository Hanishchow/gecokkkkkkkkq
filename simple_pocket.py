#!/usr/bin/env python3
"""
Simple pocket feature extractor - extracts binding site from PDB without RDKit parsing.
Just counts residues near ligand binding region.
"""

import numpy as np
from pathlib import Path
from collections import Counter

def extract_simple_pocket(pdb_path, cutoff=10.0):
    """
    Extract simple pocket features from PDB file.
    - Parse ligand position from CONECT records or center of PDB
    - Count residues within cutoff distance
    """
    if not Path(pdb_path).exists():
        return np.zeros(50, dtype=np.float32)
    
    try:
        # Read PDB
        lines = open(pdb_path).readlines()
        
        # Get atoms with coordinates
        atoms = []
        for line in lines:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    resname = line[17:20].strip()
                    resid = int(line[22:26].strip())
                    atoms.append({
                        'x': x, 'y': y, 'z': z,
                        'resname': resname,
                        'resid': resid
                    })
                except:
                    continue
        
        if not atoms:
            return np.zeros(50, dtype=np.float32)
        
        # Estimate binding site center (use centroid of all atoms as fallback)
        center_x = np.mean([a['x'] for a in atoms])
        center_y = np.mean([a['y'] for a in atoms])
        center_z = np.mean([a['z'] for a in atoms])
        
        # Try to find ligand (non-standard residues)
        het_residues = set()
        for a in atoms:
            if a['resname'] not in ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY',
                                   'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER',
                                   'THR', 'TRP', 'TYR', 'VAL', 'HOH']:
                het_residues.add((a['resname'], a['resid']))
        
        if het_residues:
            # Use ligand center
            het_atoms = [a for a in atoms if (a['resname'], a['resid']) in het_residues]
            if het_atoms:
                center_x = np.mean([a['x'] for a in het_atoms])
                center_y = np.mean([a['y'] for a in het_atoms])
                center_z = np.mean([a['z'] for a in het_atoms])
        
        # Find residues within cutoff
        res_counts = Counter()
        total_atoms = 0
        
        for a in atoms:
            dx = a['x'] - center_x
            dy = a['y'] - center_y
            dz = a['z'] - center_z
            dist = (dx*dx + dy*dy + dz*dz) ** 0.5
            
            if dist < cutoff:
                res_counts[a['resname']] += 1
                total_atoms += 1
        
        if total_atoms == 0:
            return np.zeros(50, dtype=np.float32)
        
        # Common amino acids (20)
        common_aa = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 
                    'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER',
                    'THR', 'TRP', 'TYR', 'VAL']
        
        # Normalized AA composition
        aa_features = [res_counts.get(aa, 0) / max(total_atoms, 1) for aa in common_aa]
        
        # Additional features
        # Properties of binding site
        num_residues = len(res_counts)
        num_atoms_site = total_atoms
        
        # Charge categories
        neg = res_counts.get('ASP', 0) + res_counts.get('GLU', 0)
        pos = res_counts.get('LYS', 0) + res_counts.get('ARG', 0) + res_counts.get('HIS', 0)
        polar = res_counts.get('SER', 0) + res_counts.get('THR', 0) + res_counts.get('ASN', 0) + res_counts.get('GLN', 0)
        hydrophobic = res_counts.get('ALA', 0) + res_counts.get('VAL', 0) + res_counts.get('LEU', 0) + res_counts.get('ILE', 0) + res_counts.get('MET', 0)
        
        extra = [
            num_residues / 50,  # normalized
            num_atoms_site / 500,
            neg / max(total_atoms, 1),
            pos / max(total_atoms, 1),
            polar / max(total_atoms, 1),
            hydrophobic / max(total_atoms, 1),
            (neg - pos) / max(total_atoms, 1),  # charge balance
        ]
        
        # Fill to 50
        extra = extra + [0.0] * (50 - len(aa_features) - len(extra))
        
        features = aa_features + extra[:50 - len(aa_features)]
        
        return np.array(features[:50], dtype=np.float32)
        
    except Exception as e:
        return np.zeros(50, dtype=np.float32)

if __name__ == "__main__":
    # Test
    import sys
    test_file = sys.argv[1] if len(sys.argv) > 1 else "/home/chow/.cache/geock_autoresearch/lp_pdb_files/6p87.pdb"
    feats = extract_simple_pocket(test_file)
    print(f"Features shape: {feats.shape}")
    print(f"Non-zero: {(feats > 0).sum()}")
    print(f"Sample: {feats[:10]}")