#!/usr/bin/env python3
"""
Extract ligand from PDB HETATM records and use for scoring.
"""

import os
import numpy as np

def extract_ligand_from_pdb(pdb_path, cutoff=10.0):
    """Extract ligand atoms from HETATM records in PDB.
    Returns (coords, types, resnames).
    """
    WATER = {"HOH", "WAT", "H2O", "DOD"}
    
    lig_coords = []
    lig_types = []
    ligand_resnames = set()
    
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("HETATM"):
                continue
            resname = line[17:20].strip().upper()
            if resname in WATER:
                continue
            
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            
            ligand_resnames.add(resname)
            
            el = line[76:78].strip().upper() if len(line) > 76 else ""
            name = line[12:16].strip()
            
            if not el:
                el = ''.join(c for c in name if c.isalpha())[:2]
                el = el[0] if el else "C"
            
            if el.upper() in ("H", "D"):
                continue
            
            lig_coords.append([x, y, z])
            if el == "N": lig_types.append("NA")
            elif el == "O": lig_types.append("OA")
            elif el == "S": lig_types.append("SA")
            else: lig_types.append(el)
    
    return np.array(lig_coords, dtype=np.float32), lig_types, ligand_resnames


def main():
    import sys
    sys.path.insert(0, '/mnt/c/Users/yakka/Downloads/final/geock')
    from score_compound import score_physics
    
    test_cases = [
        ('1a1e', '/mnt/c/Users/yakka/Downloads/geock_110_data/1a1e/1a1e_pocket.pdb', -8.3),
        ('3phy', '/mnt/c/Users/yakka/Downloads/geock_110_data/3phy/3phy_pocket.pdb', -9.2),
        ('1stp', '/mnt/c/Users/yakka/Downloads/geock_110_data/1stp/1stp_pocket.pdb', -8.9),
    ]
    
    print("Testing with HETATM ligands:")
    for pdb_id, pdb_path, exp_dG in test_cases:
        print(f"\n{pdb_id}: exp dG={exp_dG}, pKd={-exp_dG/1.364:.2f}")
        
        lig_coords, lig_types, resnames = extract_ligand_from_pdb(pdb_path)
        print(f"  Ligand residues: {resnames}, {len(lig_coords)} atoms")
        
        if len(lig_coords) == 0:
            print("  No ligand found!")
            continue
        
        centroid = lig_coords.mean(axis=0)
        
        rec_coords, rec_types, _ = extract_ligand_from_pdb(pdb_path)
        
        WATER = {"HOH", "WAT", "H2O", "DOD"}
        receptor_coords = []
        receptor_types = []
        
        with open(pdb_path) as f:
            for line in f:
                if not line.startswith("ATOM"):
                    continue
                resname = line[17:20].strip().upper()
                if resname in WATER:
                    continue
                
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    xyz = np.array([x, y, z])
                except ValueError:
                    continue
                
                if np.linalg.norm(xyz - centroid) > 10.0:
                    continue
                
                el = line[76:78].strip().upper() if len(line) > 76 else ""
                name = line[12:16].strip()
                
                if not el:
                    el = ''.join(c for c in name if c.isalpha())[:2]
                    el = el[0] if el else "C"
                
                if el.upper() in ("H", "D"):
                    continue
                
                receptor_coords.append(xyz)
                if el == "N": receptor_types.append("NA")
                elif el == "O": receptor_types.append("OA")
                elif el == "S": receptor_types.append("SA")
                else: receptor_types.append(el)
        
        rec_coords = np.array(receptor_coords, dtype=np.float32)
        
        print(f"  Receptor: {len(rec_coords)} atoms within 10Å")
        
        result = score_physics(rec_coords, rec_types, lig_coords, lig_types, 3)
        print(f"  Score: raw_vina={result['raw_vina']:.2f}, clashes={result['n_clashes']}")


if __name__ == "__main__":
    main()
