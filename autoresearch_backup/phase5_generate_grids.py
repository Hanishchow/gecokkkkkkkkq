#!/usr/bin/env python3
"""
PHASE 5: 3D Grid Generation for CNN
Generates 3D voxel grids from PDB files for 3D CNN training.
"""
import os
import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Atom type encoding (19 channels)
ATOM_CHANNEL = {
    'C': 0, 'N': 1, 'O': 2, 'S': 3, 'P': 4, 'F': 5, 'CL': 6, 'BR': 7, 'I': 8,
    'CA': 9, 'MG': 10, 'ZN': 11, 'FE': 12, 'MN': 13, 'K': 14, 'NA': 15,
    'CA': 16, 'HG': 17, 'XE': 18
}

# Residue type for pocket
RESIDUE_CHANNEL = {
    'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4, 'GLN': 5, 'GLU': 6,
    'GLY': 7, 'HIS': 8, 'ILE': 9, 'LEU': 10, 'LYS': 11, 'MET': 12,
    'PHE': 13, 'PRO': 14, 'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19
}

LIGAND_RESIDUES = set([
    'LIG', 'ATP', 'ADP', 'NAD', 'NAP', 'COA', 'FAD', 'SAM', 'GTP', 'GDP',
    'UNL', 'UNX', 'MAN', 'NAG', 'BGC', 'GLC', 'GAL', 'FUL', 'XYS', 'RIB'
])


def parse_pdb_atoms(pdb_path):
    """Parse PDB file and extract atom coordinates and types."""
    atoms = []
    
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                elem = line[76:78].strip().upper()
                if not elem:
                    elem = line[12:14].strip().upper().replace(' ', '')
                
                # Skip hydrogens and waters
                if elem in ['', 'H', 'HOH']:
                    continue
                
                resname = line[17:20].strip().upper()
                is_ligand = resname in LIGAND_RESIDUES
                
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                
                atom_type = ATOM_CHANNEL.get(elem, 0)
                res_type = RESIDUE_CHANNEL.get(resname, 20)
                
                atoms.append({
                    'coords': np.array([x, y, z], dtype=np.float32),
                    'element': elem,
                    'resname': resname,
                    'is_ligand': is_ligand,
                    'atom_type': atom_type,
                    'res_type': res_type
                })
    
    return atoms


def extract_binding_site(atoms, ligand_atoms=None, cutoff=10.0):
    """Extract binding site around ligand or center of mass."""
    # Find ligand or compute center of mass
    if ligand_atoms is None:
        ligand_atoms = [a for a in atoms if a['is_ligand']]
    
    if len(ligand_atoms) > 0:
        center = np.mean([a['coords'] for a in ligand_atoms], axis=0)
    else:
        # Use all atoms center
        center = np.mean([a['coords'] for a in atoms], axis=0)
    
    # Extract atoms within cutoff
    site_atoms = []
    for atom in atoms:
        dist = np.linalg.norm(atom['coords'] - center)
        if dist <= cutoff:
            site_atoms.append({
                'coords': atom['coords'] - center,  # Center at origin
                'atom_type': atom['atom_type'],
                'res_type': atom['res_type'],
                'is_ligand': atom['is_ligand']
            })
    
    return site_atoms, center


def create_3d_grid(site_atoms, grid_size=24, resolution=0.5):
    """Create 3D voxel grid with atom type channels."""
    # Initialize grid: (channels, depth, height, width)
    channels = 19 + 20 + 1  # atom types + residue types + ligand mask
    grid = np.zeros((channels, grid_size, grid_size, grid_size), dtype=np.float32)
    
    half_size = grid_size * resolution / 2
    
    for atom in site_atoms:
        # Map to grid indices
        x_idx = int((atom['coords'][0] + half_size) / resolution)
        y_idx = int((atom['coords'][1] + half_size) / resolution)
        z_idx = int((atom['coords'][2] + half_size) / resolution)
        
        # Check bounds
        if 0 <= x_idx < grid_size and 0 <= y_idx < grid_size and 0 <= z_idx < grid_size:
            # Atom type channel
            if atom['atom_type'] < 19:
                grid[atom['atom_type'], x_idx, y_idx, z_idx] = 1
            
            # Residue type channel (offset by 19)
            if atom['res_type'] < 20:
                grid[19 + atom['res_type'], x_idx, y_idx, z_idx] = 1
            
            # Ligand mask channel (last channel)
            if atom['is_ligand']:
                grid[-1, x_idx, y_idx, z_idx] = 1
    
    return grid


def process_pdb(pdb_path, grid_size=24, resolution=0.5):
    """Process single PDB file to generate 3D grid."""
    try:
        atoms = parse_pdb_atoms(pdb_path)
        
        if len(atoms) == 0:
            return None
        
        site_atoms, center = extract_binding_site(atoms, cutoff=10.0)
        
        if len(site_atoms) < 5:  # Minimum atoms
            return None
        
        grid = create_3d_grid(site_atoms, grid_size=grid_size, resolution=resolution)
        
        return {
            'grid': grid,
            'center': center,
            'num_atoms': len(site_atoms)
        }
    
    except Exception as e:
        print(f'Error processing {pdb_path}: {e}')
        return None


def generate_all_grids(pdb_dir, output_path, max_files=None, grid_size=24, resolution=0.5):
    """Generate grids for all PDB files."""
    pdb_files = sorted([f for f in os.listdir(pdb_dir) if f.endswith('.pdb')])
    
    if max_files:
        pdb_files = pdb_files[:max_files]
    
    print(f'Processing {len(pdb_files)} PDB files...')
    
    grids = []
    ids = []
    success = 0
    failed = 0
    
    for i, pdb_file in enumerate(pdb_files):
        if i % 500 == 0:
            print(f'Progress: {i}/{len(pdb_files)}')
        
        pdb_path = os.path.join(pdb_dir, pdb_file)
        pdb_id = pdb_file.replace('.pdb', '')
        
        result = process_pdb(pdb_path, grid_size=grid_size, resolution=resolution)
        
        if result is not None:
            grids.append(result['grid'])
            ids.append(pdb_id)
            success += 1
        else:
            failed += 1
    
    print(f'Done: {success} success, {failed} failed')
    
    # Save
    output = {
        'grids': np.array(grids, dtype=np.float32),
        'ids': ids,
        'grid_size': grid_size,
        'resolution': resolution
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(output, f)
    
    print(f'Saved to {output_path}')
    grids_shape = output["grids"].shape
    print(f'Grid shape: {grids_shape}')
    
    return output


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate 3D grids from PDB files')
    parser.add_argument('--pdb-dir', default='CACHE_DIR / lp_pdb_files')
    parser.add_argument('--output', default='WORK_DIR / 3d_grids.pkl')
    parser.add_argument('--max', type=int, default=None, help='Max files to process')
    parser.add_argument('--grid-size', type=int, default=24, help='Grid dimension')
    parser.add_argument('--resolution', type=float, default=0.5, help='Voxel resolution (Angstrom)')
    
    args = parser.parse_args()
    
    generate_all_grids(args.pdb_dir, args.output, max_files=args.max,
                       grid_size=args.grid_size, resolution=args.resolution)