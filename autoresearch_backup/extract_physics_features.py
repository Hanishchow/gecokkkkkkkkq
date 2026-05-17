#!/usr/bin/env python3
"""
Extract Physics Features from PDB Files
======================================
Extract Vinardo scoring terms and pocket characteristics from PDB files.

FEATURES:
- Vinardo-like scoring terms (simplified)
- Pocket characteristics (volume, surface, charge)
- Interaction counts (H-bonds, hydrophobic contacts, etc.)
- Atom type distributions in binding site
"""

import os
import sys
import pickle
import logging
import argparse
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
log = logging.getLogger(__name__)

WORK_DIR = Path("/home/chow/autoresearch")
CACHE_DIR = Path("/home/chow/.cache/geock_autoresearch")
PDB_DIR = CACHE_DIR / "lp_pdb_files"

@dataclass
class PhysicsFeatures:
    """Physics-based features extracted from PDB."""
    # Van der Waals interactions (simplified)
    vdw_clash: float = 0.0
    vdw_contact: float = 0.0
    vdw_total: float = 0.0
    
    # Hydrogen bonding
    hbond_donor_acceptor: float = 0.0
    hbond_score: float = 0.0
    hbond_dist_avg: float = 0.0
    
    # Hydrophobic interactions
    hydrophobic_contacts: float = 0.0
    hydrophobic_ratio: float = 0.0
    
    # Electrostatic
    electrostatic: float = 0.0
    charge_balance: float = 0.0
    
    # Pocket characteristics
    pocket_volume: float = 0.0
    pocket_surface: float = 0.0
    pocket_depth: float = 0.0
    pocket_polarity: float = 0.0
    
    # Ligand features
    ligand_heavy_atoms: int = 0
    ligand_hetero_atoms: int = 0
    ligand_mol_weight: float = 0.0
    ligand_logp: float = 0.0
    ligand_tpsa: float = 0.0
    ligand_rotatable_bonds: int = 0
    ligand_aromatic_rings: int = 0
    ligand_hbd: int = 0
    ligand_hba: int = 0
    
    # Atom counts in binding site
    n_carbon: int = 0
    n_nitrogen: int = 0
    n_oxygen: int = 0
    n_sulfur: int = 0
    n_halogen: int = 0
    
    # Metal interactions
    metal_contacts: int = 0
    pi_stacking: int = 0
    salt_bridges: int = 0
    
    # Water bridges
    water_bridges: int = 0
    displaced_waters: int = 0
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([
            self.vdw_clash, self.vdw_contact, self.vdw_total,
            self.hbond_donor_acceptor, self.hbond_score, self.hbond_dist_avg,
            self.hydrophobic_contacts, self.hydrophobic_ratio,
            self.electrostatic, self.charge_balance,
            self.pocket_volume, self.pocket_surface, self.pocket_depth, self.pocket_polarity,
            self.ligand_heavy_atoms, self.ligand_hetero_atoms, self.ligand_mol_weight,
            self.ligand_logp, self.ligand_tpsa, self.ligand_rotatable_bonds,
            self.ligand_aromatic_rings, self.ligand_hbd, self.ligand_hba,
            self.n_carbon, self.n_nitrogen, self.n_oxygen, self.n_sulfur, self.n_halogen,
            self.metal_contacts, self.pi_stacking, self.salt_bridges,
            self.water_bridges, self.displaced_waters
        ], dtype=np.float32)


def parse_pdb_line(line: str) -> Optional[Dict]:
    """Parse a single PDB ATOM/HETATM line."""
    if not (line.startswith('ATOM') or line.startswith('HETATM')):
        return None
    
    try:
        record_type = line[0:6].strip()
        atom_serial = int(line[6:11].strip())
        atom_name = line[12:16].strip()
        alt_loc = line[16]
        res_name = line[17:20].strip()
        chain = line[21]
        res_seq = int(line[22:26].strip())
        x = float(line[30:38].strip())
        y = float(line[38:46].strip())
        z = float(line[46:54].strip())
        occupancy = float(line[54:60].strip()) if line[54:60].strip() else 1.0
        temp_factor = float(line[60:66].strip()) if line[60:66].strip() else 0.0
        element = line[76:78].strip()
        
        return {
            'record': record_type,
            'serial': atom_serial,
            'name': atom_name,
            'alt_loc': alt_loc,
            'res_name': res_name,
            'chain': chain,
            'res_seq': res_seq,
            'x': x, 'y': y, 'z': z,
            'occupancy': occupancy,
            'temp_factor': temp_factor,
            'element': element
        }
    except Exception:
        return None


def is_hydrogen(atom: Dict) -> bool:
    """Check if atom is hydrogen."""
    return atom['element'] in ['H', 'D']


def is_ligand(atom: Dict) -> bool:
    """Check if atom is part of ligand (HETATM not WAT, ions)."""
    return atom['record'] == 'HETATM' and atom['res_name'] not in ['HOH', 'WAT', 'NA', 'CL', 'MG', 'CA', 'ZN', 'FE', 'MN', 'K', 'NA']


def is_protein(atom: Dict) -> bool:
    """Check if atom is part of protein."""
    return atom['record'] == 'ATOM' or (atom['record'] == 'HETATM' and atom['res_name'] in ['MSE', 'SEC'])


def calc_distance(p1: Tuple, p2: Tuple) -> float:
    """Calculate Euclidean distance between two points."""
    return np.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))


def calc_vdw_radius(element: str) -> float:
    """Get Van der Waals radius for element."""
    vdw_radii = {
        'H': 1.20, 'C': 1.70, 'N': 1.55, 'O': 1.52, 'S': 1.80,
        'P': 1.80, 'FE': 2.04, 'ZN': 2.01, 'CA': 1.97, 'MG': 1.41,
        'MN': 1.61, 'CL': 1.75, 'NA': 2.27, 'K': 2.75, 'CU': 1.40,
        'NI': 1.63, 'CO': 1.50, 'BR': 1.85, 'I': 1.98, 'F': 1.47,
        'SE': 1.90, 'AS': 1.85
    }
    return vdw_radii.get(element.upper(), 1.80)


def is_hbond_donor(atom_name: str, element: str, res_name: str) -> bool:
    """Check if atom can be H-bond donor."""
    if res_name in ['SER', 'THR', 'TYR', 'CYS']:
        if atom_name in ['OG', 'OG1', 'OH', 'SG']:
            return True
    if res_name in ['LYS', 'ARG']:
        if element == 'N':
            return True
    if res_name in ['ASN', 'GLN', 'HIS']:
        if element == 'N' and atom_name in ['ND1', 'NE2', 'N']:
            return True
    if res_name in ['ASP', 'GLU']:
        if element == 'O':
            return True
    if res_name == 'MET' and element == 'S':
        return True
    return False


def is_hbond_acceptor(atom_name: str, element: str, res_name: str) -> bool:
    """Check if atom can be H-bond acceptor."""
    if res_name in ['SER', 'THR', 'TYR']:
        if atom_name in ['OG', 'OG1', 'OH']:
            return True
    if res_name in ['ASN', 'GLN', 'ASP', 'GLU', 'HIS']:
        if element == 'O' or (element == 'N' and atom_name in ['ND1', 'NE2']):
            return True
    if res_name == 'MET' and element == 'S':
        return True
    return False


def is_hydrophobic(element: str, res_name: str) -> bool:
    """Check if atom/residue is hydrophobic."""
    if element in ['C', 'S']:
        return True
    if res_name in ['ALA', 'VAL', 'ILE', 'LEU', 'MET', 'PHE', 'TYR', 'TRP', 'PRO']:
        return True
    return False


def extract_physics_from_pdb(pdb_path: Path) -> Tuple[Optional[str], Optional[PhysicsFeatures]]:
    """Extract physics features from a single PDB file."""
    try:
        with open(pdb_path, 'r') as f:
            lines = f.readlines()
        
        atoms = []
        for line in lines:
            atom = parse_pdb_line(line)
            if atom:
                atoms.append(atom)
        
        if not atoms:
            return pdb_path.stem, None
        
        pdb_id = pdb_path.stem
        features = PhysicsFeatures()
        
        ligand_atoms = [a for a in atoms if is_ligand(a)]
        protein_atoms = [a for a in atoms if is_protein(a) and not is_hydrogen(a)]
        
        if not ligand_atoms:
            return pdb_id, None
        
        ligand_coords = [(a['x'], a['y'], a['z']) for a in ligand_atoms]
        protein_coords = [(a['x'], a['y'], a['z']) for a in protein_atoms]
        
        binding_site_atoms = []
        for lc in ligand_coords:
            for pc in protein_coords:
                dist = calc_distance(lc, pc)
                if dist < 8.0:
                    binding_site_atoms.append(pc)
                    break
        
        features.ligand_heavy_atoms = len([a for a in ligand_atoms if not is_hydrogen(a)])
        features.ligand_hetero_atoms = len(ligand_atoms)
        
        elem_counts = defaultdict(int)
        for a in ligand_atoms:
            elem_counts[a['element']] += 1
        
        features.n_carbon = elem_counts.get('C', 0)
        features.n_nitrogen = elem_counts.get('N', 0)
        features.n_oxygen = elem_counts.get('O', 0)
        features.n_sulfur = elem_counts.get('S', 0)
        features.n_halogen = elem_counts.get('CL', 0) + elem_counts.get('BR', 0) + elem_counts.get('F', 0) + elem_counts.get('I', 0)
        
        features.ligand_mol_weight = features.n_carbon * 12.01 + features.n_nitrogen * 14.01 + features.n_oxygen * 16.00 + features.n_sulfur * 32.07
        
        vdw_contact = 0.0
        vdw_clash = 0.0
        hbond_score = 0.0
        hbond_dists = []
        hydrophobic_contacts = 0
        electrostatic = 0.0
        metal_contacts = 0
        salt_bridges = 0
        water_bridges = 0
        
        for la in ligand_atoms:
            if is_hydrogen(la):
                continue
            
            la_pos = (la['x'], la['y'], la['z'])
            la_elem = la['element']
            la_vdw = calc_vdw_radius(la_elem)
            
            for pa in protein_atoms:
                if is_hydrogen(pa):
                    continue
                
                pa_pos = (pa['x'], pa['y'], pa['z'])
                pa_elem = pa['element']
                pa_vdw = calc_vdw_radius(pa_elem)
                pa_res = pa['res_name']
                
                dist = calc_distance(la_pos, pa_pos)
                
                combined_vdw = la_vdw + pa_vdw
                vdw_dist = dist - combined_vdw
                
                if dist < 1.0:
                    vdw_clash += 10.0
                elif dist < combined_vdw + 0.5:
                    vdw_contact += vdw_dist * -0.5
                
                if dist < 3.5 and is_hydrophobic(la_elem, la['res_name']) and is_hydrophobic(pa_elem, pa_res):
                    hydrophobic_contacts += 1
                
                if is_hbond_donor(pa['name'], pa_elem, pa_res) and is_hbond_acceptor(la['name'], la_elem, la['res_name']):
                    if 2.5 < dist < 3.5:
                        hbond_score += (3.5 - dist) * 2.0
                        hbond_dists.append(dist)
                
                if is_hbond_acceptor(pa['name'], pa_elem, pa_res) and is_hbond_donor(la['name'], la_elem, la['res_name']):
                    if 2.5 < dist < 3.5:
                        hbond_score += (3.5 - dist) * 2.0
                        hbond_dists.append(dist)
                
                if pa_elem in ['FE', 'ZN', 'CA', 'MG', 'MN', 'CU', 'NI', 'CO']:
                    if dist < 3.0:
                        metal_contacts += 1
                
                if (pa_elem in ['NA', 'K', 'CA'] and la_elem in ['O', 'N']) or (pa_elem in ['O', 'N'] and la_elem in ['NA', 'K', 'CA']):
                    if dist < 4.0:
                        salt_bridges += 1
                
                if la_elem in ['O', 'N'] and pa_elem in ['O', 'N']:
                    if 3.0 < dist < 4.5:
                        water_bridges += 0.5
        
        features.vdw_clash = min(vdw_clash, 50.0)
        features.vdw_contact = vdw_contact
        features.vdw_total = vdw_contact - vdw_clash
        
        features.hbond_donor_acceptor = len(hbond_dists)
        features.hbond_score = hbond_score
        features.hbond_dist_avg = np.mean(hbond_dists) if hbond_dists else 3.5
        
        features.hydrophobic_contacts = hydrophobic_contacts
        features.n_carbon = max(features.n_carbon, 1)
        features.hydrophobic_ratio = features.n_carbon / (features.n_carbon + features.n_nitrogen + features.n_oxygen + 1)
        
        features.electrostatic = electrostatic
        features.charge_balance = (elem_counts.get('N', 0) + elem_counts.get('O', 0)) / (features.n_carbon + 1)
        
        features.metal_contacts = metal_contacts
        features.salt_bridges = salt_bridges
        features.water_bridges = int(water_bridges)
        
        pi_rings = 0
        ring_atoms = [a for a in ligand_atoms if a['res_name'] in ['PHE', 'TYR', 'TRP', 'HIS', 'N']
                     or (a['element'] == 'C' and any(n in a['name'] for n in ['CG', 'CD', 'CE', 'CZ']))]
        for i, ra in enumerate(ring_atoms[:6]):
            for rb in ring_atoms[i+1:min(i+6, len(ring_atoms))]:
                if calc_distance((ra['x'], ra['y'], ra['z']), (rb['x'], rb['y'], rb['z'])) < 2.0:
                    pi_rings += 0.5
        features.pi_stacking = int(pi_rings)
        features.ligand_aromatic_rings = int(pi_rings / 6)
        
        all_coords = ligand_coords + protein_coords
        if len(all_coords) > 2:
            coords_array = np.array(all_coords)
            features.pocket_volume = np.prod(coords_array.max(axis=0) - coords_array.min(axis=0)) / 1000.0
        
        features.pocket_polarity = (features.n_nitrogen + features.n_oxygen) / (features.n_carbon + features.n_nitrogen + features.n_oxygen + 1)
        
        features.ligand_rotatable_bonds = int(features.ligand_heavy_atoms * 0.1)
        features.ligand_hbd = features.n_nitrogen
        features.ligand_hba = features.n_oxygen
        features.ligand_tpsa = features.n_oxygen * 12.5 + features.n_nitrogen * 15.5
        
        features.ligand_logp = features.n_carbon * 0.5 - features.n_oxygen * 0.3 - features.n_nitrogen * 0.2
        
        return pdb_id, features
        
    except Exception as e:
        return pdb_path.stem, None


def process_pdb_wrapper(args):
    """Wrapper for parallel processing."""
    return extract_physics_from_pdb(args)


def main():
    parser = argparse.ArgumentParser(description="Extract physics features from PDB files")
    parser.add_argument('--workers', type=int, default=8, help="Number of parallel workers")
    parser.add_argument('--batch-size', type=int, default=1000, help="Batch size for progress reporting")
    args = parser.parse_args()
    
    output_path = CACHE_DIR / "physics_features.pkl"
    enhanced_path = CACHE_DIR / "lp_features_enhanced.pkl"
    
    if not PDB_DIR.exists():
        log.error(f"PDB directory not found: {PDB_DIR}")
        return
    
    log.info(f"Scanning PDB directory: {PDB_DIR}")
    pdb_files = sorted(PDB_DIR.glob("*.pdb"))
    log.info(f"Found {len(pdb_files)} PDB files")
    
    if not pdb_files:
        log.error("No PDB files found")
        return
    
    pdb_ids_to_process = {p.stem: p for p in pdb_files}
    
    log.info("Loading enhanced features to get PDB IDs...")
    with open(enhanced_path, 'rb') as f:
        enhanced_data = pickle.load(f)
    
    valid_pdb_ids = {record['pdb_id']: record for record in enhanced_data if 'pdb_id' in record}
    log.info(f"Valid PDB IDs from enhanced features: {len(valid_pdb_ids)}")
    
    pdb_files_to_process = [pdb_files[i] for i, p in enumerate(pdb_files) if p.stem in valid_pdb_ids]
    log.info(f"PDB files to process: {len(pdb_files_to_process)}")
    
    physics_by_pdb = {}
    processed = 0
    failed = 0
    
    log.info(f"Extracting physics features using {args.workers} workers...")
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(extract_physics_from_pdb, pdb_file): pdb_file for pdb_file in pdb_files_to_process}
        
        for future in as_completed(futures):
            pdb_file = futures[future]
            try:
                pdb_id, features = future.result()
                if features is not None:
                    physics_by_pdb[pdb_id] = features
                    processed += 1
                else:
                    failed += 1
                
                if (processed + failed) % args.batch_size == 0:
                    log.info(f"Progress: {processed} processed, {failed} failed")
                    
            except Exception as e:
                failed += 1
                log.warning(f"Failed to process {pdb_file.name}: {e}")
    
    log.info(f"Extraction complete: {processed} successful, {failed} failed")
    
    feature_records = []
    feature_names = [
        'vdw_clash', 'vdw_contact', 'vdw_total',
        'hbond_donor_acceptor', 'hbond_score', 'hbond_dist_avg',
        'hydrophobic_contacts', 'hydrophobic_ratio',
        'electrostatic', 'charge_balance',
        'pocket_volume', 'pocket_surface', 'pocket_depth', 'pocket_polarity',
        'ligand_heavy_atoms', 'ligand_hetero_atoms', 'ligand_mol_weight',
        'ligand_logp', 'ligand_tpsa', 'ligand_rotatable_bonds',
        'ligand_aromatic_rings', 'ligand_hbd', 'ligand_hba',
        'n_carbon', 'n_nitrogen', 'n_oxygen', 'n_sulfur', 'n_halogen',
        'metal_contacts', 'pi_stacking', 'salt_bridges',
        'water_bridges', 'displaced_waters'
    ]
    
    for record in enhanced_data:
        pdb_id = record.get('pdb_id')
        if pdb_id and pdb_id in physics_by_pdb:
            physics_feat = physics_by_pdb[pdb_id]
            physics_array = physics_feat.to_array()
            
            new_record = {
                **record,
                'physics': physics_array,
                'physics_names': feature_names
            }
            feature_records.append(new_record)
    
    log.info(f"Created {len(feature_records)} records with physics features")
    
    with open(output_path, 'wb') as f:
        pickle.dump({
            'records': feature_records,
            'feature_names': list(record.keys()) + feature_names,
            'physics_names': feature_names,
            'n_records': len(feature_records),
            'n_physics_features': len(feature_names)
        }, f)
    
    log.info(f"Saved physics features to {output_path}")
    
    physics_arrays = [r['physics'] for r in feature_records]
    physics_matrix = np.array(physics_arrays)
    log.info(f"Physics feature matrix shape: {physics_matrix.shape}")
    log.info(f"Feature stats: mean={physics_matrix.mean():.3f}, std={physics_matrix.std():.3f}")
    
    for i, name in enumerate(feature_names):
        log.info(f"  {i}: {name} - mean={physics_matrix[:, i].mean():.3f}, std={physics_matrix[:, i].std():.3f}")


if __name__ == "__main__":
    main()
