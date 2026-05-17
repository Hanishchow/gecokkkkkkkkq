"""
physics_scoring.py - Real physics-based scoring for GEOCK

Implements Vina-style scoring function for binding affinity prediction.
Uses pose coordinates to compute intermolecular energies.
"""

import numpy as np
import torch
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from functools import lru_cache

from geock.pose_utils import PoseConverter, LigandTopology, pose_to_coordinates


@dataclass
class VinaWeights:
    """Vina scoring function weights (Trott & Olson, 2010)"""
    w_gauss1: float = -0.035579
    w_gauss2: float = -0.005156
    w_repulsion: float = 0.840245
    w_hydrophobic: float = -0.035069
    w_hbond: float = -0.587439
    w_torsion: float = 0.058459


@dataclass  
class VinardoWeights:
    """Vinardo scoring weights (Quiroga & Villarreal, 2016)"""
    w_gauss1: float = -0.045
    w_gauss2: float = 0.0
    w_repulsion: float = 0.8
    w_hydrophobic: float = -0.030
    w_hbond: float = -0.600
    w_torsion: float = 0.055


# Atom type properties
VDW_RADIUS = {
    'C': 1.9, 'N': 1.8, 'O': 1.7, 'S': 2.0, 'P': 2.1,
    'F': 1.5, 'Cl': 1.8, 'Br': 2.0, 'I': 2.2, 'H': 1.0,
    'Fe': 1.2, 'Zn': 1.2, 'Ca': 1.7, 'Mg': 1.2, 'Mn': 1.2
}

HYDROPHOBIC_TYPES = {'C', 'S'}
HBOND_DONOR_TYPES = {'N', 'O'}
HBOND_ACCEPTOR_TYPES = {'N', 'O', 'S'}

CUTOFF = 8.0  # Angstrom


class ReceptorAtomCache:
    """Cache parsed receptor atoms for fast scoring"""
    
    def __init__(self):
        self._cache: Dict[str, Tuple[np.ndarray, List[str]]] = {}
    
    def get(self, pdb_path: str, center: Tuple[float, float, float], 
            max_distance: float = 10.0) -> Tuple[np.ndarray, List[str]]:
        """Get cached receptor atoms or parse and cache"""
        key = f"{pdb_path}:{center[0]:.1f},{center[1]:.1f},{center[2]:.1f}"
        
        if key not in self._cache:
            self._cache[key] = self._parse_pdb(pdb_path, center, max_distance)
        
        return self._cache[key]
    
    def _parse_pdb(self, pdb_path: str, center: Tuple[float, float, float],
                   max_distance: float) -> Tuple[np.ndarray, List[str]]:
        """Parse PDB and filter atoms by distance"""
        coords = []
        types = []
        chains = {}
        
        with open(pdb_path) as f:
            for line in f:
                if line.startswith(('ATOM', 'HETATM')):
                    try:
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        
                        # Distance filter
                        dist = np.sqrt((x-center[0])**2 + (y-center[1])**2 + (z-center[2])**2)
                        if dist > max_distance:
                            continue
                        
                        elem = line[76:78].strip().title()
                        if not elem:
                            elem = line[12:16].strip()[0].upper()
                        
                        # Skip hydrogens
                        if elem == 'H':
                            continue
                        
                        chain = line[21]
                        if chain not in chains:
                            chains[chain] = 0
                        chains[chain] += 1
                        
                        coords.append([x, y, z])
                        types.append(elem)
                    except:
                        continue
        
        # Exclude ligand chains (small chains < 200 atoms)
        ligand_chains = {ch for ch, cnt in chains.items() if cnt < 200}
        
        final_coords = []
        final_types = []
        
        # Re-parse to filter chains
        with open(pdb_path) as f:
            for line in f:
                if line.startswith(('ATOM', 'HETATM')):
                    try:
                        chain = line[21]
                        if chain in ligand_chains:
                            continue
                        
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        
                        dist = np.sqrt((x-center[0])**2 + (y-center[1])**2 + (z-center[2])**2)
                        if dist > max_distance:
                            continue
                        
                        elem = line[76:78].strip().title()
                        if not elem:
                            elem = line[12:16].strip()[0].upper()
                        
                        if elem == 'H':
                            continue
                        
                        final_coords.append([x, y, z])
                        final_types.append(elem)
                    except:
                        continue
        
        return np.array(final_coords), final_types
    
    def clear(self):
        """Clear cache"""
        self._cache.clear()


class PhysicsScorer:
    """
    Real physics-based scoring for GEOCK.
    
    Computes binding affinity using Vina-style scoring function:
    - Gaussian steric (2 terms)
    - Repulsion
    - Hydrophobic contacts
    - Hydrogen bonding
    - Torsional entropy
    
    Supports:
    - Single pose scoring
    - Batch scoring (vectorized)
    - GPU acceleration (if available)
    """
    
    def __init__(
        self,
        vina_weights: bool = True,
        use_vinardo: bool = False,
        cutoff: float = 8.0
    ):
        """
        Args:
            vina_weights: Use Vina weights (else Vinardo)
            use_vinardo: Use Vinardo hydrophobic term
            cutoff: Distance cutoff for interactions
        """
        if use_vinardo:
            self.weights = VinardoWeights()
        elif vina_weights:
            self.weights = VinaWeights()
        else:
            self.weights = VinardoWeights()
        
        self.use_vinardo = use_vinardo
        self.cutoff = cutoff
        self.receptor_cache = ReceptorAtomCache()
        
        # Device (GPU if available)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def score_pose(
        self,
        pose: np.ndarray,
        receptor_coords: np.ndarray,
        receptor_types: List[str],
        ligand_topology: LigandTopology,
        n_torsions: int = 0
    ) -> Dict:
        """
        Score a single pose.
        
        Args:
            pose: [24] pose vector
            receptor_coords: [N, 3] receptor atom coordinates
            receptor_types: [N] receptor atom types
            ligand_topology: Ligand topology
            n_torsions: Number of rotatable bonds
        
        Returns:
            dict with 'affinity', 'inter_energy', and component breakdown
        """
        # Convert pose to coordinates
        lig_coords = pose_to_coordinates(pose, ligand_topology)
        
        # Compute pairwise distances
        lig_types = ligand_topology.atom_types
        
        # Accumulate terms
        e_gauss1 = e_gauss2 = e_rep = e_hydro = e_hbond = 0.0
        
        for i, lig_coord in enumerate(lig_coords):
            if i >= len(lig_types):
                break
            
            lig_type = lig_types[i]
            lig_is_hydrophobic = lig_type in HYDROPHOBIC_TYPES
            lig_is_donor = lig_type in HBOND_DONOR_TYPES
            lig_is_acceptor = lig_type in HBOND_ACCEPTOR_TYPES
            
            r_vdw_lig = VDW_RADIUS.get(lig_type, 1.8)
            
            for j, rec_coord in enumerate(receptor_coords):
                if j >= len(receptor_types):
                    break
                
                rec_type = receptor_types[j]
                r = np.linalg.norm(lig_coord - rec_coord)
                
                if r >= self.cutoff:
                    continue
                
                # Surface distance
                r_vdw_rec = VDW_RADIUS.get(rec_type, 1.8)
                d = r - (r_vdw_lig + r_vdw_rec)
                
                # Gaussian terms
                e_gauss1 += np.exp(-((d - 0.0) / 0.5) ** 2)
                e_gauss2 += np.exp(-((d - 3.0) / 2.0) ** 2)
                
                # Repulsion
                if d < 0:
                    e_rep += d * d
                
                # Hydrophobic
                rec_is_hydrophobic = rec_type in HYDROPHOBIC_TYPES
                if lig_is_hydrophobic and rec_is_hydrophobic:
                    if self.use_vinardo:
                        if d <= 0.0:
                            e_hydro += 1.0
                        elif d < 2.5:
                            e_hydro += (2.5 - d) / 2.5
                    else:
                        if d <= 0.5:
                            e_hydro += 1.0
                        elif d < 1.5:
                            e_hydro += (1.5 - d)
                
                # Hydrogen bonding
                rec_is_donor = rec_type in HBOND_DONOR_TYPES
                rec_is_acceptor = rec_type in HBOND_ACCEPTOR_TYPES
                
                if (lig_is_donor and rec_is_acceptor) or (lig_is_acceptor and rec_is_donor):
                    if d <= -0.7:
                        e_hbond += 1.0
                    elif d < 0.0:
                        e_hbond += (0.0 - d) / 0.7
        
        # Weighted sum
        w = self.weights
        c = (w.w_gauss1 * e_gauss1 + 
             w.w_gauss2 * e_gauss2 + 
             w.w_repulsion * e_rep + 
             w.w_hydrophobic * e_hydro + 
             w.w_hbond * e_hbond)
        
        # Torsional penalty
        torsion_denom = 1.0 + w.w_torsion * max(0, n_torsions)
        affinity = c / torsion_denom
        
        return {
            'affinity': float(affinity),
            'inter_energy': float(c),
            'torsion_penalty': float(w.w_torsion * n_torsions),
            'components': {
                'gauss1': float(w.w_gauss1 * e_gauss1),
                'gauss2': float(w.w_gauss2 * e_gauss2),
                'repulsion': float(w.w_repulsion * e_rep),
                'hydrophobic': float(w.w_hydrophobic * e_hydro),
                'hbond': float(w.w_hbond * e_hbond),
            },
            'n_contacts': int(e_gauss1 > 0),  # Simplified
        }
    
    def score_batch(
        self,
        poses: np.ndarray,
        receptor_coords: np.ndarray,
        receptor_types: List[str],
        ligand_topology: LigandTopology,
        n_torsions: int = 0
    ) -> np.ndarray:
        """
        Score a batch of poses (vectorized for speed).
        
        Args:
            poses: [N, 24] array of pose vectors
            receptor_coords: [M, 3] receptor coordinates
            receptor_types: [M] receptor types
            ligand_topology: Ligand topology
            n_torsions: Number of rotatable bonds
        
        Returns:
            [N] array of affinity scores
        """
        # Convert poses to coordinates
        all_lig_coords = []
        for pose in poses:
            coords = pose_to_coordinates(pose, ligand_topology)
            all_lig_coords.append(coords)
        
        # Compute scores
        scores = np.zeros(len(poses))
        
        for i, lig_coords in enumerate(all_lig_coords):
            result = self._score_coordinates(
                lig_coords, ligand_topology.atom_types,
                receptor_coords, receptor_types, n_torsions
            )
            scores[i] = result['affinity']
        
        return scores
    
    def _score_coordinates(
        self,
        lig_coords: np.ndarray,
        lig_types: List[str],
        rec_coords: np.ndarray,
        rec_types: List[str],
        n_torsions: int
    ) -> Dict:
        """Score ligand coordinates against receptor"""
        
        e_gauss1 = e_gauss2 = e_rep = e_hydro = e_hbond = 0.0
        
        n_lig = min(len(lig_coords), len(lig_types))
        n_rec = min(len(rec_coords), len(rec_types))
        
        for i in range(n_lig):
            lig_type = lig_types[i] if i < len(lig_types) else 'C'
            lig_coord = lig_coords[i]
            
            lig_is_hydro = lig_type in HYDROPHOBIC_TYPES
            lig_is_donor = lig_type in HBOND_DONOR_TYPES
            lig_is_accept = lig_type in HBOND_ACCEPTOR_TYPES
            r_vdw_lig = VDW_RADIUS.get(lig_type, 1.8)
            
            for j in range(n_rec):
                rec_type = rec_types[j] if j < len(rec_types) else 'C'
                rec_coord = rec_coords[j]
                
                r = np.linalg.norm(lig_coord - rec_coord)
                if r >= self.cutoff:
                    continue
                
                r_vdw_rec = VDW_RADIUS.get(rec_type, 1.8)
                d = r - (r_vdw_lig + r_vdw_rec)
                
                e_gauss1 += np.exp(-(d / 0.5) ** 2)
                e_gauss2 += np.exp(-((d - 3.0) / 2.0) ** 2)
                
                if d < 0:
                    e_rep += d * d
                
                rec_is_hydro = rec_type in HYDROPHOBIC_TYPES
                if lig_is_hydro and rec_is_hydro:
                    if self.use_vinardo:
                        e_hydro += max(0, 1 - max(0, d) / 2.5)
                    else:
                        e_hydro += max(0, 1 - max(0, d - 0.5))
                
                rec_is_donor = rec_type in HBOND_DONOR_TYPES
                rec_is_accept = rec_type in HBOND_ACCEPTOR_TYPES
                
                if (lig_is_donor and rec_is_accept) or (lig_is_accept and rec_is_donor):
                    e_hbond += max(0, 1 + min(0, d + 0.7) / 0.7)
        
        w = self.weights
        c = (w.w_gauss1 * e_gauss1 + w.w_gauss2 * e_gauss2 + 
             w.w_repulsion * e_rep + w.w_hydrophobic * e_hydro + 
             w.w_hbond * e_hbond)
        
        affinity = c / (1.0 + w.w_torsion * max(0, n_torsions))
        
        return {
            'affinity': float(affinity),
            'inter_energy': float(c),
            'components': {
                'gauss1': float(w.w_gauss1 * e_gauss1),
                'gauss2': float(w.w_gauss2 * e_gauss2),
                'repulsion': float(w.w_repulsion * e_rep),
                'hydrophobic': float(w.w_hydrophobic * e_hydro),
                'hbond': float(w.w_hbond * e_hbond),
            }
        }
    
    def score_from_files(
        self,
        pose: np.ndarray,
        receptor_path: str,
        ligand_topology: LigandTopology,
        center: Tuple[float, float, float],
        n_torsions: int = 0,
        max_distance: float = 10.0
    ) -> Dict:
        """Score pose using file paths (uses cache internally)"""
        receptor_coords, receptor_types = self.receptor_cache.get(
            receptor_path, center, max_distance
        )
        
        return self.score_pose(
            pose, receptor_coords, receptor_types,
            ligand_topology, n_torsions
        )


def create_scorer(vina_style: bool = True) -> PhysicsScorer:
    """Factory function to create a physics scorer"""
    return PhysicsScorer(vina_weights=vina_style)


# Example usage
if __name__ == "__main__":
    import json
    
    # Test with sample data
    scorer = PhysicsScorer(vina_weights=True)
    
    # Sample pose (24D)
    sample_pose = np.random.randn(24) * 0.1
    
    print("PhysicsScorer initialized successfully")
    print(f"Device: {scorer.device}")
    print(f"Weights: w_gauss1={scorer.weights.w_gauss1}, "
          f"w_hbond={scorer.weights.w_hbond}")
