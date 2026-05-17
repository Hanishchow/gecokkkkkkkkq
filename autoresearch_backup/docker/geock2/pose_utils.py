"""
pose_utils.py - Convert 24D pose vectors to 3D coordinates and vice versa
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict


POSE_DIM = 24
N_RIGID = 6  # tx, ty, tz, rx, ry, rz
MAX_TORSIONS = 18


@dataclass
class LigandTopology:
    """Ligand structure for coordinate conversion"""
    atom_types: List[str]           # Element types: 'C', 'N', 'O', etc.
    atom_masses: List[float]        # Atomic masses
    bonds: List[Tuple[int, int]]  # Bond pairs (atom indices)
    torsion_axes: List[Tuple[int, int]]  # Rotatable bond axes
    torsion_atoms: List[List[int]]  # Atoms affected by each torsion
    heavy_atoms: List[int]         # Indices of non-hydrogen atoms
    centroid: Optional[np.ndarray] = None  # Original centroid for centering
    
    @property
    def n_atoms(self) -> int:
        return len(self.atom_types)
    
    @property
    def n_torsions(self) -> int:
        return len(self.torsion_axes)
    
    @property
    def n_heavy_atoms(self) -> int:
        return len(self.heavy_atoms)


def parse_pdbqt_topology(pdbqt_path: str) -> LigandTopology:
    """
    Parse ligand topology from PDBQT file.
    Extracts atom types, bonds, and rotatable torsions.
    """
    atom_types = []
    atom_masses = []
    bonds = []
    torsion_axes = []
    torsion_atoms = []
    heavy_atoms = []
    
    MASSES = {
        'C': 12.0, 'N': 14.0, 'O': 16.0, 'S': 32.0, 'P': 31.0,
        'H': 1.0, 'F': 19.0, 'Cl': 35.5, 'Br': 80.0, 'I': 127.0,
        'Fe': 55.8, 'Zn': 65.4, 'Ca': 40.1, 'Mg': 24.3, 'Mn': 54.9
    }
    
    if not pdbqt_path.endswith('.pdbqt'):
        raise ValueError(f"Expected PDBQT file, got {pdbqt_path}")
    
    with open(pdbqt_path) as f:
        lines = f.readlines()
    
    # Parse atoms
    for line in lines:
        if line.startswith('ATOM') or line.startswith('HETATM'):
            # PDBQT format: columns 77-78 for type
            atom_type = line[77:79].strip()
            if not atom_type:
                atom_type = line[12:16].strip()[0]
            
            atom_types.append(atom_type)
            atom_masses.append(MASSES.get(atom_type, 12.0))
            
            if atom_type != 'H':
                heavy_atoms.append(len(atom_types) - 1)
    
    # Parse bonds (ROOT/BRANCH sections)
    current_branch = None
    branch_torsions = []
    
    for line in lines:
        if line.startswith('ROOT'):
            current_branch = None
            branch_torsions = []
        elif line.startswith('BRANCH'):
            parts = line.split()
            if len(parts) >= 4:
                # BRANCH index1 index2 - defines rotatable bond
                idx1 = int(parts[1]) - 1  # 1-indexed to 0-indexed
                idx2 = int(parts[2]) - 1
                if idx1 < len(atom_types) and idx2 < len(atom_types):
                    torsion_axes.append((idx1, idx2))
                    bonds.append((idx1, idx2))
                    current_branch = (idx1, idx2)
                    branch_torsions = []
        elif line.startswith('ENDBRANCH'):
            current_branch = None
        elif line.startswith('ATOM') or line.startswith('HETATM'):
            # Add bonds from HETATM lines if not already added
            if current_branch is not None:
                idx = int(line[6:11].strip()) - 1
                if idx < len(atom_types):
                    # Bond to parent in branch
                    parent = current_branch[1]
                    if (parent, idx) not in bonds and (idx, parent) not in bonds:
                        bonds.append((parent, idx))
    
    # Build torsion atom lists (all atoms affected by each torsion)
    for axis in torsion_axes:
        affected = [axis[0], axis[1]]
        # Find all atoms "downstream" from axis[1]
        to_visit = [axis[1]]
        visited = set([axis[0]])
        
        while to_visit:
            curr = to_visit.pop()
            if curr not in visited:
                visited.add(curr)
                affected.append(curr)
                for b in bonds:
                    if b[0] == curr and b[1] not in visited:
                        to_visit.append(b[1])
                    elif b[1] == curr and b[0] not in visited:
                        to_visit.append(b[0])
        
        torsion_atoms.append(affected)
    
    return LigandTopology(
        atom_types=atom_types,
        atom_masses=atom_masses,
        bonds=bonds,
        torsion_axes=torsion_axes,
        torsion_atoms=torsion_atoms,
        heavy_atoms=heavy_atoms
    )


def parse_sdf_topology(sdf_path: str) -> LigandTopology:
    """
    Parse ligand topology from SDF file.
    Simpler than PDBQT - no torsions marked explicitly.
    """
    atom_types = []
    atom_masses = []
    bonds = []
    heavy_atoms = []
    
    MASSES = {
        'C': 12.0, 'N': 14.0, 'O': 16.0, 'S': 32.0, 'P': 31.0,
        'H': 1.0, 'F': 19.0, 'Cl': 35.5, 'Br': 80.0, 'I': 127.0
    }
    
    with open(sdf_path) as f:
        lines = f.readlines()
    
    # V2000 format: counts line has totals
    if len(lines) < 3:
        raise ValueError("Invalid SDF file")
    
    counts = lines[2].split()
    n_atoms = int(counts[0])
    n_bonds = int(counts[1])
    
    # Parse atoms (starting line 4)
    for i in range(n_atoms):
        line = lines[3 + i]
        parts = line.split()
        if len(parts) >= 4:
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            atom_type = parts[3] if len(parts) > 3 else 'C'
            
            atom_types.append(atom_type)
            atom_masses.append(MASSES.get(atom_type, 12.0))
            
            if atom_type != 'H':
                heavy_atoms.append(len(atom_types) - 1)
    
    # Parse bonds (after atoms)
    for i in range(n_bonds):
        line = lines[3 + n_atoms + i]
        parts = line.split()
        if len(parts) >= 3:
            idx1 = int(parts[0]) - 1
            idx2 = int(parts[1]) - 1
            bonds.append((idx1, idx2))
    
    # Estimate rotatable bonds (simplified: single bonds not in rings)
    torsion_axes = []
    torsion_atoms = []
    
    # Simple heuristic: estimate ~n_atoms/8 rotatable
    n_est_tors = max(0, n_atoms // 8)
    for i in range(min(n_est_tors, len(bonds))):
        if i < len(bonds):
            torsion_axes.append(bonds[i])
            torsion_atoms.append(list(range(len(atom_types))))  # All atoms affected
    
    return LigandTopology(
        atom_types=atom_types,
        atom_masses=atom_masses,
        bonds=bonds,
        torsion_axes=torsion_axes[:5],  # Limit to 5 for SDF
        torsion_atoms=torsion_atoms[:5],
        heavy_atoms=heavy_atoms
    )


def euler_to_rotation_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    """
    Convert Euler angles (ZYX convention) to rotation matrix.
    Order: Rz * Ry * Rx
    """
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    
    # Rotation matrix (ZYX order)
    R = np.array([
        [cz*cy, cz*sy*sx - sz*cx, cz*sy*cx + sz*sx],
        [sz*cy, sz*sy*sx + cz*cx, sz*sy*cx - cz*sx],
        [-sy, cy*sx, cy*cx]
    ])
    
    return R


def apply_rotation(coords: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Apply rotation matrix to coordinates"""
    return coords @ R.T


def apply_translation(coords: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Apply translation to coordinates"""
    return coords + t


def rotation_matrix_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """Create rotation matrix from axis and angle"""
    axis = axis / np.linalg.norm(axis)
    c = np.cos(angle)
    s = np.sin(angle)
    t = 1 - c
    
    x, y, z = axis
    return np.array([
        [t*x*x + c, t*x*y - s*z, t*x*z + s*y],
        [t*x*y + s*z, t*y*y + c, t*y*z - s*x],
        [t*x*z - s*y, t*y*z + s*x, t*z*z + c]
    ])


def apply_torsion(
    coords: np.ndarray,
    axis_atoms: Tuple[int, int],
    pivot: np.ndarray,
    axis_direction: np.ndarray,
    angle: float
) -> np.ndarray:
    """
    Apply torsion rotation around an axis to affected atoms.
    
    Args:
        coords: All atom coordinates
        axis_atoms: (atom1, atom2) defining the torsion axis
        pivot: Coordinates of pivot point (atom2)
        axis_direction: Direction vector of axis
        angle: Rotation angle in radians
    """
    result = coords.copy()
    
    # Rotation matrix for this torsion
    R = rotation_matrix_from_axis_angle(axis_direction, angle)
    
    # Translate to origin, rotate, translate back
    for i in range(len(coords)):
        if i not in [axis_atoms[0]]:  # Don't rotate first atom of axis
            translated = coords[i] - pivot
            rotated = R @ translated
            result[i] = rotated + pivot
    
    return result


def get_reference_coordinates(topology: LigandTopology) -> np.ndarray:
    """
    Get reference coordinates from SDF or generate placeholder.
    For SDF files, parse coordinates. For PDBQT, need separate parsing.
    """
    # Placeholder: generate simple coordinates based on atom types
    coords = []
    current_pos = np.array([0.0, 0.0, 0.0])
    direction = np.array([1.0, 0.0, 0.0])
    
    for i, atom_type in enumerate(topology.atom_types):
        coords.append(current_pos.copy())
        
        # Move in a rough chain
        if atom_type == 'C':
            step = 1.5
        elif atom_type in ['N', 'O']:
            step = 1.4
        else:
            step = 1.6
        
        direction = direction + np.random.randn(3) * 0.3
        direction = direction / np.linalg.norm(direction)
        current_pos = current_pos + direction * step
    
    return np.array(coords)


def pose_to_coordinates(
    pose: np.ndarray,
    topology: LigandTopology,
    reference_coords: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Convert 24D pose vector to 3D coordinates.
    
    Args:
        pose: [24] vector [tx, ty, tz, rx, ry, rz, t1, t2, ..., t18]
        topology: LigandTopology with structure info
        reference_coords: Original coordinates (if available)
    
    Returns:
        coordinates: [n_atoms, 3] array of atomic coordinates
    """
    if len(pose) != POSE_DIM:
        raise ValueError(f"Expected pose dim {POSE_DIM}, got {len(pose)}")
    
    # Extract DOF
    tx, ty, tz = pose[0], pose[1], pose[2]
    rx, ry, rz = pose[3], pose[4], pose[5]
    torsions = pose[6:6 + topology.n_torsions]
    
    # Get or generate reference coordinates
    if reference_coords is None:
        reference_coords = get_reference_coordinates(topology)
    elif len(reference_coords) != topology.n_atoms:
        raise ValueError("Reference coords don't match topology")
    
    # Center at origin
    centroid = reference_coords.mean(axis=0)
    coords = reference_coords - centroid
    
    # Apply rotation
    R = euler_to_rotation_matrix(rx, ry, rz)
    coords = apply_rotation(coords, R)
    
    # Apply torsions (simplified - just rotate all affected atoms)
    for i, angle in enumerate(torsions):
        if i < len(topology.torsion_axes):
            axis = topology.torsion_axes[i]
            if axis[0] < len(coords) and axis[1] < len(coords):
                # Get axis direction
                axis_vec = coords[axis[1]] - coords[axis[0]]
                axis_vec = axis_vec / np.linalg.norm(axis_vec)
                
                # Apply rotation around this axis to all affected atoms
                affected = topology.torsion_atoms[i]
                R_tor = rotation_matrix_from_axis_angle(axis_vec, angle)
                
                pivot = coords[axis[1]]
                for j in affected:
                    if j != axis[0]:
                        translated = coords[j] - pivot
                        rotated = R_tor @ translated
                        coords[j] = rotated + pivot
    
    # Apply translation
    coords = apply_translation(coords, np.array([tx, ty, tz]))
    
    return coords


def coordinates_to_pose(
    coords: np.ndarray,
    topology: LigandTopology,
    reference_coords: np.ndarray
) -> np.ndarray:
    """
    Convert 3D coordinates back to 24D pose vector.
    This is an approximation - exact inverse may not exist.
    """
    pose = np.zeros(POSE_DIM)
    
    # Center
    centroid = coords.mean(axis=0)
    centered = coords - centroid
    
    # Reference centroid
    ref_centroid = reference_coords.mean(axis=0)
    ref_centered = reference_coords - ref_centroid
    
    # Translation (simplified)
    pose[0:3] = centroid - ref_centroid
    
    # Rotation (simplified - use SVD for best alignment)
    H = ref_centered.T @ centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # Handle reflection case
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Convert rotation matrix to Euler angles (simplified)
    # This is approximate - proper extraction needs more work
    ry = np.arcsin(-R[2, 0])
    if abs(np.cos(ry)) > 1e-6:
        rx = np.arctan2(R[2, 1], R[2, 2])
        rz = np.arctan2(R[1, 0], R[0, 0])
    else:
        rx = np.arctan2(-R[1, 2], R[1, 1])
        rz = 0
    
    pose[3:6] = [rx, ry, rz]
    
    # Torsions (set to zero as approximation)
    pose[6:6 + topology.n_torsions] = 0.0
    
    return pose


class PoseConverter:
    """Convert between pose vectors and 3D coordinates"""
    
    def __init__(self, topology: LigandTopology, reference_coords: Optional[np.ndarray] = None):
        self.topology = topology
        self.reference_coords = reference_coords
    
    def to_coordinates(self, pose: np.ndarray) -> np.ndarray:
        """Convert pose vector to 3D coordinates"""
        return pose_to_coordinates(pose, self.topology, self.reference_coords)
    
    def to_pose(self, coords: np.ndarray) -> np.ndarray:
        """Convert 3D coordinates to pose vector"""
        if self.reference_coords is None:
            raise ValueError("Need reference_coords for reverse conversion")
        return coordinates_to_pose(coords, self.topology, self.reference_coords)
    
    @classmethod
    def from_pdbqt(cls, pdbqt_path: str) -> 'PoseConverter':
        """Create from PDBQT file"""
        topology = parse_pdbqt_topology(pdbqt_path)
        return cls(topology)
    
    @classmethod
    def from_sdf(cls, sdf_path: str) -> 'PoseConverter':
        """Create from SDF file"""
        topology = parse_sdf_topology(sdf_path)
        return cls(topology)
