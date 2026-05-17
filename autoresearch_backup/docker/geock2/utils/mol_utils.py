"""
utils/mol_utils.py — Molecular File Utilities for GEOCK/DNBAP

Parses ligand files (PDBQT, SDF, MOL2) into pose vectors for pre-training.

Scientific contract:
  A pose vector is [tx, ty, tz, rx, ry, rz, t1..t18] — 24D
  - Translation: absolute xyz centroid of ligand (Å)
  - Rotation   : Euler angles (ZYX convention, radians)
  - Torsions   : rotatable bond dihedral angles (radians), padded to 18

  This representation is canonical for AutoDock-family docking.
  It matches geock_mc exactly (POSE_DIM=24, MAX_TORSIONS=18).

Why we parse ourselves (not RDKit):
  - Zero hard dependencies (RDKit install is 300MB, fails on many HPC nodes)
  - We only need: atom coords, connectivity, rotatable bonds
  - Full RDKit parity on torsion extraction is overkill for training data

What we parse:
  PDBQT: AutoDock format — has explicit torsion BRANCH records (easiest)
  SDF  : MDL Molfile — we derive torsions from bond table
  MOL2 : TRIPOS format — atom + bond table

Output is always a [24] numpy array ready for PoseVAE / SOM training.
"""

import numpy as np
import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Dict


# ──────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────

POSE_DIM     = 24
MAX_TORSIONS = 18
N_RIGID      = 6      # tx, ty, tz, rx, ry, rz

# AutoDock atom-type hydrophobicity for feature augmentation
ADTYPE_HYDROPHOBICITY = {
    "C": 0.8, "A": 0.8,
    "N": 0.2, "NA": 0.2,
    "O": 0.1, "OA": 0.1,
    "S": 0.6, "SA": 0.6,
    "F": 0.4,
    "P": 0.3,
    "Cl": 0.7, "CL": 0.7,
    "Br": 0.7, "BR": 0.7,
    "I": 0.8,
    "H": 0.0,
    "HD": 0.0,
    "Fe": 0.0, "Zn": 0.0, "Mg": 0.0, "Ca": 0.0,
}

ROTATABLE_BOND_PAIRS = {(6,6), (6,7), (6,8), (6,16), (7,8), (7,16), (8,15)}


# ──────────────────────────────────────────────────────────────────
# Data containers
# ──────────────────────────────────────────────────────────────────

@dataclass
class LigandAtom:
    idx:       int
    x:         float
    y:         float
    z:         float
    atomic_num: int
    atom_type: str


@dataclass
class LigandBond:
    i: int
    j: int
    order: int


@dataclass
class ParsedLigand:
    atoms:    List[LigandAtom]
    bonds:    List[LigandBond]
    torsions: List[Tuple[int,int,int,int]]
    name:     str = ""


# ──────────────────────────────────────────────────────────────────
# PDBQT parser
# ──────────────────────────────────────────────────────────────────

def parse_pdbqt(path: str) -> ParsedLigand:
    """Parse AutoDock PDBQT ligand file."""
    path = Path(path)
    atoms   = []
    bonds   = []
    torsion_atom_pairs = []

    ELEM_TO_NUM = {
        "H":1,"C":6,"N":7,"O":8,"F":9,"P":15,"S":16,"CL":17,"BR":35,"I":53
    }

    serial_to_idx: Dict[int, int] = {}

    with open(path) as f:
        for line in f:
            rec = line[:6].strip()

            if rec in ("ATOM", "HETATM"):
                try:
                    serial = int(line[6:11].strip())
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                except (ValueError, IndexError):
                    continue

                ad_type = line[77:].strip() if len(line) > 77 else ""
                if not ad_type:
                    ad_type = line[12:16].strip()

                elem = ad_type.upper()[:2].strip()
                elem = ''.join(c for c in elem if c.isalpha())
                anum = ELEM_TO_NUM.get(elem, 6)

                idx = len(atoms)
                serial_to_idx[serial] = idx
                atoms.append(LigandAtom(
                    idx=idx, x=x, y=y, z=z,
                    atomic_num=anum, atom_type=ad_type
                ))

            elif rec == "BRANCH":
                try:
                    parts = line.split()
                    i = int(parts[1])
                    j = int(parts[2])
                    torsion_atom_pairs.append((i, j))
                except (ValueError, IndexError):
                    pass

    torsions = []
    for (si, sj) in torsion_atom_pairs:
        ai = serial_to_idx.get(si, -1)
        aj = serial_to_idx.get(sj, -1)
        if ai >= 0 and aj >= 0:
            torsions.append((ai, aj, aj, ai))

    return ParsedLigand(
        atoms    = atoms,
        bonds    = bonds,
        torsions = torsions,
        name     = path.stem,
    )


# ──────────────────────────────────────────────────────────────────
# SDF / MOL parser
# ──────────────────────────────────────────────────────────────────

def parse_sdf(path: str, mol_idx: int = 0) -> ParsedLigand:
    """Parse MDL SDF / MOL file."""
    path = Path(path)
    ELEM_TO_NUM = {
        "H":1,"C":6,"N":7,"O":8,"F":9,"P":15,"S":16,"CL":17,"BR":35,"I":53
    }

    with open(path) as f:
        content = f.read()

    blocks = content.split("$$$$")
    if mol_idx >= len(blocks):
        raise ValueError(f"mol_idx={mol_idx} but file has {len(blocks)} molecules")

    block = blocks[mol_idx].strip().splitlines()
    if len(block) < 4:
        raise ValueError(f"SDF block too short: {len(block)} lines")

    name = block[0].strip() if block else ""

    counts_line = block[3]
    try:
        n_atoms = int(counts_line[0:3])
        n_bonds = int(counts_line[3:6])
    except (ValueError, IndexError):
        raise ValueError(f"Cannot parse counts line: {repr(counts_line)}")

    atoms = []
    for i in range(n_atoms):
        line = block[4 + i]
        try:
            x      = float(line[0:10])
            y      = float(line[10:20])
            z      = float(line[20:30])
            elem   = line[31:34].strip().upper()
            anum   = ELEM_TO_NUM.get(elem, 6)
        except (ValueError, IndexError):
            x = y = z = 0.0; anum = 6; elem = "C"

        atoms.append(LigandAtom(
            idx=i, x=x, y=y, z=z,
            atomic_num=anum, atom_type=elem,
        ))

    bonds = []
    for b in range(n_bonds):
        line = block[4 + n_atoms + b]
        try:
            ai    = int(line[0:3]) - 1
            aj    = int(line[3:6]) - 1
            btype = int(line[6:9])
        except (ValueError, IndexError):
            continue
        if 0 <= ai < n_atoms and 0 <= aj < n_atoms:
            bonds.append(LigandBond(i=ai, j=aj, order=btype))

    torsions = _derive_torsions(atoms, bonds)

    return ParsedLigand(atoms=atoms, bonds=bonds, torsions=torsions, name=name)


# ──────────────────────────────────────────────────────────────────
# MOL2 parser
# ──────────────────────────────────────────────────────────────────

def parse_mol2(path: str) -> ParsedLigand:
    """Parse TRIPOS MOL2 file."""
    path = Path(path)
    ELEM_TO_NUM = {
        "H":1,"C":6,"N":7,"O":8,"F":9,"P":15,"S":16,"CL":17,"BR":35,"I":53
    }

    atoms  = []
    bonds  = []
    name   = ""

    section = None
    atom_id_map: Dict[int,int] = {}

    with open(path) as f:
        for line in f:
            line_s = line.strip()
            if line_s.startswith("@<TRIPOS>"):
                section = line_s.split(">")[1]
                continue

            if section == "MOLECULE" and not name:
                name = line_s

            elif section == "ATOM":
                parts = line_s.split()
                if len(parts) < 6:
                    continue
                try:
                    atom_id = int(parts[0])
                    x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                    atype = parts[5].split(".")[0].upper()
                    anum  = ELEM_TO_NUM.get(atype, 6)
                except (ValueError, IndexError):
                    continue

                idx = len(atoms)
                atom_id_map[atom_id] = idx
                atoms.append(LigandAtom(
                    idx=idx, x=x, y=y, z=z,
                    atomic_num=anum, atom_type=atype,
                ))

            elif section == "BOND":
                parts = line_s.split()
                if len(parts) < 4:
                    continue
                try:
                    ai = atom_id_map.get(int(parts[1]), -1)
                    aj = atom_id_map.get(int(parts[2]), -1)
                    btype_str = parts[3]
                    btype = {"1":1,"2":2,"3":3,"ar":4,"am":1}.get(btype_str, 1)
                except (ValueError, IndexError):
                    continue
                if ai >= 0 and aj >= 0:
                    bonds.append(LigandBond(i=ai, j=aj, order=btype))

    torsions = _derive_torsions(atoms, bonds)
    return ParsedLigand(atoms=atoms, bonds=bonds, torsions=torsions, name=name)


# ──────────────────────────────────────────────────────────────────
# Torsion derivation from bond graph
# ──────────────────────────────────────────────────────────────────

def _derive_torsions(
    atoms: List[LigandAtom],
    bonds: List[LigandBond],
) -> List[Tuple[int,int,int,int]]:
    """Identify rotatable bonds and return (a,b,c,d) dihedral atom tuples."""
    if not bonds:
        return []

    n = len(atoms)
    adj: List[List[int]] = [[] for _ in range(n)]
    for bond in bonds:
        if bond.i < n and bond.j < n:
            adj[bond.i].append(bond.j)
            adj[bond.j].append(bond.i)

    single_bonds = [b for b in bonds if b.order == 1]
    ring_atoms   = _find_ring_atoms(n, adj)

    torsions = []
    seen_pairs = set()

    for bond in single_bonds:
        b, c = bond.i, bond.j
        if b >= n or c >= n:
            continue

        if len(adj[b]) < 2 or len(adj[c]) < 2:
            continue

        if b in ring_atoms and c in ring_atoms:
            continue

        if atoms[b].atomic_num <= 1 or atoms[c].atomic_num <= 1:
            continue

        pair = (min(b,c), max(b,c))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)

        a_candidates = [x for x in adj[b] if x != c and atoms[x].atomic_num > 1]
        d_candidates = [x for x in adj[c] if x != b and atoms[x].atomic_num > 1]

        if not a_candidates or not d_candidates:
            a_candidates = [x for x in adj[b] if x != c] or [c]
            d_candidates = [x for x in adj[c] if x != b] or [b]

        a = a_candidates[0]
        d = d_candidates[0]
        torsions.append((a, b, c, d))

        if len(torsions) >= MAX_TORSIONS:
            break

    return torsions


def _find_ring_atoms(n: int, adj: List[List[int]]) -> set:
    """Find all atoms that are part of a ring using DFS back-edge detection."""
    ring_atoms = set()
    visited    = [False] * n
    parent     = [-1]    * n
    in_stack   = [False] * n

    def dfs(v):
        visited[v] = True
        in_stack[v] = True
        for u in adj[v]:
            if not visited[u]:
                parent[u] = v
                dfs(u)
            elif in_stack[u] and parent[v] != u:
                cur = v
                while cur != u:
                    ring_atoms.add(cur)
                    cur = parent[cur]
                ring_atoms.add(u)
        in_stack[v] = False

    import sys
    sys.setrecursionlimit(max(1000, n * 2))
    for i in range(n):
        if not visited[i]:
            dfs(i)

    return ring_atoms


# ──────────────────────────────────────────────────────────────────
# Pose vector extraction
# ──────────────────────────────────────────────────────────────────

def ligand_to_pose_vector(ligand: ParsedLigand) -> np.ndarray:
    """Convert a ParsedLigand to a 24D pose vector."""
    coords = np.array([[a.x, a.y, a.z] for a in ligand.atoms
                       if a.atomic_num > 1], dtype=np.float32)

    if len(coords) == 0:
        return np.zeros(POSE_DIM, dtype=np.float32)

    centroid = coords.mean(axis=0)
    tx, ty, tz = centroid

    rx, ry, rz = _coords_to_euler(coords - centroid)

    torsion_angles = []
    for (a, b, c, d) in ligand.torsions[:MAX_TORSIONS]:
        if max(a, b, c, d) < len(ligand.atoms):
            pa = np.array([ligand.atoms[a].x, ligand.atoms[a].y, ligand.atoms[a].z])
            pb = np.array([ligand.atoms[b].x, ligand.atoms[b].y, ligand.atoms[b].z])
            pc = np.array([ligand.atoms[c].x, ligand.atoms[c].y, ligand.atoms[c].z])
            pd = np.array([ligand.atoms[d].x, ligand.atoms[d].y, ligand.atoms[d].z])
            angle = _dihedral_angle(pa, pb, pc, pd)
            torsion_angles.append(angle)

    torsion_angles = torsion_angles[:MAX_TORSIONS]
    n_tor = len(torsion_angles)
    padded = torsion_angles + [0.0] * (MAX_TORSIONS - n_tor)

    pose = np.array([tx, ty, tz, rx, ry, rz] + padded, dtype=np.float32)
    assert pose.shape == (POSE_DIM,), f"pose shape: {pose.shape}"
    return pose


def _coords_to_euler(centered_coords: np.ndarray) -> Tuple[float, float, float]:
    """Compute Euler angles (ZYX, radians) from PCA of centered atom coordinates."""
    if len(centered_coords) < 2:
        return 0.0, 0.0, 0.0

    cov = centered_coords.T @ centered_coords / max(len(centered_coords)-1, 1)

    try:
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]

        v1 = eigenvectors[:, 0]
        v2 = eigenvectors[:, 1]
        v3 = np.cross(v1, v2)

        R = np.stack([v1, v2, v3], axis=1)

        sy = math.sqrt(R[0,0]**2 + R[1,0]**2)
        singular = sy < 1e-6

        if not singular:
            rx = math.atan2(R[2,1], R[2,2])
            ry = math.atan2(-R[2,0], sy)
            rz = math.atan2(R[1,0], R[0,0])
        else:
            rx = math.atan2(-R[1,2], R[1,1])
            ry = math.atan2(-R[2,0], sy)
            rz = 0.0

        return float(rx), float(ry), float(rz)

    except np.linalg.LinAlgError:
        return 0.0, 0.0, 0.0


def _dihedral_angle(
    a: np.ndarray, b: np.ndarray,
    c: np.ndarray, d: np.ndarray
) -> float:
    """Compute dihedral angle (radians) for atoms a-b-c-d."""
    b1 = b - a
    b2 = c - b
    b3 = d - c

    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)

    n1_norm = np.linalg.norm(n1)
    n2_norm = np.linalg.norm(n2)

    if n1_norm < 1e-8 or n2_norm < 1e-8:
        return 0.0

    n1 /= n1_norm
    n2 /= n2_norm

    cos_angle = np.clip(np.dot(n1, n2), -1.0, 1.0)
    angle     = math.acos(cos_angle)

    if np.dot(n1, np.cross(b2 / (np.linalg.norm(b2)+1e-9), n2)) < 0:
        angle = -angle

    return float(angle)


# ──────────────────────────────────────────────────────────────────
# High-level API
# ──────────────────────────────────────────────────────────────────

def load_ligand(path: str, mol_idx: int = 0) -> ParsedLigand:
    """Auto-detect format from extension and parse ligand."""
    p = Path(path)
    ext = p.suffix.lower()

    if ext == ".pdbqt":
        return parse_pdbqt(path)
    elif ext in (".sdf", ".mol"):
        return parse_sdf(path, mol_idx)
    elif ext == ".mol2":
        return parse_mol2(path)
    else:
        try:
            return parse_sdf(path, mol_idx)
        except Exception:
            return parse_pdbqt(path)


def ligand_file_to_pose(path: str, mol_idx: int = 0) -> np.ndarray:
    """One-shot: parse a ligand file and return a 24D pose vector."""
    ligand = load_ligand(path, mol_idx)
    return ligand_to_pose_vector(ligand)


def parse_smiles_to_pose_vector(smiles: str, n_poses: int = 100) -> List[np.ndarray]:
    """Parse SMILES to pose vectors (placeholder - needs RDKit)."""
    return [np.random.randn(24) for _ in range(n_poses)]


def extract_pose_vector(pose) -> np.ndarray:
    """Extract 24D pose vector (placeholder)."""
    return np.random.randn(24)


def parse_pdb_atoms(pdb_path: str) -> Tuple[np.ndarray, List[str]]:
    """Parse PDB file to get atom coordinates and element types."""
    coords = []
    elements = []
    try:
        with open(pdb_path, 'r') as f:
            for line in f:
                if line.startswith('ATOM') or line.startswith('HETATM'):
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    elem = line[76:78].strip()
                    coords.append([x, y, z])
                    elements.append(elem)
    except FileNotFoundError:
        pass
    return np.array(coords), elements
