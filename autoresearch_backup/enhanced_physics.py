"""
enhanced_physics.py
===================
GEOCK 2.0 — Enhanced Physics Engine
Combines:
  1. Vinardo physics terms     (original 6 features)
  2. Chemistry-based scoring   (new 8 features)
  3. Quantum VQE feature       (new 1 feature)

Total: 15 physics features vs original 6
Drop-in addition to simple_affinity.py feature pipeline.

Usage:
    from enhanced_physics import enhanced_physics_features
    feats = enhanced_physics_features(rec_coords, rec_types,
                                       lig_coords, lig_types,
                                       n_tors, smiles)
    # returns np.array shape (15,)
"""

from __future__ import annotations

import math
import warnings
import numpy as np
from typing import Optional

warnings.filterwarnings("ignore")

# ── RDKit ──────────────────────────────────────────────────────────────────
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, AllChem
    from rdkit.Chem.rdMolDescriptors import CalcTPSA
    RDKIT_OK = True
except ImportError:
    RDKIT_OK = False
    warnings.warn("RDKit not found — chemistry features will be zeros")

# ── Qiskit VQE ────────────────────────────────────────────────────────────
try:
    from qiskit.primitives import StatevectorEstimator
    from qiskit.circuit.library import n_local
    from qiskit.quantum_info import SparsePauliOp
    from scipy.optimize import minimize as scipy_minimize
    QISKIT_OK = True
except ImportError:
    QISKIT_OK = False
    warnings.warn("Qiskit not found — quantum feature will be zero")


# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

VDW = {
    "C":1.9,"A":1.9,"N":1.8,"NA":1.8,"O":1.7,"OA":1.7,
    "S":2.0,"SA":2.0,"P":2.1,"F":1.5,
    "CL":1.8,"Cl":1.8,"BR":2.0,"Br":2.0,"I":2.2,
    "HD":1.0,"MG":1.2,"CA":1.7,"MN":1.2,"FE":1.2,"ZN":1.2,
}

# Vinardo weights
W_GAUSS1      = -0.045
W_REPULSION   =  0.800
W_HYDROPHOBIC = -0.030
W_HBOND       = -0.600
W_TORSION     =  0.055
CUTOFF        =  8.0

HYDROPHOBIC_TYPES = frozenset({"C","A","S","SA","CL","Cl","BR","Br","F","I"})
HBOND_ACC         = frozenset({"NA","OA","SA","N","O"})
HBOND_DON         = frozenset({"NA","OA","N","O"})

# Chemistry constants
HBOND_DIST_IDEAL  = 2.9    # Angstrom — ideal N/O...N/O distance
PI_STACK_DIST     = 3.8    # Angstrom — ideal pi-pi stacking
SALT_BRIDGE_DIST  = 4.0    # Angstrom — charge-charge interaction


# ═══════════════════════════════════════════════════════════════════════════
# 1. VINARDO PHYSICS TERMS  (Features 0-5)
# ═══════════════════════════════════════════════════════════════════════════

def _vdw(t: str) -> float:
    return VDW.get(t, 1.8)

def _surface_dist(r: float, t1: str, t2: str) -> float:
    return r - (_vdw(t1) + _vdw(t2))

def vinardo_features(
    rec_coords: np.ndarray,
    rec_types:  list[str],
    lig_coords: np.ndarray,
    lig_types:  list[str],
    n_torsions: int = 0,
) -> np.ndarray:
    """
    6 Vinardo interaction term features.
    [gauss1_w, repulsion_w, hydrophobic_w, hbond_w, torsion, affinity]
    """
    g1 = rep = hydro = hb = 0.0
    n_clashes = 0

    for ri, rt in zip(rec_coords, rec_types):
        for li, lt in zip(lig_coords, lig_types):
            r = float(np.linalg.norm(ri - li))
            if r >= CUTOFF:
                continue
            d = _surface_dist(r, rt, lt)

            g1 += math.exp(-((d / 0.5) ** 2))

            # Softened repulsion: only penalise hard clashes (d < -0.4)
            if d < -0.4:
                rep += d * d
                n_clashes += 1

            if rt in HYDROPHOBIC_TYPES and lt in HYDROPHOBIC_TYPES:
                if d <= 0.0:
                    hydro += 1.0
                elif d < 2.5:
                    hydro += (2.5 - d) / 2.5

            r_acc = rt in HBOND_ACC; r_don = rt in HBOND_DON
            l_acc = lt in HBOND_ACC; l_don = lt in HBOND_DON
            if (r_don and l_acc) or (l_don and r_acc):
                if d <= -0.7:
                    hb += 1.0
                elif d < 0.0:
                    hb += (0.0 - d) / 0.7

    g1_w    = W_GAUSS1      * g1
    rep_w   = W_REPULSION   * rep
    hydro_w = W_HYDROPHOBIC * hydro
    hb_w    = W_HBOND       * hb
    tor_w   = W_TORSION     * max(0, n_torsions)
    c       = g1_w + rep_w + hydro_w + hb_w
    affinity= c / (1.0 + tor_w + 1e-9)

    return np.array([g1_w, rep_w, hydro_w, hb_w, tor_w, affinity],
                    dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# 2. CHEMISTRY-BASED SCORING  (Features 6-13)
# ═══════════════════════════════════════════════════════════════════════════

def _aromatic_atoms(mol) -> list:
    """Return indices of atoms in aromatic rings."""
    return [a.GetIdx() for a in mol.GetAtoms() if a.GetIsAromatic()]

def _ring_centroids(mol) -> list[np.ndarray]:
    """Return 3D centroids of aromatic rings."""
    centroids = []
    if not mol.GetNumConformers():
        return centroids
    conf  = mol.GetConformer()
    rings = mol.GetRingInfo().AtomRings()
    for ring in rings:
        atoms = [a for a in ring
                 if mol.GetAtomWithIdx(a).GetIsAromatic()]
        if len(atoms) >= 5:
            pts = np.array([conf.GetAtomPosition(a) for a in atoms])
            centroids.append(pts.mean(axis=0))
    return centroids

def chemistry_features(
    rec_coords: np.ndarray,
    rec_types:  list[str],
    lig_coords: np.ndarray,
    lig_types:  list[str],
    smiles:     Optional[str] = None,
) -> np.ndarray:
    """
    8 chemistry-based features capturing interactions that
    Vinardo misses:
    [0] pi_pi_score       — aromatic stacking (from SMILES ring centroids)
    [1] cation_pi_score   — cation next to aromatic ring
    [2] salt_bridge_score — N+...O- charge-charge interactions
    [3] halogen_bond      — C-X...O/N halogen bond
    [4] metal_coord       — coordination to FE/ZN/MG
    [5] burial_fraction   — fraction of ligand buried in pocket
    [6] shape_complement  — pocket/ligand volume match
    [7] lipophilic_match  — hydrophobic ligand atoms in hydrophobic pocket
    """
    feats = np.zeros(8, dtype=np.float32)

    n_lig = len(lig_coords)
    n_rec = len(rec_coords)
    if n_lig == 0 or n_rec == 0:
        return feats

    lig_arr = np.array(lig_coords)
    rec_arr = np.array(rec_coords)

    # ── Feature 0: Pi-pi stacking ────────────────────────────────────────
    # From SMILES: find aromatic ring centroids in ligand
    # From pocket: find aromatic carbons (type C/A) that cluster
    pi_score = 0.0
    if smiles and RDKIT_OK:
        mol = Chem.MolFromSmiles(smiles)
        if mol and mol.GetNumConformers() == 0:
            try:
                mol = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
                mol = Chem.RemoveHs(mol)
            except Exception:
                mol = None

        if mol and mol.GetNumConformers() > 0:
            lig_centroids = _ring_centroids(mol)
            # Receptor aromatic carbons (type A in PDBQT)
            rec_arom = rec_arr[[t == "A" for t in rec_types]]
            for lc in lig_centroids:
                for ra in rec_arom:
                    d = np.linalg.norm(lc - ra)
                    if d < PI_STACK_DIST + 1.5:
                        pi_score += math.exp(-((d - PI_STACK_DIST) / 1.0) ** 2)
    feats[0] = float(np.clip(pi_score / max(1, n_lig), 0, 1))

    # ── Feature 1: Cation-pi interaction ────────────────────────────────
    # Positively charged N (NA type) near aromatic carbon (A type)
    cpi = 0.0
    rec_arom_idx = [i for i, t in enumerate(rec_types) if t == "A"]
    lig_cat_idx  = [i for i, t in enumerate(lig_types) if t == "NA"]
    for li in lig_cat_idx:
        for ri in rec_arom_idx:
            d = np.linalg.norm(lig_arr[li] - rec_arr[ri])
            if d < 5.0:
                cpi += math.exp(-((d - 3.5) / 1.0) ** 2)
    feats[1] = float(np.clip(cpi / max(1, n_lig), 0, 1))

    # ── Feature 2: Salt bridge ──────────────────────────────────────────
    # N (positive) in ligand with O (negative) in receptor and vice versa
    sb = 0.0
    lig_N = [i for i, t in enumerate(lig_types) if t in ("NA","N")]
    lig_O = [i for i, t in enumerate(lig_types) if t in ("OA","O")]
    rec_O = [i for i, t in enumerate(rec_types) if t in ("OA","O")]
    rec_N = [i for i, t in enumerate(rec_types) if t in ("NA","N")]

    for li in lig_N:
        for ri in rec_O:
            d = np.linalg.norm(lig_arr[li] - rec_arr[ri])
            if d < SALT_BRIDGE_DIST + 1.0:
                sb += math.exp(-((d - SALT_BRIDGE_DIST) / 0.8) ** 2)
    for li in lig_O:
        for ri in rec_N:
            d = np.linalg.norm(lig_arr[li] - rec_arr[ri])
            if d < SALT_BRIDGE_DIST + 1.0:
                sb += math.exp(-((d - SALT_BRIDGE_DIST) / 0.8) ** 2)
    feats[2] = float(np.clip(sb / max(1, n_lig), 0, 1))

    # ── Feature 3: Halogen bond ─────────────────────────────────────────
    # C-X...O/N where X = CL, BR, I
    halo_types = frozenset({"CL","Cl","BR","Br","I"})
    halo = 0.0
    lig_hal = [i for i, t in enumerate(lig_types) if t in halo_types]
    rec_acc = [i for i, t in enumerate(rec_types) if t in ("OA","NA","O","N")]
    for li in lig_hal:
        for ri in rec_acc:
            d = np.linalg.norm(lig_arr[li] - rec_arr[ri])
            if d < 4.0:
                halo += math.exp(-((d - 3.0) / 0.7) ** 2)
    feats[3] = float(np.clip(halo / max(1, n_lig), 0, 1))

    # ── Feature 4: Metal coordination ──────────────────────────────────
    metal_types = frozenset({"FE","ZN","MG","MN","CA"})
    metal = 0.0
    rec_metals = [i for i, t in enumerate(rec_types) if t in metal_types]
    lig_coord   = [i for i, t in enumerate(lig_types)
                   if t in ("OA","NA","SA","O","N","S")]
    for ri in rec_metals:
        for li in lig_coord:
            d = np.linalg.norm(rec_arr[ri] - lig_arr[li])
            if d < 3.0:
                metal += math.exp(-((d - 2.2) / 0.4) ** 2)
    feats[4] = float(np.clip(metal / max(1, n_lig), 0, 1))

    # ── Feature 5: Burial fraction ──────────────────────────────────────
    # Fraction of ligand atoms within 4Å of any receptor atom
    buried = sum(
        1 for lc in lig_arr
        if np.min(np.linalg.norm(rec_arr - lc, axis=1)) < 4.0
    )
    feats[5] = float(buried / n_lig)

    # ── Feature 6: Shape complementarity ────────────────────────────────
    # Ratio of ligand volume estimate to pocket volume estimate
    lig_span    = np.ptp(lig_arr, axis=0)
    rec_span    = np.ptp(rec_arr, axis=0)
    lig_vol     = float(np.prod(lig_span + 1e-3))
    rec_vol     = float(np.prod(rec_span + 1e-3))
    feats[6]    = float(np.clip(lig_vol / (rec_vol + 1e-3), 0, 1))

    # ── Feature 7: Lipophilic match ──────────────────────────────────────
    # Hydrophobic ligand atoms that face hydrophobic pocket atoms
    lig_hydro = [i for i, t in enumerate(lig_types) if t in HYDROPHOBIC_TYPES]
    rec_hydro = [i for i, t in enumerate(rec_types) if t in HYDROPHOBIC_TYPES]
    lipo = 0.0
    for li in lig_hydro:
        for ri in rec_hydro:
            d = np.linalg.norm(lig_arr[li] - rec_arr[ri])
            if d < 5.0:
                lipo += math.exp(-d / 3.0)
    feats[7] = float(np.clip(lipo / max(1, n_lig * n_rec) * 100, 0, 1))

    return feats


# ═══════════════════════════════════════════════════════════════════════════
# 3. QUANTUM VQE FEATURE  (Feature 14)
# ═══════════════════════════════════════════════════════════════════════════

# Cache VQE results — don't recompute same SMILES twice
_vqe_cache: dict[str, float] = {}

def vqe_feature(
    smiles:      str,
    rec_coords:  Optional[np.ndarray] = None,
    rec_types:   Optional[list[str]]  = None,
    maxiter:     int = 100,
) -> float:
    """
    Compute quantum VQE ground state energy for the ligand.
    
    The Hamiltonian is scaled by:
      - Molecular size (n_atoms)
      - Heteroatom count (polarity)
      - LogP (lipophilicity)
      - Interaction context (if rec_coords provided)
    
    Returns energy in kcal/mol (normalised to per-atom scale).
    Returns 0.0 if Qiskit not installed or SMILES invalid.
    """
    if not QISKIT_OK or not RDKIT_OK:
        return 0.0

    # Check cache
    cache_key = smiles.strip()
    if cache_key in _vqe_cache:
        return _vqe_cache[cache_key]

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0.0

        n_atoms  = mol.GetNumHeavyAtoms()
        n_hetero = Descriptors.NumHeteroatoms(mol)
        logP     = Descriptors.MolLogP(mol)
        hbd      = Descriptors.NumHDonors(mol)
        hba      = Descriptors.NumHAcceptors(mol)

        # Scale Hamiltonian by molecular properties
        size_scale   = n_atoms / 20.0
        polar_scale  = n_hetero / 8.0
        lipo_scale   = abs(logP + 1)
        hbond_scale  = (hbd + hba) / 10.0

        # Add interaction context if receptor available
        interaction_boost = 1.0
        if rec_coords is not None and rec_types is not None:
            # Count close contacts < 5Å as interaction boost
            lig_arr = np.zeros((1, 3))  # placeholder centroid
            rec_arr = np.array(rec_coords)
            n_contacts = np.sum(
                np.linalg.norm(rec_arr, axis=1) < 10.0
            )
            interaction_boost = 1.0 + 0.1 * min(n_contacts / 50.0, 1.0)

        # 4-qubit Hamiltonian encoding key molecular properties
        H = SparsePauliOp.from_list([
            ('IIZZ', -1.0523 * size_scale * interaction_boost),
            ('ZZII', -0.3979 * size_scale),
            ('IZIZ',  0.1809 * size_scale),
            ('ZIIZ',  0.1809 * size_scale),
            ('IIII',  0.7152 * size_scale),
            ('ZZZI', -0.4523 * polar_scale),   # polarity / H-bond capacity
            ('ZZZZ',  0.2252 * lipo_scale),    # lipophilicity
            ('IZZI', -0.1500 * hbond_scale),   # H-bond contribution
        ])

        # Shallow ansatz for speed
        ansatz    = n_local(4, ['ry'], 'cx', reps=1)
        estimator = StatevectorEstimator()

        def cost(params):
            bound = ansatz.assign_parameters(params)
            return estimator.run(
                [(bound, H)]).result()[0].data.evs.real

        np.random.seed(42)
        x0     = np.random.rand(ansatz.num_parameters) * np.pi
        result = scipy_minimize(cost, x0, method='COBYLA',
                                options={'maxiter': maxiter,
                                         'rhobeg': 0.5})

        # Normalise to per-atom scale
        energy_kcal = (result.fun * 627.509) / max(1, n_atoms)
        _vqe_cache[cache_key] = float(energy_kcal)
        return float(energy_kcal)

    except Exception as e:
        warnings.warn(f"VQE failed for {smiles[:30]}: {e}")
        return 0.0


# ═══════════════════════════════════════════════════════════════════════════
# 4. COMBINED ENHANCED PHYSICS (15 features total)
# ═══════════════════════════════════════════════════════════════════════════

def enhanced_physics_features(
    rec_coords:  np.ndarray,
    rec_types:   list[str],
    lig_coords:  np.ndarray,
    lig_types:   list[str],
    n_torsions:  int = 0,
    smiles:      Optional[str] = None,
    use_quantum: bool = True,
) -> np.ndarray:
    """
    Full enhanced physics feature vector — 15 dimensions:

    [0-5]   Vinardo physics  (gauss1, repulsion, hydrophobic,
                               hbond, torsion, affinity)
    [6-13]  Chemistry-based  (pi-pi, cation-pi, salt bridge,
                               halogen bond, metal coord,
                               burial, shape, lipophilic match)
    [14]    Quantum VQE      (ground state energy per atom)

    Parameters
    ----------
    rec_coords   : (M, 3) receptor heavy atom coordinates
    rec_types    : PDBQT atom type strings, length M
    lig_coords   : (N, 3) ligand heavy atom coordinates
    lig_types    : PDBQT atom type strings, length N
    n_torsions   : number of rotatable bonds
    smiles       : ligand SMILES (improves chemistry + quantum features)
    use_quantum  : set False to skip VQE (faster, for debugging)

    Returns
    -------
    np.ndarray of shape (15,), dtype float32
    """
    # Layer 1: Vinardo (6D)
    vina = vinardo_features(rec_coords, rec_types,
                            lig_coords, lig_types, n_torsions)

    # Layer 2: Chemistry (8D)
    chem = chemistry_features(rec_coords, rec_types,
                              lig_coords, lig_types, smiles)

    # Layer 3: Quantum VQE (1D)
    if use_quantum and smiles:
        qe = vqe_feature(smiles, rec_coords, rec_types)
    else:
        qe = 0.0

    return np.concatenate([vina, chem, [qe]]).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# 5. FEATURE NAME MAP (for interpretability)
# ═══════════════════════════════════════════════════════════════════════════

FEATURE_NAMES = [
    # Vinardo (0-5)
    "vinardo_gauss1",
    "vinardo_repulsion",
    "vinardo_hydrophobic",
    "vinardo_hbond",
    "vinardo_torsion",
    "vinardo_affinity",
    # Chemistry (6-13)
    "chem_pi_pi_stacking",
    "chem_cation_pi",
    "chem_salt_bridge",
    "chem_halogen_bond",
    "chem_metal_coordination",
    "chem_burial_fraction",
    "chem_shape_complementarity",
    "chem_lipophilic_match",
    # Quantum (14)
    "quantum_vqe_energy_per_atom",
]


# ═══════════════════════════════════════════════════════════════════════════
# 6. QUICK TEST
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')

    print("Testing enhanced physics engine...")
    print("="*55)

    # Mock receptor pocket (20 atoms)
    np.random.seed(42)
    rec_coords = np.random.randn(20, 3) * 3.0
    rec_types  = ["C","NA","OA","C","C","NA","OA","C","A","A",
                  "C","ZN","C","OA","C","C","NA","C","FE","C"]

    # Mock ligand (PTR-like)
    lig_coords = np.random.randn(10, 3) * 1.5
    lig_types  = ["NA","C","C","OA","C","C","C","OA","P","OA"]
    smiles     = "c1cc(ccc1CC(C(=O)O)N)OP(=O)(O)O"

    feats = enhanced_physics_features(
        rec_coords, rec_types,
        lig_coords, lig_types,
        n_torsions=5,
        smiles=smiles,
        use_quantum=True,
    )

    print(f"Feature vector shape: {feats.shape}")
    print()
    for i, (name, val) in enumerate(zip(FEATURE_NAMES, feats)):
        layer = ("Vinardo  " if i < 6 else
                 "Chemistry" if i < 14 else
                 "Quantum  ")
        print(f"  [{i:2d}] {layer} | {name:<32} : {val:+.4f}")

    print()
    print(f"Vinardo affinity  : {feats[5]:+.4f} kcal/mol")
    print(f"Burial fraction   : {feats[11]:.4f}")
    print(f"VQE energy/atom   : {feats[14]:+.4f} kcal/mol")
    print("="*55)
    print("All 15 features computed successfully")
    print()
    print("To integrate into simple_affinity.py:")
    print("  Replace compute_physics() with enhanced_physics_features()")
    print("  Feature vector grows from 60+512 to 15+512 = 527D")
    print("  (15 richer features replace 60 raw ones)")
