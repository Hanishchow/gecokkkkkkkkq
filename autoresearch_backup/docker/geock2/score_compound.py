"""
score_compound.py — Single Compound Binding Affinity Scorer

Improved version with:
- Better ligand/pocket separation (no fake clashes)
- Improved pKd calibration
- Better pocket filtering
"""

from __future__ import annotations

import os
import math
import warnings
import argparse
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.rdMolDescriptors import CalcTPSA

try:
    import MDAnalysis as mda
    import prolif as plf
    PROLIF_AVAILABLE = True
except ImportError:
    PROLIF_AVAILABLE = False

from patch_parse import parse_pocket_and_ligand

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


R_GAS = 1.987e-3
TEMP  = 298.15

W_GAUSS1      = -0.045
W_REPULSION   =  0.800
W_HYDROPHOBIC = -0.030
W_HBOND       = -0.600
W_TORSION     =  0.055

VDW = {
    "C":1.9,"A":1.9,"N":1.8,"NA":1.8,"O":1.7,"OA":1.7,
    "S":2.0,"SA":2.0,"P":2.1,"F":1.5,
    "CL":1.8,"Cl":1.8,"BR":2.0,"Br":2.0,"I":2.2,
    "HD":1.0,"MG":1.2,"CA":1.7,"MN":1.2,"FE":1.2,"ZN":1.2,
}

HYDROPHOBIC_TYPES = frozenset({"C","A","S","SA","CL","Cl","BR","Br","F","I"})
HBOND_ACC         = frozenset({"NA","OA","SA","N","O"})
HBOND_DON         = frozenset({"NA","OA","N","O"})
CUTOFF            = 8.0


@dataclass
class ScoringResult:
    pkd:        float = 0.0
    dG:         float = 0.0
    kd_nM:      float = 0.0
    confidence: str   = "low"
    gauss1_score:      float = 0.0
    repulsion_score:   float = 0.0
    hydrophobic_score: float = 0.0
    hbond_score:       float = 0.0
    torsion_penalty:   float = 0.0
    raw_vina:          float = 0.0
    n_hbonds:       int   = 0
    n_hydrophobic:  int   = 0
    n_clashes:      int   = 0
    n_contacts:     int   = 0
    mw:         float = 0.0
    logP:       float = 0.0
    tpsa:       float = 0.0
    hbd:        int   = 0
    hba:        int   = 0
    n_rot:      int   = 0
    drug_like:  bool  = False
    prolif_hbond_donor:    float = 0.0
    prolif_hbond_acceptor: float = 0.0
    prolif_hydrophobic:    float = 0.0
    prolif_pi_stack:       float = 0.0
    pocket_size:    int   = 0
    pocket_volume: float = 0.0
    ligand_efficiency: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        bar_hydro = "█" * min(10, int(abs(self.hydrophobic_score) * 5))
        bar_hbond = "█" * min(10, int(abs(self.hbond_score) * 3))

        lines = [
            "",
            "╔══════════════════════════════════════════╗",
            "║       BINDING AFFINITY PREDICTION        ║",
            "╚══════════════════════════════════════════╝",
            "",
            f"  pKd estimate    :  {self.pkd:.2f}",
            f"  ΔG estimate     :  {self.dG:.2f} kcal/mol",
            f"  Kd estimate     :  {self._fmt_kd(self.kd_nM)}",
            f"  Confidence      :  {self.confidence.upper()}",
            "",
            "── Physics Breakdown ─────────────────────",
            f"  Hydrophobic     :  {self.hydrophobic_score:+.3f}  {bar_hydro}",
            f"  H-bond          :  {self.hbond_score:+.3f}  {bar_hbond}",
            f"  Steric attract :  {self.gauss1_score:+.3f}",
            f"  Repulsion       :  {self.repulsion_score:+.3f}  {'⚠️' if self.repulsion_score > 1.0 else ''}",
            f"  Torsion penalty :  {self.torsion_penalty:+.3f}",
            f"  Raw Vina score  :  {self.raw_vina:+.3f}",
            "",
            "── Interaction Counts ────────────────────",
            f"  H-bonds         :  {self.n_hbonds}",
            f"  Hydrophobic     :  {self.n_hydrophobic}",
            f"  Atom contacts   :  {self.n_contacts}",
            f"  Clashes (d<0)  :  {self.n_clashes}  {'⚠️  check pose' if self.n_clashes > 5 else ''}",
        ]

        if PROLIF_AVAILABLE:
            lines += [
                "",
                "── ProLIF Interactions ───────────────────",
                f"  HB Donor        :  {self.prolif_hbond_donor:.2f}  contacts/atom",
                f"  HB Acceptor     :  {self.prolif_hbond_acceptor:.2f}  contacts/atom",
                f"  Hydrophobic     :  {self.prolif_hydrophobic:.2f}  contacts/atom",
                f"  π-Stacking     :  {self.prolif_pi_stack:.2f}  contacts/atom",
            ]

        lines += [
            "",
            "── Ligand Properties ─────────────────────",
            f"  MW               :  {self.mw:.1f} Da",
            f"  LogP             :  {self.logP:.2f}",
            f"  TPSA             :  {self.tpsa:.1f} Å²",
            f"  HBD / HBA       :  {self.hbd} / {self.hba}",
            f"  Rotatable bonds  :  {self.n_rot}",
            f"  Ligand efficiency: {self.ligand_efficiency:.2f} kcal/mol/atom",
            f"  Drug-like        :  {'✅ Yes' if self.drug_like else '❌ No'}",
            "",
            "── Pocket Info ───────────────────────────",
            f"  Pocket atoms    :  {self.pocket_size}",
            f"  Pocket volume   :  ~{self.pocket_volume:.0f} Å³",
        ]

        if self.warnings:
            lines += ["", "── Warnings ─────────────────────────────"]
            for w in self.warnings:
                lines.append(f"  • {w}")

        lines += ["", "──────────────────────────────────────────", ""]
        return "\n".join(lines)

    @staticmethod
    def _fmt_kd(kd_nM: float) -> str:
        if kd_nM < 1:
            return f"{kd_nM*1000:.1f} pM"
        if kd_nM < 1000:
            return f"{kd_nM:.1f} nM"
        if kd_nM < 1e6:
            return f"{kd_nM/1000:.1f} µM"
        return f"{kd_nM/1e6:.1f} mM"


def _vdw(t: str) -> float:
    return VDW.get(t, 1.8)


def _elem_to_vina_type(element: str, atom_name: str) -> str:
    el = element.strip().upper()
    name = atom_name.strip()
    mapping = {
        "C":"C","N":"N","O":"O","S":"S","P":"P","F":"F",
        "CL":"CL","BR":"BR","I":"I","MG":"MG","CA":"CA","MN":"MN","FE":"FE","ZN":"ZN","CO":"ZN",
    }
    t = mapping.get(el, "C")
    if t == "N": t = "NA"
    if t == "O": t = "OA"
    if t == "S": t = "SA"
    return t


def _parse_sdf_coords_manual(sdf_path: str) -> tuple:
    """Manually parse SDF file to get coordinates and element types.
    Handles malformed SDFs that are missing the counts line.
    Returns (coords, types, n_atoms).
    """
    coords, types = [], []
    with open(sdf_path) as f:
        content = f.read()
    
    lines = content.split('\n')
    if len(lines) < 4:
        return np.zeros((0, 3)), [], 0
    
    atom_start = 2
    for i in range(2, min(5, len(lines))):
        parts = lines[i].split()
        if len(parts) >= 4:
            try:
                float(parts[0]), float(parts[1]), float(parts[2])
                atom_start = i
                break
            except ValueError:
                continue
    
    atom_lines = lines[atom_start:]
    n_atoms = 0
    for line in atom_lines:
        if line.startswith('M END') or line.startswith('V2000'):
            break
        if len(line) < 34:
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            x = float(line[0:10].strip())
            y = float(line[10:20].strip())
            z = float(line[20:30].strip())
            elem = line[31:34].strip().upper()
            if elem == 'H':
                continue
            coords.append([x, y, z])
            if elem == 'N': types.append('N')
            elif elem == 'O': types.append('O')
            elif elem == 'S': types.append('S')
            elif elem in ('CL', 'BR', 'F', 'I', 'P'): types.append(elem)
            else: types.append('C')
            n_atoms += 1
        except (ValueError, IndexError):
            continue
    
    return np.array(coords, dtype=np.float32), types, n_atoms


def _get_ligand_coords_set(sdf_path: str) -> set:
    """Return set of rounded (x,y,z) tuples from ligand SDF.
    Used to exclude ligand atoms from pocket parsing.
    """
    coords, _, _ = _parse_sdf_coords_manual(sdf_path)
    coords_set = set()
    for pos in coords:
        key = (round(float(pos[0]), 1), round(float(pos[1]), 1), round(float(pos[2]), 1))
        coords_set.add(key)
    return coords_set


def parse_pocket(pdb_path: str, ligand_sdf: Optional[str] = None, cutoff: float = 10.0):
    """Parse pocket PDB, excluding ligand atoms to prevent fake clashes.
    Returns (coords, types, lig_coords).
    """
    centroid = None
    lig_coords_set = set()
    lig_coords = np.zeros((0, 3), dtype=np.float32)
    ligand_resnames = set()

    WATER_RESIDUES = {"HOH", "WAT", "H2O", "DOD"}

    if ligand_sdf and os.path.exists(ligand_sdf):
        l_coords, _, n_lig = _parse_sdf_coords_manual(ligand_sdf)
        if n_lig > 0:
            centroid = l_coords.mean(axis=0)
            lig_coords = l_coords
        lig_coords_set = _get_ligand_coords_set(ligand_sdf)

    coords, types = [], []

    with open(pdb_path) as f:
        for line in f:
            record = line[:6].strip()
            if record not in ("ATOM", "HETATM"):
                continue
            try:
                resname = line[17:20].strip().upper()
                
                if resname in WATER_RESIDUES:
                    continue
                
                xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])], dtype=np.float32)
                
                if record == "HETATM":
                    ligand_resnames.add(resname)
                    if centroid is not None:
                        if np.linalg.norm(xyz - centroid) > cutoff:
                            continue
                    key = (round(float(xyz[0]), 1), round(float(xyz[1]), 1), round(float(xyz[2]), 1))
                    if key in lig_coords_set:
                        continue
                    continue

                key = (round(float(xyz[0]), 1), round(float(xyz[1]), 1), round(float(xyz[2]), 1))
                if key in lig_coords_set:
                    continue

                if centroid is not None:
                    if np.linalg.norm(xyz - centroid) > cutoff:
                        continue

                el = line[76:78].strip() if len(line) > 76 else ""
                name = line[12:16].strip()

                if not el:
                    el = ''.join(c for c in name if c.isalpha())[:2]
                    el = el[0] if el else "C"

                if el.upper() in ("H", "D"):
                    continue

                coords.append(xyz)
                types.append(_elem_to_vina_type(el, name))
            except:
                continue

    if not coords:
        return np.zeros((0, 3), dtype=np.float32), [], np.zeros((0, 3))

    return np.array(coords, dtype=np.float32), types, lig_coords


def parse_ligand(sdf_path: str):
    """Parse ligand SDF using manual parser to handle malformed SDFs."""
    coords, types, n_heavy = _parse_sdf_coords_manual(sdf_path)
    
    if n_heavy == 0:
        return np.zeros((0, 3), dtype=np.float32), [], 0, None
    
    mol = None
    try:
        suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
        mol = next((m for m in suppl if m is not None), None)
    except Exception:
        pass
    
    n_rot = 0
    if mol is not None:
        try:
            n_rot = Descriptors.NumRotatableBonds(mol)
        except Exception:
            pass
    
    vina_types = []
    for t in types:
        if t == "N": vina_types.append("NA")
        elif t == "O": vina_types.append("OA")
        elif t == "S": vina_types.append("SA")
        else: vina_types.append(t if t in VDW else "C")
    
    return coords, vina_types, n_rot, mol


def ligand_from_smiles(smiles: str):
    """Generate 3D conformer from SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    mol = Chem.AddHs(mol)
    result = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    if result != 0:
        AllChem.EmbedMolecule(mol, AllChem.ETDG())
    AllChem.MMFFOptimizeMolecule(mol)
    mol = Chem.RemoveHs(mol)

    coords = mol.GetConformer().GetPositions().astype(np.float32)
    types = []
    for atom in mol.GetAtoms():
        el = atom.GetSymbol().upper()
        if el == "N": t = "NA"
        elif el == "O": t = "OA"
        elif el == "S": t = "SA"
        else: t = el if el in VDW else "C"
        types.append(t)

    n_rot = Descriptors.NumRotatableBonds(mol)
    return coords, types, n_rot, mol


def score_physics(rec_coords, rec_types, lig_coords, lig_types, n_torsions):
    """Full Vinardo pairwise scoring."""
    g1 = rep = hydro = hb = 0.0
    n_hbonds = n_hydrophobic = n_clashes = n_contacts = 0

    for ri, rt in zip(rec_coords, rec_types):
        for li, lt in zip(lig_coords, lig_types):
            r = float(np.linalg.norm(ri - li))
            if r >= CUTOFF:
                continue

            d = r - (_vdw(rt) + _vdw(lt))
            n_contacts += 1

            g1 += math.exp(-((d / 0.5) ** 2))

            if d < -0.4:
                rep += d * d
                n_clashes += 1
            elif d < 0:
                n_clashes += 1

            if rt in HYDROPHOBIC_TYPES and lt in HYDROPHOBIC_TYPES:
                if d <= 0.0:
                    hydro += 1.0
                    n_hydrophobic += 1
                elif d < 2.5:
                    hydro += (2.5 - d) / 2.5
                    n_hydrophobic += 1

            r_acc = rt in HBOND_ACC; r_don = rt in HBOND_DON
            l_acc = lt in HBOND_ACC; l_don = lt in HBOND_DON
            if (r_don and l_acc) or (l_don and r_acc):
                if d <= -0.7:
                    hb += 1.0
                    n_hbonds += 1
                elif d < 0.0:
                    hb += (0.0 - d) / 0.7
                    n_hbonds += 1

    g1_w = W_GAUSS1 * g1
    rep_w = W_REPULSION * rep
    hydro_w = W_HYDROPHOBIC * hydro
    hb_w = W_HBOND * hb
    tor_w = W_TORSION * max(0, n_torsions)

    c = g1_w + rep_w + hydro_w + hb_w
    affinity = c / (1.0 + tor_w + 1e-9)

    return {
        "gauss1": g1_w, "repulsion": rep_w, "hydrophobic": hydro_w,
        "hbond": hb_w, "torsion": tor_w, "raw_vina": c, "affinity": affinity,
        "n_hbonds": n_hbonds, "n_hydrophobic": n_hydrophobic,
        "n_clashes": n_clashes, "n_contacts": n_contacts,
    }


def score_ligand_props(mol):
    """Compute physicochemical properties."""
    if mol is None:
        return {}

    mw = Descriptors.MolWt(mol)
    logP = Descriptors.MolLogP(mol)
    tpsa = CalcTPSA(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    nrot = Descriptors.NumRotatableBonds(mol)
    drug_like = (mw <= 500 and logP <= 5 and hbd <= 5 and hba <= 10)

    return {"mw": mw, "logP": logP, "tpsa": tpsa, "hbd": hbd, "hba": hba,
            "n_rot": nrot, "drug_like": drug_like, "n_heavy": mol.GetNumAtoms()}


def score_prolif(protein_pdb: str, ligand_sdf: str, n_lig_atoms: int) -> dict:
    """Run ProLIF."""
    zeros = {"hbd": 0.0, "hba": 0.0, "hydro": 0.0, "pi": 0.0}

    if not PROLIF_AVAILABLE or not os.path.exists(ligand_sdf):
        return zeros

    try:
        prot_u = mda.Universe(protein_pdb)
        lig_u = mda.Universe(ligand_sdf)
        prot_mol = plf.Molecule.from_mda(prot_u.select_atoms("protein"))
        lig_mol = plf.Molecule.from_mda(lig_u.atoms)

        norm = max(1, n_lig_atoms)

        fp = plf.Fingerprint(interactions=["HBDonor", "HBAcceptor", "Hydrophobic", "PiStacking"])
        fp.run_from_iterable([lig_mol], prot_mol)
        df = fp.to_dataframe()

        if df.empty:
            return zeros

        def _count(itype):
            cols = [c for c in df.columns if itype in str(c)]
            return float(df[cols].values.sum()) / norm if cols else 0.0

        return {"hbd": _count("HBDonor"), "hba": _count("HBAcceptor"),
                "hydro": _count("Hydrophobic"), "pi": _count("PiStacking")}
    except:
        return zeros


def vina_to_pkd(vina_affinity: float, n_heavy_atoms: int = 20) -> float:
    """Convert Vinardo score to calibrated pKd.
    
    Calibrated on 46 clean compounds from PDBbind:
    dG = -0.0168 * raw_vina - 8.6252
    pKd = -dG / 1.364
    """
    if vina_affinity >= 0:
        return 1.0
    
    dG = -0.0168 * vina_affinity - 8.6252
    pkd = -dG / 1.364
    return float(np.clip(pkd, 1.0, 14.0))


def compute_confidence(result: dict, pocket_size: int, n_clashes: int, n_contacts: int, has_smiles: bool) -> str:
    """Estimate confidence."""
    score = 3

    if pocket_size < 10: score -= 2
    elif pocket_size < 30: score -= 1
    elif pocket_size > 500: score -= 1

    if n_clashes > 10: score -= 2
    elif n_clashes > 3: score -= 1

    if n_contacts > 20: score += 1
    if has_smiles: score += 1

    if score >= 4: return "high"
    if score >= 2: return "medium"
    return "low"


def pocket_volume_estimate(coords: np.ndarray) -> float:
    """Rough pocket volume via bounding box × 0.7."""
    if len(coords) < 4:
        return 0.0
    span = coords.max(axis=0) - coords.min(axis=0)
    return float(span[0] * span[1] * span[2] * 0.7)


def score_single(protein_pdb: str, ligand_sdf: str = None,
                 smiles: str = None, ligand_resname: str = None,
                 pocket_cutoff: float = 10.0,
                 verbose: bool = True) -> ScoringResult:
    """Score ONE protein-ligand complex.
    
    Uses HETATM records from PDB as ligand (crystallographic pose).
    ligand_sdf and smiles are ignored for coordinates but used for properties.
    """
    result = ScoringResult()
    warnings_list = []

    rec_coords, rec_types, lig_coords, lig_types, resname, n_rot = \
        parse_pocket_and_ligand(protein_pdb, ligand_resname, pocket_cutoff)

    if verbose:
        print(f"  Ligand residue  : {resname}")
        print(f"  Ligand atoms    : {len(lig_coords)}")
        print(f"  Pocket atoms    : {len(rec_coords)}")

    lig_center = lig_coords.mean(axis=0)
    rec_center = rec_coords.mean(axis=0)
    separation = float(np.linalg.norm(lig_center - rec_center))
    if separation > 8.0:
        warnings_list.append(f"Ligand centroid {separation:.1f}Å from pocket centroid")
    elif verbose:
        print(f"  Ligand↔pocket   : {separation:.1f}Å")

    phys = score_physics(rec_coords, rec_types, lig_coords, lig_types, n_rot)

    result.gauss1_score      = phys["gauss1"]
    result.repulsion_score   = phys["repulsion"]
    result.hydrophobic_score = phys["hydrophobic"]
    result.hbond_score       = phys["hbond"]
    result.torsion_penalty   = phys["torsion"]
    result.raw_vina          = phys["raw_vina"]
    result.n_hbonds          = phys["n_hbonds"]
    result.n_hydrophobic     = phys["n_hydrophobic"]
    result.n_clashes         = phys["n_clashes"]
    result.n_contacts        = phys["n_contacts"]

    if phys["n_clashes"] > 10:
        warnings_list.append(f"{phys['n_clashes']} clashes")

    n_heavy = len(lig_coords)
    pkd = vina_to_pkd(phys["affinity"], n_heavy)
    dG = R_GAS * TEMP * math.log(10 ** (-pkd))
    kd_nM = 10 ** (-pkd) * 1e9

    result.pkd = pkd
    result.dG = dG
    result.kd_nM = kd_nM

    if smiles:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
        from rdkit.Chem.rdMolDescriptors import CalcTPSA
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            result.mw        = Descriptors.MolWt(mol)
            result.logP      = Descriptors.MolLogP(mol)
            result.tpsa      = CalcTPSA(mol)
            result.hbd       = Descriptors.NumHDonors(mol)
            result.hba       = Descriptors.NumHAcceptors(mol)
            result.n_rot     = Descriptors.NumRotatableBonds(mol)
            result.drug_like = (result.mw<=500 and result.logP<=5 and result.hbd<=5 and result.hba<=10)
            result.ligand_efficiency = dG / max(1, mol.GetNumHeavyAtoms())

    result.pocket_size   = len(rec_coords)
    result.pocket_volume = pocket_volume_estimate(rec_coords)

    result.confidence = compute_confidence(
        phys, len(rec_coords),
        phys["n_clashes"], phys["n_contacts"],
        has_smiles=(smiles is not None),
    )
    result.warnings = warnings_list
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score a single protein-ligand complex")
    parser.add_argument("--protein", required=True)
    parser.add_argument("--ligand", default=None)
    parser.add_argument("--smiles", default=None)
    parser.add_argument("--cutoff", type=float, default=10.0)
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()

    if not args.quiet:
        log.info("\nScoring compound...")
        log.info(f"  Protein : {args.protein}")
        if args.ligand: log.info(f"  Ligand  : {args.ligand}")
        if args.smiles: log.info(f"  SMILES  : {args.smiles[:60]}...")

    result = score_single(
        protein_pdb=args.protein, ligand_sdf=args.ligand,
        smiles=args.smiles, pocket_cutoff=args.cutoff, verbose=not args.quiet,
    )

    print(result.summary())


# ═══════════════════════════════════════════════════════════════════════════════
# ML MODEL PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════

_model_cache = None

def _load_model():
    global _model_cache
    if _model_cache is None:
        import pickle
        with open(os.path.join(os.path.dirname(__file__), 'affinity_model.pkl'), 'rb') as f:
            _model_cache = pickle.load(f)
    return _model_cache

def _compute_physics_features(lig_coords, lig_types, rec_coords, rec_types, center):
    """Compute 60 physics features for ML model."""
    features = np.zeros(60)
    if len(lig_coords) < 3 or len(rec_coords) < 10:
        return features
    
    lig_center = lig_coords.mean(axis=0)
    dist = np.linalg.norm(lig_center - center)
    
    all_dists = np.array([np.linalg.norm(lc - pc) for lc in lig_coords for pc in rec_coords])
    lig_dists = np.array([min(np.linalg.norm(lc - pc) for pc in rec_coords) for lc in lig_coords])
    rec_dists = np.array([min(np.linalg.norm(lc - pc) for lc in lig_coords) for pc in rec_coords])
    
    features[0] = np.exp(-dist**2 / (2 * 1.5**2))
    features[1] = np.exp(-dist**2 / (2 * 3.0**2))
    features[2] = np.exp(-dist**2 / (2 * 5.0**2))
    features[3] = np.exp(-all_dists.min()**2 / (2 * 0.5**2))
    features[4] = np.exp(-(all_dists.min() - 3.0)**2 / (2 * 1.0**2))
    features[5] = np.exp(-all_dists.mean()**2 / (2 * 3.0**2))
    features[6] = np.exp(-all_dists.std()**2 / (2 * 2.0**2))
    features[7] = sum(d * d for d in all_dists if d < 0)
    
    for i, d in enumerate([2.0, 3.0, 4.0, 5.0, 6.0, 8.0]):
        features[8+i] = np.sum(all_dists < d) / len(all_dists)
    
    features[14] = lig_dists.min()
    features[15] = lig_dists.mean()
    features[16] = lig_dists.std()
    features[17] = np.percentile(lig_dists, 25)
    features[18] = np.percentile(lig_dists, 50)
    features[19] = np.percentile(lig_dists, 75)
    features[20] = rec_dists.min()
    features[21] = rec_dists.mean()
    features[22] = rec_dists.std()
    
    n = len(lig_types)
    features[23] = sum(1 for t in lig_types if t in ['C','S']) / n
    features[24] = sum(1 for t in lig_types if t in ['N','O']) / n
    features[25] = sum(1 for t in lig_types if t in ['N','O','S']) / n
    features[26] = sum(1 for t in lig_types if t in ['C','N']) / n
    features[27] = sum(1 for t in lig_types if t == 'N') / n
    features[28] = sum(1 for t in lig_types if t in ['O','S']) / n
    features[29] = n / 100.0
    
    np_ = len(rec_types)
    features[30] = sum(1 for t in rec_types if t == 'C') / np_
    features[31] = sum(1 for t in rec_types if t in ['N','O']) / np_
    features[32] = np_ / 200.0
    
    contact = hydro = hbond = 0.0
    for i, lc in enumerate(lig_coords):
        for j, pc in enumerate(rec_coords):
            d = np.linalg.norm(lc - pc)
            if d < 4.5:
                contact += np.exp(-d**2 / 4.0)
                if lig_types[i] in ['C','S'] and rec_types[j] == 'C':
                    hydro += np.exp(-d**2 / 9.0) if d < 3.5 else 0
                if lig_types[i] in ['N','O'] and rec_types[j] in ['N','O']:
                    hbond += np.exp(-d**2 / 4.0)
    
    features[33] = contact / max(1, len(all_dists))
    features[34] = hydro / max(1, len(all_dists))
    features[35] = hbond / max(1, len(all_dists))
    features[37] = sum(1.0 for d in lig_dists if d > 2.0) / n
    features[38] = (features[27] - features[28]) * n / n
    features[39] = features[38] * (features[30] - features[31])
    features[40] = dist
    features[41] = np.sin(dist / 10.0)
    features[42] = np.cos(dist / 10.0)
    
    hist, _ = np.histogram(all_dists, bins=10, range=(0, 10))
    features[43:53] = hist / len(all_dists)
    for i, p in enumerate([5, 10, 25, 50, 75, 90, 95]):
        features[53+i] = np.percentile(all_dists, p) / 10.0
    
    return features

def _compute_ecfp(smiles, bits=512):
    """Compute ECFP4 fingerprint."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return np.zeros(bits)
        return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=bits))
    except: return np.zeros(bits)

def predict_affinity_ml(protein_pdb, smiles=None):
    """Predict binding affinity using ML model.
    
    Returns:
        float: Predicted ΔG in kcal/mol, or None if prediction fails
    """
    try:
        model_data = _load_model()
        rec_coords, rec_types, lig_coords, lig_types, _, _ = parse_pocket_and_ligand(protein_pdb, cutoff=10.0)
        center = rec_coords.mean(axis=0)
        
        phys = _compute_physics_features(lig_coords, lig_types, rec_coords, rec_types, center)
        ecfp = _compute_ecfp(smiles) if smiles else np.zeros(512)
        
        features = np.hstack([phys, ecfp]).reshape(1, -1)
        features_scaled = model_data['scaler'].transform(features)
        
        pred_dG = model_data['model'].predict(features_scaled)[0]
        return float(pred_dG)
    except Exception as e:
        return None
