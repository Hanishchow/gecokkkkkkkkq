"""
patch_parse.py — Fix ligand parsing by using HETATM records from PDB
"""

import os
import numpy as np

SKIP_RESIDUES = frozenset({
    "HOH","WAT","H2O","SO4","PO4","GOL","EDO","PEG","MPD",
    "ACT","ACE","FMT","IMD","EOH","DMS","TRS","BME","DTT",
})

def _vtype(el, name=""):
    e = el.strip().upper()
    if e == "N":  return "NA"
    if e == "O":  return "OA"
    if e == "S":  return "SA"
    return {"C":"C","F":"F","P":"P","CL":"CL","BR":"BR",
            "I":"I","FE":"FE","ZN":"ZN","MG":"MG",
            "CA":"CA","MN":"MN","CO":"ZN","CU":"ZN"}.get(e, "C")


def parse_pocket_and_ligand(pocket_pdb: str,
                             ligand_resname: str = None,
                             cutoff: float = 10.0):
    """
    ONE function. ONE pass through the PDB.
    ATOM   records → receptor
    HETATM records → ligand (the crystallographic pose, all atoms)

    Returns:
        rec_coords  np.ndarray (M, 3)
        rec_types   list[str]
        lig_coords  np.ndarray (N, 3)
        lig_types   list[str]
        lig_resname str
        n_rot       int   (rough estimate)
    """
    rec_xyz, rec_t = [], []
    hetatm = {}          # resname → {xyz:[], types:[]}

    with open(pocket_pdb) as f:
        for line in f:
            rec = line[:6].strip()

            if rec == "ATOM":
                try:
                    el   = line[76:78].strip() if len(line) > 76 else ""
                    name = line[12:16].strip()
                    if not el: el = name[0] if name else "C"
                    if el.upper() in ("H","D"): continue
                    if line[17:20].strip() in ("HOH","WAT"): continue
                    xyz = np.array([float(line[30:38]),
                                    float(line[38:46]),
                                    float(line[46:54])], dtype=np.float32)
                    rec_xyz.append(xyz)
                    rec_t.append(_vtype(el, name))
                except (ValueError, IndexError):
                    continue

            elif rec == "HETATM":
                try:
                    resname = line[17:20].strip()
                    if resname in SKIP_RESIDUES: continue

                    el   = line[76:78].strip() if len(line) > 76 else ""
                    name = line[12:16].strip()
                    if not el:
                        el = "".join(c for c in name if c.isalpha())[:2]
                    el = el.capitalize()
                    if el in ("H","D",""): continue

                    xyz = np.array([float(line[30:38]),
                                    float(line[38:46]),
                                    float(line[46:54])], dtype=np.float32)

                    if resname not in hetatm:
                        hetatm[resname] = {"xyz":[], "types":[]}
                    hetatm[resname]["xyz"].append(xyz)
                    hetatm[resname]["types"].append(_vtype(el, name))
                except (ValueError, IndexError):
                    continue

    if not hetatm:
        raise ValueError(f"No HETATM ligand records in {pocket_pdb}")

    if ligand_resname and ligand_resname.upper() in hetatm:
        key = ligand_resname.upper()
    else:
        key = max(hetatm, key=lambda k: len(hetatm[k]["xyz"]))

    lig_xyz   = np.array(hetatm[key]["xyz"],   dtype=np.float32)
    lig_types = hetatm[key]["types"]

    if len(lig_xyz) < 3:
        raise ValueError(f"Ligand {key} has only {len(lig_xyz)} atoms — likely a metal ion")

    lig_center = lig_xyz.mean(axis=0)
    rec_arr    = np.array(rec_xyz, dtype=np.float32)
    dists      = np.linalg.norm(rec_arr - lig_center, axis=1)
    mask       = dists <= cutoff

    rec_coords = rec_arr[mask]
    rec_types  = [rec_t[i] for i in range(len(rec_t)) if mask[i]]

    n_rot = max(0, len(lig_xyz) // 5 - 1)

    return rec_coords, rec_types, lig_xyz, lig_types, key, n_rot
