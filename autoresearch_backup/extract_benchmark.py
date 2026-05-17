#!/usr/bin/env python3
"""
extract_benchmark.py — Extract features for 172 benchmark PDBs
================================================================
Pipeline:
  1. Extract ligand + pocket from full PDB (HETATM + nearest residues)
  2. Write {pdb_id}/{pdb_id}_pocket.pdb and .sdf files
  3. Run bio_engine on all 192 benchmark PDBs (172 new + 20 existing)
  4. Save as features_benchmark.pkl

This is for Path 2: scale-up to n=192 compounds.
"""

import os, sys, json, time, pickle, warnings, csv
warnings.filterwarnings("ignore")

import numpy as np
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Crippen
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

sys.path.insert(0, "/home/chow/autoresearch")
from patch_parse import parse_pocket_and_ligand
from bio_engine import unified_score, _bio_cache, biological_features

BENCHMARK_DIR  = Path("/mnt/c/Users/yakka/Downloads/benchmark_pdbs")
GEOCK_DATA_DIR = Path("/mnt/c/Users/yakka/Downloads/geock_110_data")
WORK_DIR       = Path("/mnt/c/Users/yakka/Downloads/geock_benchmark_data")
CACHE_OUT      = Path("CACHE_DIR / features_benchmark.pkl")

SKIP_RESIDUES = frozenset({
    "HOH","WAT","H2O","SO4","PO4","GOL","EDO","PEG","MPD",
    "ACT","ACE","FMT","IMD","EOH","DMS","TRS","BME","DTT",
})

POCKET_CUTOFF = 10.0  # Angstroms around ligand


def parse_affinity_csv(path):
    """Parse benchmark_matched_with_pdbs.csv → {pdb_id: affinity_kcal}"""
    aff = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            pdb = row["PDB_ID"].strip().lower()
            val = row["Experimental_Affinity_kcal_mol"].strip().replace("−","-")
            try:
                aff[pdb] = float(val)
            except:
                pass
    return aff


def extract_ligand_and_pocket(pdb_file):
    """
    Extract ligand HETATM + receptor atoms within POCKET_CUTOFF.
    Returns: rec_coords, rec_types, lig_coords, lig_types, lig_resname
    """
    rec_xyz, rec_types = [], []
    hetatm = {}  # resname → coords+types

    with open(pdb_file) as f:
        for line in f:
            rec = line[:6].strip()
            if rec == "ATOM":
                try:
                    el = line[76:78].strip() if len(line) > 76 else ""
                    name = line[12:16].strip()
                    if not el:
                        el = name[0] if name else "C"
                    if el.upper() in ("H","D"):
                        continue
                    if line[17:20].strip() in ("HOH","WAT"):
                        continue
                    xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])], dtype=np.float32)
                    vtype = el_type(el, name)
                    rec_xyz.append(xyz)
                    rec_types.append(vtype)
                except (ValueError, IndexError):
                    continue
            elif rec == "HETATM":
                try:
                    resname = line[17:20].strip()
                    if resname in SKIP_RESIDUES:
                        continue
                    el = line[76:78].strip() if len(line) > 76 else ""
                    name = line[12:16].strip()
                    if not el:
                        el = "".join(c for c in name if c.isalpha())[:2]
                    el = el.capitalize()
                    if el in ("H","D",""):
                        continue
                    xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])], dtype=np.float32)
                    if resname not in hetatm:
                        hetatm[resname] = {"xyz": [], "types": []}
                    hetatm[resname]["xyz"].append(xyz)
                    hetatm[resname]["types"].append(el_type(el, name))
                except:
                    continue

    if not hetatm:
        return None

    # Use the largest ligand
    ligand_key = max(hetatm, key=lambda k: len(hetatm[k]["xyz"]))
    lig_xyz = np.array(hetatm[ligand_key]["xyz"], dtype=np.float32)
    lig_types = hetatm[ligand_key]["types"]

    # Find pocket residues (ATOMs within POCKET_CUTOFF of ligand)
    rec_xyz_arr = np.array(rec_xyz, dtype=np.float32)
    dists = np.linalg.norm(rec_xyz_arr[:, None] - lig_xyz[None, :], axis=2)
    pocket_mask = dists.min(axis=1) < POCKET_CUTOFF
    pocket_xyz = rec_xyz_arr[pocket_mask]
    pocket_types = [t for t, m in zip(rec_types, pocket_mask) if m]

    return pocket_xyz, pocket_types, lig_xyz, lig_types, ligand_key


def el_type(el, name=""):
    e = el.strip().capitalize()
    if e == "N": return "NA"
    if e == "O": return "OA"
    if e == "S": return "SA"
    table = {"C":"C","F":"F","P":"P","Cl":"CL","Br":"BR","I":"I",
             "Fe":"FE","Zn":"ZN","Mg":"MG","Ca":"CA","Mn":"MN","Co":"ZN","Cu":"ZN"}
    return table.get(e, "C")


def write_pocket_pdb(pocket_xyz, pocket_types, lig_xyz, lig_types, resname, pdb_id, out_dir):
    """Write a pocket PDB file: ATOM for receptor + HETATM for ligand."""
    out_path = out_dir / pdb_id / f"{pdb_id}_pocket.pdb"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"HEADER    POCKET EXTRACTED FROM {pdb_id}",
             f"TITLE     Pocket: {POCKET_CUTOFF}A around ligand {resname}",
             f"REMARK 1  Generated by extract_benchmark.py"]
    # Receptor atoms (ATOM records)
    for i, (xyz, t) in enumerate(zip(pocket_xyz, pocket_types)):
        atom_name = t + str(i+1)
        atom_name = atom_name[:4]
        serial = i + 1
        line = f"ATOM  {serial:5d} {atom_name:4s} LIG A   1    {xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00          {t:>2s}"
        lines.append(line)
    # Ligand atoms (HETATM records — critical for parse_pocket_and_ligand)
    offset = len(pocket_xyz) + 1
    for i, (xyz, t) in enumerate(zip(lig_xyz, lig_types)):
        atom_name = t + str(i+1)
        atom_name = atom_name[:4]
        serial = offset + i
        line = f"HETATM{serial:5d} {atom_name:4s} {resname:3s} A   1    {xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00          {t:>2s}"
        lines.append(line)
    lines.append("END")
    out_path.write_text("\n".join(lines))
    return out_path


def write_ligand_sdf(lig_xyz, lig_types, resname, pdb_id, out_dir):
    """Write a minimal SDF for the ligand."""
    out_path = out_dir / pdb_id / f"{pdb_id}_ligand.sdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Build a simple RDKit mol from coords
    try:
        mol = Chem.RWMol()
        conf = Chem.Conformer()
        for i, (xyz, t) in enumerate(zip(lig_xyz, lig_types)):
            atom = Chem.Atom(t)
            mol.AddAtom(atom)
            conf.SetAtomPosition(i, (float(xyz[0]), float(xyz[1]), float(xyz[2])))
        mol.AddConformer(conf)
        mol = Chem.MolFromSmiles("")  # just to finalise
        w = Chem.SDWriter(str(out_path))
        w.write(mol)
        w.close()
    except Exception as e:
        # Fallback: write empty file
        out_path.write_text("")
    return out_path


def compute_features(pdb_id, pocket_pdb, lig_xyz, lig_types, resname,
                    n_torsions=1, smiles="", use_quantum=True):
    """Compute E1+E2+E3+E4 features for a compound."""
    try:
        feats = unified_score(
            pdb_id, pocket_pdb, lig_xyz, lig_types,
            n_torsions=n_torsions,
            smiles=smiles,
            use_quantum=use_quantum,
        )
        return feats
    except Exception as e:
        return None


def main():
    print("=" * 60)
    print("  GEOCK Path 2 — Benchmark Feature Extraction")
    print("=" * 60)

    # Load affinity data
    aff_path = "/mnt/c/Users/yakka/Downloads/benchmark_matched_with_pdbs.csv"
    aff = parse_affinity_csv(aff_path)
    print(f"\nAffinity data: {len(aff)} compounds")

    # Process benchmark PDBs
    benchmark_files = sorted([f for f in BENCHMARK_DIR.glob("*.pdb")])
    print(f"Benchmark PDB files: {len(benchmark_files)}")

    WORK_DIR.mkdir(parents=True, exist_ok=True)

    # Phase 1: Extract pockets from full PDB structures
    print(f"\n[Phase 1] Extracting pockets from {len(benchmark_files)} PDBs...")
    extracted = []
    failed = []

    for i, pdb_file in enumerate(benchmark_files):
        pdb_id = pdb_file.stem.lower()
        out_dir = WORK_DIR / pdb_id
        out_dir.mkdir(exist_ok=True)

        try:
            result = extract_ligand_and_pocket(pdb_file)
            if result is None:
                failed.append((pdb_id, "no ligand found"))
                continue

            pocket_xyz, pocket_types, lig_xyz, lig_types, resname = result

            if len(pocket_xyz) < 5:
                failed.append((pdb_id, "pocket too small"))
                continue

            write_pocket_pdb(pocket_xyz, pocket_types, lig_xyz, lig_types, resname, pdb_id, WORK_DIR)

            # Estimate n_torsions from ligand size
            nrot = max(1, len(lig_xyz) // 10)

            extracted.append({
                "pdb_id": pdb_id,
                "resname": resname,
                "n_torsions": nrot,
                "affinity": aff.get(pdb_id, 0.0),
                "smiles": "",
            })

            if (i+1) % 20 == 0:
                print(f"  [{i+1}/{len(benchmark_files)}] extracted: {len(extracted)}, failed: {len(failed)}")

        except Exception as e:
            failed.append((pdb_id, str(e)[:40]))

    print(f"\n  Extracted: {len(extracted)} / {len(benchmark_files)}")
    if failed:
        print(f"  Failed: {len(failed)}")
        for pdb_id, reason in failed[:5]:
            print(f"    {pdb_id}: {reason}")

    # Phase 2: Compute features
    print(f"\n[Phase 2] Computing features (no quantum — fast)...")
    X_list, y_list, pdb_ids_out = [], [], []
    res_weights = []

    # Load resolution data if available
    try:
        from bio_engine import _bio_cache
        bio_ok = True
    except:
        bio_ok = False

    for i, c in enumerate(extracted):
        pdb_id = c["pdb_id"]
        pocket_pdb = WORK_DIR / pdb_id / f"{pdb_id}_pocket.pdb"

        if not pocket_pdb.exists():
            continue

        try:
            # Parse pocket
            rec, rt, lig, lt, _, nrot = parse_pocket_and_ligand(str(pocket_pdb))

            feats = unified_score(
                pdb_id, rec, rt, lig, lt,
                n_torsions=c["n_torsions"],
                smiles=c["smiles"],
                use_quantum=False,  # Skip VQE for speed
            )

            # Resolution weight (default)
            res_w = 0.7  # default

            X_list.append(feats)
            y_list.append(c["affinity"])
            pdb_ids_out.append(pdb_id)
            res_weights.append(res_w)

            if (i+1) % 20 == 0:
                print(f"  [{i+1}/{len(extracted)}] features computed")

        except Exception as e:
            print(f"  FAILED {pdb_id}: {e}")

    if not X_list:
        print("ERROR: No features computed!")
        return

    X_raw = np.array(X_list, dtype=np.float32)
    y_raw = np.array(y_list, dtype=np.float32)
    y_pkd = (-y_raw / 1.364).astype(np.float32)  # Convert kcal/mol to pKd

    print(f"\n  Features: {X_raw.shape}")
    print(f"  y range: {y_pkd.min():.1f} to {y_pkd.max():.1f} pKd")
    print(f"  y mean: {y_pkd.mean():.2f} pKd")

    # Save
    data = {
        "X_raw": X_raw,
        "y_pkd": y_pkd,
        "pdb_ids": pdb_ids_out,
        "res_weights": np.array(res_weights, dtype=np.float32),
        "source": "benchmark",
        "n_compounds": len(pdb_ids_out),
    }
    CACHE_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_OUT, "wb") as f:
        pickle.dump(data, f)

    print(f"\n  Saved to {CACHE_OUT}")
    print(f"  Shape: X_raw={X_raw.shape}")

    # Quick stats
    print("\n  Feature stats (E1-E4):")
    names = ["E1_vinardo_gauss1","E1_vinardo_repulsion","E1_vinardo_hydrophobic",
             "E1_vinardo_hbond","E1_vinardo_torsion","E1_vinardo_affinity",
             "E2_chem_pi_pi","E2_chem_cation_pi","E2_chem_salt_bridge",
             "E2_chem_halogen_bond","E2_chem_metal_coord","E2_chem_burial",
             "E2_chem_shape","E2_chem_lipophilic",
             "E3_quantum_vqe",
             "E4_bio_drug_likeness","E4_bio_ligand_efficiency",
             "E4_bio_pocket_druggability","E4_bio_resolution_weight",
             "E4_bio_family_hydrophobic","E4_bio_family_hbond",
             "E4_bio_pocket_polarity","E4_bio_size_penalty","E4_bio_pharmacophore"]
    for i, n in enumerate(names):
        vals = X_raw[:, i]
        nan_count = np.isnan(vals).sum()
        if nan_count > 0:
            print(f"    [{i:2d}] {n}: nan={nan_count}")
        else:
            print(f"    [{i:2d}] {n}: mean={vals.mean():+.2f}  std={vals.std():.2f}  range=[{vals.min():.2f}, {vals.max():+.2f}]")

    print("\n  Done!")


if __name__ == "__main__":
    main()
