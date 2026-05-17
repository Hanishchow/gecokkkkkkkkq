"""
extract_casf.py — CASF-2016 Feature Extraction
==============================================
Compute E1-E4 features for CASF-2016 benchmark complexes.

Usage:
  python extract_casf.py --limit 285
"""

import argparse, json, pickle, time, sys, numpy as np
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from patch_parse import parse_pocket_and_ligand

OUT_DIR      = Path("/mnt/c/Users/yakka/Downloads/geock_casf_data")
CASF_DIR     = OUT_DIR / "CASF2016"
CACHE_FILE   = OUT_DIR / "compounds_casf.json"
FEATURE_OUT  = OUT_DIR / "features_casf.pkl"
ECFP_SIZE    = 512


def mol2_to_pocket(pdb_id: str):
    """Convert CASF mol2 ligand + protein PDB to pocket PDB file."""
    prot_file = CASF_DIR / "Crystal Structures" / pdb_id / f"{pdb_id}_protein.pdb"
    lig_file  = CASF_DIR / "Crystal Structures" / pdb_id / f"{pdb_id}_ligand.mol2"
    out_file  = CASF_DIR / "Crystal Structures" / pdb_id / f"{pdb_id}_pocket.pdb"

    if not prot_file.exists() or not lig_file.exists():
        return None

    lines = []
    # Protein atoms from PDB
    with open(prot_file) as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                elem = line[76:78].strip().upper()
                if elem not in ("H", "D", ""):
                    lines.append(line)

    # Ligand atoms from mol2
    with open(lig_file) as f:
        in_mol = False
        for line in f:
            if line.startswith("@<TRIPOS>ATOM"):
                in_mol = True
                continue
            if line.startswith("@<TRIPOS>BOND"):
                break
            if in_mol and line.strip():
                # Parse mol2 atom: idx name x y z type ...
                parts = line.split()
                if len(parts) < 6:
                    continue
                try:
                    x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                    atom_type = parts[5].upper()
                    elem_map = {
                        "C.3": "C", "C.2": "C", "C.ar": "A", "C.1": "C",
                        "N.3": "NA", "N.2": "N", "N.1": "NA", "N.am": "NA",
                        "O.3": "OA", "O.2": "O", "O.co2": "OA",
                        "S.3": "SA", "S.2": "S",
                        "F": "F", "CL": "CL", "BR": "BR", "I": "I",
                        "P": "P", "FE": "FE", "ZN": "ZN", "MG": "MG",
                        "CA": "CA", "MN": "MN",
                    }
                    elem = elem_map.get(atom_type, "C")
                    serial = len(lines) + 1
                    lines.append(
                        f"HETATM{serial:5d} {atom_type[:4]:4s} LIG L{1:4d}    "
                        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {elem:>2s}\n"
                    )
                except (ValueError, IndexError):
                    continue

    if not lines:
        return None

    with open(out_file, "w") as f:
        f.writelines(lines)

    return str(out_file)


def extract_one(pdb_id: str, smiles: str):
    """Extract features for one CASF complex."""
    # Build pocket PDB if needed
    pocket_file = CASF_DIR / "Crystal Structures" / pdb_id / f"{pdb_id}_pocket.pdb"
    if not pocket_file.exists():
        result = mol2_to_pocket(pdb_id)
        if result is None:
            return None, None

    try:
        from bio_engine import unified_score
        rec_coords, rec_types, lig_coords, lig_types, _, n_rot = \
            parse_pocket_and_ligand(str(pocket_file), cutoff=10.0)

        if len(rec_coords) < 10 or len(lig_coords) < 5:
            return None, None

        feats = unified_score(
            pdb_id, rec_coords, rec_types,
            lig_coords, lig_types,
            n_torsions=n_rot,
            smiles=smiles or None,
            use_quantum=False,
        )
        return feats, pocket_file

    except Exception as e:
        return None, None


def get_ecfp(smiles: str, fp_size: int = ECFP_SIZE) -> np.ndarray:
    if not smiles:
        return np.zeros(fp_size, dtype=np.float32)
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(fp_size, dtype=np.float32)
        fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=fp_size)
        return np.array(fpgen.GetFingerprintAsNumPy(mol), dtype=np.float32)
    except:
        return np.zeros(fp_size, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()

    if not CACHE_FILE.exists():
        print(f"[ERR] Run fetch_casf.py first: {CACHE_FILE} not found")
        return

    with open(CACHE_FILE) as f:
        compounds = json.load(f)

    if args.limit > 0:
        compounds = compounds[:args.limit]

    print(f"[INFO] Processing {len(compounds)} CASF-2016 complexes")

    cache = {}
    if not args.restart and FEATURE_OUT.exists():
        with open(FEATURE_OUT, "rb") as f:
            cache = pickle.load(f)

    X_list, y_list, pdb_ids, smiles_list = [], [], [], []
    failed = 0
    start = time.time()

    for i, c in enumerate(compounds):
        pid = c["pdb_id"]
        if not args.restart and pid in cache:
            X_list.append(cache[pid]["X"])
            y_list.append(cache[pid]["y"])
            pdb_ids.append(pid)
            smiles_list.append(c.get("smiles", ""))
            continue

        feats, _ = extract_one(pid, c.get("smiles", ""))
        if feats is not None:
            X_list.append(feats)
            y_list.append(c["experimental_affinity"])
            pdb_ids.append(pid)
            smiles_list.append(c.get("smiles", ""))
            cache[pid] = {"X": feats, "y": c["experimental_affinity"]}
        else:
            failed += 1

        if (i + 1) % 50 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            print(f"  [{i+1}/{len(compounds)}] {rate:.1f}/s  failed={failed}")
            with open(FEATURE_OUT, "wb") as f:
                pickle.dump(cache, f)

    X_raw  = np.array(X_list, dtype=np.float32)
    y_raw  = np.array(y_list, dtype=np.float32)
    y_pkd  = (-y_raw / 1.364).astype(np.float32)
    X_ecfp = np.stack([get_ecfp(s) for s in smiles_list])
    X_all  = np.hstack([X_raw, X_ecfp])

    n = len(X_all)
    n_train = int(n * 0.8)
    n_val   = int(n * 0.1)

    idx = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(idx)

    train_idx, val_idx, test_idx = idx[:n_train], idx[n_train:n_train+n_val], idx[n_train+n_val:]

    data = {
        "X_all": X_all, "X_raw": X_raw, "X_ecfp": X_ecfp,
        "y_pkd": y_pkd, "y_raw": y_raw,
        "pdb_ids": pdb_ids, "smiles_list": smiles_list,
        "X_train": X_all[train_idx], "y_train": y_pkd[train_idx],
        "X_val":   X_all[val_idx],   "y_val":   y_pkd[val_idx],
        "X_test":  X_all[test_idx],  "y_test":  y_pkd[test_idx],
        "n_train": len(train_idx), "n_val": len(val_idx), "n_test": len(test_idx),
    }

    with open(FEATURE_OUT, "wb") as f:
        pickle.dump(data, f)

    elapsed = time.time() - start
    print(f"\n[DONE] {len(pdb_ids)} complexes in {elapsed/60:.1f}min")
    print(f"[OUT]  {FEATURE_OUT}")

    # Sanity check
    from scipy.stats import pearsonr
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Lasso
    from sklearn.feature_selection import SelectKBest, f_regression

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_raw[train_idx])
    X_vl = scaler.transform(X_raw[val_idx])

    selector = SelectKBest(f_regression, k=4)
    X_tr_s = selector.fit_transform(X_tr, y_pkd[train_idx])
    X_vl_s = selector.transform(X_vl)

    model = Lasso(alpha=0.01, max_iter=5000)
    model.fit(X_tr_s, y_pkd[train_idx])
    r = pearsonr(y_pkd[val_idx], model.predict(X_vl_s))[0]
    print(f"\n[SANITY] CASF-2016 Lasso k=4: Val R = {r:.3f}")


if __name__ == "__main__":
    import os
    main()
