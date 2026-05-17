"""
extract_pdbbind.py — Feature Extraction for PDBbind Complexes
============================================================
Compute E1-E4 features for PDBbind complexes from fetch_pdbbind.py output.

Usage:
  python extract_pdbbind.py                    # all complexes
  python extract_pdbbind.py --limit 500        # first 500
  python extract_pdbbind.py --restart         # recompute from scratch
"""

import argparse
import os
import json
import pickle
import time
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bio_engine import run_all_compounds, IntegrationFilter
from patch_parse import parse_pocket_and_ligand
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.preprocessing import StandardScaler

OUT_DIR      = Path("/mnt/c/Users/yakka/Downloads/geock_pdbbind_data")
CACHE_FILE  = OUT_DIR / "compounds.json"
FEATURE_CACHE = OUT_DIR / "features_pdbbind.pkl"

FEATURE_SUBSET = slice(0, 24)   # physics only (E1+E2+E3+E4)
ECFP_SIZE = 512
TOTAL_FEATURES = 24 + ECFP_SIZE  # 536

# Threshold for druggability score (pocket features need > 0.3)
MIN_POCKET_ATOMS = 10
MIN_LIG_ATOMS = 5


def get_ecfp(smiles: str, fp_size: int = ECFP_SIZE) -> np.ndarray:
    if not smiles:
        return np.zeros(fp_size, dtype=np.float32)
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(fp_size, dtype=np.float32)
        fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=fp_size)
        return np.array(fpgen.GetFingerprintAsNumPy(mol), dtype=np.float32)
    except Exception:
        return np.zeros(fp_size, dtype=np.float32)


def extract_one(compound: dict, data_dir: Path):
    """Extract features for one compound."""
    pdb_id = compound["pdb_id"]
    smiles  = compound.get("smiles", "")

    # Try multiple possible pocket file locations
    possible_paths = [
        data_dir / pdb_id / f"{pdb_id}_pocket.pdb",
        data_dir / pdb_id / f"{pdb_id}_binding.pdb",
        data_dir / pdb_id / f"{pdb_id}.pdb",
    ]
    pocket_path = None
    for p in possible_paths:
        if p.exists():
            pocket_path = p
            break

    if pocket_path is None:
        return None, None, None

    try:
        rec_coords, rec_types, lig_coords, lig_types, _, n_rot = \
            parse_pocket_and_ligand(str(pocket_path), cutoff=10.0)

        if len(rec_coords) < MIN_POCKET_ATOMS or len(lig_coords) < MIN_LIG_ATOMS:
            return None, None, None

        from bio_engine import unified_score
        feats = unified_score(
            pdb_id, rec_coords, rec_types,
            lig_coords, lig_types,
            n_torsions=n_rot,
            smiles=smiles,
            use_quantum=False,   # VQE is too slow for 5000+ compounds
        )

        y = float(compound["experimental_affinity"])
        return feats, y, pdb_id

    except Exception as e:
        return None, None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0,
                        help="Max compounds to process (0=all)")
    parser.add_argument("--restart", action="store_true",
                        help="Ignore cache, recompute all")
    parser.add_argument("--quantum", action="store_true",
                        help="Enable VQE quantum feature (very slow)")
    parser.add_argument("--min-resolution", type=float, default=2.5,
                        help="Max resolution filter")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers (1=sequential)")
    args = parser.parse_args()

    # Load compounds
    if not CACHE_FILE.exists():
        print(f"[ERR] No compounds.json at {CACHE_FILE}")
        print(f"      Run fetch_pdbbind.py first.")
        return

    with open(CACHE_FILE) as f:
        all_compounds = json.load(f)

    # Filter by resolution
    compounds = [
        c for c in all_compounds
        if c.get("resolution", 99) <= args.min_resolution
        and c.get("n_pocket_atoms", 0) >= MIN_POCKET_ATOMS
        and c.get("n_lig_atoms", 0) >= MIN_LIG_ATOMS
    ]

    if args.limit > 0:
        compounds = compounds[:args.limit]

    print(f"[INFO] Processing {len(compounds)} / {len(all_compounds)} complexes")
    print(f"[INFO] Using {'E1+E2+E3+E4 (no VQE)' if not args.quantum else 'E1+E2+E3+E4+VQE'} features")

    # Load cache
    cache = {}
    if not args.restart and FEATURE_CACHE.exists():
        with open(FEATURE_CACHE, "rb") as f:
            cache = pickle.load(f)
        print(f"[INFO] Loaded {len(cache)} cached features")

    # Extract
    X_raw_list = []
    y_list = []
    pdb_ids = []
    failed = 0

    start = time.time()
    for i, c in enumerate(compounds):
        pid = c["pdb_id"]

        if not args.restart and pid in cache:
            X_raw_list.append(cache[pid]["X"])
            y_list.append(cache[pid]["y"])
            pdb_ids.append(pid)
            continue

        feats, y, result_pid = extract_one(c, OUT_DIR)
        if feats is not None:
            X_raw_list.append(feats)
            y_list.append(y)
            pdb_ids.append(result_pid)
            cache[pid] = {"X": feats, "y": y}
        else:
            failed += 1

        if (i + 1) % 100 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            eta = (len(compounds) - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{len(compounds)}] {rate:.1f}/s  ETA={eta/60:.1f}min  failed={failed}")
            # Checkpoint
            with open(FEATURE_CACHE, "wb") as f:
                pickle.dump(cache, f)

    # Assemble matrices
    X_raw = np.array(X_raw_list, dtype=np.float32)
    y_raw = np.array(y_list, dtype=np.float32)

    # Convert ΔG → pKd
    y_pkd = (-y_raw / 1.364).astype(np.float32)

    # ECFP fingerprints
    compound_map = {c["pdb_id"]: c for c in compounds}
    smiles_list = [compound_map.get(pid, {}).get("smiles", "") for pid in pdb_ids]
    X_ecfp = np.stack([get_ecfp(s) for s in smiles_list])

    # Combine
    X_all = np.hstack([X_raw, X_ecfp])

    # Train/val/test split (80/10/10 stratified by year if possible)
    n = len(X_all)
    n_train = int(n * 0.8)
    n_val   = int(n * 0.1)

    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)

    train_idx = indices[:n_train]
    val_idx   = indices[n_train:n_train + n_val]
    test_idx  = indices[n_train + n_val:]

    X_train = X_all[train_idx]
    y_train = y_pkd[train_idx]
    X_val   = X_all[val_idx]
    y_val   = y_pkd[val_idx]
    X_test  = X_all[test_idx]
    y_test  = y_pkd[test_idx]

    # Save cache
    data = {
        "X_raw": X_raw,
        "X_filt": X_raw,
        "X_ecfp": X_ecfp,
        "y_pkd": y_pkd,
        "y_raw": y_raw,
        "pdb_ids": pdb_ids,
        "smiles_list": smiles_list,
        "X_all": X_all,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n - n_train - n_val,
    }

    out_pkl = OUT_DIR / "features_pdbbind_v2.pkl"
    with open(out_pkl, "wb") as f:
        pickle.dump(data, f)

    elapsed = time.time() - start
    print(f"\n[COMPLETE] {len(pdb_ids)} complexes extracted in {elapsed/60:.1f}min")
    print(f"[OUTPUT]  {out_pkl}")
    print(f"  X_all:  {X_all.shape}  (24 physics + {ECFP_SIZE} ECFP)")
    print(f"  Train:  {len(y_train)}  Val: {len(y_val)}  Test: {len(y_test)}")
    print(f"  Failed: {failed} / {len(compounds)}")

    # Quick sanity check
    from scipy.stats import pearsonr
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Lasso
    from sklearn.feature_selection import SelectKBest, f_regression

    X_tr = X_train[:, :24]
    X_vl = X_val[:, :24]

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_vl_s = scaler.transform(X_vl)

    selector = SelectKBest(f_regression, k=4)
    X_tr_sel = selector.fit_transform(X_tr_s, y_train)
    X_vl_sel = selector.transform(X_vl_s)

    model = Lasso(alpha=0.01, max_iter=5000)
    model.fit(X_tr_sel, y_train)
    pred = model.predict(X_vl_sel)
    r = pearsonr(y_val, pred)[0]
    print(f"\n[SANITY] Lasso k=4 on PDBbind: Val R = {r:.3f}")
    print(f"  (Should be higher than GEOCK's 0.73 if features are informative)")


if __name__ == "__main__":
    main()
