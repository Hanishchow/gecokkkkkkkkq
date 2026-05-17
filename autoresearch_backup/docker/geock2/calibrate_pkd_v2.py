#!/usr/bin/env python3
"""Extract ligand from HETATM and use for calibration."""

import os
import sys
import json
import warnings
import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

sys.path.insert(0, '/mnt/c/Users/yakka/Downloads/final/geock')
warnings.filterwarnings("ignore")

DATA_DIR = "/mnt/c/Users/yakka/Downloads/geock_110_data"
COMPOUNDS_FILE = os.path.join(DATA_DIR, "compounds.json")

WATER = {"HOH", "WAT", "H2O", "DOD"}

def extract_ligand_from_pdb(pdb_path):
    """Extract ligand from HETATM records."""
    coords, types = [], []
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("HETATM"):
                continue
            resname = line[17:20].strip().upper()
            if resname in WATER:
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                el = line[76:78].strip().upper() if len(line) > 76 else ""
                name = line[12:16].strip()
                if not el:
                    el = ''.join(c for c in name if c.isalpha())[:2]
                    el = el[0] if el else "C"
                if el.upper() in ("H", "D"):
                    continue
                coords.append([x, y, z])
                if el == "N": types.append("NA")
                elif el == "O": types.append("OA")
                elif el == "S": types.append("SA")
                else: types.append(el)
            except:
                continue
    return np.array(coords, dtype=np.float32), types


def extract_pocket(pdb_path, centroid, cutoff=10.0):
    """Extract receptor atoms from ATOM records within cutoff."""
    coords, types = [], []
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            resname = line[17:20].strip().upper()
            if resname in WATER:
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                xyz = np.array([x, y, z])
            except:
                continue
            
            if np.linalg.norm(xyz - centroid) > cutoff:
                continue
            
            el = line[76:78].strip().upper() if len(line) > 76 else ""
            name = line[12:16].strip()
            if not el:
                el = ''.join(c for c in name if c.isalpha())[:2]
                el = el[0] if el else "C"
            if el.upper() in ("H", "D"):
                continue
            
            coords.append(xyz)
            if el == "N": types.append("NA")
            elif el == "O": types.append("OA")
            elif el == "S": types.append("SA")
            else: types.append(el)
    return np.array(coords, dtype=np.float32), types


def score_physics_simple(rec_coords, rec_types, lig_coords, lig_types, n_torsions=3):
    """Simplified physics scoring."""
    from score_compound import score_physics as sp
    return sp(rec_coords, rec_types, lig_coords, lig_types, n_torsions)


def main():
    from score_compound import W_GAUSS1, W_REPULSION, W_HYDROPHOBIC, W_HBOND, W_TORSION, VDW, CUTOFF
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    
    with open(COMPOUNDS_FILE) as f:
        compounds = json.load(f)
    
    print(f"Loaded {len(compounds)} compounds")
    
    results = []
    
    for i, compound in enumerate(compounds):
        pdb_id = compound["pdb_id"]
        pocket_file = os.path.join(DATA_DIR, pdb_id, f"{pdb_id}_pocket.pdb")
        
        if not os.path.exists(pocket_file):
            continue
        
        try:
            lig_coords, lig_types = extract_ligand_from_pdb(pocket_file)
            if len(lig_coords) < 3:
                continue
            
            centroid = lig_coords.mean(axis=0)
            rec_coords, rec_types = extract_pocket(pocket_file, centroid, cutoff=10.0)
            
            if len(rec_coords) < 10:
                continue
            
            phys = score_physics_simple(rec_coords, rec_types, lig_coords, lig_types, n_torsions=3)
            
            true_pkd = -compound["experimental_affinity"] / 1.364
            
            results.append({
                "pdb_id": pdb_id,
                "vina": phys["raw_vina"],
                "true_pkd": true_pkd,
                "exp_dG": compound["experimental_affinity"],
                "n_clashes": phys["n_clashes"],
                "n_contacts": phys["n_contacts"],
                "n_lig_atoms": len(lig_coords),
                "n_rec_atoms": len(rec_coords),
            })
            
        except Exception as e:
            continue
        
        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{len(compounds)}...")
    
    print(f"\nSuccessfully scored: {len(results)}")
    
    if len(results) < 10:
        print("❌ Not enough samples")
        return
    
    vina_scores = [r["vina"] for r in results]
    true_pkds = [r["true_pkd"] for r in results]
    
    print(f"\nScore distribution:")
    print(f"  Vina: min={min(vina_scores):.2f}, max={max(vina_scores):.2f}, mean={np.mean(vina_scores):.2f}")
    print(f"  True pKd: min={min(true_pkds):.2f}, max={max(true_pkds):.2f}, mean={np.mean(true_pkds):.2f}")
    
    n_positive = sum(1 for v in vina_scores if v > 0)
    print(f"  {n_positive}/{len(vina_scores)} have positive vina")
    
    print("\nSample (first 10):")
    for r in results[:10]:
        print(f"  {r['pdb_id']}: vina={r['vina']:+.2f}, true_pKd={r['true_pkd']:.2f}, clashes={r['n_clashes']}")
    
    valid = [r for r in results if r["vina"] < 0 and r["n_clashes"] < 30]
    print(f"\nFiltered (vina<0, clashes<30): {len(valid)}/{len(results)}")
    
    if len(valid) >= 10:
        X = np.array([r["vina"] for r in valid]).reshape(-1, 1)
        y = np.array([r["true_pkd"] for r in valid])
        
        reg = LinearRegression()
        reg.fit(X, y)
        
        preds = reg.predict(X)
        r_val = pearsonr([r["vina"] for r in valid], [r["true_pkd"] for r in valid])[0]
        mae = np.mean(np.abs(preds - y))
        
        print(f"\n{'='*60}")
        print(f"CALIBRATION RESULTS")
        print(f"{'='*60}")
        print(f"Formula: pKd = {reg.coef_[0]:.4f} * vina + {reg.intercept_:.4f}")
        print(f"Pearson R: {r_val:.4f}")
        print(f"MAE: {mae:.4f} pKd units")
        
        print(f"\nUPDATE score_compound.py vina_to_pkd():")
        print(f"def vina_to_pkd(vina_affinity: float, n_heavy_atoms: int = 20) -> float:")
        print(f"    SLOPE = {reg.coef_[0]:.6f}")
        print(f"    INTERCEPT = {reg.intercept_:.6f}")
        
        with open("/mnt/c/Users/yakka/Downloads/final/geock/calibration_params.json", "w") as f:
            json.dump({
                "slope": float(reg.coef_[0]),
                "intercept": float(reg.intercept_),
                "n_samples": len(valid),
                "pearson_r": float(r_val),
                "mae": float(mae)
            }, f, indent=2)
        print(f"\nSaved to calibration_params.json")


if __name__ == "__main__":
    main()
