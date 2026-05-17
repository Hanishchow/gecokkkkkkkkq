#!/usr/bin/env python3
"""
calibrate_pkd.py — Calibrate pKd conversion using known affinities

Usage:
    python calibrate_pkd.py
"""

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


def experimental_to_pKd(dG_kcal_mol):
    """Convert experimental ΔG (kcal/mol) to pKd."""
    return -dG_kcal_mol / 1.364


def main():
    from score_compound import score_single
    
    with open(COMPOUNDS_FILE) as f:
        compounds = json.load(f)
    
    print(f"Loaded {len(compounds)} compounds")
    
    results = []
    failed = []
    
    for i, compound in enumerate(compounds):
        pdb_id = compound["pdb_id"]
        pocket_file = os.path.join(DATA_DIR, pdb_id, f"{pdb_id}_pocket.pdb")
        ligand_file = os.path.join(DATA_DIR, pdb_id, f"{pdb_id}_ligand.sdf")
        
        if not os.path.exists(pocket_file) or not os.path.exists(ligand_file):
            failed.append((pdb_id, "missing files"))
            continue
        
        try:
            result = score_single(
                protein_pdb=pocket_file,
                ligand_sdf=ligand_file,
                pocket_cutoff=10.0,
                verbose=False
            )
            
            true_pkd = experimental_to_pKd(compound["experimental_affinity"])
            
            results.append({
                "pdb_id": pdb_id,
                "vina": result.raw_vina,
                "true_pkd": true_pkd,
                "exp_dG": compound["experimental_affinity"],
                "n_clashes": result.n_clashes,
                "n_contacts": result.n_contacts,
                "warnings": result.warnings
            })
            
        except Exception as e:
            failed.append((pdb_id, str(e)[:50]))
            continue
        
        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{len(compounds)}...")
    
    print(f"\nResults:")
    print(f"  Successfully scored: {len(results)}")
    print(f"  Failed: {len(failed)}")
    
    if len(results) < 10:
        print("\n❌ Not enough valid samples for calibration")
        return
    
    vina_scores = [r["vina"] for r in results]
    true_pkds = [r["true_pkd"] for r in results]
    
    print(f"\nScore distribution:")
    print(f"  Vina scores: min={min(vina_scores):.2f}, max={max(vina_scores):.2f}, mean={np.mean(vina_scores):.2f}")
    print(f"  True pKd: min={min(true_pkds):.2f}, max={max(true_pkds):.2f}, mean={np.mean(true_pkds):.2f}")
    
    n_positive_vina = sum(1 for v in vina_scores if v > 0)
    print(f"\n  {n_positive_vina}/{len(vina_scores)} compounds have positive vina scores")
    
    n_bad_clashes = sum(1 for r in results if r["n_clashes"] > 30)
    print(f"  {n_bad_clashes} compounds have >30 clashes")
    
    print("\nSample results (first 10):")
    for r in results[:10]:
        coord_warn = "COORD_MISMATCH" if any("mismatch" in w.lower() for w in r["warnings"]) else ""
        print(f"  {r['pdb_id']}: vina={r['vina']:+.2f}, true_pKd={r['true_pkd']:.2f}, clashes={r['n_clashes']} {coord_warn}")
    
    valid_results = [r for r in results if r["vina"] < 0 and r["n_clashes"] < 20]
    print(f"\nFiltered results (vina<0, clashes<20): {len(valid_results)}/{len(results)}")
    
    if len(valid_results) >= 10:
        X = np.array([r["vina"] for r in valid_results]).reshape(-1, 1)
        y = np.array([r["true_pkd"] for r in valid_results])
        
        reg = LinearRegression()
        reg.fit(X, y)
        
        preds = reg.predict(X)
        r_val = pearsonr([r["vina"] for r in valid_results], [r["true_pkd"] for r in valid_results])[0]
        mae = np.mean(np.abs(preds - y))
        
        print(f"\n{'='*60}")
        print(f"CALIBRATION RESULTS (filtered)")
        print(f"{'='*60}")
        print(f"Samples: {len(valid_results)}")
        print(f"Formula: pKd = {reg.coef_[0]:.4f} * vina + {reg.intercept_:.4f}")
        print(f"Pearson R: {r_val:.4f}")
        print(f"MAE: {mae:.4f} pKd units")
        
        print(f"\nUPDATE score_compound.py vina_to_pkd() with:")
        print(f"def vina_to_pkd(vina_affinity: float, n_heavy_atoms: int = 20) -> float:")
        print(f"    SLOPE = {reg.coef_[0]:.6f}")
        print(f"    INTERCEPT = {reg.intercept_:.6f}")
        print(f"    pkd = SLOPE * vina_affinity + INTERCEPT")
        print(f"    size_correction = max(0, (n_heavy_atoms - 25) * 0.03)")
        print(f"    pkd -= size_correction")
        print(f"    return float(np.clip(pkd, 1.0, 14.0))")
        
        sample = valid_results[:5]
        print(f"\nSample validation (vina, true_pKd, predicted, error):")
        for r in sample:
            pred = reg.coef_[0] * r["vina"] + reg.intercept_
            print(f"  {r['pdb_id']}: {r['vina']:+.2f} → true={r['true_pkd']:.2f}, pred={pred:.2f}, err={pred-r['true_pkd']:+.2f}")
        
        with open("/mnt/c/Users/yakka/Downloads/final/geock/calibration_params.json", "w") as f:
            json.dump({
                "slope": float(reg.coef_[0]),
                "intercept": float(reg.intercept_),
                "n_samples": len(valid_results),
                "pearson_r": float(r_val),
                "mae": float(mae)
            }, f, indent=2)
        print(f"\nSaved calibration params to calibration_params.json")
    else:
        print(f"\n⚠️  Only {len(valid_results)} valid samples after filtering (need 10)")
        print("Trying calibration with all results...")
        
        X = np.array(vina_scores).reshape(-1, 1)
        y = np.array(true_pkds)
        
        reg = LinearRegression()
        reg.fit(X, y)
        
        preds = reg.predict(X)
        r_val = pearsonr(vina_scores, true_pkds)[0]
        mae = np.mean(np.abs(preds - y))
        
        print(f"\n{'='*60}")
        print(f"CALIBRATION RESULTS (all data - WARNING: may be unreliable)")
        print(f"{'='*60}")
        print(f"Samples: {len(results)}")
        print(f"Formula: pKd = {reg.coef_[0]:.4f} * vina + {reg.intercept_:.4f}")
        print(f"Pearson R: {r_val:.4f}")
        print(f"MAE: {mae:.4f} pKd units")
        
        with open("/mnt/c/Users/yakka/Downloads/final/geock/calibration_params.json", "w") as f:
            json.dump({
                "slope": float(reg.coef_[0]),
                "intercept": float(reg.intercept_),
                "n_samples": len(results),
                "pearson_r": float(r_val),
                "mae": float(mae),
                "warning": "calibrated on all data, may be unreliable"
            }, f, indent=2)
        print(f"\nSaved calibration params to calibration_params.json")


if __name__ == "__main__":
    main()
