#!/usr/bin/env python3
"""
CASF-2016 Validation for GEOCK Binding Affinity Prediction

This script validates the GEOCK model on CASF-2016 benchmark dataset.
CASF-2016 contains 285 protein-ligand complexes with experimental binding affinities.
"""

import os
import pickle
import numpy as np
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

CASf_2016_DIR = Path("/mnt/c/Users/yakka/Downloads/CASF-2016/CASF-2016")
CORESET_DIR = CASf_2016_DIR / "coreset"
CORESET_DAT = CASf_2016_DIR / "power_scoring" / "CoreSet.dat"
MODEL_PATH = Path("geock_deep_trees_final.pkl")


def load_core_set():
    """Load CASF-2016 core set data."""
    complexes = []
    with open(CORESET_DAT, 'r') as f:
        header = f.readline().strip().split()
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                pdb_id = parts[0]
                resolution = float(parts[1])
                year = int(parts[2])
                log_ka = float(parts[3])
                ka_str = parts[4] if len(parts) > 4 else ""
                target = int(parts[5]) if len(parts) > 5 else 0
                complexes.append({
                    'pdb_id': pdb_id,
                    'resolution': resolution,
                    'year': year,
                    'log_ka': log_ka,
                    'ka_str': ka_str,
                    'target': target
                })
    return complexes


def compute_ecfp4_fingerprint(mol, radius=2, n_bits=512):
    """
    Compute ECFP4 fingerprint (Morgan fingerprint with radius=2).
    
    Args:
        mol: RDKit molecule object
        radius: Fingerprint radius (2 for ECFP4)
        n_bits: Fingerprint bit length
    
    Returns:
        numpy array of fingerprint bits
    """
    if mol is None:
        return None
    
    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol, 
        radius=radius, 
        nBits=n_bits,
        useChirality=True
    )
    arr = np.zeros((n_bits,), dtype=np.int8)
    for i in range(n_bits):
        arr[i] = fp.GetBit(i)
    
    return arr


def load_ligand_mol2(mol2_path):
    """Load ligand from MOL2 file."""
    try:
        mol = Chem.MolFromMol2File(str(mol2_path), removeHs=False)
        return mol
    except Exception as e:
        print(f"Error loading {mol2_path}: {e}")
        return None


def load_ligand_sdf(sdf_path):
    """Load ligand from SDF file (backup)."""
    try:
        suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
        mols = [mol for mol in suppl if mol is not None]
        if mols:
            return mols[0]
        return None
    except Exception as e:
        print(f"Error loading {sdf_path}: {e}")
        return None


def prepare_features(complexes):
    """
    Prepare ECFP4 features for all complexes.
    
    Returns:
        features: numpy array of shape (n_complexes, 512)
        valid_complexes: list of complexes that were successfully processed
    """
    features = []
    valid_complexes = []
    failed = []
    
    for cx in complexes:
        pdb_id = cx['pdb_id']
        mol2_path = CORESET_DIR / pdb_id / f"{pdb_id}_ligand.mol2"
        sdf_path = CORESET_DIR / pdb_id / f"{pdb_id}_ligand.sdf"
        
        mol = None
        if mol2_path.exists():
            mol = load_ligand_mol2(mol2_path)
        
        if mol is None and sdf_path.exists():
            mol = load_ligand_sdf(sdf_path)
        
        if mol is None:
            failed.append((pdb_id, "Could not load molecule"))
            continue
        
        fp = compute_ecfp4_fingerprint(mol)
        if fp is None:
            failed.append((pdb_id, "Could not compute fingerprint"))
            continue
        
        features.append(fp)
        valid_complexes.append(cx)
    
    if failed:
        print(f"\nFailed to process {len(failed)} complexes:")
        for pdb_id, reason in failed[:5]:
            print(f"  {pdb_id}: {reason}")
        if len(failed) > 5:
            print(f"  ... and {len(failed) - 5} more")
    
    return np.array(features), valid_complexes


def evaluate_predictions(y_true, y_pred):
    """
    Evaluate predictions using Pearson R and Spearman rho.
    
    Returns:
        dict with R, rho, RMSE, MAE
    """
    r, r_pvalue = pearsonr(y_true, y_pred)
    rho, rho_pvalue = spearmanr(y_true, y_pred)
    
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    
    return {
        'R': r,
        'R_pvalue': r_pvalue,
        'rho': rho,
        'rho_pvalue': rho_pvalue,
        'RMSE': rmse,
        'MAE': mae,
        'n_samples': len(y_true)
    }


def analyze_extreme_binders(y_true, y_pred, pKd_thresholds=(5.0, 9.0)):
    """
    Analyze prediction errors for extreme binders.
    
    Args:
        y_true: True pKd values
        y_pred: Predicted pKd values
        pKd_thresholds: (weak_threshold, strong_threshold)
    
    Returns:
        dict with analysis results
    """
    weak_mask = y_true < pKd_thresholds[0]
    strong_mask = y_true > pKd_thresholds[1]
    medium_mask = ~(weak_mask | strong_mask)
    
    results = {}
    
    if weak_mask.sum() > 0:
        weak_errors = y_pred[weak_mask] - y_true[weak_mask]
        results['weak_binders'] = {
            'count': weak_mask.sum(),
            'mean_error': weak_errors.mean(),
            'std_error': weak_errors.std(),
            'mean_abs_error': np.abs(weak_errors).mean()
        }
    
    if strong_mask.sum() > 0:
        strong_errors = y_pred[strong_mask] - y_true[strong_mask]
        results['strong_binders'] = {
            'count': strong_mask.sum(),
            'mean_error': strong_errors.mean(),
            'std_error': strong_errors.std(),
            'mean_abs_error': np.abs(strong_errors).mean()
        }
    
    if medium_mask.sum() > 0:
        medium_errors = y_pred[medium_mask] - y_true[medium_mask]
        results['medium_binders'] = {
            'count': medium_mask.sum(),
            'mean_error': medium_errors.mean(),
            'std_error': medium_errors.std(),
            'mean_abs_error': np.abs(medium_errors).mean()
        }
    
    return results


def save_predictions(complexes, predictions, output_path):
    """
    Save predictions in CASF scoring file format.
    
    Format: #code score
    """
    with open(output_path, 'w') as f:
        f.write("#code\tscore\n")
        for cx, pred in zip(complexes, predictions):
            f.write(f"{cx['pdb_id']}\t{pred:.4f}\n")
    print(f"Saved predictions to {output_path}")


def main():
    print("=" * 70)
    print("CASF-2016 Validation for GEOCK Binding Affinity Prediction")
    print("=" * 70)
    
    print(f"\nModel path: {MODEL_PATH}")
    print(f"CASF-2016 directory: {CASf_2016_DIR}")
    
    if not MODEL_PATH.exists():
        print(f"\nERROR: Model not found at {MODEL_PATH}")
        print("Please ensure the model has been trained and saved.")
        return
    
    print("\nStep 1: Loading model...")
    with open(MODEL_PATH, 'rb') as f:
        model_dict = pickle.load(f)
    
    if isinstance(model_dict, dict):
        model = model_dict['model']
        scaler = model_dict.get('scaler')
        selector = model_dict.get('selector')
        cv_r = model_dict.get('cv_r')
        print(f"  Model dict with keys: {list(model_dict.keys())}")
    else:
        model = model_dict
        scaler = None
        selector = None
        cv_r = None
    
    print(f"  Model type: {type(model).__name__}")
    if hasattr(model, 'n_estimators'):
        print(f"  n_estimators: {model.n_estimators}")
    if hasattr(model, 'max_depth'):
        print(f"  max_depth: {model.max_depth}")
    if cv_r is not None:
        print(f"  CV R: {cv_r:.4f}")
    
    print("\nStep 2: Loading CASF-2016 core set...")
    complexes = load_core_set()
    print(f"  Loaded {len(complexes)} complexes")
    
    print("\nStep 3: Preparing ECFP4 features...")
    features, valid_complexes = prepare_features(complexes)
    print(f"  Successfully processed {len(valid_complexes)} / {len(complexes)} complexes")
    print(f"  Feature shape: {features.shape}")
    
    print("\nStep 4: Running predictions...")
    X = features
    if scaler is not None:
        X = scaler.transform(X)
    if selector is not None:
        X = selector.transform(X)
    predictions = model.predict(X)
    
    print("\nStep 5: Evaluating predictions...")
    y_true = np.array([cx['log_ka'] for cx in valid_complexes])
    y_pred = predictions
    
    results = evaluate_predictions(y_true, y_pred)
    
    print("\n" + "=" * 70)
    print("OVERALL RESULTS")
    print("=" * 70)
    print(f"  Pearson R:      {results['R']:.4f} (p={results['R_pvalue']:.2e})")
    print(f"  Spearman ρ:     {results['rho']:.4f} (p={results['rho_pvalue']:.2e})")
    print(f"  RMSE:           {results['RMSE']:.4f} pKd")
    print(f"  MAE:            {results['MAE']:.4f} pKd")
    print(f"  N samples:      {results['n_samples']}")
    
    print("\n" + "=" * 70)
    print("EXTREME BINDER ANALYSIS")
    print("=" * 70)
    extreme_results = analyze_extreme_binders(y_true, y_pred)
    
    if 'weak_binders' in extreme_results:
        wb = extreme_results['weak_binders']
        print(f"\n  Weak binders (pKd < 5.0):")
        print(f"    Count:       {wb['count']}")
        print(f"    Mean error:  {wb['mean_error']:+.4f} pKd")
        print(f"    Std error:   {wb['std_error']:.4f} pKd")
        print(f"    MAE:         {wb['mean_abs_error']:.4f} pKd")
    
    if 'strong_binders' in extreme_results:
        sb = extreme_results['strong_binders']
        print(f"\n  Strong binders (pKd > 9.0):")
        print(f"    Count:       {sb['count']}")
        print(f"    Mean error:  {sb['mean_error']:+.4f} pKd")
        print(f"    Std error:   {sb['std_error']:.4f} pKd")
        print(f"    MAE:         {sb['mean_abs_error']:.4f} pKd")
    
    if 'medium_binders' in extreme_results:
        mb = extreme_results['medium_binders']
        print(f"\n  Medium binders (5.0 ≤ pKd ≤ 9.0):")
        print(f"    Count:       {mb['count']}")
        print(f"    Mean error:  {mb['mean_error']:+.4f} pKd")
        print(f"    Std error:   {mb['std_error']:.4f} pKd")
        print(f"    MAE:         {mb['mean_abs_error']:.4f} pKd")
    
    output_path = Path("casf2016_predictions.csv")
    save_predictions(valid_complexes, predictions, output_path)
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  CASF-2016 R = {results['R']:.4f}")
    print(f"  CASF-2016 ρ = {results['rho']:.4f}")
    print(f"  RMSE = {results['RMSE']:.4f} pKd, MAE = {results['MAE']:.4f} pKd")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    main()