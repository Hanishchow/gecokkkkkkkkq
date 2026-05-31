#!/usr/bin/env python3
"""
CASF-2007 Validation Pipeline for GEOCK v2
=========================================
Validates GEOCK on CASF-2007 benchmark (195 protein-ligand complexes)

Usage:
    python casf2007_validation.py
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
import sys
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

CASF_DIR = Path("/mnt/c/Users/yakka/Downloads/CASF")
MODEL_PATH = Path("geock_deep_trees_final.pkl")
RESULTS_DIR = Path("/mnt/c/Users/yakka/Desktop/CASF_Results")
RESULTS_DIR.mkdir(exist_ok=True)

# Feature extraction (must match training!)
FP_RADIUS = 2  # ECFP4
FP_NBITS = 512

# ============================================================================
# STEP 1: PARSE CASF-2007 INDEX
# ============================================================================

def parse_casf_index(index_file):
    """Parse PDBbind_core_set_v2007.2.lst"""
    print("\n" + "="*70)
    print("STEP 1: Parsing CASF-2007 INDEX")
    print("="*70)
    
    data = []
    with open(index_file, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if len(parts) < 4:
                continue
            
            pdb_id = parts[0].lower()
            
            # Parse affinity (can be pKd or pKi)
            try:
                pkd = float(parts[3])
            except ValueError:
                continue
            
            # Parse unit (Kd or Ki)
            unit_str = parts[4] if len(parts) > 4 else ""
            
            data.append({
                'pdb_id': pdb_id,
                'pkd': pkd,
                'unit': unit_str
            })
    
    df = pd.DataFrame(data)
    print(f"✅ Parsed {len(df)} complexes")
    print(f"   pKd range: {df['pkd'].min():.2f} - {df['pkd'].max():.2f}")
    
    return df

# ============================================================================
# STEP 2: EXTRACT SMILES FROM SDF
# ============================================================================

def extract_smiles(sdf_file):
    """Extract SMILES from SDF file"""
    try:
        suppl = Chem.SDMolSupplier(str(sdf_file), sanitize=False, removeHs=True)
        mol = suppl[0]
        if mol is not None:
            return Chem.MolToSmiles(mol)
    except Exception as e:
        return None
    return None

def process_ligands(affinity_df):
    """Extract SMILES for all complexes"""
    print("\n" + "="*70)
    print("STEP 2: Extracting SMILES from Ligands")
    print("="*70)
    
    ligand_dir = CASF_DIR / "ligand" / "ranking_scoring" / "crystal_sdf"
    
    results = []
    failed = []
    
    for _, row in affinity_df.iterrows():
        pdb_id = row['pdb_id']
        sdf_file = ligand_dir / f"{pdb_id}_ligand.sdf"
        
        if not sdf_file.exists():
            failed.append(pdb_id)
            continue
        
        smiles = extract_smiles(sdf_file)
        
        if smiles:
            results.append({
                'pdb_id': pdb_id,
                'smiles': smiles,
                'pkd_true': row['pkd']
            })
        else:
            failed.append(pdb_id)
    
    df = pd.DataFrame(results)
    
    print(f"✅ Successfully extracted: {len(df)}")
    print(f"❌ Failed: {len(failed)}")
    
    if failed:
        print(f"   Failed PDBs: {', '.join(failed[:10])}...")
    
    return df

# ============================================================================
# STEP 3: GENERATE FINGERPRINTS
# ============================================================================

def smiles_to_ecfp(smiles, radius=FP_RADIUS, n_bits=FP_NBITS):
    """Convert SMILES to ECFP4 fingerprint"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
        return np.array(fp, dtype=np.uint8)
    except:
        return None

def generate_fingerprints(smiles_df):
    """Generate ECFP4 fingerprints"""
    print("\n" + "="*70)
    print("STEP 3: Generating ECFP4 Fingerprints")
    print("="*70)
    
    X = []
    valid_indices = []
    failed = []
    
    for idx, row in smiles_df.iterrows():
        fp = smiles_to_ecfp(row['smiles'])
        
        if fp is not None:
            X.append(fp)
            valid_indices.append(idx)
        else:
            failed.append(row['pdb_id'])
    
    X = np.array(X)
    df_valid = smiles_df.iloc[valid_indices].reset_index(drop=True)
    
    print(f"✅ Generated fingerprints: {len(X)}")
    print(f"❌ Failed: {len(failed)}")
    
    return X, df_valid

# ============================================================================
# STEP 4: LOAD MODEL
# ============================================================================

def load_model():
    """Load trained GEOCK model"""
    print("\n" + "="*70)
    print("STEP 4: Loading Model")
    print("="*70)
    
    if not MODEL_PATH.exists():
        print(f"❌ ERROR: Model not found at {MODEL_PATH}")
        sys.exit(1)
    
    print(f"Loading: {MODEL_PATH}")
    
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    
    print(f"✅ Model loaded: {model.get('model_type', 'unknown')}")
    print(f"   CV R: {model.get('cv_r', 'N/A')}")
    
    return model

# ============================================================================
# STEP 5: RUN PREDICTIONS
# ============================================================================

def run_predictions(model, X, df):
    """Run GEOCK predictions"""
    print("\n" + "="*70)
    print("STEP 5: Running Predictions")
    print("="*70)
    
    # Apply scaler
    X_scaled = model['scaler'].transform(X)
    
    # Apply selector
    X_selected = model['selector'].transform(X_scaled)
    
    # Predict
    y_pred = model['model'].predict(X_selected)
    
    df = df.copy()
    df['pkd_pred'] = y_pred
    df['error'] = np.abs(df['pkd_true'] - df['pkd_true'])  # Fixed: use pkd_true
    df['error'] = np.abs(df['pkd_true'] - df['pkd_pred'])
    df['residual'] = df['pkd_pred'] - df['pkd_true']
    
    print(f"✅ Predictions completed: {len(df)}")
    print(f"   Predicted range: {y_pred.min():.2f} - {y_pred.max():.2f}")
    
    return df

# ============================================================================
# STEP 6: CALCULATE METRICS
# ============================================================================

def calculate_metrics(df):
    """Calculate validation metrics"""
    print("\n" + "="*70)
    print("STEP 6: Calculating Metrics")
    print("="*70)
    
    y_true = df['pkd_true'].values
    y_pred = df['pkd_pred'].values
    
    # Correlations
    r_pearson, p_pearson = pearsonr(y_true, y_pred)
    r_spearman, p_spearman = spearmanr(y_true, y_pred)
    
    # Errors
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Practical metrics
    within_1 = (np.abs(y_true - y_pred) <= 1.0).sum() / len(y_true) * 100
    within_2 = (np.abs(y_true - y_pred) <= 2.0).sum() / len(y_true) * 100
    extreme = (np.abs(y_true - y_pred) > 3.0).sum() / len(y_true) * 100
    
    metrics = {
        'n_samples': len(df),
        'r_pearson': r_pearson,
        'r_spearman': r_spearman,
        'mae': mae,
        'rmse': rmse,
        'within_1': within_1,
        'within_2': within_2,
        'extreme': extreme
    }
    
    return metrics

# ============================================================================
# STEP 7: ERROR ANALYSIS
# ============================================================================

def analyze_errors(df):
    """Analyze prediction errors"""
    print("\n" + "="*70)
    print("STEP 7: Error Analysis")
    print("="*70)
    
    # By affinity range
    print("\nError by Affinity Range:")
    print("-"*50)
    
    ranges = [
        (0, 5, 'Very Weak (<5)'),
        (5, 7, 'Weak (5-7)'),
        (7, 9, 'Moderate (7-9)'),
        (9, 12, 'Strong (9-12)'),
        (12, 20, 'Very Strong (>12)')
    ]
    
    error_by_range = []
    for lo, hi, label in ranges:
        mask = (df['pkd_true'] >= lo) & (df['pkd_true'] < hi)
        if mask.sum() > 0:
            mae = df.loc[mask, 'error'].mean()
            bias = df.loc[mask, 'residual'].mean()
            n = mask.sum()
            print(f"{label:20s}: n={n:3d}, MAE={mae:.2f}, Bias={bias:+.2f}")
            error_by_range.append({'range': label, 'n': n, 'mae': mae, 'bias': bias})
    
    # Worst predictions
    print("\nWorst 10 Predictions:")
    print("-"*50)
    worst = df.nlargest(10, 'error')
    for _, row in worst.iterrows():
        print(f"{row['pdb_id']:8s}: True={row['pkd_true']:.2f}, Pred={row['pkd_pred']:.2f}, Error={row['error']:.2f}")
    
    return error_by_range

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    print("\n" + "="*70)
    print("  CASF-2007 VALIDATION FOR GEOCK v2")
    print("="*70)
    
    # Step 1: Parse INDEX
    index_file = CASF_DIR / "PDBbind_core_set_v2007.2.lst"
    affinity_df = parse_casf_index(index_file)
    
    # Step 2: Extract SMILES
    smiles_df = process_ligands(affinity_df)
    
    if len(smiles_df) == 0:
        print("\n❌ FATAL: No ligands processed")
        sys.exit(1)
    
    # Step 3: Generate fingerprints
    X, valid_df = generate_fingerprints(smiles_df)
    
    # Step 4: Load model
    model = load_model()
    
    # Step 5: Predict
    results_df = run_predictions(model, X, valid_df)
    
    # Step 6: Calculate metrics
    metrics = calculate_metrics(results_df)
    
    # Step 7: Error analysis
    error_analysis = analyze_errors(results_df)
    
    # =========================================================================
    # FINAL REPORT
    # =========================================================================
    print("\n" + "="*70)
    print("  VALIDATION RESULTS")
    print("="*70)
    
    report = f"""
╔══════════════════════════════════════════════════════════════════════╗
║                    CASF-2007 BENCHMARK RESULTS                      ║
║                        GEOCK v2                                      ║
╚══════════════════════════════════════════════════════════════════════╝

📊 DATASET
────────────────────────────────────────────────────────────────────
  Complexes:      {metrics['n_samples']}
  pKd range:      {results_df['pkd_true'].min():.2f} - {results_df['pkd_true'].max():.2f}

🎯 SCORING POWER (Primary Metrics)
────────────────────────────────────────────────────────────────────
  Pearson R:       {metrics['r_pearson']:.4f}
  Spearman ρ:      {metrics['r_spearman']:.4f}

📏 ERROR METRICS
────────────────────────────────────────────────────────────────────
  MAE:            {metrics['mae']:.3f} pKd
  RMSE:           {metrics['rmse']:.3f} pKd

✅ PRACTICAL PERFORMANCE
────────────────────────────────────────────────────────────────────
  Within 1 pKd:   {metrics['within_1']:.1f}%
  Within 2 pKd:    {metrics['within_2']:.1f}%
  Extreme errors:  {metrics['extreme']:.1f}% (>3 pKd)

"""
    
    # Comparison with published methods
    report += """
📊 COMPARISON WITH PUBLISHED METHODS (CASF-2007)
────────────────────────────────────────────────────────────────────
  Method            │ Year │ R      │ MAE
  ──────────────────┼──────┼────────┼─────
  X-Score          │ 2002 │ 0.58   │ 1.7
  AutoDock Vina    │ 2010 │ 0.64   │ 1.5
  RF-Score         │ 2015 │ 0.69   │ 1.3
  Pafnucy          │ 2017 │ 0.74   │ 1.2
  GEOCK v2         │ 2026 │ {r:.4f} │ {mae:.2f}
────────────────────────────────────────────────────────────────────
"""
    
    report = report.format(r=metrics['r_pearson'], mae=metrics['mae'])
    
    # Publication readiness
    if metrics['r_pearson'] >= 0.80:
        report += """
✅ PUBLICATION READY
────────────────────────────────────────────────────────────────────
  R ≥ 0.80 → JCIM submission ready
  
  Target: Journal of Chemical Information and Modeling
  Impact: State-of-the-art on CASF-2007
"""
    elif metrics['r_pearson'] >= 0.75:
        report += """
⚠️  MODERATE PERFORMANCE
────────────────────────────────────────────────────────────────────
  0.75 ≤ R < 0.80 → Competitive with published methods
  
  Target: JCIM or JCTC
"""
    else:
        report += """
⚠️  NEEDS IMPROVEMENT
────────────────────────────────────────────────────────────────────
  R < 0.75 → Below state-of-the-art
  
  Recommend: Add physics features or ensemble methods
"""
    
    print(report)
    
    # Save results
    results_file = RESULTS_DIR / "casf2007_predictions.csv"
    results_df.to_csv(results_file, index=False)
    print(f"\n📄 Predictions saved: {results_file}")
    
    # Save metrics
    import json
    metrics_file = RESULTS_DIR / "casf2007_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"📊 Metrics saved: {metrics_file}")
    
    return metrics, results_df

if __name__ == "__main__":
    metrics, results = main()
