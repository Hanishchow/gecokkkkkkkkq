#!/usr/bin/env python3
"""
Enhanced Feature Extraction for GEOCK
===================================
Extract comprehensive molecular features:
- ECFP4 (512 bits) - Morgan fingerprints
- MACCS keys (167 bits) - Predefined structural patterns
- FCFP4 (256 bits) - Functional connectivity fingerprints
- RDKit descriptors (~200 features) - Molecular properties
- Combined: ~1135 features total

Usage:
    python extract_features_v2.py              # Extract all features
    python extract_features_v2.py --limit 100   # Test with 100 records
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, MACCSkeys, rdMolDescriptors
from rdkit.Chem import rdFingerprintGenerator
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import argparse
import time

CACHE_DIR = Path("/home/chow/.cache/geock_autoresearch")
OUTPUT_PATH = CACHE_DIR / "lp_features_enhanced.pkl"


def get_maccs_fingerprint(mol):
    """MACCS keys - 167 predefined structural keys."""
    try:
        keys = MACCSkeys.GenMACCSKeys(mol)
        return np.array(keys, dtype=np.float32)
    except:
        return np.zeros(167, dtype=np.float32)


def get_fcfp_fingerprint(mol, fp_size=256):
    """FCFP4 - Functional class fingerprints (radius=3)."""
    try:
        fcfp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=fp_size)
        return np.array(fcfp_gen.GetFingerprintAsNumPy(mol), dtype=np.float32)
    except:
        return np.zeros(fp_size, dtype=np.float32)


def get_rdkit_descriptors(mol):
    """Comprehensive RDKit molecular descriptors."""
    try:
        desc = [
            # Basic properties
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.NumHeteroatoms(mol),
            Descriptors.FractionCSP3(mol),
            
            # Ring and aromatic
            Lipinski.RingCount(mol),
            Lipinski.NumAromaticRings(mol),
            Lipinski.NumAliphaticRings(mol),
            Lipinski.NumSaturatedRings(mol),
            
            # Structural counts
            rdMolDescriptors.calcNumRings(mol),
            rdMolDescriptors.calcNumHeterocycles(mol),
            rdMolDescriptors.calcNumAmideBonds(mol),
            rdMolDescriptors.calcNumBridgeheadAtoms(mol),
            rdMolDescriptors.calcNumSpiroAtoms(mol),
            
            # Electronic properties
            Descriptors.MaxPartialCharge(mol),
            Descriptors.MinPartialCharge(mol),
            Descriptors.MaxAbsPartialCharge(mol),
            Descriptors.MinAbsPartialCharge(mol),
            
            # Polar surface area variants
            Descriptors.NumValenceElectrons(mol),
            Descriptors.NumRadicalElectrons(mol),
            
            # More descriptors
            Descriptors.HeavyAtomMolWt(mol),
            Descriptors.MaxAbsEStateIndex(mol),
            Descriptors.MinAbsEStateIndex(mol),
            Descriptors.MaxEStateIndex(mol),
            Descriptors.MinEStateIndex(mol),
            
            # Solubility related
            Descriptors.MolLogS(mol),
            Descriptors.MolMR(mol),
            
            # Shape descriptors
            rdMolDescriptors.CalcNumRotatableBonds(mol),
            rdMolDescriptors.CalcNumAmideBonds(mol),
            
            # Atom counts
            mol.GetNumAtoms(),
            mol.GetNumHeavyAtoms(),
            mol.GetNumHeteroatoms(),
            
            # Conformer
            mol.GetNumConformers(),
            
            # Additional Lipinski
            Lipinski.NumAromaticHeterocycles(mol),
            Lipinski.NumAromaticCarbocycles(mol),
            Lipinski.NumAliphaticHeterocycles(mol),
            Lipinski.NumAliphaticCarbocycles(mol),
            Lipinski.NumSaturatedHeterocycles(mol),
            Lipinski.NumSaturatedCarbocycles(mol),
            Lipinski.NumHAcceptors(mol),
            Lipinski.NumHDonors(mol),
            Lipinski.NumHeteroatoms(mol),
            
            # More complex descriptors
            rdMolDescriptors.CalcNumRotatableBonds(mol, strict=True),
            rdMolDescriptors.CalcNumHeavyAtoms(mol),
            rdMolDescriptors.CalcNumBridgeheadAtoms(mol),
        ]
        
        # Replace NaN/inf with 0
        desc = [0 if (np.isnan(x) or np.isinf(x)) else float(x) for x in desc]
        return np.array(desc, dtype=np.float32)
    except Exception as e:
        return np.zeros(47, dtype=np.float32)


def extract_features(smiles):
    """Extract all features from a SMILES string."""
    if not smiles or pd.isna(smiles):
        return None
    
    try:
        smiles = str(smiles).strip()
        if len(smiles) < 3:
            return None
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # ECFP4 (radius=2, 512 bits)
        ecfp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=512)
        ecfp = np.array(ecfp_gen.GetFingerprintAsNumPy(mol), dtype=np.float32)
        
        # MACCS keys (167 bits)
        maccs = get_maccs_fingerprint(mol)
        
        # FCFP4 (radius=3, 256 bits)
        fcfp = get_fcfp_fingerprint(mol, fp_size=256)
        
        # RDKit descriptors
        rdkit = get_rdkit_descriptors(mol)
        
        # Combine all features
        features = np.concatenate([ecfp, maccs, fcfp, rdkit])
        
        return {
            'features': features,
            'smiles': smiles,
            'ecfp': ecfp,
            'maccs': maccs,
            'fcfp': fcfp,
            'rdkit': rdkit,
        }
    except Exception as e:
        return None


def process_record(record, smiles_col='smiles'):
    """Process a single record from the dataset."""
    smiles = record.get(smiles_col)
    if not smiles:
        return None
    
    result = extract_features(smiles)
    if result:
        result['affinity'] = record.get('affinity', record.get('value'))
        result['pdb_id'] = record.get('pdb_id', record.get('Unnamed: 0', ''))
        result['source'] = record.get('source', 'lp_pdbbind')
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Enhanced feature extraction")
    parser.add_argument('--limit', type=int, default=None, help="Limit number of records")
    parser.add_argument('--workers', type=int, default=mp.cpu_count() - 1, help="Number of workers")
    parser.add_argument('--output', type=str, default=str(OUTPUT_PATH), help="Output path")
    args = parser.parse_args()
    
    print("=" * 60)
    print("ENHANCED FEATURE EXTRACTION")
    print("=" * 60)
    print(f"Workers: {args.workers}")
    print()
    
    # Load LP-PDBBind data
    print("Loading LP-PDBBind data...")
    lp_df = pd.read_csv(CACHE_DIR / "LP_PDBBind.csv")
    lp_records = []
    for _, row in lp_df.iterrows():
        lp_records.append({
            'smiles': row['smiles'],
            'affinity': row['value'],
            'pdb_id': row['Unnamed: 0'],
            'source': 'lp_pdbbind'
        })
    print(f"LP-PDBBind: {len(lp_records)} records")
    
    # Load existing enhanced data
    existing_data = []
    existing_path = CACHE_DIR / "lp_features_enhanced.pkl"
    if existing_path.exists():
        try:
            with open(existing_path, 'rb') as f:
                existing_data = pickle.load(f)
            print(f"Existing enhanced: {len(existing_data)} records")
        except:
            pass
    
    # Track what's already processed
    existing_pdb_ids = {r['pdb_id'] for r in existing_data if r.get('pdb_id')}
    
    # Filter out already processed
    to_process = [r for r in lp_records if r['pdb_id'] not in existing_pdb_ids]
    print(f"To process: {len(to_process)} new records")
    
    if args.limit:
        to_process = to_process[:args.limit]
        print(f"Limit applied: {len(to_process)} records")
    
    if not to_process:
        print("\nAll records already processed!")
        print(f"Total enhanced records: {len(existing_data)}")
        return
    
    # Process in parallel
    print(f"\nExtracting features for {len(to_process)} records...")
    start_time = time.time()
    
    all_results = list(existing_data)
    processed = 0
    failed = 0
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_record, r): r for r in to_process}
        
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                all_results.append(result)
                processed += 1
            else:
                failed += 1
            
            if (processed + failed) % 1000 == 0:
                elapsed = time.time() - start_time
                rate = (processed + failed) / elapsed
                remaining = len(to_process) - processed - failed
                eta = remaining / rate if rate > 0 else 0
                print(f"  {processed + failed}/{len(to_process)} | "
                      f"OK: {processed} | FAIL: {failed} | "
                      f"ETA: {eta/60:.1f} min")
    
    elapsed = time.time() - start_time
    
    # Save results
    print(f"\nSaving {len(all_results)} records to {args.output}...")
    with open(args.output, 'wb') as f:
        pickle.dump(all_results, f)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Processed: {processed}")
    print(f"Failed: {failed}")
    print(f"Success rate: {100*processed/(processed+failed):.1f}%")
    print(f"Total records: {len(all_results)}")
    print(f"Time: {elapsed/60:.1f} minutes")
    
    # Feature dimensions
    if all_results and 'features' in all_results[0]:
        feat = all_results[0]['features']
        print(f"\nFeature dimensions:")
        print(f"  Total: {len(feat)}")
        print(f"  ECFP: {len(all_results[0]['ecfp'])}")
        print(f"  MACCS: {len(all_results[0]['maccs'])}")
        print(f"  FCFP: {len(all_results[0]['fcfp'])}")
        print(f"  RDKit: {len(all_results[0]['rdkit'])}")
    
    print(f"\nOutput: {args.output}")


if __name__ == "__main__":
    main()
