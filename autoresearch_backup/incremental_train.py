#!/usr/bin/env python3
"""
Incremental Training Pipeline for GEOCK
Trains in chunks, saves checkpoint each iteration
"""
import pickle
import numpy as np
import os
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, KFold
import warnings
warnings.filterwarnings('ignore')

CACHE = 'CACHE_DIR / '
MODEL_DIR = 'WORK_DIR / '

def get_ecfp(smiles, radius=2, nBits=512):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    return np.array(fp, dtype=np.float32)

def get_physics(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        f = [
            Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.TPSA(mol),
            Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol),
            Descriptors.NumRotatableBonds(mol), Descriptors.HeavyAtomCount(mol),
            Descriptors.NumHeteroatoms(mol), Descriptors.RingCount(mol),
            Descriptors.NumAromaticRings(mol), Descriptors.FractionCSP3(mol),
            Descriptors.NumAliphaticCarbocycles(mol) + Descriptors.NumAliphaticHeterocycles(mol),
            Descriptors.LabuteASA(mol), Descriptors.Kappa1(mol), Descriptors.Kappa2(mol),
            Descriptors.Kappa3(mol), Descriptors.Chi0(mol), Descriptors.Chi1(mol),
            Descriptors.HallKierAlpha(mol), Descriptors.NOCount(mol), Descriptors.NHOHCount(mol),
        ]
        f = [0 if np.isnan(x) or np.isinf(x) else x for x in f]
        return np.array(f, dtype=np.float32)
    except:
        return None

def load_all_data():
    """Load and deduplicate all training data"""
    with open(CACHE + 'lp_new_features_8k.pkl', 'rb') as f:
        data1 = pickle.load(f)
    with open(CACHE + 'geock_training_data.pkl', 'rb') as f:
        data2 = pickle.load(f)
    
    all_data = data1 + data2
    unique = {}
    for d in all_data:
        pid = d['pdb_id']
        if pid not in unique:
            unique[pid] = d
    
    return list(unique.values())

def extract_features(data_list):
    """Extract features efficiently"""
    X_ecfp, X_phys, y = [], [], []
    
    for d in data_list:
        fp = get_ecfp(d['smiles'])
        phys = get_physics(d['smiles'])
        
        if fp is not None and phys is not None:
            if not np.any(np.isnan(phys)) and not np.any(np.isinf(phys)):
                X_ecfp.append(fp)
                X_phys.append(phys)
                y.append(d['affinity'])
    
    X_ecfp = np.array(X_ecfp)
    X_phys = np.array(X_phys)
    y = np.array(y)
    
    scaler = StandardScaler()
    X_phys_s = scaler.fit_transform(X_phys)
    X = np.hstack([X_ecfp, X_phys_s])
    
    return X, y, scaler

def train_chunk(data_list, chunk_name):
    """Train on a chunk of data"""
    print(f"\n=== Training: {chunk_name} ===")
    
    X, y, scaler = extract_features(data_list)
    print(f"  Samples: {len(X):,}, Features: {X.shape[1]}")
    print(f"  pKd: {y.min():.2f} - {y.max():.2f}")
    
    # Train model
    model = XGBRegressor(
        n_estimators=300,
        max_depth=10,
        learning_rate=0.05,
        reg_lambda=2.0,
        reg_alpha=0.5,
        random_state=42,
        n_jobs=-1
    )
    
    # CV
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
    print(f"  CV R²: {scores.mean():.4f} ± {scores.std():.4f}")
    
    # Train final
    model.fit(X, y)
    
    # Save checkpoint
    output = {
        'model': model,
        'scaler': scaler,
        'cv_r': float(scores.mean()),
        'cv_std': float(scores.std()),
        'n_samples': len(X),
        'chunk': chunk_name,
        'date': '2026-04-19'
    }
    
    path = f"{MODEL_DIR}/geock_chunk_{chunk_name}.pkl"
    with open(path, 'wb') as f:
        pickle.dump(output, f)
    
    print(f"  ✓ Saved: {path}")
    return output

def main():
    print("=== GEOCK Incremental Training ===\n")
    
    # Load all data
    print("Loading data...")
    all_data = load_all_data()
    print(f"Total unique: {len(all_data):,}")
    
    # Full training
    result = train_chunk(all_data, "full")
    
    print("\n=== Complete ===")
    print(f"CV R²: {result['cv_r']:.4f}")
    print(f"Samples: {result['n_samples']:,}")

if __name__ == '__main__':
    main()