"""
GEOCK v2 - Binding Affinity Predictor
Best model: CV R² = 0.8437

Usage:
    python predict_simple.py "CCO"  # SMILES -> pKD prediction
"""
import pickle
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

MODEL_PATH = "geock_deep_trees_no2016.pkl"

def load_model(path=MODEL_PATH):
    with open(path, 'rb') as f:
        return pickle.load(f)

def smiles_to_features(smiles):
    """Convert SMILES to 512-bit Morgan fingerprint (same as training)"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    
    # Morgan fingerprint (512 bits, radius 2)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=512)
    return np.array(fp, dtype=np.float32).reshape(1, -1)

def predict_affinity(smiles, model_data=None):
    if model_data is None:
        model_data = load_model()
    
    X = smiles_to_features(smiles)
    
    # Scale
    X_scaled = model_data['scaler'].transform(X)
    
    # Select top k features
    X_selected = model_data['selector'].transform(X_scaled)
    
    # Predict
    pKD = model_data['model'].predict(X_selected)[0]
    
    # Convert to Kd
    Kd_nM = 10 ** (-pKD) * 1e9
    return {'pKD': pKD, 'Kd_nM': Kd_nM}

if __name__ == "__main__":
    import sys
    smiles = sys.argv[1] if len(sys.argv) > 1 else "CCO"
    
    print(f"Loading model: {MODEL_PATH}")
    model = load_model()
    print(f"Model CV R²: {model['cv_r']:.4f}")
    print(f"Features: {model['n_features']} (selected from {model['scaler'].n_features_in_})")
    
    result = predict_affinity(smiles, model)
    print(f"\nSMILES: {smiles}")
    print(f"pKD: {result['pKD']:.2f}")
    print(f"Kd: {result['Kd_nM']:.2f} nM")