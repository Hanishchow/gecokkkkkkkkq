#!/usr/bin/env python3
"""
GEOCK Binding Affinity Predictor - Terminal Interface
=====================================================
"""

import sys
import pickle
import numpy as np

# Load model
print("Loading GEOCK model...", end=" ")
sys.stdout.flush()

try:
    with open('WORK_DIR / geock_model_final.pkl', 'rb') as f:
        model_data = pickle.load(f)
    xgb_model = model_data['xgb']
    feature_sel = model_data['sel']
    print("✓")
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

def predict_affinity(smiles):
    """Predict binding affinity from SMILES."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, "Invalid SMILES"
        
        # Generate ECFP
        ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=512)
        ecfp_arr = np.array(ecfp)
        
        # Transform and predict
        X = ecfp_arr.reshape(1, -1)
        X_sel = feature_sel.transform(X)
        pred = xgb_model.predict(X_sel)[0]
        
        return pred, None
    except ImportError:
        return None, "RDKit not installed"
    except Exception as e:
        return None, str(e)

def main():
    print("\n" + "="*60)
    print("  GEOCK - Binding Affinity Predictor")
    print("  " + "-"*56)
    print("  Model: XGBoost + ECFP (400 features)")
    print("  Expected R: ~0.71 (with PDB) / ~0.45 (SMILES only)")
    print("="*60)
    print()
    print("Enter ligand SMILES to predict binding affinity (pKd)")
    print("Type 'help' for examples, 'quit' to exit")
    print()
    
    while True:
        try:
            smiles = input("\n> ").strip()
        except EOFError:
            break
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        
        if not smiles:
            continue
        
        smiles = smiles.lower()
        
        if smiles in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if smiles == 'help':
            print("\n  Examples:")
            print("    Cc1ccc(cc1)C(=O)O                    # Aspirin")
            print("    CC(C)Cc1ccc(cc1)C(C)C(=O)O          # Ibuprofen")
            print("    CN1C=NC2=C1C(=O)N(C(=O)N2C)C       # Caffeine")
            print()
            continue
        
        pred, err = predict_affinity(smiles)
        
        if err:
            print(f"  Error: {err}")
        else:
            print(f"  Predicted pKd: {pred:.2f}")
            print(f"  Estimated Kd: {10**(-pred)*1e9:.2f} nM")

if __name__ == "__main__":
    main()
