"""Evaluate Phase 2 model on CASF-2016 (982-dim features, RDKit=zeros)"""
import pickle, numpy as np
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, rdFingerprintGenerator

BACKUP = Path(r'C:\Users\yakka\Downloads\Hanish_Brain_Vault\autoresearch_backup')
CASF_DIR = Path(r'C:\Users\yakka\Downloads\CASF-2016\CASF-2016')

# Load model
md = pickle.load(open(BACKUP / 'geock_phase2_982dim.pkl', 'rb'))
model, scaler, selector = md['model'], md['scaler'], md['selector']
print(f"Loaded model: CV R={md['cv_r']:.4f}, n_samples={md['n_samples']}")

# Load CASF
with open(CASF_DIR / 'power_scoring' / 'CoreSet.dat') as f:
    f.readline()
    casf = []
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 4:
            casf.append({'pdb_id': parts[0], 'pkd_true': float(parts[3])})

def get_mol(pid):
    mol2 = CASF_DIR / 'coreset' / pid / f'{pid}_ligand.mol2'
    sdf = CASF_DIR / 'coreset' / pid / f'{pid}_ligand.sdf'
    mol = Chem.MolFromMol2File(str(mol2), removeHs=False) if mol2.exists() else None
    if mol is None and sdf.exists():
        suppl = Chem.SDMolSupplier(str(sdf), removeHs=False)
        mol = next((m for m in suppl if m), None)
    return mol

def get_features_982(mol):
    ecfp = np.array(rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=512).GetFingerprintAsNumPy(mol), dtype=np.float32)
    maccs = np.array(MACCSkeys.GenMACCSKeys(mol), dtype=np.float32) if hasattr(MACCSkeys, 'GenMACCSKeys') else np.zeros(167, dtype=np.float32)
    fcfp = np.array(rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=256).GetFingerprintAsNumPy(mol), dtype=np.float32)
    rdkit = np.zeros(47, dtype=np.float32)
    return np.concatenate([ecfp, maccs, fcfp, rdkit])

preds, trues = [], []
for cx in casf:
    mol = get_mol(cx['pdb_id'])
    if mol is None:
        continue
    f = get_features_982(mol).reshape(1, -1)
    f = scaler.transform(f)
    f = selector.transform(f)
    preds.append(float(model.predict(f)[0]))
    trues.append(cx['pkd_true'])

preds = np.array(preds)
trues = np.array(trues)
r, _ = pearsonr(trues, preds)
rho, _ = spearmanr(trues, preds)
mae = np.mean(np.abs(trues - preds))
rmse = np.sqrt(np.mean((trues - preds)**2))

print(f"\nECFP4+MACCS+FCFP4 (982-dim, 19K training):")
print(f"  R={r:.4f} Sp={rho:.4f} MAE={mae:.2f} RMSE={rmse:.2f} (N={len(preds)})")
print(f"  vs ECFP-only XGBoost 39k: R=0.587")
print(f"  vs ECFP-only Deep Trees (39K): R=0.575")
