"""Save CASF-2016 predictions CSV from best model"""
import pickle, numpy as np
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator
import sys
sys.path.insert(0, r'C:\Users\yakka\Downloads\Hanish_Brain_Vault\autoresearch_backup')
from simple_pocket import extract_simple_pocket

BACKUP = Path(r'C:\Users\yakka\Downloads\Hanish_Brain_Vault\autoresearch_backup')
CASF16_DIR = Path(r'C:\Users\yakka\Downloads\CASF-2016\CASF-2016')

md = pickle.load(open(BACKUP / 'geock_final_best.pkl', 'rb'))
m = md['model']; s = md['scaler']; sel = md['selector']

with open(CASF16_DIR / 'power_scoring' / 'CoreSet.dat') as f:
    f.readline()
    casf = [{'pdb_id': l.split()[0], 'pkd_true': float(l.split()[3])} for l in f if len(l.strip().split()) >= 4]

rows = ['pdb_id,pkd_true,pkd_pred']
for cx in casf:
    m2 = CASF16_DIR / 'coreset' / cx['pdb_id'] / f"{cx['pdb_id']}_ligand.mol2"
    sd = CASF16_DIR / 'coreset' / cx['pdb_id'] / f"{cx['pdb_id']}_ligand.sdf"
    mol = Chem.MolFromMol2File(str(m2), removeHs=False) if m2.exists() else None
    if mol is None and sd.exists():
        mol = next((mm for mm in Chem.SDMolSupplier(str(sd), removeHs=False) if mm), None)
    if mol is None:
        continue
    ecfp = np.array(rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=512).GetFingerprintAsNumPy(mol), dtype=np.float32)
    maccs = np.array(MACCSkeys.GenMACCSKeys(mol), dtype=np.float32)
    fcfp = np.array(rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=256).GetFingerprintAsNumPy(mol), dtype=np.float32)
    lf = np.concatenate([ecfp, maccs, fcfp, np.zeros(47, dtype=np.float32)])
    pp = CASF16_DIR / 'coreset' / cx['pdb_id'] / f"{cx['pdb_id']}_pocket.pdb"
    pocket = extract_simple_pocket(str(pp)) if pp.exists() else np.zeros(50, dtype=np.float32)
    f = np.concatenate([lf, pocket]).reshape(1, -1)
    fs = s.transform(f)
    fs = sel.transform(fs)
    pred = float(m.predict(fs)[0])
    rows.append(f"{cx['pdb_id']},{cx['pkd_true']:.4f},{pred:.4f}")

with open(BACKUP / 'geock_final_casf2016_predictions.csv', 'w') as fp:
    fp.write('\n'.join(rows))
print(f'Written {len(rows)-1} predictions')
