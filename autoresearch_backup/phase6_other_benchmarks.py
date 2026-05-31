"""Phase 6: Evaluate best model on CASF-2007 and CASF-2013"""
import pickle, numpy as np
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
import warnings; warnings.filterwarnings('ignore')
from rdkit import Chem
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator
import sys
sys.path.insert(0, r'C:\Users\yakka\Downloads\Hanish_Brain_Vault\autoresearch_backup')
from simple_pocket import extract_simple_pocket

BACKUP = Path(r'C:\Users\yakka\Downloads\Hanish_Brain_Vault\autoresearch_backup')
CASF16_DIR = Path(r'C:\Users\yakka\Downloads\CASF-2016\CASF-2016')
CASF13_DIR = Path(r'C:\Users\yakka\Downloads\CASF-2013-updated\CASF-2013')
CASF07_DIR = Path(r'C:\Users\yakka\Downloads\CASF')

# Load best model
md = pickle.load(open(BACKUP / 'geock_phase5_best.pkl', 'rb'))
m = md['model']; s = md['scaler']; sel = md['selector']
print(f'Loaded best model: {md["config"]} (CASF-2016 R={md["casf16_r"]:.4f})')

def mol_from_pid(pid, base_dir):
    paths = [
        base_dir / 'coreset' / pid / f'{pid}_ligand.mol2',
        base_dir / 'coreset' / pid / f'{pid}_ligand.sdf',
        base_dir / 'coreset' / f'{pid}_ligand.mol2',
        base_dir / 'coreset' / f'{pid}_ligand.sdf',
        base_dir / 'ligand' / 'ranking_scoring' / 'crystal_mol2' / f'{pid}_ligand.mol2',
        base_dir / 'ligand' / 'ranking_scoring' / 'crystal_sdf' / f'{pid}_ligand.sdf',
    ]
    for p in paths:
        if p.suffix == '.mol2' and p.exists():
            mol = Chem.MolFromMol2File(str(p), removeHs=False)
            if mol: return mol
        elif p.suffix == '.sdf' and p.exists():
            try:
                mol = next((mm for mm in Chem.SDMolSupplier(str(p), removeHs=False) if mm), None)
                if mol: return mol
            except: pass
    return None

def feats_from_mol(mol, pid, base_dir):
    ecfp = np.array(rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=512).GetFingerprintAsNumPy(mol), dtype=np.float32)
    maccs = np.array(MACCSkeys.GenMACCSKeys(mol), dtype=np.float32)
    fcfp = np.array(rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=256).GetFingerprintAsNumPy(mol), dtype=np.float32)
    lf = np.concatenate([ecfp, maccs, fcfp, np.zeros(47, dtype=np.float32)])
    pocket_paths = [
        base_dir / 'coreset' / pid / f'{pid}_pocket.pdb',
        base_dir / 'coreset' / f'{pid}_pocket.pdb',
    ]
    pocket = np.zeros(50, dtype=np.float32)
    for pp in pocket_paths:
        if pp.exists():
            try:
                pocket = extract_simple_pocket(str(pp))
            except: pass
            break
    return np.concatenate([lf, pocket])

def eval_benchmark(pdb_list, name, base_dir):
    pr, tr = [], []
    skipped = 0
    for cx in pdb_list:
        pid = cx['pdb_id']
        mol = mol_from_pid(pid, base_dir)
        if mol is None:
            skipped += 1
            continue
        f = feats_from_mol(mol, pid, base_dir).reshape(1, -1)
        fs = s.transform(f); fs = sel.transform(fs)
        pr.append(float(m.predict(fs)[0]))
        tr.append(cx['pkd_true'])
    tr = np.array(tr); pr = np.array(pr)
    r, _ = pearsonr(tr, pr)
    rho, _ = spearmanr(tr, pr)
    mae = np.mean(np.abs(tr - pr))
    rmse = np.sqrt(np.mean((tr - pr)**2))
    print(f'{name}: R={r:.4f} Sp={rho:.4f} MAE={mae:.2f} RMSE={rmse:.2f} (N={len(tr)}, skipped={skipped})')
    return r, rho

# ===== CASF-2016 (re-eval for consistency) =====
print('\n=== CASF-2016 ===')
with open(CASF16_DIR / 'power_scoring' / 'CoreSet.dat') as f:
    f.readline()
    casf16 = [{'pdb_id': l.split()[0], 'pkd_true': float(l.split()[3])} for l in f if len(l.strip().split()) >= 4]
eval_benchmark(casf16, 'CASF-2016', CASF16_DIR)

# ===== CASF-2013 =====
print('\n=== CASF-2013 ===')
with open(CASF13_DIR / 'coreset' / 'index' / '2013_core_data.lst') as f:
    casf13 = []
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 3:
            try: casf13.append({'pdb_id': parts[0], 'pkd_true': float(parts[2])})
            except: pass
print(f'{len(casf13)} entries')
r13 = eval_benchmark(casf13, 'CASF-2013', CASF13_DIR)

# ===== CASF-2007 =====
print('\n=== CASF-2007 ===')
with open(CASF07_DIR / 'PDBbind_core_set_v2007.2.lst') as f:
    casf07 = []
    for line in f:
        if line.startswith('#') or not line.strip(): continue
        parts = line.strip().split()
        if len(parts) >= 4:
            try: casf07.append({'pdb_id': parts[0], 'pkd_true': float(parts[3])})
            except: pass
print(f'{len(casf07)} entries')
r07 = eval_benchmark(casf07, 'CASF-2007', CASF07_DIR)

print('\n=== FINAL BENCHMARKS ===')
print(f'  CASF-2007: R={r07}')
print(f'  CASF-2013: R={r13}')
print(f'  CASF-2016: R={md["casf16_r"]:.4f}')
