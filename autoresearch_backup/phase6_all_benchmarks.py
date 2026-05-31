"""Save best model (t2000) and eval on all CASF benchmarks"""
import pickle, numpy as np
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
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

X = np.load(BACKUP / 'phase2_X.npy')
y = np.load(BACKUP / 'phase2_y.npy')
pdb_ids = pickle.load(open(BACKUP / 'phase2_pdb_ids.pkl', 'rb'))
pocket_full = pickle.load(open(BACKUP / 'pocket_features_full.pkl', 'rb'))

X_aug = np.zeros((len(X), 1032), dtype=np.float32)
X_aug[:, :982] = X
for i, pid in enumerate(pdb_ids):
    if pid in pocket_full:
        X_aug[i, 982:] = pocket_full[pid]

# Train final best model: k=500, n_estimators=2000, depth=12
print('Training t2000 model (k=500, trees=2000)...', flush=True)
s = StandardScaler()
Xs = s.fit_transform(X_aug)
sel = SelectKBest(f_regression, k=500)
X_sel = sel.fit_transform(Xs, y)
m = xgb.XGBRegressor(max_depth=12, n_estimators=2000, learning_rate=0.01,
                      subsample=0.8, colsample_bytree=0.8,
                      min_child_weight=3, gamma=0.1,
                      reg_alpha=0.5, reg_lambda=2.0,
                      random_state=42, n_jobs=-1, verbosity=0)
m.fit(X_sel, y)

# Save
model_data = {'model': m, 'scaler': s, 'selector': sel,
              'config': 'k=500,n_estimators=2000,depth=12',
              'n_samples': len(y), 'n_features': 1032, 'k': 500}
pickle.dump(model_data, open(BACKUP / 'geock_final_best.pkl', 'wb'))
print('Saved final best model', flush=True)

# ===== Helper functions =====
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
            try:
                mol = Chem.MolFromMol2File(str(p), removeHs=False)
                if mol: return mol
            except: pass
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
            try: pocket = extract_simple_pocket(str(pp)); break
            except: pass
    return np.concatenate([lf, pocket])

def eval_set(name, pdb_list, base_dir):
    pr, tr = [], []
    skipped = 0
    for cx in pdb_list:
        pid = cx['pdb_id']
        yt = cx['pkd_true']
        mol = mol_from_pid(pid, base_dir)
        if mol is None:
            skipped += 1
            continue
        f = feats_from_mol(mol, pid, base_dir).reshape(1, -1)
        fs = s.transform(f); fs = sel.transform(fs)
        pr.append(float(m.predict(fs)[0]))
        tr.append(float(yt))
    tr = np.array(tr); pr = np.array(pr)
    r, _ = pearsonr(tr, pr)
    rho, _ = spearmanr(tr, pr)
    mae = np.mean(np.abs(tr - pr))
    rmse = np.sqrt(np.mean((tr - pr)**2))
    print(f'  {name}: R={r:.4f} Sp={rho:.4f} MAE={mae:.2f} RMSE={rmse:.2f} (N={len(tr)}, {skipped} skipped)')
    return {'r': r, 'rho': rho, 'mae': mae, 'rmse': rmse, 'n': len(tr)}

results = {}

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
print(f'{len(casf07)} entries in index')
results['CASF-2007'] = eval_set('CASF-2007', casf07, CASF07_DIR)

# ===== CASF-2013 =====
print('\n=== CASF-2013 ===')
with open(CASF13_DIR / 'coreset' / 'index' / '2013_core_data.lst') as f:
    casf13 = []
    for line in f:
        if line.startswith('#'): continue
        parts = line.strip().split()
        if len(parts) >= 4:
            try: casf13.append({'pdb_id': parts[0], 'pkd_true': float(parts[3])})
            except: pass
print(f'{len(casf13)} entries in index')
results['CASF-2013'] = eval_set('CASF-2013', casf13, CASF13_DIR)

# ===== CASF-2016 =====
print('\n=== CASF-2016 ===')
with open(CASF16_DIR / 'power_scoring' / 'CoreSet.dat') as f:
    f.readline()
    casf16 = [{'pdb_id': l.split()[0], 'pkd_true': float(l.split()[3])} for l in f if len(l.strip().split()) >= 4]
print(f'{len(casf16)} entries in index')
results['CASF-2016'] = eval_set('CASF-2016', casf16, CASF16_DIR)

print('\n========================================')
print('           FINAL BENCHMARK RESULTS        ')
print('========================================')
print(f'{"Benchmark":15s} {"R":8s} {"Sp":8s} {"MAE":6s} {"RMSE":6s} {"N":6s}')
print('-' * 55)
for name in ['CASF-2007', 'CASF-2013', 'CASF-2016']:
    r = results[name]
    print(f'{name:15s} {r["r"]:.4f}   {r["rho"]:.4f}  {r["mae"]:.2f}  {r["rmse"]:.2f}  {r["n"]:4d}')
print()
print(f'Best config: k=500, n_estimators=2000, max_depth=12, lr=0.01')
print(f'Features: ECFP4(512)+MACCS(167)+FCFP4(256)+RDKit(47)+Pocket(50) = 1032-dim')
