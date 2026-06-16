"""Deduplicate 24K data + train final model + evaluate all benchmarks"""
import pickle, numpy as np
from pathlib import Path
from collections import defaultdict
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
import xgboost as xgb
import warnings; warnings.filterwarnings('ignore')
from rdkit import Chem
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator
import sys
sys.path.insert(0, r'C:\Users\yakka\Downloads\Hanish_Brain_Vault\autoresearch_backup')
from simple_pocket import extract_simple_pocket

BACKUP = Path(r'C:\Users\yakka\Downloads\Hanish_Brain_Vault\autoresearch_backup')

# Load 24K data
X = np.load(BACKUP / 'all24k_X.npy')
y = np.load(BACKUP / 'all24k_y.npy')
pdb_ids = pickle.load(open(BACKUP / 'all24k_pdb_ids.pkl', 'rb'))
pocket = pickle.load(open(BACKUP / 'pocket_features_full.pkl', 'rb'))

# Build 1032-dim with pocket
X_pocket = np.zeros((len(X), 1032), dtype=np.float32)
X_pocket[:, :982] = X
pocket_count = 0
for i, pid in enumerate(pdb_ids):
    if pid in pocket:
        X_pocket[i, 982:] = pocket[pid]
        pocket_count += 1

print(f'Loaded: {len(X)} entries, {pocket_count} with pocket', flush=True)

# Deduplicate by ECFP
ecfp_hashes = [hash(row.tobytes()) for row in X[:, :512]]
uniq = defaultdict(list)
for i, h in enumerate(ecfp_hashes):
    uniq[h].append(i)

Xd = np.zeros((len(uniq), X_pocket.shape[1]), dtype=np.float32)
yd = np.zeros(len(uniq), dtype=np.float32)
has_pocket = np.zeros(len(uniq), dtype=bool)
for j, (h, idxs) in enumerate(uniq.items()):
    Xd[j] = X_pocket[idxs[0]]
    yd[j] = np.median(y[idxs])
    has_pocket[j] = any(pdb_ids[i] in pocket for i in idxs)

print(f'Deduplicated: {len(X)} -> {len(Xd)}, {has_pocket.sum()} with pocket', flush=True)

# CV on dedup 1032-dim
print('\n=== 5-Fold CV (Dedup 1032-dim) ===', flush=True)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_r = []
for fold, (tr, va) in enumerate(kf.split(Xd)):
    s = StandardScaler()
    X_tr_s = s.fit_transform(Xd[tr])
    X_va_s = s.transform(Xd[va])
    sel = SelectKBest(f_regression, k=500)
    X_tr_sel = sel.fit_transform(X_tr_s, yd[tr])
    m = xgb.XGBRegressor(max_depth=12, n_estimators=2000, learning_rate=0.01,
                          subsample=0.8, colsample_bytree=0.8,
                          min_child_weight=3, gamma=0.1,
                          reg_alpha=0.5, reg_lambda=2.0,
                          random_state=42, n_jobs=-1, verbosity=0)
    m.fit(X_tr_sel, yd[tr])
    p = m.predict(sel.transform(X_va_s))
    r, _ = pearsonr(yd[va], p)
    cv_r.append(r)
    print(f'  Fold {fold+1}: R={r:.4f}', flush=True)
print(f'CV R = {np.mean(cv_r):.4f}', flush=True)

# Train final model
print('\nTraining final model...', flush=True)
s = StandardScaler()
Xs = s.fit_transform(Xd)
sel = SelectKBest(f_regression, k=500)
X_sel = sel.fit_transform(Xs, yd)
m = xgb.XGBRegressor(max_depth=12, n_estimators=2000, learning_rate=0.01,
                      subsample=0.8, colsample_bytree=0.8,
                      min_child_weight=3, gamma=0.1,
                      reg_alpha=0.5, reg_lambda=2.0,
                      random_state=42, n_jobs=-1, verbosity=0)
m.fit(X_sel, yd)
print('Done', flush=True)

# ===== CASF evaluation helpers =====
def eval_benchmark(name, base_dir, index_path, pdb_col=0, aff_col=3):
    with open(index_path) as f:
        entries = []
        for line in f:
            if line.startswith('#'): continue
            parts = line.strip().split()
            if len(parts) > max(pdb_col, aff_col):
                try:
                    entries.append({'pdb_id': parts[pdb_col], 'pkd_true': float(parts[aff_col])})
                except: pass
    
    fgen2 = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=512)
    fgen3 = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=256)
    
    pr, tr = [], []
    for cx in entries:
        pid = cx['pdb_id']
        for p in [
            base_dir / 'coreset' / pid / f'{pid}_ligand.mol2',
            base_dir / 'coreset' / pid / f'{pid}_ligand.sdf',
            base_dir / 'coreset' / f'{pid}_ligand.mol2',
            base_dir / 'coreset' / f'{pid}_ligand.sdf',
            base_dir / 'ligand' / 'ranking_scoring' / 'crystal_mol2' / f'{pid}_ligand.mol2',
            base_dir / 'ligand' / 'ranking_scoring' / 'crystal_sdf' / f'{pid}_ligand.sdf',
        ]:
            if p.suffix == '.mol2' and p.exists():
                mol = Chem.MolFromMol2File(str(p), removeHs=False)
                if mol: break
                mol = None
            elif p.suffix == '.sdf' and p.exists():
                try:
                    mol = next((mm for mm in Chem.SDMolSupplier(str(p), removeHs=False) if mm), None)
                    if mol: break
                except: pass
                mol = None
        if mol is None: continue
        
        f = np.concatenate([
            np.array(fgen2.GetFingerprintAsNumPy(mol), dtype=np.float32),
            np.array(MACCSkeys.GenMACCSKeys(mol), dtype=np.float32),
            np.array(fgen3.GetFingerprintAsNumPy(mol), dtype=np.float32),
            np.zeros(47, dtype=np.float32),
        ])
        for pp in [base_dir / 'coreset' / pid / f'{pid}_pocket.pdb']:
            if pp.exists():
                try: f = np.concatenate([f, extract_simple_pocket(str(pp))]); break
                except: pass
        else:
            f = np.concatenate([f, np.zeros(50, dtype=np.float32)])
        
        fs = s.transform(f.reshape(1, -1))
        fs = sel.transform(fs)
        pr.append(float(m.predict(fs)[0]))
        tr.append(cx['pkd_true'])
    
    tr = np.array(tr); pr = np.array(pr)
    r, _ = pearsonr(tr, pr)
    rho, _ = spearmanr(tr, pr)
    mae = np.mean(np.abs(tr - pr))
    rmse = np.sqrt(np.mean((tr - pr)**2))
    print(f'{name}: R={r:.4f} Sp={rho:.4f} MAE={mae:.2f} RMSE={rmse:.2f} (N={len(tr)})', flush=True)
    return r, rho

# ===== Evaluate =====
CASF07_DIR = Path(r'C:\Users\yakka\Downloads\CASF')
r07 = eval_benchmark('CASF-2007', CASF07_DIR, CASF07_DIR / 'PDBbind_core_set_v2007.2.lst', 0, 3)

CASF13_DIR = Path(r'C:\Users\yakka\Downloads\CASF-2013-updated\CASF-2013')
r13 = eval_benchmark('CASF-2013', CASF13_DIR, CASF13_DIR / 'coreset' / 'index' / '2013_core_data.lst', 0, 3)

CASF16_DIR = Path(r'C:\Users\yakka\Downloads\CASF-2016\CASF-2016')
r16 = eval_benchmark('CASF-2016', CASF16_DIR, CASF16_DIR / 'power_scoring' / 'CoreSet.dat', 0, 3)

print('\n========================================')
print('        24K DEDUP FINAL RESULTS')
print('========================================')
print(f'Train: {len(Xd)} unique molecules (from {len(X)} raw entries)')
print(f'Dedup CV R = {np.mean(cv_r):.4f}')
print(f'CASF-2007: R={r07}')
print(f'CASF-2013: R={r13}')
print(f'CASF-2016: R={r16}')
print(f'\nComparison:')
print(f'Phase 5c best (19K non-dedup): CV=0.721, CASF16=0.731')
print(f'Phase 2 (19K non-dedup): CV=0.704, CASF16=0.708')
print(f'Phase 3b (19K pocket):   CV=0.712, CASF16=0.717')
