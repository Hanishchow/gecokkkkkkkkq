"""Process ChEMBL data: compute 982-dim, dedup vs existing, merge, retrain"""
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

fgen2 = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=512)
fgen3 = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=256)

# ===== Load existing training data (Phase 2) =====
X_old = np.load(BACKUP / 'phase2_X.npy')
y_old = np.load(BACKUP / 'phase2_y.npy')
old_hashes = set(hash(row.tobytes()) for row in X_old[:, :512])
print(f'Existing training: {len(X_old)} entries, {len(old_hashes)} unique ECFP')

# ===== Load ChEMBL and compute 982-dim features =====
chembl = []
for fname in ['chembl_v2.pkl', 'chembl_more.pkl']:
    chembl.extend(pickle.load(open(BACKUP / fname, 'rb')))

print(f'\nChEMBL total: {len(chembl)} entries')

# Compute 982-dim features for ChEMBL, skip overlaps
X_new_list = []
y_new_list = []
new_count = 0
dup_count = 0

for entry in chembl:
    ecfp_given = entry['ecfp']
    h = hash(ecfp_given.tobytes())
    if h in old_hashes:
        dup_count += 1
        continue
    
    try:
        mol = Chem.MolFromSmiles(entry['smiles'])
        if mol is None:
            continue
        feats = np.concatenate([
            np.array(fgen2.GetFingerprintAsNumPy(mol), dtype=np.float32),
            np.array(MACCSkeys.GenMACCSKeys(mol), dtype=np.float32),
            np.array(fgen3.GetFingerprintAsNumPy(mol), dtype=np.float32),
            np.zeros(47, dtype=np.float32),
        ])
        X_new_list.append(feats)
        y_new_list.append(entry['pKd'])
        old_hashes.add(h)  # Prevent re-adding
        new_count += 1
    except:
        pass

print(f'Duplicates skipped: {dup_count}')
print(f'New ChEMBL molecules: {new_count}')

if new_count == 0:
    print('No new molecules to add.')
    exit()

X_new = np.array(X_new_list, dtype=np.float32)
y_new = np.array(y_new_list, dtype=np.float32)

print(f'New features shape: {X_new.shape}')

# ===== Merge with existing training =====
X_merged = np.vstack([X_old, X_new])
y_merged = np.concatenate([y_old, y_new])
print(f'\nMerged: {X_merged.shape} entries')

# ===== Load pocket features, build 1032-dim =====
pocket = pickle.load(open(BACKUP / 'pocket_features_full.pkl', 'rb'))
# We don't have PDB IDs for ChEMBL, so these entries won't have pocket features
X_merged_pocket = np.zeros((len(X_merged), 1032), dtype=np.float32)
X_merged_pocket[:, :982] = X_merged
# Pocket for old entries only
old_pdbs = pickle.load(open(BACKUP / 'phase2_pdb_ids.pkl', 'rb'))
for i in range(len(X_old)):
    if old_pdbs[i] in pocket:
        X_merged_pocket[i, 982:] = pocket[old_pdbs[i]]

print(f'{sum(1 for i in range(len(X_old)) if old_pdbs[i] in pocket)}/{len(X_old)} old entries with pocket')

# ===== 5-fold CV on merged data =====
print('\n=== 5-Fold CV (Merged 1032-dim) ===')
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_r = []
for fold, (tr, va) in enumerate(kf.split(X_merged_pocket)):
    s = StandardScaler()
    X_tr_s = s.fit_transform(X_merged_pocket[tr])
    X_va_s = s.transform(X_merged_pocket[va])
    sel = SelectKBest(f_regression, k=500)
    X_tr_sel = sel.fit_transform(X_tr_s, y_merged[tr])
    m = xgb.XGBRegressor(max_depth=12, n_estimators=2000, learning_rate=0.01,
                          subsample=0.8, colsample_bytree=0.8,
                          min_child_weight=3, gamma=0.1,
                          reg_alpha=0.5, reg_lambda=2.0,
                          random_state=42, n_jobs=-1, verbosity=0)
    m.fit(X_tr_sel, y_merged[tr])
    p = m.predict(sel.transform(X_va_s))
    r, _ = pearsonr(y_merged[va], p)
    cv_r.append(r)
    print(f'  Fold {fold+1}: R={r:.4f}')
print(f'CV R = {np.mean(cv_r):.4f}')

# ===== Train final =====
print('\nTraining final model...')
s = StandardScaler()
Xs = s.fit_transform(X_merged_pocket)
sel = SelectKBest(f_regression, k=500)
X_sel = sel.fit_transform(Xs, y_merged)
m = xgb.XGBRegressor(max_depth=12, n_estimators=2000, learning_rate=0.01,
                      subsample=0.8, colsample_bytree=0.8,
                      min_child_weight=3, gamma=0.1,
                      reg_alpha=0.5, reg_lambda=2.0,
                      random_state=42, n_jobs=-1, verbosity=0)
m.fit(X_sel, y_merged)
print('Done')

# ===== CASF evaluation =====
def eval_benchmark(name, base_dir, index_path, pdb_col=0, aff_col=3):
    with open(index_path) as f:
        entries = []
        for line in f:
            if line.startswith('#'): continue
            parts = line.strip().split()
            if len(parts) > max(pdb_col, aff_col):
                try: entries.append({'pdb_id': parts[pdb_col], 'pkd_true': float(parts[aff_col])})
                except: pass
    
    pr, tr = [], []
    for cx in entries:
        pid = cx['pdb_id']
        mol = None
        for p in [
            base_dir / 'coreset' / pid / f'{pid}_ligand.mol2',
            base_dir / 'coreset' / pid / f'{pid}_ligand.sdf',
            base_dir / 'coreset' / f'{pid}_ligand.mol2',
            base_dir / 'coreset' / f'{pid}_ligand.sdf',
            base_dir / 'ligand' / 'ranking_scoring' / 'crystal_mol2' / f'{pid}_ligand.mol2',
            base_dir / 'ligand' / 'ranking_scoring' / 'crystal_sdf' / f'{pid}_ligand.sdf',
        ]:
            if p.suffix == '.mol2' and p.exists():
                try: mol = Chem.MolFromMol2File(str(p), removeHs=False)
                except: pass
                if mol: break
            elif p.suffix == '.sdf' and p.exists():
                try:
                    mol = next((mm for mm in Chem.SDMolSupplier(str(p), removeHs=False) if mm), None)
                    if mol: break
                except: pass
        if mol is None: continue
        
        f = np.concatenate([
            np.array(fgen2.GetFingerprintAsNumPy(mol), dtype=np.float32),
            np.array(MACCSkeys.GenMACCSKeys(mol), dtype=np.float32),
            np.array(fgen3.GetFingerprintAsNumPy(mol), dtype=np.float32),
            np.zeros(47, dtype=np.float32),
        ])
        for pp in [base_dir / 'coreset' / pid / f'{pid}_pocket.pdb', base_dir / 'coreset' / f'{pid}_pocket.pdb']:
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
    print(f'{name}: R={r:.4f} Sp={rho:.4f} MAE={mae:.2f} RMSE={rmse:.2f} (N={len(tr)})')
    return r, rho

CASF07_DIR = Path(r'C:\Users\yakka\Downloads\CASF')
r07 = eval_benchmark('CASF-2007', CASF07_DIR, CASF07_DIR / 'PDBbind_core_set_v2007.2.lst', 0, 3)

CASF13_DIR = Path(r'C:\Users\yakka\Downloads\CASF-2013-updated\CASF-2013')
r13 = eval_benchmark('CASF-2013', CASF13_DIR, CASF13_DIR / 'coreset' / 'index' / '2013_core_data.lst', 0, 3)

CASF16_DIR = Path(r'C:\Users\yakka\Downloads\CASF-2016\CASF-2016')
r16 = eval_benchmark('CASF-2016', CASF16_DIR, CASF16_DIR / 'power_scoring' / 'CoreSet.dat', 0, 3)

print('\n' + '='*50)
print('FINAL RESULTS WITH ChEMBL ADDITION')
print('='*50)
print(f'Training: {len(X_merged)} entries ({len(X_old)} old + {new_count} new ChEMBL)')
print(f'CV R = {np.mean(cv_r):.4f}')
print(f'CASF-2007: R={r07[0]:.4f}')
print(f'CASF-2013: R={r13[0]:.4f}')
print(f'CASF-2016: R={r16[0]:.4f}')
print(f'\nvs Best (no ChEMBL):')
print(f'  CASF-2016: {r16[0]:.4f} vs 0.731')

# Save model
pickle.dump({
    'model': m, 'scaler': s, 'selector': sel,
    'config': f'Phase2 + {new_count} ChEMBL, 1032-dim, t=2000, k=500',
    'cv_r': float(np.mean(cv_r)),
    'casf2007': float(r07[0]), 'casf2013': float(r13[0]), 'casf2016': float(r16[0]),
    'n_train': len(X_merged), 'n_new_chembl': new_count,
}, open(BACKUP / 'geock_chembl_merged.pkl', 'wb'))
print('\nSaved: geock_chembl_merged.pkl')
