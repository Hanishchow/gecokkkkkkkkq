"""Train on ALL available data + evaluate on CASF-2007, 2013, 2016"""
import pickle, numpy as np
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
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

# ===== Load ALL training data =====
print("=== Loading training data ===", flush=True)
X19 = np.load(BACKUP / 'phase2_X.npy')
y19 = np.load(BACKUP / 'phase2_y.npy')
pdb_ids = pickle.load(open(BACKUP / 'phase2_pdb_ids.pkl', 'rb'))
pocket_full = pickle.load(open(BACKUP / 'pocket_features_full.pkl', 'rb'))

# Build 1032-dim features (982 + 50 pocket)
X_aug = np.zeros((len(X19), 1032), dtype=np.float32)
X_aug[:, :982] = X19
pocket_count = 0
for i, pid in enumerate(pdb_ids):
    if pid in pocket_full:
        X_aug[i, 982:] = pocket_full[pid]
        pocket_count += 1
print(f'Total entries: {len(X_aug)}, with pocket: {pocket_count}, unique PDBs: {len(set(pdb_ids))}', flush=True)

# ECFP unique count
ecfp_hashes = [hash(row.tobytes()) for row in X19[:, :512]]
print(f'Unique ECFP (molecules): {len(set(ecfp_hashes))} / {len(X19)}', flush=True)

# ===== 5-fold CV on all data =====
print("\n=== 5-Fold CV ===", flush=True)
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_r = []
for fold, (tr, va) in enumerate(kf.split(X_aug)):
    s = StandardScaler()
    X_tr_s = s.fit_transform(X_aug[tr])
    X_va_s = s.transform(X_aug[va])
    sel = SelectKBest(f_regression, k=500)
    X_tr_sel = sel.fit_transform(X_tr_s, y19[tr])
    m = xgb.XGBRegressor(max_depth=12, n_estimators=2000, learning_rate=0.01,
                          subsample=0.8, colsample_bytree=0.8,
                          min_child_weight=3, gamma=0.1,
                          reg_alpha=0.5, reg_lambda=2.0,
                          random_state=42, n_jobs=-1, verbosity=0)
    m.fit(X_tr_sel, y19[tr])
    p = m.predict(sel.transform(X_va_s))
    r, _ = pearsonr(y19[va], p)
    cv_r.append(r)
    print(f"  Fold {fold+1}: R={r:.4f}", flush=True)
cv_mean = np.mean(cv_r)
print(f"CV R = {cv_mean:.4f} +/- {np.std(cv_r):.4f}", flush=True)

# ===== Train Final Model =====
print("\n=== Training Final Model ===", flush=True)
s = StandardScaler()
Xs = s.fit_transform(X_aug)
sel = SelectKBest(f_regression, k=500)
X_sel = sel.fit_transform(Xs, y19)
m = xgb.XGBRegressor(max_depth=12, n_estimators=2000, learning_rate=0.01,
                      subsample=0.8, colsample_bytree=0.8,
                      min_child_weight=3, gamma=0.1,
                      reg_alpha=0.5, reg_lambda=2.0,
                      random_state=42, n_jobs=-1, verbosity=0)
m.fit(X_sel, y19)
print("Model trained", flush=True)

# ===== Helper functions for CASF evaluation =====
def get_mol(pid, base_dir):
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

def get_feats(mol, pid, base_dir):
    ecfp = np.array(rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=512).GetFingerprintAsNumPy(mol), dtype=np.float32)
    maccs = np.array(MACCSkeys.GenMACCSKeys(mol), dtype=np.float32)
    fcfp = np.array(rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=256).GetFingerprintAsNumPy(mol), dtype=np.float32)
    lf = np.concatenate([ecfp, maccs, fcfp, np.zeros(47, dtype=np.float32)])
    for pp in [base_dir / 'coreset' / pid / f'{pid}_pocket.pdb', base_dir / 'coreset' / f'{pid}_pocket.pdb']:
        if pp.exists():
            try: pocket = extract_simple_pocket(str(pp)); break
            except: pass
    else:
        pocket = np.zeros(50, dtype=np.float32)
    return np.concatenate([lf, pocket])

def eval_benchmark(name, base_dir, index_path, pdb_col=0, aff_col=3, skip_header=True, delim=None):
    print(f"\n=== {name} ===", flush=True)
    if delim is None:
        with open(index_path) as f:
            if skip_header: f.readline()
            entries = []
            for line in f:
                if line.startswith('#') or not line.strip(): continue
                parts = line.strip().split()
                if len(parts) > max(pdb_col, aff_col):
                    try: entries.append({'pdb_id': parts[pdb_col], 'pkd_true': float(parts[aff_col])})
                    except: pass
    else:
        with open(index_path) as f:
            entries = []
            for line in f:
                if line.startswith('#') or not line.strip(): continue
                parts = line.strip().split(delim)
                if len(parts) > max(pdb_col, aff_col):
                    try: entries.append({'pdb_id': parts[pdb_col], 'pkd_true': float(parts[aff_col])})
                    except: pass
    print(f'{len(entries)} entries in index', flush=True)
    
    pr, tr = [], []
    for cx in entries:
        mol = get_mol(cx['pdb_id'], base_dir)
        if mol is None:
            continue
        f = get_feats(mol, cx['pdb_id'], base_dir).reshape(1, -1)
        fs = s.transform(f)
        fs = sel.transform(fs)
        pr.append(float(m.predict(fs)[0]))
        tr.append(cx['pkd_true'])
    
    tr = np.array(tr); pr = np.array(pr)
    r, _ = pearsonr(tr, pr)
    rho, _ = spearmanr(tr, pr)
    mae = np.mean(np.abs(tr - pr))
    rmse = np.sqrt(np.mean((tr - pr)**2))
    print(f'{name}: R={r:.4f} Sp={rho:.4f} MAE={mae:.2f} RMSE={rmse:.2f} (N={len(tr)})', flush=True)
    return {'r': r, 'rho': rho, 'mae': mae, 'rmse': rmse, 'n': len(tr)}

# ===== CASF-2007 =====
CASF07_DIR = Path(r'C:\Users\yakka\Downloads\CASF')
r07 = eval_benchmark('CASF-2007', CASF07_DIR, 
                     CASF07_DIR / 'PDBbind_core_set_v2007.2.lst',
                     pdb_col=0, aff_col=3, skip_header=False, delim=None)

# ===== CASF-2013 =====
CASF13_DIR = Path(r'C:\Users\yakka\Downloads\CASF-2013-updated\CASF-2013')
r13 = eval_benchmark('CASF-2013', CASF13_DIR,
                     CASF13_DIR / 'coreset' / 'index' / '2013_core_data.lst',
                     pdb_col=0, aff_col=3, skip_header=False, delim=None)

# ===== CASF-2016 =====
CASF16_DIR = Path(r'C:\Users\yakka\Downloads\CASF-2016\CASF-2016')
r16 = eval_benchmark('CASF-2016', CASF16_DIR,
                     CASF16_DIR / 'power_scoring' / 'CoreSet.dat',
                     pdb_col=0, aff_col=3, skip_header=True, delim=None)

# ===== Save model =====
model_data = {
    'model': m, 'scaler': s, 'selector': sel,
    'config': 'All data: 19087 entries, 1032-dim, k=500, t=2000',
    'cv_r': float(cv_mean), 'cv_std': float(np.std(cv_r)),
    'casf2007_r': float(r07['r']), 'casf2013_r': float(r13['r']), 'casf2016_r': float(r16['r']),
    'n_train': len(y19), 'n_unique_ecfp': len(set(ecfp_hashes)), 'n_pocket': pocket_count,
}
pickle.dump(model_data, open(BACKUP / 'geock_final_all.pkl', 'wb'))
print("\nSaved: geock_final_all.pkl", flush=True)

# ===== Also deduplicate and train =====
print("\n\n=== Deduplicated Version ===", flush=True)
uniq_ecfp = {}
for i, h in enumerate(ecfp_hashes):
    if h not in uniq_ecfp:
        uniq_ecfp[h] = []
    uniq_ecfp[h].append(i)

X_dedup = np.zeros((len(uniq_ecfp), X_aug.shape[1]), dtype=np.float32)
y_dedup = np.zeros(len(uniq_ecfp), dtype=np.float32)
for j, (h, idxs) in enumerate(uniq_ecfp.items()):
    X_dedup[j] = X_aug[idxs[0]]
    y_dedup[j] = np.median(y19[idxs])

print(f'Deduplicated: {len(X19)} -> {len(X_dedup)}', flush=True)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_r_d = []
for tr, va in kf.split(X_dedup):
    s_d = StandardScaler()
    X_tr_s = s_d.fit_transform(X_dedup[tr])
    X_va_s = s_d.transform(X_dedup[va])
    sel_d = SelectKBest(f_regression, k=500)
    X_tr_sel = sel_d.fit_transform(X_tr_s, y_dedup[tr])
    m_d = xgb.XGBRegressor(max_depth=12, n_estimators=2000, learning_rate=0.01,
                            subsample=0.8, colsample_bytree=0.8,
                            min_child_weight=3, gamma=0.1,
                            reg_alpha=0.5, reg_lambda=2.0,
                            random_state=42, n_jobs=-1, verbosity=0)
    m_d.fit(X_tr_sel, y_dedup[tr])
    p = m_d.predict(sel_d.transform(X_va_s))
    r, _ = pearsonr(y_dedup[va], p)
    cv_r_d.append(r)
print(f'Dedup CV R = {np.mean(cv_r_d):.4f}', flush=True)

s_d = StandardScaler()
Xs_d = s_d.fit_transform(X_dedup)
sel_d = SelectKBest(f_regression, k=500)
X_sel_d = sel_d.fit_transform(Xs_d, y_dedup)
m_d = xgb.XGBRegressor(max_depth=12, n_estimators=2000, learning_rate=0.01,
                        subsample=0.8, colsample_bytree=0.8,
                        min_child_weight=3, gamma=0.1,
                        reg_alpha=0.5, reg_lambda=2.0,
                        random_state=42, n_jobs=-1, verbosity=0)
m_d.fit(X_sel_d, y_dedup)

def eval_dedup(name, base_dir, index_path, pdb_col=0, aff_col=3, skip_header=True):
    with open(index_path) as f:
        if skip_header: f.readline()
        entries = []
        for line in f:
            if line.startswith('#') or not line.strip(): continue
            parts = line.strip().split()
            if len(parts) > max(pdb_col, aff_col):
                try: entries.append({'pdb_id': parts[pdb_col], 'pkd_true': float(parts[aff_col])})
                except: pass
    pr, tr = [], []
    for cx in entries:
        mol = get_mol(cx['pdb_id'], base_dir)
        if mol is None: continue
        f = get_feats(mol, cx['pdb_id'], base_dir).reshape(1, -1)
        fs = s_d.transform(f)
        fs = sel_d.transform(fs)
        pr.append(float(m_d.predict(fs)[0]))
        tr.append(cx['pkd_true'])
    r, _ = pearsonr(tr, pr)
    rho, _ = spearmanr(tr, pr)
    print(f'  Dedup {name}: R={r:.4f} Sp={rho:.4f} (N={len(tr)})', flush=True)
    return r

r07_d = eval_dedup('CASF-2007', CASF07_DIR, CASF07_DIR / 'PDBbind_core_set_v2007.2.lst', 0, 3, False)
r13_d = eval_dedup('CASF-2013', CASF13_DIR, CASF13_DIR / 'coreset' / 'index' / '2013_core_data.lst', 0, 3, False)
r16_d = eval_dedup('CASF-2016', CASF16_DIR, CASF16_DIR / 'power_scoring' / 'CoreSet.dat', 0, 3, True)

model_data_d = {
    'model': m_d, 'scaler': s_d, 'selector': sel_d,
    'config': f'Dedup {len(X_dedup)} entries, 1032-dim, k=500, t=2000',
    'cv_r': float(np.mean(cv_r_d)),
    'casf2007_r': float(r07_d), 'casf2013_r': float(r13_d), 'casf2016_r': float(r16_d),
    'n_train': len(X_dedup),
}
pickle.dump(model_data_d, open(BACKUP / 'geock_final_dedup.pkl', 'wb'))
print("Saved: geock_final_dedup.pkl", flush=True)

# ===== Final summary =====
print("\n\n========================================")
print("         FINAL COMPARISON TABLE          ")
print("========================================")
print(f"{'Model':25s} {'CV-R':8s} {'2007-R':8s} {'2013-R':8s} {'2016-R':8s}")
print("-" * 60)
print(f"{'All 19K (non-dedup)':25s} {cv_mean:.4f}   {r07['r']:.4f}   {r13['r']:.4f}   {r16['r']:.4f}")
print(f"{'Deduplicated':25s} {np.mean(cv_r_d):.4f}   {r07_d:.4f}   {r13_d:.4f}   {r16_d:.4f}")
print(f"{'(ref) ECFP-only 39K':25s} 0.847   0.877   0.870   0.587")
print(f"{'(ref) Phase 5c best':25s} 0.721   —       —       0.731")
print(f"\nNon-dedup: {len(X_aug)} entries, dedup: {len(X_dedup)} entries")
print(f"Features: 1032-dim (ECFP4+MACCS+FCFP4+RDKit47+Pocket50)")
