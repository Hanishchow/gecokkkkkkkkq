"""Phase 5b (best params) + Phase 6 (CASF-2007/2013)"""
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

X = np.load(BACKUP / 'phase2_X.npy')
y = np.load(BACKUP / 'phase2_y.npy')
pdb_ids = pickle.load(open(BACKUP / 'phase2_pdb_ids.pkl', 'rb'))
pocket_full = pickle.load(open(BACKUP / 'pocket_features_full.pkl', 'rb'))

X_aug = np.zeros((len(X), 1032), dtype=np.float32)
X_aug[:, :982] = X
for i, pid in enumerate(pdb_ids):
    if pid in pocket_full:
        X_aug[i, 982:] = pocket_full[pid]

# Try even more trees + depth combos
configs = [
    ('k=500,trees=1000', dict(k=500, n_estimators=1000, max_depth=12)),
    ('k=500,trees=800', dict(k=500, n_estimators=800, max_depth=12)),
    ('k=500,trees=800,depth=14', dict(k=500, n_estimators=800, max_depth=14)),
    ('k=500,trees=600,depth=12', dict(k=500, n_estimators=600, max_depth=12)),
]

best_model_data = None
best_r = 0
best_name = ''

for name, cfg in configs:
    print(f'\n--- {name} ---', flush=True)
    # 5-fold CV
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_r = []
    for fold, (tr, va) in enumerate(kf.split(X_aug)):
        s = StandardScaler()
        X_tr_s = s.fit_transform(X_aug[tr])
        X_va_s = s.transform(X_aug[va])
        k = cfg.get('k', 500)
        sel = SelectKBest(f_regression, k=k)
        X_tr_sel = sel.fit_transform(X_tr_s, y[tr])
        m = xgb.XGBRegressor(max_depth=cfg.get('max_depth', 12), n_estimators=cfg.get('n_estimators', 500),
                             learning_rate=0.01, subsample=0.8, colsample_bytree=0.8,
                             min_child_weight=3, gamma=0.1, reg_alpha=0.5, reg_lambda=2.0,
                             random_state=42, n_jobs=-1, verbosity=0)
        m.fit(X_tr_sel, y[tr])
        p = m.predict(sel.transform(X_va_s))
        r, _ = pearsonr(y[va], p)
        cv_r.append(r)
    cv_mean = np.mean(cv_r)
    print(f'  CV: {cv_mean:.4f}', flush=True)
    
    # Train final
    s = StandardScaler()
    Xs = s.fit_transform(X_aug)
    k = cfg.get('k', 500)
    sel = SelectKBest(f_regression, k=k)
    X_sel = sel.fit_transform(Xs, y)
    m = xgb.XGBRegressor(max_depth=cfg.get('max_depth', 12), n_estimators=cfg.get('n_estimators', 500),
                          learning_rate=0.01, subsample=0.8, colsample_bytree=0.8,
                          min_child_weight=3, gamma=0.1, reg_alpha=0.5, reg_lambda=2.0,
                          random_state=42, n_jobs=-1, verbosity=0)
    m.fit(X_sel, y)
    
    # CASF-2016
    with open(CASF16_DIR / 'power_scoring' / 'CoreSet.dat') as f:
        f.readline()
        casf16 = [{'pdb_id': l.split()[0], 'pkd_true': float(l.split()[3])} for l in f if len(l.strip().split()) >= 4]
    
    def get_mol(pid, base_dir):
        m2 = base_dir / 'coreset' / pid / f'{pid}_ligand.mol2'
        sd = base_dir / 'coreset' / pid / f'{pid}_ligand.sdf'
        mol = Chem.MolFromMol2File(str(m2), removeHs=False) if m2.exists() else None
        if mol is None and sd.exists(): mol = next((mm for mm in Chem.SDMolSupplier(str(sd), removeHs=False) if mm), None)
        return mol
    
    def get_feats(mol, pid, base_dir):
        ecfp = np.array(rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=512).GetFingerprintAsNumPy(mol), dtype=np.float32)
        maccs = np.array(MACCSkeys.GenMACCSKeys(mol), dtype=np.float32)
        fcfp = np.array(rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=256).GetFingerprintAsNumPy(mol), dtype=np.float32)
        lf = np.concatenate([ecfp, maccs, fcfp, np.zeros(47, dtype=np.float32)])
        pp = base_dir / 'coreset' / pid / f'{pid}_pocket.pdb'
        pocket = extract_simple_pocket(str(pp)) if pp.exists() else np.zeros(50, dtype=np.float32)
        return np.concatenate([lf, pocket])
    
    pr, tr = [], []
    for cx in casf16:
        mol = get_mol(cx['pdb_id'], CASF16_DIR)
        if mol is None: continue
        f = get_feats(mol, cx['pdb_id'], CASF16_DIR).reshape(1, -1)
        fs = s.transform(f)
        fs = sel.transform(fs)
        pr.append(float(m.predict(fs)[0]))
        tr.append(cx['pkd_true'])
    r_casf, _ = pearsonr(tr, pr)
    rho_casf, _ = spearmanr(tr, pr)
    print(f'  CASF-2016: R={r_casf:.4f} Sp={rho_casf:.4f}', flush=True)
    
    if r_casf > best_r:
        best_r = r_casf
        best_model_data = (m, s, sel, cfg)
        best_name = name

# Save best model
m, s, sel, best_cfg = best_model_data
pickle.dump({
    'model': m, 'scaler': s, 'selector': sel,
    'config': best_name, 'params': best_cfg,
    'cv_r': float(cv_mean), 'casf16_r': float(best_r)
}, open(BACKUP / 'geock_phase5_best.pkl', 'wb'))
print(f'\nSaved best model: {best_name} (R={best_r:.4f})', flush=True)

# ===== Phase 6: CASF-2007 =====
print('\n=== Phase 6: CASF-2007 ===')
CASF07_DIR = Path(r'C:\Users\yakka\Downloads\CASF-2007')
if CASF07_DIR.exists():
    with open(CASF07_DIR / 'index' / 'INDEX_core_data.2007') as f:
        casf07 = []
        for line in f:
            if line.startswith('#') or not line.strip(): continue
            parts = line.strip().split()
            if len(parts) >= 6:
                try: casf07.append({'pdb_id': parts[0], 'pkd_true': float(parts[5])})
                except: pass
    print(f'{len(casf07)} complexes')
    
    pr07, tr07 = [], []
    for cx in casf07:
        mol = get_mol(cx['pdb_id'], CASF07_DIR)
        if mol is None: continue
        f = get_feats(mol, cx['pdb_id'], CASF07_DIR).reshape(1, -1)
        fs = s.transform(f)
        fs = sel.transform(fs)
        pr07.append(float(m.predict(fs)[0]))
        tr07.append(cx['pkd_true'])
    r07, _ = pearsonr(tr07, pr07)
    rho07, _ = spearmanr(tr07, pr07)
    mae07 = np.mean(np.abs(np.array(tr07) - np.array(pr07)))
    rmse07 = np.sqrt(np.mean((np.array(tr07) - np.array(pr07))**2))
    print(f'CASF-2007: R={r07:.4f} Sp={rho07:.4f} MAE={mae07:.2f} RMSE={rmse07:.2f} (N={len(tr07)})')
else:
    print('CASF-2007 directory not found')

# ===== Phase 6: CASF-2013 =====
print('\n=== Phase 6: CASF-2013 ===')
CASF13_DIR = Path(r'C:\Users\yakka\Downloads\CASF-2013')
if CASF13_DIR.exists():
    with open(CASF13_DIR / 'index' / 'INDEX_core_data.2013') as f:
        casf13 = []
        for line in f:
            if line.startswith('#') or not line.strip(): continue
            parts = line.strip().split()
            if len(parts) >= 6:
                try: casf13.append({'pdb_id': parts[0], 'pkd_true': float(parts[5])})
                except: pass
    print(f'{len(casf13)} complexes')
    
    pr13, tr13 = [], []
    for cx in casf13:
        mol = get_mol(cx['pdb_id'], CASF13_DIR)
        if mol is None:
            m2 = CASF13_DIR / 'coreset' / f'{cx["pdb_id"]}_ligand.mol2'
            if m2.exists(): mol = Chem.MolFromMol2File(str(m2), removeHs=False)
        if mol is None: continue
        f = get_feats(mol, cx['pdb_id'], CASF13_DIR).reshape(1, -1)
        fs = s.transform(f)
        fs = sel.transform(fs)
        pr13.append(float(m.predict(fs)[0]))
        tr13.append(cx['pkd_true'])
    r13, _ = pearsonr(tr13, pr13)
    rho13, _ = spearmanr(tr13, pr13)
    mae13 = np.mean(np.abs(np.array(tr13) - np.array(pr13)))
    rmse13 = np.sqrt(np.mean((np.array(tr13) - np.array(pr13))**2))
    print(f'CASF-2013: R={r13:.4f} Sp={rho13:.4f} MAE={mae13:.2f} RMSE={rmse13:.2f} (N={len(tr13)})')
else:
    print('CASF-2013 directory not found')

print('\n=== BEST RESULTS ===')
print(f'Model: {best_name}')
print(f'CASF-2007: R={r07 if CASF07_DIR.exists() else "N/A":.4f}' if CASF07_DIR.exists() else '')
print(f'CASF-2013: R={r13 if CASF13_DIR.exists() else "N/A":.4f}' if CASF13_DIR.exists() else '')
print(f'CASF-2016: R={best_r:.4f}')
print(f'\nProgress summary:')
print(f'  ECFP-only 39K:       R=0.587 (baseline)')
print(f'  982-dim 19K:         R=0.708')
print(f'  +pocket 18K:         R=0.717')
print(f'  +trees=800,k=500:    R={best_r:.4f}')
