"""Phase 5c: Push further — more trees, depth combos, lower LR"""
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

configs = [
    ('t1500', dict(k=500, n_estimators=1500, max_depth=12)),
    ('t1000_d14', dict(k=500, n_estimators=1000, max_depth=14)),
    ('t1000_d12_lr005', dict(k=500, n_estimators=1000, max_depth=12, learning_rate=0.005)),
    ('t2000', dict(k=500, n_estimators=2000, max_depth=12)),
]

for name, cfg in configs:
    print(f'\n--- {name} ---', flush=True)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_r = []
    for fold, (tr, va) in enumerate(kf.split(X_aug)):
        s = StandardScaler()
        X_tr_s = s.fit_transform(X_aug[tr])
        X_va_s = s.transform(X_aug[va])
        sel = SelectKBest(f_regression, k=cfg.get('k', 500))
        X_tr_sel = sel.fit_transform(X_tr_s, y[tr])
        m = xgb.XGBRegressor(max_depth=cfg.get('max_depth', 12), n_estimators=cfg.get('n_estimators', 500),
                             learning_rate=cfg.get('learning_rate', 0.01), subsample=0.8,
                             colsample_bytree=0.8, min_child_weight=3, gamma=0.1,
                             reg_alpha=0.5, reg_lambda=2.0, random_state=42, n_jobs=-1, verbosity=0)
        m.fit(X_tr_sel, y[tr])
        p = m.predict(sel.transform(X_va_s))
        r, _ = pearsonr(y[va], p)
        cv_r.append(r)
    cv_mean = np.mean(cv_r)
    print(f'  CV: {cv_mean:.4f}', flush=True)
    
    s = StandardScaler()
    Xs = s.fit_transform(X_aug)
    sel = SelectKBest(f_regression, k=cfg.get('k', 500))
    X_sel = sel.fit_transform(Xs, y)
    m = xgb.XGBRegressor(max_depth=cfg.get('max_depth', 12), n_estimators=cfg.get('n_estimators', 500),
                          learning_rate=cfg.get('learning_rate', 0.01), subsample=0.8,
                          colsample_bytree=0.8, min_child_weight=3, gamma=0.1,
                          reg_alpha=0.5, reg_lambda=2.0, random_state=42, n_jobs=-1, verbosity=0)
    m.fit(X_sel, y)
    
    with open(CASF16_DIR / 'power_scoring' / 'CoreSet.dat') as f:
        f.readline()
        casf = [{'pdb_id': l.split()[0], 'pkd_true': float(l.split()[3])} for l in f if len(l.strip().split()) >= 4]
    
    pr, tr = [], []
    for cx in casf:
        m2 = CASF16_DIR / 'coreset' / cx['pdb_id'] / f"{cx['pdb_id']}_ligand.mol2"
        sd = CASF16_DIR / 'coreset' / cx['pdb_id'] / f"{cx['pdb_id']}_ligand.sdf"
        mol = Chem.MolFromMol2File(str(m2), removeHs=False) if m2.exists() else None
        if mol is None and sd.exists(): mol = next((mm for mm in Chem.SDMolSupplier(str(sd), removeHs=False) if mm), None)
        if mol is None: continue
        ecfp = np.array(rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=512).GetFingerprintAsNumPy(mol), dtype=np.float32)
        maccs = np.array(MACCSkeys.GenMACCSKeys(mol), dtype=np.float32)
        fcfp = np.array(rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=256).GetFingerprintAsNumPy(mol), dtype=np.float32)
        lf = np.concatenate([ecfp, maccs, fcfp, np.zeros(47, dtype=np.float32)])
        pp = CASF16_DIR / 'coreset' / cx['pdb_id'] / f"{cx['pdb_id']}_pocket.pdb"
        pocket = extract_simple_pocket(str(pp)) if pp.exists() else np.zeros(50, dtype=np.float32)
        f = np.concatenate([lf, pocket]).reshape(1, -1)
        fs = s.transform(f)
        fs = sel.transform(fs)
        pr.append(float(m.predict(fs)[0]))
        tr.append(cx['pkd_true'])
    
    r_c, _ = pearsonr(tr, pr)
    rho_c, _ = spearmanr(tr, pr)
    print(f'  CASF-2016: R={r_c:.4f} Sp={rho_c:.4f}', flush=True)

print('\nDone!')
