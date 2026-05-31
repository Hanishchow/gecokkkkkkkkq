"""Phase 5: Hyperparameter sweeps — try k values, depths, n_estimators"""
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
CASF_DIR = Path(r'C:\Users\yakka\Downloads\CASF-2016\CASF-2016')

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
    ('k=900', dict(k=900)),
    ('k=700', dict(k=700)),
    ('k=500,depth=16', dict(k=500, max_depth=16)),
    ('k=500,trees=800', dict(k=500, n_estimators=800)),
    ('k=500,lr=0.03', dict(k=500, learning_rate=0.03)),
    ('k=500,depth=8', dict(k=500, max_depth=8)),
    ('k=500,colsample=0.6', dict(k=500, colsample_bytree=0.6)),
]

results = []
for name, cfg in configs:
    kf = KFold(n_splits=3, shuffle=True, random_state=42)  # 3-fold for speed
    cv_r = []
    for tr, va in kf.split(X_aug):
        s = StandardScaler()
        X_tr_s = s.fit_transform(X_aug[tr])
        X_va_s = s.transform(X_aug[va])
        k = cfg.get('k', 500)
        sel = SelectKBest(f_regression, k=k)
        X_tr_sel = sel.fit_transform(X_tr_s, y[tr])
        m = xgb.XGBRegressor(
            max_depth=cfg.get('max_depth', 12),
            n_estimators=cfg.get('n_estimators', 500),
            learning_rate=cfg.get('learning_rate', 0.01),
            subsample=0.8,
            colsample_bytree=cfg.get('colsample_bytree', 0.8),
            min_child_weight=3, gamma=0.1,
            reg_alpha=0.5, reg_lambda=2.0,
            random_state=42, n_jobs=-1, verbosity=0
        )
        m.fit(X_tr_sel, y[tr])
        p = m.predict(sel.transform(X_va_s))
        r, _ = pearsonr(y[va], p)
        cv_r.append(r)
    cv_mean = np.mean(cv_r)
    
    # Train final & eval CASF
    s = StandardScaler()
    Xs = s.fit_transform(X_aug)
    k = cfg.get('k', 500)
    sel = SelectKBest(f_regression, k=k)
    X_sel = sel.fit_transform(Xs, y)
    m = xgb.XGBRegressor(
        max_depth=cfg.get('max_depth', 12),
        n_estimators=cfg.get('n_estimators', 500),
        learning_rate=cfg.get('learning_rate', 0.01),
        subsample=0.8,
        colsample_bytree=cfg.get('colsample_bytree', 0.8),
        min_child_weight=3, gamma=0.1,
        reg_alpha=0.5, reg_lambda=2.0,
        random_state=42, n_jobs=-1, verbosity=0
    )
    m.fit(X_sel, y)
    
    # CASF eval
    with open(CASF_DIR / 'power_scoring' / 'CoreSet.dat') as f:
        f.readline()
        casf = [{'pdb_id': l.split()[0], 'pkd_true': float(l.split()[3])} for l in f if len(l.strip().split()) >= 4]
    
    def get_mol(pid):
        m2 = CASF_DIR / 'coreset' / pid / f'{pid}_ligand.mol2'
        sd = CASF_DIR / 'coreset' / pid / f'{pid}_ligand.sdf'
        mol = Chem.MolFromMol2File(str(m2), removeHs=False) if m2.exists() else None
        if mol is None and sd.exists(): mol = next((mm for mm in Chem.SDMolSupplier(str(sd), removeHs=False) if mm), None)
        return mol
    
    def get_feats(mol, pid):
        ecfp = np.array(rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=512).GetFingerprintAsNumPy(mol), dtype=np.float32)
        maccs = np.array(MACCSkeys.GenMACCSKeys(mol), dtype=np.float32)
        fcfp = np.array(rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=256).GetFingerprintAsNumPy(mol), dtype=np.float32)
        lf = np.concatenate([ecfp, maccs, fcfp, np.zeros(47, dtype=np.float32)])
        pp = CASF_DIR / 'coreset' / pid / f'{pid}_pocket.pdb'
        pocket = extract_simple_pocket(str(pp)) if pp.exists() else np.zeros(50, dtype=np.float32)
        return np.concatenate([lf, pocket])
    
    pr, tr = [], []
    for cx in casf:
        mol = get_mol(cx['pdb_id'])
        if mol is None: continue
        f = get_feats(mol, cx['pdb_id']).reshape(1, -1)
        fs = s.transform(f)
        fs = sel.transform(fs)
        pr.append(float(m.predict(fs)[0]))
        tr.append(cx['pkd_true'])
    
    r_casf, _ = pearsonr(tr, pr)
    rho_casf, _ = spearmanr(tr, pr)
    mae = np.mean(np.abs(np.array(tr) - np.array(pr)))
    rmse = np.sqrt(np.mean((np.array(tr) - np.array(pr))**2))
    
    results.append((name, cv_mean, r_casf, rho_casf, mae, rmse))
    print(f'{name}: CV={cv_mean:.4f} CASF-R={r_casf:.4f} Sp={rho_casf:.4f} MAE={mae:.2f} RMSE={rmse:.2f}', flush=True)

print('\n=== Phase 5 Results ===')
print(f'{"Config":25s} {"CV-R":8s} {"CASF-R":8s} {"Sp":8s} {"MAE":6s} {"RMSE":6s}')
print('-'*65)
for name, cv, r, rho, mae, rmse in sorted(results, key=lambda x: -x[2]):
    print(f'{name:25s} {cv:.4f}   {r:.4f}   {rho:.4f}  {mae:.2f}  {rmse:.2f}')
