"""Phase 4: Ensemble — train with 3 seeds, evaluate CV + CASF-2016"""
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
pocket_count = 0
for i, pid in enumerate(pdb_ids):
    if pid in pocket_full:
        X_aug[i, 982:] = pocket_full[pid]
        pocket_count += 1
print(f'X={X_aug.shape}, pocket={pocket_count}/{len(X)}')

seeds = [42, 1, 123]
models_data = []

for seed in seeds:
    print(f'\nSeed {seed}:', flush=True)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_r = []
    for fold, (tr, va) in enumerate(kf.split(X_aug)):
        s = StandardScaler()
        X_tr_s = s.fit_transform(X_aug[tr])
        X_va_s = s.transform(X_aug[va])
        sel = SelectKBest(f_regression, k=500)
        X_tr_sel = sel.fit_transform(X_tr_s, y[tr])
        m = xgb.XGBRegressor(max_depth=12, n_estimators=500, learning_rate=0.01,
                             subsample=0.8, colsample_bytree=0.8,
                             min_child_weight=3, gamma=0.1,
                             reg_alpha=0.5, reg_lambda=2.0,
                             random_state=seed, n_jobs=-1, verbosity=0)
        m.fit(X_tr_sel, y[tr])
        p = m.predict(sel.transform(X_va_s))
        r, _ = pearsonr(y[va], p)
        cv_r.append(r)
        print(f'  Fold {fold+1}: R={r:.4f}', flush=True)
    print(f'  CV R = {np.mean(cv_r):.4f}', flush=True)
    
    s = StandardScaler()
    Xs = s.fit_transform(X_aug)
    sel = SelectKBest(f_regression, k=500)
    X_sel = sel.fit_transform(Xs, y)
    m = xgb.XGBRegressor(max_depth=12, n_estimators=500, learning_rate=0.01,
                          subsample=0.8, colsample_bytree=0.8,
                          min_child_weight=3, gamma=0.1,
                          reg_alpha=0.5, reg_lambda=2.0,
                          random_state=seed, n_jobs=-1, verbosity=0)
    m.fit(X_sel, y)
    models_data.append({'model': m, 'scaler': s, 'selector': sel, 'seed': seed})

pickle.dump(models_data, open(BACKUP / 'geock_phase4_ensemble.pkl', 'wb'))
print('Saved ensemble', flush=True)

# CASF-2016
with open(CASF_DIR / 'power_scoring' / 'CoreSet.dat') as f:
    f.readline()
    casf = [{'pdb_id': l.split()[0], 'pkd_true': float(l.split()[3])} for l in f if len(l.strip().split()) >= 4]

def make_feats(mol, pid):
    ecfp = np.array(rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=512).GetFingerprintAsNumPy(mol), dtype=np.float32)
    maccs = np.array(MACCSkeys.GenMACCSKeys(mol), dtype=np.float32)
    fcfp = np.array(rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=256).GetFingerprintAsNumPy(mol), dtype=np.float32)
    lf = np.concatenate([ecfp, maccs, fcfp, np.zeros(47, dtype=np.float32)])
    pp = CASF_DIR / 'coreset' / pid / f'{pid}_pocket.pdb'
    pocket = extract_simple_pocket(str(pp)) if pp.exists() else np.zeros(50, dtype=np.float32)
    return np.concatenate([lf, pocket])

all_preds, trues = [], []
for cx in casf:
    m2 = CASF_DIR / 'coreset' / cx['pdb_id'] / f"{cx['pdb_id']}_ligand.mol2"
    sd = CASF_DIR / 'coreset' / cx['pdb_id'] / f"{cx['pdb_id']}_ligand.sdf"
    mol = None
    if m2.exists(): mol = Chem.MolFromMol2File(str(m2), removeHs=False)
    if mol is None and sd.exists(): mol = next((m for m in Chem.SDMolSupplier(str(sd), removeHs=False) if m), None)
    if mol is None: continue
    f = make_feats(mol, cx['pdb_id']).reshape(1, -1)
    preds = []
    for md in models_data:
        fs = md['scaler'].transform(f)
        fs = md['selector'].transform(fs)
        preds.append(float(md['model'].predict(fs)[0]))
    all_preds.append(np.mean(preds))
    trues.append(cx['pkd_true'])

trues = np.array(trues)
all_preds = np.array(all_preds)
r, _ = pearsonr(trues, all_preds)
rho, _ = spearmanr(trues, all_preds)
mae = np.mean(np.abs(trues - all_preds))
rmse = np.sqrt(np.mean((trues - all_preds)**2))
print(f'\nPhase 4 Ensemble (3 seeds): R={r:.4f} Sp={rho:.4f} MAE={mae:.2f} RMSE={rmse:.2f}', flush=True)
