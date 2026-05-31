"""Phase 3b: Full pocket features (18,832 entries) + 982-dim"""
import pickle, numpy as np
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, rdFingerprintGenerator
import sys
sys.path.insert(0, r'C:\Users\yakka\Downloads\Hanish_Brain_Vault\autoresearch_backup')
from simple_pocket import extract_simple_pocket

BACKUP = Path(r'C:\Users\yakka\Downloads\Hanish_Brain_Vault\autoresearch_backup')
CASF_DIR = Path(r'C:\Users\yakka\Downloads\CASF-2016\CASF-2016')

# ===== Load training data =====
print("=== Loading training data ===")
X = np.load(BACKUP / 'phase2_X.npy')
y = np.load(BACKUP / 'phase2_y.npy')
pdb_ids = pickle.load(open(BACKUP / 'phase2_pdb_ids.pkl', 'rb'))
pocket_full = pickle.load(open(BACKUP / 'pocket_features_full.pkl', 'rb'))
print(f'Training: X={X.shape}, y={y.shape}, pocket={len(pocket_full)}')

# Build 1032-dim: 982 + 50 pocket
X_aug = np.zeros((len(X), 1032), dtype=np.float32)
X_aug[:, :982] = X
pocket_count = 0
for i, pid in enumerate(pdb_ids):
    if pid in pocket_full:
        X_aug[i, 982:] = pocket_full[pid]
        pocket_count += 1
print(f'Entries with pocket features: {pocket_count}/{len(X)}')

# ===== 5-fold CV =====
print("\n=== 5-Fold Cross Validation ===")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_r = []

for fold, (tr, va) in enumerate(kf.split(X_aug)):
    X_tr, X_va = X_aug[tr], X_aug[va]
    y_tr, y_va = y[tr], y[va]
    s = StandardScaler()
    X_tr_s = s.fit_transform(X_tr)
    X_va_s = s.transform(X_va)
    sel = SelectKBest(f_regression, k=500)
    X_tr_sel = sel.fit_transform(X_tr_s, y_tr)
    X_va_sel = sel.transform(X_va_s)
    m = xgb.XGBRegressor(max_depth=12, n_estimators=500, learning_rate=0.01,
                         subsample=0.8, colsample_bytree=0.8,
                         min_child_weight=3, gamma=0.1,
                         reg_alpha=0.5, reg_lambda=2.0,
                         random_state=42, n_jobs=-1, verbosity=0)
    m.fit(X_tr_sel, y_tr)
    p = m.predict(X_va_sel)
    r, _ = pearsonr(y_va, p)
    cv_r.append(r)
    print(f"  Fold {fold+1}: R={r:.4f}")

print(f"CV R = {np.mean(cv_r):.4f} +/- {np.std(cv_r):.4f}")

# ===== Train Final =====
s = StandardScaler()
Xs = s.fit_transform(X_aug)
sel = SelectKBest(f_regression, k=500)
X_sel = sel.fit_transform(Xs, y)
m = xgb.XGBRegressor(max_depth=12, n_estimators=500, learning_rate=0.01,
                      subsample=0.8, colsample_bytree=0.8,
                      min_child_weight=3, gamma=0.1,
                      reg_alpha=0.5, reg_lambda=2.0,
                      random_state=42, n_jobs=-1, verbosity=0)
m.fit(X_sel, y)

model_data = {
    'model': m, 'scaler': s, 'selector': sel,
    'cv_r': float(np.mean(cv_r)), 'cv_std': float(np.std(cv_r)),
    'n_samples': len(y), 'n_features': X_aug.shape[1], 'k': 500,
    'n_pocket': pocket_count, 'config': '982dim + 18K pocket (1032-dim)'
}
pickle.dump(model_data, open(BACKUP / 'geock_phase3b_fullpocket.pkl', 'wb'))
print("Saved: geock_phase3b_fullpocket.pkl")

# ===== CASF-2016 =====
print("\n=== CASF-2016 ===")
with open(CASF_DIR / 'power_scoring' / 'CoreSet.dat') as f:
    f.readline()
    casf = [{'pdb_id': l.split()[0], 'pkd_true': float(l.split()[3])} for l in f if len(l.strip().split()) >= 4]

def get_mol(pid):
    m2 = CASF_DIR / 'coreset' / pid / f'{pid}_ligand.mol2'
    sd = CASF_DIR / 'coreset' / pid / f'{pid}_ligand.sdf'
    mol = Chem.MolFromMol2File(str(m2), removeHs=False) if m2.exists() else None
    if mol is None and sd.exists():
        mol = next((m for m in Chem.SDMolSupplier(str(sd), removeHs=False) if m), None)
    return mol

def get_feats(mol, pid):
    ecfp = np.array(rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=512).GetFingerprintAsNumPy(mol), dtype=np.float32)
    maccs = np.array(MACCSkeys.GenMACCSKeys(mol), dtype=np.float32)
    fcfp = np.array(rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=256).GetFingerprintAsNumPy(mol), dtype=np.float32)
    rdkit = np.zeros(47, dtype=np.float32)
    lf = np.concatenate([ecfp, maccs, fcfp, rdkit])
    pp = CASF_DIR / 'coreset' / pid / f'{pid}_pocket.pdb'
    pocket = extract_simple_pocket(str(pp)) if pp.exists() else np.zeros(50, dtype=np.float32)
    return np.concatenate([lf, pocket])

pr, tr = [], []
for cx in casf:
    mol = get_mol(cx['pdb_id'])
    if mol is None:
        continue
    f = get_feats(mol, cx['pdb_id']).reshape(1, -1)
    f = s.transform(f)
    f = sel.transform(f)
    pr.append(float(m.predict(f)[0]))
    tr.append(cx['pkd_true'])

pr = np.array(pr)
tr = np.array(tr)
r, _ = pearsonr(tr, pr)
rho, _ = spearmanr(tr, pr)
mae = np.mean(np.abs(tr - pr))
rmse = np.sqrt(np.mean((tr - pr)**2))
print(f"1032-dim (full pocket 18K): R={r:.4f} Sp={rho:.4f} MAE={mae:.2f} RMSE={rmse:.2f}")
print(f"vs Phase 2 (982 no pocket): R=0.708")
print(f"vs Phase 3a (3K pocket): R=0.712")
print(f"vs ECFP-only 39K: R=0.587")
