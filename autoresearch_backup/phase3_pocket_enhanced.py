"""Phase 3: Train on 982-dim + pocket features, mixed availability"""
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

# ===== STEP 1: Load training data =====
print("=== Loading training data ===")
X = np.load(BACKUP / 'phase2_X.npy')  # 982-dim features
y = np.load(BACKUP / 'phase2_y.npy')

# Load pocket features for training
ps = pickle.load(open(BACKUP / 'training_pocket_simple.pkl', 'rb'))
pdb_ids = pickle.load(open(BACKUP / 'phase2_pdb_ids.pkl', 'rb'))
print(f'Training: X={X.shape}, y={y.shape}')
print(f'PDB IDs with pocket features: {len(ps)}')

# Build pocket-augmented features: 982 + 50 = 1032-dim
X_aug = np.zeros((len(X), 1032), dtype=np.float32)
X_aug[:, :982] = X
pocket_count = 0
for i, pid in enumerate(pdb_ids):
    if pid in ps:
        X_aug[i, 982:] = ps[pid]
        pocket_count += 1
print(f'Entries with pocket features: {pocket_count}/{len(X)}')

# ===== STEP 2: 5-fold CV =====
print("\n=== 5-Fold Cross Validation ===")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_aug)):
    X_tr, X_val = X_aug[train_idx], X_aug[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]
    
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)
    
    selector = SelectKBest(f_regression, k=500)
    X_tr_sel = selector.fit_transform(X_tr_s, y_tr)
    X_val_sel = selector.transform(X_val_s)
    
    model = xgb.XGBRegressor(
        max_depth=12, n_estimators=500, learning_rate=0.01,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=3, gamma=0.1,
        reg_alpha=0.5, reg_lambda=2.0,
        random_state=42, n_jobs=-1, verbosity=0
    )
    model.fit(X_tr_sel, y_tr)
    preds = model.predict(X_val_sel)
    r, _ = pearsonr(y_val, preds)
    cv_scores.append(r)
    print(f"  Fold {fold+1}: R={r:.4f}")

print(f"CV R = {np.mean(cv_scores):.4f} +/- {np.std(cv_scores):.4f}")

# ===== STEP 3: Train Final =====
print("\n=== Training Final Model ===")
scaler = StandardScaler()
X_s = scaler.fit_transform(X_aug)
selector = SelectKBest(f_regression, k=500)
X_sel = selector.fit_transform(X_s, y)
model = xgb.XGBRegressor(
    max_depth=12, n_estimators=500, learning_rate=0.01,
    subsample=0.8, colsample_bytree=0.8,
    min_child_weight=3, gamma=0.1,
    reg_alpha=0.5, reg_lambda=2.0,
    random_state=42, n_jobs=-1, verbosity=0
)
model.fit(X_sel, y)

model_data = {
    'model': model, 'scaler': scaler, 'selector': selector,
    'cv_r': float(np.mean(cv_scores)), 'cv_std': float(np.std(cv_scores)),
    'n_samples': len(y), 'n_features': X_aug.shape[1], 'k': 500,
    'config': '982-dim + pocket(50) mixed = 1032-dim'
}
pickle.dump(model_data, open(BACKUP / 'geock_phase3_1032dim.pkl', 'wb'))
print("Saved: geock_phase3_1032dim.pkl")

# ===== STEP 4: CASF-2016 Evaluation =====
print("\n=== CASF-2016 Evaluation ===")
with open(CASF_DIR / 'power_scoring' / 'CoreSet.dat') as f:
    f.readline()
    casf = []
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 4:
            casf.append({'pdb_id': parts[0], 'pkd_true': float(parts[3])})

def get_mol(pid):
    mol2 = CASF_DIR / 'coreset' / pid / f'{pid}_ligand.mol2'
    sdf = CASF_DIR / 'coreset' / pid / f'{pid}_ligand.sdf'
    mol = Chem.MolFromMol2File(str(mol2), removeHs=False) if mol2.exists() else None
    if mol is None and sdf.exists():
        suppl = Chem.SDMolSupplier(str(sdf), removeHs=False)
        mol = next((m for m in suppl if m), None)
    return mol

def get_1032_features(mol, pid):
    ecfp = np.array(rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=512).GetFingerprintAsNumPy(mol), dtype=np.float32)
    maccs = np.array(MACCSkeys.GenMACCSKeys(mol), dtype=np.float32)
    fcfp = np.array(rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=256).GetFingerprintAsNumPy(mol), dtype=np.float32)
    rdkit = np.zeros(47, dtype=np.float32)
    ligand = np.concatenate([ecfp, maccs, fcfp, rdkit])
    # Get pocket features from CASF pocket PDB
    pocket_pdb = CASF_DIR / 'coreset' / pid / f'{pid}_pocket.pdb'
    if pocket_pdb.exists():
        pocket = extract_simple_pocket(str(pocket_pdb))
    else:
        pocket = np.zeros(50, dtype=np.float32)
    return np.concatenate([ligand, pocket])

preds, trues = [], []
for cx in casf:
    mol = get_mol(cx['pdb_id'])
    if mol is None:
        continue
    f = get_1032_features(mol, cx['pdb_id']).reshape(1, -1)
    f = scaler.transform(f)
    f = selector.transform(f)
    preds.append(float(model.predict(f)[0]))
    trues.append(cx['pkd_true'])

preds = np.array(preds)
trues = np.array(trues)
r, _ = pearsonr(trues, preds)
rho, _ = spearmanr(trues, preds)
mae = np.mean(np.abs(trues - preds))
rmse = np.sqrt(np.mean((trues - preds)**2))
print(f"1032-dim (982+pocket): R={r:.4f} Sp={rho:.4f} MAE={mae:.2f} RMSE={rmse:.2f}")
print(f"vs Phase 2 (982-dim, no pocket): R=0.708")
print(f"vs ECFP-only 39k: R=0.587")
