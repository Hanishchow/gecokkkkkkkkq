"""Phase 1: Train XGBoost on ECFP4 + Pocket Features, evaluate on CASF-2016"""
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
from rdkit.Chem import AllChem
import sys

sys.path.insert(0, r'C:\Users\yakka\Downloads\Hanish_Brain_Vault\autoresearch_backup')
from simple_pocket import extract_simple_pocket

BACKUP = Path(r'C:\Users\yakka\Downloads\Hanish_Brain_Vault\autoresearch_backup')
CASF_DIR = Path(r'C:\Users\yakka\Downloads\CASF-2016\CASF-2016')

# ===== STEP 1: Load and prepare training data =====
print("=== Loading training data ===")

# Read data.tsv (SMILES + affinity + PDB IDs)
with open(BACKUP / 'data.tsv') as f:
    lines = f.read().strip().split('\n')
header = lines[0].split('\t')
id_col = header.index('pdb_id')
smiles_col = header.index('smiles')
pkd_col = header.index('pKd')
tsv_entries = []
for line in lines[1:]:
    parts = line.split('\t')
    tsv_entries.append({
        'pdb_id': parts[id_col],
        'smiles': parts[smiles_col],
        'pKd': float(parts[pkd_col])
    })
print(f"data.tsv: {len(tsv_entries)} entries")

# Load pre-computed pocket features
ps = pickle.load(open(BACKUP / 'training_pocket_simple.pkl', 'rb'))
print(f"training_pocket_simple.pkl: {len(ps)} PDB IDs")

# Build training set: overlap by PDB ID
train_data = [e for e in tsv_entries if e['pdb_id'] in ps]
print(f"Overlap: {len(train_data)} compounds with SMILES + affinity + pocket features")

# Compute ECFP and build feature matrix
def ecfp_from_smiles(smiles, nbits=512, radius=2):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits), dtype=np.float32)

X_list, y_list = [], []
skipped = 0
for e in train_data:
    fp = ecfp_from_smiles(e['smiles'])
    if fp is None:
        skipped += 1
        continue
    pocket = ps[e['pdb_id']]  # 50-dim
    X_list.append(np.concatenate([fp, pocket]))  # 562-dim
    y_list.append(e['pKd'])

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.float32)
print(f"Training set: X={X.shape}, y={y.shape}, skipped={skipped}")
print(f"pKd range: {y.min():.2f} - {y.max():.2f}, mean={y.mean():.2f}")

# ===== STEP 2: 5-fold Cross-Validation =====
print("\n=== 5-Fold Cross Validation ===")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]
    
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)
    
    selector = SelectKBest(f_regression, k=400)
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

# ===== STEP 3: Train Final Model on All Data =====
print("\n=== Training Final Model ===")
scaler = StandardScaler()
X_s = scaler.fit_transform(X)
selector = SelectKBest(f_regression, k=400)
X_sel = selector.fit_transform(X_s, y)

model = xgb.XGBRegressor(
    max_depth=12, n_estimators=500, learning_rate=0.01,
    subsample=0.8, colsample_bytree=0.8,
    min_child_weight=3, gamma=0.1,
    reg_alpha=0.5, reg_lambda=2.0,
    random_state=42, n_jobs=-1, verbosity=0
)
model.fit(X_sel, y)

# Save model
model_data = {
    'model': model, 'scaler': scaler, 'selector': selector,
    'cv_r': float(np.mean(cv_scores)), 'cv_std': float(np.std(cv_scores)),
    'n_samples': len(y), 'n_features': X.shape[1],
    'k': 400, 'date': 'phase1_pocket',
    'config': 'ecfp4_512 + pocket_50 (simple_pocket), XGB depth=12, 500 trees'
}
pickle.dump(model_data, open(BACKUP / 'geock_phase1_pocket.pkl', 'wb'))
print("Saved: geock_phase1_pocket.pkl")

# ===== STEP 4: Evaluate on CASF-2016 =====
print("\n=== CASF-2016 Evaluation ===")

# Load CASF complexes
with open(CASF_DIR / 'power_scoring' / 'CoreSet.dat') as f:
    f.readline()
    casf_complexes = []
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 4:
            casf_complexes.append({'pdb_id': parts[0], 'pkd_true': float(parts[3])})

def get_ligand_mol(pid):
    mol2 = CASF_DIR / 'coreset' / pid / f'{pid}_ligand.mol2'
    sdf = CASF_DIR / 'coreset' / pid / f'{pid}_ligand.sdf'
    mol = Chem.MolFromMol2File(str(mol2), removeHs=False) if mol2.exists() else None
    if mol is None and sdf.exists():
        suppl = Chem.SDMolSupplier(str(sdf), removeHs=False)
        mol = next((m for m in suppl if m), None)
    return mol

def get_pocket_features(pid):
    pocket_pdb = CASF_DIR / 'coreset' / pid / f'{pid}_pocket.pdb'
    if pocket_pdb.exists():
        return extract_simple_pocket(str(pocket_pdb))
    return np.zeros(50, dtype=np.float32)

preds, trues = [], []
for cx in casf_complexes:
    mol = get_ligand_mol(cx['pdb_id'])
    if mol is None:
        continue
    ecfp = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=512), dtype=np.float32)
    pocket = get_pocket_features(cx['pdb_id'])
    X_casf = np.concatenate([ecfp, pocket]).reshape(1, -1)
    X_casf = scaler.transform(X_casf)
    X_casf = selector.transform(X_casf)
    preds.append(float(model.predict(X_casf)[0]))
    trues.append(cx['pkd_true'])

preds = np.array(preds)
trues = np.array(trues)
r, _ = pearsonr(trues, preds)
rho, _ = spearmanr(trues, preds)
mae = np.mean(np.abs(trues - preds))
rmse = np.sqrt(np.mean((trues - preds)**2))

print(f"CASF-2016 (ECFP4 + Pocket Features):")
print(f"  R={r:.4f} Sp={rho:.4f} MAE={mae:.2f} RMSE={rmse:.2f} (N={len(preds)})")
print(f"  vs baseline ECFP-only Deep Trees: R=0.575")
print(f"  vs baseline ECFP-only XGBoost 39k: R=0.587")
