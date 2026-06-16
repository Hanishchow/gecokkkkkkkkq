"""
Train GEOCK with BindingDB augmentation + evaluate on CASF-2016
Uses same Phase 5c pipeline: XGBoost, t=2000, k=500, 1032-dim
"""
import pickle, numpy as np
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
import xgboost as xgb
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
import warnings; warnings.filterwarnings('ignore')

BACKUP = Path(r'C:\Users\yakka\Downloads\Hanish_Brain_Vault\autoresearch_backup')
CASF_DIR = Path(r'C:\Users\yakka\Downloads\CASF-2016\CASF-2016')
CORESET_DAT = CASF_DIR / 'power_scoring' / 'CoreSet.dat'

def compute_982_fp(mol):
    ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, 4, nBits=512)
    maccs = MACCSkeys.GenMACCSKeys(mol)
    fcfp = AllChem.GetMorganFingerprintAsBitVect(mol, 4, nBits=300, useFeatures=True)
    fp = np.zeros(982, dtype=np.float32)
    for i in range(512):
        fp[i] = ecfp.GetBit(i)
    for i in range(167):
        fp[512+i] = maccs.GetBit(i)
    for i in range(300):
        fp[679+i] = fcfp.GetBit(i)
    return fp

# === 1. Load training data ===
print("=== Loading Phase 2 (19K) ===")
X19 = np.load(BACKUP / 'phase2_X.npy')
y19 = np.load(BACKUP / 'phase2_y.npy')
pdb_ids = pickle.load(open(BACKUP / 'phase2_pdb_ids.pkl', 'rb'))
pocket_full = pickle.load(open(BACKUP / 'pocket_features_full.pkl', 'rb'))
# Build 1032-dim
X_phase2 = np.zeros((len(X19), 1032), dtype=np.float32)
X_phase2[:, :982] = X19
for i, pid in enumerate(pdb_ids):
    if pid in pocket_full:
        X_phase2[i, 982:] = pocket_full[pid]
print(f"  Phase 2: {X_phase2.shape}, y range: {y19.min():.2f}-{y19.max():.2f}")

print("\n=== Loading BindingDB (556K new) ===")
X_bind = np.load(BACKUP / 'bindingdb_X_new.npy')
y_bind = np.load(BACKUP / 'bindingdb_y_new.npy')
# Zero-pad to 1032-dim (no pocket features for BindingDB)
X_bind_aug = np.zeros((len(X_bind), 1032), dtype=np.float32)
X_bind_aug[:, :982] = X_bind.astype(np.float32)
print(f"  BindingDB: {X_bind_aug.shape}, y range: {y_bind.min():.2f}-{y_bind.max():.2f}")

# === 2. Combine ===
print("\n=== Combining datasets ===")
X_all = np.concatenate([X_phase2, X_bind_aug])
y_all = np.concatenate([y19, y_bind])
print(f"  Combined: {X_all.shape}, {len(X_all)} entries")
print(f"  y: mean={y_all.mean():.3f}, std={y_all.std():.3f}")

# === 3. Train Phase 5c pipeline ===
print("\n=== Training XGBoost (t=2000, k=500) ===")
ss = StandardScaler()
X_s = ss.fit_transform(X_all)
sel = SelectKBest(f_regression, k=500)
X_sel = sel.fit_transform(X_s, y_all)
model = xgb.XGBRegressor(
    max_depth=12, n_estimators=2000, learning_rate=0.01,
    subsample=0.8, colsample_bytree=0.8,
    min_child_weight=3, gamma=0.1,
    reg_alpha=0.5, reg_lambda=2.0,
    random_state=42, n_jobs=-1, verbosity=0
)
model.fit(X_sel, y_all)
print("  Model trained")

# === 4. Evaluate on CASF-2016 ===
print("\n=== CASF-2016 Evaluation ===")
complexes = []
with open(CORESET_DAT, 'r') as f:
    f.readline()
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 4:
            complexes.append({'pdb_id': parts[0], 'pkd': float(parts[3])})
print(f"  {len(complexes)} complexes")

true_vals, pred_vals = [], []
for c in complexes:
    pid = c['pdb_id']
    mol2_path = CASF_DIR / 'coreset' / pid / f'{pid}_ligand.mol2'
    sdf_path = CASF_DIR / 'coreset' / pid / f'{pid}_ligand.sdf'
    mol = None
    if mol2_path.exists():
        mol = Chem.MolFromMol2File(str(mol2_path))
    if mol is None and sdf_path.exists():
        mol = Chem.SDMolSupplier(str(sdf_path))[0]
    if mol is None:
        print(f"  WARN: Can't load ligand for {pid}")
        continue
    fp = compute_982_fp(mol)
    X_test = np.zeros(1032, dtype=np.float32)
    X_test[:982] = fp
    X_test_s = ss.transform(X_test.reshape(1, -1))
    X_test_sel = sel.transform(X_test_s)
    pred = model.predict(X_test_sel)[0]
    true_vals.append(c['pkd'])
    pred_vals.append(pred)

true_arr = np.array(true_vals)
pred_arr = np.array(pred_vals)
r_pearson, _ = pearsonr(true_arr, pred_arr)
r_spearman, _ = spearmanr(true_arr, pred_arr)
print(f"\n  CASF-2016 Results ({len(true_vals)} complexes):")
print(f"  Pearson R: {r_pearson:.4f}")
print(f"  Spearman R: {r_spearman:.4f}")
print(f"  RMSE: {np.sqrt(np.mean((true_arr - pred_arr)**2)):.4f}")
print(f"  MAE: {np.mean(np.abs(true_arr - pred_arr)):.4f}")

# Save predictions
out = BACKUP / 'geock_bindingdb_predictions.csv'
with open(out, 'w') as f:
    f.write('pdb_id,pkd_true,pkd_pred\n')
    for c, t, p in zip(complexes, true_vals, pred_vals):
        f.write(f"{c['pdb_id']},{t:.4f},{p:.4f}\n")
print(f"\n  Predictions saved to {out}")

# Save model
model_data = {
    'model': model,
    'scaler': ss,
    'selector': sel,
    'config': 'BindingDB+Phase2, k=500, t=2000',
    'n_samples': len(X_all),
    'casf2016_r': r_pearson
}
pickle.dump(model_data, open(BACKUP / 'geock_bindingdb_final.pkl', 'wb'))
print(f"  Model saved to geock_bindingdb_final.pkl")

# Comparison with best previous
print(f"\n=== Comparison ===")
print(f"  Phase 5c (19K only):  CASF-2016 R = 0.731")
print(f"  BindingDB (556K+19K): CASF-2016 R = {r_pearson:.4f}")
