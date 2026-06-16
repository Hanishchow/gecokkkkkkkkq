"""Compute comprehensive fingerprints for all 24K entries, handling missing keys"""
import pickle, numpy as np
from pathlib import Path
from collections import Counter
from rdkit import Chem
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator
import warnings; warnings.filterwarnings('ignore')

BACKUP = Path(r'C:\Users\yakka\Downloads\Hanish_Brain_Vault\autoresearch_backup')

nf = pickle.load(open(BACKUP / 'lp_new_features_8k.pkl', 'rb'))
pocket = pickle.load(open(BACKUP / 'pocket_features_full.pkl', 'rb'))
X19 = np.load(BACKUP / 'phase2_X.npy')
pdb19 = pickle.load(open(BACKUP / 'phase2_pdb_ids.pkl', 'rb'))

print(f'Loaded {len(nf)} entries, {len(pocket)} pocket features', flush=True)

new_pdbs = set(e['pdb_id'] for e in nf)
old_pdbs = set(pdb19)
print(f'PDB overlap: {len(new_pdbs & old_pdbs)}, Only new: {len(new_pdbs - old_pdbs)}', flush=True)

# Determine feature dimension: ECFP512 + MACCS167 + FCFP256 + RDKit47 = 982
# + optional physics (max 24) + pocket (50)
# Let's make it variable: 982 + 8 physchem + 50 pocket = 1040 (but physics has multiple sizes)
# 
# Actually, let's keep it simple: use the same 982-dim base for compatibility
# and have separate columns for extra features
# 
# OR just compute 982-dim and use pocket only (drop physics for comparability)
# That way we can directly compare with Phase 2 results

print('Computing features from SMILES...', flush=True)
fgen2 = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=512)
fgen3 = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=256)

X = np.zeros((len(nf), 982), dtype=np.float32)  # 982-dim, same as Phase 2
y = np.zeros(len(nf), dtype=np.float32)
pdb_ids = []
pocket_count = 0

for i, entry in enumerate(nf):
    y[i] = entry['affinity']
    pdb_ids.append(entry['pdb_id'])
    
    try:
        mol = Chem.MolFromSmiles(entry['smiles'])
        if mol is None:
            raise ValueError('MolFromSmiles failed')
        X[i, :512] = np.array(fgen2.GetFingerprintAsNumPy(mol), dtype=np.float32)
        X[i, 512:679] = np.array(MACCSkeys.GenMACCSKeys(mol), dtype=np.float32)
        X[i, 679:935] = np.array(fgen3.GetFingerprintAsNumPy(mol), dtype=np.float32)
        X[i, 935:982] = 0  # RDKit = zeros (consistent with Phase 2)
    except:
        # Fall back to stored ECFP if available
        if 'ecfp' in entry:
            X[i, :512] = entry['ecfp']
        X[i, 512:982] = 0
    
    if entry['pdb_id'] in pocket:
        pocket_count += 1
    
    if (i+1) % 5000 == 0:
        print(f'  {i+1}/{len(nf)}', flush=True)

print(f'Done. Pocket coverage: {pocket_count}/{len(nf)}', flush=True)

# Unique check
ecfp_hashes = [hash(row.tobytes()) for row in X[:, :512]]
print(f'Unique ECFP: {len(set(ecfp_hashes))} / {len(X)}', flush=True)
full_hashes = [hash(row.tobytes()) for row in X]
print(f'Unique 982-dim: {len(set(full_hashes))} / {len(X)}', flush=True)

# Now build the 1032-dim version with pocket
X_pocket = np.zeros((len(nf), 1032), dtype=np.float32)
X_pocket[:, :982] = X
for i, pid in enumerate(pdb_ids):
    if pid in pocket:
        X_pocket[i, 982:] = pocket[pid]

np.save(BACKUP / 'all24k_X.npy', X)
np.save(BACKUP / 'all24k_y.npy', y)
np.save(BACKUP / 'all24k_X_pocket.npy', X_pocket)
pickle.dump(pdb_ids, open(BACKUP / 'all24k_pdb_ids.pkl', 'wb'))
print(f'Saved X={X.shape}, X_pocket={X_pocket.shape}, y={y.shape}', flush=True)

# ===== Train on ALL 24K data =====
print('\n=== Training on all 24K data ===', flush=True)
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
import xgboost as xgb
from scipy.stats import pearsonr

# First without pocket
print('982-dim (no pocket):', flush=True)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_r = []
for fold, (tr, va) in enumerate(kf.split(X)):
    s = StandardScaler()
    X_tr_s = s.fit_transform(X[tr])
    X_va_s = s.transform(X[va])
    sel = SelectKBest(f_regression, k=500)
    X_tr_sel = sel.fit_transform(X_tr_s, y[tr])
    m = xgb.XGBRegressor(max_depth=12, n_estimators=2000, learning_rate=0.01,
                          subsample=0.8, colsample_bytree=0.8,
                          min_child_weight=3, gamma=0.1,
                          reg_alpha=0.5, reg_lambda=2.0,
                          random_state=42, n_jobs=-1, verbosity=0)
    m.fit(X_tr_sel, y[tr])
    p = m.predict(sel.transform(X_va_s))
    r, _ = pearsonr(y[va], p)
    cv_r.append(r)
    print(f'  Fold {fold+1}: R={r:.4f}', flush=True)
print(f'982-dim CV R = {np.mean(cv_r):.4f}', flush=True)

# Now with pocket
print('\n1032-dim (with pocket):', flush=True)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_r2 = []
for fold, (tr, va) in enumerate(kf.split(X_pocket)):
    s = StandardScaler()
    X_tr_s = s.fit_transform(X_pocket[tr])
    X_va_s = s.transform(X_pocket[va])
    sel = SelectKBest(f_regression, k=500)
    X_tr_sel = sel.fit_transform(X_tr_s, y[tr])
    m = xgb.XGBRegressor(max_depth=12, n_estimators=2000, learning_rate=0.01,
                          subsample=0.8, colsample_bytree=0.8,
                          min_child_weight=3, gamma=0.1,
                          reg_alpha=0.5, reg_lambda=2.0,
                          random_state=42, n_jobs=-1, verbosity=0)
    m.fit(X_tr_sel, y[tr])
    p = m.predict(sel.transform(X_va_s))
    r, _ = pearsonr(y[va], p)
    cv_r2.append(r)
    print(f'  Fold {fold+1}: R={r:.4f}', flush=True)
print(f'1032-dim CV R = {np.mean(cv_r2):.4f}', flush=True)

# Reference
print(f'\nReference: Phase 2 19K (982-dim) CV = 0.704, CASF-2016 = 0.708')
print(f'Reference: Phase 3b 19K (1032-dim) CV = 0.712, CASF-2016 = 0.717')
print(f'Reference: Phase 5c best CV = 0.721, CASF-2016 = 0.731')
