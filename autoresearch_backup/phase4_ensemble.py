"""Phase 4: Ensemble model — average 5 XGBoost models with different seeds"""
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

# Load training data
print("=== Loading ===")
X = np.load(BACKUP / 'phase2_X.npy')
y = np.load(BACKUP / 'phase2_y.npy')
pdb_ids = pickle.load(open(BACKUP / 'phase2_pdb_ids.pkl', 'rb'))
pocket_full = pickle.load(open(BACKUP / 'pocket_features_full.pkl', 'rb'))

X_aug = np.zeros((len(X), 1032), dtype=np.float32)
X_aug[:, :982] = X
for i, pid in enumerate(pdb_ids):
    if pid in pocket_full:
        X_aug[i, 982:] = pocket_full[pid]

print(f'X={X_aug.shape}, pocket coverage={sum(1 for i,p in enumerate(pdb_ids) if p in pocket_full)}/{len(X)}')

# Phase 4: Ensemble with 5 seeds
seeds = [42, 1, 123, 7, 999]
models_data = []
print("\n=== Phase 4: Training Ensemble ===")

for seed in seeds:
    # 5-fold CV to get individual score
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
                             random_state=seed, n_jobs=-1, verbosity=0)
        m.fit(X_tr_sel, y_tr)
        p = m.predict(X_va_sel)
        r, _ = pearsonr(y_va, p)
        cv_r.append(r)
    print(f"  Seed {seed}: CV R={np.mean(cv_r):.4f}")
    
    # Train final
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
    models_data.append((m, s, sel))
    print(f"  Seed {seed}: final model trained")

# Save ensemble
ensemble_data = [{'model': m, 'scaler': s, 'selector': sel, 'seed': sd}
                 for (m, s, sel), sd in zip(models_data, seeds)]
pickle.dump(ensemble_data, open(BACKUP / 'geock_phase4_ensemble.pkl', 'wb'))
print("Saved: geock_phase4_ensemble.pkl")

# ===== Evaluate on CASF-2016 =====
print("\n=== Phase 4: CASF-2016 Ensemble ===")
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

# Individual model predictions
all_preds = np.zeros((len(casf), len(seeds)))
trues = []
for i, cx in enumerate(casf):
    mol = get_mol(cx['pdb_id'])
    if mol is None:
        continue
    f = get_feats(mol, cx['pdb_id']).reshape(1, -1)
    for j, (m, s, sel) in enumerate(models_data):
        fs = s.transform(f)
        fs = sel.transform(fs)
        all_preds[i, j] = float(m.predict(fs)[0])
    trues.append(cx['pkd_true'])

trues = np.array(trues)
valid = ~np.all(all_preds == 0, axis=1)
all_preds = all_preds[valid]
trues = trues[valid]

print(f"\nIndividual model performance on CASF-2016:")
for j, sd in enumerate(seeds):
    r, _ = pearsonr(trues, all_preds[:, j])
    print(f"  Seed {sd}: R={r:.4f}")

# Ensemble: simple average
ensemble_pred = np.mean(all_preds, axis=1)
r, _ = pearsonr(trues, ensemble_pred)
rho, _ = spearmanr(trues, ensemble_pred)
mae = np.mean(np.abs(trues - ensemble_pred))
rmse = np.sqrt(np.mean((trues - ensemble_pred)**2))
print(f"\nEnsemble (avg of {len(seeds)} seeds):")
print(f"  R={r:.4f} Sp={rho:.4f} MAE={mae:.2f} RMSE={rmse:.2f}")

# Phase 5: Try k=700 (more features retained)
print("\n=== Phase 5: k=700 feature selection ===")
models_data700 = []
for seed in seeds[:3]:  # Use 3 seeds for speed
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_r = []
    for fold, (tr, va) in enumerate(kf.split(X_aug)):
        X_tr, X_va = X_aug[tr], X_aug[va]
        y_tr, y_va = y[tr], y[va]
        s = StandardScaler()
        X_tr_s = s.fit_transform(X_tr)
        X_va_s = s.transform(X_va)
        sel = SelectKBest(f_regression, k=700)
        X_tr_sel = sel.fit_transform(X_tr_s, y_tr)
        X_va_sel = sel.transform(X_va_s)
        m = xgb.XGBRegressor(max_depth=12, n_estimators=500, learning_rate=0.01,
                             subsample=0.8, colsample_bytree=0.8,
                             min_child_weight=3, gamma=0.1,
                             random_state=seed, n_jobs=-1, verbosity=0)
        m.fit(X_tr_sel, y_tr)
        p = m.predict(X_va_sel)
        r, _ = pearsonr(y_va, p)
        cv_r.append(r)
    print(f"  Seed {seed} (k=700): CV R={np.mean(cv_r):.4f}")
    s = StandardScaler()
    Xs = s.fit_transform(X_aug)
    sel = SelectKBest(f_regression, k=700)
    X_sel = sel.fit_transform(Xs, y)
    m = xgb.XGBRegressor(max_depth=12, n_estimators=500, learning_rate=0.01,
                          subsample=0.8, colsample_bytree=0.8,
                          min_child_weight=3, gamma=0.1,
                          random_state=seed, n_jobs=-1, verbosity=0)
    m.fit(X_sel, y)
    models_data700.append((m, s, sel))

# Evaluate k=700 model
all_preds700 = np.zeros((len(casf), 3))
for i, cx in enumerate(casf):
    mol = get_mol(cx['pdb_id'])
    if mol is None:
        continue
    f = get_feats(mol, cx['pdb_id']).reshape(1, -1)
    for j, (m, s, sel) in enumerate(models_data700):
        fs = s.transform(f)
        fs = sel.transform(fs)
        all_preds700[i, j] = float(m.predict(fs)[0])

all_preds700 = all_preds700[valid]
ensemble700 = np.mean(all_preds700, axis=1)
r700, _ = pearsonr(trues, ensemble700)
print(f"  Ensemble (k=700, 3 seeds): R={r700:.4f}")

# Phase 6: Evaluate best model on CASF-2007
print("\n=== Phase 6: CASF-2007 ===")
CASF07_DIR = Path(r'C:\Users\yakka\Downloads\CASF-2007')
if CASF07_DIR.exists():
    with open(CASF07_DIR / 'index' / 'INDEX_core_data.2007', 'r') as f:
        casf07 = []
        for line in f:
            if line.startswith('#') or line.strip() == '':
                continue
            parts = line.strip().split()
            if len(parts) >= 6:
                try:
                    casf07.append({'pdb_id': parts[0], 'pkd_true': float(parts[5])})
                except:
                    pass
    print(f"CASF-2007: {len(casf07)} complexes")
    
    preds07 = np.zeros((len(casf07), len(seeds)))
    trues07 = []
    for i, cx in enumerate(casf07):
        mol = get_mol(cx['pdb_id'])
        if mol is None:
            mol2 = CASF07_DIR / 'coreset' / f'{cx["pdb_id"]}_ligand.mol2'
            sdf = CASF07_DIR / 'coreset' / f'{cx["pdb_id"]}_ligand.sdf'
            if mol2.exists():
                mol = Chem.MolFromMol2File(str(mol2), removeHs=False)
            if mol is None and sdf.exists():
                mol = next((m for m in Chem.SDMolSupplier(str(sdf), removeHs=False) if m), None)
        if mol is None:
            continue
        f = get_feats(mol, cx['pdb_id']).reshape(1, -1)
        for j, (m, s, sel) in enumerate(models_data):
            fs = s.transform(f)
            fs = sel.transform(fs)
            preds07[i, j] = float(m.predict(fs)[0])
        trues07.append(cx['pkd_true'])
    
    trues07 = np.array(trues07)
    v07 = ~np.all(preds07 == 0, axis=1)
    preds07 = preds07[v07]
    trues07 = trues07[v07]
    ens07 = np.mean(preds07, axis=1)
    r07, _ = pearsonr(trues07, ens07)
    print(f"CASF-2007 ensemble: R={r07:.4f} (N={len(trues07)})")
else:
    print(f"CASF-2007 directory not found")

print("\n=== SUMMARY ===")
print(f"Baseline ECFP-only 39K:      R=0.587")
print(f"Phase 2 982-dim (19K):       R=0.708")
print(f"Phase 3b +pocket (18K):      R=0.717")
print(f"Phase 4 Ensemble (5 seeds):  R={r:.4f}")
print(f"Phase 5 k=700 ensemble:      R={r700:.4f}")
