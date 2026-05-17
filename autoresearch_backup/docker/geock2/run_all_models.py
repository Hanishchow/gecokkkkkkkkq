"""Compare Ridge, GradientBoosting, ECFP, and Hybrid approaches"""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import json, os, sys, time

sys.path.insert(0, '.')
from patch_parse import parse_pocket_and_ligand

def compute_physics_features(lig_coords, lig_types, rec_coords, rec_types, center):
    """60 physics features"""
    features = np.zeros(60)
    if len(lig_coords) < 3 or len(rec_coords) < 10:
        return features
    
    lig_center = lig_coords.mean(axis=0)
    dist_lig_center = np.linalg.norm(lig_center - center)
    
    all_dists = np.array([np.linalg.norm(lc - pc) for lc in lig_coords for pc in rec_coords])
    lig_dists = np.array([min(np.linalg.norm(lc - pc) for pc in rec_coords) for lc in lig_coords])
    rec_dists = np.array([min(np.linalg.norm(lc - pc) for lc in lig_coords) for pc in rec_coords])
    
    features[0] = np.exp(-dist_lig_center**2 / (2 * 1.5**2))
    features[1] = np.exp(-dist_lig_center**2 / (2 * 3.0**2))
    features[2] = np.exp(-dist_lig_center**2 / (2 * 5.0**2))
    features[3] = np.exp(-all_dists.min()**2 / (2 * 0.5**2))
    features[4] = np.exp(-(all_dists.min() - 3.0)**2 / (2 * 1.0**2))
    features[5] = np.exp(-all_dists.mean()**2 / (2 * 3.0**2))
    features[6] = np.exp(-all_dists.std()**2 / (2 * 2.0**2))
    
    features[7] = sum(d * d for d in all_dists if d < 0)
    
    for i, d in enumerate([2.0, 3.0, 4.0, 5.0, 6.0, 8.0]):
        features[8+i] = np.sum(all_dists < d) / len(all_dists)
    
    features[14] = lig_dists.min()
    features[15] = lig_dists.mean()
    features[16] = lig_dists.std()
    features[17] = np.percentile(lig_dists, 25)
    features[18] = np.percentile(lig_dists, 50)
    features[19] = np.percentile(lig_dists, 75)
    features[20] = rec_dists.min()
    features[21] = rec_dists.mean()
    features[22] = rec_dists.std()
    
    n_atoms = len(lig_types)
    n_hydro = sum(1 for t in lig_types if t in ['C', 'S'])
    n_hbond_don = sum(1 for t in lig_types if t in ['N', 'O'])
    n_hbond_acc = sum(1 for t in lig_types if t in ['N', 'O', 'S'])
    n_aromatic = sum(1 for t in lig_types if t in ['C', 'N'])
    n_positive = sum(1 for t in lig_types if t == 'N')
    n_negative = sum(1 for t in lig_types if t in ['O', 'S'])
    
    features[23] = n_hydro / n_atoms
    features[24] = n_hbond_don / n_atoms
    features[25] = n_hbond_acc / n_atoms
    features[26] = n_aromatic / n_atoms
    features[27] = n_positive / n_atoms
    features[28] = n_negative / n_atoms
    features[29] = n_atoms / 100.0
    
    n_pocket = len(rec_types)
    n_rec_hydro = sum(1 for t in rec_types if t == 'C')
    n_rec_hbond = sum(1 for t in rec_types if t in ['N', 'O'])
    features[30] = n_rec_hydro / n_pocket
    features[31] = n_rec_hbond / n_pocket
    features[32] = n_pocket / 200.0
    
    contact = hydro = hbond = 0.0
    for i, lc in enumerate(lig_coords):
        for j, pc in enumerate(rec_coords):
            d = np.linalg.norm(lc - pc)
            if d < 4.5:
                contact += np.exp(-d**2 / 4.0)
                if lig_types[i] in ['C', 'S'] and rec_types[j] == 'C':
                    hydro += np.exp(-d**2 / 9.0) if d < 3.5 else 0
                if lig_types[i] in ['N', 'O'] and rec_types[j] in ['N', 'O']:
                    hbond += np.exp(-d**2 / 4.0)
    
    features[33] = contact / max(1, len(all_dists))
    features[34] = hydro / max(1, len(all_dists))
    features[35] = hbond / max(1, len(all_dists))
    
    features[37] = sum(1.0 for d in lig_dists if d > 2.0) / n_atoms
    features[38] = (n_positive - n_negative) / n_atoms
    features[39] = features[38] * (n_rec_hydro - n_rec_hbond) / n_pocket
    features[40] = dist_lig_center
    features[41] = np.sin(dist_lig_center / 10.0)
    features[42] = np.cos(dist_lig_center / 10.0)
    
    hist, _ = np.histogram(all_dists, bins=10, range=(0, 10))
    features[43:53] = hist / len(all_dists)
    for i, p in enumerate([5, 10, 25, 50, 75, 90, 95]):
        features[53+i] = np.percentile(all_dists, p) / 10.0
    
    return features

def compute_ecfp_fingerprint(smiles, radius=2, bits=1024):
    """ECFP4 fingerprint"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(bits)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=bits)
        return np.array(fp)
    except:
        return np.zeros(bits)

def compute_descriptors(smiles):
    """Molecular descriptors"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(10)
        return np.array([
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumHeteroatoms(mol),
            Descriptors.RingCount(mol),
            Descriptors.FractionCSP3(mol),
            Descriptors.NumAromaticRings(mol),
        ])
    except:
        return np.zeros(10)

# Load data
data_dir = '/mnt/c/Users/yakka/Downloads/geock_110_data'
with open(f'{data_dir}/compounds.json') as f:
    compounds = json.load(f)

# Use first 30 for speed
compounds = compounds[:30]

X_phys, X_ecfp, X_desc, y = [], [], [], []
t0 = time.time()

for c in compounds:
    pdb_file = f'{data_dir}/{c["pdb_id"]}/{c["pdb_id"]}_pocket.pdb'
    try:
        rec_coords, rec_types, lig_coords, lig_types, _, _ = parse_pocket_and_ligand(pdb_file, cutoff=10.0)
        center = rec_coords.mean(axis=0)
        
        phys = compute_physics_features(lig_coords, lig_types, rec_coords, rec_types, center)
        ecfp = compute_ecfp_fingerprint(c.get('smiles', ''))
        desc = compute_descriptors(c.get('smiles', ''))
        
        X_phys.append(phys)
        X_ecfp.append(ecfp)
        X_desc.append(desc)
        y.append(c['experimental_affinity'])
    except Exception as e:
        pass

print(f"Loaded {len(X_phys)} compounds in {time.time()-t0:.1f}s")

X_phys = np.array(X_phys)
X_ecfp = np.array(X_ecfp)
X_desc = np.array(X_desc)
y = np.array(y)

print(f"Physics: {X_phys.shape}, ECFP: {X_ecfp.shape}, Desc: {X_desc.shape}")

scaler_phys = StandardScaler()
X_phys_s = scaler_phys.fit_transform(X_phys)

X_hybrid = np.hstack([X_phys, X_ecfp, X_desc])
scaler_hybrid = StandardScaler()
X_hybrid_s = scaler_hybrid.fit_transform(X_hybrid)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

def evaluate(X_scaled, name):
    preds = []
    for train_idx, val_idx in kf.split(X_scaled):
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_scaled[train_idx], y[train_idx])
        preds.extend(ridge.predict(X_scaled[val_idx]))
    r = pearsonr(preds, y)[0]
    mae = mean_absolute_error(y, preds)
    return r, mae, preds

def evaluate_gb(X, name):
    preds = []
    for train_idx, val_idx in kf.split(X):
        gb = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
        gb.fit(X[train_idx], y[train_idx])
        preds.extend(gb.predict(X[val_idx]))
    r = pearsonr(preds, y)[0]
    mae = mean_absolute_error(y, preds)
    return r, mae, preds

print(f"\n{'='*70}")
print(f"MODEL COMPARISON (5-fold CV, {len(X_phys)} compounds)")
print(f"{'='*70}")

# A) Ridge on physics only
r, mae, _ = evaluate(X_phys_s, "Ridge+Physics")
print(f"A) Ridge + Physics (60D):        r={r:.3f}, MAE={mae:.3f}")

# B) Ridge on ECFP
scaler_ecfp = StandardScaler()
X_ecfp_s = scaler_ecfp.fit_transform(X_ecfp)
r, mae, _ = evaluate(X_ecfp_s, "Ridge+ECFP")
print(f"B) Ridge + ECFP (1024D):         r={r:.3f}, MAE={mae:.3f}")

# C) Ridge on descriptors
scaler_desc = StandardScaler()
X_desc_s = scaler_desc.fit_transform(X_desc)
r, mae, _ = evaluate(X_desc_s, "Ridge+Desc")
print(f"   Ridge + Descriptors (10D):     r={r:.3f}, MAE={mae:.3f}")

# A+B) Physics + ECFP
X_phys_ecfp = np.hstack([X_phys_s, X_ecfp_s])
r, mae, _ = evaluate(X_phys_ecfp, "Ridge+Physics+ECFP")
print(f"   Ridge + Physics+ECFP (1084D): r={r:.3f}, MAE={mae:.3f}")

# B+C) ECFP + Desc
X_ecfp_desc = np.hstack([X_ecfp_s, X_desc_s])
r, mae, _ = evaluate(X_ecfp_desc, "Ridge+ECFP+Desc")
print(f"   Ridge + ECFP+Desc (1034D):    r={r:.3f}, MAE={mae:.3f}")

# C) Hybrid (all)
r, mae, _ = evaluate(X_hybrid_s, "Ridge+Hybrid")
print(f"C) Ridge + Hybrid (1094D):       r={r:.3f}, MAE={mae:.3f}")

print(f"\n{'='*70}")
print("GRADIENT BOOSTING VARIANTS")
print(f"{'='*70}")

# GB on physics
r, mae, _ = evaluate_gb(X_phys, "GB+Physics")
print(f"A) GB + Physics:                 r={r:.3f}, MAE={mae:.3f}")

# GB on hybrid
r, mae, _ = evaluate_gb(X_hybrid, "GB+Hybrid")
print(f"C) GB + Hybrid:                  r={r:.3f}, MAE={mae:.3f}")

print(f"\n{'='*70}")
print(f"Baseline (original, wrong coords): r = 0.515")
print(f"{'='*70}")

print(f"\n{'='*70}")
print("ADDITIONAL MODELS")
print(f"{'='*70}")

from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor

# PLSRegression (good for high-dim, small n)
def evaluate_pls(X, name, n_components=5):
    preds = []
    for train_idx, val_idx in kf.split(X):
        pls = PLSRegression(n_components=min(n_components, len(train_idx)-1))
        pls.fit(X[train_idx], y[train_idx])
        pred = pls.predict(X[val_idx]).flatten()
        preds.extend(pred)
    r = pearsonr(preds, y)[0]
    mae = mean_absolute_error(y, preds)
    return r, mae

# RF on hybrid
def evaluate_rf(X, name):
    preds = []
    for train_idx, val_idx in kf.split(X):
        rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
        rf.fit(X[train_idx], y[train_idx])
        preds.extend(rf.predict(X[val_idx]))
    r = pearsonr(preds, y)[0]
    mae = mean_absolute_error(y, preds)
    return r, mae

# Try different n_components for PLS
for n in [3, 5, 10, 15]:
    r, mae = evaluate_pls(X_hybrid, f"PLS({n})", n)
    print(f"PLS ({n} comp) + Hybrid:  r={r:.3f}, MAE={mae:.3f}")

# RF
r, mae = evaluate_rf(X_hybrid, "RF+Hybrid")
print(f"RF + Hybrid:                r={r:.3f}, MAE={mae:.3f}")

# Try combining physics features with different alpha for Ridge
for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
    preds = []
    for train_idx, val_idx in kf.split(X_hybrid_s):
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_hybrid_s[train_idx], y[train_idx])
        preds.extend(ridge.predict(X_hybrid_s[val_idx]))
    r = pearsonr(preds, y)[0]
    mae = mean_absolute_error(y, preds)
    print(f"Ridge (alpha={alpha}) + Hybrid: r={r:.3f}, MAE={mae:.3f}")

print(f"\n{'='*70}")
