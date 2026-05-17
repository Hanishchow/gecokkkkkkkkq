"""Ridge with Fixed HETATM - Same as original (30 compounds)"""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error
import json, os, sys, time

sys.path.insert(0, '.')
from patch_parse import parse_pocket_and_ligand

def compute_features(ligand_coords, ligand_types, pocket_coords, pocket_types, center):
    features = np.zeros(60)
    if len(ligand_coords) == 0 or len(pocket_coords) == 0:
        return features
    
    lig_center = ligand_coords.mean(axis=0)
    dist_lig_center = np.linalg.norm(lig_center - center)
    
    all_dists = np.array([np.linalg.norm(lc - pc) for lc in ligand_coords for pc in pocket_coords])
    lig_dists = np.array([min(np.linalg.norm(lc - pc) for pc in pocket_coords) for lc in ligand_coords])
    rec_dists = np.array([min(np.linalg.norm(lc - pc) for lc in ligand_coords) for pc in pocket_coords])
    
    features[0] = np.exp(-dist_lig_center**2 / (2 * 1.5**2))
    features[1] = np.exp(-dist_lig_center**2 / (2 * 3.0**2))
    features[2] = np.exp(-dist_lig_center**2 / (2 * 5.0**2))
    features[3] = np.exp(-all_dists.min()**2 / (2 * 0.5**2))
    features[4] = np.exp(-(all_dists.min() - 3.0)**2 / (2 * 1.0**2))
    features[5] = np.exp(-all_dists.mean()**2 / (2 * 3.0**2))
    features[6] = np.exp(-all_dists.std()**2 / (2 * 2.0**2))
    
    repulsion = sum(d * d for d in all_dists if d < 0)
    features[7] = repulsion
    
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
    
    n_atoms = len(ligand_types)
    n_hydro = sum(1 for t in ligand_types if t in ['C', 'S'])
    n_hbond_don = sum(1 for t in ligand_types if t in ['N', 'O'])
    n_hbond_acc = sum(1 for t in ligand_types if t in ['N', 'O', 'S'])
    n_aromatic = sum(1 for t in ligand_types if t in ['C', 'N'])
    n_positive = sum(1 for t in ligand_types if t == 'N')
    n_negative = sum(1 for t in ligand_types if t in ['O', 'S'])
    
    features[23] = n_hydro / n_atoms
    features[24] = n_hbond_don / n_atoms
    features[25] = n_hbond_acc / n_atoms
    features[26] = n_aromatic / n_atoms
    features[27] = n_positive / n_atoms
    features[28] = n_negative / n_atoms
    features[29] = n_atoms / 100.0
    
    n_pocket = len(pocket_types)
    n_rec_hydro = sum(1 for t in pocket_types if t == 'C')
    n_rec_hbond = sum(1 for t in pocket_types if t in ['N', 'O'])
    features[30] = n_rec_hydro / n_pocket
    features[31] = n_rec_hbond / n_pocket
    features[32] = n_pocket / 200.0
    
    contact_score = hydro_score = hbond_score = 0.0
    for i, lc in enumerate(ligand_coords):
        for j, pc in enumerate(pocket_coords):
            d = np.linalg.norm(lc - pc)
            if d < 4.5:
                contact_score += np.exp(-d**2 / 4.0)
                if ligand_types[i] in ['C', 'S'] and pocket_types[j] == 'C':
                    hydro_score += np.exp(-d**2 / 9.0) if d < 3.5 else 0
                if ligand_types[i] in ['N', 'O'] and pocket_types[j] in ['N', 'O']:
                    hbond_score += np.exp(-d**2 / 4.0)
    
    features[33] = contact_score / max(1, len(all_dists))
    features[34] = hydro_score / max(1, len(all_dists))
    features[35] = hbond_score / max(1, len(all_dists))
    
    surf_lig = sum(1.0 for d in lig_dists if d > 2.0)
    features[37] = surf_lig / n_atoms
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

data_dir = '/mnt/c/Users/yakka/Downloads/geock_110_data'
with open(f'{data_dir}/compounds.json') as f:
    compounds = json.load(f)

# Use same 30 compounds as original
compounds = compounds[:30]

X, y = [], []
t0 = time.time()
for c in compounds:
    pdb_file = f'{data_dir}/{c["pdb_id"]}/{c["pdb_id"]}_pocket.pdb'
    try:
        rec_coords, rec_types, lig_coords, lig_types, _, _ = parse_pocket_and_ligand(pdb_file, cutoff=10.0)
        center = rec_coords.mean(axis=0)
        features = compute_features(lig_coords, lig_types, rec_coords, rec_types, center)
        X.append(features)
        y.append(c['experimental_affinity'])
    except Exception as e:
        pass
print(f"Loaded {len(X)} compounds in {time.time()-t0:.1f}s")

X, y = np.array(X), np.array(y)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
preds = []
for train_idx, val_idx in kf.split(X_scaled):
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_scaled[train_idx], y[train_idx])
    preds.extend(ridge.predict(X_scaled[val_idx]))

r = pearsonr(preds, y)[0]
mae = mean_absolute_error(y, preds)

print(f"\n{'='*60}")
print(f"RESULTS (5-fold CV, Ridge on {len(X)} compounds)")
print(f"{'='*60}")
print(f"Pearson r: {r:.3f}")
print(f"MAE: {mae:.3f} kcal/mol")
print(f"Baseline (original on 30): r = 0.515")
print(f"Change: {r - 0.515:+.3f}")
print(f"{'='*60}")
