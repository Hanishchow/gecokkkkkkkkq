"""Ridge with Fixed HETATM Parsing - Optimized"""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error
import json, os, sys, time

sys.path.insert(0, '/mnt/c/Users/yakka/Downloads/final/geock')
from patch_parse import parse_pocket_and_ligand

def compute_features_fast(lig_coords, lig_types, rec_coords, rec_types):
    """Optimized feature computation."""
    features = np.zeros(60)
    if len(lig_coords) < 3 or len(rec_coords) < 10:
        return features
    
    # Vectorized distances
    diffs = lig_coords[:, np.newaxis, :] - rec_coords[np.newaxis, :, :]
    all_dists = np.sqrt(np.sum(diffs ** 2, axis=2))
    
    lig_dists = all_dists.min(axis=1)
    rec_dists = all_dists.min(axis=0)
    all_dists_flat = all_dists.flatten()
    
    features[0] = np.exp(-lig_dists.mean()**2 / (2 * 1.5**2))
    features[1] = np.exp(-lig_dists.mean()**2 / (2 * 3.0**2))
    features[2] = np.exp(-lig_dists.mean()**2 / (2 * 5.0**2))
    features[3] = np.exp(-all_dists.min()**2 / (2 * 0.5**2))
    features[4] = np.exp(-(all_dists.min() - 3.0)**2 / (2 * 1.0**2))
    features[5] = np.exp(-all_dists.mean()**2 / (2 * 3.0**2))
    features[6] = np.exp(-all_dists.std()**2 / (2 * 2.0**2))
    
    features[7] = np.sum(np.maximum(0, -all_dists_flat)**2)
    
    for i, d in enumerate([2.0, 3.0, 4.0, 5.0, 6.0, 8.0]):
        features[8+i] = np.sum(all_dists_flat < d) / len(all_dists_flat)
    
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
    n_N = sum(1 for t in lig_types if t == 'NA')
    n_O = sum(1 for t in lig_types if t == 'OA')
    n_C = sum(1 for t in lig_types if t == 'C')
    
    features[23] = (n_C + sum(1 for t in lig_types if t == 'SA')) / n_atoms
    features[24] = n_N / n_atoms
    features[25] = (n_N + n_O + sum(1 for t in lig_types if t == 'SA')) / n_atoms
    features[26] = 0.5
    features[27] = n_N / n_atoms
    features[28] = (n_O + sum(1 for t in lig_types if t == 'SA')) / n_atoms
    features[29] = n_atoms / 100.0
    
    n_pocket = len(rec_types)
    n_rec_C = sum(1 for t in rec_types if t == 'C')
    n_rec_N = sum(1 for t in rec_types if t in ['NA', 'N'])
    n_rec_O = sum(1 for t in rec_types if t in ['OA', 'O'])
    features[30] = n_rec_C / n_pocket
    features[31] = (n_rec_N + n_rec_O) / n_pocket
    features[32] = n_pocket / 200.0
    
    # Vectorized interaction scoring
    close_mask = all_dists < 4.5
    features[33] = np.sum(np.exp(-all_dists**2 / 4.0)) / max(1, len(all_dists_flat))
    
    lig_C = np.array([t == 'C' or t == 'SA' for t in lig_types])
    rec_C = np.array([t == 'C' for t in rec_types])
    lig_HB = np.array([t in ['NA', 'N', 'OA', 'O', 'SA'] for t in lig_types])
    rec_HB = np.array([t in ['NA', 'N', 'OA', 'O'] for t in rec_types])
    
    close_C = close_mask & lig_C[:, np.newaxis] & rec_C[np.newaxis, :]
    close_HB = close_mask & lig_HB[:, np.newaxis] & rec_HB[np.newaxis, :]
    
    hydro = np.where(close_C & (all_dists < 3.5), np.exp(-all_dists**2 / 9.0), 0)
    hbond = np.where(close_HB, np.exp(-all_dists**2 / 4.0), 0)
    
    features[34] = np.sum(hydro) / max(1, len(all_dists_flat))
    features[35] = np.sum(hbond) / max(1, len(all_dists_flat))
    
    features[37] = np.sum(lig_dists > 2.0) / n_atoms
    features[38] = (n_N - n_O) / n_atoms
    features[39] = features[38] * (n_rec_C - n_rec_N - n_rec_O) / n_pocket
    features[40] = lig_dists.mean()
    features[41] = np.sin(lig_dists.mean() / 10.0)
    features[42] = np.cos(lig_dists.mean() / 10.0)
    
    hist, _ = np.histogram(all_dists_flat, bins=10, range=(0, 10))
    features[43:53] = hist / len(all_dists_flat)
    
    for i, p in enumerate([5, 10, 25, 50, 75, 90, 95]):
        features[53+i] = np.percentile(all_dists_flat, p) / 10.0
    
    return features

def main():
    data_dir = '/mnt/c/Users/yakka/Downloads/geock_110_data'
    with open(f'{data_dir}/compounds.json') as f:
        compounds = json.load(f)
    
    X, y = [], []
    t0 = time.time()
    for c in compounds:
        pdb_file = f'{data_dir}/{c["pdb_id"]}/{c["pdb_id"]}_pocket.pdb'
        try:
            rec_coords, rec_types, lig_coords, lig_types, _, _ = \
                parse_pocket_and_ligand(pdb_file, cutoff=10.0)
            features = compute_features_fast(lig_coords, lig_types, rec_coords, rec_types)
            X.append(features)
            y.append(c['experimental_affinity'])
        except:
            pass
    print(f"Loaded {len(X)} compounds in {time.time()-t0:.1f}s")
    
    X, y = np.array(X), np.array(y)
    print(f"Data: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Affinity range: {y.min():.2f} to {y.max():.2f}")
    
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
    
    print(f"\n" + "=" * 60)
    print(f"RESULTS (5-fold CV, Ridge on {len(X)} compounds)")
    print(f"=" * 60)
    print(f"Pearson r: {r:.3f}")
    print(f"MAE: {mae:.3f} kcal/mol")
    print(f"Baseline: r = 0.515")
    print(f"Improvement: {r - 0.515:+.3f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
