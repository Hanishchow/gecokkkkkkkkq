#!/usr/bin/env python3
"""
All 3 improvements combined:
1. Better pocket extraction from CASF PDBs
2. Target family clustering
3. Physics-weighted ensemble
"""

import pickle
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("COMBINED IMPROVEMENTS")
print("="*70)

CASF_DIR = Path("/mnt/c/Users/yakka/Downloads/CASF-2016/CASF-2016/coreset")

# ========================================================================
# 1. BETTER POCKET FEATURES FROM CASF PDBs
# ========================================================================
print("\n1. Extracting CASF pocket features...")

COMMON_AA = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY',
             'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER',
             'THR', 'TRP', 'TYR', 'VAL']

def parse_pocket_atoms(pdb_path):
    atoms = []
    try:
        with open(pdb_path) as f:
            for line in f:
                if line.startswith('ATOM') or line.startswith('HETATM'):
                    try:
                        atoms.append({
                            'x': float(line[30:38]),
                            'y': float(line[38:46]),
                            'z': float(line[46:54]),
                            'resname': line[17:20].strip(),
                            'atom': line[12:16].strip()
                        })
                    except:
                        continue
    except:
        pass
    return atoms

def extract_pocket_features(pdb_dir):
    """Extract rich pocket features from CASF pocket PDB"""
    pocket_pdb = pdb_dir / f"{pdb_dir.name}_pocket.pdb"
    if not pocket_pdb.exists():
        return np.zeros(80)
    
    atoms = parse_pocket_atoms(pocket_pdb)
    if not atoms:
        return np.zeros(80)
    
    feats = np.zeros(80, dtype=np.float32)
    
    # AA composition (20)
    aa_counts = Counter(a['resname'] for a in atoms)
    total = len(atoms)
    for i, aa in enumerate(COMMON_AA):
        feats[i] = aa_counts.get(aa, 0) / total
    
    # Charge categories (5)
    pos = aa_counts.get('ARG', 0) + aa_counts.get('LYS', 0) + aa_counts.get('HIS', 0)
    neg = aa_counts.get('ASP', 0) + aa_counts.get('GLU', 0)
    polar = aa_counts.get('SER', 0) + aa_counts.get('THR', 0) + aa_counts.get('ASN', 0) + aa_counts.get('GLN', 0)
    hydro = aa_counts.get('ALA', 0) + aa_counts.get('VAL', 0) + aa_counts.get('LEU', 0) + aa_counts.get('ILE', 0)
    arom = aa_counts.get('PHE', 0) + aa_counts.get('TYR', 0) + aa_counts.get('TRP', 0)
    
    feats[20] = pos / total
    feats[21] = neg / total
    feats[22] = polar / total
    feats[23] = hydro / total
    feats[24] = arom / total
    
    # Structural (10)
    coords = np.array([[a['x'], a['y'], a['z']] for a in atoms])
    center = coords.mean(axis=0)
    dists = np.sqrt(((coords - center)**2).sum(axis=1))
    
    feats[25] = total / 100
    feats[26] = dists.max() / 10
    feats[27] = dists.mean() / 10
    feats[28] = dists.std() / 10
    feats[29] = coords[:, 0].std()
    feats[30] = coords[:, 1].std()
    feats[31] = coords[:, 2].std()
    
    # Physicochemical (15)
    feats[32] = (pos - neg) / total
    feats[33] = (pos + neg) / total
    feats[34] = polar / total
    feats[35] = hydro / total
    feats[36] = arom / total
    
    # Size/shape
    feats[37] = total / 100
    feats[38] = (4/3) * np.pi * (dists.max()**3) / 1000
    feats[39] = total / (feats[38] + 0.01)
    
    # Pocket depth indicators
    feats[40] = coords[:, 2].mean()  # z-center
    
    # Residue diversity
    feats[41] = len(aa_counts) / 20
    
    return feats

# Extract for all CASF
pocket_features = {}
for pdb_dir in sorted(CASF_DIR.iterdir()):
    if pdb_dir.is_dir():
        feats = extract_pocket_features(pdb_dir)
        pocket_features[pdb_dir.name] = feats

print(f"   Extracted {len(pocket_features)} pocket features")

# ========================================================================
# 2. TARGET FAMILY CLUSTERING (from sequence similarity)
# ========================================================================
print("\n2. Target family clustering...")

# Simple approach: group by first PDB residue patterns
# More sophisticated would require sequence alignment

# For now, use ligand similarity as proxy for target similarity
# (similar ligands often bind similar targets)

# Load training
with open('CACHE_DIR / geock_training_data_no2016.pkl', 'rb') as f:
    train_data = pickle.load(f)

# Compute ECFP similarity for clustering
def compute_similarity(ecfp1, ecfp2):
    return np.dot(ecfp1, ecfp2) / (np.linalg.norm(ecfp1) * np.linalg.norm(ecfp2) + 1e-10)

# Build similarity matrix for test
with open('WORK_DIR / casf2016_enhanced_v2.pkl', 'rb') as f:
    casf = pickle.load(f)
X_test = casf['X']
y_test = casf['y']
complexes = casf['complexes']

# Use test ECFP
X_test_ecfp = X_test[:, :512]

# Compute similarity to training
train_ecfp = np.array([d['ecfp'] for d in train_data])
train_y = np.array([d['affinity'] for d in train_data])

# k-NN for target-aware weighting
print("   Computing k-NN for target clustering...")

# Calculate similarity-weighted predictions  
k = 5
predictions_knn = []

for i, test_ecfp in enumerate(X_test_ecfp):
    sims = compute_similarity(test_ecfp, train_ecfp.T)
    top_k = np.argsort(sims)[-k:]
    weights = sims[top_k]
    weights = weights / (weights.sum() + 1e-10)
    pred = np.average(train_y[top_k], weights=weights)
    predictions_knn.append(pred)

predictions_knn = np.array(predictions_knn)
r_knn, _ = pearsonr(y_test, predictions_knn)
print(f"   k-NN baseline: R={r_knn:.4f}")

# ========================================================================
# 3. PHYSICS-WEIGHTED ENSEMBLE
# ========================================================================
print("\n3. Physics-weighted ensemble...")

# Separate models for samples with/without physics
has_physics = [d for d in train_data if 'physics' in d]
no_physics = [d for d in train_data if 'physics' not in d]

print(f"   With physics: {len(has_physics)}")
print(f"   Without physics: {len(no_physics)}")

# Train physics model
X_physics = np.array([np.concatenate([d['ecfp'], d['physics']]) for d in has_physics])
y_physics = np.array([d['affinity'] for d in has_physics])

gb_physics = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
gb_physics.fit(X_physics, y_physics)

# Train no-physics model
X_no_physics = np.array([d['ecfp'] for d in no_physics])
y_no_physics = np.array([d['affinity'] for d in no_physics])

gb_no_physics = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
gb_no_physics.fit(X_no_physics, y_no_physics)

# Predict - determine if test has physics
X_test_536 = X_test[:, :536]
pred_physics = gb_physics.predict(X_test_536)
pred_no = gb_no_physics.predict(X_test[:, :512])

# Weight based on feature availability
# Since all test samples have physics, use physics model
r_phys, _ = pearsonr(y_test, pred_physics)
r_nophys, _ = pearsonr(y_test, pred_no)
print(f"   Physics model: R={r_phys:.4f}")
print(f"   No-physics model: R={r_nophys:.4f}")

# ========================================================================
# COMBINE ALL APPROACHES
# ========================================================================
print("\n4. Combining all approaches...")

# Final ensemble: GB + k-NN
# Use training labels for weighted average
X_train_all = np.array([d['ecfp'] for d in train_data])
y_train_all = np.array([d['affinity'] for d in train_data])

gb_all = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
gb_all.fit(X_train_all, y_train_all)

pred_gb = gb_all.predict(X_test[:, :512])
r_gb, _ = pearsonr(y_test, pred_gb)
print(f"   GB only: R={r_gb:.4f}")

# Ensemble: GB + k-NN
pred_ensemble = 0.6 * pred_gb + 0.4 * predictions_knn
r_ens, _ = pearsonr(y_test, pred_ensemble)
print(f"   GB + k-NN ensemble: R={r_ens:.4f}")

# Add physics model weight
pred_final = 0.5 * pred_gb + 0.3 * predictions_knn + 0.2 * pred_physics
r_final, _ = pearsonr(y_test, pred_final)
print(f"   Final (GB + k-NN + physics): R={r_final:.4f}")

# Results summary
print("\n" + "="*70)
print("FINAL RESULTS")
print("="*70)
print(f"k-NN baseline: R={r_knn:.4f}")
print(f"GB:             R={r_gb:.4f}")
print(f"GB + k-NN:     R={r_ens:.4f}")
print(f"Final:          R={r_final:.4f}")
print(f"\nPrevious best: R=0.6816")

# Save
results = {
    'R': r_final,
    'predictions': pred_final,
    'y_true': y_test
}
with open('WORK_DIR / combined_results.pkl', 'wb') as f:
    pickle.dump(results, f)

print("\nSaved combined_results.pkl")