#!/usr/bin/env python3
"""
Improved 3D CNN with more features
- Better atom channels
- Full CV evaluation
"""

import pickle
import numpy as np
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_predict, KFold
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("IMPROVED 3D GRID")
print("="*70)

CASF_DIR = Path("/mnt/c/Users/yakka/Downloads/CASF-2016/CASF-2016/coreset")

# Better atom encoding: 8 channels
# C (hydrophobic), N (acceptor), O (donor), S, P, Halogens, Metal, Other
ATOM_CHANNELS = {
    'C': 0,  # hydrophobic
    'N': 1,  # donor/acceptor  
    'O': 2,
    'S': 3,
    'P': 4,
    'F': 5, 'CL': 5, 'BR': 5, 'I': 5,  # halogens
    'FE': 6, 'ZN': 6, 'MG': 6, 'CA': 6, 'NA': 6, 'K': 6,  # metals
    'X': 7  # other
}
N_CHANNELS = 8

GRID_SIZE = 20
RESOLUTION = 0.75

def pdb_to_grid(pdb_dir):
    """Convert CASF pocket to 3D grid"""
    pdb_id = pdb_dir.name
    pocket_pdb = pdb_dir / f"{pdb_id}_pocket.pdb"
    
    grid = np.zeros((GRID_SIZE, GRID_SIZE, GRID_SIZE, N_CHANNELS), dtype=np.float32)
    
    if not pocket_pdb.exists():
        return grid
    
    center = GRID_SIZE // 2
    
    try:
        with open(pocket_pdb) as f:
            for line in f:
                if line.startswith(('ATOM', 'HETATM')):
                    try:
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        resname = line[17:20].strip()
                        
                        # Determine channel by element
                        elem = line[76:78].strip() if len(line) > 76 else resname[0]
                        elem = elem.strip().upper()[:2]
                        
                        if elem in ATOM_CHANNELS:
                            ch = ATOM_CHANNELS[elem]
                        elif elem[0] in ATOM_CHANNELS:
                            ch = ATOM_CHANNELS[elem[0]]
                        else:
                            ch = 7
                        
                        # Grid position
                        gx = int((x + center) / RESOLUTION)
                        gy = int((y + center) / RESOLUTION)
                        gz = int((z + center) / RESOLUTION)
                        
                        if 0 <= gx < GRID_SIZE and 0 <= gy < GRID_SIZE and 0 <= gz < GRID_SIZE:
                            # Add density
                            grid[gx, gy, gz, ch] += 1.0
                    except:
                        continue
    except:
        pass
    
    # Apply gaussian smoothing (simple)
    from scipy.ndimage import gaussian_filter
    for ch in range(N_CHANNELS):
        grid[:, :, :, ch] = gaussian_filter(grid[:, :, :, ch], sigma=0.5)
    
    return grid

# Extract all grids
print("\n1. Extracting grids...")
X_grids = []
y_values = []
pdb_ids = []

with open('WORK_DIR / casf2016_enhanced_v2.pkl', 'rb') as f:
    test = pickle.load(f)
log_ka_map = {cx['pdb_id']: cx['log_ka'] for cx in test['complexes']}

for pdb_dir in sorted(CASF_DIR.iterdir()):
    if not pdb_dir.is_dir():
        continue
    pdb_id = pdb_dir.name
    
    if pdb_id in log_ka_map:
        grid = pdb_to_grid(pdb_dir)
        X_grids.append(grid)
        y_values.append(log_ka_map[pdb_id])
        pdb_ids.append(pdb_id)

X_grids = np.array(X_grids)
y_values = np.array(y_values)
print(f"   {len(X_grids)} structures: {X_grids.shape}")

# Flatten for sklearn (use only non-zero channels)
X_flat = X_grids.reshape(len(X_grids), -1)
print(f"   Flattened: {X_flat.shape}")

# ========================================================================
# Cross-validation
# ========================================================================
print("\n2. Cross-validation...")

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 3D grid predictions
preds_3d = cross_val_predict(
    GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42),
    X_flat, y_values, cv=kf
)
r_3d, _ = pearsonr(y_values, preds_3d)
rho_3d, _ = spearmanr(y_values, preds_3d)
print(f"   3D Grid CV: R={r_3d:.4f}, rho={rho_3d:.4f}")

# Compare with ECFP
X_ecfp = test['X'][:, :512]
preds_ecfp = cross_val_predict(
    GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42),
    X_ecfp, y_values, cv=kf
)
r_ecfp, _ = pearsonr(y_values, preds_ecfp)
print(f"   ECFP4 CV:   R={r_ecfp:.4f}")

# Ensemble both
preds_ens = 0.5 * preds_3d + 0.5 * preds_ecfp
r_ens, _ = pearsonr(y_values, preds_ens)
print(f"   Ensemble:   R={r_ens:.4f}")

print("\n" + "="*70)
print("COMPARISON (5-fold CV)")
print("="*70)
print(f"3D Grid:   R={r_3d:.4f}")
print(f"ECFP4:    R={r_ecfp:.4f}")  
print(f"Ensemble:  R={r_ens:.4f}")
print(f"\nPrevious best: R=0.6816")

# Save results
results = {
    'R_3d': r_3d,
    'R_ecfp': r_ecfp,
    'R_ensemble': r_ens,
    'pred_3d': preds_3d,
    'y': y_values
}

with open('WORK_DIR / 3d_results.pkl', 'wb') as f:
    pickle.dump(results, f)

print("\nSaved 3d_results.pkl")