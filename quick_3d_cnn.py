#!/usr/bin/env python3
"""
Quick 3D CNN prototype
- Generate 3D grids from CASF pocket PDBs
- Simple conv3d on pocket structures
"""

import pickle
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("3D CNN PROTOTYPE")
print("="*70)

CASF_DIR = Path("/mnt/c/Users/yakka/Downloads/CASF-2016/CASF-2016/coreset")

# ========================================================================
# 1. Generate 3D grids from pocket PDBs
# ========================================================================
print("\n1. Generating 3D grids...")

GRID_SIZE = 16  # 16x16x16 = 4096 voxels
RESOLUTION = 1.0  # 1 Angstrom per voxel

# Atom types for channels
ATOM_TYPES = ['C', 'N', 'O', 'S', 'H']  # Simplified elements
N_CHANNELS = len(ATOM_TYPES)

def pdb_to_grid(pdb_path, size=GRID_SIZE, resolution=RESOLUTION):
    """Convert PDB to 3D grid"""
    grid = np.zeros((size, size, size, N_CHANNELS), dtype=np.float32)
    
    try:
        with open(pdb_path) as f:
            for line in f:
                if line.startswith('ATOM') or line.startswith('HETATM'):
                    # Parse atom
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    elem = line[76:78].strip() if len(line) > 76 else line[12:14].strip()
                    elem = elem.strip()[0] if elem else 'X'  # First letter
                    
                    # Map to channel
                    if elem in ATOM_TYPES:
                        ch = ATOM_TYPES.index(elem)
                    else:
                        ch = 0  # Default to carbon
                    
                    # Convert to grid coordinates
                    center = size // 2
                    gx = int(x / resolution) + center
                    gy = int(y / resolution) + center
                    gz = int(z / resolution) + center
                    
                    # Place in grid (with simple distance falloff)
                    for dx in range(-1, 2):
                        for dy in range(-1, 2):
                            for dz in range(-1, 2):
                                nx, ny, nz = gx + dx, gy + dy, gz + dz
                                if 0 <= nx < size and 0 <= ny < size and 0 <= nz < size:
                                    dist = np.sqrt(dx*dx + dy*dy + dz*dz)
                                    grid[nx, ny, nz, ch] = max(grid[nx, ny, nz, ch], 1.0 / (dist + 1))
    except:
        pass
    
    return grid

# Process CASF pockets
X_3d = []
y_values = []
pdb_ids = []

with open('WORK_DIR / casf2016_enhanced_v2.pkl', 'rb') as f:
    test_data = pickle.load(f)

# Map pdb_id to log_ka
log_ka_map = {cx['pdb_id']: cx['log_ka'] for cx in test_data['complexes']}

count = 0
for pdb_dir in sorted(CASF_DIR.iterdir()):
    if not pdb_dir.is_dir():
        continue
    pdb_id = pdb_dir.name
    
    # Get pocket PDB
    pocket_pdb = pdb_dir / f"{pdb_id}_pocket.pdb"
    if pocket_pdb.exists() and pdb_id in log_ka_map:
        grid = pdb_to_grid(pocket_pdb)
        X_3d.append(grid)
        y_values.append(log_ka_map[pdb_id])
        pdb_ids.append(pdb_id)
        count += 1

X_3d = np.array(X_3d)
y_values = np.array(y_values)

print(f"   Generated {count} 3D grids: {X_3d.shape}")

# ========================================================================
# 2. Simple 3D CNN (using sklearn for speed)
# ========================================================================
print("\n2. Training placeholder...")

# For quick prototype, use flattened features + simple model
# Full 3D conv would need tensorflow/keras

# Flatten grid
X_flat = X_3d.reshape(len(X_3d), -1)
print(f"   Flattened: {X_flat.shape}")

# Replace with simple sklearn baseline
from sklearn.ensemble import GradientBoostingRegressor
from scipy.stats import pearsonr

# Split for validation
np.random.seed(42)
idx = np.random.permutation(len(X_flat))
n_train = int(0.8 * len(idx))
train_idx = idx[:n_train]
test_idx = idx[n_train:]

X_train, X_test = X_flat[train_idx], X_flat[test_idx]
y_train, y_test = y_values[train_idx], y_values[test_idx]

print(f"\n   Train: {len(X_train)}, Test: {len(X_test)}")

# Simple GB on flattened grid
gb = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
gb.fit(X_train, y_train)

pred = gb.predict(X_test)
r, p = pearsonr(y_test, pred)
print(f"\n   3D Grid (GB): R={r:.4f} on held-out")

# Compare with baseline (ECFP from test data)
with open('WORK_DIR / casf2016_enhanced_v2.pkl', 'rb') as f:
    test = pickle.load(f)

X_ecfp = test['X'][:, :512]
y_ecfp = test['y']

# Use same split
X_ecfp_train, X_ecfp_test = X_ecfp[train_idx], X_ecfp[test_idx]

gb2 = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
gb2.fit(X_ecfp_train, y_train)
pred2 = gb2.predict(X_ecfp_test)
r2, _ = pearsonr(y_test, pred2)
print(f"   ECFP4 (GB): R={r2:.4f}")

print("\n" + "="*70)
print("COMPARISON")
print("="*70)
print(f"3D Grid:     R={r:.4f}")
print(f"ECFP4:       R={r2:.4f}")
print("Note: 3D grid is very primitive (16^3 = 4096 features)")

# Save grids for later use
np.savez_compressed('WORK_DIR / 3d_grids.npz', 
                 X=X_3d, y=y_values, pdb_ids=pdb_ids)

print(f"\nSaved 3D grids to 3d_grids.npz")
print("For full 3D CNN, would need tensorflow with conv3d layers")