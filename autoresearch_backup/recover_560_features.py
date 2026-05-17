#!/usr/bin/env python3
"""
Recover 560-dim features for CASF-2016 test
- ECFP4: 512
- Molecular: 8 (rdkit descriptors)
- Physics: 20
- Interaction: 20
Total: 560
"""
import pickle
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("RECOVERING 560-DIM FEATURES")
print("="*60)

# Load CASF test
with open('casf2016_enhanced_v2.pkl', 'rb') as f:
    test = pickle.load(f)

complexes = test['complexes']
y_test = test['y']
X_base = test['X']  # 586 = 512 + 24 + 50

print(f"CASF test: {len(complexes)} samples")

# Load interaction features (30-dim)
with open('casf_interaction_features.pkl', 'rb') as f:
    interact_data = pickle.load(f)
interactions = interact_data['interactions']

# Build 560-dim features
# ECFP (512) + physics (24) + pocket (50) + interaction (30-???) 

# We have: 512 ecfp + 24 physics + 50 pocket = 586
# Need: 512 + 8 + 20 + 20 = 560

# The missing 8 molecular + 20 physics + 20 interaction = 48
# We have: 24 physics (need 20) + 50 pocket + 30 interaction (need 20)

# Build features by taking first N
X_560 = np.zeros((len(complexes), 560), dtype=np.float32)

for i, cx in enumerate(complexes):
    pdb_id = cx['pdb_id']
    
    # ECFP: 512
    X_560[i, :512] = X_base[i, :512]
    
    # Physics: use first 20 of 24
    X_560[i, 512:532] = X_base[i, 512:532]
    
    # Pocket: use 50 - but we need only 8+20=28 more
    # Actually let's reconstruct: 512 + 8 + 20 + 20 = 560
    # ECFP (512-512) + mol (8) + phys (20) + int (20)
    
# Try different mapping
X_560 = np.zeros((len(complexes), 560), dtype=np.float32)

# ECFP4: 512
X_560[:, :512] = X_base[:, :512]

# Physics features: use 20 of 24
X_560[:, 512:532] = X_base[:, 512:532]

# Interaction features: 20 (we have 30, use first 20)
for i, cx in enumerate(complexes):
    pdb_id = cx['pdb_id']
    if pdb_id in interactions:
        X_560[i, 532:552] = interactions[pdb_id][:20]

# Molecular: 8 features - set to mean values (we don't have exact mol descriptors for CASF)
# This is the gap - we'd need rdkit to compute for each ligand
# For now, use zeros
X_560[:, 552:560] = 0  # Placeholder

print(f"Feature matrix: {X_560.shape}")

# Now test with training data that has matching features
# Load training with physics
with open('CACHE_DIR / geock_training_data_no2016.pkl', 'rb') as f:
    train = pickle.load(f)

# Extract samples with physics (829)
has_phys = [d for d in train if 'physics' in d]
print(f"Training with physics: {len(has_phys)}")

# Build features
X_train_list = []
y_train_list = []

for d in has_phys:
    ecfp = d['ecfp']
    physics = d['physics'][:20]  # First 20
    # Interaction - 0 placeholder
    # Molecular - 0 placeholder
    combined = np.concatenate([ecfp, physics, np.zeros(28)])
    X_train_list.append(combined)
    y_train_list.append(d['affinity'])

X_train = np.array(X_train_list, dtype=np.float32)
y_train = np.array(y_train_list, dtype=np.float32)

print(f"Training: {X_train.shape}")

# Train XGBoost like before
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor

model = XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    random_state=42
)

print("Training...")
model.fit(X_train, y_train)

# Predict
pred = model.predict(X_560)
r, p = pearsonr(y_test, pred)
rho, _ = spearmanr(y_test, pred)

print(f"\nResult with recovered 560 features:")
print(f"  R = {r:.4f}")
print(f"  rho = {rho:.4f}")

# Compare to previous
print(f"\nPrevious: R = 0.6811")
print(f"Diff: {r - 0.6811:.4f}")

# Save
result = {'R': r, 'rho': rho, 'y_pred': pred, 'y_true': y_test}
with open('recovered_result.pkl', 'wb') as f:
    pickle.dump(result, f)
print("\nSaved recovered_result.pkl")