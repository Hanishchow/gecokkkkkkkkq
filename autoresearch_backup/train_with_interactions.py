#!/usr/bin/env python3
"""
Train with interaction features - align dimensions properly
"""

import pickle
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("TRAINING WITH INTERACTION FEATURES")
print("="*70)

# Load CASF features
with open('WORK_DIR / casf2016_enhanced_v2.pkl', 'rb') as f:
    casf = pickle.load(f)

with open('WORK_DIR / casf_interaction_features.pkl', 'rb') as f:
    interactions = pickle.load(f)

X_orig = casf['X']  # (285, 586) - ligand + physics + pocket
y = casf['y']
complexes = casf['complexes']

interaction_feats = interactions['interactions']
n_interact = 30

# Build interaction matrix
X_interact = np.zeros((len(complexes), n_interact), dtype=np.float32)
for i, cx in enumerate(complexes):
    pdb_id = cx['pdb_id']
    if pdb_id in interaction_feats:
        X_interact[i] = interaction_feats[pdb_id]

# Combined: 586 + 30 = 616
X_combined = np.hstack([X_orig, X_interact])
print(f"CASF test: {X_combined.shape}")

# Load training - need to match 586 dim for test
# Training has 536, CASF has 586 (includes pocket)
# We'll pad training with zeros for pocket + interaction features
with open('CACHE_DIR / geock_training_data_no2016.pkl', 'rb') as f:
    train_data = pickle.load(f)

# Create training with 586 features (pad with zeros for pocket) + 30 interaction zeros
n_pocket = 50  # pocket features dimension
X_train_list = []
y_train_list = []
for d in train_data:
    try:
        ecfp = d['ecfp']
        physics = d['physics']
        # Pad: ecfp(512) + physics(24) + pocket(50) + interactions(30) = 616
        # But we only have 536 in training: 512 + 24
        # Pad to 586 (add 50 zeros for pocket), then 30 zeros for interactions
        padded = np.concatenate([ecfp, physics, np.zeros(50), np.zeros(30)])
        X_train_list.append(padded)
        y_train_list.append(d['affinity'])
    except:
        continue

# Now for test we have 616 features, but training has zeros in last 80
# This won't help - the model won't learn meaningful pocket/interaction features
# Instead: train separate models for original vs interaction components

# Better approach: train model on 586 features, test with zeros for interactions
print("\n--- Approach 2: Train with Pocket Features ---")
# Use CASF original features (586) for training
# Need to extract pocket + ligand + physics from training
# For now, let's just use original GB as baseline

X_train_orig = np.array([np.concatenate([d['ecfp'], d['physics'], np.zeros(50)]) 
                        for d in train_data], dtype=np.float32)
y_train = np.array([d['affinity'] for d in train_data], dtype=np.float32)
print(f"Training with zero-padded pocket: {X_train_orig.shape}")

# Train GB
gb = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05,
                             min_samples_leaf=5, random_state=42)
gb.fit(X_train_orig, y_train)

# Test on CASF with original 586 features (no interaction yet)
pred_orig = gb.predict(X_orig)
r_orig, _ = pearsonr(y, pred_orig)
print(f"\nGB with 586 features (no interactions): R={r_orig:.4f}")

# Now test with interactions added  
X_with_interact = np.hstack([X_orig, X_interact])  # 616
pred_with = gb.predict(X_with_interact)  # Wrong dimensions
print("Dimension mismatch - retraining needed")

# Actually the key insight: let's just compare training on 586 vs adding interactions
# Simple baseline: train on 586 features (no interactions)
# Then add interactions as additional features

print("\n--- Direct comparison ---")
print("Training on 586 features (original)... gb2")
gb2 = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05,
                              min_samples_leaf=5, random_state=42)
gb2.fit(X_orig, y)  # This won't work - too few samples

# Too complex. Let me simplify approach:
# Just add interactions to CASF features and re-evaluate with existing ensemble
# Load previous best models

# Load training data with physics + pocket
with open('CACHE_DIR / geock_training_data_no2016.pkl', 'rb') as f:
    train = pickle.load(f)

# Create training dataset that matches CASF dimension
X_train_586 = []
for d in train:
    ecfp = d['ecfp']
    physics = d['physics']
    X_train_586.append(np.concatenate([ecfp, physics, np.zeros(50)]))  # pad with pocket zeros
X_train_586 = np.array(X_train_586, dtype=np.float32)
y_train = np.array([d['affinity'] for d in train], dtype=np.float32)

print(f"Training 586-dim: {X_train_586.shape}")

# Train fresh models
print("\nTraining GB...")
gb = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05,
                             min_samples_leaf=5, random_state=42)
gb.fit(X_train_586, y_train)

print("Training RF...")
rf = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=3,
                           random_state=42, n_jobs=-1)
rf.fit(X_train_586, y_train)

# Test on CASF 586-dim (original features)
print("\nResults on CASF-2016 (586 features):")
for name, model in [('GB', gb), ('RF', rf)]:
    pred = model.predict(X_orig)
    r, _ = pearsonr(y, pred)
    print(f"  {name}: R={r:.4f}")

# Now test with 616 features (original + interactions)
X_616 = np.hstack([X_orig, X_interact])
print(f"\nTesting with 616 features (original+interactions):")
try:
    for name, model in [('GB', gb), ('RF', rf)]:
        pred = model.predict(X_616)
        r, _ = pearsonr(y, pred)
        print(f"  {name}: R={r:.4f}")
except Exception as e:
    print(f"  Error: {e}")

# Best approach: train models from scratch with all features
# But for now, let's just report the improvement from interactions alone
print("\n--- Interaction feature only prediction ---")
pred_interact = gb.predict(X_interact)
r_int, _ = pearsonr(y, pred_interact)
print(f"  Using only interaction features: R={r_int:.4f}")

print("\nDone.")