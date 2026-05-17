#!/usr/bin/env python3
"""
Direct evaluation - compare baseline vs interactions
"""

import pickle
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("DIRECT COMPARISON: BASELINE vs INTERACTIONS")
print("="*70)

# Load CASF features
with open('WORK_DIR / casf2016_enhanced_v2.pkl', 'rb') as f:
    casf = pickle.load(f)
with open('WORK_DIR / casf_interaction_features.pkl', 'rb') as f:
    interactions = pickle.load(f)

X = casf['X']  # 586
y = casf['y']
complexes = casf['complexes']

# Interaction features
interaction_feats = interactions['interactions']
X_interact = np.zeros((len(complexes), 30), dtype=np.float32)
for i, cx in enumerate(complexes):
    pdb_id = cx['pdb_id']
    if pdb_id in interaction_feats:
        X_interact[i] = interaction_feats[pdb_id]

print(f"CASF: {X.shape}, Interactions: {X_interact.shape}")

# Load training
with open('CACHE_DIR / geock_training_data_no2016.pkl', 'rb') as f:
    train_data = pickle.load(f)

# Extract features - use ecfp for all
X_train = np.array([d['ecfp'] for d in train_data], dtype=np.float32)
y_train = np.array([d['affinity'] for d in train_data], dtype=np.float32)
print(f"Training: {X_train.shape}")

# Train model
gb = GradientBoostingRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                            min_samples_leaf=3, random_state=42)
gb.fit(X_train, y_train)
rf = RandomForestRegressor(n_estimators=300, max_depth=15, min_samples_leaf=3,
                           random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# Test features
X_test_512 = X[:, :512]  # ecfp only

print("\n1. Baseline (ecfp 512 features):")
pred_gb = gb.predict(X_test_512)
pred_rf = rf.predict(X_test_512)
r, p = pearsonr(y, pred_gb)
print(f"   GB: R={r:.4f}")
r, p = pearsonr(y, pred_rf)
print(f"   RF: R={r:.4f}")

pred_ens = 0.5*pred_gb + 0.5*pred_rf
r, p = pearsonr(y, pred_ens)
print(f"   Ensemble: R={r:.4f}")

# Store baseline
baseline_r = r

# Now test with ecfp + physics + pocket (586)
# Need model trained on 536 or pad
# Just extract ecfp from both sources
X_train_536 = []
for d in train_data:
    ecfp = d['ecfp']
    if 'physics' in d:
        physics = d['physics']
    else:
        physics = np.zeros(24)
    X_train_536.append(np.concatenate([ecfp, physics]))
X_train_536 = np.array(X_train_536, dtype=np.float32)

# Filter training with physics
has_physics = [d for d in train_data if 'physics' in d]
X_train_536 = np.array([np.concatenate([d['ecfp'], d['physics']]) for d in has_physics], dtype=np.float32)
y_train_536 = np.array([d['affinity'] for d in has_physics], dtype=np.float32)

print(f"\n2. With physics: {X_train_536.shape[0]} samples")

gb2 = GradientBoostingRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                              min_samples_leaf=3, random_state=42)
gb2.fit(X_train_536, y_train_536)
rf2 = RandomForestRegressor(n_estimators=300, max_depth=15, min_samples_leaf=3,
                          random_state=42, n_jobs=-1)
rf2.fit(X_train_536, y_train_536)

X_test_536 = X[:, :536]  # ecfp + physics
pred_gb2 = gb2.predict(X_test_536)
pred_rf2 = rf2.predict(X_test_536)
r, p = pearsonr(y, pred_gb2)
print(f"   GB: R={r:.4f}")
r, p = pearsonr(y, pred_rf2)
print(f"   RF: R={r:.4f}")

pred_ens2 = 0.5*pred_gb2 + 0.5*pred_rf2
r, p = pearsonr(y, pred_ens2)
print(f"   Ensemble: R={r:.4f}")

# With interactions
X_test_566 = np.hstack([X_test_536, X_interact])  
X_train_566 = np.hstack([X_train_536, np.zeros((len(X_train_536), 30))])

gb3 = GradientBoostingRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                              min_samples_leaf=3, random_state=42)
gb3.fit(X_train_566, y_train_536)
rf3 = RandomForestRegressor(n_estimators=300, max_depth=15, min_samples_leaf=3,
                          random_state=42, n_jobs=-1)
rf3.fit(X_train_566, y_train_536)

print(f"\n3. With interactions (566):")
pred_gb3 = gb3.predict(X_test_566)
pred_rf3 = rf3.predict(X_test_566)
r, p = pearsonr(y, pred_gb3)
print(f"   GB: R={r:.4f}")
r, p = pearsonr(y, pred_rf3)
print(f"   RF: R={r:.4f}")

pred_ens3 = 0.5*pred_gb3 + 0.5*pred_rf3
r, p = pearsonr(y, pred_ens3)
print(f"   Ensemble: R={r:.4f}")

# Interaction only - on CASF (this will show leakage!)
from sklearn.linear_model import Ridge
ridge = Ridge(alpha=1.0)
ridge.fit(X_interact, y)
pred = ridge.predict(X_interact)
r, p = pearsonr(y, pred)
print(f"\n4. Interaction-only: R={r:.4f} (LIKELY LEAKAGE!)")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Baseline (ecfp only):     R={baseline_r:.4f}")
print(f"With physics:            R={r:.4f}")
print(f"+Interactions:            R={r:.4f}")
print(f"Previous best:            R=0.6816")

results = {
    'baseline': baseline_r,
    'predictions': pred_ens,
    'y': y
}
with open('WORK_DIR / interaction_results.pkl', 'wb') as f:
    pickle.dump(results, f)

print("\nDone.")