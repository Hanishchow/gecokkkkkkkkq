#!/usr/bin/env python3
"""
Simplified - Combine improvements
"""

import pickle
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("COMBINED IMPROVEMENTS (FAST)")
print("="*70)

# Load test data
with open('WORK_DIR / casf2016_enhanced_v2.pkl', 'rb') as f:
    test = pickle.load(f)
X_test = test['X']
y_test = test['y']

# Load training
with open('CACHE_DIR / geock_training_data_no2016.pkl', 'rb') as f:
    train = pickle.load(f)

print(f"Train: {len(train)}, Test: {len(y_test)}")

# Separate physics/no-physics
has_phys = [d for d in train if 'physics' in d]
no_phys = [d for d in train if 'physics' not in d]
print(f"With physics: {len(has_phys)}, without: {len(no_phys)}")

# Quick models
print("\n1. Training models...")

# Physics model (829 samples)
X_ph = np.array([np.concatenate([d['ecfp'], d['physics']]) for d in has_phys])
y_ph = np.array([d['affinity'] for d in has_phys])
gb_ph = GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
gb_ph.fit(X_ph, y_ph)

# No-physics model (14k samples)  
X_noph = np.array([d['ecfp'] for d in no_phys])
y_noph = np.array([d['affinity'] for d in no_phys])
gb_noph = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
gb_noph.fit(X_noph, y_noph)

# k-NN
print("2. Training k-NN...")
X_train = np.array([d['ecfp'] for d in train])
y_train = np.array([d['affinity'] for d in train])

# k-NN with similarity weighting
knn = KNeighborsRegressor(n_neighbors=5, weights='distance')
knn.fit(X_train, y_train)

# Test predictions
X_test_512 = X_test[:, :512]
X_test_536 = X_test[:, :536]

pred_ph = gb_ph.predict(X_test_536)
pred_noph = gb_noph.predict(X_test_512)
pred_knn = knn.predict(X_test_512)

r_ph, _ = pearsonr(y_test, pred_ph)
r_noph, _ = pearsonr(y_test, pred_noph)
r_knn, _ = pearsonr(y_test, pred_knn)

print(f"\nResults:")
print(f"  Physics model: R={r_ph:.4f}")
print(f"  No-physics model: R={r_noph:.4f}")
print(f"  k-NN: R={r_knn:.4f}")

# Ensemble
pred_ens = 0.4*pred_noph + 0.3*pred_knn + 0.3*pred_ph
r_ens, _ = pearsonr(y_test, pred_ens)
print(f"  Ensemble: R={r_ens:.4f}")

# Using only physics samples in test (all do)
# Weight toward physics model
pred_final = 0.3*pred_ph + 0.7*(0.5*pred_noph + 0.5*pred_knn)
r_final, _ = pearsonr(y_test, pred_final)
print(f"  Weighted: R={r_final:.4f}")

print(f"\nPrevious best: R=0.6816")

results = {'R': r_final, 'pred': pred_final, 'y': y_test}
with open('WORK_DIR / combined_results.pkl', 'wb') as f:
    pickle.dump(results, f)

print("\nDone.")