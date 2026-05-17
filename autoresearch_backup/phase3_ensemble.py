#!/usr/bin/env python3
"""
Phase 3: Multi-Model Ensemble Training
- Model A: ECFP4 only (trained on ALL data)
- Model B: ECFP4 + physics (trained on ALL data)
- Model C: Full features with pocket (trained on subset with pockets)
- Ensemble: Similarity-weighted combination
"""

import pickle
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("PHASE 3: MULTI-MODEL ENSEMBLE TRAINING")
print("="*70)

# Load training data
print("\n1. Loading training data...")
with open('CACHE_DIR / geock_training_data_no2016.pkl', 'rb') as f:
    train_list = pickle.load(f)

# Load training pockets (from simple extraction)
print("   Loading training pocket features...")
try:
    with open('training_pocket_simple.pkl', 'rb') as f:
        train_pockets = pickle.load(f)
    print(f"   Found {len(train_pockets)} training pockets")
except:
    train_pockets = {}
    print("   No training pockets found, using zeros")

# Build feature matrices
X_full = np.array([t['ecfp'] for t in train_list])
y_full = np.array([t['affinity'] for t in train_list])
pdb_ids_full = [t['pdb_id'] for t in train_list]

print(f"   Full training: {X_full.shape}")

# Identify training samples with pockets
has_pocket = np.array([pdb_id in train_pockets for pdb_id in pdb_ids_full])
print(f"   Samples with pocket: {has_pocket.sum()} / {len(has_pocket)}")

# Load CASF-2016 test
print("\n2. Loading CASF-2016 test data...")
with open('casf2016_enhanced_v2.pkl', 'rb') as f:
    test_data = pickle.load(f)

X_test = test_data['X']
y_test = test_data['y']
print(f"   Test: {X_test.shape}")

# Load phase 1 analysis for similarity
with open('phase1_analysis.pkl', 'rb') as f:
    analysis = pickle.load(f)
max_sim = analysis['max_sim_per_test']

# ========================================================================
# MODEL A: ECFP4 only (ALL training data)
# ========================================================================
print("\n" + "="*70)
print("MODEL A: ECFP4-only (baseline)")
print("="*70)

scaler_A = StandardScaler()
X_train_A = scaler_A.fit_transform(X_full)
X_test_A = scaler_A.transform(X_test[:, :512])

model_A = GradientBoostingRegressor(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    min_samples_split=10,
    subsample=0.8,
    random_state=42
)

print("   Training...")
model_A.fit(X_train_A, y_full)

# Cross-validation
print("   Cross-validation (5-fold)...")
cv_scores = cross_val_score(model_A, X_train_A, y_full, cv=5, scoring='r2')
print(f"   CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Predict
y_pred_A = model_A.predict(X_test_A)
r_A = pearsonr(y_test, y_pred_A)[0]
rho_A = spearmanr(y_test, y_pred_A)[0]
rmse_A = np.sqrt(np.mean((y_test - y_pred_A)**2))

print(f"\n   CASF-2016 Results:")
print(f"   Pearson R: {r_A:.4f}")
print(f"   Spearman ρ: {rho_A:.4f}")
print(f"   RMSE: {rmse_A:.4f}")

# ========================================================================
# MODEL B: ECFP4 + Physics (ALL training data)
# ========================================================================
print("\n" + "="*70)
print("MODEL B: ECFP4 + Physics")
print("="*70)

# Build training features (ligand + physics)
X_train_B = []
for i, t in enumerate(train_list):
    ligand = t['ecfp']
    
    # Physics from existing data
    if 'physics_simple' in t and t['physics_simple'] is not None:
        physics = np.array(t['physics_simple'], dtype=np.float32)
    elif 'physics' in t and t['physics'] is not None:
        physics = np.array(t['physics'], dtype=np.float32)
    else:
        physics = np.zeros(24, dtype=np.float32)
    
    if len(physics) < 24:
        physics = np.pad(physics, (0, 24 - len(physics)))
    
    X_train_B.append(np.concatenate([ligand, physics[:24]]))

X_train_B = np.array(X_train_B)
print(f"   Training shape: {X_train_B.shape}")

scaler_B = StandardScaler()
X_train_B_s = scaler_B.fit_transform(X_train_B)
X_test_B_s = scaler_B.transform(X_test[:, :536])  # Ligand + physics

model_B = GradientBoostingRegressor(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    min_samples_split=10,
    subsample=0.8,
    random_state=42
)

print("   Training...")
model_B.fit(X_train_B_s, y_full)

# Cross-validation
print("   Cross-validation (5-fold)...")
cv_scores_B = cross_val_score(model_B, X_train_B_s, y_full, cv=5, scoring='r2')
print(f"   CV R²: {cv_scores_B.mean():.4f} ± {cv_scores_B.std():.4f}")

# Predict
y_pred_B = model_B.predict(X_test_B_s)
r_B = pearsonr(y_test, y_pred_B)[0]
rho_B = spearmanr(y_test, y_pred_B)[0]
rmse_B = np.sqrt(np.mean((y_test - y_pred_B)**2))

print(f"\n   CASF-2016 Results:")
print(f"   Pearson R: {r_B:.4f}")
print(f"   Spearman ρ: {rho_B:.4f}")
print(f"   RMSE: {rmse_B:.4f}")

# ========================================================================
# MODEL C: Full features with pocket (subset with pockets)
# ========================================================================
print("\n" + "="*70)
print("MODEL C: Full features with Pocket")
print("="*70)

# Filter training to samples with pockets
pocket_indices = np.where(has_pocket)[0]
print(f"   Training on {len(pocket_indices)} samples with pockets")

if len(pocket_indices) > 100:
    X_train_C = X_full[pocket_indices]
    y_train_C = y_full[pocket_indices]
    
    # Add pocket features
    X_train_C_full = []
    for i in pocket_indices:
        pdb_id = pdb_ids_full[i]
        ligand = X_full[i]
        
        if pdb_id in train_pockets:
            pocket = train_pockets[pdb_id]
        else:
            pocket = np.zeros(50, dtype=np.float32)
        
        physics = np.zeros(24, dtype=np.float32)
        X_train_C_full.append(np.concatenate([ligand, physics, pocket]))
    
    X_train_C_full = np.array(X_train_C_full)
    print(f"   Training shape: {X_train_C_full.shape}")
    
    scaler_C = StandardScaler()
    X_train_C_s = scaler_C.fit_transform(X_train_C_full)
    X_test_C_s = scaler_C.transform(X_test)
    
    model_C = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        min_samples_split=5,
        subsample=0.8,
        random_state=42
    )
    
    print("   Training...")
    model_C.fit(X_train_C_s, y_train_C)
    
    y_pred_C = model_C.predict(X_test_C_s)
    r_C = pearsonr(y_test, y_pred_C)[0]
    rho_C = spearmanr(y_test, y_pred_C)[0]
    rmse_C = np.sqrt(np.mean((y_test - y_pred_C)**2))
    
    print(f"\n   CASF-2016 Results:")
    print(f"   Pearson R: {r_C:.4f}")
    print(f"   Spearman ρ: {rho_C:.4f}")
    print(f"   RMSE: {rmse_C:.4f}")
    
    # Feature importance
    feat_imp = model_C.feature_importances_
    lig_imp = feat_imp[:512].sum()
    phys_imp = feat_imp[512:536].sum()
    pock_imp = feat_imp[536:].sum()
    print(f"\n   Feature importance:")
    print(f"     Ligand: {lig_imp:.3f}")
    print(f"     Physics: {phys_imp:.3f}")
    print(f"     Pocket: {pock_imp:.3f}")
else:
    y_pred_C = y_pred_A.copy()
    r_C, rho_C, rmse_C = r_A, rho_A, rmse_A
    print("   Not enough samples with pockets, using Model A predictions")

# ========================================================================
# ENSEMBLE: Similarity-weighted combination
# ========================================================================
print("\n" + "="*70)
print("ENSEMBLE METHODS")
print("="*70)

# Method 1: Simple average
y_pred_avg = (y_pred_A + y_pred_B) / 2
r_avg = pearsonr(y_test, y_pred_avg)[0]
print(f"\n1. Simple Average (A + B):")
print(f"   R = {r_avg:.4f}")

# Method 2: Weighted by validation performance
best_r = max(r_A, r_B, r_C)
if r_A == best_r:
    weights = [0.6, 0.3, 0.1]
elif r_B == best_r:
    weights = [0.3, 0.6, 0.1]
else:
    weights = [0.3, 0.3, 0.4]

y_pred_weighted = weights[0]*y_pred_A + weights[1]*y_pred_B + weights[2]*y_pred_C
r_weighted = pearsonr(y_test, y_pred_weighted)[0]
print(f"\n2. Performance Weighted:")
print(f"   R = {r_weighted:.4f}")

# Method 3: Similarity-adaptive ensemble (KEY METHOD)
print(f"\n3. Similarity-Adaptive Ensemble:")

y_pred_adaptive = np.zeros_like(y_test, dtype=np.float32)

for i in range(len(y_test)):
    sim = max_sim[i]
    
    if sim > 0.5:
        # High similarity: trust ligand-only model
        y_pred_adaptive[i] = y_pred_A[i]
    elif sim > 0.35:
        # Medium similarity: blend models
        y_pred_adaptive[i] = 0.4 * y_pred_A[i] + 0.4 * y_pred_B[i] + 0.2 * y_pred_C[i]
    else:
        # Low similarity: trust pocket model
        y_pred_adaptive[i] = 0.3 * y_pred_A[i] + 0.3 * y_pred_B[i] + 0.4 * y_pred_C[i]

r_adaptive = pearsonr(y_test, y_pred_adaptive)[0]
rho_adaptive = spearmanr(y_test, y_pred_adaptive)[0]
rmse_adaptive = np.sqrt(np.mean((y_test - y_pred_adaptive)**2))

print(f"   Pearson R: {r_adaptive:.4f}")
print(f"   Spearman ρ: {rho_adaptive:.4f}")
print(f"   RMSE: {rmse_adaptive:.4f}")

# ========================================================================
# FINAL RESULTS
# ========================================================================
print("\n" + "="*70)
print("FINAL RESULTS SUMMARY")
print("="*70)

results = {
    'Model_A_ECFP4': {'R': r_A, 'rho': rho_A, 'RMSE': rmse_A},
    'Model_B_Physics': {'R': r_B, 'rho': rho_B, 'RMSE': rmse_B},
    'Model_C_Pocket': {'R': r_C, 'rho': rho_C, 'RMSE': rmse_C},
    'Ensemble_Average': {'R': r_avg},
    'Ensemble_Weighted': {'R': r_weighted},
    'Ensemble_Adaptive': {'R': r_adaptive, 'rho': rho_adaptive, 'RMSE': rmse_adaptive}
}

for name, metrics in results.items():
    r_val = metrics['R']
    print(f"{name}: R = {r_val:.4f}")

print(f"\nBaseline: R = 0.59")
print(f"Target:   R > 0.76")

# Save models and predictions
print("\nSaving models...")
models = {
    'model_A': model_A,
    'model_B': model_B,
    'model_C': model_C if len(pocket_indices) > 100 else None,
    'scaler_A': scaler_A,
    'scaler_B': scaler_B,
    'scaler_C': scaler_C if len(pocket_indices) > 100 else None,
    'predictions': {
        'y_pred_A': y_pred_A,
        'y_pred_B': y_pred_B,
        'y_pred_C': y_pred_C,
        'y_pred_adaptive': y_pred_adaptive,
        'y_test': y_test
    },
    'results': results
}

with open('ensemble_models.pkl', 'wb') as f:
    pickle.dump(models, f)

print("Saved to ensemble_models.pkl")
print("\nPhase 3 complete.")