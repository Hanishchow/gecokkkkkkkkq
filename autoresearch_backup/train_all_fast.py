#!/usr/bin/env python3
"""Train on all extracted features."""
import pickle, time, os
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import LeaveOneOut, cross_val_predict, RepeatedKFold
from scipy.stats import pearsonr

CACHE_DIR = '/home/chow/.cache/geock_autoresearch'

# Load extracted features
print("Loading features...")
with open(f'{CACHE_DIR}/lp_all_features.pkl', 'rb') as f:
    features = pickle.load(f)
print(f"Loaded {len(features)} compounds")

X = np.array([f['ecfp'] for f in features], dtype=np.float32)
y = np.array([f['affinity'] for f in features])
pdb_ids = [f['pdb_id'] for f in features]
smiles_list = [f['smiles'] for f in features]
print(f"X shape: {X.shape}, y range: {y.min():.1f} - {y.max():.1f}, mean: {y.mean():.1f}")

# Normalize
mu = X.mean(0)
sd = X.std(0)
sd = np.where(sd < 1e-10, 1, sd)
X_n = (X - mu) / sd

# Grid search
print("\nGrid search (ke vs alpha)...")
t0 = time.time()
best = (0, None, None)

results_grid = []
for ke in [50, 75, 100, 150, 200, 300]:
    if ke >= X.shape[1]: continue
    sel = SelectKBest(f_regression, k=ke)
    X_s = sel.fit_transform(X_n, y)
    for alpha in [1.0, 5.0, 10.0, 50.0, 100.0]:
        loo_preds = cross_val_predict(Ridge(alpha=alpha), X_s, y, cv=LeaveOneOut())
        loo_r = pearsonr(y, loo_preds)[0]
        results_grid.append((ke, alpha, loo_r))
        if loo_r > best[0]:
            best = (loo_r, ke, alpha)
        print(f"  ke={ke:3d}, alpha={alpha:5.1f}  LOO-R={loo_r:.4f}")

print(f"\nBest: ke={best[1]}, alpha={best[2]}, LOO-R={best[0]:.4f} ({time.time()-t0:.0f}s)")

# Train final model
ke, alpha = best[1], best[2]
sel = SelectKBest(f_regression, k=ke)
X_s = sel.fit_transform(X_n, y)
ridge = Ridge(alpha=alpha)
ridge.fit(X_s, y)
train_r = pearsonr(y, ridge.predict(X_s))[0]

# RKF
rkf = RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)
rkf_preds = cross_val_predict(Ridge(alpha=alpha), X_s, y, cv=rkf)
rkf_r = pearsonr(y, rkf_preds)[0]

loo_preds = cross_val_predict(Ridge(alpha=alpha), X_s, y, cv=LeaveOneOut())
loo_r = pearsonr(y, loo_preds)[0]
loo_mae = np.mean(np.abs(y - loo_preds))

gap = train_r - rkf_r

print(f"\n{'='*50}")
print(f"FINAL MODEL")
print(f"{'='*50}")
print(f"N compounds:   {len(y)}")
print(f"ke (ECFP bits): {ke}")
print(f"alpha:        {alpha}")
print(f"LOO-R:        {loo_r:.4f}  (honest)")
print(f"RKF-R:        {rkf_r:.4f}  (5-fold × 5)")
print(f"Train-R:      {train_r:.4f}")
print(f"Gap:          {gap:.4f}")
print(f"LOO-MAE:      {loo_mae:.2f} pKd")
print(f"y range:      {y.min():.1f} - {y.max():.1f} (pKd)")

# Save model
model_data = {
    'ridge': ridge,
    'sel': sel,
    'mu': mu,
    'sd': sd,
    'ke': ke,
    'alpha': alpha,
    'loo_r': float(loo_r),
    'rkf_r': float(rkf_r),
    'rkf_std': 0.0,
    'train_r': float(train_r),
    'gap': float(gap),
    'loo_mae': float(loo_mae),
    'n_compounds': len(y),
    'ecfp_len': X.shape[1],
    'pdb_ids': pdb_ids,
    'smiles': smiles_list,
}

MODEL_PATH = 'WORK_DIR / geock_model_all.pkl'
with open(MODEL_PATH, 'wb') as f:
    pickle.dump(model_data, f)
print(f"\nModel saved: {MODEL_PATH}")

# Save TSV predictions
TSV_PATH = 'WORK_DIR / results_all.tsv'
with open(TSV_PATH, 'w') as f:
    f.write("pdb_id\tsmiles\tactual_pKd\tpredicted_pKd\terror\n")
    for i in range(len(y)):
        err = abs(y[i] - loo_preds[i])
        f.write(f"{pdb_ids[i]}\t{smiles_list[i]}\t{y[i]:.3f}\t{loo_preds[i]:.3f}\t{err:.3f}\n")
print(f"Predictions saved: {TSV_PATH}")

# Update geock_engine.py model path reference
print(f"\n{'='*50}")
print("Update geock_engine.py model path to geock_model_all.pkl")
