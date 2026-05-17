import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
RDLogger.DisableLog('rdChem.*')

print("Loading merged data...")
with open('.cache/geock_autoresearch/geock_merged_training.pkl', 'rb') as f:
    data = pickle.load(f)

smiles_list = [d[2] for d in data]
affinity = np.array([d[3] for d in data])
print(f"Total samples: {len(smiles_list)}")

# Quick simple descriptors (computes fast)
print("Computing descriptors...")
desc_data = []
for i, smi in enumerate(smiles_list):
    if i % 2000 == 0:
        print(f"  {i}/{len(smiles_list)}")
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        desc_data.append(np.zeros(6))
    else:
        desc_data.append([
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.NumRotatableBonds(mol)
        ])

X = np.array(desc_data)
y = affinity
print(f"Features: {X.shape}")

# === CORRELATIONS ===
feature_names = ['MW', 'LogP', 'TPSA', 'HDonors', 'HAcceptors', 'RotBonds']
print("\n=== CORRELATIONS WITH pKD ===")
for i, name in enumerate(feature_names):
    c = np.corrcoef(X[:, i], y)[0, 1]
    print(f"  {name}: {c:.4f}")

# === LEARNING CURVES ===
print("\n=== LEARNING CURVES ===")
kf = KFold(n_splits=3, shuffle=True, random_state=42)
results = []

for n in [500, 1000, 2000, 4000, 8000, 12000, 16000]:
    np.random.seed(42)
    idx = np.random.permutation(len(X))[:n]
    Xsub, ysub = X[idx], y[idx]
    
    model = GradientBoostingRegressor(n_estimators=50, max_depth=4, learning_rate=0.1, random_state=42)
    scores = cross_val_score(model, Xsub, ysub, cv=kf, scoring='r2')
    results.append({'n': n, 'r2': scores.mean(), 'std': scores.std()})
    print(f"N={n:5d}: R2={scores.mean():.4f} +/- {scores.std():.4f}")

# Train on ALL data and see performance
print("\n=== FULL TRAINING ===")
model = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
print(f"N={len(X)}: R2={scores.mean():.4f} +/- {scores.std():.4f}")

# Also test RandomForest
print("\n--- RF ---")
rf_scores = cross_val_score(RandomForestRegressor(n_estimators=50, max_depth=6, random_state=42, n_jobs=-1), 
                          X, y, cv=kf, scoring='r2')
print(f"RF: R2={rf_scores.mean():.4f}")

# === PLOT ===
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Learning curve
ns = [r['n'] for r in results]
r2s = [r['r2'] for r in results]
stds = [r['std'] for r in results]
ax1 = axes[0]
ax1.fill_between(ns, [r-s for r,s in zip(r2s, stds)], [r+s for r,s in zip(r2s, stds)], alpha=0.2, color='blue')
ax1.plot(ns, r2s, 'b-o', linewidth=2, markersize=8, label='GradientBoosting')
ax1.set_xlabel('Training Samples')
ax1.set_ylabel('R2 Score (CV)')
ax1.set_title('Learning Curve: Molecular Descriptors Only')
ax1.grid(True, alpha=0.3)
ax1.set_xscale('log')
ax1.legend()

# Feature correlations bar
ax2 = axes[1]
corrs = [np.corrcoef(X[:, i], y)[0, 1] for i in range(6)]
colors = ['green' if c > 0 else 'red' for c in corrs]
ax2.barh(feature_names, corrs, color=colors)
ax2.set_xlabel('Correlation with pKD')
ax2.set_title('Molecular Property Correlations')
ax2.axvline(0, color='black', linewidth=0.5)

plt.tight_layout()
plt.savefig('.cache/geock_autoresearch/learning_curves_molprops.png', dpi=150)
print(f"\nSaved: .cache/geock_autoresearch/learning_curves_molprops.png")
print("Done.")