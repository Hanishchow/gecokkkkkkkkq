import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import GradientBoostingRegressor
import warnings
warnings.filterwarnings('ignore')
import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# Load merged data with precomputed features
print("Loading data...")
with open('.cache/geock_autoresearch/geock_merged_training.pkl', 'rb') as f:
    data = pickle.load(f)

# data is list of tuples: (source, pdb_id, smiles, affinity)
# Convert to arrays
smiles_list = [d[2] for d in data]
affinity = np.array([d[3] for d in data])

# Use molecular descriptors directly (fast)
def calc_props(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(16)
    return np.array([
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.NumAromaticRings(mol),
        Descriptors.FractionCSP3(mol),
        Descriptors.NumHeteroatoms(mol),
        Descriptors.RingCount(mol),
        Descriptors.HeavyAtomCount(mol),
        Descriptors.NumValenceElectrons(mol),
        Descriptors.NumRadicalElectrons(mol),
        Descriptors.FlexibleCage(mol),
        Descriptors.BertzCT(mol),
        Descriptors.Chi0(mol)
    ])

print("Computing molecular descriptors...")
props = np.array([calc_props(s) for s in smiles_list])
print(f"Props shape: {props.shape}")

# Remove rows with all zeros
valid = ~(props == 0).all(axis=1)
props = props[valid]
affinity = affinity[valid]
print(f"Valid samples: {len(props)}")

# Add target as feature (for correlation analysis)
X = props
y = affinity

# === CORRELATION ANALYSIS ===
print("\n=== FEATURE CORRELATIONS WITH AFFINITY ===")
feature_names = ['MolWt', 'LogP', 'TPSA', 'HDonors', 'HAcceptors', 'RotBonds', 
              'AromRings', 'CSP3', 'Heteroatoms', 'RingCount', 'HeavyAtom', 
              'ValenceE', 'RadicalE', 'Flexible', 'BertzCT', 'Chi0']
correlations = []
for i in range(X.shape[1]):
    corr = np.corrcoef(X[:, i], y)[0, 1]
    correlations.append((feature_names[i], corr))
    
correlations.sort(key=lambda x: abs(x[1]), reverse=True)
for name, corr in correlations[:10]:
    print(f"  {name}: {corr:.4f}")

# === LEARNING CURVES with different sample sizes ===
print("\n=== LEARNING CURVES ===")

kf = KFold(n_splits=3, shuffle=True, random_state=42)
results = []

# Sample sizes
sizes = [500, 1000, 2000, 4000, 8000, 12000]
np.random.seed(42)

for n in sizes:
    if n > len(X):
        n = len(X)
    
    idx = np.random.permutation(len(X))[:n]
    X_sub, y_sub = X[idx], y[idx]
    
    model = GradientBoostingRegressor(
        n_estimators=50,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        subsample=0.8,
        min_samples_leaf=5
    )
    
    scores = cross_val_score(model, X_sub, y_sub, cv=kf, scoring='r2')
    results.append({'n': n, 'r2': scores.mean(), 'std': scores.std()})
    print(f"n={n:5d}: R²={scores.mean():.4f} ± {scores.std():.4f}")

# === PLOT ===
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Learning curve
ax1 = axes[0]
ns = [r['n'] for r in results]
r2s = [r['r2'] for r in results]
stds = [r['std'] for r in results]
ax1.fill_between(ns, [r-s for r,s in zip(r2s, stds)], [r+s for r,s in zip(r2s, stds)], alpha=0.2)
ax1.plot(ns, r2s, 'b-o', linewidth=2, markersize=8)
ax1.set_xlabel('Training Samples')
ax1.set_ylabel('R² Score')
ax1.set_title('Learning Curve: GradientBoosting\n(with molecular descriptors only)')
ax1.grid(True, alpha=0.3)
ax1.set_xscale('log')
ax1.set_ylim(0, 1)

# Plot 2: Feature correlations
ax2 = axes[1]
top_corrs = correlations[:10]
names = [c[0] for c in top_corrs]
vals = [c[1] for c in top_corrs]
colors = ['green' if v > 0 else 'red' for v in vals]
ax2.barh(names, vals, color=colors)
ax2.set_xlabel('Correlation with pKD')
ax2.set_title('Feature Correlations with Binding Affinity')
ax2.axvline(0, color='black', linewidth=0.5)

plt.tight_layout()
plt.savefig('.cache/geock_autoresearch/learning_curves_molprops.png', dpi=150)
print(f"\nSaved: .cache/geock_autoresearch/learning_curves_molprops.png")