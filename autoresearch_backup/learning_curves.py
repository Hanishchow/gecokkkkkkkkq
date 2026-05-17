import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
import warnings
warnings.filterwarnings('ignore')
import os

# Load PDBBind data
print("Loading data...")
import pandas as pd
df = pd.read_csv('.cache/geock_autoresearch/LP_PDBBind.csv')

# Parse Kd values
def parse_kd(val):
    if pd.isna(val):
        return np.nan
    val = str(val).strip()
    try:
        if '=' in val:
            parts = val.split('=')
            num_str = parts[1].replace('nM','').replace('uM','*1000').replace('mM','*1000000').replace('M','*1000000000')
            return eval(num_str) if '*' in num_str else float(num_str)
        return float(val)
    except:
        return np.nan

df['kd_nM'] = df['value'].apply(parse_kd)
df['pKD'] = -np.log10(df['kd_nM'].values * 1e-9 + 1e-15)
df = df.dropna(subset=['pKD', 'smiles'])

print(f"Total samples after cleaning: {len(df)}")

# Generate molecular fingerprints
def smiles_to_fp(smiles, radius=2, bits=512):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(bits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=bits)
    return np.array(fp)

print("Generating fingerprints...")
X = np.array([smiles_to_fp(s) for s in df['smiles']])
y = df['pKD'].values

print(f"Features shape: {X.shape}")
print(f"Target range: {y.min():.2f} - {y.max():.2f}")

# Add molecular properties as features
def calc_props(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [0]*8
    return [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.NumAromaticRings(mol),
        Descriptors.FractionCSP3(mol)
    ]

props = np.array([calc_props(s) for s in df['smiles']])
X = np.hstack([X, props])
print(f"Features with props: {X.shape}")

# Save feature matrix for later use
np.save('.cache/geock_autoresearch/X_features.npy', X)
np.save('.cache/geock_autoresearch/y_targets.npy', y)

# === LEARNING CURVES ===
print("\n" + "="*50)
print("LEARNING CURVES ANALYSIS")
print("="*50)

# Different data sizes to test
data_sizes = [500, 1000, 2000, 4000, 8000, 12000, 16000, len(X)]
results = []

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for n_samples in data_sizes:
    if n_samples > len(X):
        n_samples = len(X)
    
    # Random sample
    idx = np.random.RandomState(42).permutation(len(X))[:n_samples]
    X_sub, y_sub = X[idx], y[idx]
    
    # Train GradientBoosting (fixed params, no tuning)
    model = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        subsample=0.8,
        min_samples_leaf=10
    )
    
    scores = cross_val_score(model, X_sub, y_sub, cv=kf, scoring='r2')
    mean_r2 = scores.mean()
    std_r2 = scores.std()
    
    results.append({
        'n_samples': n_samples,
        'r2_mean': mean_r2,
        'r2_std': std_r2,
        'fold_scores': scores
    })
    
    print(f"N={n_samples:5d}: R² = {mean_r2:.4f} ± {std_r2:.4f}")

# Also test RandomForest
print("\n--- Testing RandomForest ---")
results_rf = []
for n_samples in data_sizes[:5]:  # Quick test
    if n_samples > len(X):
        n_samples = len(X)
    idx = np.random.RandomState(42).permutation(len(X))[:n_samples]
    X_sub, y_sub = X[idx], y[idx]
    
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    scores = cross_val_score(model, X_sub, y_sub, cv=kf, scoring='r2')
    results_rf.append({
        'n_samples': n_samples,
        'r2_mean': scores.mean(),
        'r2_std': scores.std()
    })
    print(f"N={n_samples:5d}: R² = {scores.mean():.4f} ± {scores.std():.4f}")

# === PLOT LEARNING CURVES ===
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: GB Learning Curve
ax1 = axes[0]
n_sizes = [r['n_samples'] for r in results]
r2_means = [r['r2_mean'] for r in results]
r2_stds = [r['r2_std'] for r in results]

ax1.errorbar(n_sizes, r2_means, yerr=r2_stds, marker='o', capsize=5, linewidth=2)
ax1.set_xlabel('Training Samples', fontsize=12)
ax1.set_ylabel('R² Score', fontsize=12)
ax1.set_title('Learning Curve: GradientBoosting', fontsize=14)
ax1.grid(True, alpha=0.3)
ax1.set_xscale('log')

# Plot 2: Compare models
ax2 = axes[1]
ax2.plot([r['n_samples'] for r in results], r2_means, 'b-o', label='GradientBoosting', linewidth=2)
ax2.plot([r['n_samples'] for r in results_rf], [r['r2_mean'] for r in results_rf], 'r-s', label='RandomForest', linewidth=2)
ax2.set_xlabel('Training Samples', fontsize=12)
ax2.set_ylabel('R² Score', fontsize=12)
ax2.set_title('Model Comparison: Learning Curves', fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xscale('log')

plt.tight_layout()
plt.savefig('.cache/geock_autoresearch/learning_curves.png', dpi=150)
print(f"\nSaved: .cache/geock_autoresearch/learning_curves.png")

# Save results
with open('.cache/geock_autoresearch/learning_curves_results.pkl', 'wb') as f:
    pickle.dump({
        'gradient_boosting': results,
        'random_forest': results_rf
    }, f)

print(f"\n=== SUMMARY ===")
print(f"Best R² with {results[-1]['n_samples']} samples: {results[-1]['r2_mean']:.4f}")
print("="*50)