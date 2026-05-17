#!/usr/bin/env python3
"""
Add protein context to binding affinity model using simple sequence features.
- Amino acid composition (20 features)
- Dipeptide composition (sampled 50 most common)
- Length, MW, aromaticity
- Combined with ECFP ligand features
"""
import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/chow/autoresearch")

from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut, cross_val_predict, KFold
from scipy.stats import pearsonr

AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")

AA_MW = {
    'A': 89.1, 'C': 121.2, 'D': 133.1, 'E': 147.1, 'F': 165.2,
    'G': 75.1, 'H': 155.2, 'I': 131.2, 'K': 146.2, 'L': 131.2,
    'M': 149.2, 'N': 132.1, 'P': 115.1, 'Q': 146.2, 'R': 174.2,
    'S': 105.1, 'T': 119.1, 'V': 117.1, 'W': 204.2, 'Y': 181.2
}

AROMATIC_AA = set('FWY')

def compute_aa_composition(seq):
    """Compute amino acid composition (20 features)."""
    seq = seq.upper()
    n = len(seq) if len(seq) > 0 else 1
    return np.array([seq.count(aa) / n for aa in AA_LIST], dtype=np.float32)

def compute_dipeptide_composition(seq, top_n=50):
    """Compute dipeptide composition, return top N most common."""
    seq = seq.upper()
    dipeptides = {}
    for i in range(len(seq) - 1):
        dp = seq[i:i+2]
        if 'X' not in dp and '-' not in dp:
            dipeptides[dp] = dipeptides.get(dp, 0) + 1
    
    total = sum(dipeptides.values()) if dipeptides else 1
    sorted_dps = sorted(dipeptides.items(), key=lambda x: -x[1])[:top_n]
    
    result = np.zeros(top_n, dtype=np.float32)
    for i, (dp, count) in enumerate(sorted_dps):
        result[i] = count / total
    
    return result, [dp for dp, _ in sorted_dps]

def compute_sequence_features(seq):
    """Compute simple sequence features."""
    seq = seq.upper()
    seq = seq.replace('-', '').replace('X', '')
    
    if len(seq) == 0:
        return np.zeros(25 + 50, dtype=np.float32)  # 20 AA + 5 basic + 50 dipep
    
    # Basic features
    length = len(seq)
    mw = sum(AA_MW.get(aa, 110) for aa in seq)  # avg 110 if unknown
    aromatic_count = sum(1 for aa in seq if aa in AROMATIC_AA)
    aromaticity = aromatic_count / length
    
    # Charge at pH 7
    pos_charge = sum(1 for aa in seq if aa in 'KRH')
    neg_charge = sum(1 for aa in seq if aa in 'DE')
    net_charge = (pos_charge - neg_charge) / length
    
    # Isoelectric point approximation
    dE = sum(1 for aa in seq if aa in 'DE')
    dK = sum(1 for aa in seq if aa in 'KRH')
    if dE == dK:
        pI = 7.0
    elif dE > dK:
        pI = 6.0 - 0.5 * np.log10(dE - dK + 1e-10)
    else:
        pI = 8.0 + 0.5 * np.log10(dK - dE + 1e-10)
    
    basic_feats = np.array([
        length / 1000,  # normalized
        mw / 10000,     # normalized  
        aromaticity,
        net_charge,
        pI / 14         # normalized
    ], dtype=np.float32)
    
    # AA composition (20 features)
    aa_comp = compute_aa_composition(seq)
    
    # Dipeptide composition (50 features)
    dp_comp, dp_names = compute_dipeptide_composition(seq, top_n=50)
    
    return np.concatenate([aa_comp, basic_feats, dp_comp]), dp_names

def get_ecfp(smiles, fp_size=512):
    """Compute ECFP4 fingerprint."""
    try:
        from rdkit import Chem
        from rdkit.Chem import rdFingerprintGenerator
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(fp_size, dtype=np.float32)
        fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=fp_size)
        return np.array(fpgen.GetFingerprintAsNumPy(mol), dtype=np.float32)
    except:
        return np.zeros(fp_size, dtype=np.float32)

def evaluate_r(y_true, y_pred):
    if len(y_true) < 2:
        return 0.0
    r, _ = pearsonr(y_true, y_pred)
    return float(r) if not np.isnan(r) else 0.0

def evaluate_mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("  LOADING LP-PDBBind DATA")
print("=" * 70)

DATA_PATH = "CACHE_DIR / LP_PDBBind.csv"
df = pd.read_csv(DATA_PATH)

print(f"Loaded {len(df)} entries from LP-PDBBind.csv")
print(f"Columns: {list(df.columns)}")

# Filter entries with valid sequence and SMILES
df_valid = df.dropna(subset=['seq', 'smiles', 'value'])
df_valid = df_valid[df_valid['seq'].str.len() > 10]  # Minimum sequence length
df_valid = df_valid[df_valid['value'].notna()]
print(f"Valid entries with sequence and affinity: {len(df_valid)}")

# Parse affinity values
def parse_affinity(val):
    if pd.isna(val):
        return np.nan
    try:
        return float(val)
    except:
        return np.nan

df_valid = df_valid.copy()
df_valid['affinity'] = df_valid['value'].apply(parse_affinity)
df_valid = df_valid[df_valid['affinity'].notna()]
df_valid = df_valid[(df_valid['affinity'] > 0) & (df_valid['affinity'] < 100000)]

print(f"Entries with valid affinity (0-100000 nM): {len(df_valid)}")

# Convert to pKd
df_valid['pkd'] = -np.log10(df_valid['affinity'] * 1e-9 + 1e-10)
df_valid['pkd'] = df_valid['pkd'].clip(2, 12)

print(f"pKd range: {df_valid['pkd'].min():.2f} - {df_valid['pkd'].max():.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# EXTRACT FEATURES
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  EXTRACTING PROTEIN SEQUENCE FEATURES")
print("=" * 70)

# Use train/val/test split from the CSV
split_map = {'train': 0, 'val': 1, 'test': 2}
df_valid['split_idx'] = df_valid['new_split'].map(split_map)

# Handle missing splits - assign randomly
df_valid.loc[df_valid['split_idx'].isna(), 'split_idx'] = np.random.randint(
    0, 3, size=df_valid['split_idx'].isna().sum()
)

# Get unique sequences for efficiency
print("Computing sequence features for all unique sequences...")
unique_seqs = df_valid['seq'].unique()
seq_to_features = {}

for i, seq in enumerate(unique_seqs):
    if i % 500 == 0:
        print(f"  Processing {i}/{len(unique_seqs)}...")
    feat, dp_names = compute_sequence_features(seq)
    seq_to_features[seq] = feat

print(f"Computed features for {len(unique_seqs)} unique sequences")

# Build feature matrix
protein_features = np.array([seq_to_features[s] for s in df_valid['seq']])

# ECFP for ligands
print("\nComputing ECFP fingerprints for ligands...")
ecfp_features = np.array([get_ecfp(s) for s in df_valid['smiles']])

# Combine features
X_protein = protein_features  # 25 + 50 = 75 features
X_ligand = ecfp_features      # 512 features
X_combined = np.hstack([X_protein, X_ligand])

y = df_valid['pkd'].values
pdb_ids = df_valid.iloc[:, 0].values
splits = df_valid['split_idx'].values

N_TOTAL = len(y)
print(f"\nTotal: {N_TOTAL} samples")
print(f"Protein features: {X_protein.shape[1]}")
print(f"Ligand features: {X_ligand.shape[1]}")
print(f"Combined features: {X_combined.shape[1]}")

# ─────────────────────────────────────────────────────────────────────────────
# TRAIN MODELS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  TRAINING RIDGE REGRESSION MODELS")
print("=" * 70)

# Splits
train_mask = splits == 0
val_mask = splits == 1
test_mask = splits == 2

X_train = X_combined[train_mask]
X_val = X_combined[val_mask]
X_test = X_combined[test_mask]
y_train = y[train_mask]
y_val = y[val_mask]
y_test = y[test_mask]

# Also get ligand-only data
X_train_lig = X_ligand[train_mask]
X_val_lig = X_ligand[val_mask]
X_test_lig = X_ligand[test_mask]

# Also get protein-only data
X_train_prot = X_protein[train_mask]
X_val_prot = X_protein[val_mask]
X_test_prot = X_protein[test_mask]

print(f"\nTrain: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")
print(f"Train PDBs: {pdb_ids[train_mask][:5]}...")
print(f"Val PDBs:   {pdb_ids[val_mask][:5]}...")
print(f"Test PDBs:  {pdb_ids[test_mask][:5]}...")

# Normalize
def normalize(X_tr, X_vl=None, X_te=None):
    mu = X_tr.mean(0)
    sd = X_tr.std(0)
    sd = np.where(sd == 0, 1, sd)
    X_tr_n = (X_tr - mu) / sd
    X_vl_n = (X_vl - mu) / sd if X_vl is not None else None
    X_te_n = (X_te - mu) / sd if X_te is not None else None
    return X_tr_n, X_vl_n, X_te_n, mu, sd

# Train and evaluate
def train_and_eval(X_tr, X_vl, X_te, y_tr, y_vl, y_te, name):
    X_tr_n, X_vl_n, X_te_n, mu, sd = normalize(X_tr, X_vl, X_te)
    
    best_loo_r = -999
    best_alpha = 1.0
    
    # Quick alpha search
    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
        loo = LeaveOneOut()
        lp = cross_val_predict(Ridge(alpha=alpha), X_tr_n, y_tr, cv=loo)
        lr = evaluate_r(y_tr, lp)
        if lr > best_loo_r:
            best_loo_r = lr
            best_alpha = alpha
    
    # Final model
    model = Ridge(alpha=best_alpha)
    model.fit(X_tr_n, y_tr)
    
    val_pred = model.predict(X_vl_n)
    test_pred = model.predict(X_te_n)
    
    # LOO prediction for LOO-R
    loo = LeaveOneOut()
    loo_pred = cross_val_predict(Ridge(alpha=best_alpha), X_tr_n, y_tr, cv=loo)
    loo_r = evaluate_r(y_tr, loo_pred)
    loo_mae = evaluate_mae(y_tr, loo_pred)
    
    val_r = evaluate_r(y_vl, val_pred)
    val_mae = evaluate_mae(y_vl, val_pred)
    test_r = evaluate_r(y_te, test_pred)
    test_mae = evaluate_mae(y_te, test_pred)
    
    print(f"\n{name}:")
    print(f"  Best alpha: {best_alpha}")
    print(f"  Val-R:   {val_r:.4f}, Val-MAE: {val_mae:.4f}")
    print(f"  LOO-R:   {loo_r:.4f}, LOO-MAE: {loo_mae:.4f}")
    print(f"  Test-R:  {test_r:.4f}, Test-MAE: {test_mae:.4f}")
    
    return {
        'val_r': val_r, 'val_mae': val_mae,
        'loo_r': loo_r, 'loo_mae': loo_mae,
        'test_r': test_r, 'test_mae': test_mae,
        'alpha': best_alpha, 'model': model, 'mu': mu, 'sd': sd
    }

# Train ligand-only model (baseline)
print("\n" + "-" * 50)
results_ligand = train_and_eval(
    X_train_lig, X_val_lig, X_test_lig,
    y_train, y_val, y_test,
    "LIGAND-ONLY (ECFP)"
)

# Train protein-only model
print("\n" + "-" * 50)
results_protein = train_and_eval(
    X_train_prot, X_val_prot, X_test_prot,
    y_train, y_val, y_test,
    "PROTEIN-ONLY"
)

# Train combined model
print("\n" + "-" * 50)
results_combined = train_and_eval(
    X_train, X_val, X_test,
    y_train, y_val, y_test,
    "COMBINED (Protein + Ligand)"
)

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  SUMMARY")
print("=" * 70)

print(f"\n{'Model':<25} {'Val-R':>8} {'LOO-R':>8} {'Test-R':>8} {'MAE':>8}")
print("-" * 60)
print(f"{'Ligand-only (ECFP)':<25} {results_ligand['val_r']:>8.4f} {results_ligand['loo_r']:>8.4f} {results_ligand['test_r']:>8.4f} {results_ligand['test_mae']:>8.4f}")
print(f"{'Protein-only':<25} {results_protein['val_r']:>8.4f} {results_protein['loo_r']:>8.4f} {results_protein['test_r']:>8.4f} {results_protein['test_mae']:>8.4f}")
print(f"{'Combined':<25} {results_combined['val_r']:>8.4f} {results_combined['loo_r']:>8.4f} {results_combined['test_r']:>8.4f} {results_combined['test_mae']:>8.4f}")

improvement = results_combined['loo_r'] - results_ligand['loo_r']
print(f"\nLOO-R improvement with protein: {improvement:+.4f}")

# Save results
results = {
    'ligand_only': results_ligand,
    'protein_only': results_protein,
    'combined': results_combined,
    'pdb_ids': pdb_ids.tolist(),
    'splits': splits.tolist(),
    'y': y.tolist()
}

# Save to pickle
CACHE_DIR = Path("/home/chow/.cache/geock_autoresearch")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

with open(CACHE_DIR / "protein_context_results.pkl", "wb") as f:
    pickle.dump(results, f)

# Save TSV
tsv_path = CACHE_DIR / "results_protein.tsv"
with open(tsv_path, "w") as f:
    f.write("model\tval_r\tloo_r\ttest_r\ttest_mae\n")
    f.write(f"ligand_only\t{results_ligand['val_r']:.4f}\t{results_ligand['loo_r']:.4f}\t{results_ligand['test_r']:.4f}\t{results_ligand['test_mae']:.4f}\n")
    f.write(f"protein_only\t{results_protein['val_r']:.4f}\t{results_protein['loo_r']:.4f}\t{results_protein['test_r']:.4f}\t{results_protein['test_mae']:.4f}\n")
    f.write(f"combined\t{results_combined['val_r']:.4f}\t{results_combined['loo_r']:.4f}\t{results_combined['test_r']:.4f}\t{results_combined['test_mae']:.4f}\n")

print(f"\nResults saved to:")
print(f"  {CACHE_DIR / 'protein_context_results.pkl'}")
print(f"  {tsv_path}")

# Save model
model_data = {
    'model': results_combined['model'],
    'mu': results_combined['mu'],
    'sd': results_combined['sd'],
    'alpha': results_combined['alpha'],
    'feature_names': (
        [f'aa_{aa}' for aa in AA_LIST] + 
        ['length', 'mw', 'aromaticity', 'net_charge', 'pI'] +
        [f'dp_{i}' for i in range(50)] +
        [f'ecfp_{i}' for i in range(512)]
    )
}
with open(CACHE_DIR / "model_protein.pkl", "wb") as f:
    pickle.dump(model_data, f)

print(f"  {CACHE_DIR / 'model_protein.pkl'}")
print("\nDone!")
