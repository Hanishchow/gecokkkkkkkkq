#!/usr/bin/env python3
"""
GEOCK v2 - Ultimate Training Script
Combines ALL available data sources to match/exceed original model's 39k samples
Uses exact same config as best original (CV R² = 0.84)
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski
import xgboost as xgb
import warnings

warnings.filterwarnings("ignore")
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

print("=" * 70)
print("GEOCK v2 - ULTIMATE TRAINING")
print("Combining ALL data sources")
print("=" * 70)

# Import path helpers
try:
    from geock_paths import get_cache_dir, get_work_dir

    cache_dir = get_cache_dir()
    work_dir = get_work_dir()
except ImportError:
    cache_dir = Path("/home/chow/.cache/geock_autoresearch")
    work_dir = Path("/home/chow/autoresearch")

# ==================== STEP 1: COLLECT ALL DATA ====================
print("\n[1/6] Collecting ALL data sources...")

all_smiles_affinity = []

# 1. LP-PDBBind CSV (19,443 samples)
print("  - Loading LP-PDBBind.csv...")
lp_df = pd.read_csv(cache_dir / "LP_PDBBind.csv")
for _, row in lp_df.iterrows():
    all_smiles_affinity.append(
        {
            "smiles": row["smiles"],
            "affinity": row["value"],  # pKd
            "source": "lp_pd bind",
        }
    )
print(f"    Added {len(lp_df)} samples from LP-PDBBind")

# 2. ChEMBL binding data (4,383 samples)
print("  - Loading ChEMBL...")
chembl_df = pd.read_csv(cache_dir / "chembl_binding.csv")
chembl_df = chembl_df.dropna(subset=["smiles", "affinity_nM"])
chembl_df = chembl_df[chembl_df["affinity_nM"] > 0]
for _, row in chembl_df.iterrows():
    try:
        pkd = 9 - np.log10(row["affinity_nM"])  # Convert nM to pKd
        if 0 < pkd < 20:
            all_smiles_affinity.append(
                {"smiles": row["smiles"], "affinity": pkd, "source": "chembl"}
            )
    except:
        pass
print(f"    Added {len(all_smiles_affinity) - len(lp_df)} samples from ChEMBL")

# 3. Try to add more from enhanced features
print("  - Loading lp_features_enhanced.pkl...")
try:
    with open(cache_dir / "lp_features_enhanced.pkl", "rb") as f:
        enhanced_data = pickle.load(f)
    initial_count = len(all_smiles_affinity)
    for d in enhanced_data:
        smiles = d.get("smiles")
        aff = d.get("affinity")
        if smiles and aff and isinstance(smiles, str):
            # Check if not duplicate
            if not any(s["smiles"] == smiles for s in all_smiles_affinity):
                all_smiles_affinity.append(
                    {"smiles": smiles, "affinity": aff, "source": "enhanced"}
                )
    print(
        f"    Added {len(all_smiles_affinity) - initial_count} unique samples from enhanced"
    )
except Exception as e:
    print(f"    Error: {e}")

print(f"\n  TOTAL raw samples: {len(all_smiles_affinity)}")

# ==================== STEP 2: COMPUTE FINGERPRINTS ====================
print("\n[2/6] Computing Morgan fingerprints (2048 bits)...")


def smiles_to_fp(smiles, radius=2, nBits=2048):
    if not isinstance(smiles, str):
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
        return np.array(fp, dtype=np.float32)
    except:
        return None


# Process all molecules
valid_data = []
for i, item in enumerate(all_smiles_affinity):
    if i % 5000 == 0:
        print(f"  Processing: {i}/{len(all_smiles_affinity)}")

    fp = smiles_to_fp(item["smiles"])
    if fp is not None:
        valid_data.append(
            {
                "smiles": item["smiles"],
                "affinity": item["affinity"],
                "fp": fp,
                "source": item["source"],
            }
        )

print(f"  Valid samples after fingerprinting: {len(valid_data)}")

# ==================== STEP 3: BUILD FEATURE MATRIX ====================
print("\n[3/6] Building feature matrix...")

X_list = [d["fp"] for d in valid_data]
y_list = [d["affinity"] for d in valid_data]

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.float32)

print(f"  Raw features shape: {X.shape}")
print(f"  Affinity range: {y.min():.2f} - {y.max():.2f}, mean: {y.mean():.2f}")

# ==================== STEP 4: FEATURE SELECTION ====================
print("\n[4/6] Feature selection (SelectKBest k=500)...")

selector = SelectKBest(f_regression, k=500)
X_selected = selector.fit_transform(X, y)

print(f"  Selected features shape: {X_selected.shape}")

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

# ==================== STEP 5: TRAIN WITH EXACT ORIGINAL CONFIG ====================
print("\n[5/6] Training XGBoost with original config...")

# Exact config from best original model
config = {
    "n_estimators": 200,
    "max_depth": 10,
    "learning_rate": 0.05,
    "reg_alpha": 0.5,
    "reg_lambda": 2.0,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
}

print(f"  Config: {config}")

# 5-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_scores_r = []
fold_scores_r2 = []
models = []

for fold, (tr_idx, vl_idx) in enumerate(kf.split(X_scaled)):
    X_tr, X_vl = X_scaled[tr_idx], X_scaled[vl_idx]
    y_tr, y_vl = y[tr_idx], y[vl_idx]

    model = xgb.XGBRegressor(**config, random_state=42 + fold, verbosity=0, n_jobs=-1)
    model.fit(X_tr, y_tr)

    pred = model.predict(X_vl)
    r, _ = pearsonr(y_vl, pred)
    r2 = r**2

    fold_scores_r.append(r)
    fold_scores_r2.append(r2)
    models.append(model)

    print(f"    Fold {fold + 1}: R = {r:.4f}, R² = {r2:.4f}")

cv_r = np.mean(fold_scores_r)
cv_r2 = cv_r**2
cv_std = np.std(fold_scores_r)

print(f"\n  CV Results:")
print(f"    CV R: {cv_r:.4f} ± {np.std(fold_scores_r):.4f}")
print(f"    CV R²: {cv_r2:.4f} ± {np.std(fold_scores_r2):.4f}")

# ==================== STEP 6: TRAIN FINAL MODEL ====================
print("\n[6/6] Training final model on all data...")

final_model = xgb.XGBRegressor(**config, random_state=42, verbosity=0, n_jobs=-1)
final_model.fit(X_scaled, y)

# Save model
model_data = {
    "model": final_model,
    "models": models,
    "scaler": scaler,
    "selector": selector,
    "config": config,
    "cv_r": cv_r,
    "cv_r2": cv_r2,
    "cv_std": cv_std,
    "fold_scores_r": fold_scores_r,
    "fold_scores_r2": fold_scores_r2,
    "n_features": X_selected.shape[1],
    "n_samples": len(y),
    "date": pd.Timestamp.now().isoformat(),
}

# Save with version
version = 5
output_path = work_dir / f"geock_v2_run{version}.pkl"
with open(output_path, "wb") as f:
    pickle.dump(model_data, f)

print(f"\n{'=' * 70}")
print(f"RESULTS")
print(f"{'=' * 70}")
print(f"  Samples: {len(y)}")
print(f"  Features: {X_selected.shape[1]}")
print(f"  CV R: {cv_r:.4f}")
print(f"  CV R²: {cv_r2:.4f}")
print(f"  Saved to: {output_path}")
print(f"{'=' * 70}")

# Compare with original
print(f"\n  COMPARISON:")
print(f"    Original: R² = 0.7118 (39,109 samples)")
print(f"    Current:  R² = {cv_r2:.4f} ({len(y)} samples)")
print(f"    Gap:      {0.7118 - cv_r2:.4f}")
