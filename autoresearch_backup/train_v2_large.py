#!/usr/bin/env python3
"""
Train GEOCK v2 with larger data chunks - matching original best model config.
Original: CV R² = 0.8437, n_samples = 39109, n_features = 500
Using LP-PDBBind (~19k) + ChEMBL (~4k) = ~23k samples
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
from rdkit.Chem import AllChem
import xgboost as xgb
import warnings

warnings.filterwarnings("ignore")

# ==================== LOAD DATA ====================
print("=" * 60)
print("GEOCK v2 - Large Chunk Training")
print("=" * 60)

# Import path helpers
try:
    from geock_paths import get_cache_dir, get_work_dir

    cache_dir = get_cache_dir()
    work_dir = get_work_dir()
except ImportError:
    cache_dir = Path("/home/chow/.cache/geock_autoresearch")
    work_dir = Path("/home/chow/autoresearch")

# Load LP-PDBBind
print("\n[1/5] Loading LP-PDBBind data...")
lp_df = pd.read_csv(cache_dir / "LP_PDBBind.csv")
print(f"  LP-PDBBind: {len(lp_df)} samples")

# Load ChEMBL
print("[2/5] Loading ChEMBL data...")
chembl_path = cache_dir / "chembl_binding.csv"
if chembl_path.exists():
    chembl_df = pd.read_csv(chembl_path)
    # Filter valid SMILES and values - column is 'affinity_nM'
    chembl_df = chembl_df.dropna(subset=["smiles", "affinity_nM"])
    chembl_df = chembl_df[chembl_df["affinity_nM"] > 0]
    print(f"  ChEMBL: {len(chembl_df)} samples")
else:
    chembl_df = pd.DataFrame()
    print("  ChEMBL: Not found")

# ==================== COMPUTE FINGERPRINTS ====================
print("\n[3/5] Computing Morgan fingerprints...")


def smiles_to_fp(smiles, radius=2, nBits=2048):
    """Convert SMILES to Morgan fingerprint"""
    if not isinstance(smiles, str):
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    return np.array(fp)


# Process LP-PDBBind
lp_data = []
for i, row in lp_df.iterrows():
    if i % 2000 == 0:
        print(f"  Processing LP-PDBBind: {i}/{len(lp_df)}")

    fp = smiles_to_fp(row["smiles"], nBits=2048)
    if fp is not None:
        lp_data.append(
            {
                "smiles": row["smiles"],
                "affinity": row["value"],  # pKd
                "fp": fp,
                "source": "lp",
            }
        )

print(f"  LP-PDBBind valid: {len(lp_data)}")

# Process ChEMBL - convert nM to pKd (pKd = -log10(Kd in M) = -log10(Kd_nM * 1e-9) = 9 - log10(Kd_nM))
chembl_data = []
if len(chembl_df) > 0:
    for i, row in chembl_df.iterrows():
        if i % 1000 == 0:
            print(f"  Processing ChEMBL: {i}/{len(chembl_df)}")

        fp = smiles_to_fp(row["smiles"], nBits=2048)
        if fp is not None:
            # Convert nM to pKd
            try:
                nm_value = row["affinity_nM"]
                if nm_value > 0:
                    pkd = 9 - np.log10(nm_value)
                    if 0 < pkd < 20:  # Reasonable pKd range
                        chembl_data.append(
                            {
                                "smiles": row["smiles"],
                                "affinity": pkd,
                                "fp": fp,
                                "source": "chembl",
                            }
                        )
            except:
                pass

print(f"  ChEMBL valid: {len(chembl_data)}")

# Combine data
all_data = lp_data + chembl_data
print(f"  Total combined: {len(all_data)} samples")

# ==================== BUILD FEATURES ====================
print("\n[4/5] Building feature matrix...")

X_list = [d["fp"] for d in all_data]
y_list = [d["affinity"] for d in all_data]

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.float32)

print(f"  Raw features: {X.shape}")

# Feature selection - select top 500 like original
print("  Applying SelectKBest(k=500)...")
selector = SelectKBest(f_regression, k=500)
X_selected = selector.fit_transform(X, y)
print(f"  Selected features: {X_selected.shape}")

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

# ==================== CROSS-VALIDATION ====================
print("\n[5/5] Training with 5-Fold CV...")

# Same config as best original model
config = {
    "n_estimators": 200,
    "max_depth": 10,
    "learning_rate": 0.05,
    "reg_alpha": 0.5,
    "reg_lambda": 2.0,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}

print(f"  Config: {config}")

kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_scores = []
models = []

for fold, (tr_idx, vl_idx) in enumerate(kf.split(X_scaled)):
    X_tr, X_vl = X_scaled[tr_idx], X_scaled[vl_idx]
    y_tr, y_vl = y[tr_idx], y[vl_idx]

    model = xgb.XGBRegressor(
        **config, min_child_weight=3, random_state=42 + fold, verbosity=0, n_jobs=-1
    )
    model.fit(X_tr, y_tr)

    pred = model.predict(X_vl)
    r, _ = pearsonr(y_vl, pred)
    r2 = r**2

    fold_scores.append(r2)
    models.append(model)
    print(f"    Fold {fold + 1}: R = {r:.4f}, R² = {r2:.4f}")

cv_mean = np.mean(fold_scores)
cv_std = np.std(fold_scores)
print(f"\n  CV R²: {cv_mean:.4f} ± {cv_std:.4f}")

# ==================== SAVE MODEL ====================
print("\n[Saving]...")

model_data = {
    "models": models,
    "scaler": scaler,
    "selector": selector,
    "config": config,
    "cv_r": np.sqrt(cv_mean),  # R from R²
    "cv_r2": cv_mean,
    "cv_std": cv_std,
    "fold_scores": fold_scores,
    "n_features": X_selected.shape[1],
    "n_samples": len(y),
    "date": pd.Timestamp.now().isoformat(),
}

# Save with version
version = 4
output_path = f"WORK_DIR / geock_v2_run{version}.pkl"
with open(output_path, "wb") as f:
    pickle.dump(model_data, f)

print(f"  Saved to: {output_path}")
print(f"  Samples: {len(y)}, Features: {X_selected.shape[1]}")
print(f"  CV R²: {cv_mean:.4f}")

# Also train final model on all data
print("\n[Training final model on all data]...")
final_model = xgb.XGBRegressor(
    **config, min_child_weight=3, random_state=42, verbosity=0, n_jobs=-1
)
final_model.fit(X_scaled, y)

final_data = {
    "model": final_model,
    "scaler": scaler,
    "selector": selector,
    "config": config,
    "cv_r": np.sqrt(cv_mean),
    "cv_r2": cv_mean,
    "cv_std": cv_std,
    "fold_scores": fold_scores,
    "n_features": X_selected.shape[1],
    "n_samples": len(y),
    "date": pd.Timestamp.now().isoformat(),
}

final_path = f"WORK_DIR / geock_v2_final.pkl"
with open(final_path, "wb") as f:
    pickle.dump(final_data, f)

print(f"  Final model saved to: {final_path}")
print("\n" + "=" * 60)
print(f"DONE! CV R² = {cv_mean:.4f}")
print("=" * 60)
