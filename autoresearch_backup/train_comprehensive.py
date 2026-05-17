#!/usr/bin/env python3
"""
GEOCK v2 - COMPREHENSIVE TRAINING
Combines ALL data sources + proper overfitting detection + best configs
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.model_selection import KFold, train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from rdkit import Chem
from rdkit.Chem import AllChem
import xgboost as xgb
import warnings

warnings.filterwarnings("ignore")
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

print("=" * 70)
print("GEOCK v2 - COMPREHENSIVE FINAL TRAINING")
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
print("\n[1/7] Collecting ALL unique data...")

all_data = {}

# Load from all pickle files
pkl_files = [
    "lp_new_features_8k_no2016.pkl",
    "geock_training_data_no2016.pkl",
    "geock_training_data.pkl",
    "lp_features_enhanced.pkl",
    "lp_all_features.pkl",
]

for pkl_file in pkl_files:
    path = cache_dir / pkl_file
    if path.exists():
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            for d in data:
                smiles = d.get("smiles")
                if smiles and smiles not in all_data:
                    all_data[smiles] = d
            print(f"  {pkl_file}: {len(data)}")
        except Exception as e:
            print(f"  {pkl_file}: ERROR {e}")

# Add from LP-PDBBind CSV
print("  Loading LP-PDBBind.csv...")
lp_df = pd.read_csv(cache_dir / "LP_PDBBind.csv")
for _, row in lp_df.iterrows():
    smiles = row["smiles"]
    if smiles and smiles not in all_data:
        all_data[smiles] = {
            "smiles": smiles,
            "affinity": row["value"],
            "source": "lp_csv",
        }
print(f"    Added {len(lp_df)} from CSV")

# Add from ChEMBL
print("  Loading ChEMBL...")
chembl_df = pd.read_csv(cache_dir / "chembl_binding.csv")
chembl_df = chembl_df.dropna(subset=["smiles", "affinity_nM"])
chembl_df = chembl_df[chembl_df["affinity_nM"] > 0]
added = 0
for _, row in chembl_df.iterrows():
    smiles = row["smiles"]
    if smiles and smiles not in all_data:
        try:
            pkd = 9 - np.log10(row["affinity_nM"])
            if 0 < pkd < 20:
                all_data[smiles] = {
                    "smiles": smiles,
                    "affinity": pkd,
                    "source": "chembl",
                }
                added += 1
        except:
            pass
print(f"    Added {added} unique from ChEMBL")

print(f"\n  TOTAL UNIQUE COMPOUNDS: {len(all_data)}")

# ==================== STEP 2: COMPUTE FINGERPRINTS ====================
print("\n[2/7] Computing fingerprints (2048 bits)...")


def get_fp(smiles, nBits=2048):
    if not isinstance(smiles, str):
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=nBits)
        return np.array(fp, dtype=np.float32)
    except:
        return None


# Process all
valid_data = []
for i, (smiles, data) in enumerate(all_data.items()):
    if i % 5000 == 0:
        print(f"  Processing: {i}/{len(all_data)}")

    # Use existing ECFP if available, otherwise compute
    ecfp = data.get("ecfp")
    if ecfp is not None:
        fp = np.array(ecfp, dtype=np.float32)
        if len(fp) < 512:
            # Pad or skip
            continue
        elif len(fp) > 512:
            fp = fp[:512]
    else:
        fp = get_fp(smiles)
        if fp is None:
            continue

    # For 512-bit, we need to handle differently - let's recompute all as 2048
    fp = get_fp(smiles, 2048)
    if fp is None:
        continue

    valid_data.append({"smiles": smiles, "affinity": data["affinity"], "fp": fp})

print(f"  Valid samples: {len(valid_data)}")

# ==================== STEP 3: BUILD FEATURES ====================
print("\n[3/7] Building feature matrix...")

X = np.array([d["fp"] for d in valid_data], dtype=np.float32)
y = np.array([d["affinity"] for d in valid_data], dtype=np.float32)

print(f"  X shape: {X.shape}")
print(f"  y range: {y.min():.2f} - {y.max():.2f}, mean: {y.mean():.2f}")

# ==================== STEP 4: TRAIN/VAL SPLIT ====================
print("\n[4/7] Creating train/validation split...")

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)
print(f"  Train: {len(X_train)}, Val: {len(X_val)}")

# Feature selection on training data ONLY
selector = SelectKBest(f_regression, k=500)
X_train_sel = selector.fit_transform(X_train, y_train)
X_val_sel = selector.transform(X_val)

# Standardize
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train_sel)
X_val_s = scaler.transform(X_val_sel)

print(f"  Selected features: {X_train_sel.shape[1]}")

# ==================== STEP 5: TRAIN WITH OVERFITTING CHECK ====================
print("\n[5/7] Training with overfitting detection...")

# Best configs to try
configs = [
    {
        "name": "config1",
        "max_depth": 14,
        "learning_rate": 0.025,
        "reg_alpha": 0.5,
        "reg_lambda": 2.5,
        "n_estimators": 500,
    },
    {
        "name": "config2",
        "max_depth": 16,
        "learning_rate": 0.02,
        "reg_alpha": 0.7,
        "reg_lambda": 3.0,
        "n_estimators": 600,
    },
    {
        "name": "config3",
        "max_depth": 12,
        "learning_rate": 0.03,
        "reg_alpha": 0.4,
        "reg_lambda": 2.0,
        "n_estimators": 500,
    },
]

best_config = None
best_gap = float("inf")
results = []

for cfg in configs:
    name = cfg.pop("name")

    model = xgb.XGBRegressor(
        **cfg,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        random_state=42,
        verbosity=0,
        n_jobs=-1,
    )
    model.fit(X_train_s, y_train)

    train_pred = model.predict(X_train_s)
    val_pred = model.predict(X_val_s)

    train_r, _ = pearsonr(y_train, train_pred)
    val_r, _ = pearsonr(y_val, val_pred)

    train_r2 = train_r**2
    val_r2 = val_r**2
    gap = train_r2 - val_r2

    results.append(
        {
            "name": name,
            "config": {**cfg, "name": name},
            "train_r2": train_r2,
            "val_r2": val_r2,
            "gap": gap,
            "model": model,
        }
    )

    print(f"  {name}: Train R²={train_r2:.4f}, Val R²={val_r2:.4f}, Gap={gap:.4f}")
    cfg["name"] = name

# Find best with acceptable overfitting
valid_results = [r for r in results if r["gap"] < 0.15]  # Max 15% gap
if valid_results:
    best = max(valid_results, key=lambda x: x["val_r2"])
else:
    best = min(results, key=lambda x: x["gap"])

print(
    f"\n  BEST: {best['name']} with Val R²={best['val_r2']:.4f}, Gap={best['gap']:.4f}"
)

# ==================== STEP 6: CROSS-VALIDATION ====================
print("\n[6/7] Full 5-fold cross-validation...")

# Retrain on all data with best config
X_all_sel = selector.fit_transform(X, y)
X_all_s = scaler.fit_transform(X_all_sel)

best_cfg = {k: v for k, v in best["config"].items() if k != "name"}
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_scores = []

for fold, (tr_idx, vl_idx) in enumerate(kf.split(X_all_s)):
    model = xgb.XGBRegressor(
        **best_cfg,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        random_state=42 + fold,
        verbosity=0,
        n_jobs=-1,
    )
    model.fit(X_all_s[tr_idx], y[tr_idx])
    pred = model.predict(X_all_s[vl_idx])
    r, _ = pearsonr(y[vl_idx], pred)
    fold_scores.append(r**2)
    print(f"    Fold {fold + 1}: R² = {fold_scores[-1]:.4f}")

cv_r2 = np.mean(fold_scores)
cv_r = np.sqrt(cv_r2)
print(f"\n  CV R: {cv_r:.4f}")
print(f"  CV R²: {cv_r2:.4f} ± {np.std(fold_scores):.4f}")

# ==================== STEP 7: SAVE FINAL MODEL ====================
print("\n[7/7] Training final model...")

final_model = xgb.XGBRegressor(
    **best_cfg,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    random_state=42,
    verbosity=0,
    n_jobs=-1,
)
final_model.fit(X_all_s, y)

model_data = {
    "model": final_model,
    "scaler": scaler,
    "selector": selector,
    "config": best_cfg,
    "cv_r": cv_r,
    "cv_r2": cv_r2,
    "cv_std": np.std(fold_scores),
    "fold_scores": fold_scores,
    "train_r2": best["train_r2"],
    "val_r2": best["val_r2"],
    "overfit_gap": best["gap"],
    "n_features": X_all_sel.shape[1],
    "n_samples": len(y),
    "date": pd.Timestamp.now().isoformat(),
}

output_path = work_dir / "geock_v2_final_v2.pkl"
with open(output_path, "wb") as f:
    pickle.dump(model_data, f)

print(f"\n{'=' * 70}")
print("FINAL RESULTS")
print(f"{'=' * 70}")
print(f"  Samples: {len(y)}")
print(f"  Features: {X_all_sel.shape[1]}")
print(f"  Config: {best_cfg}")
print(f"  Train R²: {best['train_r2']:.4f}")
print(f"  Val R²: {best['val_r2']:.4f}")
print(f"  Overfit Gap: {best['gap']:.4f}")
print(f"  CV R²: {cv_r2:.4f} ± {np.std(fold_scores):.4f}")
print(f"  CV R: {cv_r:.4f}")
print(f"\n  Saved: {output_path}")
print(f"{'=' * 70}")
