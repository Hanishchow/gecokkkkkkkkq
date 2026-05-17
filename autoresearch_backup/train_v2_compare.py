#!/usr/bin/env python3
"""
Train GEOCK v2 - Try different configurations to improve performance
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

# ==================== LOAD DATA ====================
print("=" * 60)
print("GEOCK v2 - Alternative Configurations")
print("=" * 60)

# Import path helpers
try:
    from geock_paths import get_cache_dir

    cache_dir = get_cache_dir()
except ImportError:
    cache_dir = Path("/home/chow/.cache/geock_autoresearch")

# Load LP-PDBBind only first (see if ChEMBL is hurting)
print("\n[1/5] Loading data...")
lp_df = pd.read_csv(cache_dir / "LP_PDBBind.csv")
print(f"  LP-PDBBind: {len(lp_df)} samples")

# ==================== COMPUTE FINGERPRINTS + PHYSICS ====================
print("\n[2/5] Computing features...")


def compute_features(smiles):
    """Compute Morgan fingerprint + physics features"""
    if not isinstance(smiles, str):
        return None, None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None

    # Morgan fingerprint
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    fp_arr = np.array(fp, dtype=np.float32)

    # Physics features
    try:
        phys = np.array(
            [
                Descriptors.MolWt(mol),
                Descriptors.MolLogP(mol),
                Descriptors.TPSA(mol),
                Lipinski.NumHDonors(mol),
                Lipinski.NumHAcceptors(mol),
                Lipinski.NumRotatableBonds(mol),
                Lipinski.NumAromaticRings(mol),
                Lipinski.NumHeteroatoms(mol),
                Lipinski.NumHeavyAtoms(mol),
                Descriptors.FractionCSP3(mol),
                Lipinski.RingCount(mol),
                Descriptors.BertzCT(mol),
                Descriptors.Chi0(mol),
                Descriptors.Chi1(mol),
            ],
            dtype=np.float32,
        )
    except:
        phys = np.zeros(14, dtype=np.float32)

    return fp_arr, phys


# Process data
data_list = []
for i, row in lp_df.iterrows():
    if i % 3000 == 0:
        print(f"  Processing: {i}/{len(lp_df)}")

    fp, phys = compute_features(row["smiles"])
    if fp is not None:
        data_list.append(
            {"smiles": row["smiles"], "affinity": row["value"], "fp": fp, "phys": phys}
        )

print(f"  Valid samples: {len(data_list)}")

# ==================== BUILD FEATURES ====================
print("\n[3/5] Building feature matrix...")

X_fp = np.array([d["fp"] for d in data_list], dtype=np.float32)
X_phys = np.array([d["phys"] for d in data_list], dtype=np.float32)
y = np.array([d["affinity"] for d in data_list], dtype=np.float32)

print(f"  Fingerprints: {X_fp.shape}, Physics: {X_phys.shape}")

# Try different feature configurations
configs = [
    # Config 1: FP only, 500 features (baseline)
    {"name": "fp_500", "fp_k": 500, "use_phys": False},
    # Config 2: FP only, 1024 features
    {"name": "fp_1024", "fp_k": 1024, "use_phys": False},
    # Config 3: FP + physics, 500 FP features
    {"name": "fp500_phys", "fp_k": 500, "use_phys": True},
    # Config 4: FP + physics, 800 FP features
    {"name": "fp800_phys", "fp_k": 800, "use_phys": True},
]

results = []

for cfg in configs:
    print(f"\n[Testing: {cfg['name']}]")

    # Feature selection on fingerprints
    if cfg["fp_k"] < 2048:
        selector = SelectKBest(f_regression, k=cfg["fp_k"])
        X_fp_sel = selector.fit_transform(X_fp, y)
    else:
        X_fp_sel = X_fp

    # Combine with physics
    if cfg["use_phys"]:
        X = np.hstack([X_fp_sel, X_phys])
    else:
        X = X_fp_sel

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # XGBoost params - try different settings
    xgb_configs = [
        {
            "max_depth": 10,
            "learning_rate": 0.05,
            "reg_alpha": 0.5,
            "reg_lambda": 2.0,
            "n_estimators": 200,
        },
        {
            "max_depth": 8,
            "learning_rate": 0.1,
            "reg_alpha": 0.3,
            "reg_lambda": 1.0,
            "n_estimators": 200,
        },
        {
            "max_depth": 12,
            "learning_rate": 0.03,
            "reg_alpha": 1.0,
            "reg_lambda": 5.0,
            "n_estimators": 300,
        },
    ]

    best_r2 = 0
    best_model = None
    best_config = None

    for xgb_cfg in xgb_configs:
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        fold_scores = []

        for fold, (tr_idx, vl_idx) in enumerate(kf.split(X_scaled)):
            model = xgb.XGBRegressor(
                **xgb_cfg,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,
                random_state=42 + fold,
                verbosity=0,
                n_jobs=-1,
            )
            model.fit(X_scaled[tr_idx], y[tr_idx])

            pred = model.predict(X_scaled[vl_idx])
            r, _ = pearsonr(y[vl_idx], pred)
            fold_scores.append(r**2)

        cv_r2 = np.mean(fold_scores)
        print(f"    {xgb_cfg}: CV R² = {cv_r2:.4f}")

        if cv_r2 > best_r2:
            best_r2 = cv_r2
            best_config = xgb_cfg

    results.append(
        {
            "name": cfg["name"],
            "cv_r2": best_r2,
            "config": best_config,
            "n_features": X.shape[1],
        }
    )
    print(f"  BEST: {best_r2:.4f}")

# ==================== SUMMARY ====================
print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)
for r in results:
    print(f"  {r['name']}: R² = {r['cv_r2']:.4f}, features = {r['n_features']}")

# Save best
best = max(results, key=lambda x: x["cv_r2"])
print(f"\nBest: {best['name']} with R² = {best['cv_r2']:.4f}")
