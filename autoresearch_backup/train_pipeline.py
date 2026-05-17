#!/usr/bin/env python3
"""
GEOCK Training Pipeline
=====================
Unified training pipeline using merged data.

Usage:
    python train_pipeline.py                    # Default: XGBoost quick
    python train_pipeline.py --model xgboost   # XGBoost
    python train_pipeline.py --model neural    # Neural network
    python train_pipeline.py --model both       # Both models
    python train_pipeline.py --epochs 200     # More epochs for NN
    python train_pipeline.py --folds 5         # CV folds
"""

import os
import sys
import pickle
import argparse
import json
import time
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler


# Use cross-platform paths
def _get_cache_dir():
    linux = Path("/home/chow/.cache/geock_autoresearch")
    if linux.exists():
        return linux
    win = Path(os.path.expanduser("~/OneDrive/.cache/geock_autoresearch"))
    if win.exists():
        return win
    return Path("./cache")


def _get_work_dir():
    linux = Path("/home/chow/autoresearch")
    if linux.exists():
        return linux
    win = Path(os.path.expanduser("~/OneDrive/autoresearch"))
    if win.exists():
        return win
    return Path(".")


CACHE_DIR = _get_cache_dir()
WORK_DIR = _get_work_dir()

print("=" * 60)
print("GEOCK Training Pipeline")
print("=" * 60)


# ===== FEATURE EXTRACTION =====
def extract_features(smiles_list, fp_size=512):
    """Extract ECFP fingerprints from SMILES."""
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, Lipinski

    features = []
    valid_idx = []

    for i, smi in enumerate(smiles_list):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue

            # ECFP
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=fp_size)
            fp_arr = np.array(fp, dtype=np.float32)

            # Basic molecular descriptors
            try:
                desc = np.array(
                    [
                        Descriptors.MolWt(mol),
                        Descriptors.MolLogP(mol),
                        Descriptors.TPSA(mol),
                        Lipinski.NumHDonors(mol),
                        Lipinski.NumHAcceptors(mol),
                        Lipinski.NumRotatableBonds(mol),
                        Lipinski.NumHeavyAtoms(mol),
                    ],
                    dtype=np.float32,
                )
            except:
                desc = np.zeros(7, dtype=np.float32)

            features.append(np.concatenate([fp_arr, desc]))
            valid_idx.append(i)

        except Exception as e:
            continue

    return np.array(features), valid_idx


# ===== XGBOOST MODEL =====
def train_xgboost(X, y, n_folds=5, n_estimators=500, max_depth=12):
    """Train XGBoost model with cross-validation."""
    import xgboost as xgb

    print(f"\n[XGBoost] Training with {n_folds}-fold CV...")
    print(f"  max_depth={max_depth}, n_estimators={n_estimators}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_scores = []
    models = []

    for fold, (tr_idx, vl_idx) in enumerate(kf.split(X_scaled)):
        print(f"\n  Fold {fold + 1}/{n_folds}...")

        # Feature selection INSIDE fold
        selector = SelectKBest(f_regression, k=min(400, X.shape[1]))
        X_tr = selector.fit_transform(X_scaled[tr_idx], y[tr_idx])
        X_vl = selector.transform(X_scaled[vl_idx])

        # XGBoost
        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            max_depth=max_depth,
            n_estimators=n_estimators,
            learning_rate=0.01,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            random_state=42 + fold,
            verbosity=0,
            n_jobs=-1,
        )

        model.fit(X_tr, y[tr_idx])

        preds = model.predict(X_vl)
        r, _ = pearsonr(y[vl_idx], preds)
        r2 = r**2
        fold_scores.append(r2)
        models.append((model, scaler, selector))

        print(f"    R² = {r2:.4f}")

    cv_r2 = np.mean(fold_scores)
    cv_std = np.std(fold_scores)

    print(f"\n  XGBoost CV R² = {cv_r2:.4f} ± {cv_std:.4f}")

    return models, fold_scores, scaler


# ===== NEURAL NETWORK MODEL =====
def train_neural(X, y, n_folds=5, epochs=100, hidden_dims=[512, 256, 128, 64]):
    """Train neural network with cross-validation."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader

    print(f"\n[Neural Network] Training with {n_folds}-fold CV...")
    print(f"  Architecture: {X.shape[1]} → {' → '.join(map(str, hidden_dims))} → 1")
    print(f"  Epochs: {epochs}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    class TabularRegressor(nn.Module):
        def __init__(self, input_dim, hidden_dims, dropout=0.3):
            super().__init__()
            layers = []
            prev = input_dim
            for h in hidden_dims:
                layers.extend(
                    [
                        nn.Linear(prev, h),
                        nn.BatchNorm1d(h),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                    ]
                )
                prev = h
            layers.append(nn.Linear(prev, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x).squeeze(-1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_scores = []
    models = []

    for fold, (tr_idx, vl_idx) in enumerate(kf.split(X_scaled)):
        print(f"\n  Fold {fold + 1}/{n_folds}...")

        # Feature selection INSIDE fold
        selector = SelectKBest(f_regression, k=min(400, X.shape[1]))
        X_tr = selector.fit_transform(X_scaled[tr_idx], y[tr_idx])
        X_vl = selector.transform(X_scaled[vl_idx])

        # Data
        train_ds = TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(y[tr_idx]))
        train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)

        val_X = torch.FloatTensor(X_vl)
        val_y = torch.FloatTensor(y[vl_idx])

        # Model
        model = TabularRegressor(X_tr.shape[1], hidden_dims).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", patience=10, factor=0.5
        )
        criterion = nn.MSELoss()

        best_r2 = 0
        best_state = None
        no_improve = 0

        for epoch in range(epochs):
            model.train()
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                loss = criterion(model(bx), by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            # Validate
            model.eval()
            with torch.no_grad():
                preds = model(val_X.to(device)).cpu().numpy()
                r, _ = pearsonr(val_y.numpy(), preds)
                r2 = r**2 if r > 0 else 0

            scheduler.step(r2)

            if r2 > best_r2:
                best_r2 = r2
                best_state = model.state_dict().copy()
                no_improve = 0
            else:
                no_improve += 1

            if (epoch + 1) % 20 == 0:
                print(f"    Epoch {epoch + 1}: R² = {r2:.4f}")

            if no_improve >= 30:
                print(f"    Early stop at epoch {epoch + 1}")
                break

        if best_state:
            model.load_state_dict(best_state)

        fold_scores.append(best_r2)
        models.append((model, scaler, selector))

        print(f"    Best R² = {best_r2:.4f}")

    cv_r2 = np.mean(fold_scores)
    cv_std = np.std(fold_scores)

    print(f"\n  Neural Net CV R² = {cv_r2:.4f} ± {cv_std:.4f}")

    return models, fold_scores, scaler


# ===== MAIN =====
def main():
    parser = argparse.ArgumentParser(description="GEOCK Training Pipeline")
    parser.add_argument(
        "--model",
        type=str,
        default="xgboost",
        choices=["xgboost", "neural", "both"],
        help="Model type",
    )
    parser.add_argument(
        "--data", type=str, default="merged_39k.pkl", help="Input data file"
    )
    parser.add_argument("--folds", type=int, default=5, help="CV folds")
    parser.add_argument("--epochs", type=int, default=100, help="Epochs for NN")
    parser.add_argument("--output", type=str, default=None, help="Output model name")
    parser.add_argument("--features", type=int, default=512, help="Fingerprint size")

    args = parser.parse_args()

    # Load data
    data_path = WORK_DIR / args.data
    if not data_path.exists():
        data_path = CACHE_DIR / args.data

    print(f"\n[1] Loading data: {data_path}")

    if data_path.suffix == ".pkl":
        with open(data_path, "rb") as f:
            data = pickle.load(f)
    else:
        df = pd.read_csv(data_path)
        data = df.to_dict("records")

    print(f"  Loaded: {len(data)} entries")

    # Extract
    print(f"\n[2] Extracting features...")
    smiles_list = [d["smiles"] for d in data]
    y = np.array([d["affinity"] for d in data], dtype=np.float32)

    X, valid_idx = extract_features(smiles_list, fp_size=args.features)
    y = y[valid_idx]

    print(f"  Features: {X.shape}")
    print(f"  Valid entries: {len(y)}")

    # Train
    results = {}

    if args.model in ["xgboost", "both"]:
        print(f"\n[3] Training XGBoost...")
        xgb_models, xgb_scores, xgb_scaler = train_xgboost(X, y, n_folds=args.folds)
        results["xgboost"] = {
            "models": xgb_models,
            "cv_r2": np.mean(xgb_scores),
            "fold_scores": xgb_scores,
            "scaler": xgb_scaler,
        }

    if args.model in ["neural", "both"]:
        print(f"\n[4] Training Neural Network...")
        nn_models, nn_scores, nn_scaler = train_neural(
            X, y, n_folds=args.folds, epochs=args.epochs
        )
        results["neural"] = {
            "models": nn_models,
            "cv_r2": np.mean(nn_scores),
            "fold_scores": nn_scores,
            "scaler": nn_scaler,
        }

    # Save
    output_name = (
        args.output
        or f"geock_model_{args.model}_{datetime.now().strftime('%Y%m%d')}.pkl"
    )
    output_path = WORK_DIR / output_name

    model_data = {
        "model_type": args.model,
        "results": results,
        "best_cv_r2": max(r["cv_r2"] for r in results.values()),
        "n_samples": len(y),
        "n_features": X.shape[1],
        "date": datetime.now().isoformat(),
        "fp_size": args.features,
    }

    with open(output_path, "wb") as f:
        pickle.dump(model_data, f)

    print(f"\n✅ Saved: {output_path}")
    print(f"  Best CV R²: {model_data['best_cv_r2']:.4f}")

    # Save JSON
    json_path = output_path.with_suffix(".json")
    json_data = {
        "model_type": args.model,
        "best_cv_r2": model_data["best_cv_r2"],
        "n_samples": len(y),
        "n_features": X.shape[1],
        "date": model_data["date"],
    }
    for name, res in results.items():
        json_data[f"{name}_cv_r2"] = res["cv_r2"]
        json_data[f"{name}_fold_scores"] = res["fold_scores"]

    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)

    print(f"  Results: {json_path}")

    return results


if __name__ == "__main__":
    main()
