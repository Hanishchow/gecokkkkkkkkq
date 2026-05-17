#!/usr/bin/env python3
"""
GEOCK v2 - Neural Network FOLD-BY-FOLD
Run ONE fold at a time: python train_nn_fold.py --fold 1
"""

import pickle
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import warnings

warnings.filterwarnings("ignore")

print("=" * 60)
print("GEOCK v2 - NEURAL NETWORK (Fold-by-Fold)")
print("=" * 60)

# Import path helpers
try:
    from geock_paths import get_cache_dir, get_work_dir

    cache_dir = get_cache_dir()
    work_dir = get_work_dir()
except ImportError:
    cache_dir = Path("/home/chow/.cache/geock_autoresearch")
    work_dir = Path("/home/chow/autoresearch")

# ========== DATA LOADING ==========
print("\n[1] Loading data...")

# Try merged first, if not available combine
merged_path = cache_dir / "merged_39k.pkl"
if merged_path.exists():
    with open(merged_path, "rb") as f:
        data = pickle.load(f)
    print(f"  Loaded merged: {len(data)} samples")
else:
    # Load separate files and merge
    f1 = cache_dir / "lp_new_features_8k_no2016.pkl"
    f2 = cache_dir / "geock_training_data_no2016.pkl"
    data = []
    if f1.exists():
        with open(f1, "rb") as f:
            d1 = pickle.load(f)
        data.extend(d1)
        print(f"  Loaded lp_new_features_8k: {len(d1)} samples")
    if f2.exists():
        with open(f2, "rb") as f:
            d2 = pickle.load(f)
        data.extend(d2)
        print(f"  Loaded geock_training_data: {len(d2)} samples")

    # Deduplicate by (smiles, pdb_id) - need to find common keys
    # For now just use what we have

    if not data:
        print("ERROR: No data files found!")
        print(f"  Looking in: {cache_dir}")
        exit(1)

X = np.array([d["ecfp"] for d in data], dtype=np.float32)
y = np.array([d["affinity"] for d in data], dtype=np.float32)
print(f"  Total: {len(data)} samples, {X.shape[1]} features")

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ========== MODEL ==========
class TabularRegressor(nn.Module):
    def __init__(
        self, input_dim=400, hidden_dims=[512, 256, 128, 64], dropout_rate=0.3
    ):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        return self.network(x).squeeze(-1)


class ChunkedDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(np.array([self.y[idx]]))


# ========== TRAINING ==========
def train_fold(fold_idx, X_scaled, y, kf, device, max_epochs=150):
    print(f"\n[2] Training FOLD {fold_idx + 1}/5...")
    print("=" * 60)

    for fold_num, (tr_idx, vl_idx) in enumerate(kf.split(X_scaled)):
        if fold_num != fold_idx:
            continue

        # Feature selection INSIDE fold
        selector = SelectKBest(f_regression, k=400)
        X_tr_sel = selector.fit_transform(X_scaled[tr_idx], y[tr_idx])
        X_vl_sel = selector.transform(X_scaled[vl_idx])

        print(f"  Train: {len(tr_idx)}, Val: {len(vl_idx)}")
        print(f"  Features after selection: {X_tr_sel.shape[1]}")

        # Datasets
        train_dataset = ChunkedDataset(X_tr_sel, y[tr_idx])
        val_dataset = ChunkedDataset(X_vl_sel, y[vl_idx])

        train_loader = DataLoader(
            train_dataset, batch_size=256, shuffle=True, num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, batch_size=256, shuffle=False, num_workers=0
        )

        # Model
        model = TabularRegressor(
            input_dim=X_tr_sel.shape[1], hidden_dims=[512, 256, 128, 64]
        )
        model = model.to(device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6
        )
        criterion = nn.MSELoss()

        best_val_loss = float("inf")
        best_model_state = None
        no_improve = 0

        for epoch in range(max_epochs):
            model.train()
            train_loss = 0.0
            for data, target in train_loader:
                data, target = data.to(device), target.to(device).squeeze(-1)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)

            # Validate
            model.eval()
            val_loss = 0.0
            val_preds = []
            val_targets = []
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device).squeeze(-1)
                    output = model(data)
                    loss = criterion(output, target)
                    val_loss += loss.item()
                    val_preds.extend(output.cpu().numpy())
                    val_targets.extend(target.cpu().numpy())

            avg_val_loss = val_loss / len(val_loader)
            r, _ = pearsonr(val_targets, val_preds)
            val_r2 = r**2

            scheduler.step(avg_val_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {epoch + 1:3d}/{max_epochs} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | R²: {val_r2:.4f} | LR: {current_lr:.6f}"
            )

            # Save best
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = model.state_dict().copy()
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= 30:
                    print(f"  Early stopping at epoch {epoch + 1}")
                    break

        # Final eval with best model
        model.load_state_dict(best_model_state)
        model.eval()
        final_preds = []
        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(device)
                output = model(data)
                final_preds.extend(output.cpu().numpy())

        r, _ = pearsonr(y[vl_idx], final_preds)
        r2 = r**2

        print(f"\n>>> FOLD {fold_idx + 1} FINAL: R² = {r2:.4f}, R = {r:.4f}")

        # Save fold result
        result = {
            "fold": fold_idx + 1,
            "r2": r2,
            "r": r,
            "val_indices": vl_idx.tolist(),
            "predictions": final_preds,
            "date": pd.Timestamp.now().isoformat(),
        }

        fold_path = work_dir / f"nn_fold_{fold_idx + 1}_result.pkl"
        with open(fold_path, "wb") as f:
            pickle.dump(result, f)
        print(f"  Saved: {fold_path}")

        return r2, r


# ========== MAIN ==========
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=1, help="Fold number 1-5")
    parser.add_argument("--epochs", type=int, default=150, help="Max epochs per fold")
    args = parser.parse_args()

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    r2, r = train_fold(args.fold - 1, X_scaled, y, kf, device, max_epochs=args.epochs)

    print(f"\n{'=' * 60}")
    print(f"FOLD {args.fold} COMPLETE: R² = {r2:.4f}")
    print(f"{'=' * 60}")
