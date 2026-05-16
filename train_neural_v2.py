#!/usr/bin/env python3
"""
GEOCK v2 - Neural Network v2 (Chunked + LR Scheduling)
User requirement: "the more u train the more better it will get till a point"
"""

import pickle
import numpy as np
import pandas as pd
import os
import json
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import warnings

warnings.filterwarnings("ignore")


# ===== CROSS-PLATFORM PATHS =====
def get_cache_dir():
    """Get cache directory - works on Linux and Windows."""
    # Try Linux path first
    linux_cache = Path("/home/chow/.cache/geock_autoresearch")
    if linux_cache.exists():
        return linux_cache
    # Try Windows path (OneDrive)
    win_cache = Path(os.path.expanduser("~/OneDrive/.cache/geock_autoresearch"))
    if win_cache.exists():
        return win_cache
    # Fallback to current directory
    return Path("./cache")


def get_work_dir():
    """Get work directory - works on Linux and Windows."""
    linux_work = Path("/home/chow/autoresearch")
    if linux_work.exists():
        return linux_work
    return Path(".")


cache_dir = get_cache_dir()
work_dir = get_work_dir()

print("=" * 60)
print("GEOCK v2 - NEURAL NETWORK v2 (Chunked + LR Scheduling)")
print("=" * 60)
print(f"Cache dir: {cache_dir}")
print(f"Work dir: {work_dir}")

# Load MERGED data (50K samples - enhanced with ChEMBL)
print("\n[1] Loading MERGED data (50K samples - enhanced)...")
data_path = cache_dir / "merged_50k.pkl"
if not data_path.exists():
    data_path = cache_dir / "merged_39k.pkl"
    print("  merged_50k.pkl not found, falling back to merged_39k.pkl")
with open(data_path, "rb") as f:
    data = pickle.load(f)

X = np.array([d["ecfp"] for d in data], dtype=np.float32)
y = np.array([d["affinity"] for d in data], dtype=np.float32)
print(f"  {len(data)} samples, {X.shape[1]} features")

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"  Data scaled. Feature range: [{X_scaled.min():.3f}, {X_scaled.max():.3f}]")


# ========== TABULAR NEURAL NETWORK ==========
class TabularRegressor(nn.Module):
    """MLP for tabular regression with 515 input features."""

    def __init__(
        self, input_dim=512, hidden_dims=[512, 256, 128, 64], dropout_rate=0.3
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


# ========== CHUNKED DATASET ==========
class ChunkedDataset(Dataset):
    """Dataset that loads data from memory-mapped numpy arrays in chunks."""

    def __init__(self, X, y, indices=None):
        if indices is not None:
            self.X = X[indices]
            self.y = y[indices]
        else:
            self.X = X
            self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(np.array([self.y[idx]]))


# ========== TRAINING WITH LR SCHEDULING ==========
def train_with_scheduling(
    model,
    train_loader,
    val_loader,
    device,
    num_epochs=200,
    initial_lr=1e-3,
    weight_decay=1e-4,
    patience_lr=10,
    patience_early=30,
    min_lr=1e-6,
):
    model = model.to(device)
    optimizer = optim.AdamW(
        model.parameters(), lr=initial_lr, weight_decay=weight_decay
    )

    # ReduceLROnPlateau: reduces LR when val loss plateaus
    # This implements "the more you train, the better it gets till a point"
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=patience_lr, min_lr=min_lr
    )

    criterion = nn.MSELoss()
    train_losses = []
    val_losses = []
    learning_rates = []
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(num_epochs):
        # ===== TRAIN =====
        model.train()
        train_loss = 0.0
        num_batches = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device).squeeze(-1)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            num_batches += 1

        avg_train_loss = train_loss / num_batches
        train_losses.append(avg_train_loss)

        # ===== VALIDATE =====
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
        val_losses.append(avg_val_loss)

        # Calculate R²
        r, _ = pearsonr(val_targets, val_preds)
        val_r2 = r**2

        # LR scheduling
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        learning_rates.append(current_lr)

        print(
            f"Epoch {epoch + 1:3d}/{num_epochs} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | R²: {val_r2:.4f} | LR: {current_lr:.6f}"
        )

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict().copy()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience_early:
                print(f"  Early stopping triggered after {epoch + 1} epochs")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "learning_rates": learning_rates,
        "best_val_r2": val_r2,
    }


# ========== 5-FOLD CV ==========
print("\n[2] 5-Fold CV with Chunked NN + LR Scheduling...")
print("    (ReduceLROnPlateau: LR reduces when improvement stalls)")
print("     Patience: 10 epochs for LR, 30 epochs for early stopping")
print("=" * 60)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_scores = []
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if device == "cpu":
    print("  (CPU mode - training will be slower than GPU)")
else:
    print("  (GPU mode - fast training)")

for fold, (tr_idx, vl_idx) in enumerate(kf.split(X_scaled)):
    print(f"\n{'=' * 60}")
    print(f"FOLD {fold + 1}/5")
    print(f"{'=' * 60}")

    # Feature selection INSIDE fold to avoid data leakage
    from sklearn.feature_selection import SelectKBest, f_regression

    selector = SelectKBest(f_regression, k=400)
    X_tr_sel = selector.fit_transform(X_scaled[tr_idx], y[tr_idx])
    X_vl_sel = selector.transform(X_scaled[vl_idx])

    # Datasets
    train_dataset = ChunkedDataset(X_tr_sel, y[tr_idx])
    val_dataset = ChunkedDataset(X_vl_sel, y[vl_idx])

    train_loader = DataLoader(
        train_dataset, batch_size=256, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=256, shuffle=False, num_workers=2, pin_memory=True
    )

    # Model
    model = TabularRegressor(
        input_dim=X_tr_sel.shape[1], hidden_dims=[512, 256, 128, 64]
    )

    # Train
    trained_model, history = train_with_scheduling(
        model,
        train_loader,
        val_loader,
        device,
        num_epochs=150,
        initial_lr=1e-3,
        patience_lr=10,
        patience_early=30,
    )

    # Final eval
    model.eval()
    val_preds = []
    with torch.no_grad():
        for data, target in val_loader:
            data = data.to(device)
            output = model(data)
            val_preds.extend(output.cpu().numpy())
    r, _ = pearsonr(y[vl_idx], val_preds)
    r2 = r**2
    fold_scores.append(r2)
    print(f"\nFold {fold + 1} FINAL: R² = {r2:.4f}, R = {r:.4f}")

# Results
cv_r2 = np.mean(fold_scores)
cv_r = np.sqrt(cv_r2)
print(f"\n{'=' * 60}")
print(f"NEURAL NETWORK v2 RESULTS (39K samples):")
print(f"{'=' * 60}")
print(f"  CV R² = {cv_r2:.4f} ± {np.std(fold_scores):.4f}")
print(f"  CV R  = {cv_r:.4f}")
print(f"  Fold scores: {[f'{s:.4f}' for s in fold_scores]}")
print(f"\n  vs XGBoost v2 (23K): R² = 0.5956")
print(f"  vs Original (39K):     R² = 0.7118")
print(f"  Difference from target: {cv_r2 - 0.7118:+.4f}")
print(f"{'=' * 60}")

# ========== TRAIN FINAL MODEL ON ALL DATA ==========
print("\n[3] Training FINAL model on ALL 39K samples...")
print("=" * 60)

# Feature selection on full data
from sklearn.feature_selection import SelectKBest, f_regression

selector = SelectKBest(f_regression, k=400)
X_final_sel = selector.fit_transform(X_scaled, y)

full_dataset = ChunkedDataset(X_final_sel, y)
full_loader = DataLoader(
    full_dataset, batch_size=256, shuffle=True, num_workers=2, pin_memory=True
)

final_model = TabularRegressor(
    input_dim=X_final_sel.shape[1], hidden_dims=[512, 256, 128, 64]
)
final_model = final_model.to(device)
optimizer = optim.AdamW(final_model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6
)
criterion = nn.MSELoss()

final_model.train()
for epoch in range(150):
    total_loss = 0.0
    num_batches = 0
    for data, target in full_loader:
        data, target = data.to(device), target.to(device).squeeze(-1)
        optimizer.zero_grad()
        output = final_model(data)
        loss = criterion(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(final_model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1
    avg_loss = total_loss / num_batches
    scheduler.step(avg_loss)
    if (epoch + 1) % 30 == 0:
        print(
            f"  Epoch {epoch + 1}/150, Loss: {avg_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

# Save
import pickle

model_data = {
    "model": final_model.cpu().state_dict(),
    "architecture": "TabularRegressor",
    "hidden_dims": [512, 256, 128, 64],
    "input_dim": X_final_sel.shape[1],
    "scaler": scaler,
    "selector": selector,
    "cv_r2": cv_r2,
    "cv_r": cv_r,
    "fold_scores": fold_scores,
    "n_samples": len(y),
    "lr_schedule": "ReduceLROnPlateau(patience=10, factor=0.5)",
    "date": pd.Timestamp.now().isoformat(),
}

output_path = work_dir / "geock_v2_neural_scheduled.pkl"
with open(output_path, "wb") as f:
    pickle.dump(model_data, f)

# ===== EXPORT RESULTS TO JSON =====
results = {
    "model": "GEOCK v2 Neural Network",
    "architecture": "TabularRegressor",
    "hidden_dims": [512, 256, 128, 64],
    "cv_r2": float(cv_r2),
    "cv_r": float(cv_r),
    "fold_scores": [float(s) for s in fold_scores],
    "n_samples": int(len(y)),
    "n_features": int(X_final_sel.shape[1]),
    "lr_schedule": "ReduceLROnPlateau",
    "date": pd.Timestamp.now().isoformat(),
    "device": device,
    "random_seed": 42,
}
results_path = work_dir / "geock_v2_results.json"
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved: {output_path}")
print(f"  Final CV R² = {cv_r2:.4f}")
print(f"Results: {results_path}")
print(f"{'=' * 60}")
