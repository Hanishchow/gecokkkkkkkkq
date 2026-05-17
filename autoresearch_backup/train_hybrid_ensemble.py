#!/usr/bin/env python3
"""Hybrid Ensemble: XGBoost + NN (trained on full data)."""

import pickle
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import warnings

warnings.filterwarnings("ignore")

# Import path helpers
try:
    from geock_paths import get_cache_dir, get_work_dir

    cache_dir = get_cache_dir()
    work_dir = get_work_dir()
except ImportError:
    cache_dir = Path("/home/chow/.cache/geock_autoresearch")
    work_dir = Path("/home/chow/autoresearch")

print("=" * 60)
print("GEOCK v2 - HYBRID ENSEMBLE (XGBoost + NN)")
print("=" * 60)

# Load data
with open(cache_dir / "merged_39k.pkl", "rb") as f:
    data = pickle.load(f)

X = np.array([d["ecfp"] for d in data], dtype=np.float32)
y = np.array([d["affinity"] for d in data], dtype=np.float32)
print(f"Loaded: {len(data)} samples, {X.shape[1]} features")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ========== TABULAR NN ==========
class TabularRegressor(nn.Module):
    def __init__(self, input_dim=400, hidden_dims=[256, 128, 64], dropout_rate=0.3):
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

    def forward(self, x):
        return self.network(x).squeeze(-1)


# ========== QUICK NN TRAINING (50 epochs only) ==========
print("\n[1] Training NN on FULL 39K data (50 epochs)...")

selector = SelectKBest(f_regression, k=400)
X_sel = selector.fit_transform(X_scaled, y)


class ChunkedDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(np.array([self.y[idx]]))


dataset = ChunkedDataset(X_sel, y)
loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=2)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  Device: {device}")

model_nn = TabularRegressor(input_dim=X_sel.shape[1]).to(device)
optimizer = optim.AdamW(model_nn.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.MSELoss()

print("  Training NN...")
for epoch in range(50):  # Only 50 epochs for speed
    model_nn.train()
    total_loss = 0.0
    for data_batch, target_batch in loader:
        data_batch, target_batch = (
            data_batch.to(device),
            target_batch.to(device).squeeze(-1),
        )
        optimizer.zero_grad()
        output = model_nn(data_batch)
        loss = criterion(output, target_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_nn.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    if (epoch + 1) % 10 == 0:
        print(f"    Epoch {epoch + 1}/50, Loss: {total_loss / len(loader):.4f}")

# ========== XGBOOST ON FULL DATA ==========
print("\n[2] Training XGBoost on FULL 39K data...")

model_xgb = xgb.XGBRegressor(
    objective="reg:squarederror",
    max_depth=12,
    n_estimators=500,
    learning_rate=0.01,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.1,
    random_state=42,
)
model_xgb.fit(X_sel, y, verbose=False)
print("  XGBoost trained.")

# ========== HYBRID PREDICTIONS ==========
print("\n[3] Creating hybrid ensemble predictions...")

# XGBoost predictions
pred_xgb = model_xgb.predict(X_sel)

# NN predictions
model_nn.eval()
pred_nn = []
with torch.no_grad():
    for data_batch, _ in loader:
        data_batch = data_batch.to(device)
        output = model_nn(data_batch)
        pred_nn.extend(output.cpu().numpy())
pred_nn = np.array(pred_nn)

# Ensemble: simple average
pred_ensemble = (pred_xgb + pred_nn) / 2.0

# Calculate R² for each
r_xgb, _ = pearsonr(y, pred_xgb)
r_nn, _ = pearsonr(y, pred_nn)
r_ensemble, _ = pearsonr(y, pred_ensemble)

print(f"\n{'=' * 60}")
print("HYBRID ENSEMBLE RESULTS (39K samples):")
print(f"{'=' * 60}")
print(f"  XGBoost R²     = {r_xgb**2:.4f} (R = {r_xgb:.4f})")
print(f"  NN R² (50 ep)  = {r_nn**2:.4f} (R = {r_nn:.4f})")
print(f"  Hybrid Ensemble = {r_ensemble**2:.4f} (R = {r_ensemble:.4f})")
print(f"\n  Target (original): R² = 0.7118")
print(f"  Best single (XGB39K): R² = 0.7169")
print(f"{'=' * 60}")

# Save ensemble
model_data = {
    "model_xgb": model_xgb,
    "model_nn_state": model_nn.cpu().state_dict(),
    "nn_architecture": "TabularRegressor",
    "nn_input_dim": X_sel.shape[1],
    "scaler": scaler,
    "selector": selector,
    "r2_xgb": float(r_xgb**2),
    "r2_nn": float(r_nn**2),
    "r2_ensemble": float(r_ensemble**2),
    "n_samples": len(y),
    "date": __import__("pandas").Timestamp.now().isoformat(),
}

output_path = work_dir / "geock_v2_hybrid_ensemble.pkl"
with open(output_path, "wb") as f:
    pickle.dump(model_data, f)

print(f"\nSaved: {output_path}")
print(f"{'=' * 60}")
