#!/usr/bin/env python3
"""
GEOCK v2 - Neural Network (PyTorch)
Learns representations from fingerprints with backpropagation
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
import warnings

warnings.filterwarnings("ignore")

print("=" * 60)
print("GEOCK v2 - NEURAL NETWORK (PyTorch)")
print("=" * 60)

# Import path helpers
try:
    from geock_paths import get_cache_dir, get_work_dir

    cache_dir = get_cache_dir()
    work_dir = get_work_dir()
except ImportError:
    cache_dir = Path("/home/chow/.cache/geock_autoresearch")
    work_dir = Path("/home/chow/autoresearch")

# Load data
print("\n[1] Loading data...")
with open(cache_dir / "lp_new_features_8k_no2016.pkl", "rb") as f:
    data = pickle.load(f)

X = np.array([d["ecfp"] for d in data], dtype=np.float32)
y = np.array([d["affinity"] for d in data], dtype=np.float32)
print(f"  {len(data)} samples, {X.shape[1]} features")

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_tensor = torch.FloatTensor(X_scaled)
y_tensor = torch.FloatTensor(y)


# Neural Network
class BindingNet(nn.Module):
    def __init__(self, input_dim=512, hidden_dims=[256, 128, 64]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, dim), nn.ReLU(), nn.Dropout(0.2)])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.network(x).squeeze(-1)


# Training function
def train_model(model, X_train, y_train, epochs=100, lr=0.001):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 50 == 0:
            print(
                f"    Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(loader):.4f}"
            )


# Evaluate
def evaluate(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        pred = model(X_test).numpy()
    r, _ = pearsonr(y_test.numpy(), pred)
    return r**2


# 5-fold CV
print("\n[2] 5-fold CV with Neural Network...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_scores = []

for fold, (tr_idx, vl_idx) in enumerate(kf.split(X_scaled)):
    print(f"\n  Fold {fold + 1}/5:")

    X_tr = torch.FloatTensor(X_scaled[tr_idx])
    y_tr = torch.FloatTensor(y[tr_idx])
    X_vl = torch.FloatTensor(X_scaled[vl_idx])
    y_vl = torch.FloatTensor(y[vl_idx])

    model = BindingNet(input_dim=X.shape[1])
    train_model(model, X_tr, y_tr, epochs=100, lr=0.001)

    r2 = evaluate(model, X_vl, y_vl)
    fold_scores.append(r2)
    print(f"    Fold {fold + 1} R² = {r2:.4f}")

cv_r2 = np.mean(fold_scores)
cv_r = np.sqrt(cv_r2)
print(f"\n{'=' * 60}")
print(f"NEURAL NETWORK RESULTS:")
print(f"  CV R² = {cv_r2:.4f} ± {np.std(fold_scores):.4f}")
print(f"  CV R = {cv_r:.4f}")
print(f"\n  vs XGBoost: R² = 0.5956")
print(f"  Difference: {cv_r2 - 0.5956:.4f}")
print(f"{'=' * 60}")

# Train final model
print("\n[3] Training final model...")
final_model = BindingNet(input_dim=X.shape[1])
final_X = torch.FloatTensor(X_scaled)
final_y = torch.FloatTensor(y)
train_model(final_model, final_X, final_y, epochs=150, lr=0.001)

# Save
model_data = {
    "model": final_model,
    "scaler": scaler,
    "config": {
        "architecture": "BindingNet",
        "hidden_dims": [256, 128, 64],
        "epochs": 150,
        "learning_rate": 0.001,
        "optimizer": "Adam",
    },
    "cv_r2": cv_r2,
    "cv_r": cv_r,
    "fold_scores": fold_scores,
    "n_samples": len(y),
    "date": pd.Timestamp.now().isoformat(),
}

output_path = work_dir / "geock_v2_neural.pkl"
with open(output_path, "wb") as f:
    pickle.dump(model_data, f)

print(f"\nSaved: {output_path}")
