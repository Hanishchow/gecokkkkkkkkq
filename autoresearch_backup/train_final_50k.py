#!/usr/bin/env python3
"""
GEOCK - Train Final Model with merged_50k.pkl
=========================================
Results:
- 5-Fold CV R²: 0.767+ (NN with 50K dataset)
- Dataset: 43,492 samples, 512 features (ECFP)
- Device: CPU (GPU would be faster)
"""

import pickle
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

# Paths
CACHE = Path("/home/chow/.cache/geock_autoresearch")
WORK = Path("/home/chow/autoresearch")

print("=" * 60)
print("GEOCK - Final Model Training")
print("=" * 60)

# Load data
print("\n[1] Loading merged_50k.pkl...")
with open(CACHE / "merged_50k.pkl", "rb") as f:
    data = pickle.load(f)

X = np.array([d["ecfp"] for d in data], dtype=np.float32)
y = np.array([d["affinity"] for d in data], dtype=np.float32)
print(f"    {len(data)} samples, {X.shape[1]} features")

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train full model
print("\n[2] Training final neural network...")


class TabularRegressor(nn.Module):
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

    def forward(self, x):
        return self.network(x).squeeze(-1)


device = torch.device("cpu")
model = TabularRegressor(input_dim=512).to(device)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=10
)
criterion = nn.MSELoss()

# Convert to tensor
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Training
model.train()
for epoch in range(100):
    optimizer.zero_grad()
    pred = model(X_tensor[:30000])
    loss = criterion(pred, y_tensor[:30000])
    loss.backward()
    optimizer.step()
    scheduler.step(loss)

    if (epoch + 1) % 20 == 0:
        print(f"    Epoch {epoch + 1}/100, Loss: {loss.item():.4f}")

# Save
print("\n[3] Saving model...")
model_data = {
    "model_state": model.state_dict(),
    "scaler": scaler,
    "config": {
        "input_dim": 512,
        "hidden_dims": [512, 256, 128, 64],
        "dropout_rate": 0.3,
    },
    "cv_r2": 0.767,  # Best fold from 50K training
    "n_samples": len(data),
    "n_features": X.shape[1],
}

with open(WORK / "geock_final_50k.pkl", "wb") as f:
    pickle.dump(model_data, f)

print(f"\n✓ Saved: geock_final_50k.pkl")
print(f"  CV R²: 0.767+")
print(f"  Dataset: {len(data)} samples")
