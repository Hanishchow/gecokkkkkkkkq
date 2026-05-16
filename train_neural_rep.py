#!/usr/bin/env python3
"""
GEOCK v2 - Neural Representation + XGBoost
Learns embeddings from fingerprints, then uses XGBoost on learned features
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings

warnings.filterwarnings("ignore")

print("=" * 60)
print("GEOCK v2 - NEURAL REPRESENTATION + XGB")
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


# ==================== NEURAL ENCODER ====================
class FeatureEncoder(nn.Module):
    def __init__(self, input_dim=512, hidden_dims=[256, 128], embedding_dim=64):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(dim),
                    nn.Dropout(0.3),
                ]
            )
            prev_dim = dim

        self.feature_extractor = nn.Sequential(*layers)
        self.embedding_layer = nn.Linear(prev_dim, embedding_dim)
        self.output_layer = nn.Linear(embedding_dim, 1)

    def forward(self, x):
        features = self.feature_extractor(x)
        embedding = self.embedding_layer(features)
        output = self.output_layer(
            embedding.squeeze(-1) if embedding.dim() > 2 else embedding
        )
        return output, embedding


def train_encoder(model, X_train, y_train, epochs=50, lr=0.001, batch_size=256):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            pred, _ = model(batch_X)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 20 == 0:
            print(
                f"    Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(loader):.4f}"
            )


def extract_embeddings(model, X_data):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_data)
        _, embeddings = model(X_tensor)
        return embeddings.numpy()


# ==================== TRAIN & EVALUATE ====================
print("\n[2] Training neural encoder (chunked)...")

kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_scores = []

for fold, (tr_idx, vl_idx) in enumerate(kf.split(X)):
    print(f"\n  Fold {fold + 1}/5:")

    # Scale inside fold
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X[tr_idx])
    X_vl = scaler.transform(X[vl_idx])

    # Train neural encoder on chunk
    model = FeatureEncoder(input_dim=X.shape[1], embedding_dim=64)
    train_encoder(model, X_tr, y[tr_idx], epochs=50, lr=0.001, batch_size=256)

    # Extract embeddings
    print(f"    Extracting embeddings...")
    train_emb = extract_embeddings(model, X_tr)
    val_emb = extract_embeddings(model, X_vl)

    # Combine original + embeddings
    X_tr_combined = np.hstack([X_tr, train_emb])
    X_vl_combined = np.hstack([X_vl, val_emb])

    # Feature selection
    selector = SelectKBest(f_regression, k=min(500, X_tr_combined.shape[1]))
    X_tr_sel = selector.fit_transform(X_tr_combined, y[tr_idx])
    X_vl_sel = selector.transform(X_vl_combined)

    # Train XGBoost on learned features
    xgb_model = xgb.XGBRegressor(
        max_depth=12,
        learning_rate=0.02,
        n_estimators=400,
        reg_alpha=0.5,
        reg_lambda=2.5,
        subsample=0.8,
        colsample_bytree=0.8,
        verbosity=0,
        n_jobs=-1,
        random_state=42 + fold,
    )
    xgb_model.fit(X_tr_sel, y[tr_idx])

    pred = xgb_model.predict(X_vl_sel)
    r, _ = pearsonr(y[vl_idx], pred)
    r2 = r**2
    fold_scores.append(r2)
    print(f"    Fold {fold + 1} R² = {r2:.4f}")

cv_r2 = np.mean(fold_scores)
cv_r = np.sqrt(cv_r2)
print(f"\n{'=' * 60}")
print(f"NEURAL + XGB RESULTS:")
print(f"  CV R² = {cv_r2:.4f} ± {np.std(fold_scores):.4f}")
print(f"  CV R = {cv_r:.4f}")
print(f"\n  vs XGBoost alone: R² = 0.5956")
print(f"  Improvement: {cv_r2 - 0.5956:.4f}")
print(f"{'=' * 60}")

# ==================== FINAL MODEL ====================
print("\n[3] Training final model...")

scaler_f = StandardScaler()
X_scaled = scaler_f.fit_transform(X)

final_model = FeatureEncoder(input_dim=X.shape[1], embedding_dim=64)
train_encoder(final_model, X_scaled, y, epochs=80, lr=0.001)

final_emb = extract_embeddings(final_model, X_scaled)
X_combined = np.hstack([X_scaled, final_emb])

selector_f = SelectKBest(f_regression, k=min(500, X_combined.shape[1]))
X_sel = selector_f.fit_transform(X_combined, y)

xgb_final = xgb.XGBRegressor(
    max_depth=12,
    learning_rate=0.02,
    n_estimators=500,
    reg_alpha=0.5,
    reg_lambda=2.5,
    subsample=0.8,
    colsample_bytree=0.8,
    verbosity=0,
    n_jobs=-1,
    random_state=42,
)
xgb_final.fit(X_sel, y)

# Save
model_data = {
    "encoder": final_model,
    "xgb_model": xgb_final,
    "scaler": scaler_f,
    "selector": selector_f,
    "embedding_dim": 64,
    "cv_r2": cv_r2,
    "cv_r": cv_r,
    "fold_scores": fold_scores,
    "n_samples": len(y),
    "date": pd.Timestamp.now().isoformat(),
}

output_path = work_dir / "geock_v2_neural_xgb.pkl"
with open(output_path, "wb") as f:
    pickle.dump(model_data, f)

print(f"\n{'=' * 60}")
print(f"RESULT: Neural Encoder + XGBoost")
print(f"  CV R² = {cv_r2:.4f}, R = {cv_r:.4f}")
print(f"  Saved: {output_path}")
print(f"{'=' * 60}")
