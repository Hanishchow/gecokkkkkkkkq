#!/usr/bin/env python3
"""
Extract Kuramoto-inspired physics features for GEOCK v2.
Based on cosmic-sync Kuramoto model concepts.
"""

import pickle
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
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
print("GEOCK v2 - KURAMOTO PHYSICS FEATURES")
print("=" * 60)

# Load merged data
with open(cache_dir / "merged_39k.pkl", "rb") as f:
    data = pickle.load(f)

X_ecfp = np.array([d["ecfp"] for d in data], dtype=np.float32)
y = np.array([d["affinity"] for d in data], dtype=np.float32)
print(f"\nLoaded: {len(data)} samples, {X_ecfp.shape[1]} ECFP features")

# ============================================================
# KURAMOTO-INSPIRED FEATURES
# Based on: r (order param), coupling, phase locking, sync time
# ============================================================
print("\n[1] Extracting Kuramoto-inspired features...")


def compute_kuramoto_features(ecfp, affinity):
    """
    Compute physics-inspired features based on Kuramoto model concepts.
    Each sample = oscillator system with ECFP bits as phase indicators.
    """
    n_samples = ecfp.shape[0]
    features = np.zeros((n_samples, 4), dtype=np.float32)

    for i in range(n_samples):
        # Treat ECFP bits as oscillator phases (0 or pi)
        bits = ecfp[i]
        n_oscillators = len(bits)

        # Order parameter r = |mean(e^(i*phase))|
        # phase = 0 if bit=0, pi if bit=1
        phases = bits * np.pi  # 0 or pi
        r = np.abs(np.mean(np.exp(1j * phases)))  # Order parameter
        features[i, 0] = r  # Synchronization level [0,1]

        # Coupling strength: fraction of "active" bits (bit=1)
        coupling = np.sum(bits) / n_oscillators
        features[i, 1] = coupling

        # Phase locking: fraction of bits in "locked" state
        # (bits that agree with majority)
        majority = 1 if np.sum(bits) > n_oscillators / 2 else 0
        locked = np.sum(bits == majority) / n_oscillators
        features[i, 2] = locked

        # Synchronization speed: how fast system syncs (1/(1 + entropy))
        # Using bit distribution entropy
        p0 = np.sum(bits == 0) / n_oscillators
        p1 = 1 - p0
        if p0 > 0 and p1 > 0:
            entropy = -p0 * np.log(p0) - p1 * np.log(p1)
        else:
            entropy = 0
        features[i, 3] = 1.0 / (1.0 + entropy)  # Sync speed

    return features


# Compute features
print("  Computing Kuramoto features for 39K samples...")
X_kuramoto = compute_kuramoto_features(X_ecfp, y)
print(f"  Features shape: {X_kuramoto.shape}")
print(
    f"  Feature means: r={X_kuramoto[:, 0].mean():.3f}, K={X_kuramoto[:, 1].mean():.3f}, lock={X_kuramoto[:, 2].mean():.3f}, speed={X_kuramoto[:, 3].mean():.3f}"
)

# ============================================================
# COMBINE: ECFP + Kuramoto Features
# ============================================================
print("\n[2] Combining ECFP + Kuramoto features...")
X_combined = np.hstack([X_ecfp, X_kuramoto])
print(f"  Combined shape: {X_combined.shape} (512 ECFP + 4 Kuramoto)")

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_combined)

# ============================================================
# 5-FOLD CV WITH XGBOOST
# ============================================================
print("\n[3] 5-Fold CV with XGBoost (ECFP + Kuramoto)...")

kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_scores = []

for fold, (tr_idx, vl_idx) in enumerate(kf.split(X_scaled)):
    print(f"\n{'=' * 60}")
    print(f"FOLD {fold + 1}/5")
    print(f"{'=' * 60}")

    # Feature selection INSIDE fold
    selector = SelectKBest(f_regression, k=400)
    X_tr = selector.fit_transform(X_scaled[tr_idx], y[tr_idx])
    X_vl = selector.transform(X_scaled[vl_idx])

    # XGBoost
    params = {
        "objective": "reg:squarederror",
        "max_depth": 12,
        "n_estimators": 500,
        "learning_rate": 0.01,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "gamma": 0.1,
        "random_state": 42,
    }

    model = xgb.XGBRegressor(**params)
    model.fit(X_tr, y[tr_idx], eval_set=[(X_vl, y[vl_idx])], verbose=False)

    preds = model.predict(X_vl)
    r, _ = pearsonr(y[vl_idx], preds)
    r2 = r**2
    fold_scores.append(r2)
    print(f"Fold {fold + 1} R² = {r2:.4f}, R = {r:.4f}")

cv_r2 = np.mean(fold_scores)
cv_r = np.sqrt(cv_r2)

print(f"\n{'=' * 60}")
print(f"KURAMOTO + ECFP RESULTS:")
print(f"{'=' * 60}")
print(f"  CV R² = {cv_r2:.4f} ± {np.std(fold_scores):.4f}")
print(f"  CV R  = {cv_r:.4f}")
print(f"  Fold scores: {[f'{s:.4f}' for s in fold_scores]}")
print(f"\n  vs ECFP only (39K): R² = 0.7169")
print(f"  vs Original (39K):   R² = 0.7118")
print(f"  Difference: {cv_r2 - 0.7118:+.4f}")
print(f"{'=' * 60}")

# ============================================================
# TRAIN FINAL MODEL ON ALL DATA
# ============================================================
print("\n[4] Training FINAL model (ECFP + Kuramoto) on ALL 39K...")

selector = SelectKBest(f_regression, k=400)
X_final = selector.fit_transform(X_scaled, y)

final_model = xgb.XGBRegressor(
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
final_model.fit(X_final, y, verbose=False)

# Save
model_data = {
    "model": final_model,
    "scaler": scaler,
    "selector": selector,
    "cv_r2": cv_r2,
    "cv_r": cv_r,
    "fold_scores": fold_scores,
    "n_samples": len(y),
    "features": [
        "ECFP_512",
        "Kuramoto_r",
        "Kuramoto_K",
        "Kuramoto_lock",
        "Kuramoto_speed",
    ],
    "date": __import__("pandas").Timestamp.now().isoformat(),
}

output_path = work_dir / "geock_v2_kuramoto.pkl"
with open(output_path, "wb") as f:
    pickle.dump(model_data, f)

print(f"Saved: {output_path}")
print(f"Final CV R² = {cv_r2:.4f}")
print(f"{'=' * 60}")
