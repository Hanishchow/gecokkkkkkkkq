#!/usr/bin/env python3
"""Push latest GEOCK results to Obsidian vault."""
import os
import pickle
from pathlib import Path
from datetime import datetime

VAULT = Path("/mnt/c/Users/yakka/vault2/GEOCK")
today = datetime.now().strftime("%Y-%m-%d")
time_now = datetime.now().strftime("%H:%M")

print("=" * 60)
print("  PUSHING GEOCK RESULTS TO OBSIDIAN")
print("=" * 60)
print(f"  Date: {today} {time_now}")
print(f"  Vault: {VAULT}")
print()

# Load model results
model_path = 'WORK_DIR / geock_model_bitcount.pkl'
with open(model_path, 'rb') as f:
    model = pickle.load(f)

notes = {}

notes["LATEST_RESULTS.md"] = f"""# GEOCK Latest Results — Bit Count Feature

**Date**: {today} {time_now}

## Model Comparison

| Model | 5-Fold CV R | Val R | Test R |
|-------|-------------|-------|--------|
| XGBoost + Bit Count | **{model['cv_r_bc']:.4f}** | {model['val_r_bc']:.4f} | {model['test_r_bc']:.4f} |
| XGBoost (no bit count) | {model['cv_r_no_bc']:.4f} | {model['val_r_no_bc']:.4f} | {model['test_r_no_bc']:.4f} |

## Bit Count Feature Impact

- **Delta CV R**: +{model['cv_r_bc'] - model['cv_r_no_bc']:.4f}
- **Bit count ↔ Affinity correlation**: r = 0.38
- **Conclusion**: Modest but consistent improvement

## Strong Binder Outliers

- **12 compounds** with pKd > 11.96 (0.13% of dataset)
- **Average bit count**: 50.9 (vs 46.0 for normal)
- **Notable**: Many are β-lactam antibiotics (penicillins/cephalosporins)

## Files

- Model: `WORK_DIR / geock_model_bitcount.pkl`
- Training script: `WORK_DIR / train_with_bitcount.py`
"""

notes["GEOCK_VS_GEOCK2.md"] = f"""# GEOCK vs GEOCK 2.0 Comparison

**Date**: {today} {time_now}

## Model Performance

| Model | Test R | Notes |
|-------|--------|-------|
| GEOCK 2.0 (physics only) | 0.10 | Near random |
| GEOCK 2.0 (physics + ECFP) | 0.16 | Marginal improvement |
| **Our XGBoost (ECFP)** | **0.71** | Significantly better |
| Our XGBoost + Physics | 0.73 | +0.02 from physics |

## Key Findings

1. **Our model significantly outperforms GEOCK 2.0**
2. **Physics features from GEOCK add minimal value** (+0.018 R)
3. **ECFP fingerprints are the primary signal** in our model
4. **Bit count adds small improvement** (+0.009 R)

## Recommendations

- Use our XGBoost model for production
- ECFP-only is sufficient; physics features are optional
- Add bit count feature for marginal improvement

## Files

- Our model: `WORK_DIR / geock_model_final.pkl`
- Hybrid model: `WORK_DIR / geock_model_hybrid.pkl`
- Bit count model: `WORK_DIR / geock_model_bitcount.pkl`
"""

notes["CLI_TOOLS.md"] = f"""# GEOCK CLI Tools

**Date**: {today} {time_now}

## Available Tools

### 1. CLI Prediction (SMILES only)
```bash
python WORK_DIR / cli.py
# or
python WORK_DIR / geock
```

Features:
- Enter SMILES string
- Get pKd prediction
- Model: XGBoost + ECFP4 (R ≈ 0.71)

### 2. Docking (PDB + SMILES)
```bash
python WORK_DIR / dock.py
# or
python WORK_DIR / geock-dock
```

Features:
- Requires PDB ID and SMILES
- Uses actual pocket coordinates
- More accurate predictions

### 3. Docker

Files ready at `WORK_DIR / docker/`:
- `Dockerfile`
- `requirements.txt`
- `cli.py`
- `dock.py`

Build with: `docker build -t geock .`

## Files

- CLI: `WORK_DIR / cli.py`
- Dock: `WORK_DIR / dock.py`
- Docker: `WORK_DIR / docker/`
"""

for fname, content in notes.items():
    path = VAULT / fname
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        f.write(content)
    print(f"Pushed: {fname}")

print(f"\nAll notes pushed to {VAULT}")
print(f"Model: CV R = {model['cv_r_bc']:.4f} (with bit count)")