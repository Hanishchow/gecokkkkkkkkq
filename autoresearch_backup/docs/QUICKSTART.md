# GEOCK Quick Start Guide

## For Windows User

Copy all modified files to your Windows machine (OneDrive/Desktop/lki).

---

## Quick Test (5 min)

```powershell
# Test imports
python -c "from geock_paths import get_cache_dir, get_work_dir; print(get_cache_dir()); print(get_work_dir())"

# Run test suite
python -m pytest tests/test_geock.py -v
```

---

## Train Model (1-2 hours)

```powershell
# Neural network training (recommended)
python train_neural_v2.py

# XGBoost training
python train_xgboost_39k.py

# Quick version (30 min)
python train_v2_quick.py
```

---

## Expected Results

| Model | Data | Expected R² |
|-------|------|-------------|
| Neural Network | 39K | 0.73-0.78 |
| XGBoost | 39K | 0.70-0.75 |
| Quick | 8K | 0.50-0.60 |

---

## Files Fixed for Cross-Platform

- `geock_paths.py` - Path helper (auto-detects Linux/Windows)
- `geock_engine.py` - Main prediction engine
- `train_neural_v2.py` - Neural network training
- `train_v2_quick.py` - Quick training
- `train_v2_full.py` - Full training
- `train_xgboost_39k.py` - XGBoost training
- `train_enhanced_v2.py` - Enhanced model
- `pipeline_train.py` - Training pipeline
- `check_overfitting.py` - Overfitting analysis
- `train_kuramoto.py` - Kuramoto features
- `train_hybrid_ensemble.py` - Hybrid ensemble
- `tests/test_geock.py` - Test suite

---

## Data Location

| OS | Cache Directory |
|-----|-------------|
| Linux | `/home/chow/.cache/geock_autoresearch/` |
| Windows | `C:\Users\yakka\OneDrive\.cache\geock_autoresearch\` |
| Windows Alt | `C:\Users\yakka\OneDrive\Desktop\lki\.cache\geock_autoresearch\` |

---

## Key Scripts

```powershell
# 1. Predict binding affinity
python geock_engine.py --smiles "CCO" --affinity 5.5

# 2. Batch prediction  
python -c "from geock_engine import GEOCK; g = GEOCK(); print(g.predict_batch(['CCO', 'c1ccccc1']))"

# 3. Train neural network
python train_neural_v2.py --epochs 150

# 4. Train XGBoost
python train_xgboost_39k.py
```

---

## Troubleshooting

**Path not found?**
```powershell
# Check paths work
python -c "from geock_paths import CACHE_DIR, WORK_DIR; print(CACHE_DIR, WORK_DIR)"
```

**Import errors?**
```powershell
# Install dependencies
pip install -r requirements.txt
```

**Data not found?**
```powershell
# Check cache directory
dir C:\Users\yakka\OneDrive\.cache\geock_autoresearch\
```

---

*Generated: May 2026*