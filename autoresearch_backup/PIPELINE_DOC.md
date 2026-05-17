# GEOCK Pipeline Documentation

## Overview

The GEOCK data acquisition and training pipeline automates:
1. **Data Acquisition** - Fetch PDB files and training data from multiple sources
2. **Feature Extraction** - Generate molecular fingerprints and features
3. **Model Training** - Train and evaluate binding affinity prediction models
4. **Deployment** - Update production model for predictions

## Pipeline Components

### 1. Data Acquisition (`pipeline_acquire.py`)

**Sources:**
- **LP-PDBBind** (GitHub: THGLab/LP-PDBBind) - 19,443 binding records
- **ChEMBL** (via API) - Drug-like compounds with activity data
- **PDBBind Core** (manual download) - 285 high-quality complexes

**Steps:**
```bash
# Download PDB files from LP-PDBBind
python pipeline_acquire.py --step fetch_pdb --limit 1000

# Extract features from downloaded PDBs
python pipeline_acquire.py --step extract_features

# Fetch additional data from ChEMBL
python pipeline_acquire.py --step fetch_chembl

# Combine all data sources
python pipeline_acquire.py --step combine_data

# Run all steps
python pipeline_acquire.py --step all
```

### 2. Model Training (`pipeline_train.py`)

**Models:**
- **Ridge Regression** - Baseline, interpretable
- **XGBoost** - Gradient boosting with regularization
- **Ensemble** - XGBoost + Ridge blend

**Usage:**
```bash
# Train Ridge model
python pipeline_train.py --model ridge

# Train XGBoost
python pipeline_train.py --model xgboost

# Train ensemble
python pipeline_train.py --model ensemble

# Compare all models
python pipeline_train.py --model compare
```

### 3. Pipeline Orchestrator (`pipeline_main.py`)

**Usage:**
```bash
# Check pipeline status
python pipeline_main.py --mode status

# Run data acquisition only
python pipeline_main.py --mode acquire

# Run training only
python pipeline_main.py --mode train

# Run full pipeline
python pipeline_main.py --mode full

# Evaluate current models
python pipeline_main.py --mode evaluate
```

## Current Status (Updated: 2026-04-05)

| Metric | Value |
|--------|-------|
| LP-PDBBind Records | 19,443 |
| PDB Files Downloaded | **19,436** ✅ |
| PDB Files Missing | 7 |
| Training Records | **39,507** |
| Best Model CV R | **0.8432** ✅ |

## Results Summary

### Model Performance

| Model | CV R | MAE | Notes |
|-------|------|-----|-------|
| **Deep Trees (final)** | **0.8432 ± 0.0027** | 0.75 | Best model |
| Original Ensemble | 0.7049 | 1.04 | Previous production |
| Baseline (ECFP only) | 0.668 | 1.09 | Starting point |

### Key Finding: Deep Trees Architecture

The breakthrough was using **deeper trees (max_depth=10)** instead of shallow trees (max_depth=6):

```python
model = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=10,           # Increased from 6
    learning_rate=0.05,
    reg_alpha=0.5,         # Reduced regularization
    reg_lambda=2.0,        # Reduced regularization
    subsample=0.8,
    colsample_bytree=0.8
)
```

### Improvement Breakdown

- **Baseline**: R = 0.668
- **After tuning**: R = 0.8432
- **Improvement**: +0.1752 (+26% relative improvement)
- **Target**: R = 0.76-0.80
- **Status**: ✓ **TARGET EXCEEDED**

## Data Locations

| Data | Path |
|------|------|
| Cache Directory | `/home/chow/.cache/geock_autoresearch/` |
| PDB Files | `lp_pdb_files/` |
| Training Data | `lp_new_features_8k.pkl` (24K), `geock_training_data.pkl` (15K) |
| Enhanced Features | `lp_features_enhanced.pkl` (19K × 982 features) |

## Model Files

| Model | Path | Performance |
|-------|------|-------------|
| **Final Production** | `geock_deep_trees_final.pkl` | CV R=0.8432 |
| Deep Trees | `geock_deep_trees.pkl` | Test R=0.8302 |
| Original Ensemble | `geock_ensemble.pkl` | R=0.7049 |

## Scripts Created

| Script | Purpose |
|--------|---------|
| `extract_features_v2.py` | Enhanced features (MACCS + RDKit + FCFP + ECFP) |
| `extract_physics_features.py` | Physics features from PDB files |
| `train_final_model.py` | Train final deep trees model |
| `improve_model.py` | Test different model architectures |

## Next Steps

1. ~~**Train Better Model**~~ - **DONE: R=0.8432**
2. **Validate on CASF-2016** - Standard benchmark for binding prediction
3. **Deploy to Production** - Update geock_engine.py with new model
