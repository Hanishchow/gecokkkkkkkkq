# GEOCK 2.0 Results

## Method

**Binding Affinity Prediction using Physics Features + ECFP4 + ElasticNet**

GEOCK 2.0 combines traditional physics-based scoring with molecular fingerprints and regularized regression to predict protein-ligand binding affinity (ΔG).

## Dataset

| Split | Compounds |
|-------|----------|
| Training | 30 (PDBbind subset) |
| Held-out | 20 |
| Total | 50 |

## Performance

| Metric | Value | Target |
|--------|-------|--------|
| **Pearson R** | **0.644** | > 0.5 |
| **MAE** | **0.637 kcal/mol** | < 2.0 |

### Held-out Validation Results (Compounds 31-50)

| PDB ID | Experimental ΔG | Predicted ΔG | Error |
|--------|----------------|--------------|-------|
| 1cpf | -8.60 | -7.90 | 0.70 |
| 1cpg | -8.20 | -7.76 | 0.44 |
| 1cpi | -9.70 | -11.65 | 1.95 |
| 1cpo | -8.40 | -7.35 | 1.05 |
| 1cpq | -8.60 | -8.85 | 0.25 |
| 1cpr | -9.50 | -9.46 | 0.04 |
| 1cps | -8.30 | -8.65 | 0.35 |
| 1cpt | -7.80 | -8.09 | 0.29 |
| 1cpu | -8.90 | -9.00 | 0.10 |
| 1cpw | -8.20 | -7.91 | 0.29 |
| 1cqb | -8.50 | -7.23 | 1.27 |
| 1cqd | -8.80 | -8.64 | 0.16 |
| 1cqe | -8.40 | -7.50 | 0.90 |
| 1cqf | -7.80 | -9.22 | 1.42 |
| 1cqi | -8.40 | -7.96 | 0.44 |
| 1cqj | -8.10 | -7.37 | 0.73 |
| 1cqp | -8.60 | -7.82 | 0.78 |
| 1cqq | -8.40 | -8.86 | 0.46 |
| 1cqs | -8.70 | -7.99 | 0.71 |
| 1cqx | -8.10 | -7.69 | 0.41 |

## Comparison to Baselines

| Method | Pearson R |
|--------|----------|
| **GEOCK 2.0** | **0.644** |
| AutoDock Vina | 0.56 |
| Physics-only | 0.03 |

GEOCK 2.0 outperforms AutoDock Vina by **0.084** in Pearson correlation.

## Physics Calibration (March 2026 Update)

The physics-only fallback scoring was calibrated on 46 clean PDBbind compounds (3 outliers excluded: 3phy, 1axo, 1cpw).

### Clash Handling Fix
- Changed clash threshold from `d < 0` to `d < -0.4`
- Soft contacts (-0.4 ≤ d < 0) now counted but not penalized
- Hard clashes (d < -0.4) still penalized

### pKd Calibration Formula
```
dG = -0.0168 × raw_vina - 8.6252
pKd = -dG / 1.364
```

### Validation on 1a1e
| Metric | Before | After | TRUE |
|--------|--------|-------|------|
| pKd | 3.41 | **6.28** | 6.09 |
| ΔG (kcal/mol) | -4.65 | **-8.56** | -8.30 |
| Error | 2.68 | **0.19** | - |

**Note**: This calibration only affects `score_single()` (physics-only display). The ML model (`predict_affinity_ml()`) is unaffected.

## Feature Engineering

### Physics Features (60 dimensions)

- Distance-based terms (ligand-pocket centroid, min/max/mean distances)
- Vina-style Gaussian terms
- Repulsion score for atomic clashes
- Contact fractions at various distance cutoffs (2-8 Å)
- Ligand composition (hydrophobic, H-bond donors/acceptors, aromatic)
- Pocket composition (carbon, polar atoms)
- Interaction scoring (van der Waals, hydrophobic, H-bond)
- Desolvation approximation
- Electrostatic features
- Distance histograms and percentiles

### Molecular Fingerprints (512 dimensions)

- ECFP4 (Extended-Connectivity Fingerprints, radius=2)
- Encodes molecular substructures relevant to binding

**Total features: 572 dimensions**

## Model Architecture

```
ElasticNet Regression
├── alpha = 0.001 (L2 regularization)
├── l1_ratio = 0.5 (50% L1 sparsity)
├── max_iter = 5000
└── Feature scaling: StandardScaler
```

The L1 component promotes sparsity, selecting relevant features from the 572-dimensional space.

## Key Innovations

1. **Fixed Coordinate Parsing**: Uses HETATM records from PDB for correct ligand coordinates (not SDF files)
2. **Hybrid Features**: Combines physics-based and fingerprint-based representations
3. **Sparse Regression**: ElasticNet balances feature selection with prediction accuracy
4. **No Neural Networks**: Simpler model works better than deep learning on small datasets

## Files

- `affinity_model.pkl` - Trained model and scaler
- `score_compound.py` - Prediction API
- `patch_parse.py` - Coordinate extraction

## Usage

```python
from score_compound import predict_affinity_ml

# Predict binding affinity
pred_dG = predict_affinity_ml('pocket.pdb', smiles='CCO')
# Returns: predicted ΔG in kcal/mol
```

---
**Date**: March 21, 2026  
**Physics Calibration**: v2.1 (March 2026)  
**Target**: Pearson R > 0.5, MAE < 2.0 kcal/mol  
**Status**: ✅ ACHIEVED
