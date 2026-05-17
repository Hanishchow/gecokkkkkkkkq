# GEOCK Binding Affinity Prediction Engine - Research Summary

## Executive Summary
We developed a binding affinity (pKd) prediction engine using machine learning on protein-ligand complexes. After extensive experimentation with different algorithms, features, and data sources, we achieved an honest cross-validated correlation coefficient of R = 0.52 on truly held-out compounds.

## 1. Data Sources

### 1.1 Primary Dataset: LP-PDBBind
- **Source**: GitHub (THGLab/LP-PDBBind)
- **Total complexes**: 19,443 protein-ligand structures
- **Fields**: SMILES, protein sequences, binding affinities (Kd/Ki), PDB IDs

### 1.2 Downloaded PDB Files
- **Location**: `/home/chow/.cache/geock_autoresearch/lp_pdb_files/`
- **Count**: 7,393 PDB files downloaded from RCSB PDB

### 1.3 ChEMBL Data
- **Source**: ChEMBL API (https://www.ebi.ac.uk/chembl/api/data/activity.json)
- **Compounds fetched**: 2,768 IC50/Ki/Kd measurements
- **Note**: Adding ChEMBL data did not improve performance due to distribution shift

### 1.4 Final Training Data
- **LP-PDBBind features**: 7,465 unique compounds with ECFP fingerprints
- **Augmented data**: 7,384 samples (with SMILES augmentation)
- **Feature dimensionality**: 512-bit ECFP4 fingerprints

## 2. Feature Engineering

### 2.1 ECFP Fingerprints
- **Type**: Morgan fingerprints (ECFP4)
- **Radius**: 2
- **Bit vector size**: 512 bits
- **Library**: RDKit

### 2.2 Feature Selection
- **Method**: SelectKBest with f_regression
- **Selected features**: 400 bits (optimized from 100, 200, 300, 400, 512)
- **Performance improved**: Yes - 400 features optimal

### 2.3 Alternative Features Tested (Rejected)
- Physics-based features (molecular weight, logP, H-bond donors/acceptors, etc.)
- **Result**: LOO-R = 0.06 - essentially noise
- **Conclusion**: ECFP-only is the best approach

## 3. Models Tested

### 3.1 Linear Models

| Model | Parameters | CV-R (Original) | CV-R (Augmented) |
|-------|------------|-----------------|------------------|
| Ridge (α=10) | ke=100 | 0.50 | - |
| Ridge (α=100) | ke=300 | 0.61 | 0.62 |
| Ridge (α=100) | ke=400 | 0.64 | 0.62 |
| Ridge (α=500) | ke=300 | 0.61 | 0.64 |
| Ridge (α=125) | ke=512 | 0.64 | 0.62 |
| ElasticNet | various | < 0.60 | - |
| Lasso | various | < 0.60 | - |

### 3.2 Tree-Based Models

| Model | Parameters | 5-Fold CV | Test on New |
|-------|------------|------------|-------------|
| RandomForest | 100 trees, depth=15 | 0.88* | 0.49 |
| GradientBoosting | 100 trees, depth=5 | - | 0.51 |
| XGBoost | 100 trees, depth=5 | - | **0.52** |

*Inflated due to augmentation leakage

### 3.3 Other Models Tested (Rejected)
- SVR: Poor performance
- GaussianProcess: Too slow, OOM
- BayesianRidge: Similar to Ridge
- Kernel Ridge (RBF): Poor with high-dimensional ECFP

## 4. Hyperparameter Tuning

### 4.1 Ridge Regression
- **Best**: ke=400, α=100
- **LOO-R**: 0.6388 (on original data)
- **Honest CV-R**: ~0.47 (on truly new compounds)

### 4.2 XGBoost
- **n_estimators**: 100
- **max_depth**: 5
- **learning_rate**: 0.1
- **Best CV-R**: 0.5238 (honest evaluation)

## 5. Key Findings

### 5.1 Data Quality
- LP-PDBBind high-quality, curated data
- ChEMBL data introduced distribution shift, hurt performance
- SMILES augmentation helped but caused CV inflation

### 5.2 Feature Importance
- ECFP alone >> ECFP + physics features
- 400 features optimal (vs 100, 200, 300, 512)
- Feature selection critical for linear models

### 5.3 Model Selection
- XGBoost best for generalization
- RF had highest CV but inflated (augmentation leakage)
- Ridge most interpretable, stable

### 5.4 Evaluation Methodology
- Always test on truly held-out compounds
- Augmentation causes CV inflation
- GroupKFold prevents leakage from augmented samples

## 6. Final Engine

### 6.1 Model Configuration
- **Algorithm**: XGBoost Regressor
- **Features**: 400-bit ECFP4, selected via f_regression
- **Training**: 7,384 augmented samples

### 6.2 Performance
- **Honest CV-R**: 0.5238 (on 1,000 truly new compounds)
- **Confidence**: Low (R < 0.6)

### 6.3 Usage
```bash
# CLI
python geock_engine.py --smiles "CCO"

# Python
from geock_engine import predict_pKd
result = predict_pKd("CCO")
print(f"pKd = {result['pKd']:.2f}")
```

## 7. Files Produced

### Models
- `geock_model_xgb.pkl` - Final production model (XGBoost)
- `geock_model_augmented.pkl` - Ridge model (LOO-R=0.64)
- `geock_model_rf.pkl` - RandomForest (inflated)

### Data
- `lp_all_features.pkl` - 7,384 augmented samples
- `lp_new_features_all.pkl` - 3,990 original
- `lp_new_features_8k.pkl` - 7,465 unique
- `chembl_v2.pkl` - 2,768 ChEMBL compounds

### Code
- `geock_engine.py` - Production CLI + Python module

## 8. Lessons Learned

1. **More data beats better physics** - Simple ECFP >> complex physics features
2. **Honest evaluation critical** - Augmentation inflates CV by 30%+
3. **XGBoost generalizes best** - Better than RF, Ridge on new compounds
4. **Feature selection matters** - 400 bits optimal for this dataset
5. **Distribution shift is real** - ChEMBL hurt performance

## 9. Recommendations for Research Paper

### Highlights to Emphasize
1. ECFP-only approach achieved competitive performance
2. Rigorous honest evaluation revealed true generalization
3. XGBoost outperformed linear models on new compounds
4. Data augmentation provides CV inflation warning

### Limitations
1. Ligand-only features (no protein context)
2. 7,384 training samples relatively small
3. Binding site information not encoded

### Future Work
1. Add protein sequence embeddings (ESM, ProtTrans)
2. Incorporate 3D structure features
3. Expand training data to 50k+ compounds
4. Try graph neural networks (GNN)
