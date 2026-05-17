# GEOCK v2 - Protein-Ligand Binding Affinity Prediction

**Best Model CV R² = 0.774 (with 50K dataset)**

Machine learning system for predicting protein-ligand binding affinity (pKd) from molecular structure using XGBoost and Neural Networks.

---

## Model Performance

| Model | Dataset | CV R² | Notes |
|-------|---------|-------|-------|
| Neural Network (v2) | merged_50k.pkl | **0.7739** | 3-fold CV |
| Neural Network (v2) | merged_50k.pkl | **0.7788** | Best fold |
| Neural Network (v2) | merged_39k.pkl | 0.7332 | Previous best |
| XGBoost | merged_50k.pkl | 0.5408 | |

### Cross-Validation Results (50K Dataset)
- Fold 1: R² = 0.7672
- Fold 2: R² = 0.7788
- Fold 3: R² = 0.7756
- **3-Fold CV Average: R² = 0.7739**

### GSTACK Analysis Summary
- **Improvement: +4.1%** over 39K dataset
- No overfitting (train-val gap ~0.2)
- No underfitting (R² > 0.7)
- Model generalizes well

---

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Test installation
python -m pytest tests/test_geock.py -v
```

---

## Quick Start

### Predict Binding Affinity

```python
from geock_engine import predict_pKd

# Single prediction
result = predict_pKd("CCO")  # ethanol
print(f"pKd: {result.get('pKd')}")

# Using engine class
from geock_engine import GEOCKEngine
engine = GEOCKEngine()
result = engine.predict("CC(=O)Oc1ccccc1C(=O)O")  # aspirin
print(f"Aspirin pKd: {result['pKd']:.2f}")
```

### Train Models

```bash
# Neural network (recommended)
python train_neural_v2.py

# XGBoost (fast)
python train_xgboost_39k.py

# Quick test
python train_v2_quick.py
```

---

## Project Structure

```
geock/
├── geock_engine.py        # Prediction engine
├── geock_paths.py         # Cross-platform paths
├── train_neural_v2.py    # Neural network training
├── train_xgboost_39k.py # XGBoost training
├── tests/
│   └── test_geock.py    # Test suite (12 tests)
├── docs/                # Documentation
└── cache/               # Data (~39K samples)
```

---

## Data

| Dataset | Samples | Location |
|---------|---------|----------|
| merged_39k.pkl | 39,109 | Auto-detect |
| LP-PDBBind.csv | 19,444 | Auto-detect |
| chembl_binding.csv | ~3,000 | Auto-detect |

---

## Cross-Platform Support

- **Linux**: Auto-detects paths
- **Windows**: OneDrive paths auto-detected
- Works on both systems without modifications

---

## Testing

```bash
# Run test suite
python -m pytest tests/test_geock.py -v
```

12 tests covering:
- Import verification
- SMILES parsing
- Batch predictions
- Path handling
- Result formats

---

## Requirements

- Python 3.9+
- RDKit (molecular fingerprints)
- scikit-learn
- XGBoost
- PyTorch
- NumPy, Pandas, SciPy

See `requirements.txt` for full list.

---

## Citation

```bibtex
@software{geock2026,
  title = {GEOCK: Protein-Ligand Binding Affinity Prediction},
  author = {GEOCK Team},
  year = {2026}
}
```

---

*Last Updated: May 2026*