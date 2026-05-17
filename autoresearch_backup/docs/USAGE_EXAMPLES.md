# GEOCK Usage Examples

## Quick Start

### Predict binding affinity for a molecule

```python
from geock_engine import predict_pKd

# Simple prediction
result = predict_pKd("CCO")  # ethanol
print(f"pKd: {result.get('pKd')}")
```

### Using the engine class

```python
from geock_engine import GEOCKEngine

# Load model
engine = GEOCKEngine()

# Single prediction
result = engine.predict("CC(=O)Oc1ccccc1C(=O)O")  # aspirin
print(f"Predicted pKd: {result['pKd']:.2f}")

# Batch prediction
smiles_list = ["CCO", "CC(=O)O", "c1ccccc1"]
results = engine.predict_batch(smiles_list)
for smiles, result in zip(smiles_list, results):
    print(f"{smiles}: pKd = {result.get('pKd', 'N/A')}")
```

---

## Training

### Neural Network (Recommended)

```python
# Train with default settings
python train_neural_v2.py

# Train with custom epochs
python train_neural_v2.py --epochs 150
```

### XGBoost

```python
python train_xgboost_39k.py
```

### Quick test

```python
python train_v2_quick.py
```

---

## Data

### Check data location

```python
from geock_paths import CACHE_DIR, WORK_DIR

print(f"Cache: {CACHE_DIR}")
print(f"Work:  {WORK_DIR}")
```

### Available datasets

| Dataset | Samples | Location |
|---------|---------|----------|
| merged_39k.pkl | 39,109 | CACHE_DIR |
| LP-PDBBind.csv | 19,444 | CACHE_DIR |

### Load custom data

```python
import pickle
from geock_paths import get_cache_dir

cache = get_cache_dir()
with open(cache / "merged_39k.pkl", "rb") as f:
    data = pickle.load(f)

print(f"Loaded {len(data)} samples")
```

---

## Testing

### Run test suite

```bash
python -m pytest tests/test_geock.py -v
```

### Run specific test

```bash
python -m pytest tests/test_geock.py::test_valid_smiles_aspirin -v
```

---

## Troubleshooting

### Model not found

```
Error: Model not found
```

**Fix:** Train a model first:
```python
python train_neural_v2.py
```

### Path issues

```
Error: FileNotFoundError
```

**Fix:** Check paths work:
```python
python -c "from geock_paths import CACHE_DIR, WORK_DIR; print(CACHE_DIR, WORK_DIR)"
```

### Import errors

**Fix:** Install dependencies:
```bash
pip install -r requirements.txt
```

---

*Generated: May 2026*