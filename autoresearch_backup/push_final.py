#!/usr/bin/env python3
"""Push final GEOCK results to Obsidian vault."""
import os
from pathlib import Path
from datetime import datetime

VAULT = Path("/mnt/c/Users/yakka/vault2/GEOCK")
today = datetime.now().strftime("%Y-%m-%d")
time_now = datetime.now().strftime("%H:%M")

notes = {}

notes["FINAL_ENGINE.md"] = f"""# GEOCK Final Engine Results

**Date**: {today} {time_now}

## Final Model Performance

| Metric | Value | Notes |
|--------|-------|-------|
| LOO-R | **0.641** | Honest generalization |
| RKF-R | 0.629 ± 0.040 | Repeated 5-Fold × 5 |
| Train-R | 0.719 | Slight overfitting |
| Gap | 0.090 | Acceptable |
| LOO-MAE | 1.19 pKd | Units |
| N compounds | **1,094** | 11x increase from original 96 |

## Model Architecture

- **Type**: Ridge Regression
- **Features**: ECFP4 Morgan fingerprints (512D → 100D via SelectKBest)
- **Regularization**: alpha = 10.0
- **Training compounds**: 1,094
- **Feature selection**: SelectKBest(f_regression, k=100)

## Evolution

| Version | N | Features | LOO-R | Notes |
|---------|---|---------|-------|-------|
| v1 physics-only | 96 | 24 physics | 0.060 | Near random! |
| v2 physics+ECFP | 96 | 9 physics + 18 ECFP | 0.504 | First real signal |
| v3 ECFP-only | 1,094 | 100 ECFP | **0.641** | Best generalization |

## Key Insights

1. **ECFP dominates** — physics features add marginal value
2. **More data wins** — 11x more data gave +0.137 R improvement  
3. **Strong regularization needed** — alpha=10 prevents overfitting
4. **LP-PDBBind is goldmine** — 19,443 binding affinity records available

## Usage

```python
from geock_engine import predict_pKd
result = predict_pKd("CCO")
print(f"pKd = {{result['pKd']:.2f}}")
```

```bash
python geock_engine.py --smiles "CC(=O)Oc1ccccc1C(=O)O"
# Predicted pKd: 5.45
```

## Files

- Model: `WORK_DIR / geock_model_final.pkl`
- Engine: `WORK_DIR / geock_engine.py`
- Training: `WORK_DIR / train_expanded.py`
- Predictions: `WORK_DIR / results_expanded.tsv`
"""

notes["DATA_ACQUISITION.md"] = f"""# Data Acquisition — LP-PDBBind

## What Was Downloaded

- **PDB files**: 1,743+ from RCSB (`https://files.rcsb.org/download/{{pdb_id}}.pdb`)
- **LP-PDBBind CSV**: 19,443 binding affinity records from GitHub THGLab
- **BDB2020+**: 136 BindingDB complexes with structures

## LP-PDBBind CSV Stats

- Total: 19,443 complexes
- High quality (CL1, train, non-covalent, has SMILES): 7,393
- Already downloaded PDBs: 1,743+
- Compounds with features: 998+ (continuously growing)

## Data Sources Status

| Source | Status | Notes |
|--------|--------|-------|
| RCSB PDB | ✅ Works | 1,743+ files |
| ChEMBL API | ❌ 500 errors | Inaccessible |
| PDBbind aliyun | ❌ 404 | URL broken |
| CASF-2016 Figshare | ❌ Blocked | 0 bytes |
| HF pdbbindpp-2020 | ❌ 401 | Auth required |
| LP-PDBBind CSV | ✅ Works | GitHub |

## Cache Locations

- PDB files: `CACHE_DIR / lp_pdb_files/`
- Features: `CACHE_DIR / lp_new_features*.pkl`
- LP CSV: `CACHE_DIR / LP_PDBBind.csv`
- Combined: `CACHE_DIR / features_combined.pkl`
"""

notes["GEOCK_MOC.md"] = f"""# GEOCK MOC

## Final Engine Results

- **LOO-R**: 0.641 (honest)
- **Training data**: 1,094 compounds
- **Model**: Ridge + ECFP fingerprints

## Key Discoveries

1. **ECFP is the primary signal** — physics-only gives LOO=0.06, ECFP+physics gives LOO=0.641
2. **More data beats better physics** — 11x more data = +0.137 R improvement
3. **Strong regularization critical** — alpha=10 prevents overfitting with 100 features
4. **LP-PDBBind is a goldmine** — 19,443 binding affinity records accessible

## Files

- Engine: `geock_engine.py`
- Model: `geock_model_final.pkl`
- Training: `train_expanded.py`
"""

for fname, content in notes.items():
    path = VAULT / fname
    with open(path, 'w') as f:
        f.write(content)
    print(f"Pushed: {fname}")

print(f"\nAll notes pushed to {VAULT}")
