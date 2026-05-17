#!/usr/bin/env python3
"""Push final GEOCK results to Obsidian vault."""
import os, shutil, pickle
from pathlib import Path
from datetime import datetime

VAULT = Path("/mnt/c/Users/yakka/vault2/GEOCK")
today = datetime.now().strftime("%Y-%m-%d")
time_now = datetime.now().strftime("%H:%M")

# Load model results
model = pickle.load(open('WORK_DIR / geock_model_all.pkl', 'rb'))

notes = {}

notes["FINAL_ENGINE.md"] = f"""# GEOCK Final Engine Results

**Date**: {today} {time_now}

## Final Model Performance

| Metric | Value | Notes |
|--------|-------|-------|
| LOO-R | **{model['loo_r']:.4f}** | Honest generalization (leave-one-out) |
| RKF-R | {model['rkf_r']:.4f} ± {model['rkf_std']:.4f} | Repeated 5-Fold × 5 repeats |
| Train-R | {model['train_r']:.4f} | Mild overfitting |
| Gap | {model['gap']:.4f} | Train-R - RKF-R |
| LOO-MAE | {model['loo_mae']:.2f} pKd | Mean absolute error |
| N compounds | **{model['n_compounds']}** | 41x from original 96 |

## Model Architecture

- **Type**: Ridge Regression
- **Features**: ECFP4 Morgan fingerprints (512D → {model['ke']}D via SelectKBest)
- **Regularization**: alpha = {model['alpha']}
- **Training compounds**: {model['n_compounds']}
- **Feature selection**: SelectKBest(f_regression, k={model['ke']})

## Evolution

| Version | N | Features | LOO-R | Notes |
|---------|---|---------|-------|-------|
| v1 physics-only | 96 | 24 physics | 0.060 | Near random! |
| v2 physics+ECFP | 96 | 9 physics + 18 ECFP | 0.504 | First real signal |
| v3 ECFP-only | 1,094 | 100 ECFP | 0.641 | Major improvement |
| v4 ECFP-large | 3,990 | 200 ECFP | **{model['loo_r']:.3f}** | 41x data, maintained R |

## Key Insights

1. **ECFP dominates** — physics features add marginal value
2. **More data maintains R** — 41x more data kept LOO-R ~0.63
3. **Strong regularization needed** — alpha=100 prevents overfitting
4. **LP-PDBBind is goldmine** — 19,443 binding affinity records available

## Usage

```python
from geock_engine import predict_pKd
result = predict_pKd("CCO")
print(f"pKd = {{result['pKd']:.2f}}")
```

```bash
python geock_engine.py --smiles "CC(=O)Oc1ccccc1C(=O)O"
```

## Files

- Model: `WORK_DIR / geock_model_all.pkl`
- Engine: `WORK_DIR / geock_engine.py`
- Predictions: `WORK_DIR / results_all.tsv`
"""

notes["DATA_ACQUISITION.md"] = f"""# Data Acquisition — LP-PDBBind

## Final Stats

- **PDB files downloaded**: 3,997 from RCSB
- **LP-PDBBind CSV**: 19,443 binding affinity records
- **Training compounds**: {model['n_compounds']} with valid ECFP + affinity
- **PDB file cache**: `CACHE_DIR / lp_pdb_files/`

## LP-PDBBind CSV Stats

- Total: 19,443 complexes
- Already downloaded PDBs: 3,997
- Valid training compounds: {model['n_compounds']}

## Data Sources Status

| Source | Status | Notes |
|--------|--------|-------|
| RCSB PDB | ✅ Works | 3,997+ files |
| ChEMBL API | ❌ 500 errors | Inaccessible |
| PDBbind aliyun | ❌ 404 | URL broken |
| CASF-2016 Figshare | ❌ Blocked | 0 bytes |
| HF pdbbindpp-2020 | ❌ 401 | Auth required |
| LP-PDBBind CSV | ✅ Works | GitHub THGLab |

## Cache Locations

- PDB files: `CACHE_DIR / lp_pdb_files/`
- Features: `CACHE_DIR / lp_all_features.pkl`
- LP CSV: `CACHE_DIR / LP_PDBBind.csv`
- Combined: `CACHE_DIR / features_combined.pkl`
"""

notes["GEOCK_MOC.md"] = f"""# GEOCK MOC

## Final Engine Results

- **LOO-R**: {model['loo_r']:.4f} (honest)
- **Training data**: {model['n_compounds']} compounds
- **Model**: Ridge + ECFP fingerprints (ke={model['ke']}, alpha={model['alpha']})

## Key Discoveries

1. **ECFP is the primary signal** — physics-only gives LOO=0.06, ECFP gives LOO={model['loo_r']:.3f}
2. **More data maintains quality** — 41x more data kept LOO-R stable
3. **Strong regularization critical** — alpha=100 prevents overfitting
4. **LP-PDBBind is a goldmine** — 19,443 binding affinity records accessible

## Files

- Engine: `geock_engine.py`
- Model: `geock_model_all.pkl`
- Predictions: `results_all.tsv`
"""

for fname, content in notes.items():
    path = VAULT / fname
    with open(path, 'w') as f:
        f.write(content)
    print(f"Pushed: {fname}")

print(f"\nAll notes pushed to {VAULT}")
print(f"Model: N={model['n_compounds']}, LOO-R={model['loo_r']:.4f}")
