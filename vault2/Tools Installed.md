---
tags: [setup, tools, reference, master-list]
type: reference
status: living-document
related: [[WSL Setup]], [[AMD GPU ROCm]]
---

# 🛠️ Tools Installed

> [!info] Living Document
> Update this every time you install something new.

---

## Python Environment

```
Python:  3.13
Manager: Miniconda3
Path:    /home/chow/miniconda3
```

---

## Installed Packages

### Cheminformatics

| Package | Install | Status | Notes |
|---------|---------|--------|-------|
| `rdkit` | `conda install -c conda-forge rdkit` | ✅ | Core molecular toolkit |
| `oddt` | `pip install six && pip install oddt` | ✅ | Needed `six` fix |
| `prolif` | `pip install prolif MDAnalysis` | ✅ | Interaction FPs |
| `MDAnalysis` | comes with prolif | ✅ | Structure loader |

### ML / Scientific

| Package | Install | Status |
|---------|---------|--------|
| `numpy` | pre-installed | ✅ |
| `scikit-learn` | `pip install scikit-learn` | 🔜 check |
| `xgboost` | `pip install xgboost` | 🔜 check |
| `torch` (ROCm) | `pip install torch --index-url .../rocm6.1` | 🔜 check |

### ❌ Not Installed (and why)

| Package | Reason |
|---------|--------|
| `gninatorch` | NVIDIA CUDA only |
| `gnina` binary | NVIDIA CUDA only |

---

## Verify All Installs

```bash
python -c "
from rdkit import Chem
import oddt, prolif
import numpy as np
print('✅ All cheminformatics imports OK')
"

python -c "import torch; print('PyTorch:', torch.__version__, '| CUDA/ROCm:', torch.cuda.is_available())"
```

---

## Full Fresh Install (from scratch)

```bash
# Step 1: heavy C packages via conda
conda install -c conda-forge rdkit

# Step 2: pip packages
pip install six
pip install oddt
pip install prolif MDAnalysis
pip install xgboost scikit-learn

# Step 3: PyTorch (AMD ROCm)
pip install torch --index-url https://download.pytorch.org/whl/rocm6.1
```
