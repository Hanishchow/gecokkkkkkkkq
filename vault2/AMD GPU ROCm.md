---
tags: [hardware, amd, gpu, rocm, constraint]
type: hardware
status: active
related: [[GNINA]], [[WSL Setup]], [[Tools Installed]]
---

# 💻 AMD GPU + ROCm

> [!warning] Permanent Constraint
> My GPU is AMD. This rules out ==all NVIDIA CUDA-only tools forever==. Design the stack around this.

---

## Hardware

```
GPU:        AMD
Framework:  ROCm (AMD's CUDA equivalent)
CUDA:       ❌ unavailable
```

---

## What This Rules Out

| Tool | Reason Blocked | My Alternative |
|------|---------------|---------------|
| [[GNINA]] | CUDA kernels | Vina + ML |
| gninatorch | libmolgrid = CUDA only | Skip |
| Some OpenMM kernels | CUDA optimised | CPU mode |

---

## PyTorch on AMD

```bash
# Install ROCm PyTorch (match version to your ROCm)
pip install torch --index-url https://download.pytorch.org/whl/rocm6.1

# Check ROCm version first
rocm-smi --showversion
# or: cat /opt/rocm/.info/version
```

| ROCm Version | Install URL suffix |
|-------------|-------------------|
| 5.7 | `rocm5.7` |
| 6.0 | `rocm6.0` |
| 6.1 | `rocm6.1` |

```python
import torch
print(torch.cuda.is_available())   # True if ROCm is working (HIP = CUDA API)
```

---

## What Works Fine on AMD

> [!check] ROCm-Compatible Stack
> - ✅ [[vina_score.py]] — pure Python/numpy
> - ✅ [[RDKit]] — CPU only, fast enough
> - ✅ [[ProLIF]] / [[oddt]] — CPU only
> - ✅ XGBoost / Random Forest — CPU, no GPU needed
> - ✅ PyTorch (standard ops) — ROCm
> - ✅ [[DiffDock]] — pure PyTorch

---

## Check Your GPU

```bash
rocm-smi          # AMD equivalent of nvidia-smi
rocminfo          # detailed device info
```
