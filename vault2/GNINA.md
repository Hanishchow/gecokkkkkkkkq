---
tags: [tool, docking, cnn, nvidia-only, skip]
type: tool
status: cannot-use
blocker: AMD GPU
related: [[AutoDock Vina]], [[DiffDock]], [[AMD GPU ROCm]]
---

# 🚫 GNINA

> [!abstract] One Line
> CNN-based rescoring on top of Vina search. State-of-art accuracy. ==Permanently blocked — NVIDIA CUDA only.==

**Paper:** McNutt et al., 2021 — *J Cheminform* 13:43

---

## What It Is

GNINA keeps Vina's **search algorithm** but replaces its scoring function with a **3D convolutional neural network** trained on PDBBind.

```
Vina search algorithm  →  CNN scoring
                             ↓
                    CNNscore + CNNaffinity + CNNvariance
```

### Three Outputs

| Score | Range | Meaning |
|-------|-------|---------|
| **CNNscore** | 0–1 | Probability pose is correct |
| **CNNaffinity** | ~1–12 | Predicted pKd (like PDBBind labels) |
| **CNNvariance** | low = good | CNN uncertainty |

> [!tip] CNNaffinity is the Holy Grail
> It's directly calibrated to PDBBind pKd scale — exactly the label format I need. This is why everyone wants GNINA.

---

## Why I ==Cannot== Use It

> [!error] Permanently Blocked
> ```
> ❌ Requires NVIDIA CUDA
> ❌ gninatorch (Python wrapper) — CUDA only
> ❌ Precompiled binary — CUDA runtime
> ✅ My GPU = AMD → ROCm
>
> Decision: SKIP GNINA FOREVER
> ```
> See [[AMD GPU ROCm]]

---

## My Replacements

| GNINA feature | My replacement |
|---------------|---------------|
| CNNaffinity | [[AutoDock Vina]] (Vinardo) + ML model |
| Pose scoring | [[DiffDock]] confidence score |
| 3D CNN features | [[ProLIF]] + [[RDKit]] features |

---

## If I Ever Get NVIDIA Access

```bash
# Binary
wget https://github.com/gnina/gnina/releases/latest/download/gnina
chmod +x gnina
./gnina --receptor rec.pdb --ligand lig.sdf \
        --autobox_ligand lig.sdf --score_only

# Python wrapper
conda install -c conda-forge gninatorch
```
