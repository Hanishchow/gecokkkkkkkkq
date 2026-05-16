---
tags: [tool, docking, deep-learning, diffusion-model]
type: tool
status: know-conceptually
related: [[AutoDock Vina]], [[GNINA]], [[Binding Affinity Prediction Model]], [[vina_score.py]]
---

# 🌀 DiffDock

> [!abstract] One Line
> Diffusion model for molecular docking. Treats pose generation as a generative problem, not a search problem.

**Paper:** Corso et al., 2023 — ICLR 2023
**Repo:** [github.com/gcorso/DiffDock](https://github.com/gcorso/DiffDock)

---

## How It Differs from Vina

| | [[AutoDock Vina]] | DiffDock |
|--|----------------|----------|
| Approach | Physics search + scoring | ==Generative diffusion model== |
| Speed | Minutes per ligand | Seconds (GPU) |
| Grid needed | ✅ Yes | ❌ No — blind docking native |
| Output | Single best pose | **N poses + confidence scores** |
| Flexibility | Fixed function | Learned from data |

---

## How It Works

```
Random pose  →  [Denoising Network]  →  Refined pose
     ↑                                        ↓
  Noise                              Confidence score
```

Three degrees of freedom learned simultaneously:
1. **Translation** — where in space
2. **Rotation** — orientation
3. **Torsion angles** — internal conformation

---

## Confidence Score — ==Critical==

> [!warning] Confidence ≠ Affinity
> The confidence score tells you how good the **pose** is, NOT how tightly the ligand binds. Don't confuse them.

| Confidence | Interpretation | Use in training? |
|-----------|---------------|-----------------|
| `> 0` | Very high — top ~20% | ✅ Yes |
| `-1.5 to 0` | Medium — plausible | ✅ Yes |
| `< -1.5` | Low — likely wrong pose | ❌ Filter out |

### How I Use It

```python
from vina_score import diffdock_rescore

# High confidence → keeps full Vina affinity
good = diffdock_rescore(-8.3, confidence=0.5)    # → -8.2

# Low confidence → shrinks toward zero
bad  = diffdock_rescore(-8.3, confidence=-3.0)   # → -1.2
```

> [!tip] Why This Matters
> Without this filter, your ML model trains on bad poses and learns garbage correlations. Always gate by confidence.

---

## Usage

```bash
git clone https://github.com/gcorso/DiffDock && cd DiffDock
pip install -e .

python inference.py \
  --protein_path  receptor.pdb \
  --ligand_description "CCO" \
  --out_dir ./results \
  --inference_steps 20 \
  --samples_per_complex 40
```

Output: 40 poses ranked by confidence. Take top N where `confidence > -1.5`.

---

## In My Pipeline

```
DiffDock → top poses (conf > -1.5)
    → Vina rescoring → [[vina_score.py]]
    → DiffDock confidence gating
    → Feature extraction → ML model
```

---

## Limitations

> [!caution]
> - Needs GPU for reasonable speed (CPU = slow but works)
> - Does **not** output binding affinity — only pose quality
> - Best for drug-like molecules; struggles with macrocycles/peptides
> - AMD GPU works ✅ (pure PyTorch)
