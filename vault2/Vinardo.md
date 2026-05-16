---
tags: [scoring-function, physics-based, pdbbind, weights, implemented]
type: concept
status: implemented
related: [[AutoDock Vina]], [[vina_score.py]], [[PDBBind]], [[Binding Affinity Prediction Model]]
---

# 🔧 Vinardo

> [!abstract] One Line
> Re-fitted AutoDock Vina weights specifically optimised for binding affinity prediction on PDBBind. Use this over standard Vina when training ML models.

**Paper:** Quiroga & Villarreal, 2016 — *PLoS ONE* 11:e0155183

---

## Why Vinardo Exists

> [!note] The Problem with Standard Vina
> Vina's original weights were optimised for **pose prediction** — is the docked pose correct?
> Vinardo re-fit the same math for **affinity correlation** on PDBBind.
> Same equations. Better numbers for my use case.

---

## Weight Comparison

| Term | Vina (2010) | Vinardo (2016) | Change |
|------|------------|----------------|--------|
| Gauss₁ | −0.035579 | −0.045 | ==stronger attraction== |
| Gauss₂ | −0.005156 | **0.0** | ==removed entirely== |
| Repulsion | +0.840245 | +0.8 | similar |
| Hydrophobic | −0.035069 | −0.030 | weaker but wider ramp |
| H-bond | −0.587439 | −0.600 | slightly stronger |
| Torsion | 0.058459 | 0.055 | slightly less penalty |

### Key Differences

> [!important] What Actually Changed
> 1. **Gauss₂ removed** — the wide Gaussian added noise for affinity prediction
> 2. **Hydrophobic ramp extended** — `[0.0, 2.5 Å]` instead of `[0.5, 1.5 Å]` → captures more buried contacts
> 3. **Fit on PDBBind core set** → directly relevant to my dataset

---

## Hydrophobic Ramp Visual

```
Vina:     |████|░░░░░░░|          ← tight: 0.5–1.5 Å only
           0   0.5  1.5  2.5

Vinardo:  |████████████|          ← wide: 0.0–2.5 Å
           0   0.5  1.5  2.5
```

---

## When to Use Which

| Use Case | Use |
|----------|-----|
| Reproducing literature docking | [[AutoDock Vina]] standard |
| Training affinity ML model | ==**Vinardo** ✅== |
| PDBBind benchmark | ==**Vinardo** ✅== |
| Pose screening only | Vina standard |

---

## Code

```python
from vina_score import VinaScorer

# Standard Vina
scorer_std = VinaScorer(use_vinardo=False)

# Vinardo — use this for PDBBind
scorer_vin = VinaScorer(use_vinardo=True)

score = scorer_vin(rec_coords, rec_types,
                   lig_coords, lig_types,
                   n_torsions=5)
```

---

## References

- Quiroga & Villarreal (2016) [doi:10.1371/journal.pone.0155183](https://doi.org/10.1371/journal.pone.0155183)
- [[vina_score.py]] — `VinardoWeights` class
