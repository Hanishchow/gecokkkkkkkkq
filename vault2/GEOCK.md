---
tags: [project, framework, bug, fixed]
type: tool
status: fixed
related: [[AutoDock Vina]], [[vina_score.py]], [[Binding Affinity Prediction Model]]
---

# ⚙️ GEOCK

> [!abstract] One Line
> The docking/scoring framework I'm building on. Had a critical stub scoring function — identified and fixed.

---

## The Bug

> [!bug] Root Cause of MSE > 60
> ```
> Stub output range:    ~[-1, 1]   ← what GEOCK returned
> Real affinity range:  [-15, 0]   ← what PDBBind labels are
>
> Result: model had no signal → MSE = 60+
> ```
> This is a **feature scale mismatch** bug. Classic and silent.

### Why It's Dangerous

The model trained without crashing. The loss looked like it was optimising. But the features and labels were completely incompatible in scale — the model was just learning noise.

---

## The Fix

Replaced the stub with a full [[AutoDock Vina]] / [[Vinardo]] reimplementation.

```python
# ❌ OLD — stub, don't use
# score = geock_stub_score(complex)  # returns ~0.3, meaningless

# ✅ NEW — real Vina scoring
from vina_score import VinaScorer
scorer = VinaScorer(use_vinardo=True)
score  = scorer(rec_coords, rec_types, lig_coords, lig_types, n_torsions=5)
# → -8.3 kcal/mol ← calibrated, meaningful
```

---

## What GEOCK Does Well

- ✅ Molecular graph construction
- ✅ Atom feature extraction
- ✅ Dataset loading pipeline
- ✅ Training loop infrastructure

## What Was Fixed

- ~~Scoring function~~ → [[vina_score.py]]
- ~~Feature scale~~ → proper kcal/mol
- ~~No protein-specific features~~ → [[ProLIF]]

---

## Lessons Learned

> [!important] Never Assume Stubs Are Labelled
> Framework stub functions fail silently. MSE > 10 on a binding affinity task almost always means a scaling issue, not a model architecture issue. Check your output range first.
