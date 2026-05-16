---
tags: [project, ml, drug-discovery, active, in-progress]
type: project
status: in-progress
related: [[AutoDock Vina]], [[Vinardo]], [[DiffDock]], [[PDBBind]], [[RDKit]], [[ProLIF]], [[PLEC Fingerprints]], [[GEOCK]], [[vina_score.py]]
---

# 🎯 Binding Affinity Prediction Model

> [!abstract] Goal
> Given a protein-ligand complex, predict how tightly the ligand binds.
> **Output:** pKd (2–12) or ΔG (0 to −16 kcal/mol)
> **Target:** Pearson R > 0.80 on [[PDBBind]] Core Set

---

## Problem History

> [!bug] What Went Wrong
> | Issue | Root Cause | Fix Applied |
> |-------|-----------|------------|
> | MSE > 60 | GEOCK stub returns \[0,1\], labels span \[2,12\] | [[vina_score.py]] |
> | Poor generalisation | No protein-specific features | Adding [[ProLIF]] |
> | Worked at 30, broke at full dataset | Scale mismatch invisible at small scale | Proper normalisation |

---

## Feature Stack

```
┌─────────────────────────────────────────────────────┐
│                FULL FEATURE VECTOR                   │
├──────────┬─────────────┬───────────┬────────────────┤
│ Vina 6D  │ ECFP4 2048D │ RDKit 7D  │  ProLIF FP     │
├──────────┼─────────────┼───────────┼────────────────┤
│ physics  │ chem ident  │physiochem │ interactions   │
│ geometry │ what it IS  │ logP,TPSA │ which residues │
└──────────┴─────────────┴───────────┴────────────────┘
```

> [!important] Why Each Layer is Needed
> - **Vina:** 3D geometry, steric, H-bonds — but blind to chemical identity
> - **ECFP4:** what the molecule structurally is — but no protein context
> - **RDKit:** lipophilicity, polarity, flexibility — physicochemical profile
> - **ProLIF:** ==which residues make which interactions== — the protein-specific signal that was missing

---

## Architecture

```
PDBBind complex
      ↓
[10Å pocket extraction]
      ↓
[vina_score.py → 6D]    [RDKit → 2055D]    [ProLIF → FP]
      └────────────────┬──────────────────────┘
                       ↓
             [Concatenate features]
                       ↓
              [XGBoost / RF regressor]
                       ↓
                  pKd prediction
```

### Why XGBoost / RF (Not Neural Net)

> [!note]
> ~5,000 samples in refined set → RF/XGBoost **beats** neural nets at this scale.
> - Handles mixed dense + sparse features natively
> - Interpretable: can see which features matter
> - Fast iteration
> - Revisit GNN when data > 50k or with augmentation

---

## Progress Checklist

- [x] Fixed GEOCK stub → real Vina scoring
- [x] Built [[vina_score.py]] with Vinardo support
- [x] [[RDKit]] installed + ECFP4 pipeline
- [x] [[ProLIF]] installed
- [x] [[oddt]] installed (PLEC fingerprints)
- [ ] Full feature pipeline integrated
- [ ] Training run on PDBBind refined set
- [ ] Benchmarked on core set (285 compounds)
- [ ] Hyperparameter tuning
- [ ] DiffDock confidence gating integrated

---

## Benchmark Targets

| Method | Pearson R | Status |
|--------|----------|--------|
| AutoDock Vina baseline | 0.56 | beat this first |
| RF-Score | 0.74 | next milestone |
| PLEC linear | 0.82 | target |
| ==My model== | ==**> 0.80**== | ==**goal**== |

---

## Key Files

| File | Purpose | Status |
|------|---------|--------|
| `vina_score.py` | Physics scoring | ✅ Done |
| `features.py` | Full feature pipeline | 🔜 TODO |
| `train.py` | XGBoost training | 🔜 TODO |
| `evaluate.py` | Core set benchmark | 🔜 TODO |
