---
tags: [tool, docking, scoring-function, physics-based, mastered]
type: tool
status: mastered
related: [[Vinardo]], [[DiffDock]], [[GNINA]], [[vina_score.py]], [[Binding Affinity Prediction Model]]
---

# ⚗️ AutoDock Vina

> [!abstract] One Line
> Open-source molecular docking engine. The industry standard. I rebuilt its scoring function from scratch.

**Paper:** Trott & Olson, 2010 — *J Comput Chem* 31:455–461
**Code:** [[vina_score.py]]

---

## What It Does

Takes a **receptor** (protein) and a **ligand** (small molecule) and:
1. Searches for the best binding pose
2. Scores each pose with an empirical scoring function
3. Returns binding affinity estimate in **kcal/mol** (negative = binding)

---

## The Scoring Function

> [!note] The Key Formula
> All terms operate on **surface-to-surface distance**, not atomic centre distance.

$$d_{ij} = r_{ij} - (vdw_i + vdw_j)$$

where $r_{ij}$ is the distance between atom centres and $vdw$ are van der Waals radii.

### Five Interaction Terms

| Term | Formula | Weight | Meaning |
|------|---------|--------|---------|
| **Gauss₁** | $e^{-(d/0.5)^2}$ | −0.035579 | Short-range steric attraction |
| **Gauss₂** | $e^{-((d-3)/2)^2}$ | −0.005156 | Medium-range steric attraction |
| **Repulsion** | $d^2$ if $d < 0$ | +0.840245 | Atomic clash penalty |
| **Hydrophobic** | ramp $[0.5, 1.5\text{ Å}]$ | −0.035069 | Buried nonpolar contacts |
| **H-bond** | ramp $[-0.7, 0.0\text{ Å}]$ | −0.587439 | Donor–acceptor pairs |

### Torsional Entropy Penalty

$$\text{affinity} = \frac{c}{1 + 0.058459 \times N_{rot}}$$

where $c$ = weighted sum of all pairwise interaction terms.

> [!tip] Intuition
> More rotatable bonds = more entropy lost on binding = worse (less negative) affinity score. A floppy ligand pays a larger penalty.

---

## Output Range Reference

```
Strong binders    →  -12 to -15 kcal/mol   ✅
Moderate binders  →   -8 to -12 kcal/mol   ✅
Weak / non-binders→    > -6 kcal/mol        ⚠️
Random / clashing →   positive values       ❌
```

> [!warning] The Bug I Fixed
> GEOCK's stub returned values in \[−1, 1\]. Real Vina outputs \[−15, 0\]. That mismatch caused MSE > 60. See [[GEOCK]] and [[vina_score.py]].

---

## Atom Types (PDBQT)

| Type | Meaning | VDW Radius |
|------|---------|-----------|
| `C`, `A` | Hydrophobic carbon | 1.9 Å |
| `N`, `NA` | Nitrogen (NA = H-bond acceptor) | 1.8 Å |
| `O`, `OA` | Oxygen (OA = H-bond acceptor) | 1.7 Å |
| `HD` | H-bond donor hydrogen | 1.0 Å |
| `SA` | Sulfur acceptor | 2.0 Å |
| `Cl`, `Br`, `I` | Halogens | 1.8–2.2 Å |

---

## What Vina ==Cannot== Capture

> [!caution] Vina Blind Spots
> - Chemical identity (doesn't know scaffold type)
> - Solvation effects (crude implicit model)
> - Protein flexibility beyond rotamers
> - Quantum / polarisation effects
>
> **Fix:** stack [[RDKit]] + [[ProLIF]] on top

---

## CLI Usage

```bash
vina --receptor receptor.pdbqt \
     --ligand ligand.pdbqt \
     --center_x 10.0 --center_y 5.0 --center_z 8.0 \
     --size_x 20 --size_y 20 --size_z 20 \
     --out docked.pdbqt \
     --log vina.log
```

## Python Drop-in (My Implementation)

```python
from vina_score import VinaScorer

scorer = VinaScorer(use_vinardo=True)   # Vinardo for PDBBind training
score  = scorer(rec_coords, rec_types,
                lig_coords, lig_types,
                n_torsions=5)
# → -8.3 kcal/mol
```

---

## Comparison: Vina vs Vinardo

| Property | Vina | [[Vinardo]] |
|----------|------|--------|
| Optimised for | Pose prediction | **Binding affinity** |
| Gauss₂ | included | removed |
| Hydrophobic ramp | 0.5–1.5 Å | **0.0–2.5 Å** (wider) |
| Use for PDBBind | ⚠️ okay | ✅ better |

---

## References

- [Trott & Olson 2010](https://doi.org/10.1002/jcc.21334)
- [AutoDock Vina GitHub](https://github.com/ccsb-scripps/AutoDock-Vina)
- [[Vinardo]] — improved weights
- [[vina_score.py]] — my Python implementation
