---
tags: [dataset, benchmark, binding-affinity, gold-standard]
type: dataset
status: using
related: [[Binding Affinity Prediction Model]], [[AutoDock Vina]], [[Vinardo]]
---

# 📦 PDBBind

> [!abstract] One Line
> The gold standard dataset for protein-ligand binding affinity. Every structure has an experimentally measured ΔG.

**Website:** [pdbbind.org.cn](http://www.pdbbind.org.cn)

---

## Three Subsets

| Subset | Size (~2020) | Use |
|--------|-------------|-----|
| **General** | ~19,000 | All complexes with affinity |
| **Refined** | ~5,000 | Higher quality — use for ==training== |
| **Core** | 285 | Benchmark set — use for ==evaluation== |

> [!tip] My Workflow
> Train on **Refined**, benchmark on **Core**. Every paper does this — it makes results comparable.

---

## Affinity Measurement Types

$$\Delta G = RT \ln K_d$$

```python
import math
R = 1.987e-3   # kcal/(mol·K)
T = 298.15     # K  (room temp)

def kd_to_dG(Kd_nM):
    Kd_M = Kd_nM * 1e-9
    return R * T * math.log(Kd_M)  # negative = binding

def kd_to_pkd(Kd_nM):
    return -math.log10(Kd_nM * 1e-9)  # pKd = 2–12 range
```

| Measure | Meaning | ML label |
|---------|---------|---------|
| **Kd** | Dissociation constant | pKd = −log₁₀(Kd) |
| **Ki** | Inhibition constant | pKi = −log₁₀(Ki) |
| **IC50** | Half-max inhibitory conc. | approximate |

---

## Label Range

```
pKd scale:     2  ──────────────────── 12
               ↑                        ↑
           weak binder             strong binder

ΔG (kcal/mol): 0  ──────────────────── -16
               ↑                        ↑
           no binding              tight binding
```

> [!warning] The Bug This Caused
> GEOCK's stub returned \[0,1\]. PDBBind labels span \[2,12\] in pKd.
> MSE > 60 because the model had zero signal. Fixed by [[vina_score.py]].

---

## Benchmark: Pearson R on Core Set (285 compounds)

| Method | Pearson R |
|--------|----------|
| AutoDock Vina | 0.56 |
| RF-Score | 0.74 |
| PLEC (linear) | 0.82 |
| GNINA CNN | 0.84 |
| GatorAffinity (2025) | ~0.90 |
| ==**My target**== | ==**> 0.80**== |

---

## File Structure

```
refined-set/
  1a1e/
    1a1e_protein.pdb       ← receptor
    1a1e_ligand.mol2       ← ligand
    1a1e_pocket.pdb        ← binding pocket (10 Å)
  ...
INDEX_refined_data.2020    ← PDB ID, resolution, pKd/pKi/IC50
```

---

## Download

```bash
# Register at pdbbind.org.cn first, then:
wget http://pdbbind.org.cn/download/PDBbind_v2020_refined.tar.gz
wget http://pdbbind.org.cn/download/PDBbind_v2020_core.tar.gz
tar -xzf PDBbind_v2020_refined.tar.gz
```
