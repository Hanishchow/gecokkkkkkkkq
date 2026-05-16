---
tags: [code, scoring-function, python, built-by-me, complete]
type: code
status: complete
related: [[AutoDock Vina]], [[Vinardo]], [[DiffDock]], [[GEOCK]]
---

# 🐍 vina_score.py

> [!abstract] One Line
> My full Python reimplementation of AutoDock Vina scoring. Drop-in replacement for GEOCK's broken stub. Outputs calibrated kcal/mol.

---

## Contents

| Class / Function | Purpose |
|-----------------|---------|
| `AtomType` | Enum of all PDBQT atom types |
| `Atom` | Dataclass: coords + type + donor/acceptor flags |
| `VinaWeights` | Standard Vina 2010 weights |
| `VinardoWeights` | Vinardo 2016 re-fit weights |
| `VinaScorer` | ==Drop-in class: numpy arrays → kcal/mol== |
| `vina_score()` | Full dict: affinity + component breakdown |
| `vina_feature_vector()` | 6D numpy array for ML input |
| `diffdock_rescore()` | DiffDock confidence-weighted affinity |
| `atoms_from_arrays()` | Convert numpy → Atom objects |

---

## Quick Usage

```python
from vina_score import VinaScorer

scorer = VinaScorer(use_vinardo=True)  # True = better for PDBBind

score = scorer(
    rec_coords,       # (M, 3) receptor heavy atom coords
    rec_types,        # ["C", "OA", "NA", ...]  PDBQT strings
    lig_coords,       # (N, 3) ligand heavy atom coords
    lig_types,        # ["C", "N", "OA", ...]
    n_torsions=5,     # rotatable bonds in ligand
    diffdock_conf=0.3 # optional DiffDock confidence
)
# → -8.3  (kcal/mol)
```

---

## Full Breakdown

```python
result = scorer.score_dict(rec_coords, rec_types, lig_coords, lig_types)
# {
#   'affinity':     -8.3,
#   'inter_energy': -9.1,
#   'torsion_term':  0.29,
#   'components': {
#       'gauss1':      -2.1,
#       'gauss2':      -0.8,
#       'repulsion':   +0.0,
#       'hydrophobic': -3.5,
#       'hbond':       -2.7
#   }
# }
```

---

## Feature Vector for ML

```python
from vina_score import vina_feature_vector

feat = vina_feature_vector(rec_atoms, lig_atoms, n_rotatable_bonds=5)
# → np.array([gauss1_w, gauss2_w, repulsion_w, hydro_w, hbond_w, torsion])
# shape: (6,) float32
```

Combine with other features:
```python
full_features = np.concatenate([
    feat,           # Vina: 6D
    ecfp4_array,    # RDKit ECFP4: 2048D
    physchem,       # RDKit descriptors: 7D
    prolif_fp,      # ProLIF: variable
])
```

---

## The Math

$$d_{ij} = r_{ij} - (vdw_i + vdw_j)$$

$$c = \sum_{i,j} \left[ w_1 \cdot G_1(d) + w_2 \cdot G_2(d) + w_3 \cdot R(d) + w_4 \cdot H_{phob}(d) + w_5 \cdot H_{bond}(d) \right]$$

$$\text{affinity} = \frac{c}{1 + w_{tor} \cdot N_{rot}}$$

---

## DiffDock Confidence Gating

```python
from vina_score import diffdock_rescore

# Sigmoid gate: confidence < -1.5 → shrink affinity toward zero
good = diffdock_rescore(-8.3, confidence=0.5)   # → -8.2 (kept)
bad  = diffdock_rescore(-8.3, confidence=-3.0)  # → -1.2 (penalised)
```

> [!tip] Why This Exists
> Prevents the ML model from learning from poses DiffDock wasn't confident about. Bad poses with high Vina scores = noise. Gate them out.
