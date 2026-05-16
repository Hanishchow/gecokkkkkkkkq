---
tags: [fingerprints, cheminformatics, protein-ligand, high-performance]
type: concept
status: know-it
related: [[oddt]], [[ProLIF]], [[RDKit]], [[Binding Affinity Prediction Model]]
---

# 🧬 PLEC Fingerprints

> [!abstract] One Line
> Protein-Ligand Extended Connectivity fingerprints. Encodes ligand atoms **in the context of which protein atoms they touch**. Linear model on PLEC beats Vina by a mile.

**Paper:** Wójcikowski et al., 2019 — *J Cheminform* 11:39

---

## The Key Insight

> [!note] ECFP vs PLEC
> ```
> ECFP:  "there's a phenyl ring"
> PLEC:  "there's a phenyl ring stacking with PHE140's environment"
> ```
> PLEC encodes the **contact context**, not just the molecule.

For each contacting atom pair *(protein atom i, ligand atom j)*:

$$\text{bit} = \text{hash}(\text{ECFP}_{ligand}(j, d_L) + \text{ECFP}_{protein}(i, d_P))$$

---

## Benchmark Performance

| Model | Pearson R (PDBBind Core) |
|-------|------------------------|
| AutoDock Vina | 0.56 |
| ECFP4 only | ~0.60 |
| ==PLEC (linear model)== | ==**0.817**== |
| GNINA CNN | 0.84 |

> [!important]
> A **linear model** on PLEC beats everything except deep CNNs. That means the features are incredibly information-rich.

---

## Usage via oddt

```bash
pip install six && pip install oddt
```

```python
import oddt
from oddt.fingerprints import PLEC

protein = next(oddt.toolkit.readfile("pdb", "protein.pdb"))
protein.protein = True   # ← don't forget this

ligand  = next(oddt.toolkit.readfile("sdf", "ligand.sdf"))

fp = PLEC(ligand, protein,
          depth_ligand=2,    # ECFP radius for ligand side (=ECFP4)
          depth_protein=4,   # ECFP radius for protein side (deeper = more context)
          size=16384,        # fingerprint length
          count_bits=True)   # count contacts vs binary

# → sparse numpy array, shape (16384,)
```

---

## Parameters

| Param | Recommended | Meaning |
|-------|-------------|---------|
| `depth_ligand` | 2 | Ligand ECFP radius |
| `depth_protein` | 4 | Protein ECFP radius (more context) |
| `size` | 16384 | Fingerprint bit length |
| `count_bits` | True | Count contacts (richer than binary) |
| `ignore_hoh` | True | Ignore water molecules |

---

## When to Use PLEC vs ProLIF

> [!tip] Decision Guide
>
> **Use PLEC when:**
> - Training a high-performance ML model (RF, XGBoost, NN)
> - You want maximum information density
> - Interpretability doesn't matter
>
> **Use [[ProLIF]] when:**
> - You want to know WHICH residue makes WHICH interaction
> - Debugging / understanding a compound
> - Interpretable features for medicinal chemistry
>
> **Best: use both** and let the model weight them.

---

## References

- Wójcikowski et al. (2019) [doi:10.1186/s13321-019-0389-5](https://doi.org/10.1186/s13321-019-0389-5)
- [[oddt]] — the package that implements PLEC
