---
tags: [tool, cheminformatics, fingerprints, descriptors, installed]
type: tool
status: installed
install: "conda install -c conda-forge rdkit"
related: [[PLEC Fingerprints]], [[ProLIF]], [[Binding Affinity Prediction Model]]
---

# 🧪 RDKit

> [!abstract] One Line
> The backbone of cheminformatics in Python. If it's about molecules in code, RDKit does it.

```bash
conda install -c conda-forge rdkit

# verify
python -c "from rdkit import Chem; print(Chem.MolFromSmiles('CCO'))"
```

---

## Core: ECFP4 / Morgan Fingerprints

> [!note] What ECFP Encodes
> Morgan fingerprints capture molecular **substructures** — which atoms are connected to what, up to a given radius. They're rotation/translation invariant and perfect for ML.

```python
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

mol = Chem.MolFromSmiles("c1ccc(NC(=O)c2ccccc2)cc1")

# ECFP4 = Morgan radius=2, 2048 bits
fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
fp_array = np.array(fp)   # shape (2048,) — use directly as ML features
```

**ECFP radius guide:**

| Name | Radius | Captures |
|------|--------|---------|
| ECFP2 | 1 | Atoms + immediate neighbours |
| ==ECFP4== | ==2== | ==Standard — most common== |
| ECFP6 | 3 | More detail, more sparse |

---

## Physicochemical Descriptors (My 7)

```python
from rdkit.Chem import Descriptors

def get_physchem(mol):
    return np.array([
        Descriptors.MolLogP(mol),            # lipophilicity
        Descriptors.TPSA(mol),               # polar surface area (Å²)
        Descriptors.NumHDonors(mol),         # H-bond donors
        Descriptors.NumHAcceptors(mol),      # H-bond acceptors
        Descriptors.NumRotatableBonds(mol),  # flexibility
        Descriptors.MolWt(mol),              # molecular weight
        Descriptors.NumAromaticRings(mol),   # aromaticity
    ])
```

> [!tip] Lipinski Rule of Five Reference
> - MolWt < 500
> - logP < 5
> - HBD ≤ 5
> - HBA ≤ 10
>
> Drug-likeness filter. Most PDBBind ligands pass this.

---

## 3D Conformer Generation

```python
from rdkit.Chem import AllChem

mol = Chem.MolFromSmiles("CCO")
mol = Chem.AddHs(mol)                        # add hydrogens first

AllChem.EmbedMolecule(mol, AllChem.ETKDGv3()) # generate 3D coords
AllChem.MMFFOptimizeMolecule(mol)             # energy minimise

coords = mol.GetConformer().GetPositions()    # numpy (N, 3)
```

---

## My Full Ligand Feature Pipeline

```python
def featurize_ligand(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # ECFP4 fingerprint (2048-D)
    fp = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048))

    # Physicochemical (7-D)
    physchem = get_physchem(mol)

    return np.concatenate([fp, physchem])  # → 2055-D vector
```

---

## Combined Feature Stack

```
[Vina 6D]  +  [ECFP4 2048D]  +  [RDKit 7D]  +  [ProLIF FP]
   ↓               ↓                  ↓              ↓
physics      chem identity       physicochem    interactions
```

> [!important]
> ECFP4 tells the model **what the molecule is**.
> Vina tells the model **how it fits in the pocket**.
> They're ==complementary== — never use one without the other.

---

## References

- [RDKit Docs](https://www.rdkit.org/docs/)
- [RDKit GitHub](https://github.com/rdkit/rdkit)
