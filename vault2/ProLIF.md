---
tags: [tool, fingerprints, interactions, protein-ligand, installed]
type: tool
status: installed
install: "pip install prolif MDAnalysis"
related: [[PLEC Fingerprints]], [[RDKit]], [[oddt]], [[Binding Affinity Prediction Model]]
---

# 🔬 ProLIF

> [!abstract] One Line
> Protein-Ligand Interaction Fingerprints. Tells you **which residues** make **which interactions** with your ligand. The protein-specific signal my ensemble was missing.

```bash
pip install prolif MDAnalysis
```

---

## Why ProLIF Solves My Problem

> [!important] The Missing Signal
> Vina gives you one number. ProLIF gives you a bitstring like:
> `HIS41_HBond | PHE140_Hydrophobic | GLU166_SaltBridge`
>
> That's protein-specific context. That's what the ensemble needed.

---

## Core Workflow

```python
import MDAnalysis as mda
import prolif as plf

# Load
protein_u = mda.Universe("protein.pdb")
ligand_u  = mda.Universe("ligand.sdf")

# Convert
prot_mol = plf.Molecule.from_mda(protein_u.select_atoms("protein"))
lig_mol  = plf.Molecule.from_mda(ligand_u.atoms)

# Fingerprint
fp = plf.Fingerprint()
fp.run_from_iterable([lig_mol], prot_mol)

# Output as DataFrame
df = fp.to_dataframe()
# Columns: (ResidueID, InteractionType)
# e.g. ('HIS41', 'HBDonor'), ('PHE140', 'Hydrophobic')

# Output as numpy vector (for ML)
bv = fp.to_bitvectors()[0].ToNumpy().astype(float)
```

---

## Interaction Types

| Interaction | Symbol | Description |
|-------------|--------|-------------|
| `HBDonor` | 🔵 | Ligand donates H-bond to protein |
| `HBAcceptor` | 🟣 | Ligand accepts H-bond from protein |
| `Hydrophobic` | 🟡 | Nonpolar C–C contacts |
| `PiStacking` | 🔴 | Aromatic ring face/edge interaction |
| `PiCation` | 🟠 | Cation–pi interaction |
| `Anionic` | ➖ | Negative–positive charge pair |
| `Cationic` | ➕ | Positive–negative charge pair |
| `VdWContact` | ⚪ | Generic van der Waals |
| `MetalCoordination` | ⚙️ | Ligand coordinates a metal ion |

---

## For My ML Pipeline

```python
def get_prolif_features(protein_pdb, ligand_sdf):
    prot_u = mda.Universe(protein_pdb)
    lig_u  = mda.Universe(ligand_sdf)

    prot_mol = plf.Molecule.from_mda(prot_u.select_atoms("protein"))
    lig_mol  = plf.Molecule.from_mda(lig_u.atoms)

    fp = plf.Fingerprint(interactions=[
        "HBDonor", "HBAcceptor", "Hydrophobic",
        "PiStacking", "Anionic", "Cationic"
    ])
    fp.run_from_iterable([lig_mol], prot_mol)
    return fp.to_bitvectors()[0].ToNumpy().astype(float)
```

---

## ProLIF vs PLEC vs Vina

| | [[AutoDock Vina]] | [[PLEC Fingerprints]] | ProLIF |
|-|------|-------|--------|
| Encodes | Energy | Atom environments | ==Interaction types== |
| Interpretable | Partial | ❌ | ==✅ residue-level== |
| Maintenance | Active | ⚠️ Abandoned | ==✅ Active== |
| Python 3.13 | ✅ | ✅ (with `six`) | ✅ |

> [!tip] Rule of Thumb
> Use **ProLIF** as primary interaction FP.
> Use **PLEC** if you want maximum information density for a powerful model.
> Ideally use ==both==.

---

## References

- [ProLIF Docs](https://prolif.readthedocs.io)
- [ProLIF GitHub](https://github.com/chemosim-lab/ProLIF)
