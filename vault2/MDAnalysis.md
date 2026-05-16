---
tags: [tool, molecular-dynamics, structure, installed]
type: tool
status: installed
install: "pip install MDAnalysis"
related: [[ProLIF]], [[AutoDock Vina]]
---

# 🔩 MDAnalysis

> [!abstract] One Line
> Reads molecular structure files (PDB, SDF, MOL2, trajectories). ==Mostly invisible — it's the plumbing under ProLIF.==

```bash
pip install MDAnalysis  # comes with prolif automatically
```

---

## What It Does For Me

Loads PDB/SDF files into objects that [[ProLIF]] can process.

```python
import MDAnalysis as mda
import prolif as plf

u        = mda.Universe("protein.pdb")
lig_u    = mda.Universe("ligand.sdf")

prot_mol = plf.Molecule.from_mda(u.select_atoms("protein"))
lig_mol  = plf.Molecule.from_mda(lig_u.atoms)
```

---

## Selection Syntax (Useful to Know)

```python
# Protein only (no water/ligand)
protein = u.select_atoms("protein")

# Binding pocket (10Å around ligand)
pocket  = u.select_atoms("protein and around 10 resname LIG")

# Specific residue
his41   = u.select_atoms("resname HIS and resid 41")

# Backbone only
bb      = u.select_atoms("backbone")
```

---

## Getting Coordinates

```python
# All atoms
coords = u.atoms.positions           # numpy (N, 3)

# Pocket only
pocket_coords = u.select_atoms(
    "protein and around 10 resname LIG"
).positions
```

---

## Supported File Formats

| Format | Use |
|--------|-----|
| `.pdb` | Protein structures |
| `.sdf` | Ligand structures |
| `.mol2` | Ligand (alternate) |
| `.prmtop` + `.dcd` | MD trajectories |

---

## References
- [MDAnalysis Docs](https://docs.mdanalysis.org)
- [GitHub](https://github.com/MDAnalysis/mdanalysis)
