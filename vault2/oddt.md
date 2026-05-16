---
tags: [tool, cheminformatics, plec, installed]
type: tool
status: installed
install: "pip install six && pip install oddt"
related: [[PLEC Fingerprints]], [[ProLIF]], [[RDKit]]
---

# 🧰 oddt — Open Drug Discovery Toolkit

> [!abstract] One Line
> Python cheminformatics toolkit. ==Use it only for PLEC fingerprints== — everything else has better alternatives.

---

## Install

> [!warning] Known Bug on Python 3.13
> oddt fails to build without `six`. Always install `six` first.

```bash
pip install six          # ← required first
pip install oddt
```

**Error you'll see if you skip `six`:**
```
ModuleNotFoundError: No module named 'six'
```

---

## Main Use: PLEC Fingerprints

See [[PLEC Fingerprints]] for full details.

```python
import oddt
from oddt.fingerprints import PLEC

protein = next(oddt.toolkit.readfile("pdb", "protein.pdb"))
protein.protein = True
ligand  = next(oddt.toolkit.readfile("sdf", "ligand.sdf"))

fp = PLEC(ligand, protein, depth_ligand=2, depth_protein=4, size=16384)
```

---

## Status

> [!caution] Mostly Unmaintained
> Last major update ~2020. Python 3.13 works with the `six` fix.
> For interaction fingerprints, prefer [[ProLIF]].
> Keep oddt **specifically for PLEC** — nothing else implements it as well.

---

## References
- [oddt GitHub](https://github.com/oddt/oddt)
