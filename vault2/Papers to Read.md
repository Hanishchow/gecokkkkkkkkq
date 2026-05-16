---
tags: [papers, research, reading-list, drug-discovery]
type: reference
status: living-document
related: [[Binding Affinity Prediction Model]], [[AutoDock Vina]], [[DiffDock]]
---

# 📚 Papers to Read / Know

---

## ✅ Already Know (from implementation)

> [!check] Mastered
> These I know well enough to implement from scratch.

| Paper | Year | Key Takeaway |
|-------|------|-------------|
| **AutoDock Vina** — Trott & Olson | 2010 | 5-term scoring function, surface-distance formulation. Built [[vina_score.py]] from this. |
| **Vinardo** — Quiroga & Villarreal | 2016 | Gauss₂ removed, wider hydrophobic ramp. Better for PDBBind. |
| **DiffDock** — Corso et al. | 2023 | Diffusion docking, confidence score interpretation (`> -1.5` = good pose). |
| **PLEC** — Wójcikowski et al. | 2019 | Linear model on PLEC = Rp 0.817. Best bang-for-buck fingerprint. |

---

## 📥 Read Next (Priority Order)

> [!todo] Queue

### 1. GatorAffinity (2025) — HIGHEST PRIORITY
- **Why:** Current SOTA, trained on 450k+ complexes. Understand what I'm competing with.
- [GitHub](https://github.com/AIDD-LiLab/GatorAffinity)

### 2. GNINA / McNutt et al. (2021)
- **Why:** Understand CNNaffinity even if I can't run it. Know the architecture.
- J Cheminform 13:43

### 3. RF-Score — Ballester & Mitchell (2010)
- **Why:** Direct ancestor of what I'm building. First ML scoring function on PDBBind.
- Bioinformatics 26:1169

### 4. EquiBind — Stärk et al. (2022)
- **Why:** Geometry-aware GNNs for docking. Future architecture direction.
- ICML 2022

### 5. DeepDTA — Öztürk et al. (2018)
- **Why:** Sequence + SMILES CNN approach. Different input modality to understand.
- Bioinformatics 34:i821

---

## 📋 Paper Reading Template

```markdown
## [Paper Title] — [Authors] ([Year])

**Problem:** what does it solve?
**Innovation:** what's actually new?
**Method:** how does it work (1 paragraph)?
**Result:** key benchmark number?
**Steal:** what can I use in my pipeline?
**Limits:** where does it fail?
```
