---
tags: [about-me, identity, second-brain, core, meta]
created: 2026-03-20
updated: 2026-03-20
type: person
status: living-document
aliases: [Hanish, Me, Brain Root]
---

# 🧠 Hanish Chow — The Brain Root

> [!quote] My Philosophy
> *I don't just learn things. I build systems to make them permanent.*

---

## 🪪 Identity

| Field | Value |
|-------|-------|
| **Name** | Hanish Chow |
| **Location** | Bengaluru, Karnataka, India 🇮🇳 |
| **Machine** | Windows PC + WSL (Kali Linux) |
| **Shell path** | `/mnt/c/Users/yakka` |
| **Python** | Miniconda3, Python 3.13 |
| **GPU** | AMD (ROCm — not CUDA) |
| **Brain Tool** | Obsidian |
| **Automation** | n8n (`hanishchow.app.n8n.cloud`) |

---

## 🧬 What I Know

> [!info] Reading this section
> Each item below is a `[[wikilink]]` — click it to open the full note on that topic.

### 🔬 Computational Drug Discovery
- [[AutoDock Vina]] — rebuilt the scoring function from scratch
- [[DiffDock]] — diffusion docking, confidence scoring
- [[GNINA]] — know it cold, can't run it (AMD GPU)
- [[Vinardo]] — better Vina weights for PDBBind training
- [[PDBBind]] — gold standard binding affinity dataset

### 🧪 Cheminformatics
- [[RDKit]] — ECFP4, Morgan fingerprints, physicochemical descriptors
- [[PLEC Fingerprints]] — protein-ligand extended connectivity via [[oddt]]
- [[ProLIF]] — residue-level interaction fingerprints
- [[MDAnalysis]] — structure loading backend

### 🤖 ML / Modelling
- Feature scale mismatch → MSE > 60 (identified + fixed)
- Combined feature stack: physics + chem + interaction FP
- XGBoost / RF for binding affinity regression
- [[GEOCK]] — framework I'm using, fixed its broken stub

### ⚙️ DevOps / Tools
- [[WSL Setup]] — Kali on Windows
- [[AMD GPU ROCm]] — ROCm PyTorch, no CUDA
- [[n8n Automation]] — self-hosted workflows
- [[Obsidian Setup]] — this second brain
- [[Tools Installed]] — master install list

---

## 💻 Hardware Reality

> [!warning] AMD GPU Constraint
> My GPU is **AMD**. This permanently rules out GNINA, gninatorch, and anything requiring NVIDIA CUDA. See [[AMD GPU ROCm]].

```
OS:      Windows + WSL2 (Kali)
GPU:     AMD → ROCm
         ❌ gninatorch  ❌ GNINA binary
         ✅ PyTorch ROCm  ✅ Everything else
Shell:   /mnt/c/Users/yakka
Python:  Miniconda3 / 3.13
```

---

## 🧠 How I Think

> [!tip] Learning Style
> **Concepts first, commands second.** I understand things deeply before I know the syntax. I ask for the command when I'm moving fast and don't want to break flow.

- **Fast and direct** — all caps = thinking out loud, not angry
- Jumps between 30,000ft insight and ground-level "give me the command"
- Learns by doing, not reading docs front to back
- Trusts instincts on architecture — they're usually right
- Builds **systems**, not one-off scripts

---

## 🚀 Active Projects

- [[Binding Affinity Prediction Model]] — predict ΔG from protein-ligand structure
- [[n8n Automation]] — Google Calendar + Gmail workflows

---

## 💡 Core Insights (Permanent)

> [!important] Physics alone is blind to chemistry
> Vina sees atom types + distances. It doesn't know the molecule is a kinase inhibitor. ECFP4 + ProLIF fix this.

> [!important] Scale mismatch kills ML models
> Stub scores in \[0,1\] vs labels in \[-15, 0\] = MSE of 60+. Always check output range matches label range before training.

> [!important] PLEC beats vanilla ECFP for binding
> It encodes which protein residue environments touch which ligand substructures. That's the protein-specific signal the ensemble was missing.

---

## 🗺️ Brain Map

```
About Me (root)
├── Projects
│   ├── Binding Affinity Prediction Model
│   └── n8n Automation
├── Drug Discovery
│   ├── AutoDock Vina → vina_score.py
│   ├── DiffDock
│   ├── GNINA (skip)
│   ├── Vinardo
│   └── PDBBind
├── Cheminformatics
│   ├── RDKit
│   ├── ProLIF
│   ├── PLEC Fingerprints → oddt
│   └── MDAnalysis
├── Setup
│   ├── WSL Setup
│   ├── AMD GPU ROCm
│   ├── Tools Installed
│   └── Obsidian Setup
└── Reference
    ├── Papers to Read
    └── GEOCK
```

---

*Last updated: 2026-03-20 | This document is alive. Update it every time you learn something new.*
