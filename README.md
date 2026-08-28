# geock-docking-research

XGBoost and neural scoring models for protein-ligand docking affinity, validated against CASF benchmark suites.

## What it does

- Trains and evaluates ligand-based and physics-informed scoring models for protein-ligand binding affinity prediction.
- Extracts simple pocket features via distance-based counting and combines them with ligand ECFP4 fingerprints and physics descriptors.
- Runs feature-extraction, training, and validation workflows across acquisition, engine, and pipeline modules in `autoresearch_backup/`.
- Validates trained models against the CASF-2007 and CASF-2013 benchmark suites, writing metric JSONs and prediction CSVs to `CASF_Results/`.
- Ships associated analysis notebooks, CASF-2016 exploration, and per-suite verification reports alongside an Obsidian notes vault.

## Requirements

- Python 3 (compiled bytecode in the repo indicates 3.13).
- XGBoost (used for the XGBoost training script).
- A deep-learning stack for the neural scoring models (PyTorch) plus scientific/cheminformatics packages (numpy, pandas, scikit-learn, RDKit for ECFP4 features) — inferred from the scripts and feature sources.

## Setup

TODO: document the entry point. The repo contains Windows batch launchers (`GEOCK-Open-Terminal.bat`, `GEOCK-OpenCode-Session*.bat`), a `GEOCK_BindingDB_Colab.ipynb` notebook, and a separate `autoresearch_backup/README.md`, but no pinned install workflow or requirements manifest was found.

## Project layout

```
CASF_Results/              # CASF benchmark metrics.json + predictions.csv
autoresearch_backup/       # pipelines: data acquisition, engines, feature extraction, training, validation
GEOCK_BindingDB_Colab.ipynb
RESULTS.md                 # results summary
CASF-2007-Verification-Report.md
GEOCK-v2-Final-Paper.md
GEOCK-v2-Complete-Documentation.md
GEOCK-QuickRef.md
.obsidian/                 # Obsidian vault configuration
GEOCK-*.bat                # session launcher scripts
*.md                       # session notes and research logs
```

## Status

Research code, not a maintained tool. This is an experimental results repository: benchmarks such as casf2007_metrics.json and casf2013_metrics.json exist under `CASF_Results/`, but performance was evaluated on small, partially-featured training sets (most training examples lacked pocket features) and no ongoing maintenance, packaging, or support is provided.
