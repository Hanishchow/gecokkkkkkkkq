# GEOCK Docker Image

Binding Affinity Prediction System using XGBoost + ECFP4 + Physics Features

## Quick Start

```bash
# Build the image
docker build -t geock:latest .

# Run prediction (SMILES only)
docker run --rm geock:latest predict "Cc1ccc(cc1)C(=O)O"

# Run docking (PDB + SMILES)
docker run --rm -v /path/to/pdb:/data geock:latest dock /data/6p87.pdb "COc1ccccc1OCC(=O)N(C)[C@@H](c1cccs1)c1c[nH]c2ccccc12"

# Interactive shell
docker run --rm -it geock:latest /bin/bash
```

## Available Commands

| Command | Description |
|--------|-------------|
| `predict <smiles>` | Predict affinity from SMILES only |
| `dock <pdb> <smiles>` | Full docking with PDB + SMILES |
| `--help` | Show help |

## Examples

```bash
# Aspirin
docker run --rm geock:latest predict "CC(=O)OC1=CC=CC=C1C(=O)O"

# Ibuprofen  
docker run --rm geock:latest predict "CC(C)Cc1ccc(cc1)C(C)C(=O)O"

# Interactive mode
docker run --rm -it geock:latest
```

## Performance

- **SMILES only**: R ≈ 0.45
- **With PDB (physics)**: R ≈ 0.71

## Requirements

- Docker 20.10+
- For docking: PDB files with ligand coordinates

## Volume Mounts

Mount PDB files to `/data`:
```bash
docker run -v /my/pdb/files:/data geock:latest dock /data/complex.pdb "SMILES"
```

## License

MIT
