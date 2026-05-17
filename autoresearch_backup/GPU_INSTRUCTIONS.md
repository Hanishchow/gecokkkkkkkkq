# GEOCK 3D Structure-Based Binding Affinity Prediction

## Summary of Work Completed

### Phase 1-3 (CPU Work - DONE)
- **Phase 1**: Integrated 20 physics features with ECFP → CV R = 0.699
- **Phase 2**: Advanced feature engineering (polynomial, ratios) → CV R = 0.699
- **Phase 3**: PCA on physics + enhanced features → CV R = 0.693

### Phase 5-6 (GPU Work - READY)
- **Phase 5**: 3D CNN scripts created → Ready to run
- **Phase 6**: GNN scripts created → Ready to run
- **Phase 7**: Ensemble script created → Ready to run

## Models Saved

| Model | CV R | Status |
|-------|------|--------|
| geock_model_best.pkl | 0.70 | Baseline |
| geock_model_physics.pkl | 0.70 | +Physics features |
| geock_model_enhanced.pkl | 0.70 | +Feature engineering |
| geock_model_prolif.pkl | 0.69 | +PCA enhanced |

## Scripts Created for GPU

### Phase 5: 3D CNN
```bash
# First generate grids (will take ~30-60 min for 7000 PDBs)
python phase5_generate_grids.py --max 7000

# Then train CNN (on GPU)
python phase5_train_cnn.py --epochs 100 --batch-size 16
```

### Phase 6: GNN
```bash
# Build graph dataset (~20 min)
python phase6_train_gnn.py --build-only

# Train GNN (on GPU)
python phase6_train_gnn.py --epochs 100 --batch-size 32
```

### Phase 7: Ensemble
```bash
# Combine all models
python phase7_ensemble.py
```

## To Run on GPU Laptop

1. **Copy files to GPU laptop:**
   ```bash
   rsync -av /home/chow/autoresearch/ user@gpu-laptop:~/geock/
   ```

2. **Generate 3D grids:**
   ```bash
   cd ~/geock
   python phase5_generate_grids.py --max 7000 --output 3d_grids.pkl
   ```

3. **Train 3D CNN:**
   ```bash
   python phase5_train_cnn.py --grids 3d_grids.pkl --labels ~/.cache/geock_autoresearch/lp_new_features_8k.pkl --output geock_cnn_3d.pt --epochs 100 --batch-size 16
   ```

4. **Build graphs:**
   ```bash
   python phase6_train_gnn.py --build-only --max 7000
   ```

5. **Train GNN:**
   ```bash
   python phase6_train_gnn.py --epochs 100 --batch-size 32
   ```

6. **Create ensemble:**
   ```bash
   python phase7_ensemble.py
   ```

## Expected Results

| Approach | Expected R |
|----------|-------------|
| Baseline (ECFP) | 0.70 |
| + Physics features | 0.70-0.71 |
| + Enhanced features | 0.70-0.71 |
| 3D CNN (GPU) | 0.75-0.80 |
| GNN (GPU) | 0.77-0.82 |
| Ensemble | 0.78-0.85 |

## Hardware Requirements

- **Current machine**: CPU only (done)
- **GPU laptop**: 6 GB VRAM minimum
  - 3D CNN: Batch size 16, ~4GB VRAM
  - GNN: Batch size 32, ~3GB VRAM

## Data Locations

- PDB files: `/home/chow/.cache/geock_autoresearch/lp_pdb_files/` (7,393 files)
- Compound data: `/home/chow/.cache/geock_autoresearch/lp_new_features_8k.pkl`
- Physics features: `/home/chow/.cache/geock_autoresearch/physics_features_8k.pkl`

## Notes

- Physics features already integrated into models
- Grid generation script creates 24³ grids at 0.5Å resolution
- GNN uses PyTorch Geometric with GINE convolutions
- All scripts use early stopping with patience=20