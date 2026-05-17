# Training Session Summary - Apr 2, 2026

## Goal
Improve binding affinity (pKd) prediction from R ~0.70 to target >0.75

## Results

### Original Best Model
- **File**: `geock_ensemble_final.pkl`
- **CV R**: 0.7358
- **Val R**: 0.6996
- **Test R**: 0.7179

### My Best Result (Chunk 9)
- **CV R**: 0.7136 ± 0.0108
- **Val R**: 0.7210
- **Test R**: 0.7112
- **Approach**: Multi-seed XGBoost ensemble with 8 different random seeds

## What Was Tried

| Chunk | Approach | CV R |
|-------|----------|------|
| 1 | LightGBM, XGBoost variants | 0.6913 |
| 2 | RF, GBM, CatBoost, feature engineering | 0.6923 |
| 3 | NN, Ridge, feature focus | 0.6857 |
| 4 | Grid search + stacking | 0.6959 |
| 5 | 24D physics features | 0.6920 |
| 6 | Combined physics + RF | 0.6916 |
| 7 | Bagging + smaller trees | 0.6839 |
| 8 | Best configs blend | 0.6991 |
| 9 | Multi-seed ensemble | **0.7136** |
| 10 | Final optimization | 0.7136 |
| 11 | Fine-tune regularization | 0.7068 |

## Key Findings

1. **Multi-seed ensemble** provided the best results (Chunk 9)
2. **Features used**: ECFP (512) + molecular (8) + physics (20) + interaction (20) = 560 features
3. **Best XGBoost config**: max_depth=8, lr=0.03, reg_alpha=0.7, reg_lambda=7.0, n_estimators=400
4. Could not replicate original CV R = 0.7358

## Files Generated

- `geock_ensemble_final.pkl` - Original best (CV 0.7358)
- `geock_final_best.pkl` - My best model (CV 0.7136)
- `chunk1.py` through `chunk11.py` - Training scripts
- `chunk*_results.pkl` - Result summaries

## Next Steps (for GPU laptop)

1. Try 3D CNN on generated grids (`3d_grids.pkl`)
2. Try GNN model (`phase6_train_gnn.py`)
3. Combine 3D features with best 2D model