# GEOCK AutoResearch — Program Instructions

You are an autonomous ML researcher optimising binding affinity prediction
for the GEOCK 2.0 drug discovery pipeline.

Your job: improve `val_pearson_r` in `train.py` by modifying that file only.
Higher val_pearson_r = better. Baseline = **0.775**.

---

## Setup (run once before starting)

```bash
# 1. Install dependencies
pip install scikit-learn scipy numpy rdkit qiskit qiskit-aer

# 2. Place these files in the same directory:
#    prepare.py, train.py, program.md,
#    bio_engine.py, enhanced_physics.py, patch_parse.py

# 3. Prepare data (caches features — run once)
python prepare.py

# 4. Verify baseline runs
python train.py
# Should print: val_pearson_r: 0.775000 (approximately)
```

---

## The Rules

### You MUST:
- Create a branch: `git checkout -b autoresearch/<tag>` before each experiment
- Modify ONLY `train.py`
- Keep the two output lines exactly as:
  ```
  val_pearson_r: {float}
  val_mae: {float}
  ```
- Run: `python train.py > run.log 2>&1`
- Read result: `grep "^val_pearson_r:\|^val_mae:" run.log`
- Record in `results.tsv` (do NOT commit this file)
- If val_pearson_r improved → commit and advance
- If val_pearson_r same or worse → `git reset --hard HEAD`

### You MUST NOT:
- Modify `prepare.py` — this defines the evaluation
- Modify `program.md`
- Change the train/val/test splits
- Install new packages (only use what's in pyproject.toml)
- Hardcode val labels into predictions
- Look at test set labels

---

## What To Try (in priority order)

### Round 1 — Feature Selection (fastest wins)
The current baseline uses k=2 features. Try systematically:

1. **k=3**: Add each remaining feature one at a time to the winning k=2 pair
   - Best k=2: `E2_chem_lipophilic` + `E4_bio_size_penalty`
   - Try adding each of: E3_quantum_vqe, E1_vinardo_hbond, E2_chem_burial, E4_bio_drug_likeness
   - Report best k=3 combination

2. **k=4, k=5**: Continue adding features greedily

3. **Quantum feature**: Specifically test if `E3_quantum_vqe` (index 14) improves R when added to the k=2 baseline
   - This is important for the paper — does quantum help?

### Round 2 — Model Changes
Try replacing Lasso with:

4. **SVR** (Support Vector Regression):
   ```python
   from sklearn.svm import SVR
   SVR(kernel='rbf', C=1.0, epsilon=0.1)
   ```

5. **GradientBoosting** (strong for small n):
   ```python
   from sklearn.ensemble import GradientBoostingRegressor
   GradientBoostingRegressor(n_estimators=50, max_depth=2)
   ```

6. **Gaussian Process** (uncertainty estimates):
   ```python
   from sklearn.gaussian_process import GaussianProcessRegressor
   from sklearn.gaussian_process.kernels import RBF, WhiteKernel
   GaussianProcessRegressor(kernel=RBF() + WhiteKernel())
   ```

7. **BayesianRidge** (automatic regularisation):
   ```python
   from sklearn.linear_model import BayesianRidge
   BayesianRidge()
   ```

### Round 3 — Feature Engineering
8. **Log transform VQE**: The VQE energy is large (-800 kcal/mol) — try log(|VQE|)
   ```python
   X_tr[:, 14] = np.log(np.abs(X_tr[:, 14]) + 1)
   ```

9. **Interaction terms**: chem_lipophilic × bio_size_penalty
   ```python
   from sklearn.preprocessing import PolynomialFeatures
   PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
   ```

10. **PCA on ECFP4**: Compress 512 ECFP bits to 10 principal components
    ```python
    from sklearn.decomposition import PCA
    # Apply PCA to X_train[:, 24:] (ECFP4 part)
    # Concatenate with physics features
    ```

### Round 4 — Hyperparameter Tuning
11. **Alpha grid search** (Lasso):
    Try: [0.001, 0.005, 0.01, 0.015, 0.02, 0.05, 0.1]
    Current best: 0.015

12. **ElasticNet l1_ratio**:
    Try: [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]

13. **SVR C and epsilon**:
    Try C in [0.1, 1, 10] × epsilon in [0.01, 0.1, 0.5]

### Round 5 — Feature Subset Exploration
14. **Physics only** (no ECFP4): `FEATURE_SUBSET = slice(0, 24)`
15. **ECFP4 only**: `FEATURE_SUBSET = slice(24, 536)`
16. **All features**: `FEATURE_SUBSET = slice(0, 536)`
17. **No quantum**: test without feature index 14

---

## Recording Results

Maintain `results.tsv` (tab-separated, not committed to git):

```
experiment_id	description	val_pearson_r	val_mae	kept	notes
baseline	Lasso k=2 alpha=0.015	0.775000	0.XXX	yes	chem_lipophilic + bio_size_penalty
exp_001	k=3 + E3_quantum_vqe	0.XXX	0.XXX	yes/no	...
exp_002	k=3 + E1_vinardo_hbond	0.XXX	0.XXX	yes/no	...
```

---

## Key Scientific Questions To Answer

These go in the paper — record the experiment that answers each:

1. **Does quantum VQE help?** (exp: add feature 14 to k=2 baseline)
2. **What is the optimal k?** (exp: greedy forward selection)
3. **Do ECFP4 bits add signal beyond 2 features?** (exp: PCA-compressed ECFP4)
4. **Does SVR beat Lasso at this scale?** (exp: round 2)
5. **What is the best single feature?** (exp: k=1 for each of 24 features)

---

## Stopping Criteria

Stop experimenting when:
- val_pearson_r has not improved in 10 consecutive experiments, OR
- val_pearson_r > 0.90 (exceptional result — stop and report), OR
- You have completed all 5 rounds above

---

## Simplicity Criterion

All else being equal, simpler is better.
A model with val_pearson_r = 0.780 and k=2 features beats
a model with val_pearson_r = 0.781 and k=10 features.
Prefer interpretable results — these go in a paper.

---

## Final Report

After stopping, report:
```
Best model:       {description}
val_pearson_r:    {value}    (baseline: 0.775)
val_mae:          {value}    (baseline: ~0.5)
Features used:    {list}
Key finding:      {one sentence}
Quantum helps?:   yes/no (R with VQE vs without VQE)
```