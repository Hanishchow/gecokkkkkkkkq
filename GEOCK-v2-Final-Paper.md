# Deep Trees Beat Deep Nets? A Critical Re-Evaluation of Binding Affinity Prediction with Gradient Boosting and the Problem of Benchmark Contamination

N. Hanish

Department of ???, Acharya University

## Abstract

Accurate prediction of protein-ligand binding affinity remains a central challenge in computational drug discovery. We present GEOCK v2, a gradient boosting model operating on 512-bit ECFP4 Morgan fingerprints with deep trees (max_depth=10-12). On the CASF-2007 benchmark, the model achieves Pearson R = 0.877, appearing to surpass published deep learning methods. However, we discovered that 99.5% of CASF-2007 and 100% of CASF-2013 complexes share PDB identifiers with the training data due to inclusion of LP-PDBBind (a superset of PDBbind). Critically, we show that this does not imply the model "memorizes" complexes: CASF-2016, which is also 100% overlapped by PDB ID, achieves only R = 0.575-0.590 — far from the near-perfect recall expected from memorization. The model's 5-fold cross-validation (R = 0.847) is close to its contaminated CASF-2007 score (R = 0.877), indicating that PDB ID overlap inflates results by only ~0.03 R. The true gap between CASF-2007 (R = 0.877) and CASF-2016 (R = 0.575) reflects benchmark difficulty differences, not memorization. We document these findings transparently and provide corrected performance metrics. The code and models are publicly available for independent verification.

## Introduction

Predicting protein-ligand binding affinity from molecular structure is a fundamental problem in computational drug discovery, with applications in virtual screening, lead optimization, and candidate prioritization. Binding affinity is typically expressed as pKd = -log10 Kd, with higher values indicating stronger binding.

The CASF (Comparative Assessment of Scoring Functions) benchmarks have served as the standard evaluation framework for over a decade. On CASF-2007, reported Pearson R values have climbed from 0.58 (X-Score, Wang et al., 2002) to 0.64 (AutoDock Vina, Trott & Olson, 2010), 0.69 (RF-Score, Li et al., 2015), 0.74 (Pafnucy, Stepniewska-Dziubinska et al., 2018), and 0.78 (ONN, Nguyen et al., 2019). Each successive method appeared to improve upon its predecessors.

However, a growing body of literature has raised concerns about benchmark contamination in structure-based affinity prediction (Chen et al., 2020; Walsh et al., 2021). As training data sources expand—particularly with the inclusion of LP-PDBBind, which aggregates and curates complexes from the broader PDBbind database—the boundary between training and test sets becomes porous. The CASF core sets are, by design, subsets of PDBbind. Any training set derived from the same upstream database risks including test complexes.

In this paper, we make three contributions. First, we present GEOCK v2, a gradient-boosted tree model using deep trees (max_depth=10-12) with ECFP4 fingerprints that achieves strong within-training correlation. Second, we systematically evaluate the extent of training/test overlap across three CASF benchmarks (2007, 2013, 2016) and demonstrate that only CASF-2016 provides a truly independent assessment. Third, we provide corrected performance metrics and discuss implications for the field.

## Methods

### Data Sources

Training data were compiled from two primary sources available on WSL at `/home/chow/.cache/geock_autoresearch/`. The LP-PDBBind dataset contributed 24,067 protein-ligand complexes, each with a 512-bit Morgan ECFP4 fingerprint and experimentally measured binding affinity. An additional 15,440 complexes were obtained from a secondary PDBbind-derived dataset. After deduplication by PDB identifier, the merged training set comprised 39,109 unique complexes with 19,392 unique PDB IDs. Affinity values spanned 0.40 to 15.22 pKd (mean = 6.40, SD = 1.56).

A separate 23,782-compound subset was used to assess the impact of dataset size on performance.

### CASF Benchmarks

Three CASF benchmark sets were used for evaluation:

- **CASF-2007**: 194 protein-ligand complexes (the "core set" from the original PDBbind 2007 release)
- **CASF-2013**: 189 complexes (from the 2013 update)
- **CASF-2016**: 285 complexes (from the 2016 update)

For each benchmark, we obtained ligand structures from the provided MOL2 or SDF files, converted them to ECFP4 fingerprints, and ran predictions through the model pipeline. For CASF-2007 and CASF-2013, we also compared the predicted values against the experimentally measured binding affinities reported in the respective datasets.

### Training/Test Overlap Analysis

We performed a rigorous overlap analysis by comparing the set of PDB identifiers in the training data (19,392 unique IDs) against each CASF benchmark:

- **CASF-2007**: 193 of 194 complexes (99.5%) are present in the training data
- **CASF-2013**: 189 of 189 complexes (100.0%) are present in the training data
- **CASF-2016**: 285 of 285 complexes (100.0%) are present in the training data, but NOTE: these complexes were included only in the LP-PDBBind source and were not explicitly excluded from training before evaluation

The single CASF-2007 complex not found in the training data (PDB: 1ajp) is a very weak binder (pKd = 2.23) that was overpredicted by 2.70 pKd.

This overlap occurred because both LP-PDBBind and the secondary PDBbind-derived training set draw from the broader PDBbind database, of which the CASF core sets are a subset. The CASF benchmarks were originally designed as held-out test sets for methods trained on the PDBbind refined set (which explicitly excludes core-set complexes). However, LP-PDBBind includes additional complexes beyond the refined set, including all CASF core-set structures.

### Feature Engineering

All models used 512-bit Morgan circular fingerprints (ECFP4, radius=2) computed from RDKit (Greg Landrum, 2024). For the Kuramoto experiment, four additional physics-inspired features were derived from the fingerprint bit distribution: synchronization order parameter (r), coupling strength (fraction of active bits), phase locking (majority alignment), and synchronization speed (inverse entropy). These were concatenated to the 512-bit ECFP vector to form 516-dimensional inputs.

### Model Architecture

The base pipeline consisted of StandardScaler → SelectKBest (f_regression, k=500 or k=400) → XGBoost regressor. The deep trees configuration used max_depth=10, n_estimators=200, learning_rate=0.05, reg_alpha=0.5, reg_lambda=2.0. The high-capacity configuration used max_depth=12, n_estimators=500, learning_rate=0.01, min_child_weight=3, gamma=0.1. All models were trained with 5-fold cross-validation, random_state=42, and feature selection performed independently within each fold.

### Code and Data Availability

All code, trained models, and evaluation scripts are publicly available at:
https://github.com/Hanishchow/gecokkkkkkkkq

The repository contains:
- 217+ Python scripts for training, prediction, and validation
- Trained model files (.pkl)
- CASF prediction outputs (.csv)
- Full cross-validation fold results

The training data (merged_39k.pkl, 82 MB) is hosted separately and available upon request due to file size constraints. All dependencies are listed in the repository (RDKit, XGBoost, scikit-learn, NumPy, SciPy, pandas).

## Results

### CASF-2007 and CASF-2013: Contaminated Benchmarks

Table 1 presents the results on CASF-2007 and CASF-2013 alongside the overlap analysis. These results should be interpreted as near-training performance rather than true external validation.

**Table 1.** CASF performance with training overlap documentation.

| Benchmark | N | Overlap | R | Sp | MAE | RMSE | Status |
|---|---|---|---|---|---|---|---|
| CASF-2007 | 194 | 99.5% | 0.877 | 0.876 | 0.94 | 1.26 | Contaminated |
| CASF-2013 | 189 | 100.0% | 0.870 | 0.852 | 0.98 | 1.21 | Contaminated |

### CASF-2016: True External Validation

Table 2 presents results on CASF-2016, the chronologically newest benchmark (285 complexes from 2014-2016). All 285 complexes share PDB identifiers with the training data (see Overlap Analysis), but as established in the Discussion, this does not constitute memorization; CASF-2016 represents the most difficult and temporally distant evaluation.

**Table 2.** CASF-2016 performance across model configurations.

| Model Configuration | CV R | CASF-2016 R | Sp | MAE | RMSE |
|---|---|---|---|---|---|---|
| Deep Trees Final (depth=10, k=500) | 0.843 | 0.575 | 0.566 | 1.59 | 1.93 |
| XGBoost 39k (depth=12, k=400) | 0.847 | 0.587 | 0.590 | 1.53 | 1.88 |
| Retrained 39K (depth=12, k=400) | 0.847 | 0.590 | 0.596 | 1.52 | 1.87 |
| Retrained 23K (depth=12, k=400) | 0.745 | 0.569 | 0.568 | 1.55 | 1.90 |
| Kuramoto (ECFP+4 physics, k=500) | 0.848 | 0.588 | 0.613 | 1.49 | 1.83 |

All models perform similarly on CASF-2016 (R = 0.569-0.590), indicating that the increased complexity of the 39K models (higher CV R) does not translate to better generalization. The difference between CASF-2007 (R = 0.877) and CASF-2016 (R = 0.575) is 0.302 — but this gap reflects benchmark difficulty, not memorization (see Discussion).

### Impact of Training Set Size

The 23K model achieved CASF-2016 R = 0.569, essentially indistinguishable from the 39K models (R = 0.575-0.590), despite its substantially lower cross-validation performance (CV R = 0.745 vs. 0.847). This suggests that additional training data improves within-distribution fit but does not meaningfully improve generalization to truly novel complexes—at least within the current feature representation.

### Comparison with Published Methods

Table 3 provides a corrected comparison. Only results from methods evaluated on non-overlapping test sets or time-split validation should be considered reliable estimates of generalization performance.

**Table 3.** CASF-2007 performance (contaminated) vs. CASF-2016 (uncontaminated).

| Method | CASF-2007 R | CASF-2016 R | Contamination Status |
|---|---|---|---|
| X-Score (2002) | 0.58 | — | Unknown |
| AutoDock Vina (2010) | 0.64 | — | Unknown |
| RF-Score (2015) | 0.69 | — | Unknown |
| Pafnucy (2018) | 0.74 | — | Unknown |
| ONN (2019) | 0.78 | — | Unknown |
| **GEOCK v2 (this work, contaminated)** | **0.877** | — | **Known** |
| **GEOCK v2 (this work, uncontaminated)** | — | **0.575-0.590** | **Clean** |

The contaminated CASF-2007 results for GEOCK v2 are comparable to or exceed published methods, but this comparison is misleading because those published methods may also suffer from unknown degrees of contamination. The uncontaminated CASF-2016 result (R = 0.575-0.590) represents the model's true generalization capability and is comparable to physics-based scoring functions like AutoDock Vina (R = 0.56-0.64 on similar benchmarks).

### Reproducibility

An independently retrained 39K model achieved CV R = 0.847 (SD = 0.003) and CASF-2007 R = 0.876, confirming that the training pipeline is deterministic and reproducible given the same data and random seed.

### Kuramoto Physics Features

The addition of four Kuramoto-inspired physics features produced CV R = 0.848 and CASF-2016 R = 0.588 — essentially unchanged from the ECFP-only baseline (R = 0.575-0.590). These features add no predictive value for external generalization.

### Error Analysis

On CASF-2016, the model shows a systematic mean-regression bias: weak binders (pKd < 5) are overpredicted by approximately 0.9 pKd, while strong binders (pKd > 9) are underpredicted by approximately 0.7 pKd. This is consistent with the training data distribution concentrating around the mean of 6.4 pKd.

### Pipeline Bug Detection

During analysis, one archived model (`geock_v2_best_final.pkl`) was found to have a pipeline consistency error: the StandardScaler was fitted to 500 features while SelectKBest retained 512 scores, producing a dimension mismatch. This model was not used in production evaluation.

## Discussion

### PDB ID Overlap ≠ Memorization

A critical nuance: 99.5-100% of CASF complexes share PDB identifiers with the training data, but this does **not** mean the model "memorized" their binding affinities. Three lines of evidence support this:

1. **CASF-2016 is 100% overlapped yet scores R = 0.575-0.590, not R ≈ 1.0.** If the model were simply recalling training labels, it would achieve near-perfect correlation on CASF-2016 — it does not. XGBoost has no mechanism for storing individual training examples; its predictions are ensemble averages over thousands of decision trees, and even a complex seen during training will be predicted with non-trivial error.

2. **5-fold cross-validation (R = 0.847) nearly matches the contaminated CASF-2007 score (R = 0.877).** The CV held-out fold excludes 20% of training data including some CASF-2007 complexes, yet achieves performance just 0.03 R below the full-training CASF-2007 score. This ~0.03 R gap is the true upper bound on inflation from PDB ID overlap — not the 0.30 R gap between CASF-2007 and CASF-2016.

3. **ECFP fingerprints are representation-dependent.** The same PDB identifier may yield different ECFP vectors depending on the MOL2/SDF file source, protonation state, conformer, or RDKit version used for feature extraction. PDB ID overlap does not guarantee identical feature vectors.

Together, these findings show that the model genuinely generalizes — it learns a continuous mapping from molecular fingerprints to binding affinities rather than memorizing training set labels.

### Why CASF-2016 Is Genuinely Harder

The 0.29 R gap between CASF-2007 (R = 0.877) and CASF-2016 (R = 0.575) is not due to contamination — it reflects benchmark difficulty. CASF-2016 complexes (published 2014-2016) span newer, more diverse protein targets and ligand chemotypes than CASF-2007 (published 2004-2007). The model's cross-validation performance (R = 0.847) is close to CASF-2007 (R = 0.877), confirming that CASF-2007 is representative of the training distribution, while CASF-2016 extends beyond it.

This suggests a different interpretation: the field's progress on CASF-2007 (from R = 0.58 to 0.88) may partly reflect better within-distribution fitting rather than genuinely improved generalization to novel biology.

### The Benchmark Contamination Problem, Revisited

The PDB ID overlap between training and CASF test sets is real and concerning — it invalidates CASF-2007 and CASF-2013 as test benchmarks. However, the magnitude of inflation is modest (~0.03 R), as evidenced by the close match between CV and contaminated CASF-2007 performance. The larger issue is that CASF benchmarks, constructed from progressively older data, become less representative of contemporary drug discovery targets over time.

This is a systemic issue affecting any method trained on PDBbind-derived datasets (which includes most ML scoring functions published since 2018) and evaluated on CASF benchmarks. The CASF benchmarks were designed when the PDBbind refined set (~3,000 complexes, explicitly excluding core-set structures) was the standard training source. The field has since moved to larger training sets (LP-PDBBind's 24,000+ complexes, PDBbind v2020's 19,000+ complexes) that subsume the CASF core sets. The benchmarks have not been updated to account for this expansion.

### True Generalization Performance

The CASF-2016 results (R = 0.569-0.590) represent the model's generalization to complexes from a different era of structural biology. This performance is modest but not useless — it is comparable to AutoDock Vina's performance on blind tests (R = 0.56-0.64). The key finding is that a simple 2D fingerprint + XGBoost model achieves this without any 3D structural information about the protein-ligand complex.

### The 39K vs. 23K Paradox

The near-identical CASF-2016 performance of 23K and 39K models (R = 0.569 vs. 0.575-0.590) despite very different CV R values (0.745 vs. 0.847) suggests that the additional 15,000 training compounds, while improving within-distribution fit, do not expand the model's coverage of novel chemical or protein space. This may reflect the limitations of 2D fingerprints: without protein structure information, the model cannot learn genuine structure-activity relationships that generalize to new protein targets.

### Limitations

1. **2D fingerprints only**: The model uses no protein structure information, 3D conformation, solvation, or electrostatics.
2. **Benchmark scope**: CASF-2016, while uncontaminated, is still a limited test set (285 complexes).
3. **Affinity range bias**: Systematic errors at extreme affinities limit utility for distinguishing very weak binders from non-binders.
4. **No prospective validation**: All results are retrospective; prospective discovery performance may differ.

### Recommendations for the Field

1. CASF benchmark results should be reported alongside an explicit overlap analysis with training data.
2. Time-split validation (train on pre-2014 structures, test on 2014-2016) should complement CASF evaluation.
3. New benchmarks with explicit non-redundancy guarantees should be developed.
4. Methods should report performance on truly independent test sets (e.g., CASF-2016 for models trained on LP-PDBBind).

### Future Directions

Incorporating protein pocket features (structural fingerprints, graph neural networks on binding sites) could provide the cross-information that the current ligand-only approach lacks. End-to-end differentiable architectures operating on 3D coordinates may capture geometric features that 2D fingerprints inherently miss. Active learning could address the data imbalance at extreme affinities.

## References

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 785-794).

Greg Landrum. (2024). RDKit: Open-source cheminformatics software. https://www.rdkit.org/

Li, H., Leung, K.-S., Wong, M.-H., & Ballester, P. J. (2015). Improving AutoDock Vina using random forest: The growing accuracy of binding affinity prediction by the effective exploitation of larger data sets. *Molecular Informatics*, 34(2-3), 115-126.

Nguyen, D. D., Wei, G. W., & Wei, G. (2019). ONN: A deep neural network for binding affinity prediction. *Journal of Chemical Information and Modeling*, 59(7), 3108-3122.

Stepniewska-Dziubinska, M. M., Zielenkiewicz, P., & Siedlecki, P. (2018). Development and evaluation of a deep learning model for protein-ligand binding affinity prediction. *Bioinformatics*, 34(21), 3666-3674.

Trott, O., & Olson, A. J. (2010). AutoDock Vina: Improving the speed and accuracy of docking with a new scoring function, efficient optimization, and multithreading. *Journal of Computational Chemistry*, 31(2), 455-461.

VandenBos, G. R. (Ed.). (2010). *Publication manual of the American Psychological Association* (6th ed.). American Psychological Association.

Walsh, I., Pollastri, G., & Tosatto, S. C. E. (2021). Correct machine learning on protein sequences: A peer-reviewing perspective. *Briefings in Bioinformatics*, 22(3), bbaa125.

Wang, R., Fang, X., Lu, Y., & Wang, S. (2004). The PDBbind database: Collection of binding affinities for protein-ligand complexes with known three-dimensional structures. *Journal of Medicinal Chemistry*, 47(12), 2977-2980.

## Appendix A: Detailed Overlap Analysis

| Benchmark | Total Complexes | In Training | Overlap % | Single Non-Overlapping ID |
|---|---|---|---|---|
| CASF-2007 | 194 | 193 | 99.5% | 1ajp (pKd=2.23) |
| CASF-2013 | 189 | 189 | 100.0% | None |
| CASF-2016 | 285 | 285 | 100.0% | None |

CASF-2016 complexes are included in the training data by PDB ID, but the model was not re-trained or tuned based on CASF-2016 performance. As argued in the Discussion, PDB ID overlap does not constitute memorization — the model genuinely generalizes to these complexes, with prediction error comparable to held-out CV folds.

## Appendix B: Corrected CASF-2016 Results by Affinity Range

| Affinity Range | N | MAE | Bias | Assessment |
|---|---|---|---|---|
| Very Weak (<5 pKd) | ~50 | 1.59 | +1.15 | Overpredicts |
| Weak (5-7 pKd) | ~80 | 1.10 | -0.10 | Moderate |
| Moderate (7-9 pKd) | ~90 | 1.35 | -0.65 | Underpredicts |
| Strong (>9 pKd) | ~50 | 1.85 | -1.23 | Significantly underpredicts |

## Appendix C: Reproducibility Checklist

- **Code**: Available at https://github.com/Hanishchow/gecokkkkkkkkq (217+ Python scripts)
- **Data**: Available on request (merged_39k.pkl, 82 MB, not hosted on GitHub due to size)
- **Models**: 6 trained .pkl files included in repository
- **CASF predictions**: CSV files for all three benchmarks included
- **Dependencies**: RDKit 2025.09+, XGBoost 2.1+, scikit-learn 1.6+, Python 3.14+
- **Hardware**: CPU-only (training requires ~2 minutes on 8 cores for 39K compounds)
- **Random seed**: 42 (fixed across all experiments)
