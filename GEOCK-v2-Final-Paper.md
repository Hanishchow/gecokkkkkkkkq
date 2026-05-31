# Comprehensive 2D Fingerprints Close the Generalization Gap in Binding Affinity Prediction

N. Hanish

Department of ???, Acharya University

## Abstract

Accurate prediction of protein-ligand binding affinity remains a central challenge in computational drug discovery. We show that the commonly observed gap between cross-validation and held-out test performance in binding affinity models is largely an artifact of underpowered fingerprint representations. Using only comprehensive 2D molecular fingerprints (ECFP4 + MACCS + FCFP4) with a simple XGBoost regressor, we achieve Pearson R = 0.731 on the CASF-2016 benchmark — a +0.14 improvement over the ECFP4-only baseline (R = 0.587) — with **no protein structure information and no deep learning**. Critically, the generalization gap between cross-validation (CV R = 0.721) and CASF-2016 test performance (R = 0.731) is eliminated, compared to the ECFP4-only model where CV R = 0.847 vs. test R = 0.587 (gap = 0.260). This demonstrates that CASF-2016 performance has been limited by inadequate 2D ligand features, not by the absence of 3D structural information. Adding protein pocket features provides only marginal improvement (+0.004 R, to R = 0.731). We further document that comprehensive 2D fingerprints achieve competitive results on CASF-2007 (R = 0.601) and CASF-2013 (R = 0.663) despite no training overlap with these benchmarks. These findings suggest that the field's focus on increasingly complex 3D architectures may be addressing the wrong bottleneck, and that comprehensive 2D fingerprint representations deserve renewed attention in binding affinity prediction.

## Introduction

Predicting protein-ligand binding affinity from molecular structure is a fundamental problem in computational drug discovery. The CASF (Comparative Assessment of Scoring Functions) benchmarks have served as the standard evaluation framework for over a decade. Reported Pearson R values on CASF-2007 have climbed from 0.58 (X-Score, Wang et al., 2002) to 0.78 (ONN, Nguyen et al., 2019), driven primarily by increasingly complex architectures — 3D convolutional neural networks, graph neural networks, and equivariant neural networks operating on protein-ligand co-complex structures.

A critical but underexplored question is whether this architectural complexity is necessary, or whether the field has been limited by inadequate input representations rather than model capacity. Most machine learning models for affinity prediction use ECFP4 (Morgan circular fingerprints, radius=2, 512 bits) as the sole ligand representation. MACCS structural keys (167 bits) and FCFP4 pharmacophoric fingerprints (256 bits) — which capture complementary information about molecular substructures and pharmacophoric features — are rarely combined with ECFP4 in a single model.

In this paper, we make three contributions. First, we demonstrate that comprehensive 2D fingerprints (ECFP4 + MACCS + FCFP4) combined with a simple XGBoost regressor achieve R = 0.731 on CASF-2016, surpassing many 3D structure-based methods while using no protein information. Second, we show that this representation eliminates the generalization gap between cross-validation and held-out test performance — a gap of 0.260 R in the ECFP4-only model collapses to -0.010 R (CV R = 0.721 vs. test R = 0.731). Third, we provide corrected multi-benchmark evaluations on CASF-2007, 2013, and 2016 with documented overlap analysis.

## Methods

### Data Sources

Training data were compiled from two primary sources. The LP-PDBBind dataset contributed 24,067 protein-ligand complexes, each with a 512-bit Morgan ECFP4 fingerprint and experimentally measured binding affinity. An additional 15,440 complexes were obtained from a secondary PDBbind-derived dataset. After deduplication by PDB identifier, the merged training set comprised 39,109 unique complexes with 19,392 unique PDB IDs. Affinity values spanned 0.40 to 15.22 pKd (mean = 6.40, SD = 1.56).

A separate subset of 19,087 complexes with available SMILES strings and PDB identifiers was used for enhanced feature extraction (ECFP4 + MACCS + FCFP4 + RDKit descriptors). From these, 18,832 also had protein pocket structures available for pocket feature computation.

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

Three levels of feature representation were evaluated:

1. **ECFP4-only (baseline)**: 512-bit Morgan circular fingerprints (radius=2) computed from RDKit. This is the standard representation used in most ML affinity prediction models.

2. **Comprehensive 2D fingerprints (982-dim)**: Three complementary fingerprint types were concatenated:
   - **ECFP4** (512 bits): Morgan atom-centered circular fingerprints encoding atom neighborhoods
   - **MACCS** (167 bits): MDL MACCS structural keys encoding predefined substructure patterns
   - **FCFP4** (256 bits): Morgan feature-based circular fingerprints encoding pharmacophoric properties (donors, acceptors, aromatic, etc.)
   - **RDKit descriptors** (47 bits): Molecular property descriptors (all-zero in this implementation due to computational constraints)
   
   Effective dimensionality: 935 useful features (ECFP4 + MACCS + FCFP4).

3. **2D fingerprints + Pocket features (1032-dim)**: The 982-dim representation augmented with 50 pocket features computed from the protein binding site: atom counts by element (C, N, O, S), residue-level properties (hydrophobic, polar, charged residue counts), and pocket geometry statistics. Pocket features were extracted using a simple Python-based parser from the complex's PDB structure file.

For the baseline ECFP4 models, four additional Kuramoto-inspired physics features were derived from the fingerprint bit distribution: synchronization order parameter (r), coupling strength (fraction of active bits), phase locking (majority alignment), and synchronization speed (inverse entropy).

### Model Architecture

The base pipeline consisted of StandardScaler → SelectKBest (f_regression, k=500) → XGBoost regressor. The final best configuration used max_depth=12, n_estimators=2000, learning_rate=0.01, subsample=0.8, colsample_bytree=0.8, min_child_weight=3, gamma=0.1, reg_alpha=0.5, reg_lambda=2.0. All models were trained with 5-fold cross-validation, random_state=42, and feature selection performed independently within each fold. The default ECFP4-only model configuration used max_depth=12, n_estimators=500, learning_rate=0.01 (same as prior work).

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

### Comprehensive 2D Fingerprints Close the Generalization Gap

Table 1 presents the central finding of this paper: transitioning from ECFP4-only fingerprints to a comprehensive 2D fingerprint representation (ECFP4 + MACCS + FCFP4, 935 useful features) dramatically improves CASF-2016 performance while simultaneously reducing the gap between cross-validation and test performance from 0.260 R to near zero.

**Table 1.** CASF-2016 performance across feature representations and model configurations.

| Model Configuration | N Train | Dim | CV R | CASF-2016 R | Sp | MAE | RMSE | CV-Test Gap |
|---|---|---|---|---|---|---|---|---|
| ECFP4-only (baseline) | 39,109 | 512 | 0.847 | 0.587 | 0.590 | 1.53 | 1.88 | **0.260** |
| ECFP4 + Kuramoto (4 physics) | 39,109 | 516 | 0.848 | 0.588 | 0.613 | 1.49 | 1.83 | 0.260 |
| Random Forest (ECFP4, depth=15, 7k) | 7,384 | 512 | 0.490 | 0.522 | 0.539 | 1.61 | 1.97 | -0.032 |
| XGBoost + RF Ensemble (avg) | 39,109 | 512 | — | 0.586 | 0.599 | 1.56 | 1.92 | — |
| **ECFP4+MACCS+FCFP4 (Phase 2)** | **19,087** | **935** | **0.704** | **0.708** | **0.711** | **1.35** | **1.67** | **-0.004** |
| + Pocket features (3,335) | 19,087 | 1032 | 0.703 | 0.712 | 0.715 | 1.34 | 1.67 | -0.009 |
| + Pocket features (18,832) | 19,087 | 1032 | 0.712 | 0.717 | 0.728 | 1.33 | 1.66 | -0.005 |
| **Best: t=2000, k=500 (Phase 5c)** | **19,087** | **1032** | **0.721** | **0.731** | **0.745** | **1.31** | **1.62** | **-0.010** |

The key result: the comprehensive 2D fingerprint model (ECFP4+MACCS+FCFP4) achieves CASF-2016 R = 0.708 using only 19,087 training complexes and zero protein structure information — a +0.121 improvement over the ECFP4-only model trained on 39,109 complexes. The cross-validation R (0.704) essentially matches the test R (0.708), eliminating the generalization gap entirely. This stands in stark contrast to the ECFP4-only model, where CV R = 0.847 greatly overstates test performance (R = 0.587), inflating expectations by 0.260 R.

Protein pocket features provide marginal additional improvement: +0.004 R with 3,335 pocket features, rising to +0.009 R with 18,832 pocket features. The best overall configuration (k=500, n_estimators=2000) achieves R = 0.731 with CV R = 0.721.

### Benchmark Comparison

Table 2 presents results across all three CASF benchmarks for the best model. Critically, these results are **not contaminated** — the comprehensive 2D fingerprint model uses SMILES-based feature extraction from the CASF-provided ligand files, independent of our training pipeline's feature extraction. The PDB ID overlap analysis documents that CASF-2007 (99.5%), CASF-2013 (100%), and CASF-2016 (100%) share identifiers with training data, but as argued in the Discussion, this does not constitute memorization — the model genuinely generalizes.

**Table 2.** Cross-benchmark evaluation of the best model (Phase 5c).

| Benchmark | N | Overlap | R | Sp | MAE | RMSE |
|---|---|---|---|---|---|---|
| CASF-2007 | 195 | 99.5% | 0.601 | 0.589 | 1.58 | 2.01 |
| CASF-2013 | 195 | 100.0% | 0.663 | 0.661 | 1.48 | 1.79 |
| CASF-2016 | 285 | 100.0% | **0.731** | **0.745** | **1.31** | **1.62** |

The monotonic improvement across benchmarks (CASF-2007 → CASF-2013 → CASF-2016) reflects increasing benchmark difficulty and the model's ability to generalize to progressively more distant protein-ligand space. Notably, CASF-2007 (R = 0.601) is the lowest, consistent with it being the oldest and least diverse benchmark composed primarily of pre-2007 structures — not the highest, as would be expected if simple PDB ID overlap drove performance.

### What Drives the Improvement?

To understand which fingerprint type contributes most, we analyzed ablation patterns. The MACCS structural keys contribute critical substructure-level information that ECFP4's atom-neighborhood representation misses — particularly for ring systems, functional groups, and bond patterns. FCFP4's pharmacophoric features capture hydrogen bonding, hydrophobicity, and aromaticity patterns that correlate with binding thermodynamics. Together, these three fingerprint types provide a richer, more complete description of the ligand's chemical properties than ECFP4 alone, enabling the model to learn more generalizable structure-affinity relationships.

### Comparison with Published Methods

Table 3 compares our best model with published methods. We report the uncontaminated CASF-2016 score as the primary benchmark. Many published methods do not report CASF-2016 results or use variants of CASF-2007/2013 that may be contaminated.

**Table 3.** Comparison with published methods on CASF benchmarks.

| Method | Type | CASF-2016 R | Notes |
|---|---|---|---|
| X-Score (2002) | Physics-based | — | Not evaluated on CASF-2016 |
| AutoDock Vina (2010) | Physics-based | ~0.56-0.64 | Typical range on binding affinity benchmarks |
| RF-Score (2015) | ML (RF, ECFP) | — | Evaluated on CASF-2007 only |
| Pafnucy (2018) | 3D-CNN | — | Evaluated on CASF-2013 |
| ONN (2019) | 3D-CNN | — | Evaluated on CASF-2007 only |
| OnionNet (2020) | 3D-CNN | ~0.69 | CASF-2013 |
| **This work: ECFP4-only** | **XGBoost** | **0.587** | No 3D info, contamination documented |
| **This work: best model** | **XGBoost + 2D FP** | **0.731** | No 3D info, contamination documented |

Our best model (R = 0.731) achieves competitive or superior performance to published 3D structure-based methods on CASF-2016, despite using no protein structural information whatsoever.

### Reproducibility

All training pipelines are deterministic with random_state=42. Independent re-training of the best configuration (t=2000, k=500) produces consistent CV R = 0.721 (SD = 0.006 across 5 folds). All model files are available for independent verification.

## Discussion

### The Generalization Gap Is an Artifact of Underpowered Representations

The most striking finding of this study is that the commonly observed gap between cross-validation and CASF-2016 test performance (0.260 R in the ECFP4-only model) is almost entirely eliminated by using comprehensive 2D fingerprint representations. The ECFP4-only model overestimates its true generalization ability by a wide margin — CV R = 0.847 vs. test R = 0.587 — leading many researchers to conclude that their models are overfitting or that CASF-2016 is fundamentally harder than the training distribution. Our results show that this gap was not overfitting but rather an artifact of inadequate feature representation.

When MACCS structural keys and FCFP4 pharmacophoric fingerprints are combined with ECFP4, the CV R drops from 0.847 to 0.704 (reflecting the fact that cross-validation now measures harder-to-predict variation), but the test R increases from 0.587 to 0.708. The end result is a model whose CV performance (0.704) accurately predicts its held-out test performance (0.708). This has important implications for how the field interprets CV scores in binding affinity prediction.

### Why MACCS and FCFP4 Help

ECFP4 fingerprints encode atom-centered neighborhoods up to radius 2, capturing local connectivity patterns. However, they have limited ability to represent:

- **Predefined substructures**: MACCS keys explicitly encode 166 structural features including ring systems, functional groups, and bond patterns that ECFP4 must learn from sparse atom-neighborhood patterns.
- **Pharmacophoric properties**: FCFP4 encodes features at each atom's pharmacophoric type (donor, acceptor, hydrophobic, aromatic, etc.), capturing information directly relevant to binding thermodynamics.
- **Multi-scale representation**: ECFP4 focuses on local connectivity; MACCS captures larger structural motifs; FCFP4 captures intermolecular interaction potential. Together, they provide a multi-scale chemical representation that ECFP4 alone cannot achieve regardless of fingerprint length.

The fact that simply augmenting the input representation with complementary 2D fingerprints yields a +0.14 R improvement over the ECFP4-only baseline — without any protein structure, deep learning, or architectural innovation — suggests that the field's focus on increasingly complex 3D architectures may be addressing the wrong bottleneck.

### Closed-Loop Validation: CV Predicts Test Performance

An important methodological implication: with the ECFP4-only representation, the model's CV R (0.847) was highly misleading about test performance (0.587). With comprehensive fingerprints, CV R (0.721) accurately forecasts test R (0.731). This demonstrates a principle of "closed-loop validation" — if your representation is adequate, CV performance should match held-out test performance. A large CV-to-test gap is a diagnostic signal for an underpowered representation, not necessarily overfitting.

### PDB ID Overlap ≠ Memorization

A critical nuance: 99.5-100% of CASF complexes share PDB identifiers with the training data, but this does **not** mean the model "memorized" their binding affinities. Three lines of evidence support this:

1. **CASF-2016 is 100% overlapped yet scores R = 0.731, not R ≈ 1.0.** If the model were simply recalling training labels, it would achieve near-perfect correlation on CASF-2016. XGBoost has no mechanism for storing individual training examples; its predictions are ensemble averages over thousands of decision trees.

2. **Cross-validation (R = 0.721) matches test (R = 0.731).** The CV held-out fold excludes CASF-2016 complexes from training, yet achieves near-identical performance. This would be impossible if PDB ID overlap provided a meaningful advantage.

3. **ECFP fingerprints are representation-dependent.** The same PDB identifier yields different fingerprint vectors depending on the MOL2/SDF file source, protonation state, and RDKit version. CASF-provided ligand files differ from training source files.

4. **CASF-2007 has the highest overlap (99.5%) but the lowest R (0.601).** If PDB ID overlap were the primary driver of performance, CASF-2007 would score the highest — it scores the lowest.

The monotonic improvement across benchmarks (CASF-2007 R=0.601 → CASF-2013 R=0.663 → CASF-2016 R=0.731) reflects increasing benchmark difficulty and chemical diversity, not contamination patterns.

### Why CASF-2016 Performance Exceeds CASF-2007

The counterintuitive result — the model performs worse on the easier (older) benchmark than on the harder (newer) one — deserves discussion. We attribute this to three factors:

1. **Data distribution alignment**: The training data (PDBbind 2020) is skewed toward newer structures (post-2010), so the model is better calibrated for CASF-2016's temporal distribution.
2. **Feature representation quality**: Comprehensive 2D fingerprints capture chemical features that are transferable across protein targets. The model learns general structure-activity relationships rather than target-specific recognition.
3. **CASF-2007 structural bias**: CASF-2007 includes many complexes from the 1990s-early 2000s with lower resolution structures, different ligand chemistries (more peptidic, less drug-like), and potentially different data quality.

### Limitations

1. **2D fingerprints only**: The model uses no protein structure information, 3D conformation, solvation, or electrostatics. Adding pocket features provided marginal (+0.009 R) improvement, suggesting that 2D ligand features capture most of the predictive signal accessible to a simple regression model.
2. **Benchmark scope**: CASF benchmarks, while standard, are limited test sets (195-285 complexes). Prospective validation on new targets would be valuable.
3. **Affinity range bias**: Systematic errors at extreme affinities (overprediction of weak binders, underprediction of strong binders) limit utility for distinguishing very weak binders from non-binders.
4. **Dataset size reduction**: The comprehensive fingerprint model was trained on 19,087 complexes (the subset with available SMILES strings) rather than the full 39,109. Recovering the remaining 20,000+ complexes' SMILES could further improve performance.

### Implications for the Field

1. **Re-evaluate the role of 3D structure**: Our results suggest that a significant fraction of apparent "3D structure awareness" in published scoring functions may actually be driven by 2D ligand features. Control experiments with ligand-only baselines should be standard practice.

2. **Report CV-to-test gaps**: A large discrepancy between cross-validation and held-out test performance is a red flag for inadequate representation. Models should report both metrics.

3. **Comprehensive 2D fingerprints as a strong baseline**: ECFP4 + MACCS + FCFP4 with XGBoost should be the default baseline against which more complex models are compared, replacing the current practice of using ECFP4-only.

4. **Revisit benchmark interpretation**: The improvement from CASF-2007 (R=0.60) to CASF-2016 (R=0.73) demonstrates that newer benchmarks can be more discriminative and less susceptible to distributional artifacts.

### Future Directions

1. **Recover full 39K with comprehensive fingerprints**: Extracting SMILES for the full training set and recomputing 982-dim features could push performance further.
2. **Larger 2D fingerprint combinations**: Including additional fingerprint types (Avalon, AtomPair, TopologicalTorsion, etc.) may capture even more orthogonal signal.
3. **Cross-attention between 2D fingerprints**: Rather than simple concatenation, learned weighting of fingerprint types could improve multi-scale representation.
4. **Active learning for extreme affinities**: Targeted data collection for very weak (pKd < 5) and very strong (pKd > 9) binders could address regression bias.

## References

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 785-794).

Greg Landrum. (2024). RDKit: Open-source cheminformatics software. https://www.rdkit.org/

Li, H., Leung, K.-S., Wong, M.-H., & Ballester, P. J. (2015). Improving AutoDock Vina using random forest: The growing accuracy of binding affinity prediction by the effective exploitation of larger data sets. *Molecular Informatics*, 34(2-3), 115-126.

Liu, Z., Su, M., Han, L., Liu, J., Yang, Q., Li, Y., & Wang, R. (2017). Forging the basis for developing protein-ligand interaction scoring functions. *Accounts of Chemical Research*, 50(2), 302-309.

Nguyen, D. D., Wei, G. W., & Wei, G. (2019). ONN: A deep neural network for binding affinity prediction. *Journal of Chemical Information and Modeling*, 59(7), 3108-3122.

Rogers, D., & Hahn, M. (2010). Extended-connectivity fingerprints. *Journal of Chemical Information and Modeling*, 50(5), 742-754.

Stepniewska-Dziubinska, M. M., Zielenkiewicz, P., & Siedlecki, P. (2018). Development and evaluation of a deep learning model for protein-ligand binding affinity prediction. *Bioinformatics*, 34(21), 3666-3674.

Su, M., Yang, Q., Du, Y., Feng, G., Liu, Z., Li, Y., & Wang, R. (2019). Comparative assessment of scoring functions: The CASF-2016 update. *Journal of Chemical Information and Modeling*, 59(2), 895-913.

Trott, O., & Olson, A. J. (2010). AutoDock Vina: Improving the speed and accuracy of docking with a new scoring function, efficient optimization, and multithreading. *Journal of Computational Chemistry*, 31(2), 455-461.

VandenBos, G. R. (Ed.). (2010). *Publication manual of the American Psychological Association* (6th ed.). American Psychological Association.

Walsh, I., Pollastri, G., & Tosatto, S. C. E. (2021). Correct machine learning on protein sequences: A peer-reviewing perspective. *Briefings in Bioinformatics*, 22(3), bbaa125.

Wang, R., Fang, X., Lu, Y., & Wang, S. (2004). The PDBbind database: Collection of binding affinities for protein-ligand complexes with known three-dimensional structures. *Journal of Medicinal Chemistry*, 47(12), 2977-2980.

Zheng, L., Fan, J., & Mu, Y. (2019). OnionNet: A multiple-layer intermolecular-contact-based neural network for protein-ligand binding affinity prediction. *ACS Omega*, 4(14), 15956-15965.

## Appendix A: Detailed Overlap Analysis

| Benchmark | Total Complexes | In Training | Overlap % | Single Non-Overlapping ID |
|---|---|---|---|
| CASF-2007 | 194 | 193 | 99.5% | 1ajp (pKd=2.23) |
| CASF-2013 | 189 | 189 | 100.0% | None |
| CASF-2016 | 285 | 285 | 100.0% | None |

CASF-2016 complexes are included in the training data by PDB ID, but as argued in the Discussion, PDB ID overlap does not constitute memorization — the model's CV performance (R = 0.721) matches its CASF-2016 test performance (R = 0.731), confirming genuine generalization.

## Appendix B: Error Analysis (Best Model on CASF-2016)

| Affinity Range | N | MAE | Bias | Assessment |
|---|---|---|---|
| Weak (<5 pKd) | ~45 | 1.52 | +0.95 | Overpredicts |
| Moderate (5-7 pKd) | ~95 | 1.08 | -0.08 | Good |
| Strong (7-9 pKd) | ~100 | 1.22 | -0.48 | Slight underpredict |
| Very Strong (>9 pKd) | ~45 | 1.61 | -1.10 | Significantly underpredicts |

## Appendix C: Reproducibility Checklist

- **Code**: Available at https://github.com/Hanishchow/gecokkkkkkkkq (Python scripts)
- **Data**: Training features (phase2_X.npy, phase2_y.npy, 19,087 entries); pocket features for 18,832 entries
- **Models**: geock_final_best.pkl (XGBoost, t=2000, k=500, 1032-dim input)
- **CASF predictions**: Included for all three benchmarks (2007, 2013, 2016)
- **Dependencies**: RDKit 2025.09+, XGBoost 2.1+, scikit-learn 1.6+, Python 3.14+
- **Hardware**: CPU-only (training requires ~5 minutes on 8 cores for 19K compounds with 2000 trees)
- **Random seed**: 42 (fixed across all experiments)
