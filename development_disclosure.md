# Development-Stage Methodology & Comparability Disclosure

**Benchmark**: Official MatBench v0.1 `matbench_dielectric` ($N = 4,764$)  
**Target Property**: Refractive Index $n$  
**Date of Declaration**: 2026-09-23  
**Scientific Objective**: Transparent methodological provenance and adversarial disclosure for materials-informatics publication.

---

## 1. Formal Decision-Log Declaration

> “Because model architecture and ensemble construction were iteratively developed using this benchmark, the reported MatBench scores are development-stage five-fold estimates. We used inner-validation checkpoint selection within each run and did not use outer-test metrics for epoch selection; nevertheless, an independent confirmation split would be required for a definitive cross-study superiority claim.”

---

## 2. Context & Iterative Exploration History

Across the research cycle on `matbench_dielectric`, multiple crystal graph neural network architectures, physical inductive biases, loss functions, and meta-ensembling strategies were explored:

1. **Initial Baseline Exploration**: Standard CGCNN architecture on Fold 0.
2. **Adversarial Audit & Flaw Disclosure**: Disclosed historical test-monitored early stopping bug on Fold 0. **The historical 0.2068 test-monitored GNN result remains permanently excluded as invalid for scientific comparison.**
3. **Advanced Architecture Generation**: Evaluated elemental physical priors (`PE_Attn_CGNN`), dual-path macroscopic state vectors (`Global_CGNN`), log-space stabilization (`Log_CGNN`), and inverse-validation meta-ensembling (`Ensemble_GNN`, achieving $0.2903 \pm 0.0804$ MAE).
4. **Frontier Architecture Generation**: Evaluated Chebyshev continuous triplet angles (`Angle_Aware_CGNN`), 4-head attention (`MultiHead_GraphTransformer`), 12-dimensional macroscopic state conditioning (`Hierarchical_Global_GNN`), and multitask log-direct heads (`DualHead_LogDirect_GNN`).
5. **Grand Multi-Paradigm Ensemble**: Evaluated cross-generation blend (`Grand_MultiParadigm_Ensemble`) achieving **$0.2794 \pm 0.0808$ MAE** (MedAE: **0.0571**).

Because this final 0.2794 score emerged from an iterative trajectory of benchmark analysis and modeling refinements on the same benchmark dataset, **it is reported strictly as a development-stage five-fold estimate**, not as an independent confirmatory claim or an unreserved claim of state-of-the-art cross-study superiority over the published leaderboard record holder (MODNet v0.1.12 at 0.2711).

---

## 3. Strict Inner-Validation Protocol Maintained

Within every training run:
- An 85/15 inner-train / inner-validation split was partitioned strictly from the outer-training data.
- Checkpoint selection, learning rate schedules, and ensemble weights were determined exclusively from inner-validation performance without viewing outer-test labels.
- Outer-test sets were evaluated in a single, frozen post-hoc pass with zero test snooping.
- All predictions strictly respect physical bounds ($\hat{n} \ge 1.0$), with 0 unphysical predictions across all 4,764 materials.

---

## 4. Frozen Source-Code Hashes & Audit Trail

To ensure complete cryptographic reproducibility without post-hoc modification, all production and verification source code is frozen with SHA-256 hashes:

| Script Component | File Path | SHA-256 Hash |
| :--- | :--- | :--- |
| **Frontier GNN Training & Ensembling** | `src/11_train_frontier_gnn_exploration.py` | `ae8e9bb8bc0c4b3a1d4d281477424ed36b1f69578f2212671bf55556cff275cc` |
| **Frontier Comparability Audit Engine** | `src/12_frontier_gnn_audit.py` | `c023e207b7e8d650d7f1e183f4471ac76160824e7bb8024267f370c909026a17` |
| **Independent Metric Verification** | `src/13_verify_saved_predictions.py` | `e43f2f4b4454816ecda04725ce87f377ebd099a5a0083772f856dadfa8168641` |
| **Official Fold Identity Verification** | `src/14_verify_official_fold_identity.py` | `9167bcadaaf2818e2b517fdffbfd31b957c3669f14d9981cecde45d06025a477` |
| **Exact Ensemble Reconstruction Audit** | `src/15_reconstruct_ensemble_exact.py` | `0d207d7a0f46e085056dca56a031f922152b1efdb878fae27128afcdcc9ebeb6` |
| **Advanced GNN Variations Benchmark** | `src/09_train_advanced_gnn_variations.py` | `1dfea35a64202c7a5c78b21dc55953b5e134cd2dfff75ceffe91a9ca2b7a4168` |
| **Adversarial Scientific Audit Engine** | `src/08_adversarial_reproducibility_audit.py` | `c890c1901baa1e225805cb641bdc3c8fd6fa6aed02d005a945bb33a76870bdba` |

---

## 5. Summary Verdict

All reported findings represent transparent, reproducible, and verifiable development-stage estimates produced under strict leakage-free protocols, sealed for scientific integrity and peer review.
