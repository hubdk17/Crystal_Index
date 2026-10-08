# Adversarial Scientific-Reproducibility Audit Report
**Benchmark**: Official MatBench v0.1 `matbench_dielectric` ($N = 4,764$)  
**Target Property**: Scalar Refractive Index $n$  
**Auditor Mode**: Adversarial Scientific-Reproducibility Auditor  
**Audit Directory**: `d:\Desktop\Material_science_qml\results\classical_gnn_audit_20260923_110016`  
**Audit Timestamp**: `20260923_110016`  
**Hardware & Environment**: Intel Core Ultra 9 285H (16 CPU Cores), 32 GB RAM, Python 3.11.9, PyTorch 2.14.0+cu126  

---

## Executive Summary & Verdict

| Audit Domain | Test Performed | Audit Status | Key Empirical Evidence |
|:---|:---|:---:|:---|
| **Part A: Dataset & Folds** | 5-Fold MatBench Partition Integrity | **PASS** | 100% disjoint ($|\mathcal{S}_{\text{train}} \cap \mathcal{S}_{\text{test}}| = 0$), completeness = 4,764. |
| **Part B: Feature & Target Leakage** | Schema audit, forward trace, label permutation | **PASS** | Shuffled labels collapse $R^2$ to -0.1840. Zero target terms in input features. |
| **Part C: Preprocessing & Checkpointing** | Scrutiny of historical code & inner validation | **CONDITIONAL / REVISED** | **Flagged historical test-monitoring**: prior GNN checked test set every epoch. Corrected via strict 85/15 inner-val. |
| **Part D: 5-Fold Official Evaluation** | 7 models over 5 official folds | **PASS** | RBF SVR: **0.3124 ± 0.0812**; CGCNN: **0.3298 ± 0.0800**. RBF-SVR achieves lower mean MAE and RMSE than GNN. |
| **Part E: Error Tail & Heteroskedasticity** | Binned error analysis ($n > 7$) & outlier audit | **PASS** | Sub-0.10 MedAE on typical bulk materials; heavy error growth on narrow-gap Penn semiconductors ($n > 7$). |
| **Part F: Generalization (OOD)** | GroupKFold by Chemical System | **PASS** | OOD Chemical System MAE = **0.3301 ± 0.0263** (measuring extrapolative transfer). |
| **Part G: Physical Interpretability** | Permutation attribution & physical baselines | **PASS** | Density + VPA captures 57.8% of performance gain. Physically plausible association, not causal mechanism. |
| **Part H: GNN Ablation Suite** | Cutoff, neighbor, layer, and bound ablations | **PASS** | Ablations tracked and reported without post-hoc selection. |
| **Part I: Candidate Screening** | Held-out non-overlapping candidate pool | **PASS** | Screened external candidates from piezoelectric dataset (zero overlap with MatBench test structures). |

---

## 1. Historical Modeling Flaws Candidly Disclosed

1. **Test-Monitored Checkpoint Selection in Prior GNN**:
   In `src/06_train_crystal_gnn.py` (lines 254–260), test MAE was evaluated inside the training loop and used to save `best_gnn_mae`. This allowed information leakage from test set performance into checkpoint selection. The historical 0.2068 Test MAE is officially re-classified as **`INVALID_FOR_COMPARISON`**.
   In our publication-grade run, checkpoints are saved strictly at the minimum **inner-validation MAE** (85/15 split of train fold) and outer-test evaluation occurs strictly **once**.
2. **Global Median Imputation in Historical Featurization**:
   In `src/02_featurize_materials.py` (lines 125–129), missing descriptor values were filled using the dataset-wide median. In our audited pipeline, `SimpleImputer` is encapsulated inside an `sklearn.pipeline.Pipeline` fit exclusively on outer-training folds.
3. **Statistical Significance Overclaim Withdrawn**:
   The claim of $p < 0.001$ significance of GNN over RBF SVR is **WITHDRAWN**. Wilcoxon signed-rank test yields $W = 212,393.0, p = 0.0796$, demonstrating comparable mean error distributions between CGCNN and RBF SVR.

---

## 2. Official Five-Fold Cross-Validation Benchmark

Evaluated under strict inner-pipeline isolation across all 5 official MatBench folds:

| Model Architecture               | MAE (Mean ± SD)   | MedAE (Mean ± SD)   | RMSE (Mean ± SD)   | R² (Mean ± SD)   | Spearman ρ (Mean ± SD)   | Mean Fit Time (s)   |
|:---------------------------------|:------------------|:--------------------|:-------------------|:-----------------|:-------------------------|:--------------------|
| CGCNN-style Periodic Crystal GNN | 0.3298 ± 0.0800   | 0.0858 ± 0.0055     | 1.7291 ± 0.9109    | 0.2826 ± 0.2090  | 0.9156 ± 0.0209          | 219.3s              |
| Linear SVR                       | 0.3756 ± 0.0807   | 0.1298 ± 0.0055     | 1.7546 ± 0.8869    | 0.2528 ± 0.1726  | 0.9027 ± 0.0203          | 44.9s               |
| Poly SVR (d=3)                   | 0.3407 ± 0.0779   | 0.1065 ± 0.0031     | 1.7496 ± 0.9131    | 0.2627 ± 0.2174  | 0.9149 ± 0.0159          | 2.8s                |
| RBF SVR                          | 0.3124 ± 0.0812   | 0.0990 ± 0.0034     | 1.6863 ± 0.9332    | 0.3250 ± 0.2328  | 0.9264 ± 0.0144          | 3.6s                |
| Random Forest                    | 0.4331 ± 0.0510   | 0.1219 ± 0.0118     | 1.9065 ± 0.7731    | 0.0481 ± 0.1257  | 0.8890 ± 0.0211          | 3.6s                |
| Ridge                            | 0.5284 ± 0.0575   | 0.2831 ± 0.0282     | 1.7712 ± 0.8602    | 0.2288 ± 0.1466  | 0.8152 ± 0.0187          | 0.2s                |
| Train-Mean Predictor             | 0.8088 ± 0.0802   | 0.6079 ± 0.0160     | 1.9728 ± 0.8120    | -0.0042 ± 0.0080 | nan ± nan                | 0.0s                |

**Critical Scientific Audit Finding**:
Classical RBF-SVR achieves lower 5-fold mean MAE (0.3124 ± 0.0812) and lower RMSE (1.6863 ± 0.9332) than the CGCNN-style periodic crystal GNN (0.3298 ± 0.0800 MAE; 1.7291 ± 0.9109 RMSE), while training ~60× faster (3.6s vs 219s). The historical claim that GNN dominates classical ML is refuted.

---

## 3. High-Index Tail & Error Distribution

Refractive index in `matbench_dielectric` is heavily right-skewed (median = 2.06, max = 62.06).
When binned across target magnitudes:
- For standard optical materials ($n \le 3.0$, representing 70.1% of materials), both models achieve **sub-0.10 Median Absolute Error**.
- For narrow-gap semiconductors ($n > 7.0$, representing 1.2% of materials), errors grow significantly due to Penn gap divergence ($\varepsilon \approx 1 + (\hbar \omega_p / E_g)^2$).

---

## 4. Reproducibility Assurance

To replicate every metric, table, and figure from raw files, run the frozen PowerShell script:
```powershell
powershell -ExecutionPolicy Bypass -File .\reproduce.ps1
```
All seeds (seed=42), dependencies, and manifests are permanently preserved in this directory.
