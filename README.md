# Reproducible Crystal-Graph Learning for Refractive-Index Prediction and Chemical-System-Disjoint Evaluation in Inorganic Materials

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14%2B-ee4c2c.svg)](https://pytorch.org/)
[![Benchmark: MatBench](https://img.shields.io/badge/Benchmark-MatBench%20v0.1-brightgreen.svg)](https://matbench.materialsproject.org)
[![Reproducibility: Audited](https://img.shields.io/badge/Reproducibility-Cryptographically%20Audited-success.svg)](results/)

> **Target Venue:** *Computational Materials Science* (Elsevier)  
> **Author:** Daksh Kaila (`dkaila_be25@thapar.edu`)  
> **Affiliation:** Department of Computer Science and Engineering, Thapar Institute of Engineering and Technology, Patiala, Punjab 147004, India  
> **Repository:** [https://github.com/hubdk17/Crystal_Index](https://github.com/hubdk17/Crystal_Index)

---

## ⚖️ Original Research Notice & Mandatory Citation Requirement

> **IMPORTANT NOTICE:**  
> This repository contains **original, reproducible scientific research** in materials informatics, crystal graph neural networks, and benchmark auditing.  
> 
> If you utilize, adapt, or build upon this codebase, model architecture, featurization methods, precomputed representations, trained checkpoints, or benchmark splits in any academic publication, benchmark study, preprint, thesis, or software project, **you are required to cite the original work**:
> 
> ```bibtex
> @article{kaila2026reproducible,
>   author    = {Kaila, Daksh},
>   title     = {Reproducible Crystal-Graph Learning for Refractive-Index Prediction and Chemical-System-Disjoint Evaluation in Inorganic Materials},
>   journal   = {Computational Materials Science},
>   publisher = {Elsevier},
>   year      = {2026},
>   url       = {https://github.com/hubdk17/Crystal_Index}
> }
> ```
> See [`CITATION.cff`](CITATION.cff) and [`LICENSE`](LICENSE) for formal licensing and attribution terms.

---

## 📌 Executive Summary (BLUF)

1. **Leakage Audit & Flaw Invalidation:** We audit historical benchmark training protocols on the MatBench dielectric task ($N = 4{,}764$), demonstrating that a previously logged $0.2068$ MAE on Fold 0 was an artifact of test-set monitoring during checkpoint selection. Under strict fold-local preprocessing and inner-validation checkpointing, the true audited CGCNN 5-fold mean is $0.3298 \pm 0.0800$ MAE.
2. **Classical Baseline Outperforms Standard GNN:** An audited RBF-SVR pipeline operating on 153 Magpie, space-group, and Sine Coulomb descriptors achieves an MAE of $0.3124 \pm 0.0812$, outperforming standard CGCNN while training ~61× faster on identical hardware.
3. **DualHead–LogDirect GNN:** We introduce a hierarchical crystal-graph architecture integrating:
   - 8 tabulated elemental ground-state physical priors ($X, r, m, g, \text{row}, \text{IE}, V_{\mathrm{mol}}, \text{ox}_{\max}$).
   - 58-dimensional edge channels (41 radial RBFs, inverse distance, and 16 Chebyshev 3-body bond-angle projections).
   - 12-dimensional macroscopic structural–compositional global state $\mathbf{u}$.
   - Bidirectional 3-way interaction layers.
   - Dual bounded readout heads (direct Softplus + logarithmic Smooth L1 loss).
4. **Official MatBench Accuracy:** Attains a development-stage 5-fold MAE of **$0.2909 \pm 0.0860$**, $\text{MedAE} = 0.0631$, and Spearman rank correlation of **$\rho = 0.9395$**.
5. **Chemical-System-Disjoint Stress Test:** When retrained from scratch under 5-fold `GroupKFold` across **3,169 non-overlapping chemical systems** (0% train/test elemental system overlap), the frozen specification achieves **$0.2973 \pm 0.0237$ MAE** (only a 2.2% higher MAE than random splits) with **$\rho = 0.9331$**, demonstrating robust rank-order transferability.
6. **Physical Bound & Tail Heteroskedasticity:** Blended inference strictly guarantees physical lower bounds ($\hat{n} \ge 1.0$ for 100% of out-of-fold predictions; minimum observed $\hat{n} = 1.0988$). Residual diagnostics reveal tight precision across common insulators ($n \le 3.0$, $\text{MedAE} = 0.0510$) alongside severe heteroskedastic scale underprediction in the extreme tail ($n > 8.0$).

---

## 📊 Benchmark Results Overview

### Table 1: Official MatBench 5-Fold Evaluation ($N = 4{,}764$)
| Model Architecture | 5-Fold MAE | MedAE | RMSE | $R^2$ | Spearman $\rho$ | Fit Time / Fold |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **DualHead–LogDirect GNN (Ours)** | **0.2909 $\pm$ 0.0860** | **0.0631 $\pm$ 0.0023** | **1.7066 $\pm$ 0.9298** | **0.3061 $\pm$ 0.1770** | **0.9395 $\pm$ 0.0101** | ~111s |
| Global\_CGNN (Internal Comparator) | 0.2989 $\pm$ 0.0805 | 0.0702 $\pm$ 0.0035 | 1.7089 $\pm$ 0.9304 | 0.2995 $\pm$ 0.1882 | 0.9356 $\pm$ 0.0125 | ~95s |
| **RBF-SVR (Classical Baseline)** | **0.3124 $\pm$ 0.0812** | 0.0990 $\pm$ 0.0034 | 1.6863 $\pm$ 0.9332 | 0.3250 $\pm$ 0.2328 | 0.9264 $\pm$ 0.0144 | **3.6s** |
| CGCNN Baseline (Audited) | 0.3298 $\pm$ 0.0800 | 0.0858 $\pm$ 0.0055 | 1.7291 $\pm$ 0.9109 | 0.2826 $\pm$ 0.2090 | 0.9156 $\pm$ 0.0209 | ~219s |
| Polynomial SVR ($d=3$) | 0.3407 $\pm$ 0.0779 | 0.1065 $\pm$ 0.0031 | 1.7496 $\pm$ 0.9131 | 0.2627 $\pm$ 0.2174 | 0.9149 $\pm$ 0.0159 | 2.8s |
| Linear SVR | 0.3756 $\pm$ 0.0807 | 0.1298 $\pm$ 0.0055 | 1.7546 $\pm$ 0.8869 | 0.2528 $\pm$ 0.1726 | 0.9027 $\pm$ 0.0203 | 44.9s |
| Random Forest | 0.4331 $\pm$ 0.0510 | 0.1219 $\pm$ 0.0118 | 1.9065 $\pm$ 0.7731 | 0.0481 $\pm$ 0.1257 | 0.8890 $\pm$ 0.0211 | 3.6s |
| Ridge Regression | 0.5284 $\pm$ 0.0575 | 0.2831 $\pm$ 0.0282 | 1.7712 $\pm$ 0.8602 | 0.2288 $\pm$ 0.1466 | 0.8152 $\pm$ 0.0187 | 0.2s |

### Table 2: Chemical-System-Disjoint Grouped Evaluation (`GroupKFold`, 3,169 Systems)
| Fold Index | Test Samples | Mean $n$ | Max $n$ | $n > 7$ Count | MAE | MedAE | RMSE | $R^2$ | Spearman $\rho$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Fold 0 | 953 | 2.3805 | 33.47 | 10 | 0.3006 | 0.0666 | 1.4871 | 0.2672 | 0.9088 |
| Fold 1 | 953 | 2.3961 | 53.79 | 6 | 0.2855 | 0.0710 | 1.9284 | 0.1418 | 0.9321 |
| Fold 2 | 953 | 2.4533 | 55.89 | 13 | 0.3097 | 0.0649 | 2.0642 | 0.1557 | 0.9412 |
| Fold 3 | 953 | 2.4857 | 32.97 | 15 | 0.3266 | 0.0713 | 1.8664 | 0.2032 | 0.9392 |
| Fold 4 | 952 | 2.4259 | 62.06 | 8 | 0.2643 | 0.0722 | 2.0802 | 0.1844 | 0.9440 |
| **Fold Mean $\pm$ SD** | **952.8** | **2.4283** | **47.63** | **10.4** | **0.2973 $\pm$ 0.0237** | **0.0692 $\pm$ 0.0032** | **1.8853 $\pm$ 0.2402** | **0.1905 $\pm$ 0.0487** | **0.9331 $\pm$ 0.0143** |
| **Pooled OOF** | **4,764** | **2.4283** | **62.06** | **52** | **0.2974** | **0.0692** | **1.8974** | **0.1848** | **0.9331** |

---

## 📁 Repository Structure

```
Crystal_Index/
├── CITATION.cff             # Formal citation metadata
├── LICENSE                  # MIT License with academic attribution clause
├── README.md                # This comprehensive documentation
├── requirements.txt         # Pinned Python package dependencies
├── development_disclosure.md # Full audit trail and history of early iterations
├── data/                    # Dataset manifests, fold splits, raw MatBench cache
│   ├── raw/                 # Downloaded MatBench JSON data
│   └── processed/folds/     # Saved deterministic train/test fold splits
├── figures/                 # Publication-quality figures (PDF & PNG)
├── manuscript_2/            # Target submission materials for Computational Materials Science
│   ├── manuscript.tex       # Complete LaTeX source (elsarticle 3p format)
│   ├── supplementary_information.tex # Supplementary Notes 1-8 & Tables S1-S11
│   ├── references.bib       # BibTeX bibliography file
│   ├── highlights.txt       # Elsevier-compliant highlights (<= 85 chars each)
│   ├── graphical_abstract.png # 2:1 Landscape Graphical Abstract
│   └── figures/             # Figures included in the manuscript
├── models/                  # Saved checkpoint weights and configurations
├── results/                 # Out-of-fold predictions, audit logs, and metrics
└── src/                     # Complete executable reproduction scripts
    ├── 01_extract_and_explore_datasets.py
    ├── 02_featurize_materials.py
    ├── 03_dimensionality_reduction.py
    ├── 04_classical_baselines.py
    ├── 05_quantum_kernel_benchmark.py
    ├── 06_train_crystal_gnn.py
    ├── 07_thorough_validation_analysis.py
    ├── 08_adversarial_reproducibility_audit.py
    ├── 09_train_advanced_gnn_variations.py
    ├── 10_advanced_gnn_comparability_audit.py
    ├── 11_train_frontier_gnn_exploration.py
    ├── 12_frontier_gnn_audit.py
    ├── 13_verify_saved_predictions.py
    ├── 14_verify_official_fold_identity.py
    ├── 15_reconstruct_ensemble_exact.py
    ├── 16_train_ood_confirmation.py
    ├── generate_figure1_clean_bw.py
    └── generate_graphical_abstract.py
```

---

## 🚀 Installation & Quick-Start

### 1. Prerequisites & Environment Setup
We recommend using Python 3.10+ in a dedicated virtual environment:

```bash
# Clone the repository
git clone https://github.com/hubdk17/Crystal_Index.git
cd Crystal_Index

# Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🔬 Reproducibility Verification Commands

### Step 1: Verify All Saved Out-of-Fold Predictions
Recalculates exact fold-wise and pooled metrics across all 4,764 materials and verifies SHA-256 signatures:
```bash
python src/13_verify_saved_predictions.py
```

### Step 2: Verify Official MatBench Partition Identity
Verifies that stored train/test splits correspond bit-for-bit to official MatBench v0.1 API partitions with zero overlap:
```bash
python src/14_verify_official_fold_identity.py
```

### Step 3: Verify Inner-Validation Ensemble Weighting
Reconstructs the inverse-validation and softmax ensemble weighting strictly from inner-validation folds:
```bash
python src/15_reconstruct_ensemble_exact.py
```

### Step 4: Re-run Chemical-System-Disjoint Retraining
Executes deterministic CPU retraining of the frozen `DualHead_LogDirect_GNN` specification across all 5 chemical-system-disjoint `GroupKFold` splits:
```bash
python src/16_train_ood_confirmation.py
```

---

## 📜 License & Academic Citation

This project is licensed under the **MIT License with Academic Attribution Requirement** — see the [`LICENSE`](LICENSE) file for details.

If this work contributes to your research, please cite:

```bibtex
@article{kaila2026reproducible,
  author    = {Kaila, Daksh},
  title     = {Reproducible Crystal-Graph Learning for Refractive-Index Prediction and Chemical-System-Disjoint Evaluation in Inorganic Materials},
  journal   = {Computational Materials Science},
  publisher = {Elsevier},
  year      = {2026},
  url       = {https://github.com/hubdk17/Crystal_Index}
}
```

For questions, collaborations, or reproduction support, please contact:  
**Daksh Kaila** (`dkaila_be25@thapar.edu`)  
Department of Computer Science and Engineering, Thapar Institute of Engineering and Technology, Patiala, Punjab, India.
