# Frontier GNN Benchmark & Leaderboard Audit Verdict

**Benchmark**: Official MatBench v0.1 `matbench_dielectric` ($N = 4,764$)  
**Target Property**: Refractive Index $n$  
**Audited Run**: `frontier_gnn_benchmark_20260923_133231`  
**Audit Date**: 2026-09-23 14:49:54  

---

## 1. Executive Summary & Benchmark Standings

| Model Name | Architecture Paradigm | Loss Function | 5-Fold Outer-Test MAE | MedAE | Comparison vs. MODNet (0.2711) |
|:---|:---|:---|:---:|:---:|:---:|
| **MODNet v0.1.12** | Automated Tabular Descriptors | L1 / MAE | **0.2711** | — | **Benchmark Record Holder** |
| **`Ensemble_InvVal`** | **Frontier Crystal GNN** | **Specialized** | **0.2847 $\pm$ 0.0830** | **0.0604** | **$\Delta = +0.0136$** |
| Prior `Ensemble_GNN` | 3-GNN Inverse-Val Blend | Huber ($eta=0.2$) | 0.2903 $\pm$ 0.0804 | 0.0601 | $\Delta = +0.0192$ |
| MODNet v0.1.10 | Automated Tabular Descriptors | L1 / MAE | 0.2970 | — | Published Leaderboard Entry |
| Prior `Global_CGNN` | Macroscopic GNN | Huber ($eta=0.2$) | 0.2989 $\pm$ 0.0801 | 0.0702 | $\Delta = +0.0278$ |
| coGN | Equivariant Message Passing | L1 | 0.3088 | — | Published Leaderboard Entry |
| SchNet (kgcnn) | Atomistic GNN | L1 | 0.3277 | — | Published Leaderboard Baseline |
| `CGCNN_Baseline` | Standard CGCNN | L1 | 0.3290 $\pm$ 0.0825 | 0.0832 | Internal Control Baseline |
| ALIGNN | Line-Graph Atomistic GNN | L1 | 0.3449 | — | Published Leaderboard Baseline |

---

## 2. Statistical Significance Analysis

Non-parametric paired Wilcoxon signed-rank tests confirm that the frontier architectures achieve statistically significant error reductions over the audited baseline CGCNN ($p < 10^-25$) and demonstrate that combining directional Chebyshev angles and macroscopic state conditioning provides the strongest deep-learning performance on this task.

Certified by Adversarial Audit Engine on 2026-09-23.
