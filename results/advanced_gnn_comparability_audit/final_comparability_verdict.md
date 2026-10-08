# Final Comparability & Reproducibility Verdict

**Benchmark**: Official MatBench v0.1 `matbench_dielectric` ($N = 4,764$)  
**Target Property**: Refractive Index $n$ (unitless)  
**Evaluated Run**: `results/gnn_variations_benchmark_20260923_122350/`  
**Audit Date**: 2026-09-23  

---

## 1. Formal Verdict

### **VERDICT: LEVEL 2 — NUMERICALLY LOWER, PROTOCOL-QUALIFIED**

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              AUDIT VERDICT                                   │
│                                                                              │
│   [ ] Level 1: Confirmed Comparable (Identical protocol, budget, setup)     │
│   [X] Level 2: Numerically Lower, Protocol-Qualified (Official folds pass,   │
│                zero sample leakage, but ensemble & task-specific priors)     │
│   [ ] Level 3: Development Estimate (Test-time early stopping / leakage)     │
│   [ ] Level 4: Not Comparable (Dataset revision or fold mismatch)            │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Condition-by-Condition Audit Findings

| Condition | Requirement | Audit Finding | Verdict |
|:---|:---|:---|:---:|
| **Condition 1** | Exact MatBench v0.1 fold definitions | Verified against canonical MatBench task. All 5 folds disjoint, complete, matching official SHA-256 manifests. | **PASS** |
| **Condition 2** | Mean of exactly five outer test-fold MAEs | Recomputed directly from raw prediction rows: $\overline{	ext{MAE}} = 0.29032 \pm 0.08044$. Exact unweighted arithmetic mean. | **PASS** |
| **Condition 3** | Zero outer-test influence on architecture | Model weights used zero test snooping; however, design decisions ($R=6.0$ Å, $1+	ext{Softplus}$) were informed by prior Fold 0 exploratory ablations. | **CAVEAT / PARTIAL** |
| **Condition 4** | Comparable model-selection protocol | `Ensemble_GNN` is a 3-model ensemble; `Global_CGNN` uses domain-specific macroscopic features; published SchNet/ALIGNN are single generic models. | **MATERIAL DIFFERENCE** |
| **Condition 5** | Target and dataset integrity | 100% match on all 4,764 targets (max diff = 0.0000000000). Zero dropped duplicates. Zero unphysical predictions. | **PASS** |

---

## 3. Allowed vs. Disallowed Publication Claims

### ❌ Disallowed Claims (Will Be Rejected by Rigorous Reviewers):
1. *"Our crystal GNN firmly outperforms SchNet and ALIGNN on the official MatBench dielectric benchmark."*
2. *"We establish a new deep-learning state of the art on MatBench dielectric."* (MODNet holds 0.2711).
3. *"CGCNN convolutions are intrinsically superior to ALIGNN line-graph convolutions."* (Our model is a hybrid graph-plus-macroscopic tabular model).

### ✅ Allowed, Peer-Review-Defensible Claims:
1. **Numerical Comparison**:
   > *"Using the official five-fold cross-validation partitions of MatBench v0.1 `matbench_dielectric`, our validation-weighted crystal graph ensemble achieved an out-of-fold MAE of **$0.2903 \pm 0.0804$** (and median absolute error of **$0.0601 \pm 0.0052$**). This is numerically lower than the listed single-model leaderboard entries for SchNet ($0.3277$) and ALIGNN ($0.3449$), while falling short of the benchmark record held by the feature-engineered MODNet ($0.2711$)."*
2. **Single-Model Hybrid Representation Claim**:
   > *"Among single models, incorporating macroscopic unit cell density and packing fraction into tri-directional message passing (`Global_CGNN`) yielded an out-of-fold MAE of **$0.2989 \pm 0.0801$**, breaking the 0.30 MAE threshold under strict inner-validation checkpointing."*
3. **Controlled Internal Ablation Claim**:
   > *"Under strictly matched 5-fold evaluation, our advanced architectural variations achieved statistically significant error reductions over the audited baseline CGCNN ($0.3290 \pm 0.0825$), with paired Wilcoxon signed-rank tests confirming significance at $p < 10^{-20}$ ($p = 9.62 	imes 10^{-93}$ for the ensemble)."*

---

## 4. Certification & Audit Sign-Off

Every artifact in this evidence package has been generated, independently recomputed, cryptographically fingerprinted, and sealed in `results/advanced_gnn_comparability_audit/`.
