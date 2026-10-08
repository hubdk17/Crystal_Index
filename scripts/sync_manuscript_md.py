import os

def main():
    out_path = r"C:\Users\HP\.gemini\antigravity-ide\brain\7e1462b3-34c9-4d76-a8c1-ddab45f77d0b\manuscript_draft.md"
    
    content = r"""# Reproducible Crystal-Graph Learning for Refractive-Index Prediction and Chemical-System-Disjoint Evaluation in Inorganic Materials

**Target Journal**: *Journal of Materials Informatics* (OAE Publishing, 2025 JIF: 8.0) / *Computational Materials Science*  
**Article Type**: Original Research Article  
**Author**: Daksh Kaila (Corresponding Author; Undergraduate Student, Department of Computer Science and Engineering, Thapar Institute of Engineering and Technology, Patiala, Punjab 147004, India; Email: `dkaila_be25@thapar.edu`)  
**Primary Publication Model**: Single Frozen DualHead--LogDirect GNN (`DualHead_LogDirect_GNN`)  
**Repository**: [GitHub Repository](https://github.com/hubdk17/Crystal_Index)  
**LaTeX Source**: [`manuscript/manuscript.tex`](file:///d:/Desktop/Material_science_qml/manuscript/manuscript.tex)  
**Supplementary Information**: [`manuscript/supplementary_information.tex`](file:///d:/Desktop/Material_science_qml/manuscript/supplementary_information.tex)  
**BibTeX References**: [`manuscript/references.bib`](file:///d:/Desktop/Material_science_qml/manuscript/references.bib)  

---

## Abstract

Predicting refractive index from crystal structure can accelerate first-pass screening of inorganic dielectric and optical materials, but reported benchmark scores can be distorted by test-set monitoring and may not describe performance on related but withheld chemistries. We audit descriptor-based and periodic graph models on the $N = 4{,}764$ `matbench_dielectric` task, explicitly excluding a historical test-monitored result. Under nested validation with fold-local preprocessing and inner-validation checkpoint selection, a radial basis function support vector regression (RBF-SVR) baseline achieved a five-fold MAE of $0.3124 \pm 0.0812$, outperforming an audited baseline crystal graph convolutional neural network ($0.3298 \pm 0.0800$). We then introduce DualHead--LogDirect GNN (`DualHead_LogDirect_GNN`), a hierarchical crystal-graph model that combines elemental-property priors, radial distance and angular edge features, a global crystal-state vector, and dual direct/logarithmic readout heads, attaining a development-stage five-fold MAE of $0.2909 \pm 0.0860$ and median absolute error of $0.0631 \pm 0.0023$. Under chemical-system-disjoint five-fold GroupKFold evaluation across 3,169 unique chemical systems with zero chemical-system overlap, the frozen architecture and training protocol achieved an MAE of $0.2973 \pm 0.0237$ (a 2.2% higher MAE than the official-fold development-stage estimate) and rank correlation $\rho = 0.9331$. The model achieves low error for the central portion of the benchmark distribution ($n \le 3.0$), whereas rare high-index entries dominate worst-case errors. These results provide a reproducible materials-informatics workflow that distinguishes model development, fold-local validation, and chemical-system-disjoint evaluation.

**Keywords**: Crystal Graph Neural Networks; Refractive Index; Materials Informatics; Chemical-System-Disjoint Evaluation; Benchmark Reproducibility; MatBench.

---

## 1. Introduction

The optical refractive index $n$ characterizes the phase velocity of light in a medium and is related to its electronic dielectric response [1]. Accurate estimation of $n$ plays a foundational role across optoelectronic engineering, from guiding optical coatings, photonic waveguides, dielectric metasurfaces, and other optical-material screening tasks [2, 3].

While density functional perturbation theory (DFPT) provides quantum-mechanical access to frequency-dependent dielectric tensors and high-frequency optical constants ($\bm{\varepsilon}_{\infty}$), DFPT dielectric calculations can be computationally demanding for high-throughput screening because their cost depends on cell size, electronic complexity, reciprocal-space sampling, and convergence requirements [1]. To accelerate materials exploration, data-driven machine learning (ML) surrogates trained on curated density functional theory (DFT) databases have emerged as an attractive alternative in materials informatics [4, 5, 6].

To establish standardized benchmarks, the MatBench v0.1 benchmark suite [3] formalized property prediction across diverse materials domains, including the `matbench_dielectric` task ($N = 4{,}764$). Over recent years, both advanced tabular pipelines featuring automated feature selection (e.g., MODNet [7]) and periodic crystal graph neural networks (GNNs) operating on atomic coordinates and unit cell lattices (e.g., CGCNN [5], MEGNet [6], SchNet [8], and ALIGNN [9]) have been benchmarked.

Despite substantial reported numerical progress, three critical methodological issues can arise in materials-ML workflows:
1. **Data Leakage and Test-Monitoring Conflation**: Complex deep architectures risk subtle test-set snooping when early stopping, checkpoint selection, or hyperparameter decisions are informed by outer-test loss trajectories [10]. Such inadvertent leakage can produce optimistically biased evaluation metrics and obscure expected performance on genuinely unseen data.
2. **Metric Incomparability and SOTA Overclaiming**: Iteratively refined architectures are often claimed to "firmly outperform" prior methods on standardized leaderboards. However, repeated architecture and hyperparameter exploration on a fixed benchmark can inflate apparent gains, while asymmetries in compute budgets, feature representations, and tuning schedules preclude controlled superiority claims.
3. **Standard Folds versus Chemical-System-Disjoint Evaluation**: The standard MatBench folds are not explicitly constrained to prevent related chemical systems from appearing across train and test partitions. In practical applications, materials-screening campaigns often target compositions whose exact elemental chemical systems are absent from the training data [11].

In this paper, we address these challenges through a transparent, reproducible investigation:
- **Audited Baselines**: We audit classical descriptor models and standard crystal graph networks on `matbench_dielectric`. We identify and exclude a historical result affected by test-monitored checkpoint selection, establishing an audited baseline under fold-local preprocessing and inner-validation checkpoint selection where classical RBF-SVR ($0.3124 \pm 0.0812$ MAE) outperforms baseline CGCNN ($0.3298 \pm 0.0800$ MAE).
- **Primary Single Model Formulation**: We introduce DualHead--LogDirect GNN (`DualHead_LogDirect_GNN`), a hierarchical crystal-graph model that combines elemental-property priors, radial distance and angular edge features, a global crystal-state vector, and dual direct/logarithmic readout heads. Under fold-local preprocessing and inner-validation checkpoint selection within the finalized evaluation run, it attains a development-stage five-fold MAE of $0.2909 \pm 0.0860$ and median absolute error (MedAE) of $0.0631 \pm 0.0023$.
- **Chemical-System-Disjoint Grouped Evaluation**: We evaluate the frozen model configuration under a five-fold `GroupKFold` partition across 3,169 unique chemical systems with zero chemical-system overlap. Retraining independently on each grouped partition from scratch, the model retains an MAE of $0.2973 \pm 0.0237$ (a 2.2% higher MAE than the official-fold development-stage estimate) and Spearman rank correlation $\rho = 0.9331$.
- **Distributional Error and Outlier Diagnostics**: We demonstrate that low error in the central target range ($n \le 3.0$, MedAE $= 0.0510$) coexists with large residuals in a sparse, high-$n$ tail, providing a realistic assessment of surrogate reliability.

---

## 2. Materials and Methods

### 2.1 Benchmark Dataset and Target Characteristics
We evaluate all models on the official MatBench v0.1 `matbench_dielectric` benchmark [3]. The dataset comprises $N = 4{,}764$ inorganic crystalline structures computed with density functional perturbation theory within the Materials Project [1, 12].

The target is the unitless scalar refractive-index value supplied by the MatBench dielectric task and derived from Materials Project dielectric calculations [1, 3]. It is a scalar benchmark representation of optical response; it does not resolve dielectric-tensor anisotropy, principal refractive indices, wavelength-dependent dispersion, or optical absorption. The target distribution is heavily right-skewed: the median refractive index is $2.071$, the mean is $2.428$, 95% of samples lie below $n = 4.35$, while extreme entries extend up to $n = 62.06$.

### 2.2 Fold-Local Evaluation and Checkpoint-Selection Protocol
To reduce within-run evaluation bias, we used the following fold-local protocol:
1. **Official Outer Five-Fold Partition**: We employ the exact five folds defined by the official MatBench v0.1 API ($N = 4{,}764$, with outer test folds of 953, 953, 953, 953, and 952 materials). Saved fold manifests were hashed for integrity and checked against the MatBench API splits for zero train--test overlap ($T_{f,\mathrm{train}} \cap T_{f,\mathrm{test}} = \emptyset$) and complete test coverage.
2. **Inner-Validation Checkpointing**: Outer training sets ($N \approx 3,811$) are partitioned into an 85% inner-train split and a 15% inner-validation split using deterministic pseudo-random seeds. Model optimization, learning rate scheduling, and checkpoint selection are monitored exclusively on inner-validation performance.
3. **Frozen Single-Pass Outer-Test Evaluation**: Within each finalized run, outer-test labels and metrics were not used for training, checkpoint selection, or hyperparameter selection. Checkpoints selected by inner validation were evaluated once on the unseen outer-test fold.

### 2.3 Classical Feature Engineering and Baseline Pipelines
To establish robust classical benchmarks, crystal structures were featurized into a 153-dimensional descriptor representation using `matminer` [13]:
- **Magpie Compositional Descriptors (132)**: Generated via `ElementProperty` with the Magpie preset [4], calculating stoichiometric statistics (mean, variance, range, minimum, maximum) across elemental atomic radii, electronegativities, valence counts, covalent radii, and ground-state properties.
- **Structural symmetry descriptors (12)**: Generated via `GlobalSymmetryFeatures` and structural metadata [14], comprising continuous space-group number ($1 \le SG \le 230$), one-hot crystal system categorical indicators (7 binary flags: cubic, hexagonal, trigonal, tetragonal, orthorhombic, monoclinic, triclinic), and Bravais lattice indicators.
- **Sine Coulomb Matrix Eigenvalues (9)**: Generated via `SineCoulombMatrix` [14], computing the top 9 eigenvalues of the periodic electrostatic interaction matrix accounting for unit-cell lattice periodicity and nuclear charge interactions.

All preprocessing transformations, including median missing-value imputation and feature standardization scalers, were fit strictly inside each inner-training fold to maintain fold-local preprocessing. Classical models evaluated include Ridge regression, Linear Support Vector Regression (SVR), Polynomial SVR ($d=3$), Radial Basis Function SVR (RBF-SVR), and Random Forest [15].

### 2.4 Primary Crystal Graph Architecture: DualHead--LogDirect GNN
To capture periodic crystal geometry directly, we construct crystal graphs as a periodic atomistic graph. For the primary DualHead--LogDirect GNN architecture (implemented as `DualHead_LogDirect_GNN`), graphs were constructed using a radial cutoff of $R_{\mathrm{cut}} = 6.0$ Å, truncated to the 12 nearest neighbors within the $R_{\mathrm{cut}} = 6.0$ Å cutoff sphere. Interatomic distances $r_{ij}$ were expanded using 41 Gaussian radial basis functions (RBF):
$$e_{ij, k} = \exp\left(-\gamma (r_{ij} - \mu_k)^2\right), \quad k \in \{1, \dots, 41\},$$
where $\mu_k$ are uniformly spaced centers from $0.0$ to $6.0$ Å with step $\Delta\mu = 0.15$ Å, and the Gaussian width parameter is $\gamma = 1/\sigma^2 \approx 44.44$ Å$^{-2}$ ($\sigma = 0.15$ Å). Distance expansions were complemented by an inverse distance channel $1/(r_{ij} + 0.01)$ and 16 Chebyshev polynomial features of the local bond-angle cosine $\cos\theta_{jik}$, averaged over all eligible neighbor triplets associated with each directed edge, yielding a 58-dimensional edge attribute vector $\mathbf{e}_{ij}$.

The DualHead--LogDirect GNN architecture incorporates three domain-specific inductive biases:
1. **Atomic Embeddings and Eight Tabulated Elemental Descriptors**: Atom nodes receive learnable embeddings from atomic number $Z \in \{1, \dots, 100\}$ (48 dimensions) concatenated with an 8-dimensional normalized elemental descriptor prior vector extracted from `pymatgen.core.Element` [14]: Pauling electronegativity ($X$), atomic radius ($r$), atomic mass ($m$), periodic table group ($g$), periodic table row ($\text{row}$), first ionization energy ($\text{IE}$), molar volume ($V_{\mathrm{mol}}$), and maximum oxidation state ($\text{ox}_{\max}$). All eight tabulated elemental descriptors are normalized using z-score standardization across $Z \in [1, 100]$.
2. **Hierarchical Convolutions, Global State Vector, and Message Aggregation**: The network employs 3 hierarchical interaction layers. In each layer, edge attributes $\mathbf{e}_{ij}$, atom node representations $\mathbf{h}_i$, and a 12-dimensional global structural--compositional descriptor vector $\mathbf{u}$ are updated via multi-layer perceptron (MLP) blocks equipped with LayerNorm and SiLU activations. Edge representations are updated as:
   $$\mathbf{e}_{ij}^{(l+1)} = \mathbf{e}_{ij}^{(l)} + \mathrm{MLP}_{\mathbf{e}}\left([\mathbf{h}_i^{(l)} \parallel \mathbf{h}_j^{(l)} \parallel \mathbf{e}_{ij}^{(l)} \parallel \mathbf{u}_{\mathrm{edge}}^{(l)}]\right).$$
   Messages were summed over retained neighbors without degree normalization:
   $$\mathbf{m}_i^{(l+1)} = \sum_{j \in \mathcal{N}(i)} \mathbf{e}_{ij}^{(l+1)}.$$
   Atom node representations are subsequently updated with residual connection:
   $$\mathbf{h}_i^{(l+1)} = \mathbf{h}_i^{(l)} + \mathrm{MLP}_{\mathbf{v}}\left([\mathbf{h}_i^{(l)} \parallel \mathbf{m}_i^{(l+1)} \parallel \mathbf{u}_{\mathrm{node}}^{(l)}]\right).$$
   The 12-dimensional global structural--compositional descriptor vector $\mathbf{u}$ combines density, cell geometry, packing, composition-derived elemental statistics, graph coordination, and lattice aspect ratio: (1) mass density $\rho$, (2) volume per atom $V/N$, (3) atomic packing fraction $\phi$, (4) element count $N_{\mathrm{elem}}$, (5) stoichiometric mean Pauling electronegativity $\bar{X}$, (6) electronegativity range $\Delta X$, (7) mean atomic radius $\bar{r}$, (8) mean periodic group number $\bar{g}$, (9) mean elemental molar volume $\bar{V}_{\mathrm{mol}}$, (10) average graph coordination number $\bar{z}$, (11) an engineered dimensionless packing descriptor $\eta_{\mathrm{pack}} = \phi (V/N)/\bar{r}^3$, and (12) lattice aspect ratio $c/a$.
3. **Dual Readout Heads and Multitask Objective**: Crystal-level pooling yields a latent vector $\mathbf{h}_{\mathrm{crystal}} = [\frac{1}{N_{\mathrm{atoms}}} \sum_i \mathbf{h}_i \parallel \mathbf{u}]$. The network jointly predicts:
   $$\hat{n}_{\mathrm{direct}} = 1.0 + \operatorname{Softplus}(\mathbf{W}_d \mathbf{h}_{\mathrm{crystal}} + b_d),$$
   $$\hat{z}_{\mathrm{log}} = \mathbf{W}_\ell \mathbf{h}_{\mathrm{crystal}} + b_\ell, \quad \text{modeling } z = \ln(n - 0.99).$$
   The network is optimized using a joint multitask Smooth L1 loss ($\beta = 0.15$):
   $$\mathcal{L} = \mathcal{L}_{\mathrm{SmoothL1}}(\hat{n}_{\mathrm{direct}}, y; \beta=0.15) + 0.5 \mathcal{L}_{\mathrm{SmoothL1}}(\hat{z}_{\mathrm{log}}, \ln(y - 0.99); \beta=0.15).$$
   The blended inference prediction is:
   $$\hat{n} = \frac{1}{2}\hat{n}_{\mathrm{direct}} + \frac{1}{2}\left(0.99 + \exp(\hat{z}_{\mathrm{log}})\right).$$

#### Architectural Novelty Relative to Existing Crystal GNNs
To clarify what is fundamentally new compared with canonical crystal graph neural networks—such as CGCNN [5], MEGNet [6], SchNet [8], and ALIGNN [9]—we highlight three key architectural distinctions:
1. **Tri-Channel Inductive Prior Featurization**: While standard crystal GNNs rely strictly on discrete atomic numbers $Z$ and scalar bond distances $r_{ij}$, DualHead--LogDirect GNN injects domain-specific physical priors across all three graph topological levels simultaneously: atom nodes receive 8 tabulated elemental ground-state physical priors; edges receive 58-dimensional representations combining RBF expansions, an inverse distance channel, and Chebyshev local bond-angle projections (capturing 3-body angular geometry without line-graph scaling); and a 12-dimensional global vector encapsulates macroscopic packing, density, and unit-cell aspect ratio.
2. **Hierarchical 3-Way Message Passing**: Unlike standard crystal convolutions where message passing is restricted to local atom--atom or atom--bond updates, our architecture implements bidirectional, three-way interaction layers. Edge representations are updated conditioned on incident node states and global state $\mathbf{u}$; node representations are updated conditioned on aggregated edge messages and global state $\mathbf{u}$; and crystal-level pooling explicitly retains $\mathbf{u}$. This enables macroscopic unit-cell constraints to modulate microscopic message propagation directly.
3. **Dual Bounded-Output Readout with Joint Direct--Log Multitask Loss**: Existing crystal GNNs employ a single unbounded linear readout head optimized with MSE or L1. For optical refractive index, this conventional design suffers from two catastrophic failure modes: generating unphysical sub-unity predictions ($n < 1.0$), and encountering severe numerical instability from the heavy right-skewed target distribution ($n$ up to $62.06$). DualHead--LogDirect GNN couples a direct readout head equipped with a Softplus barrier ($\hat{n}_{\mathrm{direct}} = 1.0 + \operatorname{Softplus}(z_d)$, guaranteeing $\hat{n}_{\mathrm{direct}} \ge 1.0$) with a logarithmic readout head ($\hat{z}_{\mathrm{log}} = \mathbf{W}_\ell \mathbf{h}_{\mathrm{crystal}} + b_\ell$, modeling $\ln(n - 0.99)$ under smooth L1 loss). This formulation simultaneously guarantees physical lower-bound integrity, compresses the dynamic range of high-index outliers, and balances gradient backpropagation between typical crystals and extreme optical materials.

![Figure 1: Architecture and Evaluation Workflow](C:\Users\HP\.gemini\antigravity-ide\brain\7e1462b3-34c9-4d76-a8c1-ddab45f77d0b\model_architecture_and_workflow.png)

*Figure 1: Architecture and evaluation workflow for DualHead--LogDirect GNN. (a) Periodic crystal structures are converted into atomistic graphs using a 6.0 Å cutoff and up to 12 neighbors. Node representations combine learned atomic-number embeddings with eight tabulated elemental descriptors. Edge attributes combine radial distance expansions, inverse distance, and averaged Chebyshev polynomial features of local bond angles. A 12-dimensional global structural--compositional descriptor vector is updated jointly with node and edge states through three hierarchical interaction layers. Final crystal representations are processed by direct and logarithmic readout heads, whose predictions are averaged. (b) Official MatBench evaluation uses fold-local preprocessing and inner-validation checkpoint selection; each outer test fold is evaluated once. The frozen model specification is then retrained under chemical-system-disjoint GroupKFold splits.*

---

## 3. Results and Discussion

### 3.1 Baseline Reproducibility Audit and Exclusion of Historical Flaw
In initial benchmarking iterations using a standard crystal graph convolutional neural network (CGCNN), an outer-test MAE of $0.2068$ was logged on Fold 0. Our internal reproducibility audit examined the original training implementation (`src/06_train_crystal_gnn.py`, lines 351--353) and identified that model checkpoints were updated by evaluating test-fold loss after each training epoch (`if avg_test_mae < best_test_mae: best_test_mae = avg_test_mae; torch.save(...)`) across 40 epochs. At epoch 37, the test-fold error reached an ephemeral minimum of $0.2068$ (recorded in `results/gnn_benchmark_results.json`). This protocol constitutes outer-test set snooping and post-selection leakage, conflating outer evaluation with optimization. Under our corrected, leakage-controlled protocol—which strictly reserves a 15% inner-validation split from the outer training set for checkpoint selection and evaluates the frozen outer-test fold exactly once—the audited Fold 0 MAE for baseline CGCNN is $0.2271$, and the five-fold mean MAE is $0.3298 \pm 0.0800$. The historical $0.2068$ result is formally classified as an artifact of test monitoring and excluded from scientific comparisons.

**Table 1: Official MatBench five-fold cross-validation results for baseline models under fold-local preprocessing and inner-validation checkpoint selection ($N = 4{,}764$).**

| Model Architecture | MAE | MedAE | RMSE | $R^2$ | Spearman $\rho$ | Fit Time / Fold |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **RBF-SVR (Classical)** | **0.3124 $\pm$ 0.0812** | 0.0990 $\pm$ 0.0034 | **1.6863 $\pm$ 0.9332** | **0.3250 $\pm$ 0.2328** | **0.9264 $\pm$ 0.0144** | **3.6s** |
| CGCNN Baseline | 0.3298 $\pm$ 0.0800 | **0.0858 $\pm$ 0.0055** | 1.7291 $\pm$ 0.9109 | 0.2826 $\pm$ 0.2090 | 0.9156 $\pm$ 0.0209 | 219.3s |
| Polynomial SVR ($d=3$) | 0.3407 $\pm$ 0.0779 | 0.1065 $\pm$ 0.0031 | 1.7496 $\pm$ 0.9131 | 0.2627 $\pm$ 0.2174 | 0.9149 $\pm$ 0.0159 | 2.8s |
| Linear SVR | 0.3756 $\pm$ 0.0807 | 0.1298 $\pm$ 0.0055 | 1.7546 $\pm$ 0.8869 | 0.2528 $\pm$ 0.1726 | 0.9027 $\pm$ 0.0203 | 44.9s |
| Random Forest | 0.4331 $\pm$ 0.0510 | 0.1219 $\pm$ 0.0118 | 1.9065 $\pm$ 0.7731 | 0.0481 $\pm$ 0.1257 | 0.8890 $\pm$ 0.0211 | 3.6s |
| Ridge Regression | 0.5284 $\pm$ 0.0575 | 0.2831 $\pm$ 0.0282 | 1.7712 $\pm$ 0.8602 | 0.2288 $\pm$ 0.1466 | 0.8152 $\pm$ 0.0187 | 0.2s |
| Dummy (Train-Mean) | 0.8088 $\pm$ 0.0802 | 0.6079 $\pm$ 0.0160 | 1.9728 $\pm$ 0.8120 | -0.0042 $\pm$ 0.0080 | N/A | 0.0s |

![Figure 2: Parity plots for audited baseline models](C:\Users\HP\.gemini\antigravity-ide\brain\7e1462b3-34c9-4d76-a8c1-ddab45f77d0b\publication_pooled_parity.png)

*Figure 2: Parity plots for audited baseline models on the official MatBench `matbench_dielectric` benchmark ($N = 4{,}764$). (a) Classical RBF-SVR using 153 Magpie, space-group, and Sine Coulomb matrix descriptors ($0.3124$ MAE). (b) Audited baseline CGCNN under strict inner-validation checkpoint selection ($0.3298$ MAE).*

### 3.2 Development-Stage MatBench Evaluation of DualHead--LogDirect GNN
DualHead--LogDirect GNN (`DualHead_LogDirect_GNN`) is a hierarchical crystal-graph model that combines elemental-property priors, radial distance and angular edge features, a global crystal-state vector, and dual direct/logarithmic readout heads. On the official MatBench folds, it achieved our best observed single-model performance, attaining a development-stage five-fold MAE of $0.2909 \pm 0.0860$, MedAE of $0.0631 \pm 0.0023$, RMSE of $1.7066 \pm 0.9298$, and Spearman rank correlation of $\rho = 0.9395 \pm 0.0101$.

**Table 2: Fold-by-fold outer-test MAE and paired differences on the official MatBench benchmark.**

| Model / Comparison | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Mean $\pm$ SD | Foldwise comparison with DualHead |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DualHead--LogDirect GNN (Ours)** | **0.1763** | **0.2557** | **0.4125** | **0.3022** | **0.3078** | **0.2909 $\pm$ 0.0860** | --- |
| CGCNN Baseline (Audited) | 0.2271 | 0.3037 | 0.4491 | 0.3412 | 0.3278 | 0.3298 $\pm$ 0.0800 | 0 lower-MAE folds; 5 higher-MAE folds |
| RBF-SVR (Classical Baseline) | 0.2163 | 0.2708 | 0.4356 | 0.3274 | 0.3121 | 0.3124 $\pm$ 0.0812 | 0 lower-MAE folds; 5 higher-MAE folds |
| Global_CGNN (Comparator) | 0.1945 | 0.2676 | 0.4154 | 0.3040 | 0.3128 | 0.2989 $\pm$ 0.0805 | 0 lower-MAE folds; 5 higher-MAE folds |
| $\Delta\mathrm{MAE}$ (DualHead vs. CGCNN Audited) | -0.0508 | -0.0480 | -0.0366 | -0.0390 | -0.0200 | -0.0389 $\pm$ 0.0121 | DualHead lower in 5/5 folds |
| $\Delta\mathrm{MAE}$ (DualHead vs. RBF-SVR) | -0.0400 | -0.0151 | -0.0231 | -0.0252 | -0.0043 | -0.0215 $\pm$ 0.0132 | DualHead lower in 5/5 folds |
| $\Delta\mathrm{MAE}$ (DualHead vs. Global_CGNN) | -0.0182 | -0.0119 | -0.0029 | -0.0018 | -0.0050 | -0.0080 $\pm$ 0.0069 | DualHead lower in 5/5 folds |

### 3.3 Systematic Component Ablation Study
To isolate the quantitative contribution of each architectural inductive bias, we conducted a systematic stepwise ablation study across identical official MatBench five-fold cross-validation splits ($N = 4{,}764$). Starting from the canonical periodic crystal graph baseline (`CGCNN_Baseline`), inductive biases were introduced incrementally: tabulated elemental property priors, logarithmic target parameterization, Chebyshev bond-angle features, global state conditioning, hierarchical three-way interaction, and dual bounded readout heads.

**Table 3: Systematic component ablation study on official MatBench five-fold cross-validation ($N = 4{,}764$). Negative $\Delta\mathrm{MAE}$ denotes error reduction relative to the canonical CGCNN baseline ($0.3290 \pm 0.0825$ MAE).**

| Model / Ablation Stage | Incremental Architectural Component | 5-Fold MAE | $\Delta\mathrm{MAE}$ vs. Base | MedAE | RMSE | Spearman $\rho$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 1. `CGCNN_Baseline` | Base periodic crystal graph (48 RBF, 8.0 Å, linear head) | 0.3290 $\pm$ 0.0825 | Ref. | 0.0832 | 1.7349 | 0.9174 |
| 2. `+ Property Embeddings` | Node concatenation with 8 elemental priors + attention | 0.3070 $\pm$ 0.0783 | -0.0220 | 0.0734 | 1.7157 | 0.9359 |
| 3. `+ Log Target Head` | Log-target parameterization $\ln(n - 0.99)$ with exponential readout | 0.3057 $\pm$ 0.0804 | -0.0233 | 0.0667 | 1.7264 | 0.9390 |
| 4. `+ Angular Geometry` | 16-dim Chebyshev local bond-angle projections on edges | 0.3004 $\pm$ 0.0848 | -0.0286 | 0.0666 | 1.7244 | 0.9289 |
| 5. `+ Global State Conditioning` | Conditioning layers on 4-dim structural state ($V/N, \rho, \bar{Z}, \bar{z}$) | 0.2989 $\pm$ 0.0801 | -0.0301 | 0.0702 | 1.7089 | 0.9356 |
| 6. `+ Hierarchical Message Passing` | 12-dim structural--compositional vector $\mathbf{u}$ + 3-way coupling | 0.2917 $\pm$ 0.0802 | -0.0373 | 0.0647 | 1.7183 | 0.9347 |
| 7. **DualHead--LogDirect GNN** | **Dual Softplus/Log heads + joint smooth L1 multitask loss** | **0.2909 $\pm$ 0.0860** | **-0.0381** | **0.0631** | **1.7066** | **0.9395** |

The systematic ablation results reveal three clear insights:
First, incorporating tabulated elemental property priors yields the single largest individual reduction in MAE ($\Delta\mathrm{MAE} = -0.0220$, a $6.7\%$ relative error drop), confirming that injecting atomic ground-state physics provides strong inductive bias that discrete atomic numbers alone cannot efficiently learn from modest training sets.
Second, geometric enrichment via local bond-angle Chebyshev expansions and 12-dimensional macroscopic global state conditioning reduces MAE by an additional $0.0153$ while pushing rank correlation to $\rho = 0.9347$, demonstrating that both 3-body coordination and macroscopic cell packing govern optical response.
Third, combining the direct Softplus head with the logarithmic head under joint smooth L1 loss establishes the lowest overall MAE ($0.2909 \pm 0.0860$), lowest typical error ($\mathrm{MedAE} = 0.0631$), and highest rank correlation ($\rho = 0.9395$), while strictly enforcing the physical bound $\hat{n} \ge 1.0$.

### 3.4 Contextual Comparison with MatBench Leaderboard Snapshot
Table 4 positions our results within the public MatBench leaderboard snapshot for `matbench_dielectric`.

**Table 4: Contextual, non-head-to-head comparison with selected MatBench leaderboard entries.**

| Model / Framework | Architecture Paradigm | Reported MAE | MAE difference relative to DualHead | Evaluation Nature |
| :--- | :--- | :---: | :---: | :--- |
| MODNet v0.1.12 [7] | Tabular Feature Selection + MLP | 0.2711 | -0.0198 | Lowest score among selected entries |
| **Grand Multi-Paradigm Ensemble (Ours)** | Multi-Paradigm Graph Blend | **0.2794 $\pm$ 0.0808** | -0.0115 | Supplementary exploratory blend |
| Frontier Ensemble (`Ensemble_Softmax`) | Graph Attention + Chebyshev | 0.2825 $\pm$ 0.0828 | -0.0084 | Multi-GNN meta-ensemble |
| Frontier Ensemble (`Ensemble_InvVal`) | Inverse-Val Weighted GNNs | 0.2847 $\pm$ 0.0830 | -0.0062 | Multi-GNN meta-ensemble |
| **DualHead--LogDirect GNN (Ours)** | **Single Multitask Crystal GNN** | **0.2909 $\pm$ 0.0860** | **0.0000** | **Primary frozen single model** |
| Hierarchical_Global_GNN (Ours) | 12-dim Global GNN | 0.2917 $\pm$ 0.0802 | +0.0008 | Single graph model |
| MODNet v0.1.10 [7] | Tabular Feature Selection + MLP | 0.2970 | +0.0061 | Published leaderboard entry |
| coGN | Equivariant Message Passing | 0.3088 | +0.0179 | Published leaderboard entry |
| RBF-SVR (Ours, Audited) | 153 Magpie + Symmetry Descriptors | 0.3124 $\pm$ 0.0812 | +0.0215 | Verified classical baseline |
| SchNet (kgcnn implementation) [8] | Atomistic Continuous Filters | 0.3277 | +0.0368 | Published leaderboard baseline |
| CGCNN Baseline (Ours, Audited) | Standard Gated Periodic GNN | 0.3298 $\pm$ 0.0800 | +0.0389 | Verified internal control |
| ALIGNN [9] | Line-Graph Atomistic GNN | 0.3449 | +0.0540 | Published leaderboard baseline |

### 3.5 Chemical-System-Disjoint Grouped Evaluation
To evaluate robustness when exact elemental chemical systems are absent from the training partition, we evaluated DualHead--LogDirect GNN under the 5-fold chemical-system-disjoint partition. For each grouped outer fold, we retrained the frozen DualHead--LogDirect GNN specification from scratch on that fold's training partition, using the same architecture, preprocessing, hyperparameters, and inner-validation checkpointing rule, without post hoc adjustment.

**Table 5: Detailed fold-by-fold evaluation and target distribution under chemical-system-disjoint 5-fold `GroupKFold` evaluation ($N = 4{,}764$, 3,169 unique chemical systems, 0% train/test system overlap).**

| Fold | Test $N$ | Median $n$ | Mean $n$ | P95 $n$ | Max $n$ | $n > 7$ | MAE | MedAE | RMSE | $R^2$ | Spearman $\rho$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 953 | 1.9858 | 2.3805 | 4.5649 | 33.47 | 10 | 0.3006 | 0.0666 | 1.4871 | 0.2672 | 0.9088 |
| 1 | 953 | 2.0400 | 2.3961 | 4.3409 | 53.79 | 6 | 0.2855 | 0.0710 | 1.9284 | 0.1418 | 0.9321 |
| 2 | 953 | 2.0679 | 2.4533 | 4.3987 | 55.89 | 13 | 0.3097 | 0.0649 | 2.0642 | 0.1557 | 0.9412 |
| 3 | 953 | 2.1071 | 2.4857 | 4.2762 | 32.97 | 15 | 0.3266 | 0.0713 | 1.8664 | 0.2032 | 0.9392 |
| 4 | 952 | 2.0769 | 2.4259 | 4.1945 | 62.06 | 8 | 0.2643 | 0.0722 | 2.0802 | 0.1844 | 0.9440 |
| **Fold mean $\pm$ SD** | **952.8** | **2.0555** | **2.4283** | **4.3550** | **47.63** | **10.4** | **0.2973 $\pm$ 0.0237** | **0.0692 $\pm$ 0.0032** | **1.8853 $\pm$ 0.2402** | **0.1905 $\pm$ 0.0487** | **0.9331 $\pm$ 0.0143** |
| **Pooled OOF** | **4,764** | **2.0515** | **2.4283** | **4.3547** | **62.06** | **52** | **0.2974** | **0.0692** | **1.8974** | **0.1848** | **0.9331** |

Under chemical-system-disjoint five-fold GroupKFold evaluation, the frozen DualHead--LogDirect GNN specification achieved $0.2973 \pm 0.0237$ MAE. This was a 2.2% higher MAE than the official-fold development-stage estimate ($0.2909 \pm 0.0860$), while rank correlation remained high ($\rho = 0.9331$).

![Figure 3: OOD Generalization Confirmation](C:\Users\HP\.gemini\antigravity-ide\brain\7e1462b3-34c9-4d76-a8c1-ddab45f77d0b\ood_generalization_confirmation.png)

*Figure 3: Chemical-system-disjoint grouped evaluation for DualHead--LogDirect GNN. (Left) Out-of-fold parity scatter plot across all 4,764 materials evaluated under five-fold GroupKFold splits with zero chemical-system overlap. (Right) Generalization error comparison between official MatBench folds ($0.2909 \pm 0.0860$ MAE) and chemical-system-disjoint folds ($0.2973 \pm 0.0237$ MAE), showing a 2.2% higher MAE than the official-fold development-stage estimate under the grouped split.*

### 3.6 Error Characterization and the High-Index Tail
The model's apparent strength is central-regime accuracy. Low MedAE values and high rank correlation indicate accurate rank ordering and low typical error within the central benchmark regime over much of this benchmark.

**Table 6: Binned prediction error analysis across target refractive index regimes on official out-of-fold predictions for DualHead--LogDirect GNN ($N = 4{,}764$). Bias denotes mean signed error ($\hat{n} - y$).**

| Target Regime ($n$) | Count | Fraction | MAE | MedAE | RMSE | P95 Error | Max AE | Mean Bias |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $n \le 2.0$ | 2,229 | 46.8% | 0.0893 | 0.0337 | 0.4187 | 0.1616 | 6.8470 | +0.0451 |
| $2.0 < n \le 3.0$ | 1,790 | 37.6% | 0.1344 | 0.0882 | 0.2049 | 0.3950 | 2.2537 | -0.0151 |
| **Combined $n \le 3.0$ (Central Regime)** | **4,019** | **84.4%** | **0.1094** | **0.0510** | **0.3405** | **0.3246** | **6.8470** | **+0.0183** |
| $3.0 < n \le 5.0$ | 609 | 12.8% | 0.3721 | 0.2329 | 0.5535 | 1.1338 | 3.0704 | -0.1276 |
| $5.0 < n \le 8.0$ | 92 | 1.9% | 1.3860 | 0.8651 | 1.8786 | 4.1852 | 4.8614 | -0.8443 |
| $n > 8.0$ (Extreme Tail) | 44 | 0.9% | 13.4575 | 7.5989 | 19.1842 | 49.2311 | 59.1997 | -13.4575 |
| **Overall Dataset** | **4,764** | **100.0%** | **0.2909** | **0.0637** | **1.8985** | **0.6758** | **59.1997** | **-0.1414** |

As documented in Table 6 and Figure 4, our residual analyses reveal distinct error behaviors across target regimes. In Figure 4a, absolute prediction errors remain tightly concentrated near zero across typical dielectric materials ($n \le 3.0$, encompassing 84.4% of the benchmark), where DualHead--LogDirect GNN achieves exceptional precision ($\text{MedAE} = 0.0510$, $\text{MAE} = 0.1094$, and 95th percentile absolute error of $0.3246$). Beyond $n > 5.0$, however, the binned median error (red curve) and scatter envelope expand rapidly, illustrating severe heteroskedastic variance. In Figure 4b, the out-of-fold residual distribution ($\hat{n} - y$) demonstrates that DualHead--LogDirect GNN produces a substantially sharper central peak around zero error than either RBF-SVR or baseline CGCNN, but exhibits a pronounced negative skew in the outer tail. Notably, all 44 materials in the $n > 8.0$ extreme-tail regime were underpredicted ($\hat{n} < y$ for 44 of 44 materials, 100.0%), directly accounting for the identity $\text{Mean Bias} = -\mathrm{MAE} = -13.4575$.

![Figure 4: Residuals and Heteroskedasticity Diagnostics](C:\Users\HP\.gemini\antigravity-ide\brain\7e1462b3-34c9-4d76-a8c1-ddab45f77d0b\publication_residuals_and_heteroskedasticity.png)

*Figure 4: Error diagnostics and heteroskedasticity for DualHead--LogDirect GNN across all $N = 4{,}764$ official out-of-fold predictions. (a) Absolute prediction error $|y - \hat{n}|$ versus DFT target refractive index $n$. Vertical shaded bands delineate the four target regimes ($n \le 3.0$, $3.0 < n \le 5.0$, $5.0 < n \le 8.0$, and $n > 8.0$); the red line traces binned median absolute error (MedAE), highlighting the onset of heteroskedastic error inflation beyond $n > 5.0$. (b) Out-of-fold residual distribution ($\hat{n} - y$) comparing DualHead--LogDirect GNN (purple filled density) against classical RBF-SVR (blue dashed) and baseline CGCNN (red dotted), demonstrating a sharper central peak around zero error alongside heavy non-Gaussian underprediction tails for rare high-index crystals.*

### 3.7 Bounded-Output Integrity
The DualHead--LogDirect GNN uses two output pathways:
$$\hat{n}_{\mathrm{direct}} = 1 + \operatorname{Softplus}(z_d),$$
which strictly guarantees $\hat{n}_{\mathrm{direct}} \ge 1$. In contrast, the logarithmic pathway:
$$\hat{n}_{\log} = 0.99 + \exp(z_\ell),$$
guarantees only $\hat{n}_{\log} > 0.99$. The final blended prediction is $\hat{n} = \frac{1}{2}(\hat{n}_{\mathrm{direct}} + \hat{n}_{\log})$. In empirical evaluation across all 4,764 held-out predictions on the official MatBench folds, the blended output satisfies $\hat{n} \ge 1.0$ for all 4,764 official-fold OOF predictions ($0.0\%$ unphysical rate), with an observed minimum blended prediction of $\min_i \hat{n}_i = 1.0988$.

### 3.8 Pooled Out-of-Fold Paired-Error Comparison (Descriptive)
To examine sample-level error differences, we conducted paired Wilcoxon signed-rank tests [18] on the pooled out-of-fold absolute errors across all 4,764 materials:
- DualHead--LogDirect GNN vs. Baseline CGCNN: $W = 3,421,800$, $p = 1.4 \times 10^{-48}$.
- DualHead--LogDirect GNN vs. Classical RBF-SVR: $W = 4,112,500$, $p = 4.2 \times 10^{-24}$.
- DualHead--LogDirect GNN vs. `Global_CGNN`: $W = 4,891,200$, $p = 1.1 \times 10^{-7}$.

In the pooled OOF comparison, absolute errors were lower for DualHead--LogDirect GNN than for the specified comparators. These p-values are descriptive because predictions are clustered by five outer-fold training contexts and the model family was developed iteratively on the benchmark. The foldwise win/loss metrics reported in Table 2 provide the primary evidence for consistent fold-level performance.

### 3.9 Frontier Ensembles and Exploratory Multi-Paradigm Blends
In addition to our primary single model, we investigated multi-model ensemble strategies combining diverse crystal GNN architectures developed on the benchmark:
1. **Inverse-Validation Weighted Ensemble (`Ensemble_InvVal`)**: For $M$ candidate crystal GNN architectures ($m \in \{1, \dots, M\}$), weights on each outer fold are assigned inversely proportional to their inner-validation MAE:
   $$w_m = \frac{(\mathrm{MAE}_{\mathrm{val}, m})^{-1}}{\sum_{j=1}^M (\mathrm{MAE}_{\mathrm{val}, j})^{-1}}, \quad \hat{n}_{\mathrm{InvVal}} = \sum_{m=1}^M w_m \hat{n}_m.$$
   By giving proportionally greater weight to models with superior held-out inner-validation generalization, `Ensemble_InvVal` achieved an official five-fold MAE of $0.2847 \pm 0.0830$ and MedAE of $0.0604 \pm 0.0037$.
2. **Softmax-Weighted Ensemble (`Ensemble_Softmax`)**: Applies a temperature-scaled Boltzmann distribution ($\tau = 10.0$) over negative inner-validation MAE:
   $$w_m = \frac{\exp(-\tau \cdot \mathrm{MAE}_{\mathrm{val}, m})}{\sum_{j=1}^M \exp(-\tau \cdot \mathrm{MAE}_{\mathrm{val}, j})}, \quad \hat{n}_{\mathrm{Softmax}} = \sum_{m=1}^M w_m \hat{n}_m.$$
   The temperature parameter $\tau$ exponentially penalizes underperforming models, sharpening ensemble selection and attaining an official five-fold MAE of $0.2825 \pm 0.0828$.
3. **Grand Multi-Paradigm Ensemble**: Combining five distinct model paradigms (`Ensemble_NNLS`, `Ensemble_InvVal`, `Hierarchical_Global_GNN`, `Global_CGNN`, and `Log_CGNN`), this exploratory blend attained our lowest observed development-stage MAE of $0.2794 \pm 0.0808$ (Fold 0: $0.1754$, Fold 1: $0.2475$, Fold 2: $0.3982$, Fold 3: $0.2886$, Fold 4: $0.2873$; MedAE: $0.0571$).

We present these ensemble results strictly in a supplementary, exploratory capacity. While multi-model ensembling achieves marginal MAE gains ($0.2794$--$0.2847$ vs. $0.2909$), ensembles substantially multiply inference latency and training cost, obscure which specific inductive representations drive predictive gains, and cannot be cleanly retrained under single-model chemical-system-disjoint evaluation. For primary reporting and prospective screening, the single frozen DualHead--LogDirect GNN ($0.2909 \pm 0.0860$ development MAE, $0.2973 \pm 0.0237$ chemical-system-disjoint MAE) represents our recommended balance of predictive accuracy, physical bound enforcement, computational efficiency, and reproducibility.

---

## 4. Conclusions

In this study, we conducted a rigorous benchmarking and reproducibility audit on the MatBench `matbench_dielectric` task ($N = 4{,}764$). We identified and excluded a historical test-monitored checkpointing flaw, establishing an audited baseline where classical RBF-SVR ($0.3124 \pm 0.0812$ MAE) outperforms baseline CGCNN ($0.3298 \pm 0.0800$ MAE). We then introduced DualHead--LogDirect GNN (`DualHead_LogDirect_GNN`), a hierarchical crystal-graph model that combines elemental-property priors, radial distance and angular edge features, a global crystal-state vector, and dual direct/logarithmic readout heads, achieving a development-stage five-fold MAE of $0.2909 \pm 0.0860$ and MedAE of $0.0631 \pm 0.0023$.

Under frozen chemical-system-disjoint five-fold GroupKFold evaluation across 3,169 unique chemical systems with zero overlap, DualHead--LogDirect GNN achieved $0.2973 \pm 0.0237$ MAE. This represents a 2.2% higher MAE than the official-fold development-stage estimate, while rank correlation remained robust at $\rho = 0.9331$. Detailed diagnostics show that low median error across typical crystals ($n \le 3.0$, $\text{MedAE} = 0.0510$) coexists with large residuals in a sparse, high-$n$ tail. These results highlight the value of distinguishing development-stage benchmark estimates from evaluations performed with a frozen model specification under chemical-system-disjoint splits.

Several limitations must be underscored: (1) The prediction target is a scalar benchmark reduction from DFT calculations and does not capture full optical dielectric anisotropy or wavelength-dependent dispersion. (2) The chemical-system-disjoint split prevents identical chemical systems from appearing across folds, but does not prevent shared elemental families or identical crystal prototypes from bridging the partitions; hence it does not establish broad generalization across novel materials classes. (3) Predicted screening candidates represent computational hypotheses that require independent first-principles DFPT and experimental verification. (4) The current study evaluates a single scalar optical target derived from DFT calculations and does not establish prospective accuracy for experimentally measured refractive indices, which can be influenced by temperature, defects, stoichiometry variations, and frequency-dependent dispersion.

---

## Declarations

### Authors' Contributions
Conceptualization, Methodology, Software, Internal Reproducibility Audit, Data Analysis, Visualization, Writing - Original Draft, and Writing - Review & Editing: D.K.

### Availability of Data and Materials
All benchmark crystal structures and target dielectric properties originate from the publicly available MatBench v0.1 repository (https://matbench.materialsproject.org). The reproduction suite and trained models are available at [GitHub Repository](https://github.com/hubdk17/Crystal_Index), with frozen cryptographic SHA-256 signatures.

### Financial Support and Sponsorship
This work was supported by computational resources at the Thapar Institute of Engineering and Technology. No external funding directly influenced the design, execution, or interpretation of this study.

### Conflicts of Interest
The author declares no competing financial or commercial conflicts of interest.

### Ethical Approval and Consent to Participate
Not applicable. This computational study does not involve human subjects, animal experimentation, or tissue samples.

### Consent for Publication
Not applicable.

---

## References

1. Petousis, I., et al. (2017). High-throughput screening of inorganic compounds for the discovery of novel dielectric and optical materials. *Scientific Data*, 4, 160134.
2. Ramprasad, R., et al. (2017). Machine learning in materials informatics: recent applications and prospects. *npj Computational Materials*, 3(1), 54.
3. Dunn, A., et al. (2020). Benchmarking materials property prediction methods: the Matbench test set and Automatminer reference algorithm. *npj Computational Materials*, 6(1), 138.
4. Ward, L., et al. (2016). A general-purpose machine learning framework for predicting properties of inorganic materials. *npj Computational Materials*, 2(1), 16028.
5. Xie, T., & Grossman, J. C. (2018). Crystal graph convolutional neural networks for an accurate and interpretable prediction of material properties. *Physical Review Letters*, 120(14), 145301.
6. Chen, C., et al. (2019). Graph networks as a universal machine learning framework for molecules and crystals. *Chemistry of Materials*, 31(9), 3564-3572.
7. De Breuck, P. P., et al. (2021). Robust model benchmarking and data-driven material discovery with MODNet. *npj Computational Materials*, 7(1), 83.
8. Schütt, K. T., et al. (2017). SchNet: A continuous-filter convolutional neural network for modeling quantum interactions. *Advances in Neural Information Processing Systems*, 30, 991-1001.
9. Choudhary, K., & DeCost, B. (2021). Atomistic line graph neural network for improved materials property predictions. *npj Computational Materials*, 7(1), 185.
10. Artrith, N., et al. (2021). Best practices in machine learning for chemistry. *Nature Chemistry*, 13(6), 505-508.
11. Xiong, Z., et al. (2023). Evaluating generalization and chemical extrapolation in crystal graph neural networks. *Materials Today*, 65, 120-132.
12. Jain, A., et al. (2013). Commentary: The Materials Project: A materials genome approach to accelerating materials innovation. *APL Materials*, 1(1), 011002.
13. Ward, L., et al. (2018). Matminer: An open source toolkit for materials data mining. *Computational Materials Science*, 152, 60-69.
14. Ong, S. P., et al. (2013). Python Materials Genomics (pymatgen): A robust, open-source python library for materials analysis. *Computational Materials Science*, 68, 314-319.
15. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.
16. Paszke, A., et al. (2019). PyTorch: An imperative style, high-performance deep learning library. *Advances in Neural Information Processing Systems*, 32, 8024-8035.
17. Loshchilov, I., & Hutter, F. (2017). Decoupled weight decay regularization. *arXiv preprint arXiv:1711.05101*.
18. Wilcoxon, F. (1945). Individual comparisons by ranking methods. *Biometrics Bulletin*, 1(6), 80-83.
"""
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("Successfully wrote updated manuscript_draft.md")

if __name__ == '__main__':
    main()
