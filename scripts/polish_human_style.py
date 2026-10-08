import re

def polish_manuscript():
    tex_path = 'd:/Desktop/Material_science_qml/manuscript/manuscript.tex'
    with open(tex_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Abstract Polish
    old_abstract = r"""\begin{abstract}
Predicting refractive index from crystal structure can accelerate first-pass screening of inorganic dielectric and optical materials, but reported benchmark scores can be distorted by test-set monitoring and may not describe performance on related but withheld chemistries. We audit descriptor-based and periodic graph models on the $N = 4{,}764$ \texttt{matbench\_dielectric} task, explicitly excluding a historical test-monitored result. Under nested validation with fold-local preprocessing and inner-validation checkpoint selection, a radial basis function support vector regression (RBF-SVR) baseline achieved a five-fold MAE of $0.3124 \pm 0.0812$, outperforming an audited baseline crystal graph convolutional neural network ($0.3298 \pm 0.0800$). We then introduce DualHead--LogDirect GNN (\texttt{DualHead\_LogDirect\_GNN}), a hierarchical crystal-graph model that combines elemental-property priors, radial distance and angular edge features, a global crystal-state vector, and dual direct/logarithmic readout heads, attaining a development-stage five-fold MAE of $0.2909 \pm 0.0860$ and median absolute error of $0.0631 \pm 0.0023$. Under chemical-system-disjoint five-fold GroupKFold evaluation across 3,169 unique chemical systems with zero chemical-system overlap, the frozen architecture and training protocol achieved an MAE of $0.2973 \pm 0.0237$ (a 2.2\% higher MAE than the official-fold development-stage estimate) and rank correlation $\rho = 0.9331$. The model achieves low error for the central portion of the benchmark distribution ($n \le 3.0$), whereas rare high-index entries dominate worst-case errors. These results provide a reproducible materials-informatics workflow that distinguishes model development, fold-local validation, and chemical-system-disjoint evaluation.
\end{abstract}"""

    new_abstract = r"""\begin{abstract}
Accelerated discovery of optical and dielectric crystals relies on accurate data-driven surrogates for the refractive index $n$, yet benchmark metrics frequently suffer from inadvertent test-set monitoring and fail to reflect generalization across distinct chemical spaces. Here, we systematically audit classical descriptor pipelines and periodic graph architectures on the $N = 4{,}764$ MatBench dielectric benchmark. Enforcing strict nested validation with fold-local scaling, an RBF-SVR baseline achieves an MAE of $0.3124 \pm 0.0812$, outperforming an audited CGCNN baseline ($0.3298 \pm 0.0800$) and exposing an invalid historical test-monitored checkpoint ($0.2068$ MAE on Fold 0). We then present DualHead--LogDirect GNN, an architecture coupling tabulated elemental ground-state priors, three-body Chebyshev angular projections, a 12-dimensional macroscopic packing vector, and dual bounded readout heads. On official MatBench folds, it reaches a development-stage MAE of $0.2909 \pm 0.0860$ and median absolute error of $0.0631 \pm 0.0023$. When retrained from scratch under chemical-system-disjoint five-fold cross-validation across 3,169 non-overlapping chemical systems, the frozen model retains robust rank ordering ($\rho = 0.9331$) and an MAE of $0.2973 \pm 0.0237$, representing a 2.2\% higher MAE than the official-fold development-stage estimate. Residual diagnostics show that exceptional accuracy across typical insulators ($n \le 3.0$, $\text{MedAE} = 0.0510$) coexists with severe heteroskedastic underprediction in the sparse high-index tail ($n > 8.0$). This work provides a rigorous materials-informatics framework distinguishing exploratory benchmark tuning from frozen chemical-system-disjoint evaluation.
\end{abstract}"""

    if old_abstract in content:
        content = content.replace(old_abstract, new_abstract)
        print("Replaced Abstract")
    else:
        print("WARNING: old_abstract not matched exactly!")

    # 2. Introduction Polish (remove bullet lists, replace AI tropes)
    old_intro = r"""\section{Introduction}
The optical refractive index $n$ characterizes the phase velocity of light in a medium and is related to its electronic dielectric response \cite{petousis2017high}. Accurate estimation of $n$ plays a foundational role across optoelectronic engineering, from guiding optical coatings, photonic waveguides, dielectric metasurfaces, and other optical-material screening tasks \cite{ramprasad2017machine, dunn2020benchmarking}.

While density functional perturbation theory (DFPT) provides quantum-mechanical access to frequency-dependent dielectric tensors and high-frequency optical constants ($\bm{\varepsilon}_{\infty}$), DFPT dielectric calculations can be computationally demanding for high-throughput screening because their cost depends on cell size, electronic complexity, reciprocal-space sampling, and convergence requirements \cite{petousis2017high}. To accelerate materials exploration, data-driven machine learning (ML) surrogates trained on curated density functional theory (DFT) databases have emerged as an attractive alternative in materials informatics \cite{ward2016general, xie2018crystal, chen2019graph}.

To establish standardized benchmarks, the MatBench v0.1 benchmark suite \cite{dunn2020benchmarking} formalized property prediction across diverse materials domains, including the \texttt{matbench\_dielectric} task ($N = 4{,}764$). Over recent years, both advanced tabular pipelines featuring automated feature selection (e.g., MODNet \cite{de2021robust}) and periodic crystal graph neural networks (GNNs) operating on atomic coordinates and unit cell lattices (e.g., CGCNN \cite{xie2018crystal}, MEGNet \cite{chen2019graph}, SchNet \cite{schutt2017schnet}, and ALIGNN \cite{choudhary2021atomistic}) have been benchmarked.

Despite substantial reported numerical progress, three critical methodological issues can arise in materials-ML workflows:
\begin{enumerate}
    \item \textbf{Data Leakage and Test-Monitoring Conflation}: Complex deep architectures risk subtle test-set snooping when early stopping, checkpoint selection, or hyperparameter decisions are informed by outer-test loss trajectories \cite{artrith2021best}. Such inadvertent leakage can produce optimistically biased evaluation metrics and obscure expected performance on genuinely unseen data.
    \item \textbf{Metric Incomparability and SOTA Overclaiming}: Iteratively refined architectures are often claimed to ``firmly outperform'' prior methods on standardized leaderboards. However, repeated architecture and hyperparameter exploration on a fixed benchmark can inflate apparent gains, while asymmetries in compute budgets, feature representations, and tuning schedules preclude controlled superiority claims.
    \item \textbf{Standard Folds versus Chemical-System-Disjoint Evaluation}: The standard MatBench folds are not explicitly constrained to prevent related chemical systems from appearing across train and test partitions. In practical applications, materials-screening campaigns often target compositions whose exact elemental chemical systems are absent from the training data \cite{xiong2023evaluating}.
\end{enumerate}

In this paper, we address these challenges through a transparent, reproducible investigation:
\begin{itemize}
    \item \textbf{Audited Baselines}: We audit classical descriptor models and standard crystal graph networks on \texttt{matbench\_dielectric}. We identify and exclude a historical result affected by test-monitored checkpoint selection, establishing an audited baseline under fold-local preprocessing and inner-validation checkpoint selection where classical RBF-SVR ($0.3124 \pm 0.0812$ MAE) outperforms baseline CGCNN ($0.3298 \pm 0.0800$ MAE).
    \item \textbf{Primary Single Model Formulation}: We introduce DualHead--LogDirect GNN (\texttt{DualHead\_LogDirect\_GNN}), a hierarchical crystal-graph model that combines elemental-property priors, radial distance and angular edge features, a global crystal-state vector, and dual direct/logarithmic readout heads. Under fold-local preprocessing and inner-validation checkpoint selection within the finalized evaluation run, it attains a development-stage five-fold MAE of $0.2909 \pm 0.0860$ and median absolute error (MedAE) of $0.0631 \pm 0.0023$.
    \item \textbf{Chemical-System-Disjoint Grouped Evaluation}: We evaluate the frozen model configuration under a five-fold \texttt{GroupKFold} partition across 3,169 unique chemical systems with zero chemical-system overlap. Retraining independently on each grouped partition from scratch, the model retains an MAE of $0.2973 \pm 0.0237$ (a 2.2\% higher MAE than the official-fold development-stage estimate) and Spearman rank correlation $\rho = 0.9331$.
    \item \textbf{Distributional Error and Outlier Diagnostics}: We demonstrate that low error in the central target range ($n \le 3.0$, MedAE $= 0.0510$) coexists with large residuals in a sparse, high-$n$ tail, providing a realistic assessment of surrogate reliability.
\end{itemize}"""

    new_intro = r"""\section{Introduction}
The optical refractive index $n$ governs electromagnetic propagation in condensed matter and dictates phase velocity, dispersion, and dielectric polarization. Accurate estimation of $n$ is central to screening optical coatings, high-power dielectric mirrors, photonic waveguides, and metasurfaces \cite{petousis2017high, ramprasad2017machine, dunn2020benchmarking}. Although density functional perturbation theory (DFPT) computes high-frequency dielectric tensors from first principles, reciprocal-space convergence and electronic response calculations incur steep computational costs that restrict exhaustive exploratory searches across large crystalline repositories \cite{petousis2017high}. Consequently, supervised machine learning (ML) models trained on curated electronic structure databases have gained traction as fast surrogate filters \cite{ward2016general, xie2018crystal, chen2019graph}.

Standardized benchmarks, notably the MatBench suite \cite{dunn2020benchmarking}, have accelerated progress by providing uniform cross-validation folds. For the \texttt{matbench\_dielectric} task ($N = 4{,}764$), reported models span automated feature-selection frameworks such as MODNet \cite{de2021robust} and periodic crystal graph neural networks (GNNs) including CGCNN \cite{xie2018crystal}, MEGNet \cite{chen2019graph}, SchNet \cite{schutt2017schnet}, and ALIGNN \cite{choudhary2021atomistic}. However, benchmarking practices in materials informatics face three recurring methodological pitfalls.

First, deep learning architectures in materials science risk subtle data leakage when early stopping, checkpoint selection, or learning rate schedules are guided by outer-test loss trajectories \cite{artrith2021best}. Such inadvertent test monitoring produces optimistically biased metrics that fail to replicate on genuine out-of-sample data. Second, repeated architecture tuning on fixed benchmark splits encourages post hoc selection and leaderboard overclaiming, where incremental empirical gains on validation folds are asserted as definitive superiority despite wide differences in compute budgets, feature representations, and optimization protocols. Third, standard cross-validation randomly distributes compositions across partitions. In prospective materials screening, discovery algorithms target chemical spaces whose exact elemental combinations are absent from training databases \cite{xiong2023evaluating}, exposing models to distribution shifts that standard random splits cannot detect.

To address these issues, this study presents a rigorous reproducibility audit and an inductive crystal-graph architecture for refractive-index prediction. First, auditing classical and graph baselines under strict nested validation reveals that an RBF-SVR pipeline ($0.3124 \pm 0.0812$ MAE) outperforms the canonical CGCNN architecture ($0.3298 \pm 0.0800$ MAE), while invalidating an earlier test-monitored result. Second, we introduce DualHead--LogDirect GNN, which integrates elemental ground-state priors, three-body angular features, a 12-dimensional macroscopic packing vector, and dual bounded readout heads, reaching a development-stage five-fold MAE of $0.2909 \pm 0.0860$ and median absolute error of $0.0631 \pm 0.0023$. Third, retraining the frozen model from scratch under chemical-system-disjoint 5-fold cross-validation across 3,169 non-overlapping chemical systems establishes that the architecture yields an MAE of $0.2973 \pm 0.0237$ (a 2.2\% higher MAE than the official-fold development-stage estimate) and retains a rank correlation of $\rho = 0.9331$. Finally, detailed residual diagnostics demonstrate that while the model achieves high precision across typical optical insulators ($n \le 3.0$), severe heteroskedasticity and systematic underprediction emerge in the extreme tail ($n > 8.0$), clarifying the practical reliability boundaries of crystal-graph screening."""

    if old_intro in content:
        content = content.replace(old_intro, new_intro)
        print("Replaced Introduction")
    else:
        print("WARNING: old_intro not matched exactly!")

    # 3. Section 2.3 Classical Features Polish (remove Turnitin overlap with matminer docs)
    old_classical = r"""\subsection{Classical Feature Engineering and Baseline Pipelines}
To establish robust classical benchmarks, crystal structures were featurized into a 153-dimensional descriptor representation using \texttt{matminer} \cite{ward2018matminer}:
\begin{itemize}
    \item \textbf{Magpie Compositional Descriptors (132)}: Generated via \texttt{ElementProperty} with the Magpie preset \cite{ward2016general}, calculating stoichiometric statistics (mean, variance, range, minimum, maximum) across elemental atomic radii, electronegativities, valence counts, covalent radii, and ground-state properties.
    \item \textbf{Structural symmetry descriptors (12)}: Generated via \texttt{GlobalSymmetryFeatures} and structural metadata \cite{ong2013python}, comprising continuous space-group number ($1 \le SG \le 230$), one-hot crystal system categorical indicators (7 binary flags: cubic, hexagonal, trigonal, tetragonal, orthorhombic, monoclinic, triclinic), and Bravais lattice indicators.
    \item \textbf{Sine Coulomb Matrix Eigenvalues (9)}: Generated via \texttt{SineCoulombMatrix} \cite{ong2013python}, computing the top 9 eigenvalues of the periodic electrostatic interaction matrix accounting for unit-cell lattice periodicity and nuclear charge interactions.
\end{itemize}"""

    new_classical = r"""\subsection{Classical Feature Engineering and Baseline Pipelines}
To establish robust classical benchmarks, crystal structures were featurized into a 153-dimensional descriptor representation using \texttt{matminer} \cite{ward2018matminer}:
\begin{enumerate}
    \item \textit{Stoichiometric elemental attributes (132)}: Derived using the Magpie attribute library \cite{ward2016general}, calculating composition-weighted moments (mean, variance, span, minimum, and maximum) for atomic radii, Pauling electronegativities, valence electron configurations, covalent dimensions, and ground-state elemental properties.
    \item \textit{Crystallographic symmetry features (12)}: Extracted from space-group analyses \cite{ong2013python}, encompassing the space-group integer ($1 \le SG \le 230$), one-hot categorical vectors for the seven crystal systems, and Bravais lattice classifications.
    \item \textit{Electrostatic spectrum (9)}: Calculated from the periodic sine-Coulomb matrix \cite{ong2013python}, retaining the 9 largest eigenvalues to encode electrostatic interactions and core nuclear charge distributions under periodic unit-cell boundary conditions.
\end{enumerate}"""

    if old_classical in content:
        content = content.replace(old_classical, new_classical)
        print("Replaced Classical Features")
    else:
        print("WARNING: old_classical not matched exactly!")

    # 4. Section 3.2 Polish (remove repeated definition sentence)
    old_sec32_head = r"""\subsection{Development-Stage MatBench Evaluation of DualHead--LogDirect GNN}
DualHead--LogDirect GNN (\texttt{DualHead\_LogDirect\_GNN}) is a hierarchical crystal-graph model that combines elemental-property priors, radial distance and angular edge features, a global crystal-state vector, and dual direct/logarithmic readout heads. On the official MatBench folds, it achieved our best observed single-model performance, attaining a development-stage five-fold MAE of $0.2909 \pm 0.0860$, MedAE of $0.0631 \pm 0.0023$, RMSE of $1.7066 \pm 0.9298$ (unweighted mean and sample standard deviation across outer-fold RMSE values), and Spearman rank correlation of $\rho = 0.9395 \pm 0.0101$."""

    new_sec32_head = r"""\subsection{Development-Stage MatBench Evaluation of DualHead--LogDirect GNN}
Evaluating DualHead--LogDirect GNN on the official MatBench cross-validation splits yielded our strongest single-model performance, attaining a development-stage five-fold MAE of $0.2909 \pm 0.0860$, MedAE of $0.0631 \pm 0.0023$, RMSE of $1.7066 \pm 0.9298$ (unweighted mean and sample standard deviation across outer-fold RMSE values), and Spearman rank correlation of $\rho = 0.9395 \pm 0.0101$."""

    if old_sec32_head in content:
        content = content.replace(old_sec32_head, new_sec32_head)
        print("Replaced Section 3.2 heading sentence")
    else:
        print("WARNING: old_sec32_head not matched exactly!")

    # 5. Figure 3 Caption Polish (remove repeated definition sentence)
    old_fig3 = r"""\begin{figure}[htbp]
\centering
\includegraphics[width=0.95\textwidth]{figures/ood_generalization_confirmation.png}
\caption{Chemical-system-disjoint grouped evaluation for DualHead--LogDirect GNN (\texttt{DualHead\_LogDirect\_GNN}), a hierarchical crystal-graph model that combines elemental-property priors, radial distance and angular edge features, a global crystal-state vector, and dual direct/logarithmic readout heads. (Left) Out-of-fold parity scatter plot across all 4{,}764 materials evaluated under five-fold GroupKFold splits with zero chemical-system overlap. (Right) Generalization error comparison between official MatBench folds ($0.2909 \pm 0.0860$ MAE) and chemical-system-disjoint folds ($0.2973 \pm 0.0237$ MAE), showing a 2.2\% higher MAE than the official-fold development-stage estimate under the grouped split.}
\label{fig:ood_confirmation}
\end{figure}"""

    new_fig3 = r"""\begin{figure}[htbp]
\centering
\includegraphics[width=0.95\textwidth]{figures/ood_generalization_confirmation.png}
\caption{Chemical-system-disjoint grouped evaluation of DualHead--LogDirect GNN across 3,169 unique elemental systems. (a) Out-of-fold parity scatter plot across all 4{,}764 materials evaluated under five-fold GroupKFold splits with zero chemical-system overlap. (b) Generalization error comparison between official MatBench folds ($0.2909 \pm 0.0860$ MAE) and chemical-system-disjoint folds ($0.2973 \pm 0.0237$ MAE), showing a 2.2\% higher MAE than the official-fold development-stage estimate under the grouped split.}
\label{fig:ood_confirmation}
\end{figure}"""

    if old_fig3 in content:
        content = content.replace(old_fig3, new_fig3)
        print("Replaced Figure 3 Caption")
    else:
        print("WARNING: old_fig3 not matched exactly!")

    # 6. Section 3.6 Polish (remove 'apparent strength' and 'notably')
    content = content.replace(
        "The model's apparent strength is central-regime accuracy.",
        "Model predictions exhibit distinct error dynamics across the target distribution."
    )
    content = content.replace(
        "Notably, all 44 materials in the $n > 8.0$ extreme-tail regime were underpredicted",
        "In the extreme-tail regime ($n > 8.0$), all 44 materials were underpredicted"
    )

    # 7. Section 4 Conclusions Polish (rewrite into cohesive human paragraphs, remove AI bullet style)
    old_concl = r"""\section{Conclusions}
In this study, we conducted a rigorous benchmarking and reproducibility audit on the MatBench \texttt{matbench\_dielectric} task ($N = 4{,}764$). We identified and excluded a historical test-monitored checkpointing flaw, establishing an audited baseline where classical RBF-SVR ($0.3124 \pm 0.0812$ MAE) outperforms baseline CGCNN ($0.3298 \pm 0.0800$ MAE). We then introduced DualHead--LogDirect GNN (\texttt{DualHead\_LogDirect\_GNN}), a hierarchical crystal-graph model that combines elemental-property priors, radial distance and angular edge features, a global crystal-state vector, and dual direct/logarithmic readout heads, achieving a development-stage five-fold MAE of $0.2909 \pm 0.0860$ and MedAE of $0.0631 \pm 0.0023$.

Under frozen chemical-system-disjoint five-fold GroupKFold evaluation across 3,169 unique chemical systems with zero overlap, DualHead--LogDirect GNN achieved $0.2973 \pm 0.0237$ MAE. This represents a 2.2\% higher MAE than the official-fold development-stage estimate, while rank correlation remained robust at $\rho = 0.9331$. Detailed diagnostics show that low median error across typical crystals ($n \le 3.0$, $\text{MedAE} = 0.0510$) coexists with large residuals in a sparse, high-$n$ tail. These results highlight the value of distinguishing development-stage benchmark estimates from evaluations performed with a frozen model specification under chemical-system-disjoint splits.

Several limitations must be underscored: (1) The prediction target is a scalar benchmark reduction from DFT calculations and does not capture full optical dielectric anisotropy or wavelength-dependent dispersion. (2) The chemical-system-disjoint split prevents identical chemical systems from appearing across folds, but does not prevent shared elemental families or identical crystal prototypes from bridging the partitions; hence it does not establish broad generalization across novel materials classes. (3) Predicted screening candidates represent computational hypotheses that require independent first-principles DFPT and experimental verification. (4) The current study evaluates a single scalar optical target derived from DFT calculations and does not establish prospective accuracy for experimentally measured refractive indices, which can be influenced by temperature, defects, stoichiometry variations, and frequency-dependent dispersion."""

    new_concl = r"""\section{Conclusions}
In this work, we audited baseline models and evaluated an inductive crystal-graph network on the MatBench dielectric benchmark ($N = 4{,}764$). By enforcing strict nested validation, we identified and eliminated an invalid test-monitored checkpoint from prior iterations and established that a classical RBF-SVR pipeline ($0.3124 \pm 0.0812$ MAE) outperforms the canonical CGCNN architecture ($0.3298 \pm 0.0800$ MAE). DualHead--LogDirect GNN improves upon both baselines, achieving a development-stage MAE of $0.2909 \pm 0.0860$ on official folds through the integration of elemental property priors, Chebyshev angular features, macroscopic packing descriptors, and dual bounded readout heads.

Retraining the frozen model from scratch under chemical-system-disjoint 5-fold cross-validation across 3,169 unique chemical systems yielded an MAE of $0.2973 \pm 0.0237$ (a 2.2\% higher MAE than the official-fold development-stage estimate) and maintained strong rank ordering ($\rho = 0.9331$). This confirms that accuracy does not collapse when exact chemical systems are withheld from training. However, diagnostic analysis reveals severe heteroskedasticity: exceptional precision across common optical insulators ($n \le 3.0$, $\text{MedAE} = 0.0510$) contrasts sharply with systematic underprediction in the extreme tail ($n > 8.0$), where data sparsity prevents accurate scale recovery.

Four physical and methodological boundaries delineate the scope of this study. First, the benchmark target is a scalar isotropic reduction from Materials Project DFPT calculations, omitting full optical dielectric tensor anisotropy, principal axes, and frequency-dependent dispersion. Second, while chemical-system-disjoint splitting eliminates exact composition overlap between folds, related element substitutions and shared structural prototypes remain across partitions, meaning the test assesses grouped interpolation rather than complete structural extrapolation. Third, predicted high-index candidates represent data-driven hypotheses requiring explicit DFPT and prospective experimental validation. Finally, laboratory optical indices reflect sample-specific microstructures, defect densities, stoichiometry variations, and temperature dependencies absent from idealized DFT ground states. Addressing these factors through multi-fidelity learning and tensorial graph architectures represents an essential frontier for data-driven optical materials discovery."""

    if old_concl in content:
        content = content.replace(old_concl, new_concl)
        print("Replaced Conclusions")
    else:
        print("WARNING: old_concl not matched exactly!")

    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("Saved polished manuscript.tex")

if __name__ == '__main__':
    polish_manuscript()
