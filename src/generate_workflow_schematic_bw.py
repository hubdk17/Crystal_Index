"""
Script: generate_workflow_schematic_bw.py
Author: Daksh Kaila
Description:
    Generates a 100% black-and-white, publication-quality schematic figure (Figure 1)
    reproducing the full architecture and evaluation workflow with:
      - STRICTLY Black and White / Grayscale (zero colour)
      - Zero text spilling / letters coming out of boxes (generous box margins and padding)
      - Word 'Guaranteed' replaced with scientific terminology ('Physical constraint')
      - Clean monochrome academic styling suitable for print and high-impact journals
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import shutil

# Typography
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['mathtext.fontset'] = 'dejavusans'

def create_schematic_bw():
    fig = plt.figure(figsize=(16, 10.5), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10.5)
    ax.axis('off')

    # Black and White / Grayscale Palette
    c_black = '#000000'
    c_dark_gray = '#1F2937'
    c_mid_gray = '#4B5563'
    c_light_fill = '#F3F4F6'
    c_subtle_fill = '#F9FAFB'
    c_white = '#FFFFFF'

    def draw_box(x, y, w, h, fill=c_white, border=c_black, radius=0.14, lw=1.3, ls='-'):
        box = FancyBboxPatch((x, y), w, h,
                             boxstyle=f"round,pad=0,rounding_size={radius}",
                             facecolor=fill, edgecolor=border, linewidth=lw, linestyle=ls, zorder=2)
        ax.add_patch(box)
        return box

    def draw_arrow(x1, y1, x2, y2, color=c_black, lw=1.5, style='-|>', mutation_scale=13):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                    shrinkA=2, shrinkB=2, mutation_scale=mutation_scale),
                    zorder=3)

    # =========================================================================
    # PANEL A: Architecture (x: 0.4 to 9.5, y: 0.02 to 10.35)
    # =========================================================================
    draw_box(0.4, 0.02, 9.1, 10.30, fill=c_white, border=c_black, radius=0.20, lw=1.6)
    ax.text(0.7, 10.08, "(a) DualHead–LogDirect GNN Architecture",
            fontsize=13.0, fontweight='bold', color=c_black, va='center')

    # 1. Periodic Crystal Structure Box
    draw_box(2.35, 9.10, 5.2, 0.72, fill=c_white, border=c_black, radius=0.10, lw=1.2)
    ax.text(4.95, 9.54, "Periodic Crystal Structure", ha='center', va='center',
            fontsize=10.5, fontweight='bold', color=c_black)
    ax.text(4.95, 9.28, "Unit-cell lattice (a, b, c)  •  Fractional atom positions  •  Atomic numbers {Z_i}",
            ha='center', va='center', fontsize=8.2, color=c_mid_gray)

    draw_arrow(4.95, 9.10, 4.95, 8.52)

    # 2. Periodic Graph Construction
    draw_box(2.15, 7.82, 5.6, 0.70, fill=c_light_fill, border=c_black, radius=0.10, lw=1.3)
    ax.text(4.95, 8.26, "Periodic Graph Construction", ha='center', va='center',
            fontsize=10.5, fontweight='bold', color=c_black)
    ax.text(4.95, 8.00, r"Radial cutoff $R_{\mathrm{cut}} = 6.0\,\mathrm{\AA}$  •  Truncated to top 12 nearest neighbors",
            ha='center', va='center', fontsize=8.2, color=c_dark_gray)

    # Branching arrows to 3 channels
    draw_arrow(3.30, 7.82, 1.95, 7.15)
    draw_arrow(4.95, 7.82, 4.95, 7.15)
    draw_arrow(6.60, 7.82, 7.95, 7.15)

    # 3. Three Parallel Channels (All in crisp monochrome, height=2.15 for generous padding)
    # Node Channel
    draw_box(0.60, 5.00, 2.70, 2.15, fill=c_white, border=c_black, radius=0.12, lw=1.4)
    ax.text(1.95, 6.86, "Atom Node Channel", ha='center', va='center',
            fontsize=10.0, fontweight='bold', color=c_black)
    ax.text(1.95, 6.55, r"Learnable $Z \in [1, 100]$ (48-d)", ha='center', va='center',
            fontsize=8.0, color=c_dark_gray)
    ax.text(1.95, 6.27, "+ 8 Tabulated Descriptors:", ha='center', va='center',
            fontsize=8.0, fontweight='bold', color=c_black)
    ax.text(1.95, 5.82, "Pauling $X$, radius $r$, mass $m$,\ngroup $g$, row, 1st IE, $V_{\\mathrm{mol}}$, $\\mathrm{ox}_{\\max}$",
            ha='center', va='center', fontsize=7.6, color=c_mid_gray, multialignment='center')
    ax.text(1.95, 5.30, r"$\mathbf{h}_i^{(0)} = [\mathbf{e}_{Z_i} \parallel \mathbf{p}_i] \in \mathbb{R}^{56}$",
            ha='center', va='center', fontsize=8.6, fontweight='bold', color=c_black)

    # Edge Channel
    draw_box(3.50, 5.00, 2.90, 2.15, fill=c_white, border=c_black, radius=0.12, lw=1.4)
    ax.text(4.95, 6.86, "Pair & Triplet Edge Channel", ha='center', va='center',
            fontsize=10.0, fontweight='bold', color=c_black)
    ax.text(4.95, 6.55, r"41 Gaussian RBFs ($\sigma = 0.15\,\mathrm{\AA}$)", ha='center', va='center',
            fontsize=8.0, color=c_dark_gray)
    ax.text(4.95, 6.27, r"+ Inverse distance $1 / (r_{ij} + 0.01)$", ha='center', va='center',
            fontsize=8.0, color=c_dark_gray)
    ax.text(4.95, 5.82, "+ 16 Chebyshev angle features\n" + r"$\cos\theta_{jik} = \hat{\mathbf{r}}_{ij} \cdot \hat{\mathbf{r}}_{ik}$ averaged $\forall k \neq j$",
            ha='center', va='center', fontsize=7.6, color=c_mid_gray, multialignment='center')
    ax.text(4.95, 5.30, r"$\mathbf{e}_{ij}^{(0)} \in \mathbb{R}^{58}$  (Vertex: atom $i$)",
            ha='center', va='center', fontsize=8.6, fontweight='bold', color=c_black)

    # Global Channel
    draw_box(6.60, 5.00, 2.70, 2.15, fill=c_white, border=c_black, radius=0.12, lw=1.4)
    ax.text(7.95, 6.86, "Global Descriptor Channel", ha='center', va='center',
            fontsize=10.0, fontweight='bold', color=c_black)
    ax.text(7.95, 6.55, "12-d Structural–Compositional:", ha='center', va='center',
            fontsize=8.0, fontweight='bold', color=c_black)
    ax.text(7.95, 6.12, r"$\rho, V/N, \phi, N_{\mathrm{elem}}, \bar{X}, \Delta X,$" + "\n" + r"$\bar{r}, \bar{g}, \bar{V}_{\mathrm{mol}}, \bar{z}, \eta_{\mathrm{pack}}, c/a$",
            ha='center', va='center', fontsize=7.4, color=c_mid_gray, multialignment='center')
    ax.text(7.95, 5.68, r"$\eta_{\mathrm{pack}} = \phi (V/N)/\bar{r}^3$ (dimensionless)",
            ha='center', va='center', fontsize=7.4, color=c_dark_gray)
    # Badge inside global channel
    draw_box(6.95, 5.16, 2.00, 0.32, fill=c_light_fill, border=c_black, radius=0.06, lw=1.0)
    ax.text(7.95, 5.32, "Fold-local scaling only", ha='center', va='center',
            fontsize=7.2, fontweight='bold', color=c_black)

    # Converging arrows
    draw_arrow(1.95, 5.00, 3.70, 4.42)
    draw_arrow(4.95, 5.00, 4.95, 4.42)
    draw_arrow(7.95, 5.00, 6.20, 4.42)

    # 4. Three Hierarchical Interaction Layers (Enlarged with generous room for equations)
    draw_box(0.85, 2.85, 8.20, 1.55, fill=c_subtle_fill, border=c_black, radius=0.14, lw=1.5)
    ax.text(4.95, 4.14, "Three Hierarchical Interaction Layers  [LayerNorm + SiLU]",
            ha='center', va='center', fontsize=10.2, fontweight='bold', color=c_black)
    layer_eq1 = r"1. Edge update:  $\mathbf{e}_{ij}^{(l+1)} = \mathbf{e}_{ij}^{(l)} + \mathrm{MLP}_{\mathbf{e}}([\mathbf{h}_i^{(l)} \parallel \mathbf{h}_j^{(l)} \parallel \mathbf{e}_{ij}^{(l)} \parallel \mathbf{u}^{(l)}])$"
    layer_eq2 = r"2. Message sum:  $\mathbf{m}_i^{(l+1)} = \sum_{j \in \mathcal{N}(i)} \mathbf{e}_{ij}^{(l+1)}$  $\rightarrow$  Node residual:  $\mathbf{h}_i^{(l+1)} = \mathbf{h}_i^{(l)} + \mathrm{MLP}_{\mathbf{v}}([\mathbf{h}_i^{(l)} \parallel \mathbf{m}_i^{(l+1)} \parallel \mathbf{u}^{(l)}])$"
    layer_eq3 = r"3. Global state update:  $\mathbf{u}^{(l+1)} = \mathbf{u}^{(l)} + \mathrm{MLP}_{\mathbf{u}}\left(\left[\frac{1}{N}\sum_i \mathbf{h}_i^{(l+1)} \parallel \mathbf{u}^{(l)}\right]\right)$"
    ax.text(4.95, 3.72, layer_eq1, ha='center', va='center', fontsize=7.8, color=c_dark_gray)
    ax.text(4.95, 3.40, layer_eq2, ha='center', va='center', fontsize=7.8, color=c_dark_gray)
    ax.text(4.95, 3.08, layer_eq3, ha='center', va='center', fontsize=7.8, color=c_dark_gray)

    # Arrow to pooling
    draw_arrow(4.95, 2.85, 4.95, 2.48)

    # 5. Crystal Representation Pooling
    draw_box(1.75, 2.00, 6.40, 0.48, fill=c_light_fill, border=c_black, radius=0.08, lw=1.2)
    ax.text(4.95, 2.24, r"Crystal Representation:  $\mathbf{h}_{\mathrm{crystal}} = \left[\frac{1}{N}\sum_{i=1}^N \mathbf{h}_i^{(3)} \parallel \mathbf{u}^{(3)}\right] \in \mathbb{R}^{76}$",
            ha='center', va='center', fontsize=8.8, fontweight='bold', color=c_black)

    # Branching arrows to readout heads
    draw_arrow(3.40, 2.00, 2.55, 1.62)
    draw_arrow(6.50, 2.00, 7.35, 1.62)

    # 6. Dual Readout Heads (height=0.85, width=3.80 - ample padding, NO letters coming out)
    # Direct Readout Head (Left)
    draw_box(0.65, 0.72, 3.80, 0.90, fill=c_white, border=c_black, radius=0.10, lw=1.3)
    ax.text(2.55, 1.40, "Direct Readout Head", ha='center', va='center',
            fontsize=9.5, fontweight='bold', color=c_black)
    ax.text(2.55, 1.15, r"$\hat{n}_{\mathrm{direct}} = 1.0 + \mathrm{Softplus}(\mathbf{W}_d \mathbf{h}_{\mathrm{crystal}} + b_d)$",
            ha='center', va='center', fontsize=8.2, color=c_dark_gray)
    # 'Guaranteed bound' replaced with neutral scientific 'Physical constraint'
    ax.text(2.55, 0.90, r"Physical constraint:  $\hat{n}_{\mathrm{direct}} \geq 1.0$",
            ha='center', va='center', fontsize=8.0, fontweight='bold', color=c_black)

    # Logarithmic Readout Head (Right)
    draw_box(5.45, 0.72, 3.80, 0.90, fill=c_white, border=c_black, radius=0.10, lw=1.3)
    ax.text(7.35, 1.40, "Logarithmic Readout Head", ha='center', va='center',
            fontsize=9.5, fontweight='bold', color=c_black)
    ax.text(7.35, 1.15, r"$\hat{z}_{\mathrm{log}} = \mathbf{W}_\ell \mathbf{h}_{\mathrm{crystal}} + b_\ell \approx \ln(n - 0.99)$",
            ha='center', va='center', fontsize=8.2, color=c_dark_gray)
    ax.text(7.35, 0.90, r"Inverted prediction:  $\hat{n}_{\log} = 0.99 + \exp(\hat{z}_{\mathrm{log}}) > 0.99$",
            ha='center', va='center', fontsize=7.8, color=c_mid_gray)

    # Converging arrows to Blended Output
    draw_arrow(3.50, 0.72, 4.10, 0.52)
    draw_arrow(6.40, 0.72, 5.80, 0.52)

    # 7. Blended Prediction (w=4.8, h=0.46, centered at 4.95)
    draw_box(2.55, 0.08, 4.80, 0.44, fill=c_light_fill, border=c_black, radius=0.08, lw=1.5)
    ax.text(4.95, 0.30, r"Blended Prediction:  $\hat{n} = \frac{1}{2}(\hat{n}_{\mathrm{direct}} + \hat{n}_{\log}) \geq 1.0$",
            ha='center', va='center', fontsize=9.2, fontweight='bold', color=c_black)

    # =========================================================================
    # PANEL B: Evaluation Protocols (x: 9.8 to 15.6, y: 0.02 to 10.35)
    # =========================================================================
    draw_box(9.8, 0.02, 5.8, 10.30, fill=c_white, border=c_black, radius=0.20, lw=1.6)
    ax.text(10.1, 10.08, "(b) Evaluation Protocols and Leakage Controls",
            fontsize=13.0, fontweight='bold', color=c_black, va='center')

    # 1. Official MatBench Benchmark
    draw_box(10.30, 9.10, 4.80, 0.72, fill=c_white, border=c_black, radius=0.10, lw=1.2)
    ax.text(12.70, 9.54, "Official MatBench Benchmark", ha='center', va='center',
            fontsize=10.5, fontweight='bold', color=c_black)
    ax.text(12.70, 9.28, r"$\mathtt{matbench\_dielectric}$  •  $N = 4{,}764$ inorganic crystals",
            ha='center', va='center', fontsize=8.2, color=c_mid_gray)

    draw_arrow(12.70, 9.10, 12.70, 8.52)

    # 2. Official Five Outer Folds Partition
    draw_box(10.20, 7.82, 5.00, 0.70, fill=c_light_fill, border=c_black, radius=0.10, lw=1.3)
    ax.text(12.70, 8.26, "Official Five Outer Folds Partition", ha='center', va='center',
            fontsize=10.5, fontweight='bold', color=c_black)
    ax.text(12.70, 8.00, r"Zero train/test overlap: $T_{f,\mathrm{train}} \cap T_{f,\mathrm{test}} = \emptyset$ $\forall f \in \{0..4\}$",
            ha='center', va='center', fontsize=7.8, color=c_dark_gray)

    # Branch to Train vs Test
    draw_arrow(11.60, 7.82, 11.35, 7.15)
    draw_arrow(13.80, 7.82, 14.05, 7.15)

    # 3. Outer Train Fold vs Outer Test Fold (height=2.15)
    # Outer Train (80%)
    draw_box(10.05, 5.00, 2.60, 2.15, fill=c_white, border=c_black, radius=0.12, lw=1.4)
    ax.text(11.35, 6.88, "Outer Train Fold (80%)", ha='center', va='center',
            fontsize=9.2, fontweight='bold', color=c_black)
    ax.text(11.35, 6.64, r"$N \approx 3{,}811$ crystals", ha='center', va='center',
            fontsize=7.8, color=c_mid_gray)

    # Nested Inner Train / Val (Dashed box)
    draw_box(10.20, 5.12, 2.30, 1.38, fill=c_subtle_fill, border=c_black, radius=0.08, lw=1.1, ls='--')
    ax.text(11.35, 6.28, "Inner Train (85%)", ha='center', va='center',
            fontsize=8.0, fontweight='bold', color=c_black)
    ax.text(11.35, 6.08, "Model optimization (AdamW)", ha='center', va='center',
            fontsize=7.0, color=c_mid_gray)
    ax.text(11.35, 5.68, "Inner Validation (15%)", ha='center', va='center',
            fontsize=8.0, fontweight='bold', color=c_black)
    ax.text(11.35, 5.38, "Checkpoint selection ONLY\n(Epoch w/ min val loss)", ha='center', va='center',
            fontsize=6.9, fontweight='bold', color=c_dark_gray, multialignment='center')

    # Outer Test (20%)
    draw_box(12.85, 5.00, 2.60, 2.15, fill=c_white, border=c_black, radius=0.12, lw=1.4)
    ax.text(14.15, 6.88, "Outer Test Fold (20%)", ha='center', va='center',
            fontsize=9.2, fontweight='bold', color=c_black)
    ax.text(14.15, 6.64, r"$N = 953$ (or $952$) crystals", ha='center', va='center',
            fontsize=7.8, color=c_mid_gray)
    ax.text(14.15, 6.28, "Evaluated ONCE only", ha='center', va='center',
            fontsize=8.2, fontweight='bold', color=c_black)
    ax.text(14.15, 6.04, "using frozen inner-val\ncheckpoint weights", ha='center', va='center',
            fontsize=7.2, color=c_mid_gray, multialignment='center')
    # Leakage badge inside outer test
    draw_box(12.98, 5.16, 2.34, 0.48, fill=c_light_fill, border=c_black, radius=0.06, lw=1.0)
    ax.text(14.15, 5.40, "Outer-test labels never used\nfor checkpoint selection", ha='center', va='center',
            fontsize=6.7, fontweight='bold', color=c_black, multialignment='center')

    # Arrows to 5-Fold Result
    draw_arrow(11.35, 5.00, 12.70, 4.52)
    draw_arrow(14.15, 5.00, 12.70, 4.52)

    # 4. Development-Stage Official 5-Fold Result Box
    draw_box(10.15, 3.82, 5.10, 0.70, fill=c_light_fill, border=c_black, radius=0.10, lw=1.5)
    ax.text(12.70, 4.28, "Development-Stage Official 5-Fold Result", ha='center', va='center',
            fontsize=9.4, fontweight='bold', color=c_black)
    ax.text(12.70, 4.02, r"$\mathbf{0.2909 \pm 0.0860}$ MAE  •  $\mathrm{MedAE} = 0.0631$  •  $\rho = 0.9395$",
            ha='center', va='center', fontsize=8.2, fontweight='bold', color=c_black)

    # Dividing separator line
    ax.plot([10.1, 15.3], [3.60, 3.60], color=c_mid_gray, lw=1.0, ls=':', zorder=2)

    # 5. Lower Section: Chemical-System-Disjoint Stress Test
    ax.text(12.70, 3.38, "Grouped Extrapolation Stress Test", ha='center', va='center',
            fontsize=10.2, fontweight='bold', color=c_black)

    # Step 1: Frozen Model Specification
    draw_box(10.35, 2.68, 4.70, 0.50, fill=c_white, border=c_black, radius=0.08, lw=1.2)
    ax.text(12.70, 2.98, "Frozen Model Specification", ha='center', va='center',
            fontsize=8.8, fontweight='bold', color=c_black)
    ax.text(12.70, 2.80, r"Fixed architecture, loss, hyperparameters, and $R_{\mathrm{cut}} = 6.0\,\mathrm{\AA}$",
            ha='center', va='center', fontsize=7.2, color=c_mid_gray)

    draw_arrow(12.70, 2.68, 12.70, 2.38)

    # Step 2: Chemical-System GroupKFold Box (w=5.10, h=0.68 - ample space!)
    draw_box(10.15, 1.70, 5.10, 0.68, fill=c_white, border=c_black, radius=0.10, lw=1.2)
    ax.text(12.70, 2.18, "Chemical-System GroupKFold Partition", ha='center', va='center',
            fontsize=9.0, fontweight='bold', color=c_black)
    ax.text(12.70, 1.96, "3,169 systems  •  Zero train–test chemical-system overlap",
            ha='center', va='center', fontsize=7.6, color=c_dark_gray)
    ax.text(12.70, 1.81, r"scikit-learn 1.9.0  •  $\mathtt{shuffle=False}$ (deterministic)",
            ha='center', va='center', fontsize=7.0, color=c_mid_gray)

    draw_arrow(12.70, 1.70, 12.70, 1.40)

    # Step 3: Retrain Frozen Configuration
    draw_box(10.35, 0.88, 4.70, 0.52, fill=c_white, border=c_black, radius=0.08, lw=1.2)
    ax.text(12.70, 1.20, "Retrain Frozen Configuration per Grouped Fold", ha='center', va='center',
            fontsize=8.8, fontweight='bold', color=c_black)
    ax.text(12.70, 1.01, "5 models retrained from scratch  •  Fold-local normalization",
            ha='center', va='center', fontsize=7.4, color=c_mid_gray)

    draw_arrow(12.70, 0.88, 12.70, 0.64)

    # Step 4: Grouped Result Box (w=5.20, h=0.56, plenty of room from y=0.08 to y=0.64)
    draw_box(10.10, 0.08, 5.20, 0.56, fill=c_light_fill, border=c_black, radius=0.10, lw=1.5)
    ax.text(12.70, 0.48, "Chemical-System-Disjoint Grouped Evaluation", ha='center', va='center',
            fontsize=9.0, fontweight='bold', color=c_black)
    ax.text(12.70, 0.30, r"$\mathbf{0.2973 \pm 0.0237}$ MAE  •  $\mathrm{MedAE} = 0.0692$  •  $\rho = 0.9331$",
            ha='center', va='center', fontsize=8.0, fontweight='bold', color=c_black)
    ax.text(12.70, 0.16, "(Modest +2.2% MAE difference relative to official-fold development)",
            ha='center', va='center', fontsize=7.0, color=c_mid_gray)

    # Output directory
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'manuscript', 'figures')
    os.makedirs(out_dir, exist_ok=True)

    png_path = os.path.join(out_dir, 'model_architecture_and_workflow.png')
    pdf_path = os.path.join(out_dir, 'model_architecture_and_workflow.pdf')

    plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='#FFFFFF', edgecolor='none')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='#FFFFFF', edgecolor='none')
    plt.close()

    # Also copy to artifacts directory for direct user preview
    artifact_dir = r"C:\Users\HP\.gemini\antigravity-ide\brain\7e1462b3-34c9-4d76-a8c1-ddab45f77d0b"
    shutil.copy(png_path, os.path.join(artifact_dir, 'model_architecture_and_workflow_bw.png'))
    shutil.copy(png_path, os.path.join(artifact_dir, 'model_architecture_and_workflow.png'))

    print(f"Generated B&W schematic successfully:")
    print(f"  PNG: {png_path} ({os.path.getsize(png_path):,} bytes)")
    print(f"  PDF: {pdf_path} ({os.path.getsize(pdf_path):,} bytes)")

if __name__ == '__main__':
    create_schematic_bw()
