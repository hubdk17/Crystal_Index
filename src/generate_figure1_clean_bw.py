"""
Script: generate_figure1_clean_bw.py
Author: Daksh Kaila
Description:
    Generates a streamlined, high-legibility, 100% black-and-white publication schematic (Figure 1).
    - Preserves complete architecture flow and evaluation protocol
    - Eliminates micro-equation clutter and overflowing text
    - Uses larger, clear fonts legible at print scale
    - Strictly black and white / grayscale with consistent line weights
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import shutil

matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['mathtext.fontset'] = 'dejavusans'

def create_figure1_clean():
    fig = plt.figure(figsize=(15.5, 9.8), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 15.5)
    ax.set_ylim(0, 9.8)
    ax.axis('off')

    c_black = '#000000'
    c_dark = '#1E293B'
    c_muted = '#475569'
    c_white = '#FFFFFF'
    c_gray_bg = '#F8FAFC'
    c_gray_subtle = '#F1F5F9'

    def draw_box(x, y, w, h, fill=c_white, border=c_black, radius=0.12, lw=1.3, ls='-'):
        box = FancyBboxPatch((x, y), w, h,
                             boxstyle=f"round,pad=0,rounding_size={radius}",
                             facecolor=fill, edgecolor=border, linewidth=lw, linestyle=ls, zorder=2)
        ax.add_patch(box)
        return box

    def draw_arrow(x1, y1, x2, y2, lw=1.4):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='-|>', color=c_black, lw=lw,
                                    shrinkA=2, shrinkB=2, mutation_scale=13),
                    zorder=3)

    # =========================================================================
    # PANEL A: Architecture (x: 0.4 to 9.3, y: 0.25 to 9.55)
    # =========================================================================
    draw_box(0.4, 0.25, 8.9, 9.30, fill=c_white, border=c_black, radius=0.18, lw=1.5)
    ax.text(0.7, 9.25, "(a) DualHead–LogDirect GNN Architecture",
            fontsize=12.5, fontweight='bold', color=c_black, va='center')

    # 1. Periodic Crystal Structure
    draw_box(2.35, 8.35, 5.0, 0.65, fill=c_white, border=c_black, radius=0.10, lw=1.2)
    ax.text(4.85, 8.74, "Periodic Crystal Structure", ha='center', va='center',
            fontsize=10.2, fontweight='bold', color=c_black)
    ax.text(4.85, 8.52, r"Lattice vectors $(\mathbf{a}, \mathbf{b}, \mathbf{c})$  •  Fractional atomic positions  •  Species $\{Z_i\}$",
            ha='center', va='center', fontsize=8.0, color=c_muted)

    draw_arrow(4.85, 8.35, 4.85, 7.82)

    # 2. Periodic Graph Construction
    draw_box(2.25, 7.18, 5.2, 0.64, fill=c_gray_bg, border=c_black, radius=0.10, lw=1.2)
    ax.text(4.85, 7.56, "Periodic Graph Construction", ha='center', va='center',
            fontsize=10.2, fontweight='bold', color=c_black)
    ax.text(4.85, 7.34, r"Radial cutoff $R_{\mathrm{cut}} = 6.0\,\mathrm{\AA}$  •  Truncated to top 12 nearest neighbors",
            ha='center', va='center', fontsize=8.0, color=c_dark)

    # Branching arrows
    draw_arrow(3.25, 7.18, 1.95, 6.55)
    draw_arrow(4.85, 7.18, 4.85, 6.55)
    draw_arrow(6.45, 7.18, 7.75, 6.55)

    # 3. Three Information Channels (Clean conceptual bullets, ample padding)
    # Channel 1: Nodes
    draw_box(0.65, 4.85, 2.60, 1.70, fill=c_white, border=c_black, radius=0.12, lw=1.3)
    ax.text(1.95, 6.30, "Atom Node Channel", ha='center', va='center',
            fontsize=9.6, fontweight='bold', color=c_black)
    ax.text(1.95, 5.98, r"$\mathbf{h}_i^{(0)} \in \mathbb{R}^{56}$", ha='center', va='center',
            fontsize=8.4, fontweight='bold', color=c_dark)
    ax.text(1.95, 5.48, "• Learnable $Z$ embedding (48-d)\n• 8 elemental ground-state priors\n  (radii, electronegativity, volume,\n  ionization, mass, row, group)",
            ha='center', va='center', fontsize=7.6, color=c_muted)

    # Channel 2: Edges
    draw_box(3.50, 4.85, 2.70, 1.70, fill=c_white, border=c_black, radius=0.12, lw=1.3)
    ax.text(4.85, 6.30, "Edge Geometry Channel", ha='center', va='center',
            fontsize=9.6, fontweight='bold', color=c_black)
    ax.text(4.85, 5.98, r"$\mathbf{e}_{ij}^{(0)} \in \mathbb{R}^{58}$", ha='center', va='center',
            fontsize=8.4, fontweight='bold', color=c_dark)
    ax.text(4.85, 5.48, "• 41 Gaussian radial RBFs\n• $1/(r_{ij} + 0.01)$ inverse channel\n• 16 Chebyshev bond-angle\n  projections (3-body geometry)",
            ha='center', va='center', fontsize=7.6, color=c_muted)

    # Channel 3: Global
    draw_box(6.45, 4.85, 2.60, 1.70, fill=c_white, border=c_black, radius=0.12, lw=1.3)
    ax.text(7.75, 6.30, "Global Descriptor Channel", ha='center', va='center',
            fontsize=9.6, fontweight='bold', color=c_black)
    ax.text(7.75, 5.98, r"$\mathbf{u} \in \mathbb{R}^{12}$", ha='center', va='center',
            fontsize=8.4, fontweight='bold', color=c_dark)
    ax.text(7.75, 5.48, "• Density $\\rho$ & packing fraction $\\phi$\n• Stoichiometric composition stats\n• Mean coordination & aspect ratio\n• Fold-local scaling only",
            ha='center', va='center', fontsize=7.6, color=c_muted)

    # Converging arrows
    draw_arrow(1.95, 4.85, 3.65, 4.25)
    draw_arrow(4.85, 4.85, 4.85, 4.25)
    draw_arrow(7.75, 4.85, 6.05, 4.25)

    # 4. Hierarchical Interaction Layers (3 Layers)
    draw_box(0.85, 2.95, 8.00, 1.30, fill=c_gray_bg, border=c_black, radius=0.12, lw=1.4)
    ax.text(4.85, 3.98, "Three Hierarchical Interaction Layers  [LayerNorm + SiLU]",
            ha='center', va='center', fontsize=9.8, fontweight='bold', color=c_black)
    ax.text(4.85, 3.70, "Bidirectional, three-way mutual information coupling at each layer:",
            ha='center', va='center', fontsize=8.0, color=c_dark)
    ax.text(4.85, 3.32, "1. Edge update conditioned on incident nodes and global state:   " + r"$\mathbf{e}_{ij}^{(l+1)} = \mathbf{e}_{ij}^{(l)} + \mathrm{MLP}_{\mathbf{e}}(\dots)$" + "\n" + \
                        "2. Node update via unnormalized message aggregation:   " + r"$\mathbf{h}_i^{(l+1)} = \mathbf{h}_i^{(l)} + \mathrm{MLP}_{\mathbf{v}}(\dots)$" + "\n" + \
                        "3. Global state update conditioned on pooled crystal graph:   " + r"$\mathbf{u}^{(l+1)} = \mathbf{u}^{(l)} + \mathrm{MLP}_{\mathbf{u}}(\dots)$",
            ha='center', va='center', fontsize=7.6, color=c_muted)

    draw_arrow(4.85, 2.95, 4.85, 2.45)

    # 5. Crystal Representation Pooling
    draw_box(1.85, 1.95, 6.00, 0.50, fill=c_white, border=c_black, radius=0.08, lw=1.2)
    ax.text(4.85, 2.20, r"Crystal Representation:   $\mathbf{h}_{\mathrm{crystal}} = \left[\frac{1}{N}\sum_{i=1}^N \mathbf{h}_i^{(3)} \parallel \mathbf{u}^{(3)}\right] \in \mathbb{R}^{76}$",
            ha='center', va='center', fontsize=8.6, fontweight='bold', color=c_black)

    # Branching arrows to readout heads
    draw_arrow(3.35, 1.95, 2.50, 1.55)
    draw_arrow(6.35, 1.95, 7.20, 1.55)

    # 6. Dual Readout Heads
    # Direct Head (Left)
    draw_box(0.65, 0.72, 3.70, 0.83, fill=c_white, border=c_black, radius=0.10, lw=1.2)
    ax.text(2.50, 1.35, "Direct Softplus Readout Head", ha='center', va='center',
            fontsize=9.0, fontweight='bold', color=c_black)
    ax.text(2.50, 1.10, r"$\hat{n}_{\mathrm{direct}} = 1.0 + \mathrm{Softplus}(\mathbf{W}_d \mathbf{h}_{\mathrm{crystal}} + b_d)$",
            ha='center', va='center', fontsize=8.0, color=c_dark)
    ax.text(2.50, 0.88, r"Physical constraint: $\hat{n}_{\mathrm{direct}} \geq 1.0$",
            ha='center', va='center', fontsize=7.8, fontweight='bold', color=c_black)

    # Log Head (Right)
    draw_box(5.35, 0.72, 3.70, 0.83, fill=c_white, border=c_black, radius=0.10, lw=1.2)
    ax.text(7.20, 1.35, "Logarithmic Smooth-L1 Readout Head", ha='center', va='center',
            fontsize=9.0, fontweight='bold', color=c_black)
    ax.text(7.20, 1.10, r"$\hat{z}_{\mathrm{log}} = \mathbf{W}_\ell \mathbf{h}_{\mathrm{crystal}} + b_\ell \approx \ln(n - 0.99)$",
            ha='center', va='center', fontsize=8.0, color=c_dark)
    ax.text(7.20, 0.88, r"Inverted: $\hat{n}_{\log} = 0.99 + \exp(\hat{z}_{\mathrm{log}}) > 0.99$",
            ha='center', va='center', fontsize=7.8, color=c_muted)

    # Arrows to Blended Output
    draw_arrow(3.40, 0.72, 4.05, 0.48)
    draw_arrow(6.30, 0.72, 5.65, 0.48)

    # 7. Blended Output
    draw_box(2.55, 0.06, 4.60, 0.42, fill=c_gray_subtle, border=c_black, radius=0.08, lw=1.4)
    ax.text(4.85, 0.27, r"Blended Prediction:   $\hat{n} = \frac{1}{2}(\hat{n}_{\mathrm{direct}} + \hat{n}_{\log}) \geq 1.0$",
            ha='center', va='center', fontsize=8.8, fontweight='bold', color=c_black)

    # =========================================================================
    # PANEL B: Evaluation Protocols (x: 9.6 to 15.1, y: 0.25 to 9.55)
    # =========================================================================
    draw_box(9.6, 0.25, 5.5, 9.30, fill=c_white, border=c_black, radius=0.18, lw=1.5)
    ax.text(9.9, 9.25, "(b) Evaluation Protocols and Leakage Controls",
            fontsize=12.5, fontweight='bold', color=c_black, va='center')

    # 1. Official MatBench Benchmark
    draw_box(10.05, 8.35, 4.60, 0.65, fill=c_white, border=c_black, radius=0.10, lw=1.2)
    ax.text(12.35, 8.74, "Official MatBench Benchmark", ha='center', va='center',
            fontsize=10.0, fontweight='bold', color=c_black)
    ax.text(12.35, 8.52, r"$\mathtt{matbench\_dielectric}$  •  $N = 4{,}764$ inorganic crystals",
            ha='center', va='center', fontsize=8.0, color=c_muted)

    draw_arrow(12.35, 8.35, 12.35, 7.82)

    # 2. Official 5 Outer Folds
    draw_box(9.95, 7.18, 4.80, 0.64, fill=c_gray_bg, border=c_black, radius=0.10, lw=1.2)
    ax.text(12.35, 7.56, "Official Five Outer Folds Partition", ha='center', va='center',
            fontsize=10.0, fontweight='bold', color=c_black)
    ax.text(12.35, 7.34, r"Strict zero train/test overlap: $T_{f,\mathrm{train}} \cap T_{f,\mathrm{test}} = \emptyset$",
            ha='center', va='center', fontsize=7.8, color=c_dark)

    # Branch into Train vs Test
    draw_arrow(11.25, 7.18, 11.00, 6.55)
    draw_arrow(13.45, 7.18, 13.70, 6.55)

    # 3. Outer Train (80%) vs Outer Test (20%)
    # Outer Train
    draw_box(9.85, 4.85, 2.45, 1.70, fill=c_white, border=c_black, radius=0.10, lw=1.2)
    ax.text(11.07, 6.35, "Outer Train (80%)", ha='center', va='center',
            fontsize=9.0, fontweight='bold', color=c_black)
    ax.text(11.07, 6.15, r"$N \approx 3{,}811$ crystals", ha='center', va='center',
            fontsize=7.4, color=c_muted)
    # Inner dashed box
    draw_box(9.98, 4.96, 2.19, 1.05, fill=c_gray_bg, border=c_black, radius=0.06, lw=1.0, ls='--')
    ax.text(11.07, 5.82, "Inner Train (85%)", ha='center', va='center',
            fontsize=7.8, fontweight='bold', color=c_black)
    ax.text(11.07, 5.64, "AdamW optimization", ha='center', va='center',
            fontsize=7.0, color=c_muted)
    ax.text(11.07, 5.35, "Inner Val (15%)", ha='center', va='center',
            fontsize=7.8, fontweight='bold', color=c_black)
    ax.text(11.07, 5.14, "Checkpoint selection only", ha='center', va='center',
            fontsize=6.8, color=c_dark)

    # Outer Test
    draw_box(12.45, 4.85, 2.45, 1.70, fill=c_white, border=c_black, radius=0.10, lw=1.2)
    ax.text(13.67, 6.35, "Outer Test (20%)", ha='center', va='center',
            fontsize=9.0, fontweight='bold', color=c_black)
    ax.text(13.67, 6.15, r"$N = 953$ (or $952$) crystals", ha='center', va='center',
            fontsize=7.4, color=c_muted)
    ax.text(13.67, 5.75, "Evaluated ONCE only\non frozen weights", ha='center', va='center',
            fontsize=7.8, fontweight='bold', color=c_black, multialignment='center')
    draw_box(12.58, 4.98, 2.19, 0.44, fill=c_gray_bg, border=c_black, radius=0.06, lw=0.9)
    ax.text(13.67, 5.20, "Outer-test labels never\nused for checkpointing", ha='center', va='center',
            fontsize=6.6, color=c_dark, multialignment='center')

    # Arrows to Result
    draw_arrow(11.07, 4.85, 12.35, 4.28)
    draw_arrow(13.67, 4.85, 12.35, 4.28)

    # 4. Development Result Box
    draw_box(9.90, 3.65, 4.90, 0.63, fill=c_gray_bg, border=c_black, radius=0.08, lw=1.3)
    ax.text(12.35, 4.08, "Development-Stage 5-Fold Benchmark Result", ha='center', va='center',
            fontsize=9.0, fontweight='bold', color=c_black)
    ax.text(12.35, 3.84, r"$\mathbf{0.2909 \pm 0.0860}$ MAE  •  $\mathrm{MedAE} = 0.0631$  •  $\rho = 0.9395$",
            ha='center', va='center', fontsize=8.0, fontweight='bold', color=c_black)

    # Dividing line
    ax.plot([9.9, 14.8], [3.45, 3.45], color=c_muted, lw=0.9, ls=':', zorder=2)

    # 5. Lower Section: Chemical-System-Disjoint Stress Test
    ax.text(12.35, 3.20, "Grouped Extrapolation Stress Test", ha='center', va='center',
            fontsize=9.8, fontweight='bold', color=c_black)

    # Step 1: Frozen specification
    draw_box(10.05, 2.50, 4.60, 0.48, fill=c_white, border=c_black, radius=0.08, lw=1.1)
    ax.text(12.35, 2.80, "Frozen Model Specification", ha='center', va='center',
            fontsize=8.6, fontweight='bold', color=c_black)
    ax.text(12.35, 2.62, r"Fixed architecture, loss, hyperparameters, and $R_{\mathrm{cut}} = 6.0\,\mathrm{\AA}$",
            ha='center', va='center', fontsize=7.2, color=c_muted)

    draw_arrow(12.35, 2.50, 12.35, 2.15)

    # Step 2: GroupKFold
    draw_box(9.90, 1.55, 4.90, 0.60, fill=c_white, border=c_black, radius=0.08, lw=1.1)
    ax.text(12.35, 1.98, "Chemical-System GroupKFold Partition", ha='center', va='center',
            fontsize=8.8, fontweight='bold', color=c_black)
    ax.text(12.35, 1.76, "3,169 unique chemical systems  •  Zero train–test overlap",
            ha='center', va='center', fontsize=7.4, color=c_dark)
    ax.text(12.35, 1.63, r"Deterministic scikit-learn $\mathtt{shuffle=False}$",
            ha='center', va='center', fontsize=6.8, color=c_muted)

    draw_arrow(12.35, 1.55, 12.35, 1.25)

    # Step 3: Retraining
    draw_box(10.05, 0.76, 4.60, 0.49, fill=c_white, border=c_black, radius=0.08, lw=1.1)
    ax.text(12.35, 1.06, "Retrain Frozen Configuration per Grouped Fold", ha='center', va='center',
            fontsize=8.6, fontweight='bold', color=c_black)
    ax.text(12.35, 0.88, "5 models retrained from scratch  •  Fold-local normalization",
            ha='center', va='center', fontsize=7.2, color=c_muted)

    draw_arrow(12.35, 0.76, 12.35, 0.52)

    # Step 4: Grouped Evaluation Result
    draw_box(9.80, 0.06, 5.10, 0.46, fill=c_gray_bg, border=c_black, radius=0.08, lw=1.4)
    ax.text(12.35, 0.38, "Chemical-System-Disjoint Grouped Evaluation", ha='center', va='center',
            fontsize=8.8, fontweight='bold', color=c_black)
    ax.text(12.35, 0.22, r"$\mathbf{0.2973 \pm 0.0237}$ MAE  •  $\mathrm{MedAE} = 0.0692$  •  $\rho = 0.9331$",
            ha='center', va='center', fontsize=7.8, fontweight='bold', color=c_black)
    ax.text(12.35, 0.10, "(Modest +2.2% MAE difference relative to official-fold development)",
            ha='center', va='center', fontsize=6.8, color=c_muted)

    # Output paths
    out_dir_1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'manuscript_2', 'figures')
    out_dir_2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'manuscript', 'figures')
    art_dir = r"C:\Users\HP\.gemini\antigravity-ide\brain\7e1462b3-34c9-4d76-a8c1-ddab45f77d0b"

    for d in [out_dir_1, out_dir_2]:
        os.makedirs(d, exist_ok=True)
        plt.savefig(os.path.join(d, 'model_architecture_and_workflow.png'), dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
        plt.savefig(os.path.join(d, 'model_architecture_and_workflow.pdf'), dpi=300, bbox_inches='tight', facecolor='#FFFFFF')

    # Copy to artifacts
    shutil.copy(os.path.join(out_dir_1, 'model_architecture_and_workflow.png'),
                os.path.join(art_dir, 'model_architecture_and_workflow_clean_bw.png'))
    plt.close()
    print("Streamlined B&W Figure 1 successfully generated.")

if __name__ == '__main__':
    create_figure1_clean()
