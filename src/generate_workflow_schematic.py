"""
Script: generate_workflow_schematic.py
Author: Daksh Kaila
Description:
    Generates a publication-quality schematic figure (Figure 1) for the manuscript
    illustrating:
      (a) The DualHead-LogDirect GNN hierarchical architecture (Node, Edge, and Global channels,
          hierarchical message passing, and dual direct/log readout heads).
      (b) The official fold-local nested evaluation protocol and the chemical-system-disjoint
          GroupKFold stress test.
    Output: Saved as both high-resolution PNG (600 dpi) and vector PDF in manuscript/figures/
"""

import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle, PathPatch
import matplotlib.patheffects as patheffects

# Configure matplotlib for clean, publication-grade font rendering
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['mathtext.fontset'] = 'dejavusans'

def create_schematic():
    fig = plt.figure(figsize=(16, 10.2), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10.2)
    ax.axis('off')

    # Color Palette - 4 Muted Theme Colors + Neutrals
    # Neutral slate
    c_bg = '#FFFFFF'
    c_panel_bg = '#F8FAFC'
    c_panel_border = '#CBD5E1'
    c_text_dark = '#0F172A'
    c_text_muted = '#475569'

    # Theme colors
    # Blue: Node
    c_blue_border = '#2563EB'
    c_blue_fill = '#EFF6FF'
    c_blue_text = '#1E3A8A'

    # Orange: Edge
    c_orange_border = '#EA580C'
    c_orange_fill = '#FFF7ED'
    c_orange_text = '#9A3412'

    # Green: Global
    c_green_border = '#16A34A'
    c_green_fill = '#F0FDF4'
    c_green_text = '#14532D'

    # Purple: Readout & Blended Output
    c_purple_border = '#9333EA'
    c_purple_fill = '#FAF5FF'
    c_purple_text = '#581C87'

    # Callout / Badge colors
    c_badge_bg = '#FEF3C7'
    c_badge_border = '#D97706'
    c_badge_text = '#92400E'

    c_leak_badge_bg = '#FEE2E2'
    c_leak_badge_border = '#DC2626'
    c_leak_badge_text = '#991B1B'

    # Helper function for rounded boxes
    def draw_box(x, y, w, h, fill, border, radius=0.18, lw=1.5, ls='-'):
        box = FancyBboxPatch((x, y), w, h,
                             boxstyle=f"round,pad=0,rounding_size={radius}",
                             facecolor=fill, edgecolor=border, linewidth=lw, linestyle=ls, zorder=2)
        ax.add_patch(box)
        return box

    # Helper function for clean arrows
    def draw_arrow(x1, y1, x2, y2, color='#64748B', lw=1.8, style='-|>', mutation_scale=14):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                    shrinkA=2, shrinkB=2, mutation_scale=mutation_scale),
                    zorder=3)

    # Helper function for badges
    def draw_badge(x, y, text, fill=c_badge_bg, border=c_badge_border, text_color=c_badge_text, fontsize=7.2):
        lines = text.split('\n')
        max_len = max(len(l) for l in lines)
        w = max_len * 0.068 + 0.32
        h = 0.22 * len(lines) + 0.12
        draw_box(x - w/2, y - h/2, w, h, fill=fill, border=border, radius=0.06, lw=1.2)
        ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, fontweight='bold',
                color=text_color, zorder=4, multialignment='center')

    # -------------------------------------------------------------
    # PANEL A: Model Architecture (x: 0.4 to 9.6, y: 0.4 to 9.8)
    # -------------------------------------------------------------
    draw_box(0.4, 0.4, 9.2, 9.4, fill='#FFFFFF', border=c_panel_border, radius=0.25, lw=1.5)
    
    # Panel A Title
    ax.text(0.7, 9.45, "(a) DualHead–LogDirect GNN Architecture",
            fontsize=13.5, fontweight='bold', color=c_text_dark, va='center')

    # Top: Periodic Crystal Structure Box
    draw_box(2.8, 8.55, 4.4, 0.65, fill=c_panel_bg, border='#94A3B8', radius=0.12, lw=1.4)
    ax.text(5.0, 8.95, "Periodic Crystal Structure", ha='center', va='center',
            fontsize=10.5, fontweight='bold', color=c_text_dark)
    ax.text(5.0, 8.70, "Unit-cell lattice (a, b, c)  •  Fractional atom positions  •  Atomic numbers {Z_i}",
            ha='center', va='center', fontsize=8.5, color=c_text_muted)

    # Arrow to Graph Construction
    draw_arrow(5.0, 8.55, 5.0, 8.05)

    # Periodic Graph Construction
    draw_box(2.6, 7.45, 4.8, 0.60, fill='#F1F5F9', border='#64748B', radius=0.12, lw=1.4)
    ax.text(5.0, 7.82, "Periodic Graph Construction", ha='center', va='center',
            fontsize=10.5, fontweight='bold', color=c_text_dark)
    ax.text(5.0, 7.60, r"Cutoff $R_{\mathrm{cut}} = 6.0\,\mathrm{\AA}$  •  Truncated to top 12 nearest neighbors",
            ha='center', va='center', fontsize=8.5, color=c_text_muted)

    # Branching arrows to 3 channels
    draw_arrow(3.6, 7.45, 1.9, 6.75)  # to Node
    draw_arrow(5.0, 7.45, 5.0, 6.75)  # to Edge
    draw_arrow(6.4, 7.45, 8.1, 6.75)  # to Global

    # 1. Node Channel (Blue)
    draw_box(0.7, 4.85, 2.6, 1.90, fill=c_blue_fill, border=c_blue_border, radius=0.15, lw=1.6)
    ax.text(2.0, 6.52, "Atom Node Channel", ha='center', va='center',
            fontsize=10.5, fontweight='bold', color=c_blue_text)
    ax.text(2.0, 6.22, r"Learnable $Z \in [1, 100]$ (48-d)", ha='center', va='center',
            fontsize=8.5, color=c_blue_text)
    ax.text(2.0, 5.92, "+ 8 Tabulated Descriptors:", ha='center', va='center',
            fontsize=8.5, fontweight='bold', color=c_blue_text)
    node_desc = r"Pauling $X$, radius $r$, mass $m$," + "\n" + r"group $g$, row, 1st IE, $V_{\mathrm{mol}}$, $\mathrm{ox}_{\max}$"
    ax.text(2.0, 5.48, node_desc, ha='center', va='center',
            fontsize=8.0, color=c_blue_text, multialignment='center')
    ax.text(2.0, 5.05, r"$\mathbf{h}_i^{(0)} = [\mathbf{e}_{Z_i} \parallel \mathbf{p}_i] \in \mathbb{R}^{56}$",
            ha='center', va='center', fontsize=9.0, fontweight='bold', color=c_blue_text)

    # 2. Edge Channel (Orange)
    draw_box(3.5, 4.85, 3.0, 1.90, fill=c_orange_fill, border=c_orange_border, radius=0.15, lw=1.6)
    ax.text(5.0, 6.52, "Pair & Triplet Edge Channel", ha='center', va='center',
            fontsize=10.5, fontweight='bold', color=c_orange_text)
    ax.text(5.0, 6.22, r"41 Gaussian RBFs ($\sigma = 0.15\,\mathrm{\AA}$)", ha='center', va='center',
            fontsize=8.5, color=c_orange_text)
    ax.text(5.0, 5.92, r"+ Inverse distance $1 / (r_{ij} + 0.01)$", ha='center', va='center',
            fontsize=8.5, color=c_orange_text)
    ax.text(5.0, 5.58, r"+ 16 Chebyshev angle features" + "\n" + r"$\cos\theta_{jik} = \hat{\mathbf{r}}_{ij} \cdot \hat{\mathbf{r}}_{ik}$ averaged $\forall k \neq j$",
            ha='center', va='center', fontsize=8.0, color=c_orange_text, multialignment='center')
    ax.text(5.0, 5.05, r"$\mathbf{e}_{ij}^{(0)} \in \mathbb{R}^{58}$  (Vertex: atom $i$)",
            ha='center', va='center', fontsize=9.0, fontweight='bold', color=c_orange_text)

    # 3. Global Channel (Green)
    draw_box(6.7, 4.85, 2.6, 1.90, fill=c_green_fill, border=c_green_border, radius=0.15, lw=1.6)
    ax.text(8.0, 6.52, "Global Descriptor Channel", ha='center', va='center',
            fontsize=10.5, fontweight='bold', color=c_green_text)
    ax.text(8.0, 6.24, r"12-d Structural–Compositional:", ha='center', va='center',
            fontsize=8.2, fontweight='bold', color=c_green_text)
    global_desc = r"$\rho, V/N, \phi, N_{\mathrm{elem}}, \bar{X}, \Delta X$," + "\n" + \
                  r"$\bar{r}, \bar{g}, \bar{V}_{\mathrm{mol}}, \bar{z}, \eta_{\mathrm{pack}}, c/a$"
    ax.text(8.0, 5.80, global_desc, ha='center', va='center',
            fontsize=8.0, color=c_green_text, multialignment='center')
    ax.text(8.0, 5.40, r"$\eta_{\mathrm{pack}} = \phi (V/N)/\bar{r}^3$ (dimensionless)",
            ha='center', va='center', fontsize=7.5, color=c_green_text)
    # Badge: Fold-local scaling only (inside green box)
    draw_badge(8.0, 5.08, "Fold-local scaling only", fill='#DCFCE7', border='#16A34A', text_color='#166534', fontsize=7.2)

    # Convergence arrows to Hierarchical Block
    draw_arrow(2.0, 4.85, 3.8, 4.05)
    draw_arrow(5.0, 4.85, 5.0, 4.05)
    draw_arrow(8.0, 4.85, 6.2, 4.05)

    # Hierarchical Interaction Block (Repeated x3)
    draw_box(1.2, 2.70, 7.6, 1.35, fill='#F8FAFC', border='#475569', radius=0.18, lw=1.8)
    ax.text(5.0, 3.82, "Three Hierarchical Interaction Layers  [LayerNorm + SiLU]",
            ha='center', va='center', fontsize=10.5, fontweight='bold', color=c_text_dark)
    layer_eqs = r"1. Edge update:  $\mathbf{e}_{ij}^{(l+1)} = \mathbf{e}_{ij}^{(l)} + \mathrm{MLP}_{\mathbf{e}}([\mathbf{h}_i^{(l)} \parallel \mathbf{h}_j^{(l)} \parallel \mathbf{e}_{ij}^{(l)} \parallel \mathbf{u}^{(l)}])$" + "\n" + \
                r"2. Message sum:  $\mathbf{m}_i^{(l+1)} = \sum_{j \in \mathcal{N}(i)} \mathbf{e}_{ij}^{(l+1)}$  $\rightarrow$  Node residual:  $\mathbf{h}_i^{(l+1)} = \mathbf{h}_i^{(l)} + \mathrm{MLP}_{\mathbf{v}}([\mathbf{h}_i^{(l)} \parallel \mathbf{m}_i^{(l+1)} \parallel \mathbf{u}^{(l)}])$" + "\n" + \
                r"3. Global state update:  $\mathbf{u}^{(l+1)} = \mathbf{u}^{(l)} + \mathrm{MLP}_{\mathbf{u}}\left(\left[\frac{1}{N}\sum_i \mathbf{h}_i^{(l+1)} \parallel \mathbf{u}^{(l)}\right]\right)$"
    ax.text(5.0, 3.20, layer_eqs, ha='center', va='center', fontsize=8.2, color=c_text_dark, multialignment='center')

    # Arrow to Crystal Pooling
    draw_arrow(5.0, 2.70, 5.0, 2.30)

    # Crystal Representation Pooling
    draw_box(2.2, 1.85, 5.6, 0.45, fill='#F1F5F9', border='#64748B', radius=0.10, lw=1.4)
    ax.text(5.0, 2.07, r"Crystal Representation:  $\mathbf{h}_{\mathrm{crystal}} = \left[\frac{1}{N}\sum_{i=1}^N \mathbf{h}_i^{(3)} \parallel \mathbf{u}^{(3)}\right] \in \mathbb{R}^{76}$",
            ha='center', va='center', fontsize=9.2, fontweight='bold', color=c_text_dark)

    # Branching arrows to Dual Readout Heads
    draw_arrow(3.6, 1.85, 2.5, 1.50)
    draw_arrow(6.4, 1.85, 7.5, 1.50)

    # Dual Heads (Purple)
    # Direct Head (Left)
    draw_box(0.7, 0.75, 3.6, 0.75, fill=c_purple_fill, border=c_purple_border, radius=0.12, lw=1.5)
    ax.text(2.5, 1.28, "Direct Readout Head", ha='center', va='center',
            fontsize=9.5, fontweight='bold', color=c_purple_text)
    ax.text(2.5, 1.05, r"$\hat{n}_{\mathrm{direct}} = 1.0 + \mathrm{Softplus}(\mathbf{W}_d \mathbf{h}_{\mathrm{crystal}} + b_d)$",
            ha='center', va='center', fontsize=8.5, color=c_purple_text)
    ax.text(2.5, 0.86, r"Guaranteed bound:  $\hat{n}_{\mathrm{direct}} \geq 1.0$",
            ha='center', va='center', fontsize=8.0, fontweight='bold', color='#059669')

    # Logarithmic Head (Right)
    draw_box(5.7, 0.75, 3.6, 0.75, fill=c_purple_fill, border=c_purple_border, radius=0.12, lw=1.5)
    ax.text(7.5, 1.28, "Logarithmic Readout Head", ha='center', va='center',
            fontsize=9.5, fontweight='bold', color=c_purple_text)
    ax.text(7.5, 1.05, r"$\hat{z}_{\mathrm{log}} = \mathbf{W}_\ell \mathbf{h}_{\mathrm{crystal}} + b_\ell \approx \ln(n - 0.99)$",
            ha='center', va='center', fontsize=8.5, color=c_purple_text)
    ax.text(7.5, 0.86, r"Inverted prediction:  $\hat{n}_{\log} = 0.99 + \exp(\hat{z}_{\mathrm{log}}) > 0.99$",
            ha='center', va='center', fontsize=8.0, color=c_purple_text)

    # Converging arrows to Blended Output
    draw_arrow(3.6, 0.75, 4.4, 0.52)
    draw_arrow(6.4, 0.75, 5.6, 0.52)

    # Blended Prediction Box
    draw_box(3.2, 0.12, 3.6, 0.40, fill='#FAF5FF', border='#7E22CE', radius=0.08, lw=1.6)
    ax.text(5.0, 0.32, r"Blended Prediction:  $\hat{n} = \frac{1}{2}(\hat{n}_{\mathrm{direct}} + \hat{n}_{\log}) \geq 1.0$",
            ha='center', va='center', fontsize=9.0, fontweight='bold', color='#581C87')

    # -------------------------------------------------------------
    # PANEL B: Evaluation Protocol (x: 10.0 to 15.6, y: 0.4 to 9.8)
    # -------------------------------------------------------------
    draw_box(10.0, 0.4, 5.6, 9.4, fill='#FFFFFF', border=c_panel_border, radius=0.25, lw=1.5)

    # Panel B Title
    ax.text(10.3, 9.45, "(b) Evaluation Protocols and Leakage Controls",
            fontsize=13.5, fontweight='bold', color=c_text_dark, va='center')

    # Top: MatBench Dielectric Benchmark
    draw_box(10.6, 8.55, 4.4, 0.65, fill=c_panel_bg, border='#94A3B8', radius=0.12, lw=1.4)
    ax.text(12.8, 8.95, "Official MatBench Benchmark", ha='center', va='center',
            fontsize=10.5, fontweight='bold', color=c_text_dark)
    ax.text(12.8, 8.70, r"$\mathtt{matbench\_dielectric}$  •  $N = 4{,}764$ inorganic crystals",
            ha='center', va='center', fontsize=8.5, color=c_text_muted)

    # Arrow to 5-Fold Partition
    draw_arrow(12.8, 8.55, 12.8, 8.05)

    # 5-Fold Outer Partition Box
    draw_box(10.5, 7.45, 4.6, 0.60, fill='#F1F5F9', border='#64748B', radius=0.12, lw=1.4)
    ax.text(12.8, 7.82, "Official Five Outer Folds Partition", ha='center', va='center',
            fontsize=10.5, fontweight='bold', color=c_text_dark)
    ax.text(12.8, 7.60, r"Zero train/test overlap: $T_{f,\mathrm{train}} \cap T_{f,\mathrm{test}} = \emptyset$ $\forall f \in \{0..4\}$",
            ha='center', va='center', fontsize=8.2, color=c_text_muted)

    # Branch into Outer Train vs Outer Test
    draw_arrow(11.8, 7.45, 11.5, 6.85)
    draw_arrow(13.8, 7.45, 14.1, 6.85)

    # Outer Train (80%, N ~ 3,811)
    draw_box(10.3, 4.90, 2.5, 1.95, fill='#F8FAFC', border='#3B82F6', radius=0.12, lw=1.5)
    ax.text(11.55, 6.65, "Outer Train Fold (80%)", ha='center', va='center',
            fontsize=9.2, fontweight='bold', color='#1D4ED8')
    ax.text(11.55, 6.42, r"$N \approx 3{,}811$ crystals", ha='center', va='center',
            fontsize=7.8, color='#1E40AF')
    
    # Nested 85 / 15 inner partition
    draw_box(10.45, 5.02, 2.2, 1.25, fill='#EFF6FF', border='#60A5FA', radius=0.08, lw=1.2, ls='--')
    ax.text(11.55, 6.08, "Inner Train (85%)", ha='center', va='center',
            fontsize=8.2, fontweight='bold', color='#1E3A8A')
    ax.text(11.55, 5.88, "Model optimization (AdamW)", ha='center', va='center',
            fontsize=7.2, color='#1E40AF')
    ax.text(11.55, 5.50, "Inner Validation (15%)", ha='center', va='center',
            fontsize=8.2, fontweight='bold', color='#9333EA')
    ax.text(11.55, 5.25, "Checkpoint selection ONLY\n(Epoch w/ min val loss)", ha='center', va='center',
            fontsize=7.0, fontweight='bold', color='#9333EA', multialignment='center')

    # Outer Test (20%, N = 953)
    draw_box(13.0, 4.90, 2.4, 1.95, fill='#FEF2F2', border='#EF4444', radius=0.12, lw=1.5)
    ax.text(14.2, 6.65, "Outer Test Fold (20%)", ha='center', va='center',
            fontsize=9.2, fontweight='bold', color='#B91C1C')
    ax.text(14.2, 6.42, r"$N = 953$ (or $952$) crystals", ha='center', va='center',
            fontsize=7.8, color='#991B1B')
    ax.text(14.2, 6.08, "Evaluated ONCE only", ha='center', va='center',
            fontsize=8.2, fontweight='bold', color='#B91C1C')
    ax.text(14.2, 5.88, "using frozen inner-val\ncheckpoint weights", ha='center', va='center',
            fontsize=7.2, color='#7F1D1D', multialignment='center')

    # Leakage badge (fits cleanly within box width 2.4)
    draw_badge(14.2, 5.25, "Outer-test labels never\nused for checkpoint selection",
               fill=c_leak_badge_bg, border=c_leak_badge_border, text_color=c_leak_badge_text, fontsize=6.5)

    # Arrow to official fold result
    draw_arrow(11.55, 4.90, 12.8, 4.45)
    draw_arrow(14.2, 4.90, 12.8, 4.45)

    # Official Development Result Box
    draw_box(10.5, 3.80, 4.6, 0.65, fill='#F0FDF4', border='#16A34A', radius=0.10, lw=1.5)
    ax.text(12.8, 4.25, "Development-Stage Official 5-Fold Result", ha='center', va='center',
            fontsize=9.5, fontweight='bold', color='#14532D')
    ax.text(12.8, 4.00, r"$\mathbf{0.2909 \pm 0.0860}$ MAE  •  $\mathrm{MedAE} = 0.0631$  •  $\rho = 0.9395$",
            ha='center', va='center', fontsize=8.5, fontweight='bold', color='#166534')

    # Dividing separator
    ax.plot([10.3, 15.3], [3.60, 3.60], color='#CBD5E1', lw=1.2, ls=':', zorder=2)

    # Lower Section: Chemical-System-Disjoint Stress Test
    ax.text(12.8, 3.40, "Grouped Extrapolation Stress Test", ha='center', va='center',
            fontsize=10.5, fontweight='bold', color='#1E293B')

    # Step 1: Frozen Model Specification
    draw_box(10.6, 2.72, 4.4, 0.48, fill='#F1F5F9', border='#64748B', radius=0.08, lw=1.3)
    ax.text(12.8, 2.96, "Frozen Model Specification", ha='center', va='center',
            fontsize=9.0, fontweight='bold', color='#1E293B')
    ax.text(12.8, 2.82, r"Fixed architecture, loss, hyperparameters, and $R_{\mathrm{cut}} = 6.0\,\mathrm{\AA}$",
            ha='center', va='center', fontsize=7.2, color='#475569')

    draw_arrow(12.8, 2.72, 12.8, 2.45)

    # Step 2: Chemical-System GroupKFold Box
    draw_box(10.5, 1.85, 4.6, 0.60, fill='#EEF2FF', border='#6366F1', radius=0.10, lw=1.4)
    ax.text(12.8, 2.28, "Chemical-System GroupKFold Partition", ha='center', va='center',
            fontsize=9.2, fontweight='bold', color='#4338CA')
    ax.text(12.8, 2.08, "3,169 systems  •  Zero train–test chemical-system overlap",
            ha='center', va='center', fontsize=7.8, color='#3730A3')
    ax.text(12.8, 1.94, r"scikit-learn 1.9.0  •  $\mathtt{shuffle=False}$ (deterministic)",
            ha='center', va='center', fontsize=7.0, color='#475569')

    draw_arrow(12.8, 1.85, 12.8, 1.55)

    # Step 3: Retrain Frozen Configuration
    draw_box(10.6, 1.05, 4.4, 0.50, fill='#EFF6FF', border='#2563EB', radius=0.08, lw=1.3)
    ax.text(12.8, 1.37, "Retrain Frozen Configuration per Grouped Fold", ha='center', va='center',
            fontsize=8.8, fontweight='bold', color='#1E40AF')
    ax.text(12.8, 1.18, "5 models retrained from scratch  •  Fold-local normalization",
            ha='center', va='center', fontsize=7.4, color='#1E3A8A')

    draw_arrow(12.8, 1.05, 12.8, 0.75)

    # Step 4: Grouped Result Box
    draw_box(10.3, 0.12, 5.0, 0.63, fill='#FAF5FF', border='#9333EA', radius=0.10, lw=1.6)
    ax.text(12.8, 0.56, "Chemical-System-Disjoint Grouped Evaluation", ha='center', va='center',
            fontsize=9.5, fontweight='bold', color='#581C87')
    ax.text(12.8, 0.36, r"$\mathbf{0.2973 \pm 0.0237}$ MAE  •  $\mathrm{MedAE} = 0.0692$  •  $\rho = 0.9331$",
            ha='center', va='center', fontsize=8.5, fontweight='bold', color='#6B21A8')
    ax.text(12.8, 0.20, "(Modest +2.2% MAE difference relative to official-fold development)",
            ha='center', va='center', fontsize=7.2, color='#7E22CE')

    # Save to outputs
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'manuscript', 'figures')
    os.makedirs(out_dir, exist_ok=True)
    
    png_path = os.path.join(out_dir, 'model_architecture_and_workflow.png')
    pdf_path = os.path.join(out_dir, 'model_architecture_and_workflow.pdf')

    plt.savefig(png_path, dpi=600, bbox_inches='tight', facecolor='#FFFFFF', edgecolor='none')
    plt.savefig(pdf_path, dpi=600, bbox_inches='tight', facecolor='#FFFFFF', edgecolor='none')
    plt.close()

    print(f"Generated schematic successfully:")
    print(f"  PNG: {png_path} ({os.path.getsize(png_path):,} bytes)")
    print(f"  PDF: {pdf_path} ({os.path.getsize(pdf_path):,} bytes)")

if __name__ == '__main__':
    create_schematic()
