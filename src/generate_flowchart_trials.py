"""
Script: generate_flowchart_trials.py
Author: Daksh Kaila
Description:
    Generates three clean, uncluttered, publication-grade trial variations
    for Figure 1 (Workflow & Architecture Schematic) without rainbow colors
    and without overloaded text/equations.
    
    Trial 1: Minimalist Slate & Navy (Nature / Clean Academic Style)
    Trial 2: Ultra-Clean Monochromatic with Sapphire Accent (Physical Review Style)
    Trial 3: Modern Informatics Two-Tone (Cohesive Indigo Architecture + Muted Teal Evaluation)
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

def draw_rounded_box(ax, x, y, w, h, fill, border, radius=0.14, lw=1.3, ls='-'):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle=f"round,pad=0,rounding_size={radius}",
                         facecolor=fill, edgecolor=border, linewidth=lw, linestyle=ls, zorder=2)
    ax.add_patch(box)
    return box

def draw_arrow(ax, x1, y1, x2, y2, color='#475569', lw=1.5, style='-|>', mutation_scale=12):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                shrinkA=2, shrinkB=2, mutation_scale=mutation_scale),
                zorder=3)

# ==============================================================================
# TRIAL 1: Minimalist Slate & Navy (Clean Academic / Nature Style)
# ==============================================================================
def generate_trial_1(output_png, output_pdf):
    fig = plt.figure(figsize=(15.5, 9.6), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 15.5)
    ax.set_ylim(0, 9.6)
    ax.axis('off')

    # Palette
    c_bg = '#FFFFFF'
    c_panel_bg = '#F8FAFC'
    c_panel_border = '#CBD5E1'
    c_box_fill = '#FFFFFF'
    c_box_border = '#1E3A8A'
    c_box_border_light = '#94A3B8'
    c_text_dark = '#0F172A'
    c_text_muted = '#475569'
    c_navy = '#1E3A8A'
    c_accent_fill = '#EFF6FF'
    c_accent_border = '#2563EB'

    # Outer Panels
    # Panel A: Architecture (x: 0.5 to 9.2)
    draw_rounded_box(ax, 0.5, 0.4, 8.7, 8.8, fill=c_panel_bg, border=c_panel_border, radius=0.20, lw=1.4)
    ax.text(0.8, 8.85, "(a) DualHead–LogDirect GNN Architecture", fontsize=13, fontweight='bold', color=c_text_dark, va='center')

    # 1. Input: Crystal Structure
    draw_rounded_box(ax, 2.6, 7.95, 4.5, 0.65, fill=c_box_fill, border=c_box_border_light, radius=0.10, lw=1.2)
    ax.text(4.85, 8.35, "Periodic Crystal Structure", ha='center', va='center', fontsize=10.5, fontweight='bold', color=c_text_dark)
    ax.text(4.85, 8.12, "Unit-cell lattice  •  Atomic coordinates  •  Species {Z_i}", ha='center', va='center', fontsize=8.2, color=c_text_muted)

    draw_arrow(ax, 4.85, 7.95, 4.85, 7.45, color=c_navy)

    # 2. Graph Construction
    draw_rounded_box(ax, 2.5, 6.85, 4.7, 0.60, fill=c_box_fill, border=c_navy, radius=0.10, lw=1.3)
    ax.text(4.85, 7.22, "Periodic Graph Construction", ha='center', va='center', fontsize=10.5, fontweight='bold', color=c_navy)
    ax.text(4.85, 7.00, r"Radial cutoff $R_{\mathrm{cut}} = 6.0\,\mathrm{\AA}$  •  Top 12 nearest neighbors", ha='center', va='center', fontsize=8.2, color=c_text_muted)

    # Branching arrows
    draw_arrow(ax, 3.4, 6.85, 1.9, 6.20, color=c_navy)
    draw_arrow(ax, 4.85, 6.85, 4.85, 6.20, color=c_navy)
    draw_arrow(ax, 6.3, 6.85, 7.8, 6.20, color=c_navy)

    # 3. Three Parallel Channels (All unified in Navy/Slate)
    # Node Channel
    draw_rounded_box(ax, 0.7, 4.75, 2.4, 1.45, fill=c_box_fill, border=c_navy, radius=0.12, lw=1.4)
    ax.text(1.9, 5.95, "Atom Nodes (56-d)", ha='center', va='center', fontsize=9.8, fontweight='bold', color=c_navy)
    ax.text(1.9, 5.62, "• Learned Z embeddings (48-d)\n• 8 elemental ground-state\n  priors (radii, electronegativity,\n  molar volume, ionization)",
            ha='center', va='center', fontsize=7.6, color=c_text_muted)

    # Edge Channel
    draw_rounded_box(ax, 3.4, 4.75, 2.9, 1.45, fill=c_box_fill, border=c_navy, radius=0.12, lw=1.4)
    ax.text(4.85, 5.95, "Directed Edges (58-d)", ha='center', va='center', fontsize=9.8, fontweight='bold', color=c_navy)
    ax.text(4.85, 5.62, "• 41 Gaussian RBF distances\n• Inverse distance channel\n• 16 Chebyshev bond-angle\n  projections (3-body geometry)",
            ha='center', va='center', fontsize=7.6, color=c_text_muted)

    # Global Channel
    draw_rounded_box(ax, 6.6, 4.75, 2.4, 1.45, fill=c_box_fill, border=c_navy, radius=0.12, lw=1.4)
    ax.text(7.8, 5.95, "Global State (12-d)", ha='center', va='center', fontsize=9.8, fontweight='bold', color=c_navy)
    ax.text(7.8, 5.62, "• Density & packing fraction\n• Mean atomic volume\n• Coordination statistics\n• Unit-cell aspect ratio",
            ha='center', va='center', fontsize=7.6, color=c_text_muted)

    # Converging arrows
    draw_arrow(ax, 1.9, 4.75, 3.6, 4.10, color=c_navy)
    draw_arrow(ax, 4.85, 4.75, 4.85, 4.10, color=c_navy)
    draw_arrow(ax, 7.8, 4.75, 6.1, 4.10, color=c_navy)

    # 4. Hierarchical Interaction Layers
    draw_rounded_box(ax, 1.2, 2.80, 7.3, 1.30, fill=c_box_fill, border=c_navy, radius=0.14, lw=1.6)
    ax.text(4.85, 3.82, "Three Hierarchical Interaction Layers", ha='center', va='center', fontsize=10.5, fontweight='bold', color=c_navy)
    ax.text(4.85, 3.55, "Bidirectional coupling with LayerNorm & SiLU activations:", ha='center', va='center', fontsize=8.2, color=c_text_dark)
    ax.text(4.85, 3.25, "1. Edge states updated conditioned on incident nodes and global state\n2. Node representations updated via unnormalized message aggregation\n3. Global state updated conditioned on pooled node representations",
            ha='center', va='center', fontsize=7.7, color=c_text_muted)

    draw_arrow(ax, 4.85, 2.80, 4.85, 2.30, color=c_navy)

    # 5. Dual Readout Heads
    draw_rounded_box(ax, 0.7, 1.15, 3.9, 1.15, fill=c_box_fill, border=c_navy, radius=0.12, lw=1.3)
    ax.text(2.65, 2.05, "Direct Readout Head", ha='center', va='center', fontsize=9.5, fontweight='bold', color=c_navy)
    ax.text(2.65, 1.75, r"$\hat{n}_{\mathrm{direct}} = 1.0 + \mathrm{Softplus}(\mathbf{W}_d \mathbf{h}_{\mathrm{crystal}} + b_d)$", ha='center', va='center', fontsize=8.2, color=c_text_dark)
    ax.text(2.65, 1.42, "Guarantees physical lower bound:  " + r"$\hat{n}_{\mathrm{direct}} \geq 1.0$", ha='center', va='center', fontsize=7.8, fontweight='bold', color='#15803D')

    draw_rounded_box(ax, 5.1, 1.15, 3.9, 1.15, fill=c_box_fill, border=c_navy, radius=0.12, lw=1.3)
    ax.text(7.05, 2.05, "Logarithmic Readout Head", ha='center', va='center', fontsize=9.5, fontweight='bold', color=c_navy)
    ax.text(7.05, 1.75, r"$\hat{z}_{\mathrm{log}} = \mathbf{W}_\ell \mathbf{h}_{\mathrm{crystal}} + b_\ell \approx \ln(n - 0.99)$", ha='center', va='center', fontsize=8.2, color=c_text_dark)
    ax.text(7.05, 1.42, "Compresses extreme high-index dynamic range", ha='center', va='center', fontsize=7.8, color=c_text_muted)

    # Arrows to Blended Output
    draw_arrow(ax, 2.65, 1.15, 4.0, 0.80, color=c_navy)
    draw_arrow(ax, 7.05, 1.15, 5.7, 0.80, color=c_navy)

    # 6. Blended Output
    draw_rounded_box(ax, 2.7, 0.50, 4.3, 0.50, fill=c_accent_fill, border=c_accent_border, radius=0.10, lw=1.6)
    ax.text(4.85, 0.75, r"Blended Prediction:  $\hat{n} = \frac{1}{2}(\hat{n}_{\mathrm{direct}} + \hat{n}_{\log}) \geq 1.0$",
            ha='center', va='center', fontsize=9.2, fontweight='bold', color=c_navy)

    # --------------------------------------------------------------------------
    # Panel B: Evaluation Protocols (x: 9.6 to 15.0)
    # --------------------------------------------------------------------------
    draw_rounded_box(ax, 9.6, 0.4, 5.4, 8.8, fill=c_panel_bg, border=c_panel_border, radius=0.20, lw=1.4)
    ax.text(9.9, 8.85, "(b) Evaluation Protocols", fontsize=13, fontweight='bold', color=c_text_dark, va='center')

    # Stream 1: MatBench Official Protocol
    draw_rounded_box(ax, 9.9, 7.95, 4.8, 0.65, fill=c_box_fill, border=c_box_border_light, radius=0.10, lw=1.2)
    ax.text(12.3, 8.35, "Official MatBench Benchmark", ha='center', va='center', fontsize=10.0, fontweight='bold', color=c_text_dark)
    ax.text(12.3, 8.12, r"$\mathtt{matbench\_dielectric}$  •  $N = 4{,}764$ crystals", ha='center', va='center', fontsize=8.2, color=c_text_muted)

    draw_arrow(ax, 12.3, 7.95, 12.3, 7.45, color=c_navy)

    draw_rounded_box(ax, 9.9, 6.70, 4.8, 0.75, fill=c_box_fill, border=c_navy, radius=0.10, lw=1.3)
    ax.text(12.3, 7.22, "Strict Nested Cross-Validation (5 Folds)", ha='center', va='center', fontsize=9.5, fontweight='bold', color=c_navy)
    ax.text(12.3, 6.95, "• 80% Outer Train (85% inner-train / 15% inner-val for checkpoints)\n• 20% Outer Test: Evaluated ONCE on frozen checkpoint weights",
            ha='center', va='center', fontsize=7.4, color=c_text_muted)

    draw_arrow(ax, 12.3, 6.70, 12.3, 6.10, color=c_navy)

    # Result Box 1
    draw_rounded_box(ax, 9.9, 5.35, 4.8, 0.75, fill='#F0FDF4', border='#16A34A', radius=0.10, lw=1.4)
    ax.text(12.3, 5.85, "Development-Stage 5-Fold Result", ha='center', va='center', fontsize=9.2, fontweight='bold', color='#14532D')
    ax.text(12.3, 5.55, r"$\mathbf{0.2909 \pm 0.0860}$ MAE  •  $\mathrm{MedAE} = 0.0631$  •  $\rho = 0.9395$",
            ha='center', va='center', fontsize=8.2, fontweight='bold', color='#15803D')

    # Separator Line
    ax.plot([9.9, 14.7], [4.95, 4.95], color='#CBD5E1', lw=1.0, ls='--')

    # Stream 2: Chemical-System-Disjoint Test
    ax.text(12.3, 4.60, "Chemical-System-Disjoint Stress Test", ha='center', va='center', fontsize=10.5, fontweight='bold', color=c_navy)

    draw_rounded_box(ax, 9.9, 3.60, 4.8, 0.75, fill=c_box_fill, border=c_navy, radius=0.10, lw=1.3)
    ax.text(12.3, 4.12, "GroupKFold Partition (3,169 Unique Chemistries)", ha='center', va='center', fontsize=9.2, fontweight='bold', color=c_navy)
    ax.text(12.3, 3.85, "• Zero chemical-system overlap between train and test partitions\n• Prevents identical chemical compositions across folds",
            ha='center', va='center', fontsize=7.4, color=c_text_muted)

    draw_arrow(ax, 12.3, 3.60, 12.3, 3.00, color=c_navy)

    draw_rounded_box(ax, 9.9, 2.25, 4.8, 0.75, fill=c_box_fill, border=c_navy, radius=0.10, lw=1.3)
    ax.text(12.3, 2.77, "Retrain Frozen Configuration from Scratch", ha='center', va='center', fontsize=9.2, fontweight='bold', color=c_navy)
    ax.text(12.3, 2.50, "• Identical architecture, hyperparameters, and inner-val rule\n• Evaluates robustness on completely unseen chemical spaces",
            ha='center', va='center', fontsize=7.4, color=c_text_muted)

    draw_arrow(ax, 12.3, 2.25, 12.3, 1.65, color=c_navy)

    # Result Box 2
    draw_rounded_box(ax, 9.9, 0.65, 4.8, 1.00, fill=c_accent_fill, border=c_accent_border, radius=0.12, lw=1.5)
    ax.text(12.3, 1.38, "Chemical-System-Disjoint Evaluation", ha='center', va='center', fontsize=9.2, fontweight='bold', color=c_navy)
    ax.text(12.3, 1.10, r"$\mathbf{0.2973 \pm 0.0237}$ MAE  •  $\mathrm{MedAE} = 0.0692$  •  $\rho = 0.9331$",
            ha='center', va='center', fontsize=8.2, fontweight='bold', color=c_navy)
    ax.text(12.3, 0.85, "(Only 2.2% higher MAE than official-fold estimate)", ha='center', va='center', fontsize=7.4, color='#1E40AF')

    plt.savefig(output_png, dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.savefig(output_pdf, dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close()
    print(f"Trial 1 generated: {output_png}")

# ==============================================================================
# TRIAL 2: Monochromatic High-Contrast with Sapphire Accent (Physical Review Style)
# ==============================================================================
def generate_trial_2(output_png, output_pdf):
    fig = plt.figure(figsize=(15.5, 9.6), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 15.5)
    ax.set_ylim(0, 9.6)
    ax.axis('off')

    # Monochrome Palette with Sapphire Accent
    c_panel_bg = '#FFFFFF'
    c_panel_border = '#0F172A'
    c_box_fill = '#FFFFFF'
    c_box_border = '#334155'
    c_text_dark = '#0F172A'
    c_text_muted = '#475569'
    c_accent_blue = '#1D4ED8'
    c_accent_fill = '#F8FAFC'

    # Outer Panels
    # Panel A
    draw_rounded_box(ax, 0.5, 0.4, 8.7, 8.8, fill=c_panel_bg, border=c_panel_border, radius=0.16, lw=1.6)
    ax.text(0.8, 8.85, "(a) Architecture: DualHead–LogDirect GNN", fontsize=12.5, fontweight='bold', color=c_text_dark, va='center')

    # 1. Input Box
    draw_rounded_box(ax, 2.5, 7.95, 4.7, 0.65, fill=c_box_fill, border=c_box_border, radius=0.08, lw=1.2)
    ax.text(4.85, 8.35, "Periodic Crystal Input", ha='center', va='center', fontsize=10.0, fontweight='bold', color=c_text_dark)
    ax.text(4.85, 8.12, "Lattice vectors  •  Fractional atomic coordinates  •  Species {Z}", ha='center', va='center', fontsize=8.0, color=c_text_muted)

    draw_arrow(ax, 4.85, 7.95, 4.85, 7.45, color=c_text_dark)

    # 2. Graph Construction
    draw_rounded_box(ax, 2.5, 6.85, 4.7, 0.60, fill=c_box_fill, border=c_box_border, radius=0.08, lw=1.2)
    ax.text(4.85, 7.22, "Periodic Graph Construction", ha='center', va='center', fontsize=10.0, fontweight='bold', color=c_text_dark)
    ax.text(4.85, 7.00, r"Cutoff $R_{\mathrm{cut}} = 6.0\,\mathrm{\AA}$  •  12 nearest neighbors", ha='center', va='center', fontsize=8.0, color=c_text_muted)

    draw_arrow(ax, 3.4, 6.85, 1.9, 6.20, color=c_text_dark)
    draw_arrow(ax, 4.85, 6.85, 4.85, 6.20, color=c_text_dark)
    draw_arrow(ax, 6.3, 6.85, 7.8, 6.20, color=c_text_dark)

    # 3. Channels
    draw_rounded_box(ax, 0.7, 4.75, 2.4, 1.45, fill=c_box_fill, border=c_box_border, radius=0.08, lw=1.2)
    ax.text(1.9, 5.95, "Node Prior Channel", ha='center', va='center', fontsize=9.2, fontweight='bold', color=c_text_dark)
    ax.text(1.9, 5.58, "• Learned Z embedding (48-d)\n• 8 elemental ground-state\n  properties (radii, Pauling X,\n  ionization, volume)",
            ha='center', va='center', fontsize=7.4, color=c_text_muted)

    draw_rounded_box(ax, 3.4, 4.75, 2.9, 1.45, fill=c_box_fill, border=c_box_border, radius=0.08, lw=1.2)
    ax.text(4.85, 5.95, "Edge Geometry Channel", ha='center', va='center', fontsize=9.2, fontweight='bold', color=c_text_dark)
    ax.text(4.85, 5.58, "• 41 Gaussian distance RBFs\n• 1/(r + 0.01) inverse channel\n• 16 Chebyshev bond-angle\n  projections (3-body)",
            ha='center', va='center', fontsize=7.4, color=c_text_muted)

    draw_rounded_box(ax, 6.6, 4.75, 2.4, 1.45, fill=c_box_fill, border=c_box_border, radius=0.08, lw=1.2)
    ax.text(7.8, 5.95, "Global State Channel", ha='center', va='center', fontsize=9.2, fontweight='bold', color=c_text_dark)
    ax.text(7.8, 5.58, "• Macroscopic density\n• Atomic packing fraction\n• Mean coordination\n• Cell aspect ratio c/a",
            ha='center', va='center', fontsize=7.4, color=c_text_muted)

    draw_arrow(ax, 1.9, 4.75, 3.6, 4.10, color=c_text_dark)
    draw_arrow(ax, 4.85, 4.75, 4.85, 4.10, color=c_text_dark)
    draw_arrow(ax, 7.8, 4.75, 6.1, 4.10, color=c_text_dark)

    # 4. Hierarchical Interaction
    draw_rounded_box(ax, 1.2, 2.75, 7.3, 1.35, fill=c_accent_fill, border=c_accent_blue, radius=0.10, lw=1.5)
    ax.text(4.85, 3.82, "Hierarchical Interaction Network  (3 Layers)", ha='center', va='center', fontsize=10.0, fontweight='bold', color=c_accent_blue)
    ax.text(4.85, 3.55, "Edge, node, and global states mutually coupled at each layer:", ha='center', va='center', fontsize=8.0, color=c_text_dark)
    ax.text(4.85, 3.20, "• Edges updated conditioned on incident nodes and global state\n• Nodes updated via aggregated neighbor messages and global state\n• Global state updated via pooled node representation",
            ha='center', va='center', fontsize=7.6, color=c_text_muted)

    draw_arrow(ax, 4.85, 2.75, 4.85, 2.30, color=c_accent_blue)

    # 5. Dual Readout Heads
    draw_rounded_box(ax, 0.7, 1.15, 3.9, 1.15, fill=c_box_fill, border=c_box_border, radius=0.08, lw=1.2)
    ax.text(2.65, 2.05, "Direct Softplus Readout", ha='center', va='center', fontsize=9.2, fontweight='bold', color=c_text_dark)
    ax.text(2.65, 1.75, r"$\hat{n}_{\mathrm{direct}} = 1.0 + \mathrm{Softplus}(\mathbf{W}_d \mathbf{h} + b_d)$", ha='center', va='center', fontsize=8.2, color=c_text_dark)
    ax.text(2.65, 1.42, "Strict physical lower bound:  " + r"$\hat{n} \geq 1.0$", ha='center', va='center', fontsize=7.8, fontweight='bold', color='#15803D')

    draw_rounded_box(ax, 5.1, 1.15, 3.9, 1.15, fill=c_box_fill, border=c_box_border, radius=0.08, lw=1.2)
    ax.text(7.05, 2.05, "Logarithmic Smooth-L1 Readout", ha='center', va='center', fontsize=9.2, fontweight='bold', color=c_text_dark)
    ax.text(7.05, 1.75, r"$\hat{z}_{\mathrm{log}} = \mathbf{W}_\ell \mathbf{h} + b_\ell \approx \ln(n - 0.99)$", ha='center', va='center', fontsize=8.2, color=c_text_dark)
    ax.text(7.05, 1.42, "Compresses extreme high-index gradients", ha='center', va='center', fontsize=7.8, color=c_text_muted)

    draw_arrow(ax, 2.65, 1.15, 4.0, 0.80, color=c_text_dark)
    draw_arrow(ax, 7.05, 1.15, 5.7, 0.80, color=c_text_dark)

    # 6. Final Output
    draw_rounded_box(ax, 2.6, 0.48, 4.5, 0.52, fill='#0F172A', border='#0F172A', radius=0.08, lw=1.2)
    ax.text(4.85, 0.74, r"Blended Prediction:  $\hat{n} = \frac{1}{2}(\hat{n}_{\mathrm{direct}} + \hat{n}_{\log}) \geq 1.0$",
            ha='center', va='center', fontsize=9.0, fontweight='bold', color='#FFFFFF')

    # --------------------------------------------------------------------------
    # Panel B
    # --------------------------------------------------------------------------
    draw_rounded_box(ax, 9.6, 0.4, 5.4, 8.8, fill=c_panel_bg, border=c_panel_border, radius=0.16, lw=1.6)
    ax.text(9.9, 8.85, "(b) Rigorous Evaluation Protocols", fontsize=12.5, fontweight='bold', color=c_text_dark, va='center')

    # Official Benchmark
    draw_rounded_box(ax, 9.9, 7.95, 4.8, 0.65, fill=c_box_fill, border=c_box_border, radius=0.08, lw=1.2)
    ax.text(12.3, 8.35, "MatBench Dielectric Benchmark", ha='center', va='center', fontsize=9.8, fontweight='bold', color=c_text_dark)
    ax.text(12.3, 8.12, r"4,764 materials  •  DFT optical refractive index $n$", ha='center', va='center', fontsize=8.0, color=c_text_muted)

    draw_arrow(ax, 12.3, 7.95, 12.3, 7.45, color=c_text_dark)

    draw_rounded_box(ax, 9.9, 6.70, 4.8, 0.75, fill=c_box_fill, border=c_box_border, radius=0.08, lw=1.2)
    ax.text(12.3, 7.22, "Leakage-Controlled Nested Cross-Validation", ha='center', va='center', fontsize=9.2, fontweight='bold', color=c_text_dark)
    ax.text(12.3, 6.95, "• 15% inner-validation split for checkpoint selection\n• Outer-test fold evaluated strictly ONCE on chosen weights",
            ha='center', va='center', fontsize=7.4, color=c_text_muted)

    draw_arrow(ax, 12.3, 6.70, 12.3, 6.10, color=c_text_dark)

    # Result Box 1
    draw_rounded_box(ax, 9.9, 5.35, 4.8, 0.75, fill=c_accent_fill, border=c_accent_blue, radius=0.08, lw=1.4)
    ax.text(12.3, 5.85, "Development-Stage 5-Fold Benchmark", ha='center', va='center', fontsize=9.0, fontweight='bold', color=c_accent_blue)
    ax.text(12.3, 5.55, r"$\mathbf{0.2909 \pm 0.0860}$ MAE  •  $\mathrm{MedAE} = 0.0631$  •  $\rho = 0.9395$",
            ha='center', va='center', fontsize=8.0, fontweight='bold', color=c_text_dark)

    ax.plot([9.9, 14.7], [4.95, 4.95], color='#94A3B8', lw=0.8, ls=':')

    # Chemical-System-Disjoint
    ax.text(12.3, 4.60, "Chemical-System-Disjoint Extrapolation", ha='center', va='center', fontsize=10.0, fontweight='bold', color=c_text_dark)

    draw_rounded_box(ax, 9.9, 3.60, 4.8, 0.75, fill=c_box_fill, border=c_box_border, radius=0.08, lw=1.2)
    ax.text(12.3, 4.12, "3,169 Non-Overlapping Chemical Systems", ha='center', va='center', fontsize=9.0, fontweight='bold', color=c_text_dark)
    ax.text(12.3, 3.85, "• GroupKFold split strictly grouped by elemental composition\n• Zero chemical-system overlap between train and test",
            ha='center', va='center', fontsize=7.4, color=c_text_muted)

    draw_arrow(ax, 12.3, 3.60, 12.3, 3.00, color=c_text_dark)

    draw_rounded_box(ax, 9.9, 2.25, 4.8, 0.75, fill=c_box_fill, border=c_box_border, radius=0.08, lw=1.2)
    ax.text(12.3, 2.77, "Retrain Frozen Architecture from Scratch", ha='center', va='center', fontsize=9.0, fontweight='bold', color=c_text_dark)
    ax.text(12.3, 2.50, "• Identical hyperparameters and inner-validation checkpoint rule\n• Validates generalization on withheld chemical domains",
            ha='center', va='center', fontsize=7.4, color=c_text_muted)

    draw_arrow(ax, 12.3, 2.25, 12.3, 1.65, color=c_text_dark)

    # Result Box 2
    draw_rounded_box(ax, 9.9, 0.65, 4.8, 1.00, fill=c_accent_fill, border=c_accent_blue, radius=0.10, lw=1.5)
    ax.text(12.3, 1.38, "Chemical-System-Disjoint Evaluation", ha='center', va='center', fontsize=9.0, fontweight='bold', color=c_accent_blue)
    ax.text(12.3, 1.10, r"$\mathbf{0.2973 \pm 0.0237}$ MAE  •  $\mathrm{MedAE} = 0.0692$  •  $\rho = 0.9331$",
            ha='center', va='center', fontsize=8.0, fontweight='bold', color=c_text_dark)
    ax.text(12.3, 0.85, "(Only 2.2% higher MAE than official-fold estimate)", ha='center', va='center', fontsize=7.4, color=c_text_muted)

    plt.savefig(output_png, dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.savefig(output_pdf, dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close()
    print(f"Trial 2 generated: {output_png}")

# ==============================================================================
# TRIAL 3: Modern Informatics Two-Tone (Cohesive Indigo & Muted Teal)
# ==============================================================================
def generate_trial_3(output_png, output_pdf):
    fig = plt.figure(figsize=(15.5, 9.6), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 15.5)
    ax.set_ylim(0, 9.6)
    ax.axis('off')

    # Palette: Indigo for Panel A, Teal for Panel B
    c_indigo_dark = '#312E81'
    c_indigo_border = '#4338CA'
    c_indigo_fill = '#EEF2FF'
    c_indigo_light = '#F5F3FF'

    c_teal_dark = '#134E4A'
    c_teal_border = '#0F766E'
    c_teal_fill = '#F0FDFA'
    c_teal_light = '#F0FDF4'

    c_text_dark = '#0F172A'
    c_text_muted = '#475569'

    # Panel A: Architecture (Indigo theme)
    draw_rounded_box(ax, 0.5, 0.4, 8.7, 8.8, fill='#FAFAFA', border='#E2E8F0', radius=0.20, lw=1.4)
    ax.text(0.8, 8.85, "(a) DualHead–LogDirect GNN Architecture", fontsize=13, fontweight='bold', color=c_indigo_dark, va='center')

    # 1. Input Box
    draw_rounded_box(ax, 2.5, 7.95, 4.7, 0.65, fill='#FFFFFF', border='#CBD5E1', radius=0.10, lw=1.2)
    ax.text(4.85, 8.35, "Periodic Crystal Structure", ha='center', va='center', fontsize=10.0, fontweight='bold', color=c_text_dark)
    ax.text(4.85, 8.12, "Lattice constants  •  Atomic positions  •  Atomic numbers", ha='center', va='center', fontsize=8.0, color=c_text_muted)

    draw_arrow(ax, 4.85, 7.95, 4.85, 7.45, color=c_indigo_border)

    # 2. Graph Construction
    draw_rounded_box(ax, 2.5, 6.85, 4.7, 0.60, fill='#FFFFFF', border=c_indigo_border, radius=0.10, lw=1.3)
    ax.text(4.85, 7.22, "Periodic Crystal Graph", ha='center', va='center', fontsize=10.0, fontweight='bold', color=c_indigo_dark)
    ax.text(4.85, 7.00, r"Cutoff $R_{\mathrm{cut}} = 6.0\,\mathrm{\AA}$  •  Truncated to 12 nearest neighbors", ha='center', va='center', fontsize=8.0, color=c_text_muted)

    draw_arrow(ax, 3.4, 6.85, 1.9, 6.20, color=c_indigo_border)
    draw_arrow(ax, 4.85, 6.85, 4.85, 6.20, color=c_indigo_border)
    draw_arrow(ax, 6.3, 6.85, 7.8, 6.20, color=c_indigo_border)

    # 3. Channels (All in unified Indigo)
    draw_rounded_box(ax, 0.7, 4.75, 2.4, 1.45, fill='#FFFFFF', border=c_indigo_border, radius=0.10, lw=1.3)
    ax.text(1.9, 5.95, "Node Features (56-d)", ha='center', va='center', fontsize=9.2, fontweight='bold', color=c_indigo_dark)
    ax.text(1.9, 5.58, "• 48-d learned embedding\n• 8 elemental priors\n  (radii, electronegativity,\n  volume, ionization)",
            ha='center', va='center', fontsize=7.4, color=c_text_muted)

    draw_rounded_box(ax, 3.4, 4.75, 2.9, 1.45, fill='#FFFFFF', border=c_indigo_border, radius=0.10, lw=1.3)
    ax.text(4.85, 5.95, "Edge Features (58-d)", ha='center', va='center', fontsize=9.2, fontweight='bold', color=c_indigo_dark)
    ax.text(4.85, 5.58, "• 41 Gaussian distance RBFs\n• Inverse distance channel\n• 16 Chebyshev bond-angle\n  projections (3-body)",
            ha='center', va='center', fontsize=7.4, color=c_text_muted)

    draw_rounded_box(ax, 6.6, 4.75, 2.4, 1.45, fill='#FFFFFF', border=c_indigo_border, radius=0.10, lw=1.3)
    ax.text(7.8, 5.95, "Global State (12-d)", ha='center', va='center', fontsize=9.2, fontweight='bold', color=c_indigo_dark)
    ax.text(7.8, 5.58, "• Macroscopic density\n• Packing fraction\n• Mean atomic volume\n• Lattice aspect ratio",
            ha='center', va='center', fontsize=7.4, color=c_text_muted)

    draw_arrow(ax, 1.9, 4.75, 3.6, 4.10, color=c_indigo_border)
    draw_arrow(ax, 4.85, 4.75, 4.85, 4.10, color=c_indigo_border)
    draw_arrow(ax, 7.8, 4.75, 6.1, 4.10, color=c_indigo_border)

    # 4. Hierarchical Interaction
    draw_rounded_box(ax, 1.2, 2.75, 7.3, 1.35, fill=c_indigo_fill, border=c_indigo_border, radius=0.12, lw=1.5)
    ax.text(4.85, 3.82, "3 Hierarchical Interaction Layers  [LayerNorm + SiLU]", ha='center', va='center', fontsize=10.0, fontweight='bold', color=c_indigo_dark)
    ax.text(4.85, 3.55, "Mutual information exchange between edges, nodes, and global state:", ha='center', va='center', fontsize=8.0, color=c_text_dark)
    ax.text(4.85, 3.20, "1. Edge update conditioned on incident nodes and global vector\n2. Node update via neighbor message aggregation and global vector\n3. Global vector update via pooled crystal representation",
            ha='center', va='center', fontsize=7.6, color=c_text_muted)

    draw_arrow(ax, 4.85, 2.75, 4.85, 2.30, color=c_indigo_border)

    # 5. Dual Readout Heads
    draw_rounded_box(ax, 0.7, 1.15, 3.9, 1.15, fill='#FFFFFF', border=c_indigo_border, radius=0.10, lw=1.3)
    ax.text(2.65, 2.05, "Direct Softplus Head", ha='center', va='center', fontsize=9.2, fontweight='bold', color=c_indigo_dark)
    ax.text(2.65, 1.75, r"$\hat{n}_{\mathrm{direct}} = 1.0 + \mathrm{Softplus}(\mathbf{W}_d \mathbf{h} + b_d)$", ha='center', va='center', fontsize=8.2, color=c_text_dark)
    ax.text(2.65, 1.42, "Guarantees physical bound:  " + r"$\hat{n} \geq 1.0$", ha='center', va='center', fontsize=7.8, fontweight='bold', color='#15803D')

    draw_rounded_box(ax, 5.1, 1.15, 3.9, 1.15, fill='#FFFFFF', border=c_indigo_border, radius=0.10, lw=1.3)
    ax.text(7.05, 2.05, "Logarithmic Smooth-L1 Head", ha='center', va='center', fontsize=9.2, fontweight='bold', color=c_indigo_dark)
    ax.text(7.05, 1.75, r"$\hat{z}_{\mathrm{log}} = \mathbf{W}_\ell \mathbf{h} + b_\ell \approx \ln(n - 0.99)$", ha='center', va='center', fontsize=8.2, color=c_text_dark)
    ax.text(7.05, 1.42, "Mitigates heavy-tailed outlier gradient shock", ha='center', va='center', fontsize=7.8, color=c_text_muted)

    draw_arrow(ax, 2.65, 1.15, 4.0, 0.80, color=c_indigo_border)
    draw_arrow(ax, 7.05, 1.15, 5.7, 0.80, color=c_indigo_border)

    # 6. Blended Output
    draw_rounded_box(ax, 2.7, 0.48, 4.3, 0.52, fill=c_indigo_fill, border=c_indigo_border, radius=0.08, lw=1.4)
    ax.text(4.85, 0.74, r"Blended Prediction:  $\hat{n} = \frac{1}{2}(\hat{n}_{\mathrm{direct}} + \hat{n}_{\log}) \geq 1.0$",
            ha='center', va='center', fontsize=9.0, fontweight='bold', color=c_indigo_dark)

    # --------------------------------------------------------------------------
    # Panel B: Evaluation (Teal theme)
    # --------------------------------------------------------------------------
    draw_rounded_box(ax, 9.6, 0.4, 5.4, 8.8, fill='#FAFAFA', border='#E2E8F0', radius=0.20, lw=1.4)
    ax.text(9.9, 8.85, "(b) Evaluation Protocols", fontsize=13, fontweight='bold', color=c_teal_dark, va='center')

    # Official Benchmark
    draw_rounded_box(ax, 9.9, 7.95, 4.8, 0.65, fill='#FFFFFF', border='#CBD5E1', radius=0.10, lw=1.2)
    ax.text(12.3, 8.35, "MatBench Dielectric Task", ha='center', va='center', fontsize=9.8, fontweight='bold', color=c_text_dark)
    ax.text(12.3, 8.12, "4,764 inorganic crystalline structures", ha='center', va='center', fontsize=8.0, color=c_text_muted)

    draw_arrow(ax, 12.3, 7.95, 12.3, 7.45, color=c_teal_border)

    draw_rounded_box(ax, 9.9, 6.70, 4.8, 0.75, fill='#FFFFFF', border=c_teal_border, radius=0.10, lw=1.3)
    ax.text(12.3, 7.22, "Strict Nested Cross-Validation (5 Folds)", ha='center', va='center', fontsize=9.2, fontweight='bold', color=c_teal_dark)
    ax.text(12.3, 6.95, "• 15% inner-validation split for model checkpointing\n• Outer-test fold evaluated ONCE on frozen weights",
            ha='center', va='center', fontsize=7.4, color=c_text_muted)

    draw_arrow(ax, 12.3, 6.70, 12.3, 6.10, color=c_teal_border)

    # Result Box 1
    draw_rounded_box(ax, 9.9, 5.35, 4.8, 0.75, fill=c_teal_fill, border=c_teal_border, radius=0.10, lw=1.4)
    ax.text(12.3, 5.85, "Development-Stage 5-Fold Result", ha='center', va='center', fontsize=9.0, fontweight='bold', color=c_teal_dark)
    ax.text(12.3, 5.55, r"$\mathbf{0.2909 \pm 0.0860}$ MAE  •  $\mathrm{MedAE} = 0.0631$  •  $\rho = 0.9395$",
            ha='center', va='center', fontsize=8.0, fontweight='bold', color=c_teal_dark)

    ax.plot([9.9, 14.7], [4.95, 4.95], color='#94A3B8', lw=0.8, ls=':')

    # Chemical-System-Disjoint
    ax.text(12.3, 4.60, "Chemical-System-Disjoint Generalization", ha='center', va='center', fontsize=10.0, fontweight='bold', color=c_teal_dark)

    draw_rounded_box(ax, 9.9, 3.60, 4.8, 0.75, fill='#FFFFFF', border=c_teal_border, radius=0.10, lw=1.3)
    ax.text(12.3, 4.12, "3,169 Non-Overlapping Chemical Systems", ha='center', va='center', fontsize=9.0, fontweight='bold', color=c_teal_dark)
    ax.text(12.3, 3.85, "• GroupKFold split strictly by constituent chemical system\n• Zero train–test chemical-system overlap",
            ha='center', va='center', fontsize=7.4, color=c_text_muted)

    draw_arrow(ax, 12.3, 3.60, 12.3, 3.00, color=c_teal_border)

    draw_rounded_box(ax, 9.9, 2.25, 4.8, 0.75, fill='#FFFFFF', border=c_teal_border, radius=0.10, lw=1.3)
    ax.text(12.3, 2.77, "Retrain Frozen Architecture from Scratch", ha='center', va='center', fontsize=9.0, fontweight='bold', color=c_teal_dark)
    ax.text(12.3, 2.50, "• Identical hyperparameters and inner-validation rule\n• Validates generalization on unseen chemical spaces",
            ha='center', va='center', fontsize=7.4, color=c_text_muted)

    draw_arrow(ax, 12.3, 2.25, 12.3, 1.65, color=c_teal_border)

    # Result Box 2
    draw_rounded_box(ax, 9.9, 0.65, 4.8, 1.00, fill=c_teal_fill, border=c_teal_border, radius=0.12, lw=1.5)
    ax.text(12.3, 1.38, "Chemical-System-Disjoint Evaluation", ha='center', va='center', fontsize=9.0, fontweight='bold', color=c_teal_dark)
    ax.text(12.3, 1.10, r"$\mathbf{0.2973 \pm 0.0237}$ MAE  •  $\mathrm{MedAE} = 0.0692$  •  $\rho = 0.9331$",
            ha='center', va='center', fontsize=8.0, fontweight='bold', color=c_teal_dark)
    ax.text(12.3, 0.85, "(Only 2.2% higher MAE than official-fold estimate)", ha='center', va='center', fontsize=7.4, color=c_teal_dark)

    plt.savefig(output_png, dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.savefig(output_pdf, dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close()
    print(f"Trial 3 generated: {output_png}")

def main():
    trials_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'manuscript', 'figures', 'trials')
    os.makedirs(trials_dir, exist_ok=True)

    t1_png = os.path.join(trials_dir, 'trial_1_minimalist_navy.png')
    t1_pdf = os.path.join(trials_dir, 'trial_1_minimalist_navy.pdf')
    generate_trial_1(t1_png, t1_pdf)

    t2_png = os.path.join(trials_dir, 'trial_2_monochrome_accent.png')
    t2_pdf = os.path.join(trials_dir, 'trial_2_monochrome_accent.pdf')
    generate_trial_2(t2_png, t2_pdf)

    t3_png = os.path.join(trials_dir, 'trial_3_modern_two_tone.png')
    t3_pdf = os.path.join(trials_dir, 'trial_3_modern_two_tone.pdf')
    generate_trial_3(t3_png, t3_pdf)

    # Copy to artifacts directory
    artifact_dir = r"C:\Users\HP\.gemini\antigravity-ide\brain\7e1462b3-34c9-4d76-a8c1-ddab45f77d0b"
    shutil.copy(t1_png, os.path.join(artifact_dir, 'trial_1_minimalist_navy.png'))
    shutil.copy(t2_png, os.path.join(artifact_dir, 'trial_2_monochrome_accent.png'))
    shutil.copy(t3_png, os.path.join(artifact_dir, 'trial_3_modern_two_tone.png'))
    print("All trials copied to artifact directory.")

if __name__ == '__main__':
    main()
