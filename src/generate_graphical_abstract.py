"""
Script: generate_graphical_abstract.py
Author: Daksh Kaila
Description:
    Generates a clean, modern, restrained professional Graphical Abstract for Elsevier submission
    (Computational Materials Science).
    Ratio: 2:1 landscape (12 x 6 inches, 300 dpi).
    Flow: Crystal Structures -> Multi-Channel Graph -> DualHead-LogDirect GNN -> Rigorous Evaluation & Outcomes.
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Polygon, Wedge
import numpy as np

matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['mathtext.fontset'] = 'dejavusans'

def create_graphical_abstract():
    # 12 x 6 inches at 300 dpi = 3600 x 1800 px (2:1 ratio, ideal for Elsevier graphical abstract)
    fig = plt.figure(figsize=(12, 6), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Color Palette: Professional restrained scientific navy/slate/teal
    c_bg = '#FFFFFF'
    c_panel_bg = '#F8FAFC'
    c_border = '#CBD5E1'
    c_border_focus = '#0284C7'
    c_navy = '#0F172A'
    c_slate = '#334155'
    c_muted = '#64748B'
    c_accent = '#0284C7' # Cerulean / cyan accent
    c_card_bg = '#FFFFFF'
    c_pill = '#E0F2FE'
    c_pill_text = '#0369A1'
    c_highlight = '#F1F5F9'

    ax.set_facecolor(c_bg)

    def draw_card(x, y, w, h, fill=c_card_bg, border=c_border, radius=0.15, lw=1.2, shadow=True):
        if shadow:
            s_box = FancyBboxPatch((x+0.04, y-0.04), w, h,
                                  boxstyle=f"round,pad=0,rounding_size={radius}",
                                  facecolor='#E2E8F0', edgecolor='none', zorder=1)
            ax.add_patch(s_box)
        box = FancyBboxPatch((x, y), w, h,
                             boxstyle=f"round,pad=0,rounding_size={radius}",
                             facecolor=fill, edgecolor=border, linewidth=lw, zorder=2)
        ax.add_patch(box)
        return box

    def draw_arrow(x1, y1, x2, y2, lw=1.6, color=c_accent):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='-|>', color=color, lw=lw,
                                    shrinkA=3, shrinkB=3, mutation_scale=14),
                    zorder=4)

    # Header / Title Banner
    ax.text(6.0, 5.55, "Reproducible Crystal-Graph Learning for Refractive-Index Prediction",
            ha='center', va='center', fontsize=13.5, fontweight='bold', color=c_navy)
    ax.text(6.0, 5.25, "DualHead–LogDirect Architecture & Chemical-System-Disjoint Generalization",
            ha='center', va='center', fontsize=9.5, fontweight='normal', color=c_slate)

    # ---------------------------------------------------------
    # STAGE 1: Crystal Structure Dataset (x: 0.5 to 2.8)
    # ---------------------------------------------------------
    draw_card(0.5, 0.45, 2.45, 4.45, fill='#F8FAFC', border='#94A3B8', lw=1.3)
    ax.text(1.725, 4.60, "1. Crystal Structures", ha='center', va='center',
            fontsize=10.5, fontweight='bold', color=c_navy)

    # Crystal Lattice Icon/Illustration
    # Unit cell wireframe
    pts_front = np.array([[1.15, 2.75], [1.95, 2.75], [1.95, 3.55], [1.15, 3.55], [1.15, 2.75]])
    offset = np.array([0.35, 0.35])
    pts_back = pts_front + offset

    # back edges
    for i in range(4):
        ax.plot([pts_front[i,0], pts_back[i,0]], [pts_front[i,1], pts_back[i,1]],
                color='#94A3B8', lw=1.1, ls='--', zorder=3)
    ax.plot(pts_back[:,0], pts_back[:,1], color='#94A3B8', lw=1.1, ls='--', zorder=3)
    ax.plot(pts_front[:,0], pts_front[:,1], color=c_slate, lw=1.3, zorder=3)

    # Atoms at vertices/center
    atom_coords = [
        (1.15, 2.75, '#0284C7', 0.08), (1.95, 2.75, '#0284C7', 0.08),
        (1.95, 3.55, '#0284C7', 0.08), (1.15, 3.55, '#0284C7', 0.08),
        (1.50, 3.10, '#DC2626', 0.09), # center
        (1.50, 3.90, '#059669', 0.07), (2.30, 3.10, '#D97706', 0.07)
    ]
    for x_a, y_a, c_a, r_a in atom_coords:
        c = Circle((x_a, y_a), r_a, facecolor=c_a, edgecolor=c_navy, lw=0.9, zorder=4)
        ax.add_patch(c)

    # Info card within Stage 1
    draw_card(0.65, 0.65, 2.15, 1.85, fill=c_card_bg, border='#CBD5E1', radius=0.10, lw=1.0)
    ax.text(1.725, 2.22, "MatBench Dielectric", ha='center', va='center',
            fontsize=8.8, fontweight='bold', color=c_slate)
    ax.text(1.725, 1.95, "• 4,764 Inorganic Crystals", ha='center', va='center',
            fontsize=7.8, color=c_navy)
    ax.text(1.725, 1.68, "• 3,169 Chemical Systems", ha='center', va='center',
            fontsize=7.8, color=c_navy)
    ax.text(1.725, 1.41, "• PBE Sol + DFPT Labels", ha='center', va='center',
            fontsize=7.8, color=c_navy)
    ax.text(1.725, 1.05, "Refractive index: n ≥ 1.0", ha='center', va='center',
            fontsize=7.8, fontweight='bold', color=c_accent)

    draw_arrow(2.95, 2.65, 3.35, 2.65)

    # ---------------------------------------------------------
    # STAGE 2: Multi-Scale Graph Featurization (x: 3.35 to 5.75)
    # ---------------------------------------------------------
    draw_card(3.35, 0.45, 2.45, 4.45, fill='#F8FAFC', border='#94A3B8', lw=1.3)
    ax.text(4.575, 4.60, "2. Multi-Channel Graph", ha='center', va='center',
            fontsize=10.5, fontweight='bold', color=c_navy)

    # Atom Box
    draw_card(3.50, 3.25, 2.15, 1.05, fill=c_card_bg, border='#CBD5E1', radius=0.08, lw=1.0)
    ax.text(4.575, 4.05, "Atom Nodes: hi (56-d)", ha='center', va='center',
            fontsize=8.3, fontweight='bold', color=c_navy)
    ax.text(4.575, 3.75, "48-d Z Embeddings +", ha='center', va='center',
            fontsize=7.4, color=c_slate)
    ax.text(4.575, 3.48, "8 Elemental Physical Priors", ha='center', va='center',
            fontsize=7.4, fontweight='bold', color='#0284C7')

    # Edge Box
    draw_card(3.50, 1.95, 2.15, 1.15, fill=c_card_bg, border='#CBD5E1', radius=0.08, lw=1.0)
    ax.text(4.575, 2.85, "Edges: eij (58-d)", ha='center', va='center',
            fontsize=8.3, fontweight='bold', color=c_navy)
    ax.text(4.575, 2.58, "41 Radial RBFs + 1/(rij+0.01)", ha='center', va='center',
            fontsize=7.2, color=c_slate)
    ax.text(4.575, 2.32, "16-d Chebyshev 3-Body Angles", ha='center', va='center',
            fontsize=7.2, fontweight='bold', color='#059669')
    ax.text(4.575, 2.08, "Rcut = 6.0 Å, k = 12 neighbors", ha='center', va='center',
            fontsize=6.8, color=c_muted)

    # Global Box
    draw_card(3.50, 0.65, 2.15, 1.15, fill=c_card_bg, border='#CBD5E1', radius=0.08, lw=1.0)
    ax.text(4.575, 1.55, "Global State: u (12-d)", ha='center', va='center',
            fontsize=8.3, fontweight='bold', color=c_navy)
    ax.text(4.575, 1.28, "Density, Packing Fraction,", ha='center', va='center',
            fontsize=7.2, color=c_slate)
    ax.text(4.575, 1.02, "Stoichiometry & Coord Stats", ha='center', va='center',
            fontsize=7.2, color=c_slate)
    ax.text(4.575, 0.78, "Fold-local scaling (no leakage)", ha='center', va='center',
            fontsize=6.8, color=c_muted)

    draw_arrow(5.80, 2.65, 6.20, 2.65)

    # ---------------------------------------------------------
    # STAGE 3: DualHead-LogDirect GNN (x: 6.20 to 8.60)
    # ---------------------------------------------------------
    draw_card(6.20, 0.45, 2.45, 4.45, fill='#F8FAFC', border='#94A3B8', lw=1.3)
    ax.text(7.425, 4.60, "3. DualHead GNN", ha='center', va='center',
            fontsize=10.5, fontweight='bold', color=c_navy)

    # Interaction box
    draw_card(6.35, 3.20, 2.15, 1.10, fill=c_card_bg, border='#CBD5E1', radius=0.08, lw=1.0)
    ax.text(7.425, 4.02, "3× Interaction Layers", ha='center', va='center',
            fontsize=8.3, fontweight='bold', color=c_navy)
    ax.text(7.425, 3.75, "Bidirectional Node-Edge-Global", ha='center', va='center',
            fontsize=7.2, color=c_slate)
    ax.text(7.425, 3.50, "LayerNorm + SiLU + Residuals", ha='center', va='center',
            fontsize=7.2, color=c_muted)
    ax.text(7.425, 3.30, "Mean Pool + Global -> 76-d", ha='center', va='center',
            fontsize=7.0, color=c_navy)

    # Dual Heads Box
    draw_card(6.35, 1.70, 2.15, 1.35, fill='#EFF6FF', border='#3B82F6', radius=0.08, lw=1.1)
    ax.text(7.425, 2.82, "Dual Readout Heads", ha='center', va='center',
            fontsize=8.5, fontweight='bold', color='#1D4ED8')
    ax.text(7.425, 2.52, "Head 1: Softplus Direct", ha='center', va='center',
            fontsize=7.4, fontweight='bold', color=c_navy)
    ax.text(7.425, 2.34, "n_direct = 1.0 + Softplus(...)", ha='center', va='center',
            fontsize=6.8, color=c_slate)
    ax.text(7.425, 2.05, "Head 2: Inverted Log", ha='center', va='center',
            fontsize=7.4, fontweight='bold', color=c_navy)
    ax.text(7.425, 1.87, "n_log = 0.99 + exp(z_log)", ha='center', va='center',
            fontsize=6.8, color=c_slate)

    # Blended Prediction
    draw_card(6.35, 0.65, 2.15, 0.90, fill='#DBEAFE', border='#1D4ED8', radius=0.08, lw=1.2)
    ax.text(7.425, 1.28, "Blended Ensembling", ha='center', va='center',
            fontsize=8.0, fontweight='bold', color='#1E40AF')
    ax.text(7.425, 1.00, "n_hat = 0.5 * (n_direct + n_log)", ha='center', va='center',
            fontsize=7.4, fontweight='bold', color=c_navy)
    ax.text(7.425, 0.78, "Strict Physical Constraint: n ≥ 1.0", ha='center', va='center',
            fontsize=7.0, color='#1E40AF')

    draw_arrow(8.65, 2.65, 9.05, 2.65)

    # ---------------------------------------------------------
    # STAGE 4: Rigorous Evaluation & Outcomes (x: 9.05 to 11.50)
    # ---------------------------------------------------------
    draw_card(9.05, 0.45, 2.45, 4.45, fill='#F0FDF4', border='#16A34A', lw=1.4)
    ax.text(10.275, 4.60, "4. Rigorous Outcomes", ha='center', va='center',
            fontsize=10.5, fontweight='bold', color='#14532D')

    # Card 1: Official MatBench
    draw_card(9.20, 3.25, 2.15, 1.05, fill=c_card_bg, border='#86EFAC', radius=0.08, lw=1.0)
    ax.text(10.275, 4.05, "Official MatBench 5-Fold", ha='center', va='center',
            fontsize=8.0, fontweight='bold', color='#15803D')
    ax.text(10.275, 3.75, "MAE = 0.2909 ± 0.0860", ha='center', va='center',
            fontsize=8.5, fontweight='bold', color=c_navy)
    ax.text(10.275, 3.45, "MedAE = 0.0631  •  R2 = 0.865", ha='center', va='center',
            fontsize=7.2, color=c_slate)

    # Card 2: Chemical-System-Disjoint
    draw_card(9.20, 2.05, 2.15, 1.05, fill=c_card_bg, border='#86EFAC', radius=0.08, lw=1.0)
    ax.text(10.275, 2.85, "Disjoint Chemical Systems", ha='center', va='center',
            fontsize=8.0, fontweight='bold', color='#15803D')
    ax.text(10.275, 2.55, "MAE = 0.2973 ± 0.0237", ha='center', va='center',
            fontsize=8.5, fontweight='bold', color=c_navy)
    ax.text(10.275, 2.25, "Spearman ρ = 0.9331", ha='center', va='center',
            fontsize=8.0, fontweight='bold', color='#0284C7')

    # Card 3: Reproducibility & Audit
    draw_card(9.20, 0.65, 2.15, 1.25, fill=c_card_bg, border='#86EFAC', radius=0.08, lw=1.0)
    ax.text(10.275, 1.65, "Leakage Audit & Insights", ha='center', va='center',
            fontsize=8.0, fontweight='bold', color='#15803D')
    ax.text(10.275, 1.38, "• Corrected 0.2068 Test Leak", ha='center', va='center',
            fontsize=7.2, color=c_navy)
    ax.text(10.275, 1.15, "• RBF-SVR Beats Base CGCNN", ha='center', va='center',
            fontsize=7.2, color=c_navy)
    ax.text(10.275, 0.90, "• High-Index Tail Outliers: n > 8", ha='center', va='center',
            fontsize=7.2, fontweight='bold', color='#B91C1C')

    # Save to manuscript_2 and artifacts
    out_dir_2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'manuscript_2')
    art_dir = r"C:\Users\HP\.gemini\antigravity-ide\brain\7e1462b3-34c9-4d76-a8c1-ddab45f77d0b"

    p_png = os.path.join(out_dir_2, 'graphical_abstract.png')
    p_pdf = os.path.join(out_dir_2, 'graphical_abstract.pdf')
    p_art = os.path.join(art_dir, 'graphical_abstract.png')

    plt.savefig(p_png, dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.savefig(p_pdf, dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.savefig(p_art, dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close()
    print("Graphical abstract successfully generated at:", p_png)

if __name__ == '__main__':
    create_graphical_abstract()
