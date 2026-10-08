"""
Script: generate_figure4_diagnostics.py
Author: Daksh Kaila
Description:
    Generates Figure 4: Error diagnostics and heteroskedasticity for DualHead-LogDirect GNN.
    Panel (a): Absolute error |y - y_hat| vs. target refractive index n across regimes (n <= 3, 3-5, 5-8, >8).
    Panel (b): Out-of-fold residual distribution (y_hat - y) comparing DualHead-LogDirect GNN with baselines.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Publication typography
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['mathtext.fontset'] = 'dejavusans'

PROJECT_ROOT = Path(r"d:\Desktop\Material_science_qml")
FRONTIER_CSV = PROJECT_ROOT / "results" / "frontier_gnn_benchmark_20260923_133231" / "predictions" / "frontier_gnns_all_predictions.csv"
AUDIT_CSV = PROJECT_ROOT / "results" / "classical_gnn_audit_20260923_110016" / "predictions" / "official_matbench_all_predictions.csv"
OUT_PNG = PROJECT_ROOT / "manuscript" / "figures" / "publication_residuals_and_heteroskedasticity.png"

def main():
    # Load DualHead predictions
    df_frontier = pd.read_csv(FRONTIER_CSV)
    dh = df_frontier[df_frontier['model'] == 'DualHead_LogDirect_GNN'].copy()
    dh['residual'] = dh['y_pred'] - dh['y_true']
    dh['abs_error'] = np.abs(dh['residual'])

    # Load baseline predictions
    df_audit = pd.read_csv(AUDIT_CSV)
    rbf = df_audit[df_audit['model'] == 'RBF SVR'].copy()
    rbf['residual'] = rbf['y_pred'] - rbf['y_true']
    rbf['abs_error'] = np.abs(rbf['residual'])

    cgcnn = df_audit[df_audit['model'] == 'CGCNN-style Periodic Crystal GNN'].copy()
    cgcnn['residual'] = cgcnn['y_pred'] - cgcnn['y_true']
    cgcnn['abs_error'] = np.abs(cgcnn['residual'])

    # Create 2-panel figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    sns.set_theme(style="whitegrid", palette="muted")

    # ----------------------------------------------------
    # Panel (a): Absolute Error vs Target Refractive Index
    # ----------------------------------------------------
    ax1 = axes[0]
    
    # Highlight target regimes with background bands
    ax1.axvspan(1.0, 3.0, color='#F0FDF4', alpha=0.8, zorder=0, label=r'Central ($n \leq 3.0, 84.4\%$)')
    ax1.axvspan(3.0, 5.0, color='#EFF6FF', alpha=0.8, zorder=0, label=r'Intermediate ($3.0 < n \leq 5.0$)')
    ax1.axvspan(5.0, 8.0, color='#FFFBEB', alpha=0.8, zorder=0, label=r'High ($5.0 < n \leq 8.0$)')
    ax1.axvspan(8.0, 10.0, color='#FEF2F2', alpha=0.8, zorder=0, label=r'Extreme Tail ($n > 8.0$)')

    # Scatter points for DualHead-LogDirect GNN
    ax1.scatter(dh['y_true'], dh['abs_error'], color='#7C3AED', alpha=0.45, s=18, edgecolors='none', zorder=3, label='DualHead–LogDirect GNN')
    
    # Binned median error line
    bins = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
    bin_centers = []
    bin_medians = []
    for i in range(len(bins)-1):
        mask = (dh['y_true'] >= bins[i]) & (dh['y_true'] < bins[i+1])
        if mask.sum() > 0:
            bin_centers.append(0.5 * (bins[i] + bins[i+1]))
            bin_medians.append(np.median(dh.loc[mask, 'abs_error']))
    ax1.plot(bin_centers, bin_medians, color='#1E1B4B', lw=2.4, marker='o', markersize=5, zorder=4, label='Binned MedAE')

    ax1.set_xlim(1.0, 10.0)
    ax1.set_ylim(0.0, 4.5)
    ax1.set_title("(a) Absolute Error vs Target Refractive Index (Heteroskedasticity)", fontsize=11, fontweight='bold', pad=10)
    ax1.set_xlabel("DFT Target Refractive Index $n$", fontsize=10.5, fontweight='bold')
    ax1.set_ylabel(r"Absolute Error $|y - \hat{n}|$", fontsize=10.5, fontweight='bold')
    ax1.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9, fontsize=8.5)

    # ----------------------------------------------------
    # Panel (b): Residual Distribution (Pred - True)
    # ----------------------------------------------------
    ax2 = axes[1]

    # Plot KDE for DualHead, RBF SVR, and CGCNN
    sns.kdeplot(dh['residual'], ax=ax2, color='#7C3AED', lw=2.4, fill=True, alpha=0.25, label=f'DualHead–LogDirect (MAE={dh["abs_error"].mean():.4f}, MedAE={dh["abs_error"].median():.4f})')
    sns.kdeplot(rbf['residual'], ax=ax2, color='#2563EB', lw=1.8, linestyle='--', label=f'RBF-SVR (MAE={rbf["abs_error"].mean():.4f}, MedAE={rbf["abs_error"].median():.4f})')
    sns.kdeplot(cgcnn['residual'], ax=ax2, color='#DC2626', lw=1.6, linestyle=':', label=f'CGCNN Baseline (MAE={cgcnn["abs_error"].mean():.4f}, MedAE={cgcnn["abs_error"].median():.4f})')

    ax2.axvline(0.0, color='black', linestyle='-', lw=1.2, alpha=0.7)
    ax2.set_xlim(-2.5, 2.5)
    ax2.set_title("(b) Out-of-Fold Residual Error Distribution ($\hat{n} - y$)", fontsize=11, fontweight='bold', pad=10)
    ax2.set_xlabel(r"Prediction Residual Error $(\hat{n} - y)$", fontsize=10.5, fontweight='bold')
    ax2.set_ylabel("Probability Density", fontsize=10.5, fontweight='bold')
    ax2.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9, fontsize=8.5)

    plt.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PNG, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated Figure 4 successfully: {OUT_PNG} ({OUT_PNG.stat().st_size:,} bytes)")

if __name__ == '__main__':
    main()
