import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import shutil

def main():
    csv_path = Path("d:/Desktop/Material_science_qml/results/chemistry_grouped_ood_confirmation/predictions/ood_predictions.csv")
    df = pd.read_csv(csv_path)

    ood_mae_mean = 0.2973
    ood_mae_std = 0.0237
    standard_matbench_mae = 0.2909
    mae_diff = ood_mae_mean - standard_matbench_mae
    pct_diff = (mae_diff / standard_matbench_mae) * 100.0

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel (a): Parity Plot
    ax = axes[0]
    y_t = df["y_true"].values
    y_p = df["y_pred"].values
    ax.scatter(y_t, y_p, alpha=0.35, s=16, color="#4C72B0", edgecolors="none")
    max_v = min(max(np.max(y_t), np.max(y_p)), 15.0)
    ax.plot([1.0, max_v], [1.0, max_v], "r--", lw=1.8, label="Ideal Line (y = x)")
    ax.set_xlim(0.8, max_v)
    ax.set_ylim(0.8, max_v)
    ax.set_xlabel("DFT True Refractive Index $n$", fontsize=12)
    ax.set_ylabel("Predicted Refractive Index $\\hat{n}$", fontsize=12)
    ax.set_title(f"Chemical-System-Disjoint Grouped Parity\n5-Fold MAE: {ood_mae_mean:.4f} $\\pm$ {ood_mae_std:.4f}", fontsize=13, fontweight="bold")
    ax.legend(frameon=True)

    # Panel (b): Comparison Bar Chart
    ax2 = axes[1]
    bars = ax2.bar(["Standard MatBench\n(Official Folds)", "Chemical-System-Disjoint\n(Unseen Chemical Systems)"],
                   [standard_matbench_mae, ood_mae_mean],
                   color=["#2b5c8f", "#d95f02"], width=0.45, edgecolor="black")
    ax2.set_ylabel("5-Fold Mean Absolute Error (MAE)", fontsize=12)
    ax2.set_title(f"Evaluation Under Unseen Chemical Systems\nDifference: {mae_diff:+.4f} ({pct_diff:+.1f}% higher MAE)", fontsize=13, fontweight="bold")
    ax2.set_ylim(0, max(ood_mae_mean, standard_matbench_mae) * 1.3)
    for bar in bars:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, h + 0.01, f"{h:.4f}", ha="center", va="bottom", fontsize=12, fontweight="bold")

    plt.tight_layout()
    fig_dir = Path("d:/Desktop/Material_science_qml/manuscript/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_dir / "ood_generalization_confirmation.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Figure 3 saved to: {fig_path}")

    artifact_fig = Path(r"C:\Users\HP\.gemini\antigravity-ide\brain\7e1462b3-34c9-4d76-a8c1-ddab45f77d0b\ood_generalization_confirmation.png")
    shutil.copy(fig_path, artifact_fig)
    print(f"Figure 3 copied to artifact: {artifact_fig}")

if __name__ == "__main__":
    main()
