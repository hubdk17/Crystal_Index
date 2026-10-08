import os
os.environ["PYTHONIOENCODING"] = "utf-8"
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

"""
04_classical_baselines.py
=========================
Trains and rigorously benchmarks classical regression models across official
MatBench 5-fold cross validation for matbench_dielectric.

Evaluates on:
- Full Physical Features (~148 dims)
- PCA-12 Features (12 dims)
- PCA-10 Features (10 dims)
- PCA-8 Features  (8 dims)

Models:
- Ridge Regression
- Linear SVR
- SVR with RBF Kernel
- SVR with Polynomial Kernel
- Random Forest Regressor

Produces official MatBench MAE, RMSE, R² metrics and comparative visual figures.
"""

import json
import time
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(r"d:\Desktop\Material_science_qml")
DATA_PROC    = PROJECT_ROOT / "data" / "processed"
FOLDS_DIR    = DATA_PROC / "folds"
RESULTS_DIR  = PROJECT_ROOT / "results"
FIGURES_DIR  = PROJECT_ROOT / "figures"

for d in [RESULTS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")
    sys.stdout.flush()

def get_models():
    return {
        "Ridge": Ridge(alpha=1.0),
        "Linear SVR": SVR(kernel="linear", C=1.0),
        "RBF SVR": SVR(kernel="rbf", C=10.0, gamma="scale", epsilon=0.1),
        "Poly SVR": SVR(kernel="poly", degree=3, C=1.0, epsilon=0.1),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
    }

def main():
    log("=" * 70)
    log("CLASSICAL BENCHMARKING ACROSS OFFICIAL MATBENCH 5 FOLDS")
    log("=" * 70)

    feature_sets = ["pca8", "pca10", "pca12", "full"]
    results = {f_set: {} for f_set in feature_sets}

    for f_set in feature_sets:
        models = get_models()
        for m_name in models.keys():
            results[f_set][m_name] = {
                "fold_mae": [],
                "fold_rmse": [],
                "fold_r2": [],
            }

    n_folds = 5
    start_time = time.time()

    for fold in range(n_folds):
        log(f"\n{'='*30} FOLD {fold} {'='*30}")
        y_train = np.load(FOLDS_DIR / f"fold_{fold}_y_train.npy")
        y_test  = np.load(FOLDS_DIR / f"fold_{fold}_y_test.npy")

        for f_set in feature_sets:
            X_train = np.load(FOLDS_DIR / f"fold_{fold}_{f_set}_train.npy")
            X_test  = np.load(FOLDS_DIR / f"fold_{fold}_{f_set}_test.npy")

            models = get_models()
            for m_name, model in models.items():
                t0 = time.time()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                fit_time = time.time() - t0

                mae  = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2   = r2_score(y_test, y_pred)

                results[f_set][m_name]["fold_mae"].append(float(mae))
                results[f_set][m_name]["fold_rmse"].append(float(rmse))
                results[f_set][m_name]["fold_r2"].append(float(r2))

                log(f"  [{f_set:<5s}] {m_name:<14s} | MAE: {mae:.4f} | RMSE: {rmse:.4f} | R2: {r2:.4f} ({fit_time:.1f}s)")

    # ── Aggregate 5-Fold Metrics ──────────────────────────────────────
    log("\n" + "=" * 70)
    log("5-FOLD AGGREGATE SUMMARY (MEAN ± STD)")
    log("=" * 70)

    summary_table = []
    final_json = {}

    for f_set in feature_sets:
        final_json[f_set] = {}
        for m_name in results[f_set].keys():
            maes  = results[f_set][m_name]["fold_mae"]
            rmses = results[f_set][m_name]["fold_rmse"]
            r2s   = results[f_set][m_name]["fold_r2"]

            mean_mae,  std_mae  = float(np.mean(maes)),  float(np.std(maes))
            mean_rmse, std_rmse = float(np.mean(rmses)), float(np.std(rmses))
            mean_r2,   std_r2   = float(np.mean(r2s)),   float(np.std(r2s))

            final_json[f_set][m_name] = {
                "mean_mae": round(mean_mae, 4),
                "std_mae":  round(std_mae, 4),
                "mean_rmse": round(mean_rmse, 4),
                "std_rmse":  round(std_rmse, 4),
                "mean_r2":   round(mean_r2, 4),
                "std_r2":    round(std_r2, 4),
                "fold_maes": [round(x, 4) for x in maes],
            }

            summary_table.append({
                "Feature Set": f_set.upper(),
                "Model": m_name,
                "MAE": f"{mean_mae:.4f} ± {std_mae:.4f}",
                "RMSE": f"{mean_rmse:.4f} ± {std_rmse:.4f}",
                "R2": f"{mean_r2:.4f} ± {std_r2:.4f}",
            })

    summary_df = pd.DataFrame(summary_table)
    print("\n" + summary_df.to_string(index=False))

    # Save scores to JSON
    json_path = RESULTS_DIR / "classical_baseline_scores.json"
    with open(json_path, "w") as f:
        json.dump(final_json, f, indent=2)
    log(f"\nSaved benchmark metrics to: {json_path}")

    # ── Plot Comparison Chart ─────────────────────────────────────────
    plot_df = []
    for f_set in feature_sets:
        for m_name in final_json[f_set].keys():
            plot_df.append({
                "Feature Set": f_set.upper(),
                "Model": m_name,
                "Mean MAE": final_json[f_set][m_name]["mean_mae"],
                "Std MAE": final_json[f_set][m_name]["std_mae"]
            })
    plot_df = pd.DataFrame(plot_df)

    plt.figure(figsize=(10, 6), dpi=300)
    sns.set_theme(style="whitegrid")
    palette = sns.color_palette("muted", len(plot_df["Model"].unique()))

    ax = sns.barplot(
        data=plot_df,
        x="Feature Set",
        y="Mean MAE",
        hue="Model",
        palette="viridis",
        edgecolor="black",
        linewidth=0.8
    )

    # Reference line for published MODNet SOTA and CGCNN
    plt.axhline(y=0.2711, color="red", linestyle="--", linewidth=1.5, label="MODNet SOTA (0.2711)")
    plt.axhline(y=0.5988, color="gray", linestyle=":", linewidth=1.5, label="CGCNN baseline (0.5988)")

    plt.title("Classical Regression Baselines across Input Feature Dimensionalities\n(matbench_dielectric 5-Fold Official CV)", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Input Feature Representation", fontsize=11, fontweight="bold")
    plt.ylabel("Mean Absolute Error (MAE, lower is better)", fontsize=11, fontweight="bold")
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
    plt.tight_layout()

    fig_path = FIGURES_DIR / "classical_baselines_mae_comparison.png"
    plt.savefig(fig_path)
    plt.close()
    log(f"Saved comparison figure to: {fig_path}")

    log(f"\n[SUCCESS] Classical benchmarking complete in {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    main()
