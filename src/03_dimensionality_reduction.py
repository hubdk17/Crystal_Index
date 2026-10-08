import os
os.environ["PYTHONIOENCODING"] = "utf-8"
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

"""
03_dimensionality_reduction.py
==============================
Applies StandardScaler and Principal Component Analysis (PCA) to compress
the ~150-dimensional physical feature space down to quantum-compatible dimensions:
- PCA-8  (8 qubits)
- PCA-10 (10 qubits)
- PCA-12 (12 qubits)

Normalizes PCA components to [0, π] for direct injection into quantum rotation gates.
Splits data according to official MatBench 5-fold cross-validation folds.
Saves fold matrices and produces variance explained figures.
"""

import json
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from matbench.bench import MatbenchBenchmark

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(r"d:\Desktop\Material_science_qml")
DATA_PROC    = PROJECT_ROOT / "data" / "processed"
FOLDS_DIR    = DATA_PROC / "folds"
RESULTS_DIR  = PROJECT_ROOT / "results"
FIGURES_DIR  = PROJECT_ROOT / "figures"

for d in [FOLDS_DIR, RESULTS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")
    sys.stdout.flush()

def main():
    log("=" * 70)
    log("STAGE 1: Loading Featurized Dataset")
    log("=" * 70)

    feature_path = DATA_PROC / "matbench_dielectric_features.parquet"
    if not feature_path.exists():
        log(f"[ERROR] Featurized file not found at: {feature_path}")
        return

    df = pd.read_parquet(feature_path)
    log(f"Loaded {len(df)} samples with {len(df.columns)} columns")

    target_col = "n"
    feature_cols = [c for c in df.columns if c != target_col]
    X_full = df[feature_cols].values
    y = df[target_col].values

    log(f"Feature matrix shape: {X_full.shape}, Target shape: {y.shape}")

    # ── Global PCA Analysis ───────────────────────────────────────────
    log("\n" + "=" * 70)
    log("STAGE 2: Global PCA & Explained Variance Analysis")
    log("=" * 70)

    scaler_global = StandardScaler()
    X_scaled_global = scaler_global.fit_transform(X_full)

    n_comp_max = min(50, X_full.shape[1])
    pca_global = PCA(n_components=n_comp_max)
    pca_global.fit(X_scaled_global)

    exp_var = pca_global.explained_variance_ratio_
    cum_var = np.cumsum(exp_var)

    for k in [8, 10, 12, 16, 20]:
        if k <= len(cum_var):
            log(f"  PCA-{k:2d}: {cum_var[k-1]*100:.2f}% cumulative variance explained")

    # Plot explained variance
    plt.figure(figsize=(9, 5), dpi=300)
    sns.set_theme(style="whitegrid")
    plt.plot(range(1, len(cum_var) + 1), cum_var * 100, marker='o', markersize=4, color='#2563eb', linewidth=2, label="Cumulative Variance")
    plt.bar(range(1, len(exp_var) + 1), exp_var * 100, alpha=0.4, color='#64748b', label="Individual Component Variance")

    # Highlight quantum dimension boundaries
    for q_dim, col in [(8, '#dc2626'), (10, '#ea580c'), (12, '#16a34a')]:
        plt.axvline(x=q_dim, color=col, linestyle='--', alpha=0.8, label=f"PCA-{q_dim} ({cum_var[q_dim-1]*100:.1f}%)")

    plt.title("Principal Component Analysis on Materials Descriptors (matbench_dielectric)", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Number of Principal Components", fontsize=11)
    plt.ylabel("Variance Explained (%)", fontsize=11)
    plt.xlim(1, 35)
    plt.ylim(0, 105)
    plt.legend(loc='lower right', frameon=True)
    plt.tight_layout()

    fig_path = FIGURES_DIR / "pca_variance_explained.png"
    plt.savefig(fig_path)
    plt.close()
    log(f"Saved PCA variance figure to: {fig_path}")

    # ── Official MatBench 5-Fold Decomposition ────────────────────────
    log("\n" + "=" * 70)
    log("STAGE 3: Decomposing into Official MatBench 5-Fold Cross-Validation")
    log("=" * 70)

    log("Initializing MatbenchBenchmark('matbench_v0.1', subset=['matbench_dielectric']) ...")
    mb = MatbenchBenchmark(subset=["matbench_dielectric"], autoload=True)
    task = list(mb.tasks)[0]

    fold_summaries = {}

    for fold_idx in task.folds:
        log(f"\n── Processing Fold {fold_idx} ──")
        X_train_df, y_train_s = task.get_train_and_val_data(fold_idx)
        X_test_df = task.get_test_data(fold_idx, include_target=False)

        train_indices = [int(x.split('-')[-1]) - 1 for x in X_train_df.index]
        test_indices  = [int(x.split('-')[-1]) - 1 for x in X_test_df.index]

        X_train_raw = X_full[train_indices]
        y_train     = y[train_indices]
        X_test_raw  = X_full[test_indices]
        y_test      = y[test_indices]

        # 1. Scale on train only (prevent data leakage)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_test_scaled  = scaler.transform(X_test_raw)

        # Save full scaled
        np.save(FOLDS_DIR / f"fold_{fold_idx}_full_train.npy", X_train_scaled)
        np.save(FOLDS_DIR / f"fold_{fold_idx}_full_test.npy",  X_test_scaled)
        np.save(FOLDS_DIR / f"fold_{fold_idx}_y_train.npy",    y_train)
        np.save(FOLDS_DIR / f"fold_{fold_idx}_y_test.npy",     y_test)

        fold_meta = {
            "n_train": len(train_indices),
            "n_test": len(test_indices),
            "pca_variance": {}
        }

        # 2. PCA projections for 8, 10, 12 dimensions
        for d in [8, 10, 12]:
            pca = PCA(n_components=d)
            X_tr_pca = pca.fit_transform(X_train_scaled)
            X_te_pca = pca.transform(X_test_scaled)

            # 3. Angle-encoding scaling: normalize each component to [0, π]
            # Use train min-max to fit angle scaler
            angle_scaler = MinMaxScaler(feature_range=(0.0, np.pi))
            X_tr_angle = angle_scaler.fit_transform(X_tr_pca)
            X_te_angle = np.clip(angle_scaler.transform(X_te_pca), 0.0, np.pi)

            np.save(FOLDS_DIR / f"fold_{fold_idx}_pca{d}_train.npy", X_tr_pca)
            np.save(FOLDS_DIR / f"fold_{fold_idx}_pca{d}_test.npy",  X_te_pca)
            np.save(FOLDS_DIR / f"fold_{fold_idx}_angle{d}_train.npy", X_tr_angle)
            np.save(FOLDS_DIR / f"fold_{fold_idx}_angle{d}_test.npy",  X_te_angle)

            fold_meta["pca_variance"][f"pca_{d}"] = float(np.sum(pca.explained_variance_ratio_))

        log(f"  Train: {len(train_indices)}, Test: {len(test_indices)}")
        log(f"  PCA explained variances: {fold_meta['pca_variance']}")
        fold_summaries[f"fold_{fold_idx}"] = fold_meta

    # ── Summary JSON ──────────────────────────────────────────────────
    summary = {
        "dataset": "matbench_dielectric",
        "total_samples": len(df),
        "total_features": X_full.shape[1],
        "target": "n",
        "global_pca_variance": {
            f"pca_{k}": float(cum_var[k-1]) for k in [8, 10, 12, 16, 20] if k <= len(cum_var)
        },
        "folds": fold_summaries
    }

    out_json = RESULTS_DIR / "pca_decomposition_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    log("\n" + "=" * 70)
    log(f"[SUCCESS] Dimensionality reduction complete! Files saved to: {FOLDS_DIR}")
    log("=" * 70)

if __name__ == "__main__":
    main()
