"""
================================================================================
SCRIPT 16: FROZEN CHEMISTRY-GROUPED OOD CONFIRMATION BENCHMARK
BENCHMARK: MATBENCH v0.1 `matbench_dielectric` (N = 4,764)
MODEL: FROZEN DualHead_LogDirect_GNN (Single Publication Confirmation Model)
SPLIT: 5-Fold GroupKFold by Chemical System (ZERO Chemical System Overlap)
================================================================================
Scientific Objective:
Evaluate out-of-distribution (OOD) generalization of the frozen DualHead_LogDirect_GNN
to completely unseen chemical systems. Pre-save group assignments and fold manifests.
Maintain exact frozen architecture, loss, hyperparameters, and inner-validation
checkpointing. Report degradation relative to standard MatBench folds.
================================================================================
"""

import os
os.environ["PYTHONIOENCODING"] = "utf-8"
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import time
import json
import importlib
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, median_absolute_error, mean_squared_error, r2_score

import torch
from torch.utils.data import DataLoader

from matbench.bench import MatbenchBenchmark

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import exact frozen architecture, dataset, and loss from Script 11
frontier_mod = importlib.import_module("src.11_train_frontier_gnn_exploration")
DualHead_LogDirect_GNN = frontier_mod.DualHead_LogDirect_GNN
MultitaskLoss = frontier_mod.MultitaskLoss
FrontierCrystalDataset = frontier_mod.FrontierCrystalDataset
collate_frontier_graphs = frontier_mod.collate_frontier_graphs
compute_metrics = frontier_mod.compute_metrics

# Device strictly CPU to avoid GPU Blackwell sm_120 driver mismatch
DEVICE = torch.device("cpu")
RUN_DIR = PROJECT_ROOT / "results" / "chemistry_grouped_ood_confirmation"
MANIFEST_DIR = RUN_DIR / "manifests"
PRED_DIR = RUN_DIR / "predictions"
TABLE_DIR = RUN_DIR / "tables"
FIG_DIR = RUN_DIR / "figures"
CKPT_DIR = RUN_DIR / "checkpoints"

for d in [RUN_DIR, MANIFEST_DIR, PRED_DIR, TABLE_DIR, FIG_DIR, CKPT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

LOG_FILE = RUN_DIR / "run_log.txt"

def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    formatted = f"[{ts}] {msg}"
    print(formatted)
    sys.stdout.flush()
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(formatted + "\n")
    except Exception:
        pass

def main():
    log("=" * 80)
    log("STARTING FROZEN CHEMISTRY-GROUPED OOD CONFIRMATION BENCHMARK")
    log("=" * 80)
    log(f"Execution Device: {DEVICE}")

    # 1. Load Canonical MatBench Dataset & Extract Chemical Systems
    log("\n[Step 1] Loading official MatBench dataset and computing chemical systems...")
    bm = MatbenchBenchmark(benchmark="matbench_v0.1", subset=["matbench_dielectric"])
    task = list(bm.tasks)[0]
    task.load()
    df_official = task.df
    total_samples = len(df_official)
    log(f"Total canonical samples: {total_samples}")

    chemical_systems = []
    reduced_formulas = []
    sample_ids = []
    y_true_all = df_official["n"].values.astype(np.float64)

    for i in range(total_samples):
        st = df_official["structure"].iloc[i]
        comp = st.composition
        el_symbols = sorted(list(set(el.symbol for el in comp.elements)))
        chem_sys = "-".join(el_symbols)
        chemical_systems.append(chem_sys)
        reduced_formulas.append(comp.reduced_formula)
        sample_ids.append(f"mb-dielectric-{i+1:04d}")

    unique_chem_systems = sorted(list(set(chemical_systems)))
    log(f"Unique chemical systems: {len(unique_chem_systems)}")

    # 2. Pre-Save Group Assignments & Partition 5-Fold GroupKFold
    log("\n[Step 2] Partitioning 5-Fold GroupKFold strictly by Chemical System...")
    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(np.arange(total_samples), groups=chemical_systems))

    df_groups = pd.DataFrame({
        "original_index": np.arange(total_samples),
        "matbench_id": sample_ids,
        "reduced_formula": reduced_formulas,
        "chemical_system": chemical_systems,
        "target_n": y_true_all
    })

    fold_assignments = np.zeros(total_samples, dtype=int)
    for f_idx, (tr_idx, te_idx) in enumerate(splits):
        fold_assignments[te_idx] = f_idx
    df_groups["ood_fold"] = fold_assignments
    df_groups.to_csv(MANIFEST_DIR / "group_assignments.csv", index=False)
    log(f"Pre-saved group assignments to: {MANIFEST_DIR / 'group_assignments.csv'}")

    # Export Manifests for Each Fold and Verify Zero Overlap
    manifest_audit_rows = []
    for f_idx, (tr_idx, te_idx) in enumerate(splits):
        df_tr = df_groups.iloc[tr_idx].copy()
        df_te = df_groups.iloc[te_idx].copy()

        df_tr.to_csv(MANIFEST_DIR / f"fold_{f_idx}_train_manifest.csv", index=False)
        df_te.to_csv(MANIFEST_DIR / f"fold_{f_idx}_test_manifest.csv", index=False)

        tr_sys = set(df_tr["chemical_system"])
        te_sys = set(df_te["chemical_system"])
        overlap_sys = tr_sys & te_sys
        assert len(overlap_sys) == 0, f"FATAL: Chemical system overlap in fold {f_idx}!"

        manifest_audit_rows.append({
            "fold": f_idx,
            "train_samples": len(df_tr),
            "test_samples": len(df_te),
            "train_systems": len(tr_sys),
            "test_systems": len(te_sys),
            "chemical_system_overlap": len(overlap_sys),
            "overlap_status": "ZERO_OVERLAP_PASS"
        })
        log(f"Fold {f_idx} OOD Manifest -> Train: {len(df_tr)} (Sys: {len(tr_sys)}) | Test: {len(df_te)} (Sys: {len(te_sys)}) | Overlap: {len(overlap_sys)}")

    df_manifest_audit = pd.DataFrame(manifest_audit_rows)
    df_manifest_audit.to_csv(TABLE_DIR / "ood_manifest_audit.csv", index=False)

    # 3. Load or Build Graph Cache
    log("\n[Step 3] Loading or computing frontier graph cache...")
    cache_path = PROJECT_ROOT / "data" / "processed" / "frontier_graphs_cache.pt"
    if cache_path.exists():
        all_graphs = torch.load(cache_path, weights_only=False)
        log(f"Loaded {len(all_graphs)} graphs from cache.")
    else:
        log("Precomputing frontier crystal graphs with Chebyshev angles and 12-dim macroscopic physics...")
        gdf6 = frontier_mod.GaussianDistance(dmin=0.0, dmax=6.0, num_bins=41)
        all_graphs = [
            frontier_mod.build_frontier_structure_graph(
                df_full["structure"].iloc[i],
                float(y_raw_full[i]),
                radius=6.0,
                max_nbr=12,
                gdf=gdf6
            )
            for i in range(len(df_full))
        ]
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(all_graphs, cache_path)
        log(f"Precomputed and saved {len(all_graphs)} graphs to cache: {cache_path}")

    # 4. Execute 5-Fold OOD Training & Evaluation
    log("\n" + "=" * 80)
    log("[Step 4] EXECUTING FROZEN DualHead_LogDirect_GNN ON 5 OOD FOLDS")
    log("=" * 80)

    EPOCHS = 25
    BATCH_SIZE = 48
    LR = 1.8e-3
    WEIGHT_DECAY = 1e-4

    all_test_predictions = []
    fold_metric_records = []

    for f_idx, (tr_idx, te_idx) in enumerate(splits):
        log("\n" + "-" * 70)
        log(f"--- EXECUTING OOD FOLD {f_idx} / 4 (GroupKFold Unseen Chemical Systems) ---")
        log("-" * 70)

        # Inner-train and inner-validation split (85/15) strictly from outer-train
        np.random.seed(200 + f_idx)
        n_outer_train = len(tr_idx)
        perm = np.random.permutation(n_outer_train)
        n_inner_val = int(0.15 * n_outer_train)

        val_sub_idx = [tr_idx[k] for k in perm[:n_inner_val]]
        tr_sub_idx = [tr_idx[k] for k in perm[n_inner_val:]]

        log(f"  Outer Train: {n_outer_train} -> Inner Train: {len(tr_sub_idx)}, Inner Val: {len(val_sub_idx)}")
        log(f"  Outer Test (Held-Out Unseen Chem Systems): {len(te_idx)}")

        # Fit StandardScaler strictly on inner-training global descriptors
        inner_tr_global_raw = np.array([all_graphs[i]["global_12d"] for i in tr_sub_idx], dtype=np.float32)
        scaler = StandardScaler().fit(inner_tr_global_raw)

        train_ds = FrontierCrystalDataset([all_graphs[i] for i in tr_sub_idx], global_scaler=scaler)
        val_ds   = FrontierCrystalDataset([all_graphs[i] for i in val_sub_idx], global_scaler=scaler)
        test_ds  = FrontierCrystalDataset([all_graphs[i] for i in te_idx], global_scaler=scaler)

        loader_tr  = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_frontier_graphs)
        loader_val = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_frontier_graphs)
        loader_te  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_frontier_graphs)

        # Initialize Frozen Model
        torch.manual_seed(300 + f_idx)
        model = DualHead_LogDirect_GNN(n_atom_prior=8, edge_dim=58, n_global=12, hidden_dim=64, n_conv=3).to(DEVICE)
        criterion = MultitaskLoss(beta=0.15, lambda_log=0.5)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

        best_val_mae = float("inf")
        best_state = None
        best_epoch = 0

        t0 = time.time()
        for epoch in range(1, EPOCHS + 1):
            model.train()
            for batch in loader_tr:
                optimizer.zero_grad()
                p_dir, p_log = model(batch)
                loss = criterion(p_dir, p_log, batch["targets"], batch["targets_log"])
                loss.backward()
                optimizer.step()

            scheduler.step()

            # Inner-Validation checkpointing strictly on inner-val set
            model.eval()
            val_trues, val_preds = [], []
            with torch.no_grad():
                for batch in loader_val:
                    p_dir, p_log = model(batch)
                    p_blended = 0.5 * p_dir + 0.5 * (0.99 + torch.exp(p_log))
                    val_trues.extend(batch["targets"].view(-1).cpu().numpy())
                    val_preds.extend(p_blended.view(-1).cpu().numpy())

            cur_val_mae = float(mean_absolute_error(val_trues, val_preds))
            if cur_val_mae < best_val_mae:
                best_val_mae = cur_val_mae
                best_epoch = epoch
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        fit_time = time.time() - t0
        log(f"  Training complete in {fit_time:.1f}s | Best Epoch: {best_epoch} | Best Inner-Val MAE: {best_val_mae:.4f}")

        # Save checkpoint
        ckpt_path = CKPT_DIR / f"dualhead_ood_fold_{f_idx}.pt"
        torch.save({"model_state": best_state, "fold": f_idx, "best_val_mae": best_val_mae}, ckpt_path)

        # Single-pass outer-test evaluation on held-out unseen chemical systems
        model.load_state_dict(best_state)
        model.eval()
        test_preds, test_trues = [], []
        with torch.no_grad():
            for batch in loader_te:
                p_dir, p_log = model(batch)
                p_blended = 0.5 * p_dir + 0.5 * (0.99 + torch.exp(p_log))
                test_preds.extend(p_blended.view(-1).cpu().numpy())
                test_trues.extend(batch["targets"].view(-1).cpu().numpy())

        test_preds_arr = np.array(test_preds, dtype=np.float64)
        test_trues_arr = np.array(test_trues, dtype=np.float64)

        fold_m = compute_metrics(test_trues_arr, test_preds_arr)
        fold_metric_records.append({
            "fold": f_idx,
            "model": "DualHead_LogDirect_GNN_OOD",
            "val_mae": best_val_mae,
            "best_epoch": best_epoch,
            "fit_time_s": round(fit_time, 2),
            **fold_m
        })

        log(f"  OOD Outer Test -> MAE: {fold_m['MAE']:.4f} | MedAE: {fold_m['MedAE']:.4f} | RMSE: {fold_m['RMSE']:.4f} | R2: {fold_m['R2']:.4f} | rho: {fold_m['Spearman_rho']:.4f}")

        # Record test predictions
        for j, global_idx in enumerate(te_idx):
            all_test_predictions.append({
                "ood_fold": f_idx,
                "model": "DualHead_LogDirect_GNN_OOD",
                "identifier": sample_ids[global_idx],
                "reduced_formula": reduced_formulas[global_idx],
                "chemical_system": chemical_systems[global_idx],
                "y_true": float(test_trues_arr[j]),
                "y_pred": float(test_preds_arr[j]),
                "abs_error": float(abs(test_preds_arr[j] - test_trues_arr[j]))
            })

    # 5. Export Predictions and Summary Tables
    df_all_preds = pd.DataFrame(all_test_predictions)
    preds_csv = PRED_DIR / "ood_predictions.csv"
    df_all_preds.to_csv(preds_csv, index=False)
    log(f"\nAll OOD predictions saved to {preds_csv} ({len(df_all_preds)} rows).")

    df_fold_results = pd.DataFrame(fold_metric_records)
    df_fold_results.to_csv(TABLE_DIR / "ood_fold_results.csv", index=False)

    # Compute 5-fold OOD summary
    ood_mae_mean = float(df_fold_results["MAE"].mean())
    ood_mae_std = float(df_fold_results["MAE"].std())
    ood_medae_mean = float(df_fold_results["MedAE"].mean())
    ood_medae_std = float(df_fold_results["MedAE"].std())
    ood_rmse_mean = float(df_fold_results["RMSE"].mean())
    ood_r2_mean = float(df_fold_results["R2"].mean())
    ood_rho_mean = float(df_fold_results["Spearman_rho"].mean())

    standard_matbench_mae = 0.2909
    standard_matbench_medae = 0.0631
    mae_degradation = ood_mae_mean - standard_matbench_mae
    pct_degradation = (mae_degradation / standard_matbench_mae) * 100.0

    log("\n" + "=" * 80)
    log("CHEMISTRY-GROUPED OOD CONFIRMATION RESULTS SUMMARY")
    log("=" * 80)
    log(f"Model: Frozen DualHead_LogDirect_GNN (Single Publication Confirmation Model)")
    log(f"Split: 5-Fold GroupKFold by Chemical System (N = {total_samples})")
    log(f"5-Fold OOD MAE:          {ood_mae_mean:.4f} +/- {ood_mae_std:.4f}")
    log(f"5-Fold OOD MedAE:        {ood_medae_mean:.4f} +/- {ood_medae_std:.4f}")
    log(f"5-Fold OOD RMSE:         {ood_rmse_mean:.4f}")
    log(f"5-Fold OOD R2:           {ood_r2_mean:.4f}")
    log(f"5-Fold OOD Spearman rho: {ood_rho_mean:.4f}")
    log(f"\nComparison vs Standard MatBench Folds:")
    log(f"  Standard MatBench MAE: {standard_matbench_mae:.4f}")
    log(f"  OOD Extrapolation MAE: {ood_mae_mean:.4f}")
    log(f"  Degradation:           {mae_degradation:+.4f} ({pct_degradation:+.2f}%)")

    # 6. Save Verdict and Safe Interpretation
    verdict = {
        "benchmark": "matbench_v0.1_matbench_dielectric_GroupKFold_OOD",
        "model": "DualHead_LogDirect_GNN",
        "split_type": "GroupKFold_by_chemical_system",
        "n_samples": total_samples,
        "n_unique_chemical_systems": len(unique_chem_systems),
        "zero_chemical_system_overlap": True,
        "ood_5fold_mae_mean": ood_mae_mean,
        "ood_5fold_mae_std": ood_mae_std,
        "ood_5fold_medae_mean": ood_medae_mean,
        "ood_5fold_medae_std": ood_medae_std,
        "ood_rmse_mean": ood_rmse_mean,
        "ood_r2_mean": ood_r2_mean,
        "ood_spearman_rho_mean": ood_rho_mean,
        "standard_matbench_mae": standard_matbench_mae,
        "mae_degradation": mae_degradation,
        "pct_degradation": pct_degradation,
        "safe_interpretation": "The frozen model was evaluated under chemistry-grouped OOD splits to assess extrapolation to unseen chemical systems. This supplementary result is not directly comparable with MatBench leaderboard scores."
    }

    with open(RUN_DIR / "ood_confirmation_verdict.json", "w", encoding="utf-8") as f:
        json.dump(verdict, f, indent=2)

    # 7. Generate OOD Parity and Error Degradation Figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Parity Plot
        ax = axes[0]
        y_t = df_all_preds["y_true"].values
        y_p = df_all_preds["y_pred"].values
        ax.scatter(y_t, y_p, alpha=0.35, s=16, color="#4C72B0", edgecolors="none")
        max_v = min(max(np.max(y_t), np.max(y_p)), 15.0)
        ax.plot([1.0, max_v], [1.0, max_v], "r--", lw=1.8, label="Ideal Line (y = x)")
        ax.set_xlim(0.8, max_v)
        ax.set_ylim(0.8, max_v)
        ax.set_xlabel("DFT True Refractive Index $n$", fontsize=12)
        ax.set_ylabel("Predicted Refractive Index $\hat{n}$", fontsize=12)
        ax.set_title(f"Chemistry-Grouped OOD Parity Plot\n5-Fold MAE: {ood_mae_mean:.4f} $\pm$ {ood_mae_std:.4f}", fontsize=13, fontweight="bold")
        ax.legend(frameon=True)

        # Degradation Bar Chart
        ax2 = axes[1]
        bars = ax2.bar(["Standard MatBench\n(Official Folds)", "Chemistry-Grouped OOD\n(Unseen Chemical Systems)"],
                       [standard_matbench_mae, ood_mae_mean],
                       color=["#2b5c8f", "#d95f02"], width=0.45, edgecolor="black")
        ax2.set_ylabel("5-Fold Mean Absolute Error (MAE)", fontsize=12)
        ax2.set_title(f"Generalization Degradation to Unseen Chem Systems\nDegradation: {mae_degradation:+.4f} ({pct_degradation:+.1f}%)", fontsize=13, fontweight="bold")
        ax2.set_ylim(0, max(ood_mae_mean, standard_matbench_mae) * 1.3)
        for bar in bars:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2.0, h + 0.01, f"{h:.4f}", ha="center", va="bottom", fontsize=12, fontweight="bold")

        plt.tight_layout()
        fig_path = FIG_DIR / "ood_generalization_confirmation.png"
        plt.savefig(fig_path, dpi=300)
        plt.close()
        log(f"Figure saved to: {fig_path}")

        # Copy to artifact directory
        import shutil
        artifact_fig = Path(r"C:\Users\HP\.gemini\antigravity-ide\brain\7e1462b3-34c9-4d76-a8c1-ddab45f77d0b\ood_generalization_confirmation.png")
        shutil.copy(fig_path, artifact_fig)
        log(f"Figure copied to artifact directory: {artifact_fig}")
    except Exception as e:
        log(f"Warning: Figure generation skipped due to: {e}")

    log("\n" + "=" * 80)
    log("OOD CONFIRMATION BENCHMARK COMPLETE")
    log(f"Safe Interpretation: \"{verdict['safe_interpretation']}\"")
    log("=" * 80)

if __name__ == "__main__":
    main()
