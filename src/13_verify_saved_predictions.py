"""
================================================================================
SCRIPT 13: INDEPENDENT RECONSTRUCTION & VERIFICATION OF METRICS FROM SAVED PREDICTIONS
NO RETRAINING. STRICT POST-HOC AUDIT OF SERIALIZED ARTIFACTS.
================================================================================
Checks:
1. Exactly 4,764 out-of-fold rows per model.
2. Each material appears exactly once as an outer-test prediction.
3. All five fold IDs are present (0, 1, 2, 3, 4) with correct sample counts.
4. Zero duplicate material/fold pairs.
5. Zero missing predictions, NaNs, or infinite values.
6. y_true matches canonical MatBench values for matching immutable index.
7. Recompute fold-by-fold and 5-fold MAE, MedAE, RMSE, R2, Spearman rho.
8. Evaluate Grand Multi-Paradigm Ensemble reconstructed from saved predictions.
9. Verify whether 0.2794 +/- 0.0808 matches within rounding tolerance.
================================================================================
"""

import sys
import io
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, median_absolute_error, mean_squared_error, r2_score

# Ensure clean UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTIER_CSV = PROJECT_ROOT / "results" / "frontier_gnn_benchmark_20260923_133231" / "predictions" / "frontier_gnns_all_predictions.csv"
ADVANCED_CSV = PROJECT_ROOT / "results" / "gnn_variations_benchmark_20260923_122350" / "predictions" / "advanced_gnns_all_predictions.csv"
OUT_DIR = PROJECT_ROOT / "results" / "saved_predictions_independent_verification"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    ae = np.abs(y_true - y_pred)
    se = (y_true - y_pred) ** 2
    
    r_val, _ = pearsonr(y_true, y_pred) if len(np.unique(y_pred)) > 1 else (np.nan, np.nan)
    rho_val, _ = spearmanr(y_true, y_pred) if len(np.unique(y_pred)) > 1 else (np.nan, np.nan)
    
    return {
        "MAE": float(np.mean(ae)),
        "MedAE": float(np.median(ae)),
        "RMSE": float(np.sqrt(np.mean(se))),
        "R2": float(r2_score(y_true, y_pred)),
        "Pearson_r": float(r_val),
        "Spearman_rho": float(rho_val),
        "Max_AE": float(np.max(ae)),
        "Fraction_n_less_1": float(np.mean(y_pred < 1.0))
    }

def main():
    print("=" * 80)
    print("INDEPENDENT RECONSTRUCTION & VERIFICATION OF SAVED GNN PREDICTIONS")
    print("=" * 80)

    # 1. Load Canonical MatBench Dataset
    print("\n[Step 1] Loading Canonical MatBench v0.1 dielectric dataset...")
    from matbench.bench import MatbenchBenchmark
    bm = MatbenchBenchmark(benchmark="matbench_v0.1", subset=["matbench_dielectric"])
    task = list(bm.tasks)[0]
    task.load()
    canon_df = task.df
    canonical_n_samples = len(canon_df)
    canonical_identifiers = set(canon_df.index)
    canonical_y_true = canon_df["n"].to_dict()
    print(f"  Canonical sample count: {canonical_n_samples}")
    print(f"  Target variable: 'n'")

    # Also extract official test indices for each fold
    canonical_fold_test_ids = {}
    for f in range(5):
        test_inputs, test_outputs = task.get_test_data(f, include_target=True)
        canonical_fold_test_ids[f] = set(test_inputs.index)
        print(f"  Official Fold {f} outer-test size: {len(test_inputs)}")

    # 2. Load Saved Predictions
    print("\n[Step 2] Loading saved prediction CSVs...")
    assert FRONTIER_CSV.exists(), f"Missing frontier predictions: {FRONTIER_CSV}"
    assert ADVANCED_CSV.exists(), f"Missing advanced predictions: {ADVANCED_CSV}"
    
    df_front = pd.read_csv(FRONTIER_CSV)
    df_adv = pd.read_csv(ADVANCED_CSV)
    print(f"  Frontier predictions loaded: {len(df_front)} rows from {FRONTIER_CSV.name}")
    print(f"  Advanced predictions loaded: {len(df_adv)} rows from {ADVANCED_CSV.name}")

    frontier_models = df_front["model"].unique().tolist()
    advanced_models = df_adv["model"].unique().tolist()
    print(f"  Frontier models ({len(frontier_models)}): {frontier_models}")
    print(f"  Advanced models ({len(advanced_models)}): {advanced_models}")

    # 3. Construct Grand Multi-Paradigm Ensemble from raw predictions
    # Blend: Ensemble_NNLS, Ensemble_InvVal, Hierarchical_Global_GNN from Frontier
    #        Global_CGNN, Log_CGNN from Advanced
    piv_front = df_front.pivot_table(index=["fold", "identifier", "y_true"], columns="model", values="y_pred").reset_index()
    piv_adv = df_adv.pivot_table(index=["fold", "identifier", "y_true"], columns="model", values="y_pred").reset_index()
    merged = pd.merge(piv_front, piv_adv, on=["fold", "identifier", "y_true"], suffixes=("_front", "_adv"))
    
    grand_models = ["Ensemble_NNLS", "Ensemble_InvVal", "Hierarchical_Global_GNN", "Global_CGNN", "Log_CGNN"]
    for gm in grand_models:
        assert gm in merged.columns, f"Required grand ensemble component {gm} missing!"
    
    merged["y_pred_grand"] = merged[grand_models].mean(axis=1)
    
    # Create a DataFrame for Grand Multi-Paradigm Ensemble
    df_grand = pd.DataFrame({
        "fold": merged["fold"],
        "model": "Grand_MultiParadigm_Ensemble",
        "identifier": merged["identifier"],
        "y_true": merged["y_true"],
        "y_pred": merged["y_pred_grand"],
        "abs_error": np.abs(merged["y_true"] - merged["y_pred_grand"])
    })

    # Combined dictionary of all models to verify
    all_models_dict = {}
    for m in frontier_models:
        all_models_dict[m] = df_front[df_front["model"] == m].copy()
    for m in advanced_models:
        all_models_dict[m] = df_adv[df_adv["model"] == m].copy()
    all_models_dict["Grand_MultiParadigm_Ensemble"] = df_grand

    # 4. Rigorous Check Suite for Each Model
    print("\n" + "=" * 80)
    print("[Step 3] RUNNING 6-POINT INTEGRITY VERIFICATION ON SAVED PREDICTIONS")
    print("=" * 80)

    verification_results = []
    
    for m_name, df_m in all_models_dict.items():
        # Check 1: Exactly 4,764 out-of-fold rows
        n_rows = len(df_m)
        c1_pass = (n_rows == 4764)

        # Check 2: Each material appears exactly once as outer-test prediction
        unique_ids = df_m["identifier"].unique()
        n_unique_ids = len(unique_ids)
        ids_set = set(unique_ids)
        c2_pass = (n_unique_ids == 4764) and (ids_set == canonical_identifiers)

        # Check 3: All five fold IDs present with exact match to official partitions
        folds_present = sorted(df_m["fold"].unique().tolist())
        c3_folds = (folds_present == [0, 1, 2, 3, 4])
        fold_counts = df_m["fold"].value_counts().to_dict()
        
        # Verify that for each fold, the set of identifiers exactly matches canonical test set
        fold_id_matches = True
        for f in range(5):
            m_fold_ids = set(df_m[df_m["fold"] == f]["identifier"])
            if m_fold_ids != canonical_fold_test_ids[f]:
                fold_id_matches = False
                break
        c3_pass = c3_folds and fold_id_matches

        # Check 4: No duplicate material/fold pairs
        n_dups = df_m.duplicated(subset=["identifier", "fold"]).sum()
        c4_pass = (n_dups == 0)

        # Check 5: No missing predictions, NaNs, or infinite values
        nan_preds = df_m["y_pred"].isna().sum()
        inf_preds = np.isinf(df_m["y_pred"].values).sum()
        c5_pass = (nan_preds == 0) and (inf_preds == 0)

        # Check 6: y_true matches canonical MatBench values for matching identifier
        diffs = []
        for idx, row in df_m.iterrows():
            c_val = canonical_y_true[row["identifier"]]
            diffs.append(abs(row["y_true"] - c_val))
        max_target_diff = max(diffs)
        c6_pass = (max_target_diff < 1e-10)

        all_checks_passed = all([c1_pass, c2_pass, c3_pass, c4_pass, c5_pass, c6_pass])

        verification_results.append({
            "model": m_name,
            "n_rows": n_rows,
            "c1_exact_4764": c1_pass,
            "c2_unique_materials_4764": c2_pass,
            "c3_official_folds_match": c3_pass,
            "c4_zero_duplicates": c4_pass,
            "c5_zero_nans_infs": c5_pass,
            "c6_y_true_canonical_match": c6_pass,
            "max_target_diff": max_target_diff,
            "all_integrity_passed": all_checks_passed
        })

        status_str = "PASS" if all_checks_passed else "FAIL"
        print(f"[{status_str}] {m_name:<30} | Rows: {n_rows} | Uniq: {n_unique_ids} | Folds: {folds_present} | NaNs/Infs: {nan_preds}/{inf_preds} | Max y_true diff: {max_target_diff:.2e}")

    df_verif = pd.DataFrame(verification_results)
    df_verif.to_csv(OUT_DIR / "integrity_verification_checklist.csv", index=False)

    # 5. Recompute Metrics from Saved Predictions
    print("\n" + "=" * 80)
    print("[Step 4] RECOMPUTING 5-FOLD BENCHMARK METRICS FROM SAVED PREDICTIONS")
    print("=" * 80)

    recomputed_summary = []
    fold_details = []

    for m_name, df_m in all_models_dict.items():
        fold_maes = []
        fold_medaes = []
        fold_rmses = []
        fold_r2s = []
        fold_rhos = []

        for f in range(5):
            sub_f = df_m[df_m["fold"] == f]
            y_t = sub_f["y_true"].values
            y_p = sub_f["y_pred"].values
            met = compute_metrics(y_t, y_p)

            fold_maes.append(met["MAE"])
            fold_medaes.append(met["MedAE"])
            fold_rmses.append(met["RMSE"])
            fold_r2s.append(met["R2"])
            fold_rhos.append(met["Spearman_rho"])

            fold_details.append({
                "model": m_name,
                "fold": f,
                "n_samples": len(y_t),
                **met
            })

        mae_mean = float(np.mean(fold_maes))
        mae_std = float(np.std(fold_maes, ddof=1))
        medae_mean = float(np.mean(fold_medaes))
        medae_std = float(np.std(fold_medaes, ddof=1))
        rmse_mean = float(np.mean(fold_rmses))
        r2_mean = float(np.mean(fold_r2s))
        rho_mean = float(np.mean(fold_rhos))

        # Also compute pooled metrics across all 4,764 predictions
        pooled_met = compute_metrics(df_m["y_true"].values, df_m["y_pred"].values)

        recomputed_summary.append({
            "model": m_name,
            "5Fold_MAE_Mean": mae_mean,
            "5Fold_MAE_Std": mae_std,
            "5Fold_MedAE_Mean": medae_mean,
            "5Fold_MedAE_Std": medae_std,
            "5Fold_RMSE_Mean": rmse_mean,
            "5Fold_R2_Mean": r2_mean,
            "5Fold_Spearman_rho_Mean": rho_mean,
            "Pooled_MAE": pooled_met["MAE"],
            "Pooled_MedAE": pooled_met["MedAE"],
            "Pooled_RMSE": pooled_met["RMSE"],
            "Pooled_R2": pooled_met["R2"],
            "Pooled_Spearman_rho": pooled_met["Spearman_rho"],
            "Fold_0_MAE": fold_maes[0],
            "Fold_1_MAE": fold_maes[1],
            "Fold_2_MAE": fold_maes[2],
            "Fold_3_MAE": fold_maes[3],
            "Fold_4_MAE": fold_maes[4]
        })

    df_recomp_summary = pd.DataFrame(recomputed_summary).sort_values("5Fold_MAE_Mean")
    df_fold_details = pd.DataFrame(fold_details)

    df_recomp_summary.to_csv(OUT_DIR / "recomputed_summary_from_saved_predictions.csv", index=False)
    df_fold_details.to_csv(OUT_DIR / "recomputed_fold_by_fold_details.csv", index=False)

    print("\nRecomputed Benchmark Summary (Sorted by 5-Fold Outer-Test MAE):")
    print("-" * 110)
    print(f"{'Model':<30} | {'5-Fold MAE':<20} | {'5-Fold MedAE':<18} | {'RMSE':<10} | {'R2':<8} | {'Spearman rho':<10}")
    print("-" * 110)
    for _, row in df_recomp_summary.iterrows():
        mae_str = f"{row['5Fold_MAE_Mean']:.4f} +/- {row['5Fold_MAE_Std']:.4f}"
        med_str = f"{row['5Fold_MedAE_Mean']:.4f} +/- {row['5Fold_MedAE_Std']:.4f}"
        print(f"{row['model']:<30} | {mae_str:<20} | {med_str:<18} | {row['5Fold_RMSE_Mean']:<10.4f} | {row['5Fold_R2_Mean']:<8.4f} | {row['5Fold_Spearman_rho_Mean']:<10.4f}")
    print("-" * 110)

    # 6. Specifically Verify Grand Multi-Paradigm Ensemble Pass Criterion
    print("\n" + "=" * 80)
    print("[Step 5] VERIFYING PASS CRITERION FOR GRAND MULTI-PARADIGM ENSEMBLE")
    print("=" * 80)
    grand_row = df_recomp_summary[df_recomp_summary["model"] == "Grand_MultiParadigm_Ensemble"].iloc[0]
    
    reported_grand_mae = 0.2794
    reported_grand_std = 0.0808
    recomputed_grand_mae = grand_row["5Fold_MAE_Mean"]
    recomputed_grand_std = grand_row["5Fold_MAE_Std"]

    mae_diff = abs(recomputed_grand_mae - reported_grand_mae)
    std_diff = abs(recomputed_grand_std - reported_grand_std)

    print(f"  Fold 0 MAE: {grand_row['Fold_0_MAE']:.4f}")
    print(f"  Fold 1 MAE: {grand_row['Fold_1_MAE']:.4f}")
    print(f"  Fold 2 MAE: {grand_row['Fold_2_MAE']:.4f}")
    print(f"  Fold 3 MAE: {grand_row['Fold_3_MAE']:.4f}")
    print(f"  Fold 4 MAE: {grand_row['Fold_4_MAE']:.4f}")
    print(f"\n  Reported 5-Fold Score:   {reported_grand_mae:.4f} +/- {reported_grand_std:.4f}")
    print(f"  Recomputed 5-Fold Score: {recomputed_grand_mae:.4f} +/- {recomputed_grand_std:.4f}")
    print(f"  Difference (MAE):        {mae_diff:.6f}")
    print(f"  Difference (Std):        {std_diff:.6f}")

    pass_tolerance = 1e-4
    passes_criterion = (mae_diff < pass_tolerance) and (std_diff < pass_tolerance)

    # Also verify other reported frontier models
    frontier_checks = {
        "Ensemble_Softmax": (0.2825, 0.0828),
        "Ensemble_InvVal": (0.2847, 0.0830),
        "DualHead_LogDirect_GNN": (0.2909, 0.0860),
        "Hierarchical_Global_GNN": (0.2917, 0.0802),
        "Angle_Aware_CGNN": (0.3004, 0.0848),
        "MultiHead_GraphTransformer": (0.3600, 0.0860),
        "Ensemble_GNN": (0.2903, 0.0804),
        "Global_CGNN": (0.2989, 0.0801),
        "Log_CGNN": (0.3057, 0.0804),
        "PE_Attn_CGNN": (0.3070, 0.0783),
        "CGCNN_Baseline": (0.3290, 0.0825)
    }

    print("\nVerification Across All Reported Models:")
    print("-" * 80)
    all_models_match = True
    for m, (rep_m, rep_s) in frontier_checks.items():
        if m in df_recomp_summary["model"].values:
            r = df_recomp_summary[df_recomp_summary["model"] == m].iloc[0]
            d_m = abs(r["5Fold_MAE_Mean"] - rep_m)
            d_s = abs(r["5Fold_MAE_Std"] - rep_s)
            m_pass = (d_m < pass_tolerance) and (d_s < pass_tolerance)
            if not m_pass:
                all_models_match = False
            status = "MATCH" if m_pass else "MISMATCH"
            print(f"  [{status}] {m:<28}: Rep={rep_m:.4f}+/-{rep_s:.4f} | Rec={r['5Fold_MAE_Mean']:.4f}+/-{r['5Fold_MAE_Std']:.4f} | diff=({d_m:.5f}, {d_s:.5f})")

    verdict_text = {
        "benchmark": "matbench_v0.1_matbench_dielectric",
        "n_samples": canonical_n_samples,
        "n_models_verified": len(all_models_dict),
        "grand_ensemble_recomputed_mae": float(recomputed_grand_mae),
        "grand_ensemble_recomputed_std": float(recomputed_grand_std),
        "grand_ensemble_reported_mae": float(reported_grand_mae),
        "grand_ensemble_reported_std": float(reported_grand_std),
        "mae_diff": float(mae_diff),
        "std_diff": float(std_diff),
        "pass_criterion": bool(passes_criterion),
        "all_reported_models_match": bool(all_models_match),
        "authorized_statement": (
            "The reported five-fold score was independently reconstructed from saved out-of-fold predictions."
            if passes_criterion else "RECONSTRUCTION_FAILED"
        )
    }

    with open(OUT_DIR / "verification_verdict.json", "w", encoding="utf-8") as f:
        json.dump(verdict_text, f, indent=2)

    print("\n" + "=" * 80)
    if passes_criterion and all_models_match:
        print("FINAL VERDICT: ALL PASS CRITERIA SATISFIED (100% RECONSTRUCTION)")
        print(f"  \"{verdict_text['authorized_statement']}\"")
    else:
        print("FINAL VERDICT: CRITERION FAILED")
    print("=" * 80)

if __name__ == "__main__":
    main()
