import os
os.environ["PYTHONIOENCODING"] = "utf-8"
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

"""
01_extract_and_explore_datasets.py
===================================
QML for Materials Property Prediction — Dataset Extraction & Exploration

Downloads all 6 standard datasets via Matminer/MatBench, performs EDA,
computes statistics, checks data quality, and cross-references overlapping
materials by material_id.

Datasets:
  - dielectric_constant      (1,056 structures)
  - piezoelectric_tensor      (941 structures)
  - elastic_tensor_2015       (1,181 structures)
  - matbench_dielectric       (4,764 structures)
  - matbench_log_gvrh         (10,987 structures)
  - phonon_dielectric_mp      (1,296 structures)

Python 3.11.9 required.
"""

import json
import warnings
import traceback
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# ── Project paths ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(r"d:\Desktop\Material_science_qml")
DATA_RAW     = PROJECT_ROOT / "data" / "raw"
DATA_PROC    = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR  = PROJECT_ROOT / "results"

for d in [DATA_RAW, DATA_PROC, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Logging helper ─────────────────────────────────────────────────────
def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")
    sys.stdout.flush()

# ══════════════════════════════════════════════════════════════════════
# SECTION 1: Load Matminer datasets
# ══════════════════════════════════════════════════════════════════════
log("=" * 70)
log("SECTION 1: Loading Matminer datasets")
log("=" * 70)

from matminer.datasets import load_dataset

MATMINER_DATASETS = {
    "dielectric_constant":   {"expected_rows": 1056},
    "piezoelectric_tensor":  {"expected_rows": 941},
    "elastic_tensor_2015":   {"expected_rows": 1181},
    "matbench_dielectric":   {"expected_rows": 4764},
    "matbench_log_gvrh":     {"expected_rows": 10987},
    "phonon_dielectric_mp":  {"expected_rows": 1296},
}

datasets = {}
for name, meta in MATMINER_DATASETS.items():
    log(f"  Loading '{name}' ...")
    try:
        df = load_dataset(name)
        datasets[name] = df
        status = "[OK]" if len(df) == meta["expected_rows"] else f"[WARN] Expected {meta['expected_rows']}, got {len(df)}"
        log(f"    Shape: {df.shape}  {status}")
    except Exception as e:
        log(f"    [FAILED]: {e}")
        traceback.print_exc()

# ══════════════════════════════════════════════════════════════════════
# SECTION 2: Inspect columns and dtypes
# ══════════════════════════════════════════════════════════════════════
log("\n" + "=" * 70)
log("SECTION 2: Column inspection")
log("=" * 70)

dataset_info = {}
for name, df in datasets.items():
    log(f"\n── {name} ({len(df)} rows, {len(df.columns)} cols) ──")
    cols_info = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        n_null = df[col].isnull().sum()
        sample = None
        try:
            sample = str(df[col].dropna().iloc[0])[:80] if len(df[col].dropna()) > 0 else "N/A"
        except:
            sample = "N/A"
        cols_info.append({
            "column": col,
            "dtype": dtype,
            "null_count": int(n_null),
            "null_pct": round(100 * n_null / len(df), 2),
            "sample": sample
        })
        null_str = f" (nulls: {n_null})" if n_null > 0 else ""
        log(f"    {col:<30s} {dtype:<15s}{null_str}  sample: {sample}")
    
    dataset_info[name] = {
        "shape": list(df.shape),
        "columns": cols_info
    }

# ══════════════════════════════════════════════════════════════════════
# SECTION 3: Target variable statistics
# ══════════════════════════════════════════════════════════════════════
log("\n" + "=" * 70)
log("SECTION 3: Target variable statistics")
log("=" * 70)

target_stats = {}

# ── dielectric_constant targets ──
if "dielectric_constant" in datasets:
    df = datasets["dielectric_constant"]
    log("\n── dielectric_constant targets ──")
    for col in ["n", "poly_electronic", "poly_total", "band_gap"]:
        if col in df.columns:
            s = df[col].dropna()
            stats = {
                "count": int(len(s)),
                "mean": round(float(s.mean()), 4),
                "std": round(float(s.std()), 4),
                "min": round(float(s.min()), 4),
                "25%": round(float(s.quantile(0.25)), 4),
                "50%": round(float(s.quantile(0.50)), 4),
                "75%": round(float(s.quantile(0.75)), 4),
                "max": round(float(s.max()), 4),
                "skew": round(float(s.skew()), 4),
                "kurtosis": round(float(s.kurtosis()), 4),
            }
            target_stats[f"dielectric_constant.{col}"] = stats
            log(f"  {col}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
                f"range=[{stats['min']:.4f}, {stats['max']:.4f}], skew={stats['skew']:.4f}")

# ── piezoelectric_tensor targets ──
if "piezoelectric_tensor" in datasets:
    df = datasets["piezoelectric_tensor"]
    log("\n── piezoelectric_tensor targets ──")
    for col in ["eij_max"]:
        if col in df.columns:
            s = df[col].dropna()
            stats = {
                "count": int(len(s)),
                "mean": round(float(s.mean()), 4),
                "std": round(float(s.std()), 4),
                "min": round(float(s.min()), 4),
                "25%": round(float(s.quantile(0.25)), 4),
                "50%": round(float(s.quantile(0.50)), 4),
                "75%": round(float(s.quantile(0.75)), 4),
                "max": round(float(s.max()), 4),
                "skew": round(float(s.skew()), 4),
                "kurtosis": round(float(s.kurtosis()), 4),
            }
            target_stats[f"piezoelectric_tensor.{col}"] = stats
            log(f"  {col}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
                f"range=[{stats['min']:.4f}, {stats['max']:.4f}], skew={stats['skew']:.4f}")
    
    # Compute derived target: log(1 + max|e_iJ|)
    if "eij_max" in df.columns:
        log_eij = np.log(1 + df["eij_max"].dropna())
        stats = {
            "count": int(len(log_eij)),
            "mean": round(float(log_eij.mean()), 4),
            "std": round(float(log_eij.std()), 4),
            "min": round(float(log_eij.min()), 4),
            "max": round(float(log_eij.max()), 4),
            "skew": round(float(log_eij.skew()), 4),
        }
        target_stats["piezoelectric_tensor.log_1_plus_eij_max"] = stats
        log(f"  log(1+eij_max): mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
            f"range=[{stats['min']:.4f}, {stats['max']:.4f}], skew={stats['skew']:.4f}")

# ── elastic_tensor_2015 targets ──
if "elastic_tensor_2015" in datasets:
    df = datasets["elastic_tensor_2015"]
    log("\n── elastic_tensor_2015 targets ──")
    for col in ["G_VRH", "K_VRH", "poisson_ratio", "elastic_anisotropy"]:
        if col in df.columns:
            s = df[col].dropna()
            stats = {
                "count": int(len(s)),
                "mean": round(float(s.mean()), 4),
                "std": round(float(s.std()), 4),
                "min": round(float(s.min()), 4),
                "25%": round(float(s.quantile(0.25)), 4),
                "50%": round(float(s.quantile(0.50)), 4),
                "75%": round(float(s.quantile(0.75)), 4),
                "max": round(float(s.max()), 4),
                "skew": round(float(s.skew()), 4),
                "kurtosis": round(float(s.kurtosis()), 4),
            }
            target_stats[f"elastic_tensor_2015.{col}"] = stats
            log(f"  {col}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
                f"range=[{stats['min']:.4f}, {stats['max']:.4f}], skew={stats['skew']:.4f}")
    
    # Compute log10(G_VRH)
    if "G_VRH" in df.columns:
        valid = df["G_VRH"].dropna()
        valid = valid[valid > 0]
        log_g = np.log10(valid)
        stats = {
            "count": int(len(log_g)),
            "mean": round(float(log_g.mean()), 4),
            "std": round(float(log_g.std()), 4),
            "min": round(float(log_g.min()), 4),
            "max": round(float(log_g.max()), 4),
            "skew": round(float(log_g.skew()), 4),
        }
        target_stats["elastic_tensor_2015.log10_G_VRH"] = stats
        log(f"  log10(G_VRH): mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
            f"range=[{stats['min']:.4f}, {stats['max']:.4f}], skew={stats['skew']:.4f}")

# ── matbench_dielectric target ──
if "matbench_dielectric" in datasets:
    df = datasets["matbench_dielectric"]
    log("\n── matbench_dielectric target ──")
    col = "n"
    if col in df.columns:
        s = df[col].dropna()
        stats = {
            "count": int(len(s)),
            "mean": round(float(s.mean()), 4),
            "std": round(float(s.std()), 4),
            "min": round(float(s.min()), 4),
            "25%": round(float(s.quantile(0.25)), 4),
            "50%": round(float(s.quantile(0.50)), 4),
            "75%": round(float(s.quantile(0.75)), 4),
            "max": round(float(s.max()), 4),
            "skew": round(float(s.skew()), 4),
            "kurtosis": round(float(s.kurtosis()), 4),
        }
        target_stats[f"matbench_dielectric.{col}"] = stats
        log(f"  {col}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
            f"range=[{stats['min']:.4f}, {stats['max']:.4f}], skew={stats['skew']:.4f}")

# ── matbench_log_gvrh target ──
if "matbench_log_gvrh" in datasets:
    df = datasets["matbench_log_gvrh"]
    log("\n── matbench_log_gvrh target ──")
    col = "log10(G_VRH)"
    if col in df.columns:
        s = df[col].dropna()
        stats = {
            "count": int(len(s)),
            "mean": round(float(s.mean()), 4),
            "std": round(float(s.std()), 4),
            "min": round(float(s.min()), 4),
            "25%": round(float(s.quantile(0.25)), 4),
            "50%": round(float(s.quantile(0.50)), 4),
            "75%": round(float(s.quantile(0.75)), 4),
            "max": round(float(s.max()), 4),
            "skew": round(float(s.skew()), 4),
            "kurtosis": round(float(s.kurtosis()), 4),
        }
        target_stats[f"matbench_log_gvrh.{col}"] = stats
        log(f"  {col}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
            f"range=[{stats['min']:.4f}, {stats['max']:.4f}], skew={stats['skew']:.4f}")

# ══════════════════════════════════════════════════════════════════════
# SECTION 4: Structure complexity analysis
# ══════════════════════════════════════════════════════════════════════
log("\n" + "=" * 70)
log("SECTION 4: Structure complexity (nsites distribution)")
log("=" * 70)

structure_stats = {}
for name, df in datasets.items():
    struct_col = None
    if "structure" in df.columns:
        struct_col = "structure"
    
    if struct_col is not None:
        try:
            nsites = df[struct_col].apply(lambda s: len(s) if hasattr(s, '__len__') else np.nan).dropna()
            stats = {
                "count": int(len(nsites)),
                "mean_nsites": round(float(nsites.mean()), 2),
                "std_nsites": round(float(nsites.std()), 2),
                "min_nsites": int(nsites.min()),
                "max_nsites": int(nsites.max()),
                "median_nsites": int(nsites.median()),
            }
            structure_stats[name] = stats
            log(f"  {name}: nsites mean={stats['mean_nsites']:.1f}, "
                f"range=[{stats['min_nsites']}, {stats['max_nsites']}], "
                f"median={stats['median_nsites']}")
        except Exception as e:
            log(f"  {name}: Could not compute nsites — {e}")
    elif "nsites" in df.columns:
        s = df["nsites"].dropna()
        stats = {
            "count": int(len(s)),
            "mean_nsites": round(float(s.mean()), 2),
            "std_nsites": round(float(s.std()), 2),
            "min_nsites": int(s.min()),
            "max_nsites": int(s.max()),
            "median_nsites": int(s.median()),
        }
        structure_stats[name] = stats
        log(f"  {name}: nsites mean={stats['mean_nsites']:.1f}, "
            f"range=[{stats['min_nsites']}, {stats['max_nsites']}], "
            f"median={stats['median_nsites']}")

# ══════════════════════════════════════════════════════════════════════
# SECTION 5: Cross-reference materials across datasets
# ══════════════════════════════════════════════════════════════════════
log("\n" + "=" * 70)
log("SECTION 5: Cross-referencing materials by material_id")
log("=" * 70)

cross_ref = {}
id_sets = {}

for name, df in datasets.items():
    if "material_id" in df.columns:
        ids = set(df["material_id"].dropna().astype(str))
        id_sets[name] = ids
        log(f"  {name}: {len(ids)} unique material_ids")

# Compute pairwise overlaps
overlap_matrix = {}
id_set_names = list(id_sets.keys())
for i, name_a in enumerate(id_set_names):
    for j, name_b in enumerate(id_set_names):
        if i < j:
            overlap = id_sets[name_a] & id_sets[name_b]
            key = f"{name_a} ∩ {name_b}"
            overlap_matrix[key] = len(overlap)
            log(f"  Overlap: {name_a} ∩ {name_b} = {len(overlap)} materials")
            cross_ref[key] = {
                "count": len(overlap),
                "fraction_of_a": round(len(overlap) / len(id_sets[name_a]) * 100, 1),
                "fraction_of_b": round(len(overlap) / len(id_sets[name_b]) * 100, 1),
            }

# ══════════════════════════════════════════════════════════════════════
# SECTION 6: Space group distribution (for symmetry analysis)
# ══════════════════════════════════════════════════════════════════════
log("\n" + "=" * 70)
log("SECTION 6: Space group distribution (symmetry analysis)")
log("=" * 70)

spacegroup_info = {}
for name in ["dielectric_constant", "elastic_tensor_2015"]:
    if name in datasets and "space_group" in datasets[name].columns:
        df = datasets[name]
        sg_counts = df["space_group"].value_counts()
        n_unique = len(sg_counts)
        top5 = sg_counts.head(5).to_dict()
        spacegroup_info[name] = {
            "n_unique_spacegroups": n_unique,
            "top5": {str(k): int(v) for k, v in top5.items()},
        }
        log(f"  {name}: {n_unique} unique space groups")
        log(f"    Top 5: {top5}")

# Piezoelectric: check non-centrosymmetric count
if "piezoelectric_tensor" in datasets:
    df = datasets["piezoelectric_tensor"]
    if "space_group" in df.columns:
        sg_counts = df["space_group"].value_counts()
        n_unique = len(sg_counts)
        top5 = sg_counts.head(5).to_dict()
        spacegroup_info["piezoelectric_tensor"] = {
            "n_unique_spacegroups": n_unique,
            "top5": {str(k): int(v) for k, v in top5.items()},
        }
        log(f"  piezoelectric_tensor: {n_unique} unique space groups")
        log(f"    Top 5: {top5}")
        log(f"    Note: All 941 materials should be non-centrosymmetric")

# ══════════════════════════════════════════════════════════════════════
# SECTION 7: MatBench official fold validation
# ══════════════════════════════════════════════════════════════════════
log("\n" + "=" * 70)
log("SECTION 7: MatBench official fold validation")
log("=" * 70)

try:
    from matbench.bench import MatbenchBenchmark
    
    for task_name in ["matbench_dielectric", "matbench_log_gvrh"]:
        log(f"\n  Validating {task_name} ...")
        mb = MatbenchBenchmark(subset=[task_name], autoload=True)
        task = list(mb.tasks)[0]
        
        log(f"    Number of folds: {len(task.folds)}")
        
        for fold_idx in task.folds:
            X_train, y_train = task.get_train_and_val_data(fold_idx)
            X_test = task.get_test_data(fold_idx, include_target=False)
            log(f"    Fold {fold_idx}: train={len(X_train)}, test={len(X_test)}, "
                f"total={len(X_train)+len(X_test)}")
        
        log(f"    [OK] Official folds validated for {task_name}")

except Exception as e:
    log(f"  [WARN] MatBench fold validation failed: {e}")
    traceback.print_exc()

# ══════════════════════════════════════════════════════════════════════
# SECTION 8: Data quality checks
# ══════════════════════════════════════════════════════════════════════
log("\n" + "=" * 70)
log("SECTION 8: Data quality checks")
log("=" * 70)

quality_report = {}
for name, df in datasets.items():
    issues = []
    
    # Check for null values in key columns
    null_cols = df.columns[df.isnull().any()].tolist()
    if null_cols:
        null_info = {col: int(df[col].isnull().sum()) for col in null_cols}
        issues.append(f"Null values in {len(null_cols)} columns: {null_info}")
    
    # Check for duplicate structures/formulas
    if "formula" in df.columns:
        n_dup = df["formula"].duplicated().sum()
        if n_dup > 0:
            issues.append(f"Duplicate formulas: {n_dup}")
    
    if "material_id" in df.columns:
        n_dup = df["material_id"].duplicated().sum()
        if n_dup > 0:
            issues.append(f"Duplicate material_ids: {n_dup}")
    
    # Check for negative values in targets that should be positive
    for col in ["n", "G_VRH", "K_VRH", "eij_max", "poly_electronic", "poly_total"]:
        if col in df.columns:
            n_neg = (df[col] < 0).sum()
            if n_neg > 0:
                issues.append(f"Negative values in {col}: {n_neg}")
    
    quality_report[name] = {
        "n_issues": len(issues),
        "issues": issues,
        "total_nulls": int(df.isnull().sum().sum()),
    }
    
    if issues:
        log(f"  {name}: {len(issues)} issues found")
        for iss in issues:
            log(f"    - {iss}")
    else:
        log(f"  {name}: [OK] No quality issues detected")

# ══════════════════════════════════════════════════════════════════════
# SECTION 9: Save comprehensive summary
# ══════════════════════════════════════════════════════════════════════
log("\n" + "=" * 70)
log("SECTION 9: Saving comprehensive summary")
log("=" * 70)

summary = {
    "generated_at": datetime.now().isoformat(),
    "python_version": sys.version,
    "datasets_loaded": list(datasets.keys()),
    "dataset_shapes": {name: list(df.shape) for name, df in datasets.items()},
    "dataset_info": dataset_info,
    "target_statistics": target_stats,
    "structure_complexity": structure_stats,
    "cross_references": cross_ref,
    "spacegroup_distribution": spacegroup_info,
    "quality_report": quality_report,
}

summary_path = RESULTS_DIR / "dataset_exploration_summary.json"
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2, default=str)
log(f"  Summary saved to: {summary_path}")

# ══════════════════════════════════════════════════════════════════════
# SECTION 10: Print final overview table
# ══════════════════════════════════════════════════════════════════════
log("\n" + "=" * 70)
log("FINAL OVERVIEW")
log("=" * 70)

print("\n{:<30s} {:>8s} {:>6s} {:>20s} {:>15s}".format(
    "Dataset", "Samples", "Cols", "Primary Target", "Quality"))
print("-" * 85)

ds_meta = {
    "dielectric_constant":   ("n (refractive idx)", ),
    "piezoelectric_tensor":  ("eij_max", ),
    "elastic_tensor_2015":   ("G_VRH (shear mod.)", ),
    "matbench_dielectric":   ("n (refractive idx)", ),
    "matbench_log_gvrh":     ("log10(G_VRH)", ),
    "phonon_dielectric_mp":  ("phonon+dielectric", ),
}

for name, df in datasets.items():
    target = ds_meta.get(name, ("-",))[0]
    q = quality_report.get(name, {})
    quality = "Clean" if q.get("n_issues", 0) == 0 else f"{q['n_issues']} issues"
    print(f"{name:<30s} {len(df):>8d} {len(df.columns):>6d} {target:>20s} {quality:>15s}")

print()
log("Dataset extraction and exploration complete!")
log(f"Results saved to: {RESULTS_DIR}")
