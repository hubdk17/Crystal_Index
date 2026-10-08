import os
os.environ["PYTHONIOENCODING"] = "utf-8"
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

"""
02_featurize_materials.py
=========================
Extracts physical composition, stoichiometry, valence orbital, symmetry,
and density descriptors from crystal structures using Matminer.

Target Dataset: matbench_dielectric (4,764 samples)
Target Property: Refractive index n
"""

import json
import time
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from matminer.datasets import load_dataset
from matminer.featurizers.composition import ElementProperty, Stoichiometry, ValenceOrbital
from matminer.featurizers.structure import GlobalSymmetryFeatures, DensityFeatures

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(r"d:\Desktop\Material_science_qml")
DATA_RAW     = PROJECT_ROOT / "data" / "raw"
DATA_PROC    = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR  = PROJECT_ROOT / "results"

for d in [DATA_PROC, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")
    sys.stdout.flush()

def main():
    start_total = time.time()
    log("=" * 70)
    log("STAGE 1: Loading matbench_dielectric dataset")
    log("=" * 70)

    log("Loading dataset via matminer load_dataset('matbench_dielectric') ...")
    df = load_dataset("matbench_dielectric")
    log(f"Loaded {len(df)} structures. Target column: 'n'")

    # Extract composition from pymatgen structure
    log("Extracting compositions from pymatgen structures ...")
    df["composition"] = df["structure"].apply(lambda s: s.composition)

    # ── 1. Composition: Magpie elemental properties ───────────────────
    log("\n" + "=" * 70)
    log("STAGE 2: Featurizing with Magpie ElementProperty preset (132 features)")
    log("=" * 70)
    t0 = time.time()
    ep = ElementProperty.from_preset("magpie")
    ep.set_n_jobs(1)  # Single-process to prevent Windows pagefile exhaustion
    df = ep.featurize_dataframe(df, "composition", ignore_errors=True, pbar=True)
    log(f"Magpie featurization done in {time.time() - t0:.2f}s")

    # ── 2. Composition: Stoichiometry & Valence Orbital ───────────────
    log("\n" + "=" * 70)
    log("STAGE 3: Featurizing Stoichiometry & Valence Orbitals (11 features)")
    log("=" * 70)
    t0 = time.time()
    stoich = Stoichiometry()
    stoich.set_n_jobs(1)
    df = stoich.featurize_dataframe(df, "composition", ignore_errors=True, pbar=True)

    val = ValenceOrbital()
    val.set_n_jobs(1)
    df = val.featurize_dataframe(df, "composition", ignore_errors=True, pbar=True)
    log(f"Stoichiometry & Valence featurization done in {time.time() - t0:.2f}s")

    # ── 3. Structure: Symmetry & Density ──────────────────────────────
    log("\n" + "=" * 70)
    log("STAGE 4: Featurizing Global Symmetry & Density (7 features)")
    log("=" * 70)
    t0 = time.time()
    symm = GlobalSymmetryFeatures()
    symm.set_n_jobs(1)
    df = symm.featurize_dataframe(df, "structure", ignore_errors=True, pbar=True)

    dens = DensityFeatures()
    dens.set_n_jobs(1)
    df = dens.featurize_dataframe(df, "structure", ignore_errors=True, pbar=True)
    log(f"Symmetry & Density featurization done in {time.time() - t0:.2f}s")

    # ── 4. Cleaning & Imputation ──────────────────────────────────────
    log("\n" + "=" * 70)
    log("STAGE 5: Cleaning and Post-Processing")
    log("=" * 70)

    # Drop non-feature non-numeric columns except 'n'
    cols_to_drop = ["structure", "composition"]
    if "crystal_system" in df.columns:
        cols_to_drop.append("crystal_system")  # crystal_system_int is already present

    feature_df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Convert booleans to int
    bool_cols = feature_df.select_dtypes(include="bool").columns
    for c in bool_cols:
        feature_df[c] = feature_df[c].astype(int)

    feature_cols = [c for c in feature_df.columns if c != "n"]
    target_col = "n"

    log(f"Total raw features: {len(feature_cols)}")
    log(f"Total samples: {len(feature_df)}")

    # Check for NaNs
    nan_counts = feature_df[feature_cols].isnull().sum()
    nan_cols = nan_counts[nan_counts > 0]
    if len(nan_cols) > 0:
        log(f"Imputing NaNs in {len(nan_cols)} columns with column median ...")
        for c in nan_cols.index:
            med = feature_df[c].median()
            feature_df[c] = feature_df[c].fillna(med)

    # Verify zero NaNs remain
    remaining_nans = feature_df.isnull().sum().sum()
    log(f"Remaining NaNs in feature table: {remaining_nans}")

    # ── 5. Save Processed Data ────────────────────────────────────────
    log("\n" + "=" * 70)
    log("STAGE 6: Saving Processed Feature Tables")
    log("=" * 70)

    out_parquet = DATA_PROC / "matbench_dielectric_features.parquet"
    out_csv     = DATA_PROC / "matbench_dielectric_features.csv"

    feature_df.to_parquet(out_parquet, index=False)
    log(f"Saved parquet to: {out_parquet} ({out_parquet.stat().st_size / (1024*1024):.2f} MB)")

    # Save metadata summary
    feature_meta = {
        "dataset": "matbench_dielectric",
        "n_samples": int(len(feature_df)),
        "n_features": int(len(feature_cols)),
        "target": target_col,
        "feature_names": feature_cols,
        "target_summary": {
            "mean": float(feature_df[target_col].mean()),
            "std": float(feature_df[target_col].std()),
            "min": float(feature_df[target_col].min()),
            "max": float(feature_df[target_col].max()),
            "median": float(feature_df[target_col].median()),
            "skew": float(feature_df[target_col].skew()),
        },
        "featurization_time_seconds": round(time.time() - start_total, 2)
    }

    out_json = RESULTS_DIR / "featurization_metadata.json"
    with open(out_json, "w") as f:
        json.dump(feature_meta, f, indent=2)
    log(f"Saved metadata to: {out_json}")

    log(f"\n[SUCCESS] Featurization completed in {time.time() - start_total:.2f}s")

if __name__ == "__main__":
    main()
