"""
================================================================================
SCRIPT 09: ADVANCED CRYSTAL GNN VARIATIONS (LEAKAGE-FREE BENCHMARK)
MATBENCH v0.1 `matbench_dielectric` (N = 4,764)
================================================================================
Scientific Objective:
Implement, train, and evaluate advanced periodic crystal Graph Neural Network
(GNN) architectural and training variations to achieve state-of-the-art predictive
accuracy on refractive index n under strict zero-data-leakage protocols.

Key Innovations:
1. Physics-Enriched Attentive Crystal GNN (PE-Attn-CGNN):
   - Embedding(Z) + 8 normalized elemental physical priors
   - Multi-head edge-conditioned attention mechanism
   - Pruned interaction graph (R_cut = 6.0 A, k = 12)
   - Bounded physical readout (n >= 1.0)
2. Global-Conditioned Dual-Path GNN (Global-CGNN / MEGNet style):
   - Macroscopic unit cell state vector u = [density, vpa, packing_fraction, n_elements]
   - Tri-directional message passing (node <-> edge <-> global state)
   - Strictly isolated inner-train normalization of global state
3. Log-Space Stabilized Robust GNN (Log-CGNN):
   - Target formulation: z = log(n - 0.99)
   - Huber / Smooth L1 loss in log space to eliminate Penn gap divergence gradient shocks
   - Guaranteed physical lower bound n >= 0.99 by exponential construction
4. Multi-Architecture Leakage-Free Ensemble (Ensemble-GNN):
   - Out-of-fold weighted blend of top architectures
   - Weights determined strictly by inverse inner-validation MAE

Protocol & Isolation:
- Official MatBench 5-fold partitions
- 85% inner-train / 15% inner-val checkpointing (zero test snooping)
- Single-pass outer-test evaluation
- All outputs sealed in: results/gnn_variations_benchmark_YYYYMMDD_HHMMSS/
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
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, median_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr, wilcoxon, ttest_rel

from pymatgen.core import Element, Structure
from matminer.datasets import load_dataset
from matbench.bench import MatbenchBenchmark

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# RUN INITIALIZATION & DIRECTORIES
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(r"d:\Desktop\Material_science_qml")
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = PROJECT_ROOT / "results" / f"gnn_variations_benchmark_{RUN_TIMESTAMP}"

DIR_CKPTS    = RUN_DIR / "checkpoints"
DIR_PREDICTS = RUN_DIR / "predictions"
DIR_TABLES   = RUN_DIR / "tables"
DIR_FIGURES  = RUN_DIR / "figures"

for d in [DIR_CKPTS, DIR_PREDICTS, DIR_TABLES, DIR_FIGURES]:
    d.mkdir(parents=True, exist_ok=True)

LOG_FILE = RUN_DIR / "run_log.txt"

def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    formatted = f"[{ts}] {msg}"
    try:
        print(formatted)
        sys.stdout.flush()
    except Exception:
        pass
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(formatted + "\n")
    except Exception:
        pass

# Set CPU threading
device = torch.device("cpu")
n_threads = os.cpu_count() or 4
torch.set_num_threads(n_threads)

# ─────────────────────────────────────────────────────────────────────────────
# 1. ELEMENTAL PHYSICAL PRIORS & DISTANCE EXPANSION
# ─────────────────────────────────────────────────────────────────────────────

def build_elemental_prior_lookup() -> np.ndarray:
    """
    Builds an 8-dimensional normalized elemental property lookup table for Z in [0, 100].
    Properties:
    0: Pauling Electronegativity (X)
    1: Atomic Radius (angstrom)
    2: Atomic Mass (amu)
    3: Periodic Table Group
    4: Periodic Table Period (row)
    5: First Ionization Energy (eV)
    6: Molar Volume (cm^3/mol)
    7: Maximum Oxidation State
    """
    matrix = np.zeros((101, 8), dtype=np.float32)
    for z in range(1, 101):
        el = Element.from_Z(z)
        try: x = float(el.X) if (el.X is not None and not np.isnan(el.X)) else 1.5
        except Exception: x = 1.5
        try: r = float(el.atomic_radius) if el.atomic_radius is not None else 1.2
        except Exception: r = 1.2
        try: m = float(el.atomic_mass) if el.atomic_mass is not None else 50.0
        except Exception: m = 50.0
        try: g = float(el.group) if el.group is not None else 9.0
        except Exception: g = 9.0
        try: row = float(el.row) if el.row is not None else 4.0
        except Exception: row = 4.0
        try: ie = float(el.ionization_energy) if (el.ionization_energy is not None and not np.isnan(el.ionization_energy)) else 8.0
        except Exception: ie = 8.0
        try: mv = float(el.molar_volume) if (el.molar_volume is not None and not np.isnan(el.molar_volume)) else 15.0
        except Exception: mv = 15.0
        try: ox = float(el.max_oxidation_state) if el.max_oxidation_state is not None else 2.0
        except Exception: ox = 2.0
        matrix[z] = [x, r, m, g, row, ie, mv, ox]

    mean = matrix[1:].mean(axis=0)
    std = matrix[1:].std(axis=0) + 1e-6
    lookup = (matrix - mean) / std
    lookup[0] = 0.0  # Zero for padding index
    return lookup

ELEM_LOOKUP = build_elemental_prior_lookup()

class GaussianDistance:
    """Expands interatomic distances into Gaussian radial basis functions."""
    def __init__(self, dmin=0.0, dmax=6.0, num_bins=41, var=None):
        self.centers = np.linspace(dmin, dmax, num_bins)
        self.var = var if var is not None else (dmax - dmin) / (num_bins - 1)
        self.dim = len(self.centers)

    def expand(self, distances: np.ndarray) -> np.ndarray:
        return np.exp(-((distances[:, None] - self.centers[None, :]) ** 2) / (self.var ** 2))

# ─────────────────────────────────────────────────────────────────────────────
# 2. CRYSTAL GRAPH PRECOMPUTATION & DATASET
# ─────────────────────────────────────────────────────────────────────────────

def build_structure_graph(structure: Structure, target: float, radius: float, max_nbr: int, gdf: GaussianDistance):
    """Constructs graph with node priors, RBF+inv-dist edges, and macroscopic scalars."""
    atomic_numbers = [site.specie.number for site in structure]
    atom_priors = ELEM_LOOKUP[atomic_numbers]
    all_nbrs = structure.get_all_neighbors(radius, include_index=True)

    edge_src, edge_dst, edge_dist = [], [], []
    for i, nbrs in enumerate(all_nbrs):
        if len(nbrs) > 0:
            nbrs_sorted = sorted(nbrs, key=lambda x: x.nn_distance)[:max_nbr]
            for nbr in nbrs_sorted:
                edge_src.append(i)
                edge_dst.append(nbr.index)
                edge_dist.append(float(nbr.nn_distance))

    if len(edge_dist) == 0:
        edge_src = [0]
        edge_dst = [0]
        edge_dist = [1.0]

    edge_dist_arr = np.array(edge_dist, dtype=np.float32)
    rbf_attr = gdf.expand(edge_dist_arr).astype(np.float32)
    inv_dist = (1.0 / (edge_dist_arr + 1e-2))[:, None]
    edge_attr = np.hstack([rbf_attr, inv_dist]).astype(np.float32)

    # Macroscopic crystal features: density, vpa, packing_fraction, n_elements
    density = float(structure.density)
    vpa = float(structure.volume / structure.num_sites)
    spheres_vol = 0.0
    for site in structure:
        el = site.specie
        r = float(el.atomic_radius) if el.atomic_radius is not None else 1.2
        spheres_vol += (4.0 / 3.0) * np.pi * (r ** 3)
    packing = float(spheres_vol / structure.volume)
    n_elements = float(len(structure.composition.elements))
    global_raw = np.array([density, vpa, packing, n_elements], dtype=np.float32)

    y_val = float(target)
    y_log = float(np.log(max(y_val - 0.99, 1e-4)))

    return {
        "atom_types": np.array(atomic_numbers, dtype=np.int64),
        "atom_priors": atom_priors.astype(np.float32),
        "edge_src": np.array(edge_src, dtype=np.int64),
        "edge_dst": np.array(edge_dst, dtype=np.int64),
        "edge_attr": edge_attr,
        "edge_attr_baseline": rbf_attr,  # 41-dim pure RBF for baseline CGCNN
        "global_raw": global_raw,
        "target": y_val,
        "target_log": y_log,
        "n_atoms": len(atomic_numbers)
    }

class CrystalDataset(Dataset):
    def __init__(self, graphs: list, global_scaler: StandardScaler = None):
        self.graphs = graphs
        if global_scaler is not None:
            raw_mat = np.array([g["global_raw"] for g in graphs], dtype=np.float32)
            self.global_feats = global_scaler.transform(raw_mat).astype(np.float32)
        else:
            self.global_feats = np.array([g["global_raw"] for g in graphs], dtype=np.float32)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        g = self.graphs[idx]
        return {
            **g,
            "global_feat": self.global_feats[idx]
        }

def collate_graphs(batch):
    batch_atom_types, batch_atom_priors = [], []
    batch_edge_src, batch_edge_dst, batch_edge_attr, batch_edge_attr_base = [], [], [], []
    batch_global_feat, batch_crystal_idx = [], []
    batch_targets, batch_targets_log = [], []

    atom_offset = 0
    for c_idx, g in enumerate(batch):
        n_a = g["n_atoms"]
        batch_atom_types.append(torch.tensor(g["atom_types"], dtype=torch.long))
        batch_atom_priors.append(torch.tensor(g["atom_priors"], dtype=torch.float32))
        batch_edge_src.append(torch.tensor(g["edge_src"] + atom_offset, dtype=torch.long))
        batch_edge_dst.append(torch.tensor(g["edge_dst"] + atom_offset, dtype=torch.long))
        batch_edge_attr.append(torch.tensor(g["edge_attr"], dtype=torch.float32))
        batch_edge_attr_base.append(torch.tensor(g["edge_attr_baseline"], dtype=torch.float32))
        batch_global_feat.append(torch.tensor(g["global_feat"], dtype=torch.float32))
        batch_crystal_idx.append(torch.full((n_a,), c_idx, dtype=torch.long))
        batch_targets.append(g["target"])
        batch_targets_log.append(g["target_log"])
        atom_offset += n_a

    return {
        "atom_types": torch.cat(batch_atom_types, dim=0),
        "atom_priors": torch.cat(batch_atom_priors, dim=0),
        "edge_src": torch.cat(batch_edge_src, dim=0),
        "edge_dst": torch.cat(batch_edge_dst, dim=0),
        "edge_attr": torch.cat(batch_edge_attr, dim=0),
        "edge_attr_base": torch.cat(batch_edge_attr_base, dim=0),
        "global_feat": torch.stack(batch_global_feat, dim=0),
        "crystal_idx": torch.cat(batch_crystal_idx, dim=0),
        "targets": torch.tensor(batch_targets, dtype=torch.float32).unsqueeze(1),
        "targets_log": torch.tensor(batch_targets_log, dtype=torch.float32).unsqueeze(1),
        "n_crystals": len(batch)
    }

# ─────────────────────────────────────────────────────────────────────────────
# 3. GNN ARCHITECTURES
# ─────────────────────────────────────────────────────────────────────────────

# --- Model 1: Baseline CGCNN (Audited Baseline Standard) ---
class CGCNNConv(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int):
        super().__init__()
        self.fc_full = nn.Linear(2 * node_dim + edge_dim, 2 * node_dim)
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()
        self.layer_norm = nn.LayerNorm(node_dim)

    def forward(self, x, edge_src, edge_dst, edge_attr):
        vi = x[edge_src]
        vj = x[edge_dst]
        z = torch.cat([vi, vj, edge_attr], dim=-1)
        z_cf = self.fc_full(z)
        z_core, z_gate = z_cf.chunk(2, dim=-1)
        msg = self.softplus(z_core) * self.sigmoid(z_gate)

        out = torch.zeros_like(x)
        out.index_add_(0, edge_src, msg)
        out = self.layer_norm(out)
        return self.softplus(x + out)

class CGCNN_Baseline(nn.Module):
    def __init__(self, orig_atom_fea_len=101, edge_fea_len=41, atom_fea_len=64, n_conv=4, h_fea_len=64):
        super().__init__()
        self.embedding = nn.Embedding(orig_atom_fea_len, atom_fea_len)
        self.convs = nn.ModuleList([CGCNNConv(atom_fea_len, edge_fea_len) for _ in range(n_conv)])
        self.fc1 = nn.Linear(atom_fea_len, h_fea_len)
        self.softplus = nn.Softplus()
        self.fc_out = nn.Linear(h_fea_len, 1)

    def forward(self, batch):
        x = self.embedding(batch["atom_types"])
        for conv in self.convs:
            x = conv(x, batch["edge_src"], batch["edge_dst"], batch["edge_attr_base"])

        n_crystals = batch["n_crystals"]
        crystal_idx = batch["crystal_idx"]
        pooled = torch.zeros((n_crystals, x.size(-1)), device=x.device)
        counts = torch.zeros((n_crystals, 1), device=x.device)
        ones = torch.ones((x.size(0), 1), device=x.device)
        pooled.index_add_(0, crystal_idx, x)
        counts.index_add_(0, crystal_idx, ones)
        crys_feat = pooled / counts.clamp(min=1.0)

        h = self.softplus(self.fc1(crys_feat))
        return self.fc_out(h)

# --- Model 2: Physics-Enriched Attentive Crystal GNN (PE-Attn-CGNN) ---
class AttentiveConvLayer(nn.Module):
    def __init__(self, node_dim: int):
        super().__init__()
        self.fc_msg = nn.Linear(node_dim * 3, node_dim)
        self.fc_attn = nn.Linear(node_dim * 3, 1)
        self.fc_node = nn.Sequential(
            nn.Linear(node_dim * 2, node_dim),
            nn.LayerNorm(node_dim),
            nn.SiLU()
        )

    def forward(self, x, e, edge_src, edge_dst):
        vi = x[edge_src]
        vj = x[edge_dst]
        pair = torch.cat([vi, vj, e], dim=-1)
        msg = F.silu(self.fc_msg(pair))

        score = F.leaky_relu(self.fc_attn(pair), 0.2).squeeze(-1)
        exp_s = torch.exp(score - score.max())
        denom = torch.zeros(x.size(0), device=x.device)
        denom.index_add_(0, edge_src, exp_s)
        alpha = (exp_s / (denom[edge_src] + 1e-8)).unsqueeze(-1)

        agg = torch.zeros_like(x)
        agg.index_add_(0, edge_src, alpha * msg)
        out = self.fc_node(torch.cat([x, agg], dim=-1))
        return x + out

class PE_Attn_CGNN(nn.Module):
    def __init__(self, n_atom_prior=8, edge_dim=42, node_dim=64, n_conv=3, output_mode="bounded"):
        super().__init__()
        self.embedding = nn.Embedding(101, 48)
        self.node_proj = nn.Sequential(nn.Linear(48 + n_atom_prior, node_dim), nn.LayerNorm(node_dim), nn.SiLU())
        self.edge_proj = nn.Sequential(nn.Linear(edge_dim, node_dim), nn.LayerNorm(node_dim), nn.SiLU())
        self.convs = nn.ModuleList([AttentiveConvLayer(node_dim) for _ in range(n_conv)])
        self.output_mode = output_mode

        self.readout = nn.Sequential(
            nn.Linear(node_dim * 2, 64),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

    def forward(self, batch):
        emb = self.embedding(batch["atom_types"])
        x = self.node_proj(torch.cat([emb, batch["atom_priors"]], dim=-1))
        e = self.edge_proj(batch["edge_attr"])

        for conv in self.convs:
            x = conv(x, e, batch["edge_src"], batch["edge_dst"])

        n_crystals = batch["n_crystals"]
        crystal_idx = batch["crystal_idx"]

        mean_pool = torch.zeros((n_crystals, x.size(-1)), device=x.device)
        counts = torch.zeros((n_crystals, 1), device=x.device)
        ones = torch.ones((x.size(0), 1), device=x.device)
        mean_pool.index_add_(0, crystal_idx, x)
        counts.index_add_(0, crystal_idx, ones)
        mean_pool = mean_pool / counts.clamp(min=1.0)

        # Max pooling
        max_pool = torch.zeros((n_crystals, x.size(-1)), device=x.device)
        for c in range(n_crystals):
            mask = (crystal_idx == c)
            if mask.any():
                max_pool[c] = x[mask].max(dim=0)[0]

        crys_feat = torch.cat([mean_pool, max_pool], dim=-1)
        raw_out = self.readout(crys_feat)

        if self.output_mode == "bounded":
            return 1.0 + F.softplus(raw_out)
        elif self.output_mode == "log_space":
            return raw_out  # Predicts z = log(n - 0.99)
        return raw_out

# --- Model 3: Global-Conditioned Dual-Path GNN (Global-CGNN / MEGNet) ---
class GlobalConvLayer(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, global_dim: int):
        super().__init__()
        self.fc_e = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim + global_dim, edge_dim),
            nn.LayerNorm(edge_dim),
            nn.SiLU()
        )
        self.fc_v = nn.Sequential(
            nn.Linear(node_dim + edge_dim + global_dim, node_dim),
            nn.LayerNorm(node_dim),
            nn.SiLU()
        )
        self.fc_u = nn.Sequential(
            nn.Linear(global_dim + node_dim + edge_dim, global_dim),
            nn.LayerNorm(global_dim),
            nn.SiLU()
        )

    def forward(self, x, e, u, edge_src, edge_dst, crystal_idx, n_crystals, node_counts):
        edge_crys = crystal_idx[edge_src]
        u_edges = u[edge_crys]
        u_nodes = u[crystal_idx]

        # 1. Edge update
        vi = x[edge_src]
        vj = x[edge_dst]
        e_in = torch.cat([vi, vj, e, u_edges], dim=-1)
        e_new = e + self.fc_e(e_in)

        # 2. Node update
        e_agg = torch.zeros_like(x)
        e_agg.index_add_(0, edge_src, e_new)
        v_in = torch.cat([x, e_agg, u_nodes], dim=-1)
        x_new = x + self.fc_v(v_in)

        # 3. Global update
        v_mean = torch.zeros((n_crystals, x.size(-1)), device=x.device)
        v_mean.index_add_(0, crystal_idx, x_new)
        v_mean = v_mean / node_counts.clamp(min=1.0)

        e_mean = torch.zeros((n_crystals, e.size(-1)), device=e.device)
        e_counts = torch.zeros((n_crystals, 1), device=e.device)
        ones_e = torch.ones((e.size(0), 1), device=e.device)
        e_mean.index_add_(0, edge_crys, e_new)
        e_counts.index_add_(0, edge_crys, ones_e)
        e_mean = e_mean / e_counts.clamp(min=1.0)

        u_in = torch.cat([u, v_mean, e_mean], dim=-1)
        u_new = u + self.fc_u(u_in)

        return x_new, e_new, u_new

class Global_CGNN(nn.Module):
    def __init__(self, n_atom_prior=8, edge_dim=42, n_global=4, hidden_dim=64, n_conv=3):
        super().__init__()
        self.embedding = nn.Embedding(101, 48)
        self.node_proj = nn.Sequential(nn.Linear(48 + n_atom_prior, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU())
        self.edge_proj = nn.Sequential(nn.Linear(edge_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU())
        self.global_proj = nn.Sequential(nn.Linear(n_global, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU())

        self.convs = nn.ModuleList([GlobalConvLayer(hidden_dim, hidden_dim, hidden_dim) for _ in range(n_conv)])
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

    def forward(self, batch):
        emb = self.embedding(batch["atom_types"])
        x = self.node_proj(torch.cat([emb, batch["atom_priors"]], dim=-1))
        e = self.edge_proj(batch["edge_attr"])
        u = self.global_proj(batch["global_feat"])

        n_crystals = batch["n_crystals"]
        crystal_idx = batch["crystal_idx"]

        counts = torch.zeros((n_crystals, 1), device=x.device)
        ones = torch.ones((x.size(0), 1), device=x.device)
        counts.index_add_(0, crystal_idx, ones)

        for conv in self.convs:
            x, e, u = conv(x, e, u, batch["edge_src"], batch["edge_dst"], crystal_idx, n_crystals, counts)

        v_mean = torch.zeros((n_crystals, x.size(-1)), device=x.device)
        v_mean.index_add_(0, crystal_idx, x)
        v_mean = v_mean / counts.clamp(min=1.0)

        crys_feat = torch.cat([v_mean, u], dim=-1)
        raw_out = self.readout(crys_feat)
        return 1.0 + F.softplus(raw_out)

# ─────────────────────────────────────────────────────────────────────────────
# 4. METRIC COMPUTATION HELPER
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    ae = np.abs(y_true - y_pred)
    se = (y_true - y_pred) ** 2
    r_val = pearsonr(y_true, y_pred)[0] if len(np.unique(y_pred)) > 1 else np.nan
    rho_val = spearmanr(y_true, y_pred)[0] if len(np.unique(y_pred)) > 1 else np.nan

    return {
        "MAE": float(np.mean(ae)),
        "MedAE": float(np.median(ae)),
        "RMSE": float(np.sqrt(np.mean(se))),
        "R2": float(r2_score(y_true, y_pred)),
        "Pearson_r": float(r_val),
        "Spearman_rho": float(rho_val),
        "Mean_Signed_Error": float(np.mean(y_pred - y_true)),
        "P90_AE": float(np.percentile(ae, 90)),
        "P95_AE": float(np.percentile(ae, 95)),
        "Max_AE": float(np.max(ae)),
        "Fraction_n_less_1": float(np.mean(y_pred < 1.0))
    }

# ─────────────────────────────────────────────────────────────────────────────
# 5. MAIN TRAINING ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def main():
    log("=" * 80)
    log("ADVANCED CRYSTAL GNN VARIATIONS — LEAKAGE-FREE FIVE-FOLD BENCHMARK")
    log("=" * 80)
    log(f"Isolated run directory: {RUN_DIR}")
    log(f"Device: {device} ({n_threads} worker threads)")

    # 1. Load official MatBench benchmark
    log("\nLoading official MatbenchBenchmark('matbench_v0.1', subset=['matbench_dielectric']) ...")
    bm = MatbenchBenchmark(benchmark="matbench_v0.1", subset=["matbench_dielectric"])
    task = list(bm.tasks)[0]
    task.load()

    df_full = task.df
    total_samples = len(df_full)
    y_raw_full = df_full["n"].values
    log(f"Loaded {total_samples} materials from official benchmark.")

    # 2. Precompute crystal graphs (R=6.0 A, 42 RBF+inv bins)
    log("\nPrecomputing crystal graphs with R_cut=6.0 A, k=12, and elemental physics priors ...")
    t0_graph = time.time()
    gdf6 = GaussianDistance(dmin=0.0, dmax=6.0, num_bins=41)
    all_graphs = [
        build_structure_graph(
            df_full["structure"].iloc[i],
            float(y_raw_full[i]),
            radius=6.0,
            max_nbr=12,
            gdf=gdf6
        )
        for i in range(total_samples)
    ]
    log(f"Precomputed and cached {len(all_graphs)} graphs in {time.time() - t0_graph:.2f}s.")

    # 3. Model specifications
    models_to_evaluate = [
        "CGCNN_Baseline",
        "PE_Attn_CGNN",
        "Global_CGNN",
        "Log_CGNN"
    ]

    all_fold_results = []
    all_test_predictions = []
    training_curves = {m: {f: [] for f in task.folds} for m in models_to_evaluate}
    val_maes_per_fold = {f: {} for f in task.folds}

    # Outer 5-Fold Evaluation Loop
    for fold_idx in task.folds:
        log("\n" + "#" * 60)
        log(f"### EXECUTING OFFICIAL FOLD {fold_idx} / 4")
        log("#" * 60)

        X_tr_df, _ = task.get_train_and_val_data(fold_idx)
        X_te_df    = task.get_test_data(fold_idx, include_target=False)

        tr_idx = [int(x.split("-")[-1]) - 1 for x in X_tr_df.index]
        te_idx = [int(x.split("-")[-1]) - 1 for x in X_te_df.index]
        y_te = y_raw_full[te_idx]
        formulas_te = [df_full["structure"].iloc[i].composition.reduced_formula for i in te_idx]
        ids_te      = [f"mb-dielectric-{i+1:04d}" for i in te_idx]

        # Inner-validation partition: 85% inner train, 15% inner val
        np.random.seed(42 + fold_idx)
        n_outer_train = len(tr_idx)
        perm = np.random.permutation(n_outer_train)
        n_inner_val = int(0.15 * n_outer_train)
        val_sub_idx = [tr_idx[k] for k in perm[:n_inner_val]]
        tr_sub_idx  = [tr_idx[k] for k in perm[n_inner_val:]]

        # Strict isolation: fit global scaler solely on inner-training samples
        inner_tr_global_raw = np.array([all_graphs[i]["global_raw"] for i in tr_sub_idx])
        global_scaler = StandardScaler()
        global_scaler.fit(inner_tr_global_raw)

        ds_inner_tr  = CrystalDataset([all_graphs[i] for i in tr_sub_idx], global_scaler=global_scaler)
        ds_inner_val = CrystalDataset([all_graphs[i] for i in val_sub_idx], global_scaler=global_scaler)
        ds_outer_te  = CrystalDataset([all_graphs[i] for i in te_idx], global_scaler=global_scaler)

        loader_tr  = DataLoader(ds_inner_tr,  batch_size=64, shuffle=True,  collate_fn=collate_graphs)
        loader_val = DataLoader(ds_inner_val, batch_size=64, shuffle=False, collate_fn=collate_graphs)
        loader_te  = DataLoader(ds_outer_te,  batch_size=64, shuffle=False, collate_fn=collate_graphs)

        fold_model_preds = {}
        fold_fit_times = {}

        for model_name in models_to_evaluate:
            log(f"\n--- Training {model_name} (Fold {fold_idx}) ---")
            torch.manual_seed(42 + fold_idx)

            if model_name == "CGCNN_Baseline":
                model = CGCNN_Baseline(orig_atom_fea_len=101, edge_fea_len=41, atom_fea_len=64, n_conv=4, h_fea_len=64)
                criterion = nn.L1Loss()
                lr = 2e-3
            elif model_name == "PE_Attn_CGNN":
                model = PE_Attn_CGNN(n_atom_prior=8, edge_dim=42, node_dim=64, n_conv=3, output_mode="bounded")
                criterion = nn.SmoothL1Loss(beta=0.2)
                lr = 1.5e-3
            elif model_name == "Global_CGNN":
                model = Global_CGNN(n_atom_prior=8, edge_dim=42, n_global=4, hidden_dim=64, n_conv=3)
                criterion = nn.SmoothL1Loss(beta=0.2)
                lr = 1.5e-3
            elif model_name == "Log_CGNN":
                model = PE_Attn_CGNN(n_atom_prior=8, edge_dim=42, node_dim=64, n_conv=3, output_mode="log_space")
                criterion = nn.SmoothL1Loss(beta=0.15)
                lr = 1.5e-3
            else:
                raise ValueError(f"Unknown model: {model_name}")

            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=1e-5)

            best_val_mae = float("inf")
            best_state = None
            best_epoch = 0
            t0_fit = time.time()

            for epoch in range(1, 26):
                model.train()
                train_loss_accum = 0.0
                n_batches = 0
                for batch in loader_tr:
                    optimizer.zero_grad()
                    out = model(batch)
                    if model_name == "Log_CGNN":
                        target = batch["targets_log"]
                    else:
                        target = batch["targets"]
                    loss = criterion(out, target)
                    loss.backward()
                    optimizer.step()
                    train_loss_accum += loss.item()
                    n_batches += 1

                scheduler.step()

                # Inner Validation Checkpoint Selection (Measured in original n space)
                model.eval()
                val_preds_list, val_true_list = [], []
                with torch.no_grad():
                    for batch in loader_val:
                        p = model(batch)
                        if model_name == "Log_CGNN":
                            # Invert z -> n = 0.99 + exp(z)
                            p_actual = 0.99 + torch.exp(p)
                        else:
                            p_actual = p
                        val_preds_list.extend(p_actual.view(-1).cpu().numpy())
                        val_true_list.extend(batch["targets"].view(-1).cpu().numpy())

                val_mae = float(mean_absolute_error(val_true_list, val_preds_list))
                training_curves[model_name][fold_idx].append(val_mae)

                if val_mae < best_val_mae:
                    best_val_mae = val_mae
                    best_epoch = epoch
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            fit_time = time.time() - t0_fit
            log(f"  Best Val MAE: {best_val_mae:.4f} at epoch {best_epoch} (Fit time: {fit_time:.1f}s)")
            val_maes_per_fold[fold_idx][model_name] = best_val_mae

            # Save best checkpoint
            ckpt_path = DIR_CKPTS / f"{model_name}_fold_{fold_idx}.pt"
            torch.save(best_state, ckpt_path)

            # Locked Single-Pass Held-Out Test Evaluation
            model.load_state_dict(best_state)
            model.eval()
            test_preds_list = []
            with torch.no_grad():
                for batch in loader_te:
                    p = model(batch)
                    if model_name == "Log_CGNN":
                        p_actual = 0.99 + torch.exp(p)
                    else:
                        p_actual = p
                    test_preds_list.extend(p_actual.view(-1).cpu().numpy())

            test_preds_arr = np.array(test_preds_list, dtype=np.float32)
            fold_model_preds[model_name] = test_preds_arr

            fold_fit_times[model_name] = fit_time

            # Metrics
            m = compute_metrics(y_te, test_preds_arr)
            all_fold_results.append({
                "fold": fold_idx,
                "model": model_name,
                "val_mae": best_val_mae,
                "best_epoch": best_epoch,
                "fit_time_s": round(fit_time, 2),
                **m
            })
            log(f"  Outer Test -> MAE: {m['MAE']:.4f} | MedAE: {m['MedAE']:.4f} | RMSE: {m['RMSE']:.4f} | R2: {m['R2']:.4f} | rho: {m['Spearman_rho']:.4f}")

            # Record predictions
            for i in range(len(test_preds_arr)):
                all_test_predictions.append({
                    "fold": fold_idx,
                    "model": model_name,
                    "identifier": ids_te[i],
                    "formula": formulas_te[i],
                    "y_true": float(y_te[i]),
                    "y_pred": float(test_preds_arr[i]),
                    "abs_error": float(abs(test_preds_arr[i] - y_te[i]))
                })

        # --- Variation 4: Ensemble-GNN (Weighted blend of top models) ---
        log(f"\n--- Constructing Ensemble-GNN (Fold {fold_idx}) ---")
        blend_models = ["PE_Attn_CGNN", "Global_CGNN", "Log_CGNN"]
        # Inverse validation MAE weighting
        inv_maes = [1.0 / val_maes_per_fold[fold_idx][m] for m in blend_models]
        weights = [w / sum(inv_maes) for w in inv_maes]
        for bm_name, w in zip(blend_models, weights):
            log(f"  Model {bm_name} weight: {w:.4f} (Val MAE: {val_maes_per_fold[fold_idx][bm_name]:.4f})")

        ens_test_preds = sum(w * fold_model_preds[bm_name] for w, bm_name in zip(weights, blend_models))
        ens_metrics = compute_metrics(y_te, ens_test_preds)
        all_fold_results.append({
            "fold": fold_idx,
            "model": "Ensemble_GNN",
            "val_mae": sum(w * val_maes_per_fold[fold_idx][bm_name] for w, bm_name in zip(weights, blend_models)),
            "best_epoch": 25,
            "fit_time_s": round(sum(fold_fit_times[m] for m in blend_models), 2),
            **ens_metrics
        })
        log(f"  Ensemble_GNN Test -> MAE: {ens_metrics['MAE']:.4f} | MedAE: {ens_metrics['MedAE']:.4f} | RMSE: {ens_metrics['RMSE']:.4f} | R2: {ens_metrics['R2']:.4f}")

        for i in range(len(ens_test_preds)):
            all_test_predictions.append({
                "fold": fold_idx,
                "model": "Ensemble_GNN",
                "identifier": ids_te[i],
                "formula": formulas_te[i],
                "y_true": float(y_te[i]),
                "y_pred": float(ens_test_preds[i]),
                "abs_error": float(abs(ens_test_preds[i] - y_te[i]))
            })

    # 4. Save results and prediction CSVs
    results_df = pd.DataFrame(all_fold_results)
    results_csv_path = DIR_TABLES / "advanced_gnns_5fold_results.csv"
    results_df.to_csv(results_csv_path, index=False)

    preds_df = pd.DataFrame(all_test_predictions)
    preds_csv_path = DIR_PREDICTS / "advanced_gnns_all_predictions.csv"
    preds_df.to_csv(preds_csv_path, index=False)
    log(f"\nSaved all fold results to {results_csv_path}")
    log(f"Saved all predictions to {preds_csv_path}")

    # 5. Summary Statistics Table
    summary_rows = []
    for model_name, grp in results_df.groupby("model", sort=False):
        summary_rows.append({
            "Model Architecture": model_name,
            "MAE (Mean +/- SD)": f"{grp['MAE'].mean():.4f} +/- {grp['MAE'].std():.4f}",
            "MedAE (Mean +/- SD)": f"{grp['MedAE'].mean():.4f} +/- {grp['MedAE'].std():.4f}",
            "RMSE (Mean +/- SD)": f"{grp['RMSE'].mean():.4f} +/- {grp['RMSE'].std():.4f}",
            "R2 (Mean +/- SD)": f"{grp['R2'].mean():.4f} +/- {grp['R2'].std():.4f}",
            "Spearman rho (Mean +/- SD)": f"{grp['Spearman_rho'].mean():.4f} +/- {grp['Spearman_rho'].std():.4f}",
            "Fraction n < 1.0": f"{grp['Fraction_n_less_1'].mean()*100:.1f}%",
            "Mean Fit Time": f"{grp['fit_time_s'].mean():.1f}s"
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_md_path = DIR_TABLES / "advanced_gnns_5fold_summary.md"
    summary_df.to_markdown(summary_md_path, index=False)
    log("\n" + "=" * 80)
    log("FIVE-FOLD BENCHMARK SUMMARY TABLE:")
    log("=" * 80)
    try:
        log(summary_df.to_string(index=False))
    except Exception:
        pass

    # 6. Binned Tail Performance Analysis
    log("\nAnalyzing binned error across refractive index regimes ...")
    bins = [0.0, 2.0, 3.0, 5.0, 7.0, 100.0]
    bin_labels = ["n <= 2.0", "2.0 < n <= 3.0", "3.0 < n <= 5.0", "5.0 < n <= 7.0", "n > 7.0"]

    preds_df["target_bin"] = pd.cut(preds_df["y_true"], bins=bins, labels=bin_labels)

    binned_records = []
    for b in bin_labels:
        sub = preds_df[preds_df["target_bin"] == b]
        n_samples = len(sub[sub["model"] == "CGCNN_Baseline"])
        row = {"Refractive Index Regime": b, "Sample Count": n_samples}
        for m in models_to_evaluate + ["Ensemble_GNN"]:
            m_sub = sub[sub["model"] == m]
            if len(m_sub) > 0:
                row[f"{m} MAE"] = round(float(m_sub["abs_error"].mean()), 4)
                row[f"{m} MedAE"] = round(float(m_sub["abs_error"].median()), 4)
        binned_records.append(row)

    binned_df = pd.DataFrame(binned_records)
    binned_csv_path = DIR_TABLES / "binned_tail_mae_comparison.csv"
    binned_df.to_csv(binned_csv_path, index=False)
    binned_md_path = DIR_TABLES / "binned_tail_mae_comparison.md"
    binned_df.to_markdown(binned_md_path, index=False)
    log("\nBinned Performance Breakdown:")
    try:
        log(binned_df.to_string(index=False))
    except Exception:
        pass

    # 7. Statistical Significance Tests vs CGCNN Baseline
    log("\nComputing Wilcoxon signed-rank and paired t-tests vs CGCNN Baseline ...")
    stat_records = []
    base_preds = preds_df[preds_df["model"] == "CGCNN_Baseline"].sort_values(["fold", "identifier"])["abs_error"].values

    for m in [m for m in models_to_evaluate + ["Ensemble_GNN"] if m != "CGCNN_Baseline"]:
        m_preds = preds_df[preds_df["model"] == m].sort_values(["fold", "identifier"])["abs_error"].values
        diff = base_preds - m_preds
        w_stat, w_p = wilcoxon(base_preds, m_preds, alternative="greater")
        t_stat, t_p = ttest_rel(base_preds, m_preds)
        stat_records.append({
            "Comparison vs CGCNN_Baseline": f"{m} vs CGCNN_Baseline",
            "Mean Absolute Error Reduction": f"{np.mean(diff):.4f}",
            "Median Absolute Error Reduction": f"{np.median(diff):.4f}",
            "Wilcoxon W": float(w_stat),
            "Wilcoxon p-value": f"{w_p:.2e}",
            "Paired t-statistic": float(t_stat),
            "Paired t-test p-value": f"{t_p:.2e}",
            "Statistically Significant (p < 0.05)": "YES" if w_p < 0.05 else "NO"
        })

    stat_df = pd.DataFrame(stat_records)
    stat_csv_path = DIR_TABLES / "statistical_significance_tests.csv"
    stat_df.to_csv(stat_csv_path, index=False)
    stat_md_path = DIR_TABLES / "statistical_significance_tests.md"
    stat_df.to_markdown(stat_md_path, index=False)
    log("\nStatistical Significance Tests:")
    try:
        log(stat_df.to_string(index=False))
    except Exception:
        pass

    # 8. Publication Figures Generation
    log("\nGenerating publication figures ...")
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Figure 1: 4-Panel Pooled Out-of-Fold Parity Plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=300)
    plot_models = ["CGCNN_Baseline", "PE_Attn_CGNN", "Global_CGNN", "Ensemble_GNN"]
    titles = [
        "CGCNN Baseline (Audited Protocol)",
        "PE-Attn-CGNN (Physics Priors + Attention)",
        "Global-CGNN (Dual-Path Macroscopic State)",
        "Ensemble-GNN (Weighted Multi-Architecture Blend)"
    ]

    for ax, m_name, title in zip(axes.flatten(), plot_models, titles):
        sub = preds_df[preds_df["model"] == m_name]
        y_t = sub["y_true"].values
        y_p = sub["y_pred"].values
        mae = mean_absolute_error(y_t, y_p)
        r2 = r2_score(y_t, y_p)
        medae = median_absolute_error(y_t, y_p)

        hb = ax.hexbin(y_t, y_p, gridsize=50, cmap="viridis", mincnt=1, bins="log", extent=[1, 10, 1, 10])
        ax.plot([1, 10], [1, 10], "r--", lw=1.8, label="Ideal Parity y = x")
        ax.set_xlim(1.0, 8.5)
        ax.set_ylim(1.0, 8.5)
        ax.set_xlabel("DFT Refractive Index $n$ (Ground Truth)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Predicted Refractive Index $\\hat{n}$", fontsize=11, fontweight="bold")
        ax.set_title(f"{title}\nMAE = {mae:.4f} | MedAE = {medae:.4f} | $R^2$ = {r2:.4f}", fontsize=11, fontweight="bold", pad=8)
        ax.legend(loc="upper left", frameon=True)
        cb = fig.colorbar(hb, ax=ax)
        cb.set_label("Log10(Count)", fontsize=9)

    plt.suptitle("Out-of-Fold Parity Benchmark Across Advanced Crystal GNN Architectures (N = 4,764)", fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout()
    fig1_path = DIR_FIGURES / "oof_parity_plots_all_models.png"
    plt.savefig(fig1_path)
    plt.close()
    log(f"  Saved parity plot to {fig1_path}")

    # Figure 2: Error across Refractive Index Regimes
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    x = np.arange(len(bin_labels))
    width = 0.18

    colors = ["#94a3b8", "#38bdf8", "#34d399", "#f43f5e"]
    for idx, (m_name, color) in enumerate(zip(plot_models, colors)):
        maes = [binned_df.loc[binned_df["Refractive Index Regime"] == b, f"{m_name} MAE"].values[0] for b in bin_labels]
        ax.bar(x + idx * width, maes, width, label=m_name, color=color, edgecolor="black", linewidth=0.5)

    ax.set_xlabel("Refractive Index Regime", fontsize=12, fontweight="bold")
    ax.set_ylabel("Mean Absolute Error (MAE)", fontsize=12, fontweight="bold")
    ax.set_title("Heteroskedasticity and High-Index Tail Error Across GNN Architectures", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(bin_labels, fontsize=10)
    ax.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    fig2_path = DIR_FIGURES / "error_vs_refractive_index.png"
    plt.savefig(fig2_path)
    plt.close()
    log(f"  Saved tail error plot to {fig2_path}")

    # Figure 3: Model Comparison Bar Chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    all_models_ordered = ["CGCNN_Baseline", "PE_Attn_CGNN", "Global_CGNN", "Log_CGNN", "Ensemble_GNN"]

    mean_maes = [results_df[results_df["model"] == m]["MAE"].mean() for m in all_models_ordered]
    std_maes  = [results_df[results_df["model"] == m]["MAE"].std()  for m in all_models_ordered]
    mean_meds = [results_df[results_df["model"] == m]["MedAE"].mean() for m in all_models_ordered]
    std_meds  = [results_df[results_df["model"] == m]["MedAE"].std()  for m in all_models_ordered]

    x_m = np.arange(len(all_models_ordered))
    w = 0.35
    ax1.bar(x_m - w/2, mean_maes, w, yerr=std_maes, capsize=4, label="Mean MAE", color="#60a5fa", edgecolor="black")
    ax1.bar(x_m + w/2, mean_meds, w, yerr=std_meds, capsize=4, label="Median AE", color="#34d399", edgecolor="black")
    ax1.set_xticks(x_m)
    ax1.set_xticklabels(all_models_ordered, rotation=15, ha="right", fontsize=9, fontweight="bold")
    ax1.set_ylabel("Error Metric", fontsize=11, fontweight="bold")
    ax1.set_title("5-Fold Cross-Validation Error Metrics", fontsize=12, fontweight="bold")
    ax1.legend(frameon=True)

    mean_r2 = [results_df[results_df["model"] == m]["R2"].mean() for m in all_models_ordered]
    std_r2  = [results_df[results_df["model"] == m]["R2"].std()  for m in all_models_ordered]
    ax2.bar(x_m, mean_r2, 0.5, yerr=std_r2, capsize=4, color="#a78bfa", edgecolor="black")
    ax2.set_xticks(x_m)
    ax2.set_xticklabels(all_models_ordered, rotation=15, ha="right", fontsize=9, fontweight="bold")
    ax2.set_ylabel("Coefficient of Determination $R^2$", fontsize=11, fontweight="bold")
    ax2.set_title("5-Fold Cross-Validation Explained Variance ($R^2$)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig3_path = DIR_FIGURES / "model_comparison_bar_chart.png"
    plt.savefig(fig3_path)
    plt.close()
    log(f"  Saved comparison chart to {fig3_path}")

    # Figure 4: Validation Learning Curves
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    epochs_range = list(range(1, 26))
    for m_name, color in zip(models_to_evaluate, ["#94a3b8", "#38bdf8", "#34d399", "#f59e0b"]):
        curves = np.array([training_curves[m_name][f] for f in task.folds])
        mean_c = curves.mean(axis=0)
        std_c  = curves.std(axis=0)
        ax.plot(epochs_range, mean_c, label=m_name, color=color, lw=2)
        ax.fill_between(epochs_range, mean_c - std_c, mean_c + std_c, color=color, alpha=0.15)

    ax.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax.set_ylabel("Inner-Validation MAE", fontsize=11, fontweight="bold")
    ax.set_title("5-Fold Inner-Validation Convergence Curves", fontsize=12, fontweight="bold", pad=10)
    ax.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    fig4_path = DIR_FIGURES / "validation_loss_curves.png"
    plt.savefig(fig4_path)
    plt.close()
    log(f"  Saved validation convergence curve to {fig4_path}")

    log("\n" + "=" * 80)
    log("ADVANCED GNN BENCHMARK RUN COMPLETED SUCCESSFULLY!")
    log(f"Artifacts and tables sealed in: {RUN_DIR}")
    log("=" * 80)

if __name__ == "__main__":
    main()
