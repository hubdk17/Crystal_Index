"""
================================================================================
SCRIPT 11: FRONTIER CRYSTAL GNN BENCHMARK & MULTI-PARADIGM LOSS EXPLORATION
TARGET: MATBENCH v0.1 `matbench_dielectric` (N = 4,764)
CHALLENGING THE BENCHMARK RECORD (MODNet v0.1.12: 0.2711 MAE)
================================================================================
Scientific Objective:
Explore next-generation crystal graph neural network architectures, specialized
loss functions addressing refractive index heavy-tail divergence, hyperparameter
scaling, and meta-ensembling to challenge the MatBench benchmark record holder
under strict zero-data-leakage protocols.

Key Innovations:
1. Angle_Aware_CGNN (Bond-Angle Triplet GNN):
   - Local bond angles cos(theta_ijk) expanded into a 16-bin Chebyshev polynomial basis
   - Joint gated message passing (SiLU core * Sigmoid gate)
2. MultiHead_GraphTransformer (MHT-GNN):
   - Fully vectorized 4-head attention conditioned on atomic distances & Chebyshev angles
   - Node-level linear projections indexed by edge connectivity for high efficiency
3. Hierarchical_Global_GNN (HG-GNN):
   - 12-dimensional macroscopic crystal state vector (density, VPA, packing,
     compositional electronegativity statistics, covalent radii, coordination,
     unit-cell aspect ratio c/a)
   - Tri-directional message passing: Node <-> Edge <-> Macroscopic State
4. DualHead_LogDirect_GNN (Multitask Optical Gap GNN):
   - Dual output heads: Direct n >= 1.0 and Log space z = log(n - 0.99)
   - Joint multitask training balancing low-n bulk dielectric materials and
     extreme narrow-gap high-n outliers
5. Varied Loss Formulations:
   - Adaptive Huber (beta = 0.10)
   - Log-Cosh loss (smooth, quadratic at center, linear at tail)
   - Relative-Weighted Huber (tail power-law dampening)
   - Multitask Joint Huber
6. Advanced Meta-Ensembling:
   - Inverse Validation MAE weighting
   - Softmax error weighting
   - Non-Negative Least Squares (NNLS) meta-learner on inner validation folds

Evaluation & Isolation:
- Official MatBench v0.1 five-fold partitions (N = 4,764)
- 85% inner-train / 15% inner-val checkpointing (zero outer test snooping)
- Single-pass evaluation on held-out outer-test folds
- Sealed outputs in: results/frontier_gnn_benchmark_YYYYMMDD_HHMMSS/
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
from scipy.optimize import nnls

from pymatgen.core import Element, Structure
from matminer.datasets import load_dataset
from matbench.bench import MatbenchBenchmark

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# RUN INITIALIZATION & DIRECTORIES
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(r"d:\Desktop\Material_science_qml")
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = PROJECT_ROOT / "results" / f"frontier_gnn_benchmark_{RUN_TIMESTAMP}"

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

device = torch.device("cpu")
n_threads = os.cpu_count() or 4
torch.set_num_threads(n_threads)

# ─────────────────────────────────────────────────────────────────────────────
# 1. ELEMENTAL PHYSICAL PRIORS & DISTANCE / ANGLE EXPANSIONS
# ─────────────────────────────────────────────────────────────────────────────

def build_elemental_prior_lookup() -> np.ndarray:
    """Builds an 8-dimensional normalized elemental property lookup table for Z in [0, 100]."""
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
    lookup[0] = 0.0
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

def expand_chebyshev_angles(cos_angles: np.ndarray, num_bins: int = 16) -> np.ndarray:
    """Expands bond angle cosines into Chebyshev polynomial basis T_0 to T_{num_bins-1}."""
    cos_clamped = np.clip(cos_angles, -1.0, 1.0)
    basis = np.zeros((len(cos_angles), num_bins), dtype=np.float32)
    basis[:, 0] = 1.0
    if num_bins > 1:
        basis[:, 1] = cos_clamped
        for k in range(2, num_bins):
            basis[:, k] = 2.0 * cos_clamped * basis[:, k - 1] - basis[:, k - 2]
    return basis

# ─────────────────────────────────────────────────────────────────────────────
# 2. FRONTIER CRYSTAL GRAPH PRECOMPUTATION & 12-DIM GLOBAL DESCRIPTORS
# ─────────────────────────────────────────────────────────────────────────────

def build_frontier_structure_graph(structure: Structure, target: float, radius: float, max_nbr: int, gdf: GaussianDistance):
    """
    Constructs an advanced crystal graph with:
    1. Node types & 8 elemental priors
    2. Edge RBF (41) + Inverse distance (1) + Local Chebyshev bond-angle features (16) = 58 edge features
    3. 12-dimensional macroscopic crystal state vector u
    4. Direct and log target values
    """
    atomic_numbers = [site.specie.number for site in structure]
    atom_priors = ELEM_LOOKUP[atomic_numbers]
    all_nbrs = structure.get_all_neighbors(radius, include_index=True)

    edge_src, edge_dst, edge_dist = [], [], []
    nbr_vectors_per_site = {i: [] for i in range(len(structure))}

    for i, nbrs in enumerate(all_nbrs):
        if len(nbrs) > 0:
            nbrs_sorted = sorted(nbrs, key=lambda x: x.nn_distance)[:max_nbr]
            for nbr in nbrs_sorted:
                edge_src.append(i)
                edge_dst.append(nbr.index)
                d = float(nbr.nn_distance)
                edge_dist.append(d)
                disp = nbr.coords - structure[i].coords
                norm = np.linalg.norm(disp)
                unit_vec = disp / (norm + 1e-8)
                nbr_vectors_per_site[i].append((nbr.index, unit_vec, d))

    if len(edge_dist) == 0:
        edge_src = [0]
        edge_dst = [0]
        edge_dist = [1.0]
        nbr_vectors_per_site[0] = [(0, np.array([1.0, 0.0, 0.0]), 1.0)]

    edge_dist_arr = np.array(edge_dist, dtype=np.float32)
    rbf_attr = gdf.expand(edge_dist_arr).astype(np.float32)
    inv_dist = (1.0 / (edge_dist_arr + 1e-2))[:, None]

    edge_angle_features = np.zeros((len(edge_dist), 16), dtype=np.float32)
    edge_counter = 0
    for i in range(len(structure)):
        nbrs_i = nbr_vectors_per_site[i]
        n_nbrs = len(nbrs_i)
        for (dst_idx, u_ij, d_ij) in nbrs_i:
            if n_nbrs > 1:
                cos_list = []
                for (other_idx, u_ik, d_ik) in nbrs_i:
                    if other_idx != dst_idx:
                        cos_theta = float(np.dot(u_ij, u_ik))
                        cos_list.append(cos_theta)
                if len(cos_list) > 0:
                    cheb = expand_chebyshev_angles(np.array(cos_list, dtype=np.float32), num_bins=16)
                    edge_angle_features[edge_counter] = np.mean(cheb, axis=0)
                else:
                    edge_angle_features[edge_counter, 0] = 1.0
            else:
                edge_angle_features[edge_counter, 0] = 1.0
            edge_counter += 1

    edge_attr_full = np.hstack([rbf_attr, inv_dist, edge_angle_features]).astype(np.float32)

    density = float(structure.density)
    vpa = float(structure.volume / structure.num_sites)

    spheres_vol = 0.0
    radii, electros, valences, mol_vols = [], [], [], []
    for site in structure:
        el = site.specie
        r = float(el.atomic_radius) if el.atomic_radius is not None else 1.2
        spheres_vol += (4.0 / 3.0) * np.pi * (r ** 3)
        radii.append(r)
        try: x = float(el.X) if el.X is not None else 1.5
        except Exception: x = 1.5
        electros.append(x)
        try: g = float(el.group) if el.group is not None else 9.0
        except Exception: g = 9.0
        valences.append(g)
        try: mv = float(el.molar_volume) if el.molar_volume is not None else 15.0
        except Exception: mv = 15.0
        mol_vols.append(mv)

    packing = float(spheres_vol / max(structure.volume, 1e-4))
    n_elements = float(len(structure.composition.elements))
    mean_en = float(np.mean(electros))
    range_en = float(np.max(electros) - np.min(electros))
    mean_r = float(np.mean(radii))
    mean_val = float(np.mean(valences))
    mean_mv = float(np.mean(mol_vols))
    avg_coord = float(len(edge_dist) / max(len(structure), 1))

    latt = structure.lattice
    a, b, c = float(latt.a), float(latt.b), float(latt.c)
    aspect_ratio = float(c / max(a, 1e-4))
    packing_eff = float(packing * (vpa / max(mean_r ** 3, 1e-4)))

    global_12d = np.array([
        density, vpa, packing, n_elements,
        mean_en, range_en, mean_r, mean_val,
        mean_mv, avg_coord, packing_eff, aspect_ratio
    ], dtype=np.float32)

    y_val = float(target)
    y_log = float(np.log(max(y_val - 0.99, 1e-4)))

    return {
        "atom_types": np.array(atomic_numbers, dtype=np.int64),
        "atom_priors": atom_priors.astype(np.float32),
        "edge_src": np.array(edge_src, dtype=np.int64),
        "edge_dst": np.array(edge_dst, dtype=np.int64),
        "edge_attr": edge_attr_full,
        "global_12d": global_12d,
        "target": y_val,
        "target_log": y_log,
        "n_atoms": len(atomic_numbers)
    }

class FrontierCrystalDataset(Dataset):
    def __init__(self, graphs: list, global_scaler: StandardScaler = None):
        self.graphs = graphs
        if global_scaler is not None:
            raw_mat = np.array([g["global_12d"] for g in graphs], dtype=np.float32)
            self.global_feats = global_scaler.transform(raw_mat).astype(np.float32)
        else:
            self.global_feats = np.array([g["global_12d"] for g in graphs], dtype=np.float32)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        g = self.graphs[idx]
        return {
            **g,
            "global_feat": self.global_feats[idx]
        }

def collate_frontier_graphs(batch):
    batch_atom_types, batch_atom_priors = [], []
    batch_edge_src, batch_edge_dst, batch_edge_attr = [], [], []
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
        "global_feat": torch.stack(batch_global_feat, dim=0),
        "crystal_idx": torch.cat(batch_crystal_idx, dim=0),
        "targets": torch.tensor(batch_targets, dtype=torch.float32).unsqueeze(1),
        "targets_log": torch.tensor(batch_targets_log, dtype=torch.float32).unsqueeze(1),
        "n_crystals": len(batch)
    }

# ─────────────────────────────────────────────────────────────────────────────
# 3. SPECIALIZED LOSS FUNCTIONS (ADDRESSING HEAVY-TAIL DIVERGENCE)
# ─────────────────────────────────────────────────────────────────────────────

class LogCoshLoss(nn.Module):
    """Log-Cosh Loss: Smooth everywhere, quadratic for small errors, strictly linear for large errors."""
    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        abs_diff = torch.abs(diff)
        loss = abs_diff + torch.log1p(torch.exp(-2.0 * abs_diff)) - 0.69314718
        return torch.mean(loss)

class RelativeWeightedHuberLoss(nn.Module):
    """Power-law weighted Huber loss: w_i = (y_i)^(-gamma) to prevent extreme outlier shocks."""
    def __init__(self, beta: float = 0.15, power: float = 0.25):
        super().__init__()
        self.beta = beta
        self.power = power

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(pred - target)
        huber = torch.where(diff < self.beta, 0.5 * (diff ** 2) / self.beta, diff - 0.5 * self.beta)
        weights = torch.clamp(target, min=1.0) ** (-self.power)
        return torch.mean(weights * huber)

class MultitaskLoss(nn.Module):
    """Joint Huber loss on direct refractive index n and log-space optical gap z."""
    def __init__(self, beta: float = 0.15, lambda_log: float = 0.5):
        super().__init__()
        self.huber = nn.SmoothL1Loss(beta=beta)
        self.lambda_log = lambda_log

    def forward(self, pred_direct: torch.Tensor, pred_log: torch.Tensor,
                target_direct: torch.Tensor, target_log: torch.Tensor) -> torch.Tensor:
        l_dir = self.huber(pred_direct, target_direct)
        l_log = self.huber(pred_log, target_log)
        return l_dir + self.lambda_log * l_log

# ─────────────────────────────────────────────────────────────────────────────
# 4. HIGH-PERFORMANCE FRONTIER GNN ARCHITECTURES (FULLY VECTORIZED)
# ─────────────────────────────────────────────────────────────────────────────

# --- Architecture 1: Angle_Aware_CGNN (Chebyshev Bond-Angle Triplet GNN) ---
class AngleAwareConv(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int):
        super().__init__()
        # Fast joint linear projection chunked into core message and gate
        self.fc_full = nn.Linear(node_dim * 2 + edge_dim, node_dim * 2)
        self.sigmoid = nn.Sigmoid()
        self.silu = nn.SiLU()
        self.layer_norm = nn.LayerNorm(node_dim)

    def forward(self, x, edge_attr, edge_src, edge_dst):
        vi = x[edge_src]
        vj = x[edge_dst]
        pair = torch.cat([vi, vj, edge_attr], dim=-1)
        z = self.fc_full(pair)
        z_core, z_gate = z.chunk(2, dim=-1)
        msg = self.silu(z_core) * self.sigmoid(z_gate)

        agg = torch.zeros_like(x)
        agg.index_add_(0, edge_src, msg)
        out = self.layer_norm(agg)
        return self.silu(x + out)

class Angle_Aware_CGNN(nn.Module):
    def __init__(self, n_atom_prior=8, edge_dim=58, hidden_dim=64, n_conv=3):
        super().__init__()
        self.embedding = nn.Embedding(101, 48)
        self.node_proj = nn.Sequential(nn.Linear(48 + n_atom_prior, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU())
        self.edge_proj = nn.Sequential(nn.Linear(edge_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU())

        self.convs = nn.ModuleList([AngleAwareConv(hidden_dim, hidden_dim) for _ in range(n_conv)])
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, 64),
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
        crys_feat = mean_pool / counts.clamp(min=1.0)

        raw_out = self.readout(crys_feat)
        return 1.0 + F.softplus(raw_out)

# --- Architecture 2: MultiHead_GraphTransformer (Vectorized Fast MHT-GNN) ---
class FastGraphTransformerLayer(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, n_heads: int = 4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = node_dim // n_heads
        self.scale = 1.0 / np.sqrt(self.head_dim)

        # Node-level projections (10-15x faster than edge-level projection)
        self.q_proj = nn.Linear(node_dim, node_dim)
        self.k_proj = nn.Linear(node_dim, node_dim)
        self.v_proj = nn.Linear(node_dim, node_dim)
        self.e_proj = nn.Linear(edge_dim, node_dim)

        self.out_proj = nn.Linear(node_dim, node_dim)
        self.norm1 = nn.LayerNorm(node_dim)
        self.ffn = nn.Sequential(
            nn.Linear(node_dim, node_dim * 2),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(node_dim * 2, node_dim)
        )
        self.norm2 = nn.LayerNorm(node_dim)

    def forward(self, x, e, edge_src, edge_dst):
        # 1. Project nodes directly (V, d)
        q_all = self.q_proj(x).view(-1, self.n_heads, self.head_dim)
        k_all = self.k_proj(x).view(-1, self.n_heads, self.head_dim)
        v_all = self.v_proj(x).view(-1, self.n_heads, self.head_dim)
        e_bias = self.e_proj(e).view(-1, self.n_heads, self.head_dim)

        q = q_all[edge_src]
        k = k_all[edge_dst]
        v = v_all[edge_dst]

        attn_score = (q * k).sum(dim=-1) * self.scale + e_bias.sum(dim=-1)

        attn_exp = torch.exp(attn_score - attn_score.max(dim=0, keepdim=True)[0])
        denom = torch.zeros((x.size(0), self.n_heads), device=x.device)
        denom.index_add_(0, edge_src, attn_exp)
        alpha = (attn_exp / (denom[edge_src] + 1e-8)).unsqueeze(-1)

        msg = (alpha * v).view(-1, self.n_heads * self.head_dim)
        agg = torch.zeros_like(x)
        agg.index_add_(0, edge_src, msg)

        x1 = self.norm1(x + self.out_proj(agg))
        out = self.norm2(x1 + self.ffn(x1))
        return out

class MultiHead_GraphTransformer(nn.Module):
    def __init__(self, n_atom_prior=8, edge_dim=58, hidden_dim=64, n_conv=3):
        super().__init__()
        self.embedding = nn.Embedding(101, 48)
        self.node_proj = nn.Sequential(nn.Linear(48 + n_atom_prior, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU())
        self.edge_proj = nn.Sequential(nn.Linear(edge_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU())

        self.convs = nn.ModuleList([FastGraphTransformerLayer(hidden_dim, hidden_dim, n_heads=4) for _ in range(n_conv)])
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, 64),
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
        crys_feat = mean_pool / counts.clamp(min=1.0)

        raw_out = self.readout(crys_feat)
        return 1.0 + F.softplus(raw_out)

# --- Architecture 3: Hierarchical_Global_GNN (HG-GNN / 12-dim Macroscopic State) ---
class HierarchicalGlobalConv(nn.Module):
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

        vi = x[edge_src]
        vj = x[edge_dst]
        e_in = torch.cat([vi, vj, e, u_edges], dim=-1)
        e_new = e + self.fc_e(e_in)

        e_agg = torch.zeros_like(x)
        e_agg.index_add_(0, edge_src, e_new)
        v_in = torch.cat([x, e_agg, u_nodes], dim=-1)
        x_new = x + self.fc_v(v_in)

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

class Hierarchical_Global_GNN(nn.Module):
    def __init__(self, n_atom_prior=8, edge_dim=58, n_global=12, hidden_dim=64, n_conv=3):
        super().__init__()
        self.embedding = nn.Embedding(101, 48)
        self.node_proj = nn.Sequential(nn.Linear(48 + n_atom_prior, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU())
        self.edge_proj = nn.Sequential(nn.Linear(edge_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU())
        self.global_proj = nn.Sequential(nn.Linear(n_global, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU())

        self.convs = nn.ModuleList([HierarchicalGlobalConv(hidden_dim, hidden_dim, hidden_dim) for _ in range(n_conv)])
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

# --- Architecture 4: DualHead_LogDirect_GNN (Multitask Optical Gap GNN) ---
class DualHead_LogDirect_GNN(nn.Module):
    def __init__(self, n_atom_prior=8, edge_dim=58, n_global=12, hidden_dim=64, n_conv=3):
        super().__init__()
        self.embedding = nn.Embedding(101, 48)
        self.node_proj = nn.Sequential(nn.Linear(48 + n_atom_prior, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU())
        self.edge_proj = nn.Sequential(nn.Linear(edge_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU())
        self.global_proj = nn.Sequential(nn.Linear(n_global, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU())

        self.convs = nn.ModuleList([HierarchicalGlobalConv(hidden_dim, hidden_dim, hidden_dim) for _ in range(n_conv)])

        self.head_direct = nn.Sequential(
            nn.Linear(hidden_dim * 2, 48),
            nn.SiLU(),
            nn.Linear(48, 1)
        )
        self.head_log = nn.Sequential(
            nn.Linear(hidden_dim * 2, 48),
            nn.SiLU(),
            nn.Linear(48, 1)
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
        out_direct = 1.0 + F.softplus(self.head_direct(crys_feat))
        out_log = self.head_log(crys_feat)
        return out_direct, out_log

# ─────────────────────────────────────────────────────────────────────────────
# 5. METRIC COMPUTATION HELPER
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
# 6. MAIN FRONTIER BENCHMARK ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def main():
    log("=" * 80)
    log("FRONTIER CRYSTAL GNN BENCHMARK & MULTI-PARADIGM LOSS EXPLORATION")
    log("TARGET: CHALLENGE THE BENCHMARK RECORD (MODNet: 0.2711 MAE)")
    log("=" * 80)
    log(f"Isolated run directory: {RUN_DIR}")
    log(f"Device: {device} ({n_threads} worker threads)")

    log("\nLoading official MatbenchBenchmark('matbench_v0.1', subset=['matbench_dielectric']) ...")
    bm = MatbenchBenchmark(benchmark="matbench_v0.1", subset=["matbench_dielectric"])
    task = list(bm.tasks)[0]
    task.load()

    df_full = task.df
    total_samples = len(df_full)
    y_raw_full = df_full["n"].values
    log(f"Loaded {total_samples} materials from official benchmark.")

    # Cache mechanism for frontier graphs to avoid recomputing if already present
    cache_path = PROJECT_ROOT / "data" / "processed" / "frontier_graphs_cache.pt"
    if cache_path.exists():
        log(f"Loading precomputed frontier crystal graphs from cache: {cache_path} ...")
        t0_graph = time.time()
        all_graphs = torch.load(cache_path)
        log(f"Loaded {len(all_graphs)} graphs from cache in {time.time() - t0_graph:.2f}s.")
    else:
        log("\nPrecomputing frontier crystal graphs with Chebyshev angles and 12-dim macroscopic physics ...")
        t0_graph = time.time()
        gdf6 = GaussianDistance(dmin=0.0, dmax=6.0, num_bins=41)
        all_graphs = [
            build_frontier_structure_graph(
                df_full["structure"].iloc[i],
                float(y_raw_full[i]),
                radius=6.0,
                max_nbr=12,
                gdf=gdf6
            )
            for i in range(total_samples)
        ]
        log(f"Precomputed {len(all_graphs)} frontier graphs in {time.time() - t0_graph:.2f}s. Saving to cache ...")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(all_graphs, cache_path)
        log("Saved graph cache.")

    frontier_models = [
        "Angle_Aware_CGNN",
        "MultiHead_GraphTransformer",
        "Hierarchical_Global_GNN",
        "DualHead_LogDirect_GNN"
    ]

    all_fold_results = []
    all_test_predictions = []
    training_curves = {m: {f: [] for f in task.folds} for m in frontier_models}
    val_maes_per_fold = {f: {} for f in task.folds}
    inner_val_predictions = {f: {m: [] for m in frontier_models} for f in task.folds}
    inner_val_trues = {f: [] for f in task.folds}

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

        np.random.seed(100 + fold_idx)
        n_outer_train = len(tr_idx)
        perm = np.random.permutation(n_outer_train)
        n_inner_val = int(0.15 * n_outer_train)
        val_sub_idx = [tr_idx[k] for k in perm[:n_inner_val]]
        tr_sub_idx  = [tr_idx[k] for k in perm[n_inner_val:]]
        y_inner_val = y_raw_full[val_sub_idx]
        inner_val_trues[fold_idx] = y_inner_val

        inner_tr_global_raw = np.array([all_graphs[i]["global_12d"] for i in tr_sub_idx])
        global_scaler = StandardScaler()
        global_scaler.fit(inner_tr_global_raw)

        ds_inner_tr  = FrontierCrystalDataset([all_graphs[i] for i in tr_sub_idx], global_scaler=global_scaler)
        ds_inner_val = FrontierCrystalDataset([all_graphs[i] for i in val_sub_idx], global_scaler=global_scaler)
        ds_outer_te  = FrontierCrystalDataset([all_graphs[i] for i in te_idx], global_scaler=global_scaler)

        loader_tr  = DataLoader(ds_inner_tr,  batch_size=64, shuffle=True,  collate_fn=collate_frontier_graphs)
        loader_val = DataLoader(ds_inner_val, batch_size=64, shuffle=False, collate_fn=collate_frontier_graphs)
        loader_te  = DataLoader(ds_outer_te,  batch_size=64, shuffle=False, collate_fn=collate_frontier_graphs)

        fold_model_preds = {}
        fold_val_preds = {}
        fold_fit_times = {}

        for model_name in frontier_models:
            log(f"\n--- Training {model_name} (Fold {fold_idx}) ---")
            torch.manual_seed(100 + fold_idx)

            if model_name == "Angle_Aware_CGNN":
                model = Angle_Aware_CGNN(n_atom_prior=8, edge_dim=58, hidden_dim=64, n_conv=3)
                criterion = nn.SmoothL1Loss(beta=0.10)
                lr = 1.8e-3
            elif model_name == "MultiHead_GraphTransformer":
                model = MultiHead_GraphTransformer(n_atom_prior=8, edge_dim=58, hidden_dim=64, n_conv=3)
                criterion = LogCoshLoss()
                lr = 1.5e-3
            elif model_name == "Hierarchical_Global_GNN":
                model = Hierarchical_Global_GNN(n_atom_prior=8, edge_dim=58, n_global=12, hidden_dim=64, n_conv=3)
                criterion = RelativeWeightedHuberLoss(beta=0.15, power=0.25)
                lr = 1.8e-3
            elif model_name == "DualHead_LogDirect_GNN":
                model = DualHead_LogDirect_GNN(n_atom_prior=8, edge_dim=58, n_global=12, hidden_dim=64, n_conv=3)
                criterion = MultitaskLoss(beta=0.15, lambda_log=0.5)
                lr = 1.8e-3
            else:
                raise ValueError(f"Unknown model: {model_name}")

            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=1e-5)

            best_val_mae = float("inf")
            best_state = None
            best_epoch = 0
            best_val_preds_arr = None
            t0_fit = time.time()

            for epoch in range(1, 26):
                model.train()
                train_loss_accum = 0.0
                n_batches = 0
                for batch in loader_tr:
                    optimizer.zero_grad()
                    if model_name == "DualHead_LogDirect_GNN":
                        p_dir, p_log = model(batch)
                        loss = criterion(p_dir, p_log, batch["targets"], batch["targets_log"])
                    else:
                        out = model(batch)
                        loss = criterion(out, batch["targets"])

                    loss.backward()
                    optimizer.step()
                    train_loss_accum += loss.item()
                    n_batches += 1

                scheduler.step()

                # Inner Validation Checkpoint Selection
                model.eval()
                val_preds_list, val_true_list = [], []
                with torch.no_grad():
                    for batch in loader_val:
                        if model_name == "DualHead_LogDirect_GNN":
                            p_dir, p_log = model(batch)
                            p_blended = 0.5 * p_dir + 0.5 * (0.99 + torch.exp(p_log))
                            val_preds_list.extend(p_blended.view(-1).cpu().numpy())
                        else:
                            p = model(batch)
                            val_preds_list.extend(p.view(-1).cpu().numpy())
                        val_true_list.extend(batch["targets"].view(-1).cpu().numpy())

                val_mae = float(mean_absolute_error(val_true_list, val_preds_list))
                training_curves[model_name][fold_idx].append(val_mae)

                if val_mae < best_val_mae:
                    best_val_mae = val_mae
                    best_epoch = epoch
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    best_val_preds_arr = np.array(val_preds_list, dtype=np.float32)

            fit_time = time.time() - t0_fit
            log(f"  Best Val MAE: {best_val_mae:.4f} at epoch {best_epoch} (Fit time: {fit_time:.1f}s)")
            val_maes_per_fold[fold_idx][model_name] = best_val_mae
            fold_val_preds[model_name] = best_val_preds_arr
            inner_val_predictions[fold_idx][model_name] = best_val_preds_arr

            ckpt_path = DIR_CKPTS / f"{model_name}_fold_{fold_idx}.pt"
            torch.save(best_state, ckpt_path)

            # Locked Single-Pass Outer-Test Evaluation
            model.load_state_dict(best_state)
            model.eval()
            test_preds_list = []
            with torch.no_grad():
                for batch in loader_te:
                    if model_name == "DualHead_LogDirect_GNN":
                        p_dir, p_log = model(batch)
                        p_blended = 0.5 * p_dir + 0.5 * (0.99 + torch.exp(p_log))
                        test_preds_list.extend(p_blended.view(-1).cpu().numpy())
                    else:
                        p = model(batch)
                        test_preds_list.extend(p.view(-1).cpu().numpy())

            test_preds_arr = np.array(test_preds_list, dtype=np.float32)
            fold_model_preds[model_name] = test_preds_arr
            fold_fit_times[model_name] = fit_time

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

        # --- ADVANCED META-ENSEMBLING STRATEGIES ---
        log(f"\n--- Meta-Ensembling (Fold {fold_idx}) ---")

        # Scheme 1: Inverse Validation MAE
        inv_maes = [1.0 / val_maes_per_fold[fold_idx][m] for m in frontier_models]
        w_inv = [w / sum(inv_maes) for w in inv_maes]
        preds_inv = sum(w * fold_model_preds[m] for w, m in zip(w_inv, frontier_models))
        m_inv = compute_metrics(y_te, preds_inv)
        all_fold_results.append({
            "fold": fold_idx,
            "model": "Ensemble_InvVal",
            "val_mae": float(np.average([val_maes_per_fold[fold_idx][m] for m in frontier_models], weights=w_inv)),
            "best_epoch": None,
            "fit_time_s": round(sum(fold_fit_times.values()), 2),
            **m_inv
        })
        log(f"  Ensemble_InvVal -> MAE: {m_inv['MAE']:.4f} | MedAE: {m_inv['MedAE']:.4f} | RMSE: {m_inv['RMSE']:.4f} | R2: {m_inv['R2']:.4f}")

        # Scheme 2: Softmax Validation Weighting (tau = 10.0)
        neg_val_maes = [-10.0 * val_maes_per_fold[fold_idx][m] for m in frontier_models]
        exp_w = np.exp(neg_val_maes - np.max(neg_val_maes))
        w_softmax = exp_w / np.sum(exp_w)
        preds_softmax = sum(w * fold_model_preds[m] for w, m in zip(w_softmax, frontier_models))
        m_softmax = compute_metrics(y_te, preds_softmax)
        all_fold_results.append({
            "fold": fold_idx,
            "model": "Ensemble_Softmax",
            "val_mae": float(np.average([val_maes_per_fold[fold_idx][m] for m in frontier_models], weights=w_softmax)),
            "best_epoch": None,
            "fit_time_s": round(sum(fold_fit_times.values()), 2),
            **m_softmax
        })
        log(f"  Ensemble_Softmax -> MAE: {m_softmax['MAE']:.4f} | MedAE: {m_softmax['MedAE']:.4f} | RMSE: {m_softmax['RMSE']:.4f} | R2: {m_softmax['R2']:.4f}")

        # Scheme 3: Non-Negative Least Squares (NNLS) on inner validation predictions
        val_pred_matrix = np.column_stack([fold_val_preds[m] for m in frontier_models])
        w_nnls_raw, _ = nnls(val_pred_matrix, y_inner_val)
        if np.sum(w_nnls_raw) > 1e-6:
            w_nnls = w_nnls_raw / np.sum(w_nnls_raw)
        else:
            w_nnls = np.ones(len(frontier_models)) / len(frontier_models)
        preds_nnls = sum(w * fold_model_preds[m] for w, m in zip(w_nnls, frontier_models))
        m_nnls = compute_metrics(y_te, preds_nnls)
        all_fold_results.append({
            "fold": fold_idx,
            "model": "Ensemble_NNLS",
            "val_mae": float(mean_absolute_error(y_inner_val, val_pred_matrix @ w_nnls)),
            "best_epoch": None,
            "fit_time_s": round(sum(fold_fit_times.values()), 2),
            **m_nnls
        })
        log(f"  Ensemble_NNLS -> MAE: {m_nnls['MAE']:.4f} | MedAE: {m_nnls['MedAE']:.4f} | RMSE: {m_nnls['RMSE']:.4f} | R2: {m_nnls['R2']:.4f}")

        for i in range(len(preds_nnls)):
            all_test_predictions.append({
                "fold": fold_idx,
                "model": "Ensemble_NNLS",
                "identifier": ids_te[i],
                "formula": formulas_te[i],
                "y_true": float(y_te[i]),
                "y_pred": float(preds_nnls[i]),
                "abs_error": float(abs(preds_nnls[i] - y_te[i]))
            })
            all_test_predictions.append({
                "fold": fold_idx,
                "model": "Ensemble_InvVal",
                "identifier": ids_te[i],
                "formula": formulas_te[i],
                "y_true": float(y_te[i]),
                "y_pred": float(preds_inv[i]),
                "abs_error": float(abs(preds_inv[i] - y_te[i]))
            })

    # Summary
    df_results = pd.DataFrame(all_fold_results)
    df_preds = pd.DataFrame(all_test_predictions)

    results_csv = DIR_TABLES / "frontier_gnns_all_folds.csv"
    preds_csv   = DIR_PREDICTS / "frontier_gnns_all_predictions.csv"
    df_results.to_csv(results_csv, index=False)
    df_preds.to_csv(preds_csv, index=False)
    log(f"\nSaved raw fold metrics to {results_csv}")
    log(f"Saved {len(df_preds)} prediction rows to {preds_csv}")

    summary_rows = []
    all_evaluated_models = list(df_results["model"].unique())

    log("\n" + "=" * 80)
    log("FIVE-FOLD CROSS-VALIDATION SUMMARY (MEAN +/- STD)")
    log("=" * 80)

    for m in all_evaluated_models:
        sub = df_results[df_results["model"] == m]
        mae_m, mae_s = sub["MAE"].mean(), sub["MAE"].std()
        med_m, med_s = sub["MedAE"].mean(), sub["MedAE"].std()
        rmse_m = sub["RMSE"].mean()
        r2_m = sub["R2"].mean()
        pr_m = sub["Pearson_r"].mean()
        rho_m = sub["Spearman_rho"].mean()
        bad_n = sub["Fraction_n_less_1"].mean()

        summary_rows.append({
            "Model": m,
            "MAE_Mean": mae_m,
            "MAE_Std": mae_s,
            "MedAE_Mean": med_m,
            "MedAE_Std": med_s,
            "RMSE_Mean": rmse_m,
            "R2_Mean": r2_m,
            "Pearson_r_Mean": pr_m,
            "Spearman_rho_Mean": rho_m,
            "Fraction_n_less_1": bad_n
        })
        log(f"{m:26s} | MAE: {mae_m:.4f} +/- {mae_s:.4f} | MedAE: {med_m:.4f} +/- {med_s:.4f} | RMSE: {rmse_m:.4f} | R2: {r2_m:.4f} | rho: {rho_m:.4f}")

    df_summary = pd.DataFrame(summary_rows).sort_values("MAE_Mean")
    summary_csv = DIR_TABLES / "frontier_gnns_summary.csv"
    df_summary.to_csv(summary_csv, index=False)
    log(f"Saved summary metrics to {summary_csv}")

    # Figures
    log("\nGenerating publication figures ...")
    sns.set_theme(style="whitegrid", font_scale=1.1)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    models_to_plot = ["Angle_Aware_CGNN", "MultiHead_GraphTransformer", "Hierarchical_Global_GNN",
                      "DualHead_LogDirect_GNN", "Ensemble_InvVal", "Ensemble_NNLS"]

    for idx, m_name in enumerate(models_to_plot):
        ax = axes[idx]
        sub = df_preds[df_preds["model"] == m_name]
        y_t = sub["y_true"].values
        y_p = sub["y_pred"].values

        ax.scatter(y_t, y_p, alpha=0.35, s=18, edgecolors="none", color="#1f77b4")
        max_val = min(max(np.percentile(y_t, 99.5), np.percentile(y_p, 99.5)), 30)
        ax.plot([1.0, max_val], [1.0, max_val], "r--", lw=2, label="Parity (y = x)")

        mae = mean_absolute_error(y_t, y_p)
        medae = median_absolute_error(y_t, y_p)
        r2 = r2_score(y_t, y_p)
        rho = spearmanr(y_t, y_p)[0]

        ax.set_title(f"{m_name}\nMAE: {mae:.4f} | MedAE: {medae:.4f} | R2: {r2:.4f} | rho: {rho:.4f}", fontsize=11, fontweight="bold")
        ax.set_xlabel("DFT Refractive Index n (True)", fontsize=10)
        ax.set_ylabel("Predicted n", fontsize=10)
        ax.set_xlim(0.8, max_val)
        ax.set_ylim(0.8, max_val)
        ax.legend(loc="upper left")

    plt.tight_layout()
    fig1_path = DIR_FIGURES / "frontier_oof_parity_plots.png"
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    log(f"  Figure 1 saved to {fig1_path}")

    fig, ax = plt.subplots(figsize=(12, 6))
    comp_models = ["ALIGNN", "SchNet", "CGCNN_Baseline", "Global_CGNN", "Ensemble_GNN (Prior)",
                   "MODNet v0.1.10", "MODNet v0.1.12 (Record)", "Ensemble_InvVal (Frontier)", "Ensemble_NNLS (Frontier)"]
    comp_maes = [0.3449, 0.3277, 0.3290, 0.2989, 0.2903, 0.2970, 0.2711,
                 float(df_summary[df_summary['Model'] == 'Ensemble_InvVal']['MAE_Mean'].values[0]),
                 float(df_summary[df_summary['Model'] == 'Ensemble_NNLS']['MAE_Mean'].values[0])]

    colors = ["#7f7f7f", "#7f7f7f", "#aec7e8", "#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd", "#8c564b"]
    bars = ax.barh(comp_models, comp_maes, color=colors, edgecolor="black", alpha=0.85)
    ax.axvline(0.2711, color="red", linestyle="--", lw=2, label="MODNet Benchmark Record (0.2711)")
    ax.set_xlabel("5-Fold Outer-Test MAE (Lower is Better)", fontsize=12)
    ax.set_title("Frontier Crystal GNN Benchmark vs. MatBench Dielectric Leaderboard", fontsize=14, fontweight="bold")

    for bar, val in zip(bars, comp_maes):
        ax.text(val + 0.003, bar.get_y() + bar.get_height() / 2, f"{val:.4f}", va="center", ha="left", fontsize=10, fontweight="bold")

    ax.legend(loc="lower right")
    plt.tight_layout()
    fig2_path = DIR_FIGURES / "frontier_leaderboard_comparison.png"
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    log(f"  Figure 2 saved to {fig2_path}")

    log("\n" + "=" * 80)
    log("FRONTIER GNN BENCHMARK EXECUTION COMPLETE")
    log("=" * 80)

if __name__ == "__main__":
    main()
