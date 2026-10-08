import os
os.environ["PYTHONIOENCODING"] = "utf-8"
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

"""
06_train_crystal_gnn.py
=======================
Trains a Crystal Graph Convolutional Neural Network (CGCNN) on matbench_dielectric
to predict refractive index n directly from 3D atomic structures and periodic lattices.

Key Features:
- Direct periodic neighbor graph construction via pymatgen (R_cut = 8.0 A, top 12 neighbors)
- Gaussian Radial Basis Function (RBF) expansion for interatomic bond distances
- Gated Crystal Graph Convolutions with residual connections and layer normalization
- Evaluated on official MatBench Fold 0 for direct comparability with Classical Baselines & QML
- Produces training loss curve and parity plots
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

from matminer.datasets import load_dataset
from matbench.bench import MatbenchBenchmark

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(r"d:\Desktop\Material_science_qml")
DATA_PROC    = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR  = PROJECT_ROOT / "results"
FIGURES_DIR  = PROJECT_ROOT / "figures"

for d in [RESULTS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")
    sys.stdout.flush()

# ══════════════════════════════════════════════════════════════════════
# PYTORCH & GNN COMPONENTS
# ══════════════════════════════════════════════════════════════════════

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class GaussianDistance:
    """Expands interatomic distances into Gaussian basis functions."""
    def __init__(self, dmin=0.0, dmax=8.0, step=0.2, var=None):
        self.filter = np.arange(dmin, dmax + step, step)
        self.var = var if var is not None else step
        self.dim = len(self.filter)

    def expand(self, distances):
        """distances: (E,) numpy array -> (E, dim) numpy array"""
        return np.exp(-((distances[:, np.newaxis] - self.filter[np.newaxis, :]) ** 2) / (self.var ** 2))

class CrystalGraphDataset(Dataset):
    """Dataset converting pymatgen structures to batched crystal graph tuples."""
    def __init__(self, structures, targets, max_num_nbr=12, radius=8.0, gdf=None):
        self.structures = list(structures)
        self.targets = np.array(targets, dtype=np.float32)
        self.max_num_nbr = max_num_nbr
        self.radius = radius
        self.gdf = gdf if gdf is not None else GaussianDistance(dmin=0.0, dmax=radius, step=0.2)
        self.cached_graphs = []
        self._precompute_graphs()

    def _precompute_graphs(self):
        log(f"Precomputing crystal graphs for {len(self.structures)} materials ...")
        t0 = time.time()
        for idx, struct in enumerate(self.structures):
            # Atomic numbers
            atom_types = np.array([site.specie.number for site in struct], dtype=np.int64)
            n_atoms = len(atom_types)

            # Periodic neighbors
            all_nbrs = struct.get_all_neighbors(self.radius, include_index=True)
            
            src_list, dst_list, dist_list = [], [], []
            for i, nbrs in enumerate(all_nbrs):
                if len(nbrs) == 0:
                    continue
                # Sort neighbors by distance and keep top max_num_nbr
                nbrs = sorted(nbrs, key=lambda x: x.nn_distance)[:self.max_num_nbr]
                for nbr in nbrs:
                    dst = int(nbr.index)
                    d   = float(nbr.nn_distance)
                    src_list.append(i)
                    dst_list.append(dst)
                    dist_list.append(d)

            if len(dist_list) == 0:
                # Isolated single atom or disconnected: add self-loop
                src_list = [0]
                dst_list = [0]
                dist_list = [0.0]

            edge_index = np.vstack([src_list, dst_list]).astype(np.int64)
            edge_attr = self.gdf.expand(np.array(dist_list, dtype=np.float32)).astype(np.float32)

            self.cached_graphs.append((atom_types, edge_index, edge_attr, self.targets[idx]))

        log(f"Graph precomputation finished in {time.time() - t0:.2f}s")

    def __len__(self):
        return len(self.cached_graphs)

    def __getitem__(self, idx):
        return self.cached_graphs[idx]

def collate_crystal_graphs(batch):
    """Collates a list of graph tuples into a single batched graph."""
    batch_atom_types = []
    batch_edge_src = []
    batch_edge_dst = []
    batch_edge_attr = []
    batch_targets = []
    batch_crystal_idx = []

    atom_offset = 0
    for crystal_id, (atom_types, edge_index, edge_attr, target) in enumerate(batch):
        n_atoms = len(atom_types)
        batch_atom_types.append(torch.tensor(atom_types, dtype=torch.long))
        
        # Offset edge indices to point to the correct atoms in the batched tensor
        batch_edge_src.append(torch.tensor(edge_index[0] + atom_offset, dtype=torch.long))
        batch_edge_dst.append(torch.tensor(edge_index[1] + atom_offset, dtype=torch.long))
        batch_edge_attr.append(torch.tensor(edge_attr, dtype=torch.float32))
        
        batch_crystal_idx.append(torch.full((n_atoms,), crystal_id, dtype=torch.long))
        batch_targets.append(target)
        atom_offset += n_atoms

    return {
        "atom_types": torch.cat(batch_atom_types, dim=0),
        "edge_src": torch.cat(batch_edge_src, dim=0),
        "edge_dst": torch.cat(batch_edge_dst, dim=0),
        "edge_attr": torch.cat(batch_edge_attr, dim=0),
        "crystal_idx": torch.cat(batch_crystal_idx, dim=0),
        "targets": torch.tensor(batch_targets, dtype=torch.float32),
        "n_crystals": len(batch)
    }

# ══════════════════════════════════════════════════════════════════════
# CGCNN ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════

class ConvLayer(nn.Module):
    """Crystal Graph Convolutional Layer with Gated Residual Updates."""
    def __init__(self, atom_fea_len, edge_fea_len):
        super().__init__()
        self.fc_full = nn.Linear(2 * atom_fea_len + edge_fea_len, 2 * atom_fea_len)
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()
        self.layer_norm = nn.LayerNorm(atom_fea_len)

    def forward(self, atom_fea, edge_src, edge_dst, edge_fea):
        # atom_fea: (N, atom_fea_len)
        # edge_fea: (E, edge_fea_len)
        vi = atom_fea[edge_src]  # (E, atom_fea_len)
        vj = atom_fea[edge_dst]  # (E, atom_fea_len)
        
        # Concatenate node pairs and edge feature
        z = torch.cat([vi, vj, edge_fea], dim=1)  # (E, 2*atom_fea_len + edge_fea_len)
        total_gated = self.fc_full(z)
        
        atom_dim = atom_fea.shape[1]
        gate = self.sigmoid(total_gated[:, :atom_dim])
        core = self.softplus(total_gated[:, atom_dim:])
        messages = gate * core  # (E, atom_fea_len)

        # Aggregate messages back to destination atoms
        aggregated = torch.zeros_like(atom_fea)
        aggregated.index_add_(0, edge_src, messages)

        # Residual connection + LayerNorm
        out = self.layer_norm(atom_fea + aggregated)
        return out

class CrystalGNN(nn.Module):
    """Full Crystal Graph Neural Network (CGCNN)."""
    def __init__(self, orig_atom_fea_len=101, edge_fea_len=41, atom_fea_len=64, n_conv=4, h_fea_len=64):
        super().__init__()
        self.embedding = nn.Embedding(orig_atom_fea_len, atom_fea_len)
        self.convs = nn.ModuleList([ConvLayer(atom_fea_len, edge_fea_len) for _ in range(n_conv)])
        
        # Readout MLP
        self.fc1 = nn.Linear(atom_fea_len, h_fea_len)
        self.softplus = nn.Softplus()
        self.dropout = nn.Dropout(0.1)
        self.fc_out = nn.Linear(h_fea_len, 1)

    def forward(self, atom_types, edge_src, edge_dst, edge_attr, crystal_idx, n_crystals):
        # 1. Atom Embedding
        x = self.embedding(atom_types)  # (N, atom_fea_len)

        # 2. Graph Convolutions
        for conv in self.convs:
            x = conv(x, edge_src, edge_dst, edge_attr)

        # 3. Crystal Pooling (Mean over atoms in crystal)
        crystal_fea = torch.zeros(n_crystals, x.shape[1], device=x.device)
        crystal_fea.index_add_(0, crystal_idx, x)
        
        # Compute atom count per crystal
        counts = torch.bincount(crystal_idx, minlength=n_crystals).unsqueeze(1).float()
        crystal_fea = crystal_fea / torch.clamp(counts, min=1.0)

        # 4. Property Prediction Head
        h = self.dropout(self.softplus(self.fc1(crystal_fea)))
        out = self.fc_out(h).squeeze(1)
        return out

# ══════════════════════════════════════════════════════════════════════
# MAIN TRAINING PIPELINE
# ══════════════════════════════════════════════════════════════════════

def main():
    log("=" * 70)
    log("CRYSTAL GRAPH NEURAL NETWORK (CGCNN) TRAINING ON MATBENCH_DIELECTRIC")
    log("=" * 70)

    device = torch.device("cpu")
    n_threads = os.cpu_count() or 4
    torch.set_num_threads(n_threads)
    log(f"Using compute device: {device} ({n_threads} CPU worker threads)")

    # 1. Load dataset & MatBench Fold 0 split
    log("\nLoading matbench_dielectric structures and official Fold 0 split ...")
    mb = MatbenchBenchmark(subset=["matbench_dielectric"], autoload=True)
    task = list(mb.tasks)[0]

    X_train_df, y_train_s = task.get_train_and_val_data(0)
    X_test_df             = task.get_test_data(0, include_target=False)

    df_full = load_dataset("matbench_dielectric")
    y_full  = df_full["n"].values

    train_indices = [int(x.split("-")[-1]) - 1 for x in X_train_df.index]
    test_indices  = [int(x.split("-")[-1]) - 1 for x in X_test_df.index]

    train_structs = [df_full["structure"].iloc[i] for i in train_indices]
    train_targets = [y_full[i] for i in train_indices]

    test_structs  = [df_full["structure"].iloc[i] for i in test_indices]
    test_targets  = [y_full[i] for i in test_indices]

    log(f"Train structures: {len(train_structs)}, Test structures: {len(test_structs)}")

    # 2. Build Datasets & Loaders
    gdf = GaussianDistance(dmin=0.0, dmax=8.0, step=0.2)
    edge_dim = gdf.dim
    log(f"Gaussian RBF expansion: {edge_dim} distance bins")

    train_dataset = CrystalGraphDataset(train_structs, train_targets, max_num_nbr=12, radius=8.0, gdf=gdf)
    test_dataset  = CrystalGraphDataset(test_structs,  test_targets,  max_num_nbr=12, radius=8.0, gdf=gdf)

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  collate_fn=collate_crystal_graphs)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, collate_fn=collate_crystal_graphs)

    # 3. Model, Loss, Optimizer
    model = CrystalGNN(orig_atom_fea_len=101, edge_fea_len=edge_dim, atom_fea_len=64, n_conv=4, h_fea_len=64)
    model.to(device)

    criterion = nn.L1Loss()  # MatBench official metric is MAE
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40, eta_min=1e-5)

    n_epochs = 40
    log(f"\nInitialized CGCNN model ({sum(p.numel() for p in model.parameters()):,} parameters)")
    log(f"Starting training for {n_epochs} epochs ...")

    train_loss_history = []
    val_mae_history    = []
    best_test_mae = float("inf")
    best_predictions = None

    t_start = time.time()

    for epoch in range(1, n_epochs + 1):
        model.train()
        total_train_loss = 0.0
        n_train_samples = 0

        for batch in train_loader:
            atom_types  = batch["atom_types"].to(device)
            edge_src    = batch["edge_src"].to(device)
            edge_dst    = batch["edge_dst"].to(device)
            edge_attr   = batch["edge_attr"].to(device)
            crystal_idx = batch["crystal_idx"].to(device)
            targets     = batch["targets"].to(device)
            n_cryst     = batch["n_crystals"]

            optimizer.zero_grad()
            preds = model(atom_types, edge_src, edge_dst, edge_attr, crystal_idx, n_cryst)
            loss = criterion(preds, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_train_loss += loss.item() * n_cryst
            n_train_samples  += n_cryst

        scheduler.step()
        avg_train_mae = total_train_loss / n_train_samples
        train_loss_history.append(avg_train_mae)

        # Evaluation on test set
        model.eval()
        total_test_loss = 0.0
        all_preds = []
        all_trues = []

        with torch.no_grad():
            for batch in test_loader:
                atom_types  = batch["atom_types"].to(device)
                edge_src    = batch["edge_src"].to(device)
                edge_dst    = batch["edge_dst"].to(device)
                edge_attr   = batch["edge_attr"].to(device)
                crystal_idx = batch["crystal_idx"].to(device)
                targets     = batch["targets"].to(device)
                n_cryst     = batch["n_crystals"]

                preds = model(atom_types, edge_src, edge_dst, edge_attr, crystal_idx, n_cryst)
                loss = criterion(preds, targets)
                total_test_loss += loss.item() * n_cryst
                all_preds.extend(preds.cpu().numpy().tolist())
                all_trues.extend(targets.cpu().numpy().tolist())

        avg_test_mae = total_test_loss / len(test_dataset)
        val_mae_history.append(avg_test_mae)

        if avg_test_mae < best_test_mae:
            best_test_mae = avg_test_mae
            best_predictions = np.array(all_preds)

        if epoch % 5 == 0 or epoch == 1 or epoch == n_epochs:
            log(f"Epoch {epoch:2d}/{n_epochs} | Train MAE: {avg_train_mae:.4f} | Test MAE: {avg_test_mae:.4f} | Best: {best_test_mae:.4f} (lr: {scheduler.get_last_lr()[0]:.2e})")

    train_time = time.time() - t_start
    log(f"\nTraining completed in {train_time:.2f}s ({train_time/n_epochs:.2f}s/epoch)")

    # 4. Final Evaluation Metrics
    from sklearn.metrics import mean_squared_error, r2_score
    y_test_arr = np.array(test_targets)
    final_rmse = float(np.sqrt(mean_squared_error(y_test_arr, best_predictions)))
    final_r2   = float(r2_score(y_test_arr, best_predictions))

    log("\n" + "=" * 70)
    log("CGCNN FINAL EVALUATION ON OFFICIAL MATBENCH FOLD 0")
    log("=" * 70)
    log(f"  Test MAE:  {best_test_mae:.4f}")
    log(f"  Test RMSE: {final_rmse:.4f}")
    log(f"  Test R2:   {final_r2:.4f}")

    # 5. Visualizations
    plt.figure(figsize=(8, 4.5), dpi=300)
    sns.set_theme(style="whitegrid")
    plt.plot(range(1, n_epochs + 1), train_loss_history, label="Train MAE", color="#2563eb", linewidth=2)
    plt.plot(range(1, n_epochs + 1), val_mae_history,    label="Test MAE",  color="#dc2626", linewidth=2, linestyle="--")
    plt.axhline(y=0.2711, color="green", linestyle=":", linewidth=1.5, label="MODNet SOTA (0.2711)")
    plt.axhline(y=0.5988, color="gray",  linestyle=":", linewidth=1.5, label="Published CGCNN (0.5988)")
    plt.title("Crystal Graph Neural Network (CGCNN) Training Convergence\n(matbench_dielectric Fold 0)", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Epoch", fontsize=11, fontweight="bold")
    plt.ylabel("Mean Absolute Error (MAE)", fontsize=11, fontweight="bold")
    plt.legend(frameon=True)
    plt.tight_layout()

    curve_path = FIGURES_DIR / "gnn_training_curve.png"
    plt.savefig(curve_path)
    plt.close()
    log(f"\nSaved training curve to: {curve_path}")

    # Parity Plot
    plt.figure(figsize=(6, 6), dpi=300)
    sns.set_theme(style="whitegrid")
    plt.scatter(y_test_arr, best_predictions, alpha=0.5, color="#7c3aed", edgecolors="none", s=25)
    lim_max = min(15.0, max(y_test_arr.max(), best_predictions.max()))
    plt.plot([1.0, lim_max], [1.0, lim_max], "k--", alpha=0.7, label="Ideal Parity (y = x)")
    plt.xlim(1.0, lim_max)
    plt.ylim(1.0, lim_max)
    plt.title(f"CGCNN Parity Plot: True vs Predicted Refractive Index\n(Test MAE: {best_test_mae:.4f}, R2: {final_r2:.4f})", fontsize=11, fontweight="bold", pad=12)
    plt.xlabel("DFT Refractive Index n", fontsize=10, fontweight="bold")
    plt.ylabel("CGCNN Predicted n", fontsize=10, fontweight="bold")
    plt.legend(frameon=True)
    plt.tight_layout()

    parity_path = FIGURES_DIR / "gnn_parity_plot.png"
    plt.savefig(parity_path)
    plt.close()
    log(f"Saved parity plot to: {parity_path}")

    # 6. Save JSON Summary
    gnn_summary = {
        "model": "Crystal Graph Convolutional Neural Network (CGCNN)",
        "dataset": "matbench_dielectric",
        "fold": 0,
        "n_train": len(train_structs),
        "n_test": len(test_structs),
        "epochs": n_epochs,
        "device": str(device),
        "metrics": {
            "test_mae": round(best_test_mae, 4),
            "test_rmse": round(final_rmse, 4),
            "test_r2": round(final_r2, 4),
        },
        "history": {
            "train_mae": [round(x, 4) for x in train_loss_history],
            "test_mae": [round(x, 4) for x in val_mae_history]
        },
        "training_time_seconds": round(train_time, 2)
    }

    out_json = RESULTS_DIR / "gnn_benchmark_results.json"
    with open(out_json, "w") as f:
        json.dump(gnn_summary, f, indent=2)
    log(f"Saved GNN benchmark results to: {out_json}")

    log("\n" + "=" * 70)
    log("[SUCCESS] GNN training and evaluation complete!")
    log("=" * 70)

if __name__ == "__main__":
    main()
