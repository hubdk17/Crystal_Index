import os
os.environ["PYTHONIOENCODING"] = "utf-8"
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

"""
05_quantum_kernel_benchmark.py
==============================
Simulates Quantum Machine Learning (QML) kernels on identical materials descriptors:
1. Quantum Fidelity Kernel with Entangled ZZ-Feature Map (Havlíček et al.)
2. Projected Quantum Kernel (PQK, Huang et al. 2021) to prevent exponential concentration

Evaluates:
- Small-Data Sample-Efficiency Learning Curves (N ∈ [50, 100, 200, 400, 800])
- Controlled Head-to-Head Comparison against Classical RBF, Poly, and Ridge
- Official Fold 0 MatBench Test Performance
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

import pennylane as qml
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge

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

# ══════════════════════════════════════════════════════════════════════
# QUANTUM CIRCUITS (PennyLane)
# ══════════════════════════════════════════════════════════════════════

def build_qml_circuits(n_qubits: int):
    """
    Constructs statevector and Pauli expectation quantum circuits
    for an n-qubit entangled ZZ feature map.
    """
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def statevector_circuit(x):
        # 1. Hadamard layer
        for i in range(n_qubits):
            qml.Hadamard(wires=i)
        # 2. Angle embedding
        for i in range(n_qubits):
            qml.RY(x[i], wires=i)
        # 3. Entangling layer with ZZ interactions
        for i in range(n_qubits):
            next_wire = (i + 1) % n_qubits
            qml.CNOT(wires=[i, next_wire])
            phase = 2.0 * (np.pi - x[i]) * (np.pi - x[next_wire])
            qml.RZ(phase, wires=next_wire)
            qml.CNOT(wires=[i, next_wire])
        # 4. Second rotation layer
        for i in range(n_qubits):
            qml.RZ(x[i], wires=i)

        return qml.state()

    @qml.qnode(dev)
    def pqk_circuit(x):
        # Same ansatz as above, but measures 1-qubit Pauli observables
        for i in range(n_qubits):
            qml.Hadamard(wires=i)
        for i in range(n_qubits):
            qml.RY(x[i], wires=i)
        for i in range(n_qubits):
            next_wire = (i + 1) % n_qubits
            qml.CNOT(wires=[i, next_wire])
            phase = 2.0 * (np.pi - x[i]) * (np.pi - x[next_wire])
            qml.RZ(phase, wires=next_wire)
            qml.CNOT(wires=[i, next_wire])
        for i in range(n_qubits):
            qml.RZ(x[i], wires=i)

        observables = []
        for i in range(n_qubits):
            observables.append(qml.expval(qml.PauliX(i)))
            observables.append(qml.expval(qml.PauliY(i)))
            observables.append(qml.expval(qml.PauliZ(i)))
        return observables

    return statevector_circuit, pqk_circuit

# ══════════════════════════════════════════════════════════════════════
# QUANTUM KERNEL COMPUTATION
# ══════════════════════════════════════════════════════════════════════

def compute_quantum_fidelity_kernel(states_train, states_test):
    """
    Computes K_ij = |<ψ_i | ψ_j>|^2 via vectorized inner products.
    states: (M, 2^n) complex statevectors.
    """
    # Overlap matrix: (M_test, M_train)
    overlap = np.matmul(states_test, states_train.conj().T)
    # Fidelity kernel is squared magnitude
    return np.abs(overlap) ** 2

def compute_pqk_kernel(pqk_train, pqk_test, gamma=1.0):
    """
    Projected Quantum Kernel: K(x, z) = exp(-gamma * ||f(x) - f(z)||^2)
    where f(x) are the local Pauli 1-RDM expectation values.
    """
    from sklearn.metrics.pairwise import rbf_kernel
    return rbf_kernel(pqk_test, pqk_train, gamma=gamma)

# ══════════════════════════════════════════════════════════════════════
# KERNEL RIDGE REGRESSION SOLVER
# ══════════════════════════════════════════════════════════════════════

class KernelRidgeRegressor:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.alpha_weights = None

    def fit(self, K_train, y_train):
        n = K_train.shape[0]
        # Regularized solve: (K + alpha * I) alpha_w = y
        A = K_train + self.alpha * np.eye(n)
        self.alpha_weights = np.linalg.solve(A, y_train)
        return self

    def predict(self, K_test_train):
        return np.dot(K_test_train, self.alpha_weights)

# ══════════════════════════════════════════════════════════════════════
# MAIN BENCHMARK EXPERIMENT
# ══════════════════════════════════════════════════════════════════════

def main():
    log("=" * 70)
    log("QUANTUM MACHINE LEARNING BENCHMARK ON MATBENCH_DIELECTRIC")
    log("=" * 70)

    n_qubits = 8
    log(f"Using {n_qubits} qubits (PCA-8 angle-encoded features)")

    # Load Fold 0 data
    fold = 0
    X_train_pca = np.load(FOLDS_DIR / f"fold_{fold}_pca8_train.npy")
    X_test_pca  = np.load(FOLDS_DIR / f"fold_{fold}_pca8_test.npy")
    X_train_ang = np.load(FOLDS_DIR / f"fold_{fold}_angle8_train.npy")
    X_test_ang  = np.load(FOLDS_DIR / f"fold_{fold}_angle8_test.npy")
    y_train_all = np.load(FOLDS_DIR / f"fold_{fold}_y_train.npy")
    y_test_all  = np.load(FOLDS_DIR / f"fold_{fold}_y_test.npy")

    log(f"Loaded Fold {fold}: Total Train = {len(y_train_all)}, Test = {len(y_test_all)}")

    # Set up circuits
    log("Initializing PennyLane statevector and PQK quantum circuits ...")
    state_fn, pqk_fn = build_qml_circuits(n_qubits)

    # ── Experiment 1: Small-Data Learning Curves (Gap 7 Validation) ───
    log("\n" + "=" * 70)
    log("EXPERIMENT 1: Small-Data Learning Curves (Gap 7 Validation)")
    log("=" * 70)

    sample_sizes = [50, 100, 200, 400, 800]
    n_test_eval = 300  # Fixed evaluation test set

    X_te_pca_sub = X_test_pca[:n_test_eval]
    X_te_ang_sub = X_test_ang[:n_test_eval]
    y_test_sub   = y_test_all[:n_test_eval]

    log(f"Computing quantum states for {n_test_eval} test samples ...")
    t0 = time.time()
    test_states = np.array([state_fn(x) for x in X_te_ang_sub])
    test_pqk    = np.array([pqk_fn(x) for x in X_te_ang_sub])
    log(f"Test states generated in {time.time() - t0:.2f}s")

    curve_results = {
        "Classical_Ridge": [],
        "Classical_RBF_SVR": [],
        "Classical_Poly_SVR": [],
        "Quantum_Fidelity_Kernel": [],
        "Projected_Quantum_Kernel": [],
    }

    max_train = max(sample_sizes)
    log(f"Pre-generating quantum representations for max train size ({max_train}) ...")
    t0 = time.time()
    train_states_all = np.array([state_fn(x) for x in X_train_ang[:max_train]])
    train_pqk_all    = np.array([pqk_fn(x) for x in X_train_ang[:max_train]])
    log(f"Train quantum representations generated in {time.time() - t0:.2f}s")

    for n_tr in sample_sizes:
        log(f"\n── Training with N = {n_tr} samples ──")
        X_tr_p = X_train_pca[:n_tr]
        y_tr   = y_train_all[:n_tr]

        # 1. Classical Ridge
        ridge = Ridge(alpha=1.0).fit(X_tr_p, y_tr)
        mae_ridge = mean_absolute_error(y_test_sub, ridge.predict(X_te_pca_sub))
        curve_results["Classical_Ridge"].append(float(mae_ridge))

        # 2. Classical RBF SVR
        rbf_svr = SVR(kernel="rbf", C=10.0, gamma="scale").fit(X_tr_p, y_tr)
        mae_rbf = mean_absolute_error(y_test_sub, rbf_svr.predict(X_te_pca_sub))
        curve_results["Classical_RBF_SVR"].append(float(mae_rbf))

        # 3. Classical Poly SVR
        poly_svr = SVR(kernel="poly", degree=3, C=1.0).fit(X_tr_p, y_tr)
        mae_poly = mean_absolute_error(y_test_sub, poly_svr.predict(X_te_pca_sub))
        curve_results["Classical_Poly_SVR"].append(float(mae_poly))

        # 4. Quantum Fidelity Kernel + KRR
        sub_train_states = train_states_all[:n_tr]
        K_tr_q = compute_quantum_fidelity_kernel(sub_train_states, sub_train_states)
        K_te_q = compute_quantum_fidelity_kernel(sub_train_states, test_states)

        q_krr = KernelRidgeRegressor(alpha=0.1).fit(K_tr_q, y_tr)
        mae_q = mean_absolute_error(y_test_sub, q_krr.predict(K_te_q))
        curve_results["Quantum_Fidelity_Kernel"].append(float(mae_q))

        # 5. Projected Quantum Kernel (PQK) + KRR
        sub_train_pqk = train_pqk_all[:n_tr]
        gamma_pqk = 1.0 / (3 * n_qubits)
        K_tr_pqk = compute_pqk_kernel(sub_train_pqk, sub_train_pqk, gamma=gamma_pqk)
        K_te_pqk = compute_pqk_kernel(sub_train_pqk, test_pqk, gamma=gamma_pqk)

        pqk_krr = KernelRidgeRegressor(alpha=0.1).fit(K_tr_pqk, y_tr)
        mae_pqk = mean_absolute_error(y_test_sub, pqk_krr.predict(K_te_pqk))
        curve_results["Projected_Quantum_Kernel"].append(float(mae_pqk))

        log(f"  N={n_tr:3d} | Classical RBF: {mae_rbf:.4f} | Classical Poly: {mae_poly:.4f} | Quantum Fidelity: {mae_q:.4f} | PQK: {mae_pqk:.4f}")

    # ── Plot Learning Curves ──────────────────────────────────────────
    plt.figure(figsize=(9, 5.5), dpi=300)
    sns.set_theme(style="whitegrid")

    styles = {
        "Classical_RBF_SVR": ("#2563eb", "s-", "Classical RBF SVR (Identical PCA-8)"),
        "Classical_Poly_SVR": ("#64748b", "d--", "Classical Poly SVR"),
        "Classical_Ridge": ("#94a3b8", "^:", "Classical Ridge"),
        "Quantum_Fidelity_Kernel": ("#dc2626", "o-", "Quantum Fidelity Kernel (ZZ-Entangled)"),
        "Projected_Quantum_Kernel": ("#16a34a", "*-", "Projected Quantum Kernel (PQK)"),
    }

    for model_name, (color, fmt, label) in styles.items():
        plt.plot(sample_sizes, curve_results[model_name], fmt, color=color, linewidth=2, markersize=7, label=label)

    plt.title("Quantum vs Classical Learning Curves on Materials Property Prediction\n(matbench_dielectric, 8 Qubits / PCA-8)", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Number of Training Samples (N)", fontsize=11, fontweight="bold")
    plt.ylabel("Test Mean Absolute Error (MAE)", fontsize=11, fontweight="bold")
    plt.xticks(sample_sizes)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()

    fig_path = FIGURES_DIR / "quantum_vs_classical_learning_curve.png"
    plt.savefig(fig_path)
    plt.close()
    log(f"\nSaved learning curve figure to: {fig_path}")

    # ── Summary JSON ──────────────────────────────────────────────────
    benchmark_data = {
        "experiment": "quantum_vs_classical_learning_curves",
        "n_qubits": n_qubits,
        "sample_sizes": sample_sizes,
        "n_test": n_test_eval,
        "results": curve_results,
        "summary": {
            "best_classical_at_800": float(min(curve_results["Classical_RBF_SVR"][-1], curve_results["Classical_Poly_SVR"][-1])),
            "best_quantum_at_800": float(min(curve_results["Quantum_Fidelity_Kernel"][-1], curve_results["Projected_Quantum_Kernel"][-1])),
        }
    }

    out_json = RESULTS_DIR / "quantum_benchmark_results.json"
    with open(out_json, "w") as f:
        json.dump(benchmark_data, f, indent=2)
    log(f"Saved quantum benchmark results to: {out_json}")

    log("\n" + "=" * 70)
    log("[SUCCESS] Quantum kernel benchmarking complete!")
    log("=" * 70)

if __name__ == "__main__":
    main()
