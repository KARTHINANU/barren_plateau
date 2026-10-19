"""
Quantum-Classical Gradient Comparison (Module 3)
================================================
Compares gradient estimation mechanisms between:
1. Quantum Parameter-Shift Rule:
   g_PS = [C(theta + pi/2) - C(theta - pi/2)] / 2
   (exact statevector and finite-shot sampled)
2. Numerical Finite Differences (Quantum & Classical):
   g_FD = [Cost(theta + eps) - Cost(theta - eps)] / (2 * eps)
3. Classical Backpropagation (Reverse-Mode Automatic Differentiation):
   Evaluated on a classical Multi-Layer Perceptron (MLP) implemented in pure NumPy.

Scientific Grounding:
- Methodological comparison, NOT a direct performance competition.
- Quantum and classical neural models operate on fundamentally distinct mathematical spaces:
  - Quantum parameters are unitary rotation angles acting on a 2^n complex Hilbert space.
  - Classical parameters are continuous weights and biases acting on affine hyperplanes.
- Computational evaluation requirements:
  - Quantum parameter-shift requires 2 distinct circuit evaluations per parameter (2P total).
  - Classical backpropagation computes all P gradients in a single backward pass (O(1) passes).
- Vanishing gradient origins:
  - Classical: activation function saturation (sigmoid/tanh) or deep layer weight contraction.
  - Quantum: Haar-measure concentration of measure on unitary group SU(2^n) and expressibility.
"""

import os
import sys
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, SparsePauliOp


# ==============================================================================
# 1. QUANTUM GRADIENT COMPUTATION ENGINE
# ==============================================================================

def build_quantum_circuit(params, n_qubits, n_layers):
    """Builds parameterized circuit with Rx-Ry-Rz rotations and linear CNOT entanglement."""
    qc = QuantumCircuit(n_qubits)
    idx = 0
    for l in range(n_layers):
        for q in range(n_qubits):
            qc.rx(params[idx], q); idx += 1
            qc.ry(params[idx], q); idx += 1
            qc.rz(params[idx], q); idx += 1
        for q in range(n_qubits - 1):
            qc.cx(q, q + 1)
    return qc


def get_quantum_observable(n_qubits, target_qubit=0):
    """Returns local Pauli-Z observable on target_qubit in Qiskit reverse convention."""
    pauli_list = ["I"] * n_qubits
    pauli_list[n_qubits - 1 - target_qubit] = "Z"
    return SparsePauliOp("".join(pauli_list))


def compute_quantum_cost_statevector(params, n_qubits, n_layers, observable):
    """Computes exact expectation value <psi|O|psi> using Statevector."""
    qc = build_quantum_circuit(params, n_qubits, n_layers)
    sv = Statevector.from_instruction(qc)
    return float(sv.expectation_value(observable).real)


def compute_quantum_cost_shots(params, n_qubits, n_layers, observable, shots, rng):
    """Simulates finite-shot expectation value via exact marginal probabilities + binomial sampling."""
    qc = build_quantum_circuit(params, n_qubits, n_layers)
    sv = Statevector.from_instruction(qc)
    probs = sv.probabilities()
    # Marginal probability of qubit 0 = 0 (eigenvalue +1)
    p_plus = float(np.sum([probs[k] for k in range(len(probs)) if (k & 1) == 0]))
    n_plus = rng.binomial(n=shots, p=p_plus)
    n_minus = shots - n_plus
    return (n_plus - n_minus) / float(shots)


def quantum_parameter_shift_gradient(params, n_qubits, n_layers, param_idx, observable):
    """Exact parameter-shift rule: g = [C(theta + pi/2) - C(theta - pi/2)] / 2."""
    p_plus = np.array(params, dtype=float); p_plus[param_idx] += np.pi / 2.0
    p_minus = np.array(params, dtype=float); p_minus[param_idx] -= np.pi / 2.0
    c_plus = compute_quantum_cost_statevector(p_plus, n_qubits, n_layers, observable)
    c_minus = compute_quantum_cost_statevector(p_minus, n_qubits, n_layers, observable)
    return (c_plus - c_minus) / 2.0, c_plus, c_minus


def quantum_finite_difference_gradient(params, n_qubits, n_layers, param_idx, observable, eps=1e-5):
    """Two-point central finite differences: g_FD = [C(theta + eps) - C(theta - eps)] / (2*eps)."""
    p_plus = np.array(params, dtype=float); p_plus[param_idx] += eps
    p_minus = np.array(params, dtype=float); p_minus[param_idx] -= eps
    c_plus = compute_quantum_cost_statevector(p_plus, n_qubits, n_layers, observable)
    c_minus = compute_quantum_cost_statevector(p_minus, n_qubits, n_layers, observable)
    return (c_plus - c_minus) / (2.0 * eps)


def quantum_shot_parameter_shift_gradient(params, n_qubits, n_layers, param_idx, observable, shots, rng):
    """Finite-shot sampled parameter-shift gradient."""
    p_plus = np.array(params, dtype=float); p_plus[param_idx] += np.pi / 2.0
    p_minus = np.array(params, dtype=float); p_minus[param_idx] -= np.pi / 2.0
    c_plus_hat = compute_quantum_cost_shots(p_plus, n_qubits, n_layers, observable, shots, rng)
    c_minus_hat = compute_quantum_cost_shots(p_minus, n_qubits, n_layers, observable, shots, rng)
    return (c_plus_hat - c_minus_hat) / 2.0


# ==============================================================================
# 2. CLASSICAL NEURAL NETWORK & BACKPROPAGATION ENGINE
# ==============================================================================

class SimpleClassicalMLP:
    """
    Two-layer classical Multi-Layer Perceptron (MLP) implemented in pure NumPy.
    Architecture:
      Input: x in R^{d_in}
      Hidden: z1 = W1 @ x + b1,  a1 = sigmoid(z1)
      Output: y_hat = W2 @ a1 + b2
      Loss: Mean Squared Error L = 0.5 * (y_hat - y_target)^2
    """
    def __init__(self, d_in=4, d_hidden=8, d_out=1, seed=42):
        rng = np.random.RandomState(seed)
        # Xavier/Glorot normal initialization
        self.W1 = rng.randn(d_hidden, d_in) * np.sqrt(2.0 / (d_in + d_hidden))
        self.b1 = np.zeros((d_hidden, 1))
        self.W2 = rng.randn(d_out, d_hidden) * np.sqrt(2.0 / (d_hidden + d_out))
        self.b2 = np.zeros((d_out, 1))

    @staticmethod
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    @staticmethod
    def sigmoid_prime(z):
        s = SimpleClassicalMLP.sigmoid(z)
        return s * (1.0 - s)

    def forward(self, x):
        self.x = x.reshape(-1, 1)
        self.z1 = self.W1 @ self.x + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = self.W2 @ self.a1 + self.b2
        self.y_hat = self.z2
        return self.y_hat

    def compute_loss(self, y_hat, y_target):
        return 0.5 * float(np.sum((y_hat - y_target) ** 2))

    def backward(self, y_target):
        """Analytical backpropagation: single backward pass computes all gradients."""
        dL_dyhat = (self.y_hat - y_target)  # (d_out, 1)
        # Output layer gradients
        grad_W2 = dL_dyhat @ self.a1.T      # (d_out, d_hidden)
        grad_b2 = dL_dyhat                  # (d_out, 1)

        # Hidden layer gradients via chain rule
        dL_da1 = self.W2.T @ dL_dyhat       # (d_hidden, 1)
        dL_dz1 = dL_da1 * self.sigmoid_prime(self.z1)  # (d_hidden, 1)
        grad_W1 = dL_dz1 @ self.x.T         # (d_hidden, d_in)
        grad_b1 = dL_dz1                    # (d_hidden, 1)

        return {
            "grad_W1": grad_W1,
            "grad_b1": grad_b1,
            "grad_W2": grad_W2,
            "grad_b2": grad_b2
        }

    def compute_finite_difference_weight(self, x, y_target, layer="W1", row=0, col=0, eps=1e-5):
        """Numerical finite differences for a specific classical weight parameter."""
        target_param = getattr(self, layer)
        orig_val = target_param[row, col]

        # + eps
        target_param[row, col] = orig_val + eps
        y_plus = self.forward(x)
        loss_plus = self.compute_loss(y_plus, y_target)

        # - eps
        target_param[row, col] = orig_val - eps
        y_minus = self.forward(x)
        loss_minus = self.compute_loss(y_minus, y_target)

        # Restore original parameter
        target_param[row, col] = orig_val
        self.forward(x)

        return (loss_plus - loss_minus) / (2.0 * eps)


# ==============================================================================
# 3. EXPERIMENT EXECUTION & COMPARISON
# ==============================================================================

def run_gradient_comparison(n_trials=30, base_seed=42, output_csv="results_quantum_classical_comparison.csv",
                            output_plot="quantum_classical_comparison.png"):
    """
    Executes systematic comparison of:
    1. Quantum Parameter-Shift vs Quantum Finite Differences (Mathematical Validation)
    2. Quantum Parameter-Shift vs Finite-Shot Sampling (Noise Impact: M=100, 1000, 10000)
    3. Classical Backpropagation vs Classical Finite Differences (Mathematical Validation)
    """
    print("\n" + "=" * 78)
    print("MODULE 3: QUANTUM VS CLASSICAL GRADIENT ESTIMATION COMPARISON")
    print("=" * 78)
    print(f"Independent Trials:     {n_trials}")
    print(f"Base Random Seed:       {base_seed}")
    print(f"Quantum Architecture:   n=4 qubits, L=2 layers (24 rotation angles, local Z_0)")
    print(f"Classical Architecture: 4 -> 8 -> 1 MLP (Sigmoid activation, MSE loss)")
    print("=" * 78)

    n_qubits = 4
    n_layers = 2
    n_params = 3 * n_qubits * n_layers
    observable = get_quantum_observable(n_qubits, target_qubit=0)

    records = []

    diff_ps_fd_quantum = []
    diff_bp_fd_classical = []
    shot_errors_m100 = []
    shot_errors_m1000 = []
    shot_errors_m10000 = []

    for trial_id in range(n_trials):
        seed_trial = base_seed + trial_id
        rng_q = np.random.RandomState(seed_trial)

        # ----------------- Quantum Gradient Evaluation -----------------
        q_params = rng_q.uniform(0, 2 * np.pi, n_params)
        target_param_idx = 0  # theta[0]: Layer 1, Qubit 0, Rx

        # Exact Parameter-Shift
        g_ps, c_p, c_m = quantum_parameter_shift_gradient(
            q_params, n_qubits, n_layers, target_param_idx, observable
        )
        # Quantum Finite Differences (eps = 1e-5)
        g_q_fd = quantum_finite_difference_gradient(
            q_params, n_qubits, n_layers, target_param_idx, observable, eps=1e-5
        )
        # Shot-based parameter-shift estimates
        rng_s1 = np.random.RandomState(seed_trial + 1000)
        rng_s2 = np.random.RandomState(seed_trial + 2000)
        rng_s3 = np.random.RandomState(seed_trial + 3000)
        g_shot_100 = quantum_shot_parameter_shift_gradient(
            q_params, n_qubits, n_layers, target_param_idx, observable, shots=100, rng=rng_s1
        )
        g_shot_1000 = quantum_shot_parameter_shift_gradient(
            q_params, n_qubits, n_layers, target_param_idx, observable, shots=1000, rng=rng_s2
        )
        g_shot_10000 = quantum_shot_parameter_shift_gradient(
            q_params, n_qubits, n_layers, target_param_idx, observable, shots=10000, rng=rng_s3
        )

        q_fd_discrepancy = abs(g_ps - g_q_fd)
        diff_ps_fd_quantum.append(q_fd_discrepancy)
        shot_errors_m100.append(abs(g_shot_100 - g_ps))
        shot_errors_m1000.append(abs(g_shot_1000 - g_ps))
        shot_errors_m10000.append(abs(g_shot_10000 - g_ps))

        # ----------------- Classical Gradient Evaluation -----------------
        rng_c = np.random.RandomState(seed_trial + 500)
        mlp = SimpleClassicalMLP(d_in=4, d_hidden=8, d_out=1, seed=seed_trial + 500)
        x_sample = rng_c.randn(4, 1)
        y_target = np.array([[1.0]])

        # Forward & Analytical Backprop
        y_hat = mlp.forward(x_sample)
        grads_bp = mlp.backward(y_target)
        g_backprop_target = grads_bp["grad_W1"][0, 0]

        # Classical Finite Differences (eps = 1e-5)
        g_c_fd = mlp.compute_finite_difference_weight(
            x_sample, y_target, layer="W1", row=0, col=0, eps=1e-5
        )
        c_fd_discrepancy = abs(g_backprop_target - g_c_fd)
        diff_bp_fd_classical.append(c_fd_discrepancy)

        # Verification check: independent numerical differentiation must match analytical gradients
        assert q_fd_discrepancy < 1e-4, f"Quantum FD check failed on trial {trial_id}!"
        assert c_fd_discrepancy < 1e-4, f"Classical FD check failed on trial {trial_id}!"

        records.append({
            "trial_id": trial_id,
            "quantum_exact_ps_gradient": g_ps,
            "quantum_finite_diff_gradient": g_q_fd,
            "quantum_fd_discrepancy": q_fd_discrepancy,
            "quantum_shot100_gradient": g_shot_100,
            "quantum_shot1000_gradient": g_shot_1000,
            "quantum_shot10000_gradient": g_shot_10000,
            "classical_backprop_gradient": g_backprop_target,
            "classical_finite_diff_gradient": g_c_fd,
            "classical_fd_discrepancy": c_fd_discrepancy
        })

    # Save to CSV
    fieldnames = list(records[0].keys())
    with open(output_csv, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"\n[CSV Saved] Gradient comparison data written to: {output_csv}")

    # Print Summary Statistics
    print("\n--- GRADIENT VALIDATION & ERROR SUMMARY ---")
    print(f"Quantum Parameter-Shift vs Quantum Finite Differences (|g_PS - g_FD|):")
    print(f"  Mean Discrepancy: {np.mean(diff_ps_fd_quantum):.4e} | Max: {np.max(diff_ps_fd_quantum):.4e}")
    print(f"Classical Backpropagation vs Classical Finite Differences (|g_BP - g_FD|):")
    print(f"  Mean Discrepancy: {np.mean(diff_bp_fd_classical):.4e} | Max: {np.max(diff_bp_fd_classical):.4e}")
    print(f"Quantum Parameter-Shift Sampling Error under Shot Noise (|g_hat_M - g_exact|):")
    print(f"  M = 100 shots:   Mean Abs Error = {np.mean(shot_errors_m100):.4e} | RMSE = {np.sqrt(np.mean(np.array(shot_errors_m100)**2)):.4e}")
    print(f"  M = 1,000 shots: Mean Abs Error = {np.mean(shot_errors_m1000):.4e} | RMSE = {np.sqrt(np.mean(np.array(shot_errors_m1000)**2)):.4e}")
    print(f"  M = 10,000 shots:Mean Abs Error = {np.mean(shot_errors_m10000):.4e} | RMSE = {np.sqrt(np.mean(np.array(shot_errors_m10000)**2)):.4e}")

    # Generate Visualization
    generate_comparison_plot(records, shot_errors_m100, shot_errors_m1000, shot_errors_m10000,
                             output_plot)

    return records


def generate_comparison_plot(records, shot_errs_100, shot_errs_1000, shot_errs_10000, output_plot):
    """
    Renders 4 panels illustrating quantum vs classical gradient mechanisms:
    1. Quantum Parameter-Shift vs Quantum Finite Differences (Parity Check)
    2. Classical Backpropagation vs Classical Finite Differences (Parity Check)
    3. Quantum Shot-Noise Gradient Error Decay vs M
    4. Gradient Magnitude Distribution: Quantum (angles) vs Classical (synaptic weights)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Quantum PS vs FD
    ax1 = axes[0, 0]
    ps_vals = [r["quantum_exact_ps_gradient"] for r in records]
    fd_vals = [r["quantum_finite_diff_gradient"] for r in records]
    ax1.plot([-1, 1], [-1, 1], "k--", alpha=0.6, label="Ideal Line ($y=x$)")
    ax1.scatter(ps_vals, fd_vals, color="#2b83ba", s=50, alpha=0.8, edgecolors="none", label="Trials")
    ax1.set_xlabel(r"Exact Parameter-Shift Gradient $g_{\mathrm{PS}}$", fontsize=11)
    ax1.set_ylabel(r"Finite Differences Gradient $g_{\mathrm{FD}}$ ($\epsilon=10^{-5}$)", fontsize=11)
    ax1.set_title("Quantum Gradient Validation: Parameter-Shift vs Finite Differences", fontsize=11, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Panel 2: Classical Backprop vs FD
    ax2 = axes[0, 1]
    bp_vals = [r["classical_backprop_gradient"] for r in records]
    cfd_vals = [r["classical_finite_diff_gradient"] for r in records]
    min_c = min(min(bp_vals), min(cfd_vals))
    max_c = max(max(bp_vals), max(cfd_vals))
    ax2.plot([min_c, max_c], [min_c, max_c], "k--", alpha=0.6, label="Ideal Line ($y=x$)")
    ax2.scatter(bp_vals, cfd_vals, color="#d7191c", s=50, alpha=0.8, edgecolors="none", label="Trials")
    ax2.set_xlabel(r"Classical Backpropagation Gradient $\partial \mathcal{L} / \partial W_{1,00}$", fontsize=11)
    ax2.set_ylabel(r"Classical Finite Differences Gradient ($\epsilon=10^{-5}$)", fontsize=11)
    ax2.set_title("Classical Gradient Validation: Backpropagation vs Finite Differences", fontsize=11, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Panel 3: Quantum Shot-Noise Gradient Error vs M
    ax3 = axes[1, 0]
    ms = [100, 1000, 10000]
    mean_errs = [np.mean(shot_errs_100), np.mean(shot_errs_1000), np.mean(shot_errs_10000)]
    rmses = [
        np.sqrt(np.mean(np.array(shot_errs_100)**2)),
        np.sqrt(np.mean(np.array(shot_errs_1000)**2)),
        np.sqrt(np.mean(np.array(shot_errs_10000)**2))
    ]
    ax3.loglog(ms, rmses, "s-", color="#7570b3", linewidth=2, markersize=7, label="Empirical Gradient RMSE")
    ax3.loglog(ms, mean_errs, "o--", color="#1b9e77", linewidth=1.8, markersize=6, label="Mean Absolute Gradient Error")
    # Theoretical asymptote for gradient shot error: sqrt(1/(2M))
    theo_grad_se = [np.sqrt(1.0 / (2.0 * m)) for m in ms]
    ax3.loglog(ms, theo_grad_se, "k:", linewidth=1.5, label=r"Theoretical Noise Floor $\sim \sqrt{1/(2M)}$")
    ax3.set_xlabel("Measurement Shots ($M$)", fontsize=11)
    ax3.set_ylabel("Gradient Estimation Error", fontsize=11)
    ax3.set_title(r"Quantum Parameter-Shift Sampling Error vs Shots ($M$)", fontsize=11, fontweight="bold")
    ax3.grid(True, which="both", alpha=0.3)
    ax3.legend()

    # Panel 4: Gradient Magnitude Distribution Comparison
    ax4 = axes[1, 1]
    q_abs = np.abs(ps_vals)
    c_abs = np.abs(bp_vals)
    ax4.boxplot([q_abs, c_abs], tick_labels=["Quantum\nParameter-Shift", "Classical\nBackprop"], patch_artist=True,
                boxprops=dict(facecolor="#ccebc5", color="gray"))
    ax4.set_ylabel("Absolute Gradient Magnitude $|g|$", fontsize=11)
    ax4.set_title("Gradient Magnitude Distributions (Methodological Reference)", fontsize=11, fontweight="bold")
    ax4.grid(True, alpha=0.3)

    fig.suptitle("Module 3: Quantum vs Classical Gradient Estimation Comparison\n"
                 "(Parameter-Shift Rule vs Backpropagation & Numerical Validation)",
                 fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_plot, dpi=150)
    plt.close(fig)
    print(f"[Plot Saved] Figure saved to: {output_plot}")


def main():
    parser = argparse.ArgumentParser(description="Quantum vs Classical Gradient Comparison (Module 3)")
    parser.add_argument("--trials", type=int, default=30, help="Number of comparison trials (default: 30)")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed (default: 42)")
    parser.add_argument("--csv", type=str, default="results_quantum_classical_comparison.csv", help="Output CSV path")
    parser.add_argument("--plot", type=str, default="quantum_classical_comparison.png", help="Output plot path")

    args = parser.parse_args()

    run_gradient_comparison(
        n_trials=args.trials,
        base_seed=args.seed,
        output_csv=args.csv,
        output_plot=args.plot
    )


if __name__ == "__main__":
    main()
