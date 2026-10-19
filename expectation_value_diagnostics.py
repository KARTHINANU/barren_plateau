"""
Expectation Value Diagnostics & Barren Plateau Foundations (Module 1)
====================================================================
Implements exact quantum statevector simulation to systematically investigate:
1. Parameterized state generation: |psi(theta)> = U(theta)|0>^{\otimes n}
2. Expectation value of local observable O = Z_0 (SparsePauliOp("I"*(n-1) + "Z")):
   C(theta) = <psi(theta)|O|psi(theta)>
3. Parameter-shift rule gradient for theta[0] (Layer 1, Qubit 0, Rx):
   g = [C(theta + pi/2 * e_0) - C(theta - pi/2 * e_0)] / 2
4. Statistical diagnostics across random parameter trials:
   - Expectation value C(theta)
   - Mean expectation value E[C]
   - Variance of expectation values Var(C)
   - Gradient g
   - Mean gradient E[g]
   - Gradient variance Var(g)
   - Mean absolute gradient E[|g|]
   - Mean squared gradient E[g^2]

Critical scientific distinctions:
- E[C] ~ 0 does NOT indicate a barren plateau (traceless Pauli operators have E[C]=0 by Haar symmetry).
- E[g] ~ 0 does NOT indicate gradient vanishing (positive and negative gradients cancel).
- True gradient vanishing is characterized by Var(g) -> 0 and E[|g|] -> 0.
- Var(C) -> 0 measures cost function concentration across the state space.
"""

import os
import sys
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, SparsePauliOp


def build_circuit(params_list, n_qubits, n_layers):
    """
    Constructs a parameterized quantum circuit using direct numerical angle injection.
    Architecture:
      n_layers repetitions of:
        - Rx(theta), Ry(theta), Rz(theta) on every qubit (3 * n_qubits parameters per layer)
        - Linear nearest-neighbor CNOT chain: qc.cx(q, q+1) for q in [0, n_qubits-2]
    """
    qc = QuantumCircuit(n_qubits)
    idx = 0
    for l in range(n_layers):
        for q in range(n_qubits):
            qc.rx(params_list[idx], q); idx += 1
            qc.ry(params_list[idx], q); idx += 1
            qc.rz(params_list[idx], q); idx += 1
        for q in range(n_qubits - 1):
            qc.cx(q, q + 1)
    return qc


def get_observable(n_qubits, target_qubit=0):
    """
    Constructs Pauli-Z observable on target_qubit (default: qubit 0).
    In Qiskit's SparsePauliOp convention, strings are little-endian (right-to-left),
    so Z on qubit 0 is represented as 'I' * (n_qubits - 1) + 'Z'.
    """
    pauli_list = ["I"] * n_qubits
    # In Qiskit string convention, qubit 0 corresponds to the rightmost character: index n_qubits - 1 - target_qubit
    pauli_list[n_qubits - 1 - target_qubit] = "Z"
    pauli_str = "".join(pauli_list)
    return SparsePauliOp(pauli_str)


def compute_cost_and_gradient(params, n_qubits, n_layers, param_idx, observable):
    """
    Evaluates:
      1. Base cost: C(theta) = <psi(theta)|O|psi(theta)>
      2. Shifted costs: C(theta + pi/2), C(theta - pi/2)
      3. Parameter-shift gradient: g = [C(theta + pi/2) - C(theta - pi/2)] / 2
    """
    # 1. Base evaluation
    qc_base = build_circuit(params, n_qubits, n_layers)
    sv_base = Statevector.from_instruction(qc_base)
    cost_base = sv_base.expectation_value(observable).real

    # 2. Positive shift
    params_plus = list(params)
    params_plus[param_idx] += np.pi / 2
    qc_plus = build_circuit(params_plus, n_qubits, n_layers)
    sv_plus = Statevector.from_instruction(qc_plus)
    cost_plus = sv_plus.expectation_value(observable).real

    # 3. Negative shift
    params_minus = list(params)
    params_minus[param_idx] -= np.pi / 2
    qc_minus = build_circuit(params_minus, n_qubits, n_layers)
    sv_minus = Statevector.from_instruction(qc_minus)
    cost_minus = sv_minus.expectation_value(observable).real

    # 4. Parameter-shift gradient
    gradient = (cost_plus - cost_minus) / 2.0

    return cost_base, gradient


def run_diagnostics(qubit_counts, depths, n_trials=20, base_seed=42,
                    output_csv="results_expectation_diagnostics.csv",
                    output_plot="expectation_diagnostics.png"):
    """
    Runs expectation value and gradient diagnostics across a grid of qubit counts and depths.
    Uses synchronized seeds: seed = base_seed + trial_id for trial-level reproducibility.
    """
    print("\n" + "=" * 75)
    print("EXPECTATION VALUE & GRADIENT DIAGNOSTICS (MODULE 1)")
    print("=" * 75)
    print(f"Qubit configurations: {qubit_counts}")
    print(f"Evaluated depths:     {depths}")
    print(f"Trials per point:     {n_trials}")
    print(f"Base seed:            {base_seed}")
    print(f"Target parameter:     theta[0] (Layer 1, Qubit 0, Rx)")
    print("=" * 75)

    results = []

    for n_qubits in qubit_counts:
        observable = get_observable(n_qubits, target_qubit=0)
        print(f"\n[Qubits n = {n_qubits}] Observable: {observable.paulis[0]}")

        for depth in depths:
            n_params = 3 * n_qubits * depth
            costs = []
            gradients = []

            for trial_id in range(n_trials):
                trial_seed = base_seed + trial_id
                rng = np.random.RandomState(trial_seed)
                # Uniform random parameters in [0, 2pi)
                params = rng.uniform(0, 2 * np.pi, n_params)

                cost, grad = compute_cost_and_gradient(
                    params, n_qubits, depth, param_idx=0, observable=observable
                )
                costs.append(cost)
                gradients.append(grad)

            costs = np.array(costs)
            gradients = np.array(gradients)

            # Statistical calculations
            mean_c = float(np.mean(costs))
            var_c = float(np.var(costs))
            std_c = float(np.std(costs))

            mean_g = float(np.mean(gradients))
            var_g = float(np.var(gradients))
            std_g = float(np.std(gradients))
            mean_abs_g = float(np.mean(np.abs(gradients)))
            mean_sq_g = float(np.mean(gradients ** 2))

            # Mathematical consistency validation: Var(g) == E[g^2] - (E[g])^2
            theoretical_var_g = mean_sq_g - (mean_g ** 2)
            assert abs(var_g - theoretical_var_g) < 1e-12, "Statistical identity check failed!"

            record = {
                "n_qubits": n_qubits,
                "depth": depth,
                "n_trials": n_trials,
                "mean_expectation": mean_c,
                "var_expectation": var_c,
                "std_expectation": std_c,
                "mean_gradient": mean_g,
                "var_gradient": var_g,
                "std_gradient": std_g,
                "mean_abs_gradient": mean_abs_g,
                "mean_sq_gradient": mean_sq_g,
            }
            results.append(record)

            print(f"  Depth {depth:2d} | E[C]={mean_c:+.4e}, Var(C)={var_c:.4e} | "
                  f"E[g]={mean_g:+.4e}, Var(g)={var_g:.4e}, E[|g|]={mean_abs_g:.4e}, E[g^2]={mean_sq_g:.4e}")

    # Write CSV
    fieldnames = [
        "n_qubits", "depth", "n_trials",
        "mean_expectation", "var_expectation", "std_expectation",
        "mean_gradient", "var_gradient", "std_gradient",
        "mean_abs_gradient", "mean_sq_gradient"
    ]
    with open(output_csv, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n[CSV Saved] Diagnostic metrics written to: {output_csv}")

    # Generate Plot
    generate_diagnostics_plot(results, qubit_counts, depths, output_plot)

    return results


def generate_diagnostics_plot(results, qubit_counts, depths, output_plot):
    """
    Renders 4 diagnostic panels:
    1. Expectation Value Variance Var(C) vs Depth
    2. Gradient Variance Var(g) vs Depth (log scale)
    3. Mean Absolute Gradient E[|g|] vs Depth
    4. Gradient Variance vs Expectation Value Variance (Correlation / Concentration)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = {4: "#1f77b4", 6: "#d62728", 8: "#2ca02c"}
    markers = {4: "o-", 6: "s-", 8: "^-"}

    # 1. Expectation Variance vs Depth
    ax1 = axes[0, 0]
    for n in qubit_counts:
        pts = [r for r in results if r["n_qubits"] == n]
        ds = [r["depth"] for r in pts]
        var_cs = [r["var_expectation"] for r in pts]
        c = colors.get(n, "black")
        m = markers.get(n, "o-")
        ax1.plot(ds, var_cs, m, color=c, label=f"n = {n} qubits", linewidth=2, markersize=6)
    ax1.set_xlabel("Circuit Depth L (Layers)", fontsize=11)
    ax1.set_ylabel(r"Expectation Value Variance $\mathrm{Var}(C)$", fontsize=11)
    ax1.set_title(r"Cost Landscape Concentration $\mathrm{Var}(\langle Z_0 \rangle)$ vs Depth", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 2. Gradient Variance vs Depth (Log Scale)
    ax2 = axes[0, 1]
    for n in qubit_counts:
        pts = [r for r in results if r["n_qubits"] == n]
        ds = [r["depth"] for r in pts]
        var_gs = [r["var_gradient"] for r in pts]
        c = colors.get(n, "black")
        m = markers.get(n, "o-")
        ax2.semilogy(ds, var_gs, m, color=c, label=f"n = {n} qubits", linewidth=2, markersize=6)
    ax2.set_xlabel("Circuit Depth L (Layers)", fontsize=11)
    ax2.set_ylabel(r"Gradient Variance $\mathrm{Var}(\partial_{\theta_0} C)$ (log scale)", fontsize=11)
    ax2.set_title(r"Gradient Variance $\mathrm{Var}(g)$ vs Depth", fontsize=12, fontweight="bold")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend()

    # 3. Mean Absolute Gradient vs Depth
    ax3 = axes[1, 0]
    for n in qubit_counts:
        pts = [r for r in results if r["n_qubits"] == n]
        ds = [r["depth"] for r in pts]
        mag_gs = [r["mean_abs_gradient"] for r in pts]
        c = colors.get(n, "black")
        m = markers.get(n, "o-")
        ax3.plot(ds, mag_gs, m, color=c, label=f"n = {n} qubits", linewidth=2, markersize=6)
    ax3.set_xlabel("Circuit Depth L (Layers)", fontsize=11)
    ax3.set_ylabel(r"Mean Absolute Gradient $\mathbb{E}[|\partial_{\theta_0} C|]$", fontsize=11)
    ax3.set_title(r"Typical Step Magnitude $\mathbb{E}[|g|]$ vs Depth", fontsize=12, fontweight="bold")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # 4. Var(g) vs Var(C) Scatter / Relationship
    ax4 = axes[1, 1]
    for n in qubit_counts:
        pts = [r for r in results if r["n_qubits"] == n]
        var_cs = [r["var_expectation"] for r in pts]
        var_gs = [r["var_gradient"] for r in pts]
        c = colors.get(n, "black")
        ax4.scatter(var_cs, var_gs, color=c, s=60, label=f"n = {n} qubits", zorder=3)
        for r in pts:
            ax4.annotate(f"L={r['depth']}", (r["var_expectation"], r["var_gradient"]),
                         textcoords="offset points", xytext=(5, 3), fontsize=9)
    ax4.set_xlabel(r"$\mathrm{Var}(C)$ (Landscape Variance)", fontsize=11)
    ax4.set_ylabel(r"$\mathrm{Var}(g)$ (Gradient Variance)", fontsize=11)
    ax4.set_title(r"Gradient Variance vs Cost Variance", fontsize=12, fontweight="bold")
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    fig.suptitle("Module 1: Expectation Value & Gradient Diagnostics\n(Demonstrating Cost Concentration & Gradient Suppression)",
                 fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_plot, dpi=150)
    plt.close(fig)
    print(f"[Plot Saved] Diagnostic figure written to: {output_plot}")


def main():
    parser = argparse.ArgumentParser(description="Expectation Value and Gradient Diagnostics (Module 1)")
    parser.add_argument("--qubits", type=int, nargs="+", default=[4, 6],
                        help="List of qubit counts to evaluate (default: 4 6)")
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 4, 8],
                        help="List of circuit depths to evaluate (default: 1 4 8)")
    parser.add_argument("--trials", type=int, default=20,
                        help="Number of random parameter trials per point (default: 20)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed (default: 42)")
    parser.add_argument("--csv", type=str, default="results_expectation_diagnostics.csv",
                        help="Output CSV path")
    parser.add_argument("--plot", type=str, default="expectation_diagnostics.png",
                        help="Output plot path")

    args = parser.parse_args()

    run_diagnostics(
        qubit_counts=args.qubits,
        depths=args.depths,
        n_trials=args.trials,
        base_seed=args.seed,
        output_csv=args.csv,
        output_plot=args.plot
    )


if __name__ == "__main__":
    main()
