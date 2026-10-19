"""
Barren Plateau Demonstration (Qiskit) — Baseline only, no mitigation
=====================================================================
Extends your original 3-qubit script into a loop over qubit count
(n = 2..7), replicating McClean et al. (2018). For each n we build a
random circuit (Rx-Ry-Rz per qubit + linear CNOT entanglers), compute
the gradient of theta[0] via the parameter-shift rule over many random
samples, and record the gradient variance. The variance vs n curve
should fall off sharply (exponential decay -> looks like a steep
downward curve, "parabolic downwards" on a linear plot).
"""

import csv
import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector, SparsePauliOp

np.random.seed(42)

N_EXPERIMENTS = 250          # random circuits per qubit count
QUBIT_RANGE = range(2, 8)    # n = 2, 3, 4, 5, 6, 7
N_LAYERS = 20                 # depth: needed for circuits to reach ~2-design
                              # randomness. A single layer is NOT deep enough
                              # to show the barren-plateau effect.


def build_circuit(n_qubits, n_layers=N_LAYERS):
    """n_layers repetitions of [Rx-Ry-Rz on every qubit] + [linear CNOT chain].
    Sufficient depth is what makes the circuit approximate a unitary 2-design,
    which is the actual cause of the barren plateau (McClean et al., 2018)."""
    qc = QuantumCircuit(n_qubits)
    n_params = 3 * n_qubits * n_layers
    theta = ParameterVector("theta", n_params)

    idx = 0
    for layer in range(n_layers):
        for q in range(n_qubits):
            qc.rx(theta[idx], q); idx += 1
            qc.ry(theta[idx], q); idx += 1
            qc.rz(theta[idx], q); idx += 1
        for q in range(n_qubits - 1):
            qc.cx(q, q + 1)

    return qc, theta


def observable_for(n_qubits):
    """Z on qubit 0, identity elsewhere -> string is reversed in Qiskit,
    so Z goes at the END of the Pauli string, e.g. n=3 -> 'IIZ'."""
    return SparsePauliOp("I" * (n_qubits - 1) + "Z")


def gradient_stats_for_n(n_qubits, n_experiments=N_EXPERIMENTS):
    qc, theta = build_circuit(n_qubits)
    observable = observable_for(n_qubits)
    n_params = 3 * n_qubits * N_LAYERS

    gradients = []

    for i in range(n_experiments):
        random_values = np.random.uniform(0, 2 * np.pi, n_params)

        # Shift theta[0] by +pi/2
        values_plus = random_values.copy()
        values_plus[0] += np.pi / 2

        # Shift theta[0] by -pi/2
        values_minus = random_values.copy()
        values_minus[0] -= np.pi / 2

        circuit_plus = qc.assign_parameters(dict(zip(theta, values_plus)))
        state_plus = Statevector.from_instruction(circuit_plus)
        cost_plus = state_plus.expectation_value(observable).real

        circuit_minus = qc.assign_parameters(dict(zip(theta, values_minus)))
        state_minus = Statevector.from_instruction(circuit_minus)
        cost_minus = state_minus.expectation_value(observable).real

        gradient = (cost_plus - cost_minus) / 2
        gradients.append(gradient)

    mean_gradient = np.mean(gradients)
    gradient_variance = np.var(gradients)

    print(f"\nn_qubits = {n_qubits}")
    print(f"  experiments      = {n_experiments}")
    print(f"  mean gradient    = {mean_gradient:.6e}")
    print(f"  gradient variance= {gradient_variance:.6e}")

    return mean_gradient, gradient_variance


def main():
    qubit_counts = []
    variances = []
    means = []

    for n in QUBIT_RANGE:
        mean_g, var_g = gradient_stats_for_n(n)
        qubit_counts.append(n)
        variances.append(var_g)
        means.append(mean_g)

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    csv_filename = "results_baseline_qubits.csv"
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["n_qubits", "mean_gradient", "gradient_variance"])
        for n, m, v in zip(qubit_counts, means, variances):
            print(f"  n={n}:  mean={m:.6e}, variance={v:.6e}")
            writer.writerow([n, m, v])
            
    print(f"\nSaved structured data to {csv_filename}")

    # ---- Plot: variance vs number of qubits ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Linear scale: this is the "steep downward curve" shape
    axes[0].plot(qubit_counts, variances, "o-", color="crimson", linewidth=2)
    axes[0].set_xlabel("Number of Qubits (n)")
    axes[0].set_ylabel("Gradient Variance")
    axes[0].set_title("Barren Plateau: Variance vs Qubits (linear scale)")
    axes[0].grid(True, alpha=0.3)

    # Log scale: confirms it's exponential decay, not just any curve
    axes[1].semilogy(qubit_counts, variances, "o-", color="navy", linewidth=2)
    axes[1].set_xlabel("Number of Qubits (n)")
    axes[1].set_ylabel("Gradient Variance (log scale)")
    axes[1].set_title("Barren Plateau: Variance vs Qubits (log scale)")
    axes[1].grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig("barren_plateau_qiskit.png", dpi=150)
    print("\nSaved plot: barren_plateau_qiskit.png")
    # plt.show()


if __name__ == "__main__":
    main()