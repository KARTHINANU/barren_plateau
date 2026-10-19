import csv
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector, SparsePauliOp

np.random.seed(42)

N_EXPERIMENTS = 250
QUBIT_RANGE = range(2, 8)
N_LAYERS = 20

def build_circuit(n_qubits, n_layers=N_LAYERS):
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
    return SparsePauliOp("I" * (n_qubits - 1) + "Z")

def gradient_stats_for_n(n_qubits, n_experiments=N_EXPERIMENTS):
    qc, theta = build_circuit(n_qubits)
    observable = observable_for(n_qubits)
    n_params = 3 * n_qubits * N_LAYERS

    gradients = []

    for i in range(n_experiments):
        # MITIGATION CHANGE: Normal(0, 0.1) instead of Uniform(0, 2*pi)
        random_values = np.random.normal(0, 0.1, n_params)

        # Shift theta[0] by +pi/2
        values_plus = random_values.copy()
        values_plus[0] += np.pi / 2
        circuit_plus = qc.assign_parameters(dict(zip(theta, values_plus)))
        cost_plus = Statevector.from_instruction(circuit_plus).expectation_value(observable).real

        # Shift theta[0] by -pi/2
        values_minus = random_values.copy()
        values_minus[0] -= np.pi / 2
        circuit_minus = qc.assign_parameters(dict(zip(theta, values_minus)))
        cost_minus = Statevector.from_instruction(circuit_minus).expectation_value(observable).real

        gradient = (cost_plus - cost_minus) / 2
        gradients.append(gradient)

    mean_gradient = np.mean(gradients)
    gradient_variance = np.var(gradients)
    gradient_std = np.std(gradients)

    return mean_gradient, gradient_variance, gradient_std

def main():
    qubit_counts = []
    variances = []
    means = []
    stds = []

    print("\n" + "=" * 50)
    print("MITIGATION EXPERIMENT: CONTROLLED INITIALIZATION")
    print("=" * 50)
    
    for n in QUBIT_RANGE:
        mean_g, var_g, std_g = gradient_stats_for_n(n)
        qubit_counts.append(n)
        variances.append(var_g)
        means.append(mean_g)
        stds.append(std_g)
        print(f"n={n}: mean={mean_g:.6e}, var={var_g:.6e}, std={std_g:.6e}")

    csv_filename = "results_mitigation_controlled_init.csv"
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["n_qubits", "mean_gradient", "gradient_variance", "standard_deviation"])
        for n, m, v, s in zip(qubit_counts, means, variances, stds):
            writer.writerow([n, m, v, s])
            
    print(f"\nSaved structured data to {csv_filename}")

    # Load baseline for comparison plot
    baseline_var = []
    try:
        with open("results_baseline_qubits.csv", "r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                baseline_var.append(float(row["gradient_variance"]))
    except:
        print("Could not load baseline data for plotting.")
        pass

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    axes[0].plot(qubit_counts, variances, "o-", color="green", linewidth=2, label="Mitigated (Normal(0, 0.1))")
    if baseline_var:
        axes[0].plot(qubit_counts, baseline_var, "x--", color="crimson", linewidth=1.5, label="Baseline (Uniform)")
    axes[0].set_xlabel("Number of Qubits (n)")
    axes[0].set_ylabel("Gradient Variance")
    axes[0].set_title("Variance vs Qubits (Linear)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogy(qubit_counts, variances, "o-", color="limegreen", linewidth=2, label="Mitigated")
    if baseline_var:
        axes[1].semilogy(qubit_counts, baseline_var, "x--", color="navy", linewidth=1.5, label="Baseline")
    axes[1].set_xlabel("Number of Qubits (n)")
    axes[1].set_ylabel("Gradient Variance (log scale)")
    axes[1].set_title("Variance vs Qubits (Log)")
    axes[1].legend()
    axes[1].grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig("mitigation_controlled_init.png", dpi=150)
    print("Saved plot: mitigation_controlled_init.png")

if __name__ == "__main__":
    main()
