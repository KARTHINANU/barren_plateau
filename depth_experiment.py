import csv
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector, SparsePauliOp

np.random.seed(42)

N_EXPERIMENTS = 250
FIXED_QUBITS = 6
DEPTHS = [1, 2, 4, 8, 12, 16, 20]

def build_circuit(n_qubits, n_layers):
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

def run_depth_experiment():
    observable = observable_for(FIXED_QUBITS)
    
    means = []
    variances = []
    
    csv_filename = "results_baseline_depth.csv"
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["depth", "mean_gradient", "gradient_variance"])
        
        print("\n" + "=" * 50)
        print(f"DEPTH EXPERIMENT (Fixed at {FIXED_QUBITS} qubits)")
        print("=" * 50)
        
        for depth in DEPTHS:
            qc, theta = build_circuit(FIXED_QUBITS, depth)
            n_params = 3 * FIXED_QUBITS * depth
            
            gradients = []
            for i in range(N_EXPERIMENTS):
                random_values = np.random.uniform(0, 2 * np.pi, n_params)
                
                v_plus = random_values.copy()
                v_plus[0] += np.pi / 2
                c_plus = qc.assign_parameters(dict(zip(theta, v_plus)))
                cost_plus = Statevector.from_instruction(c_plus).expectation_value(observable).real
                
                v_minus = random_values.copy()
                v_minus[0] -= np.pi / 2
                c_minus = qc.assign_parameters(dict(zip(theta, v_minus)))
                cost_minus = Statevector.from_instruction(c_minus).expectation_value(observable).real
                
                gradients.append((cost_plus - cost_minus) / 2)
                
            var_g = np.var(gradients)
            mean_g = np.mean(gradients)
            
            means.append(mean_g)
            variances.append(var_g)
            
            print(f"  depth={depth:2d}: mean={mean_g:.6e}, variance={var_g:.6e}")
            writer.writerow([depth, mean_g, var_g])
            
    print(f"\nSaved structured data to {csv_filename}")
    
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    axes[0].plot(DEPTHS, variances, "o-", color="purple", linewidth=2)
    axes[0].set_xlabel("Circuit Depth (Layers)")
    axes[0].set_ylabel("Gradient Variance")
    axes[0].set_title(f"Variance vs Depth ({FIXED_QUBITS} qubits) - Linear")
    axes[0].grid(True, alpha=0.3)
    
    axes[1].semilogy(DEPTHS, variances, "o-", color="darkorange", linewidth=2)
    axes[1].set_xlabel("Circuit Depth (Layers)")
    axes[1].set_ylabel("Gradient Variance (log scale)")
    axes[1].set_title(f"Variance vs Depth ({FIXED_QUBITS} qubits) - Log")
    axes[1].grid(True, which="both", alpha=0.3)
    
    fig.tight_layout()
    fig.savefig("barren_plateau_depth.png", dpi=150)
    print("Saved plot: barren_plateau_depth.png")

if __name__ == "__main__":
    run_depth_experiment()
