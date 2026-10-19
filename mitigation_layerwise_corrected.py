import csv
import time
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector, SparsePauliOp

np.random.seed(42)

N_EXPERIMENTS = 100
FIXED_QUBITS = 6
DEPTHS_TO_MEASURE = [1, 2, 4, 8, 12, 16, 20]
MAX_DEPTH = 20

LEARNING_RATE = 0.5
TRAIN_STEPS = 2

def observable_for(n_qubits):
    return SparsePauliOp("I" * (n_qubits - 1) + "Z")

def run_layerwise_corrected_experiment():
    observable = observable_for(FIXED_QUBITS)
    gradients_at_depth = {d: [] for d in DEPTHS_TO_MEASURE}
    
    print("\n" + "=" * 50)
    print("MITIGATION EXPERIMENT: TRUE SEQUENTIAL LAYER-WISE TRAINING")
    print("=" * 50)
    print(f"Qubits: {FIXED_QUBITS}")
    print(f"Trials: {N_EXPERIMENTS}")
    print(f"Train Steps per Layer: {TRAIN_STEPS}")
    print(f"Learning Rate: {LEARNING_RATE}")
    
    start_time = time.time()
    
    for trial in range(N_EXPERIMENTS):
        # 1. Start with a circuit containing ONLY Layer 1 (empty initially, built sequentially)
        current_qc = QuantumCircuit(FIXED_QUBITS)
        
        for layer in range(1, MAX_DEPTH + 1):
            # 5. Add Layer `layer` to the existing circuit
            layer_theta = ParameterVector(f"theta_l{layer}", 3 * FIXED_QUBITS)
            idx = 0
            for q in range(FIXED_QUBITS):
                current_qc.rx(layer_theta[idx], q); idx += 1
                current_qc.ry(layer_theta[idx], q); idx += 1
                current_qc.rz(layer_theta[idx], q); idx += 1
            for q in range(FIXED_QUBITS - 1):
                current_qc.cx(q, q + 1)
                
            # 7. Initialize only the newly added Layer
            # (Uniform 0 to 2pi as requested)
            new_params = np.random.uniform(0, 2 * np.pi, 3 * FIXED_QUBITS)
            
            # 8. Train only the newly added Layer
            for step in range(TRAIN_STEPS):
                layer_gradients = np.zeros(3 * FIXED_QUBITS)
                # Calculate gradient for each parameter in the active layer
                for i in range(3 * FIXED_QUBITS):
                    v_plus = new_params.copy()
                    v_plus[i] += np.pi / 2
                    cp = Statevector.from_instruction(current_qc.assign_parameters(dict(zip(layer_theta, v_plus)))).expectation_value(observable).real
                    
                    v_minus = new_params.copy()
                    v_minus[i] -= np.pi / 2
                    cm = Statevector.from_instruction(current_qc.assign_parameters(dict(zip(layer_theta, v_minus)))).expectation_value(observable).real
                    
                    layer_gradients[i] = (cp - cm) / 2
                
                # Update current layer parameters
                new_params -= LEARNING_RATE * layer_gradients
                
            # 9. Record the gradient statistics for the trained/active layer
            # We measure the gradient of the first parameter in this layer after training
            if layer in DEPTHS_TO_MEASURE:
                v_plus = new_params.copy()
                v_plus[0] += np.pi / 2
                cp = Statevector.from_instruction(current_qc.assign_parameters(dict(zip(layer_theta, v_plus)))).expectation_value(observable).real
                
                v_minus = new_params.copy()
                v_minus[0] -= np.pi / 2
                cm = Statevector.from_instruction(current_qc.assign_parameters(dict(zip(layer_theta, v_minus)))).expectation_value(observable).real
                
                grad = (cp - cm) / 2
                gradients_at_depth[layer].append(grad)
                
            # 6. Keep previously trained layers fixed
            # By assigning the trained parameters, they become fixed floats in `current_qc`.
            # In the next iteration, they will not be Parameter objects anymore.
            current_qc = current_qc.assign_parameters(dict(zip(layer_theta, new_params)))
            
        if (trial + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"  Completed trial {trial + 1}/{N_EXPERIMENTS} ({elapsed:.1f}s)")

    # Calculate statistics
    depths = []
    means = []
    variances = []
    
    csv_filename = "results_mitigation_layerwise_corrected.csv"
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["depth", "mean_gradient", "gradient_variance"])
        
        for d in DEPTHS_TO_MEASURE:
            grads = gradients_at_depth[d]
            mean_g = np.mean(grads)
            var_g = np.var(grads)
            
            depths.append(d)
            means.append(mean_g)
            variances.append(var_g)
            
            print(f"depth={d:2d}: mean={mean_g:.6e}, variance={var_g:.6e}")
            writer.writerow([d, mean_g, var_g])
            
    print(f"\nSaved structured data to {csv_filename}")

    # Load baseline for comparison
    baseline_var = []
    try:
        with open("results_baseline_depth.csv", "r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                baseline_var.append(float(row["gradient_variance"]))
    except:
        print("Could not load baseline depth data for plotting.")

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    axes[0].plot(depths, variances, "o-", color="purple", linewidth=2, label="Corrected Layer-wise")
    if baseline_var:
        axes[0].plot(depths, baseline_var, "x--", color="gray", linewidth=1.5, label="Standard Baseline")
    axes[0].set_xlabel("Circuit Total Depth (Layers)")
    axes[0].set_ylabel("Gradient Variance (Active Layer)")
    axes[0].set_title(f"Variance vs Depth (n={FIXED_QUBITS}) - Linear")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].semilogy(depths, variances, "o-", color="darkmagenta", linewidth=2, label="Corrected Layer-wise")
    if baseline_var:
        axes[1].semilogy(depths, baseline_var, "x--", color="gray", linewidth=1.5, label="Standard Baseline")
    axes[1].set_xlabel("Circuit Total Depth (Layers)")
    axes[1].set_ylabel("Gradient Variance (log scale)")
    axes[1].set_title(f"Variance vs Depth (n={FIXED_QUBITS}) - Log")
    axes[1].legend()
    axes[1].grid(True, which="both", alpha=0.3)
    
    fig.tight_layout()
    fig.savefig("mitigation_layerwise_corrected.png", dpi=150)
    print("Saved plot: mitigation_layerwise_corrected.png")

if __name__ == "__main__":
    run_layerwise_corrected_experiment()
