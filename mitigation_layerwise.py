import csv
import time
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector, SparsePauliOp

np.random.seed(42)

# Reduced to 100 because full gradient descent in an exact simulator is computationally heavy
N_EXPERIMENTS = 100   
FIXED_QUBITS = 6
DEPTHS_TO_MEASURE = [1, 2, 4, 8, 12, 16, 20]
MAX_DEPTH = 20

LEARNING_RATE = 0.5
TRAIN_STEPS = 2       # Small number of steps per layer to keep compute time reasonable

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

def run_layerwise_experiment():
    qc, theta = build_circuit(FIXED_QUBITS, MAX_DEPTH)
    observable = observable_for(FIXED_QUBITS)
    
    # We will track the gradient of the FIRST parameter of the newly added layer
    gradients_at_depth = {d: [] for d in DEPTHS_TO_MEASURE}
    
    print("\n" + "=" * 50)
    print("MITIGATION EXPERIMENT: ACTUAL LAYER-WISE TRAINING")
    print("=" * 50)
    print(f"Qubits: {FIXED_QUBITS}")
    print(f"Trials: {N_EXPERIMENTS}")
    print(f"Train Steps per Layer: {TRAIN_STEPS}")
    print(f"Learning Rate: {LEARNING_RATE}")
    
    start_time = time.time()
    
    for trial in range(N_EXPERIMENTS):
        # Initialize all parameters uniformly (0, 2pi)
        current_params = np.random.uniform(0, 2 * np.pi, 3 * FIXED_QUBITS * MAX_DEPTH)
        
        for layer in range(1, MAX_DEPTH + 1):
            # Indices for the current active layer
            # Each layer has 3 * FIXED_QUBITS parameters
            layer_start = (layer - 1) * 3 * FIXED_QUBITS
            layer_end = layer * 3 * FIXED_QUBITS
            
            # Record the gradient of the FIRST parameter in this new layer
            # BEFORE we train this layer (this shows the gradient variance we face when starting to train it)
            if layer in DEPTHS_TO_MEASURE:
                v_plus = current_params.copy()
                v_plus[layer_start] += np.pi / 2
                cost_plus = Statevector.from_instruction(qc.assign_parameters(dict(zip(theta, v_plus)))).expectation_value(observable).real
                
                v_minus = current_params.copy()
                v_minus[layer_start] -= np.pi / 2
                cost_minus = Statevector.from_instruction(qc.assign_parameters(dict(zip(theta, v_minus)))).expectation_value(observable).real
                
                grad = (cost_plus - cost_minus) / 2
                gradients_at_depth[layer].append(grad)
            
            # Train the current layer (gradient descent)
            for step in range(TRAIN_STEPS):
                layer_gradients = np.zeros(3 * FIXED_QUBITS)
                # Calculate gradient for each parameter in the active layer
                for i, param_idx in enumerate(range(layer_start, layer_end)):
                    v_plus = current_params.copy()
                    v_plus[param_idx] += np.pi / 2
                    cp = Statevector.from_instruction(qc.assign_parameters(dict(zip(theta, v_plus)))).expectation_value(observable).real
                    
                    v_minus = current_params.copy()
                    v_minus[param_idx] -= np.pi / 2
                    cm = Statevector.from_instruction(qc.assign_parameters(dict(zip(theta, v_minus)))).expectation_value(observable).real
                    
                    layer_gradients[i] = (cp - cm) / 2
                
                # Update current layer parameters (theta = theta - lr * grad)
                current_params[layer_start:layer_end] -= LEARNING_RATE * layer_gradients
                
        if (trial + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"  Completed trial {trial + 1}/{N_EXPERIMENTS} ({elapsed:.1f}s)")

    # Calculate statistics
    depths = []
    means = []
    variances = []
    
    csv_filename = "results_mitigation_layerwise.csv"
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
    
    axes[0].plot(depths, variances, "o-", color="purple", linewidth=2, label="Layer-wise Training")
    if baseline_var:
        axes[0].plot(depths, baseline_var, "x--", color="gray", linewidth=1.5, label="Standard Baseline")
    axes[0].set_xlabel("Circuit Depth (Layers)")
    axes[0].set_ylabel("Gradient Variance")
    axes[0].set_title(f"Variance vs Depth (n={FIXED_QUBITS}) - Linear")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].semilogy(depths, variances, "o-", color="darkmagenta", linewidth=2, label="Layer-wise")
    if baseline_var:
        axes[1].semilogy(depths, baseline_var, "x--", color="gray", linewidth=1.5, label="Standard Baseline")
    axes[1].set_xlabel("Circuit Depth (Layers)")
    axes[1].set_ylabel("Gradient Variance (log scale)")
    axes[1].set_title(f"Variance vs Depth (n={FIXED_QUBITS}) - Log")
    axes[1].legend()
    axes[1].grid(True, which="both", alpha=0.3)
    
    fig.tight_layout()
    fig.savefig("mitigation_layerwise.png", dpi=150)
    print("Saved plot: mitigation_layerwise.png")

if __name__ == "__main__":
    run_layerwise_experiment()
