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
INIT_SIGMA = 0.1

def observable_for(n_qubits):
    return SparsePauliOp("I" * (n_qubits - 1) + "Z")

def run_adaptive_experiment():
    observable = observable_for(FIXED_QUBITS)
    gradients_at_depth = {d: [] for d in DEPTHS_TO_MEASURE}
    
    print("\n" + "=" * 50)
    print("PHASE 5: ADAPTIVE MITIGATION (Layer-wise + Normal Init)")
    print("=" * 50)
    print(f"Qubits: {FIXED_QUBITS}")
    print(f"Trials: {N_EXPERIMENTS}")
    print(f"Train Steps per Layer: {TRAIN_STEPS}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Init Distribution: Normal(0, {INIT_SIGMA})")
    
    start_time = time.time()
    
    for trial in range(N_EXPERIMENTS):
        current_qc = QuantumCircuit(FIXED_QUBITS)
        
        for layer in range(1, MAX_DEPTH + 1):
            layer_theta = ParameterVector(f"theta_l{layer}", 3 * FIXED_QUBITS)
            idx = 0
            for q in range(FIXED_QUBITS):
                current_qc.rx(layer_theta[idx], q); idx += 1
                current_qc.ry(layer_theta[idx], q); idx += 1
                current_qc.rz(layer_theta[idx], q); idx += 1
            for q in range(FIXED_QUBITS - 1):
                current_qc.cx(q, q + 1)
                
            # ADAPTIVE CHANGE: Initialize new layer with Normal(0, 0.1) instead of Uniform
            new_params = np.random.normal(0, INIT_SIGMA, 3 * FIXED_QUBITS)
            
            for step in range(TRAIN_STEPS):
                layer_gradients = np.zeros(3 * FIXED_QUBITS)
                for i in range(3 * FIXED_QUBITS):
                    v_plus = new_params.copy()
                    v_plus[i] += np.pi / 2
                    cp = Statevector.from_instruction(current_qc.assign_parameters(dict(zip(layer_theta, v_plus)))).expectation_value(observable).real
                    
                    v_minus = new_params.copy()
                    v_minus[i] -= np.pi / 2
                    cm = Statevector.from_instruction(current_qc.assign_parameters(dict(zip(layer_theta, v_minus)))).expectation_value(observable).real
                    
                    layer_gradients[i] = (cp - cm) / 2
                
                new_params -= LEARNING_RATE * layer_gradients
                
            if layer in DEPTHS_TO_MEASURE:
                v_plus = new_params.copy()
                v_plus[0] += np.pi / 2
                cp = Statevector.from_instruction(current_qc.assign_parameters(dict(zip(layer_theta, v_plus)))).expectation_value(observable).real
                
                v_minus = new_params.copy()
                v_minus[0] -= np.pi / 2
                cm = Statevector.from_instruction(current_qc.assign_parameters(dict(zip(layer_theta, v_minus)))).expectation_value(observable).real
                
                grad = (cp - cm) / 2
                gradients_at_depth[layer].append(grad)
                
            current_qc = current_qc.assign_parameters(dict(zip(layer_theta, new_params)))
            
        if (trial + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"  Completed trial {trial + 1}/{N_EXPERIMENTS} ({elapsed:.1f}s)")

    depths = []
    means = []
    variances = []
    stds = []
    
    csv_filename = "results_adaptive_mitigation.csv"
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["depth", "mean_gradient", "gradient_variance", "standard_deviation"])
        
        for d in DEPTHS_TO_MEASURE:
            grads = gradients_at_depth[d]
            mean_g = np.mean(grads)
            var_g = np.var(grads)
            std_g = np.std(grads)
            
            depths.append(d)
            means.append(mean_g)
            variances.append(var_g)
            stds.append(std_g)
            
            print(f"depth={d:2d}: mean={mean_g:.6e}, variance={var_g:.6e}, std={std_g:.6e}")
            writer.writerow([d, mean_g, var_g, std_g])
            
    print(f"\nSaved structured data to {csv_filename}")

    # Load comparison data
    baseline_var = []
    try:
        with open("results_baseline_depth.csv", "r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                baseline_var.append(float(row["gradient_variance"]))
    except:
        pass
        
    layerwise_var = []
    try:
        with open("results_mitigation_layerwise_corrected.csv", "r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                layerwise_var.append(float(row["gradient_variance"]))
    except:
        pass

    # Quick calculation of Controlled Init (across depths) to fulfill plotting requirement
    controlled_init_var = []
    print("\nCalculating Controlled Init data for plotting comparison...")
    for d in DEPTHS_TO_MEASURE:
        grads = []
        for _ in range(N_EXPERIMENTS):
            qc = QuantumCircuit(FIXED_QUBITS)
            theta = ParameterVector(f"t", 3 * FIXED_QUBITS * d)
            idx = 0
            for layer in range(d):
                for q in range(FIXED_QUBITS):
                    qc.rx(theta[idx], q); idx+=1
                    qc.ry(theta[idx], q); idx+=1
                    qc.rz(theta[idx], q); idx+=1
                for q in range(FIXED_QUBITS - 1):
                    qc.cx(q, q+1)
            
            rv = np.random.normal(0, INIT_SIGMA, 3 * FIXED_QUBITS * d)
            vp = rv.copy(); vp[0] += np.pi / 2
            cp = Statevector.from_instruction(qc.assign_parameters(vp)).expectation_value(observable).real
            vm = rv.copy(); vm[0] -= np.pi / 2
            cm = Statevector.from_instruction(qc.assign_parameters(vm)).expectation_value(observable).real
            grads.append((cp - cm) / 2)
        controlled_init_var.append(np.var(grads))

    # Calculate improvement factor
    print("\nIMPROVEMENT FACTOR (Adaptive / Baseline):")
    for i, d in enumerate(depths):
        if i < len(baseline_var):
            factor = variances[i] / baseline_var[i]
            print(f"Depth {d:2d}: {factor:.2f}x")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    axes[0].plot(depths, variances, "o-", color="red", linewidth=2.5, label="Adaptive (Layerwise + Init)")
    if baseline_var:
        axes[0].plot(depths, baseline_var, "x--", color="gray", linewidth=1.5, label="Standard Baseline")
    if controlled_init_var:
        axes[0].plot(depths, controlled_init_var, "s--", color="green", linewidth=1.5, label="Controlled Init Only")
    if layerwise_var:
        axes[0].plot(depths, layerwise_var, "^--", color="purple", linewidth=1.5, label="Layerwise Only")
        
    axes[0].set_xlabel("Circuit Depth (Layers)")
    axes[0].set_ylabel("Gradient Variance")
    axes[0].set_title(f"Adaptive Mitigation Comparison (n={FIXED_QUBITS}) - Linear")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].semilogy(depths, variances, "o-", color="darkred", linewidth=2.5, label="Adaptive")
    if baseline_var:
        axes[1].semilogy(depths, baseline_var, "x--", color="gray", linewidth=1.5, label="Standard Baseline")
    if controlled_init_var:
        axes[1].semilogy(depths, controlled_init_var, "s--", color="green", linewidth=1.5, label="Controlled Init Only")
    if layerwise_var:
        axes[1].semilogy(depths, layerwise_var, "^--", color="purple", linewidth=1.5, label="Layerwise Only")
        
    axes[1].set_xlabel("Circuit Depth (Layers)")
    axes[1].set_ylabel("Gradient Variance (log scale)")
    axes[1].set_title(f"Adaptive Mitigation Comparison (n={FIXED_QUBITS}) - Log")
    axes[1].legend()
    axes[1].grid(True, which="both", alpha=0.3)
    
    fig.tight_layout()
    fig.savefig("adaptive_mitigation.png", dpi=150)
    print("\nSaved plot: adaptive_mitigation.png")

if __name__ == "__main__":
    run_adaptive_experiment()
