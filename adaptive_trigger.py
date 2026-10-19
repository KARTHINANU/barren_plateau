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
TRIGGER_RATIO = 0.05

def observable_for(n_qubits):
    return SparsePauliOp("I" * (n_qubits - 1) + "Z")

def run_adaptive_trigger_experiment():
    observable = observable_for(FIXED_QUBITS)
    
    print("\n" + "=" * 50)
    print("PHASE 5: PER-TRIAL ADAPTIVE DECISION-RULE EXPERIMENT")
    print("=" * 50)
    print(f"Qubits: {FIXED_QUBITS}, Trials: {N_EXPERIMENTS}")
    print(f"Trigger: Variance of active layer < {TRIGGER_RATIO*100}% of Depth-1 Variance")
    
    start_time = time.time()
    
    # To store population statistics at measurement depths
    population_stats = {d: {"v_currs": [], "v_refs": [], "mitigated_count": 0} for d in DEPTHS_TO_MEASURE}
    trigger_depths = []
    
    for trial in range(N_EXPERIMENTS):
        qc = QuantumCircuit(FIXED_QUBITS)
        all_params = []
        all_thetas = []
        
        mitigation_active = False
        v_ref = 0
        triggered_at = -1
        
        for layer in range(1, MAX_DEPTH + 1):
            layer_theta = ParameterVector(f"t_{layer}_{trial}", 3 * FIXED_QUBITS)
            all_thetas.extend(layer_theta)
            
            idx = 0
            for q in range(FIXED_QUBITS):
                qc.rx(layer_theta[idx], q); idx += 1
                qc.ry(layer_theta[idx], q); idx += 1
                qc.rz(layer_theta[idx], q); idx += 1
            for q in range(FIXED_QUBITS - 1):
                qc.cx(q, q + 1)
                
            if mitigation_active:
                new_params = np.random.normal(0, INIT_SIGMA, 3 * FIXED_QUBITS)
            else:
                new_params = np.random.uniform(0, 2 * np.pi, 3 * FIXED_QUBITS)
                
            all_params.extend(new_params)
            
            # Helper to calculate the 18 gradients of the NEW layer
            def get_layer_gradients():
                grads = np.zeros(3 * FIXED_QUBITS)
                base_idx = len(all_params) - 3 * FIXED_QUBITS
                for i in range(3 * FIXED_QUBITS):
                    p_idx = base_idx + i
                    vp = np.array(all_params).copy(); vp[p_idx] += np.pi / 2
                    cp = Statevector.from_instruction(qc.assign_parameters(dict(zip(all_thetas, vp)))).expectation_value(observable).real
                    vm = np.array(all_params).copy(); vm[p_idx] -= np.pi / 2
                    cm = Statevector.from_instruction(qc.assign_parameters(dict(zip(all_thetas, vm)))).expectation_value(observable).real
                    grads[i] = (cp - cm) / 2
                return grads

            layer_grads = get_layer_gradients()
            v_curr = np.var(layer_grads)
            
            if layer == 1:
                v_ref = v_curr
                
            if not mitigation_active and layer > 1:
                if v_curr < TRIGGER_RATIO * v_ref:
                    mitigation_active = True
                    triggered_at = layer
                    # Switch init to Normal(0, 0.1)
                    new_params = np.random.normal(0, INIT_SIGMA, 3 * FIXED_QUBITS)
                    all_params[-3 * FIXED_QUBITS:] = new_params
                    # Recalculate gradients with new init
                    layer_grads = get_layer_gradients()
                    v_curr = np.var(layer_grads)
            
            # Record measurement
            if layer in DEPTHS_TO_MEASURE:
                # We record the gradient of the FIRST parameter for population statistics
                population_stats[layer]["v_currs"].append(layer_grads[0])
                population_stats[layer]["v_refs"].append(v_ref)
                if mitigation_active:
                    population_stats[layer]["mitigated_count"] += 1
                    
            # Train active parameters
            active_start = (len(all_params) - 3 * FIXED_QUBITS) if mitigation_active else 0
            active_end = len(all_params)
            
            for step in range(TRAIN_STEPS):
                active_grads = np.zeros(active_end - active_start)
                for i, p_idx in enumerate(range(active_start, active_end)):
                    vp = np.array(all_params).copy(); vp[p_idx] += np.pi / 2
                    cp = Statevector.from_instruction(qc.assign_parameters(dict(zip(all_thetas, vp)))).expectation_value(observable).real
                    vm = np.array(all_params).copy(); vm[p_idx] -= np.pi / 2
                    cm = Statevector.from_instruction(qc.assign_parameters(dict(zip(all_thetas, vm)))).expectation_value(observable).real
                    active_grads[i] = (cp - cm) / 2
                
                for i, p_idx in enumerate(range(active_start, active_end)):
                    all_params[p_idx] -= LEARNING_RATE * active_grads[i]
                    
        if triggered_at != -1:
            trigger_depths.append(triggered_at)
            
        if (trial + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"  Completed trial {trial + 1}/{N_EXPERIMENTS} ({elapsed:.1f}s)")

    # Analyze trigger stats
    trigger_pct = (len(trigger_depths) / N_EXPERIMENTS) * 100
    avg_trigger_depth = np.mean(trigger_depths) if trigger_depths else -1
    print("\nTRIGGER STATISTICS:")
    print(f"Trials triggered: {len(trigger_depths)}/{N_EXPERIMENTS} ({trigger_pct:.1f}%)")
    print(f"Average trigger depth: {avg_trigger_depth:.2f}")
    
    # Save results
    results = []
    csv_filename = "results_adaptive_trigger.csv"
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["depth", "current_variance", "depth_1_reference", "trigger_threshold", "mitigation_active_pct"])
        
        for d in DEPTHS_TO_MEASURE:
            pop_var = np.var(population_stats[d]["v_currs"])
            avg_ref = np.mean(population_stats[d]["v_refs"])
            mit_pct = (population_stats[d]["mitigated_count"] / N_EXPERIMENTS) * 100
            
            results.append({
                "depth": d,
                "var": pop_var,
                "ref": avg_ref,
                "thresh": TRIGGER_RATIO * avg_ref,
                "pct": mit_pct
            })
            writer.writerow([d, pop_var, avg_ref, TRIGGER_RATIO * avg_ref, mit_pct])
            print(f"Depth {d:2d}: Var={pop_var:.6e} | Mitigated={mit_pct}%")

    print(f"\nSaved structured data to {csv_filename}")

    # Plotting
    depths = [r["depth"] for r in results]
    variances = [r["var"] for r in results]
    
    baseline_var = []
    try:
        with open("results_baseline_depth.csv", "r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                baseline_var.append(float(row["gradient_variance"]))
    except:
        pass
        
    fixed_adaptive_var = []
    try:
        with open("results_adaptive_mitigation.csv", "r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                fixed_adaptive_var.append(float(row["gradient_variance"]))
    except:
        pass

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    axes[0].plot(depths, variances, "o-", color="blue", linewidth=2.5, label="Dynamic Trigger (Empirical)")
    if baseline_var:
        axes[0].plot(depths, baseline_var, "x--", color="gray", linewidth=1.5, label="Standard Baseline")
    if fixed_adaptive_var:
        axes[0].plot(depths, fixed_adaptive_var, "s--", color="red", linewidth=1.5, label="Fixed Adaptive")
        
    axes[0].set_xlabel("Circuit Depth (Layers)")
    axes[0].set_ylabel("Gradient Variance")
    axes[0].set_title(f"Dynamic Adaptive Rule (n={FIXED_QUBITS}) - Linear")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].semilogy(depths, variances, "o-", color="darkblue", linewidth=2.5, label="Dynamic Trigger")
    if baseline_var:
        axes[1].semilogy(depths, baseline_var, "x--", color="gray", linewidth=1.5, label="Standard Baseline")
    if fixed_adaptive_var:
        axes[1].semilogy(depths, fixed_adaptive_var, "s--", color="red", linewidth=1.5, label="Fixed Adaptive")
        
    axes[1].set_xlabel("Circuit Depth (Layers)")
    axes[1].set_ylabel("Gradient Variance (log scale)")
    axes[1].set_title(f"Dynamic Adaptive Rule (n={FIXED_QUBITS}) - Log")
    axes[1].legend()
    axes[1].grid(True, which="both", alpha=0.3)
    
    fig.tight_layout()
    fig.savefig("adaptive_trigger.png", dpi=150)
    print("\nSaved plot: adaptive_trigger.png")

if __name__ == "__main__":
    run_adaptive_trigger_experiment()
