"""
Validation Script: Adaptive Trigger Per-Trial Diagnostics
===========================================================
Performs a rigorous, documented diagnostic validation of the adaptive trigger
algorithm to extract per-trial quantities that were not saved in the original
aggregate CSV (results_adaptive_trigger.csv).

Preserves 100% of the algorithm from adaptive_trigger.py:
- n = 6 qubits, depths = 1..20
- Rx-Ry-Rz on each qubit + linear CNOT chain
- Observable: Z_0 on qubit 0 (SparsePauliOp)
- Parameter shift rule: (C(+pi/2) - C(-pi/2)) / 2
- Trigger condition: v_curr < 0.05 * v_ref (intra-layer variance across 18 components)
- Reinitialization: Normal(0, 0.1) on active layer upon triggering
- Freezing: Prior layers trained before trigger; frozen after trigger
- Learning rate = 0.5, 2 gradient descent steps
- Fixed documented random seed: seed = 42

Saves:
1. validation_adaptive_trials.csv (detailed per-trial, per-depth log)
2. validation_adaptive_summary.csv (depth-level aggregates)
3. validation_adaptive_diagnostics.png (trajectory of v_curr vs threshold per trial)
"""

import time
import csv
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, SparsePauliOp

RANDOM_SEED = 42
N_VALIDATION_TRIALS = 10   # Exactly matches trials 0..9 of the original experiment
FIXED_QUBITS = 6
MAX_DEPTH = 20
DEPTHS_TO_RECORD = [1, 2, 4, 8, 12, 16, 20]

LEARNING_RATE = 0.5
TRAIN_STEPS = 2
INIT_SIGMA = 0.1
TRIGGER_RATIO = 0.05

def build_circuit(params_list, n_qubits, n_layers):
    """Direct numeric QuantumCircuit construction for exact, high-performance simulation."""
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

def run_validation():
    print("=" * 70)
    print("STARTING ADAPTIVE TRIGGER DIAGNOSTIC VALIDATION")
    print(f"Random Seed: {RANDOM_SEED} | Validation Trials: {N_VALIDATION_TRIALS}")
    print("=" * 70)

    np.random.seed(RANDOM_SEED)
    observable = SparsePauliOp("I" * (FIXED_QUBITS - 1) + "Z")

    trial_log_records = []
    trial_trigger_depths = []

    start_time = time.time()

    for trial in range(N_VALIDATION_TRIALS):
        trial_t0 = time.time()
        all_params = []
        mitigation_active = False
        v_ref = 0.0
        triggered_at = -1

        for layer in range(1, MAX_DEPTH + 1):
            # 1. Initialize parameters
            if mitigation_active:
                new_params = list(np.random.normal(0, INIT_SIGMA, 3 * FIXED_QUBITS))
            else:
                new_params = list(np.random.uniform(0, 2 * np.pi, 3 * FIXED_QUBITS))

            all_params.extend(new_params)

            # 2. Compute the 18 gradients of the NEW layer
            def compute_active_layer_gradients(params):
                grads = np.zeros(3 * FIXED_QUBITS)
                base_idx = len(params) - 3 * FIXED_QUBITS
                for i in range(3 * FIXED_QUBITS):
                    p_idx = base_idx + i

                    vp = list(params)
                    vp[p_idx] += np.pi / 2
                    cp = Statevector.from_instruction(build_circuit(vp, FIXED_QUBITS, layer)).expectation_value(observable).real

                    vm = list(params)
                    vm[p_idx] -= np.pi / 2
                    cm = Statevector.from_instruction(build_circuit(vm, FIXED_QUBITS, layer)).expectation_value(observable).real

                    grads[i] = (cp - cm) / 2
                return grads

            layer_grads = compute_active_layer_gradients(all_params)
            v_curr_pre = float(np.var(layer_grads))

            # 3. Reference establishment at Layer 1
            if layer == 1:
                v_ref = v_curr_pre

            threshold = TRIGGER_RATIO * v_ref
            triggered_this_layer = False

            # 4. Trigger evaluation
            if (not mitigation_active) and (layer > 1):
                if v_curr_pre < threshold:
                    mitigation_active = True
                    triggered_this_layer = True
                    triggered_at = layer

                    # Reinitialize active layer with Normal(0, 0.1)
                    new_params = list(np.random.normal(0, INIT_SIGMA, 3 * FIXED_QUBITS))
                    all_params[-3 * FIXED_QUBITS:] = new_params

                    # Recalculate gradients
                    layer_grads = compute_active_layer_gradients(all_params)

            v_curr_effective = float(np.var(layer_grads))
            g0_component = float(layer_grads[0])

            # Consistency check
            is_consistent = True
            if triggered_this_layer and (v_curr_pre >= threshold):
                is_consistent = False
            if (not mitigation_active) and (layer > 1) and (v_curr_pre < threshold):
                is_consistent = False

            # Log record
            trial_log_records.append({
                "trial_id": trial,
                "layer": layer,
                "v_ref": v_ref,
                "threshold": threshold,
                "v_curr_pre": v_curr_pre,
                "v_curr_effective": v_curr_effective,
                "triggered_this_layer": triggered_this_layer,
                "mitigation_active": mitigation_active,
                "triggered_at": triggered_at,
                "g0_component": g0_component,
                "is_consistent": is_consistent
            })

            # 5. Training
            active_start = (len(all_params) - 3 * FIXED_QUBITS) if mitigation_active else 0
            active_end = len(all_params)

            for step in range(TRAIN_STEPS):
                active_grads = np.zeros(active_end - active_start)
                for i, p_idx in enumerate(range(active_start, active_end)):
                    vp = list(all_params)
                    vp[p_idx] += np.pi / 2
                    cp = Statevector.from_instruction(build_circuit(vp, FIXED_QUBITS, layer)).expectation_value(observable).real

                    vm = list(all_params)
                    vm[p_idx] -= np.pi / 2
                    cm = Statevector.from_instruction(build_circuit(vm, FIXED_QUBITS, layer)).expectation_value(observable).real

                    active_grads[i] = (cp - cm) / 2

                for i, p_idx in enumerate(range(active_start, active_end)):
                    all_params[p_idx] -= LEARNING_RATE * active_grads[i]

        if triggered_at != -1:
            trial_trigger_depths.append(triggered_at)

        print(f"Trial {trial+1:2d}/{N_VALIDATION_TRIALS} finished in {time.time()-trial_t0:.1f}s | Triggered at layer: {triggered_at}")

    print(f"\nAll {N_VALIDATION_TRIALS} trials completed in {time.time()-start_time:.1f}s.")

    # 1. Save detailed per-trial CSV
    trials_csv_path = "validation_adaptive_trials.csv"
    fieldnames = [
        "trial_id", "layer", "v_ref", "threshold",
        "v_curr_pre", "v_curr_effective", "triggered_this_layer",
        "mitigation_active", "triggered_at", "g0_component", "is_consistent"
    ]
    with open(trials_csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(trial_log_records)
    print(f"[SUCCESS] Saved detailed diagnostic per-trial log: {trials_csv_path}")

    # 2. Save depth summary CSV
    summary_csv_path = "validation_adaptive_summary.csv"
    summary_fieldnames = [
        "depth", "mean_v_ref", "mean_threshold", "mean_v_curr_effective",
        "cross_trial_var_g0", "mitigation_active_pct"
    ]
    summary_rows = []
    for d in DEPTHS_TO_RECORD:
        records_d = [r for r in trial_log_records if r["layer"] == d]
        mean_ref = np.mean([r["v_ref"] for r in records_d])
        mean_thresh = np.mean([r["threshold"] for r in records_d])
        mean_vcurr = np.mean([r["v_curr_effective"] for r in records_d])
        cross_var_g0 = np.var([r["g0_component"] for r in records_d])
        active_cnt = sum(1 for r in records_d if r["mitigation_active"])
        pct = (active_cnt / N_VALIDATION_TRIALS) * 100

        summary_rows.append({
            "depth": d,
            "mean_v_ref": mean_ref,
            "mean_threshold": mean_thresh,
            "mean_v_curr_effective": mean_vcurr,
            "cross_trial_var_g0": cross_var_g0,
            "mitigation_active_pct": pct
        })

    with open(summary_csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"[SUCCESS] Saved validation depth summary: {summary_csv_path}")

    # 3. Create diagnostic visualization
    plot_path = "validation_adaptive_diagnostics.png"
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Subplot 1: Per-trial trajectories of v_curr vs threshold
    for trial in range(N_VALIDATION_TRIALS):
        records_t = [r for r in trial_log_records if r["trial_id"] == trial]
        layers = [r["layer"] for r in records_t]
        v_currs = [r["v_curr_pre"] for r in records_t]
        thresh = records_t[0]["threshold"]
        trig_at = records_t[0]["triggered_at"]

        line = axes[0].plot(layers, v_currs, alpha=0.6, linewidth=1.2, label=f"Trial {trial} (Trig: L{trig_at})")
        axes[0].axhline(thresh, color=line[0].get_color(), linestyle=":", alpha=0.4)

    axes[0].set_yscale("log")
    axes[0].set_xlabel("Layer Depth")
    axes[0].set_ylabel("Within-Trial Active Layer Variance (v_curr, log scale)")
    axes[0].set_title("Per-Trial Within-Layer Variance Trajectories\n(Dotted line = Trial's own 5% threshold)")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=7, loc="upper right")

    # Subplot 2: Cross-trial variance vs Mitigation active rate
    depths = [s["depth"] for s in summary_rows]
    vars_ = [s["cross_trial_var_g0"] for s in summary_rows]
    pcts = [s["mitigation_active_pct"] for s in summary_rows]

    ax2_color = 'tab:blue'
    axes[1].set_xlabel("Circuit Depth")
    axes[1].set_ylabel("Cross-Trial Population Variance (Var(g0))", color=ax2_color)
    axes[1].plot(depths, vars_, "o-", color=ax2_color, linewidth=2, label="Validation Var(g0)")
    axes[1].tick_params(axis='y', labelcolor=ax2_color)
    axes[1].set_yscale("log")

    ax2_twin = axes[1].twinx()
    ax2_twin_color = 'tab:red'
    ax2_twin.set_ylabel("Mitigation Active Rate (%)", color=ax2_twin_color)
    ax2_twin.plot(depths, pcts, "s--", color=ax2_twin_color, linewidth=2, label="Active %")
    ax2_twin.tick_params(axis='y', labelcolor=ax2_twin_color)
    ax2_twin.set_ylim(-5, 105)

    axes[1].set_title("Validation: Cross-Trial Variance vs Active %")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    print(f"[SUCCESS] Saved diagnostic plots: {plot_path}")

    # Summary prints
    print("\nVALIDATION SUMMARY:")
    print(f"Trials Triggered: {len(trial_trigger_depths)}/{N_VALIDATION_TRIALS} ({(len(trial_trigger_depths)/N_VALIDATION_TRIALS)*100:.1f}%)")
    if trial_trigger_depths:
        print(f"Trigger Depths: {trial_trigger_depths}")
        print(f"Average Trigger Depth: {np.mean(trial_trigger_depths):.2f}")

if __name__ == "__main__":
    run_validation()
