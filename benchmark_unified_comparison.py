"""
Benchmark Unified Comparison & Methodological Audit
====================================================
Audits and compares existing depth-experiment CSV results:
- results_baseline_depth.csv
- results_mitigation_layerwise_corrected.csv
- results_adaptive_mitigation.csv
- results_adaptive_trigger.csv

Explicitly documents methodology differences, parameter scopes,
measurement timing (pre- vs post-training), and statistic definitions.

Generates:
1. benchmark_unified_comparison.csv
2. benchmark_unified_comparison.png (multi-panel, methodology-aware visualization)
3. benchmark_unified_comparison_report.txt
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt

def load_csv_data(filepath):
    """Safely loads CSV data if the file exists."""
    if not os.path.exists(filepath):
        print(f"[WARNING] File not found: {filepath}")
        return None
    data = []
    with open(filepath, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data

def build_comparison_records():
    records = []

    # 1. Baseline Depth
    baseline_file = "results_baseline_depth.csv"
    baseline_data = load_csv_data(baseline_file)
    if baseline_data:
        for row in baseline_data:
            d = int(row["depth"])
            var = float(row["gradient_variance"])
            records.append({
                "method_name": "Baseline Depth (Uniform)",
                "circuit_depth": d,
                "reported_gradient_variance": var,
                "trial_count": 250,
                "measurement_timing": "At Initialization (0 training steps)",
                "parameter_measured": "Layer 0, Qubit 0, Rx (theta[0])",
                "initialization_protocol": "Uniform(0, 2*pi)",
                "training_protocol": "No training; Full circuit initialized at once; No freezing",
                "directly_comparable_to_baseline": "Self (Reference)",
                "comparability_notes": "Measures earliest parameter (Layer 0) followed by (L-1) scrambling layers at initialization."
            })

    # 2. Corrected Layer-Wise
    layerwise_file = "results_mitigation_layerwise_corrected.csv"
    layerwise_data = load_csv_data(layerwise_file)
    if layerwise_data:
        for row in layerwise_data:
            d = int(row["depth"])
            var = float(row["gradient_variance"])
            records.append({
                "method_name": "Layer-Wise Corrected (Uniform)",
                "circuit_depth": d,
                "reported_gradient_variance": var,
                "trial_count": 100,
                "measurement_timing": "AFTER Training (2 gradient steps, lr=0.5)",
                "parameter_measured": "Active Layer l, Qubit 0, Rx (theta_l[0])",
                "initialization_protocol": "Uniform(0, 2*pi) on active layer",
                "training_protocol": "2 GD steps on active layer; Prior layers permanently frozen",
                "directly_comparable_to_baseline": "NOT Directly Comparable",
                "comparability_notes": "Measures active layer AFTER 2 training steps; gate is at end of circuit, not Layer 0."
            })

    # 3. Fixed Combined (Adaptive Mitigation)
    adaptive_mit_file = "results_adaptive_mitigation.csv"
    adaptive_mit_data = load_csv_data(adaptive_mit_file)
    if adaptive_mit_data:
        for row in adaptive_mit_data:
            d = int(row["depth"])
            var = float(row["gradient_variance"])
            records.append({
                "method_name": "Fixed Combined (Layer-wise + Normal Init)",
                "circuit_depth": d,
                "reported_gradient_variance": var,
                "trial_count": 100,
                "measurement_timing": "AFTER Training (2 gradient steps, lr=0.5)",
                "parameter_measured": "Active Layer l, Qubit 0, Rx (theta_l[0])",
                "initialization_protocol": "Normal(0, 0.1) on active layer",
                "training_protocol": "2 GD steps on active layer; Prior layers permanently frozen",
                "directly_comparable_to_baseline": "NOT Directly Comparable",
                "comparability_notes": "Measures active layer AFTER 2 training steps; uses near-identity initialization."
            })

    # 4. Adaptive Trigger
    trigger_file = "results_adaptive_trigger.csv"
    trigger_data = load_csv_data(trigger_file)
    if trigger_data:
        for row in trigger_data:
            d = int(row["depth"])
            var = float(row["current_variance"])
            mit_pct = float(row["mitigation_active_pct"])
            records.append({
                "method_name": "Dynamic Adaptive Trigger",
                "circuit_depth": d,
                "reported_gradient_variance": var,
                "trial_count": 100,
                "measurement_timing": "BEFORE Training (at layer addition, step 0)",
                "parameter_measured": "Active Layer l, Qubit 0, Rx (theta_l[0])",
                "initialization_protocol": f"Dynamic: Uniform(0, 2*pi) until trigger, then Normal(0, 0.1) [{mit_pct:.0f}% active]",
                "training_protocol": "Full-circuit GD before trigger; Active-layer GD + frozen prior layers after trigger",
                "directly_comparable_to_baseline": "NOT Directly Comparable",
                "comparability_notes": "Measures active layer BEFORE training. Decision rule operates on intra-layer variance across 18 components, not cross-trial variance."
            })

    return records, baseline_data, layerwise_data, adaptive_mit_data, trigger_data

def save_comparison_csv(records, output_path="benchmark_unified_comparison.csv"):
    if not records:
        print("[WARNING] No records to save.")
        return
    fieldnames = [
        "method_name",
        "circuit_depth",
        "reported_gradient_variance",
        "trial_count",
        "measurement_timing",
        "parameter_measured",
        "initialization_protocol",
        "training_protocol",
        "directly_comparable_to_baseline",
        "comparability_notes"
    ]
    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"[SUCCESS] Saved unified comparison table to: {output_path}")

def generate_visualizations(baseline_data, layerwise_data, adaptive_mit_data, trigger_data, output_png="benchmark_unified_comparison.png"):
    """
    Generates a scientifically rigorous multi-panel plot that highlights
    protocol differences rather than drawing false equivalences.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Helper to parse depths and variances
    def parse_depth_var(data, var_key="gradient_variance"):
        if not data:
            return [], []
        depths = [int(r["depth"]) for r in data]
        vars_ = [float(r[var_key]) for r in data]
        return depths, vars_

    b_d, b_v = parse_depth_var(baseline_data, "gradient_variance")
    l_d, l_v = parse_depth_var(layerwise_data, "gradient_variance")
    a_d, a_v = parse_depth_var(adaptive_mit_data, "gradient_variance")
    t_d, t_v = parse_depth_var(trigger_data, "current_variance")

    # PANEL 1: Reported Variances with Methodology Distinction
    ax1 = axes[0, 0]
    if b_d:
        ax1.plot(b_d, b_v, "x--", color="gray", linewidth=2, label="Baseline (L0 param, Pre-training, N=250)")
    if t_d:
        ax1.plot(t_d, t_v, "s-", color="blue", linewidth=2.5, label="Dynamic Trigger (Active param, Pre-training, N=100)")
    if l_d:
        ax1.plot(l_d, l_v, "^:", color="purple", linewidth=2, label="Layer-wise Corrected (Active param, Post-training, N=100)")
    if a_d:
        ax1.plot(a_d, a_v, "o-.", color="red", linewidth=2, label="Fixed Combined (Active param, Post-training, N=100)")

    ax1.set_yscale("log")
    ax1.set_xlabel("Circuit Depth (Layers)")
    ax1.set_ylabel("Reported Gradient Variance (log scale)")
    ax1.set_title("Panel A: Reported Variances (Direct Curve Overlay)\n[CAUTION: Methods differ in timing and parameter target]")
    ax1.legend(loc="best", fontsize=8)
    ax1.grid(True, which="both", alpha=0.3)

    # PANEL 2: Grouped by Timing: Pre-Training vs Post-Training
    ax2 = axes[0, 1]
    if b_d:
        ax2.plot(b_d, b_v, "x--", color="black", linewidth=1.8, label="Baseline [Pre-training, Layer 0]")
    if t_d:
        ax2.plot(t_d, t_v, "s-", color="royalblue", linewidth=2, label="Dynamic Trigger [Pre-training, Active Layer]")
    ax2.set_yscale("log")
    ax2.set_xlabel("Circuit Depth (Layers)")
    ax2.set_ylabel("Gradient Variance (log scale)")
    ax2.set_title("Panel B: Pre-Training Measurements\n(Baseline vs Dynamic Trigger)")
    ax2.legend(loc="best", fontsize=8)
    ax2.grid(True, which="both", alpha=0.3)

    # PANEL 3: Grouped by Timing: Post-Training (Trained Active Layer)
    ax3 = axes[1, 0]
    if l_d:
        ax3.plot(l_d, l_v, "^-", color="darkmagenta", linewidth=2, label="Layer-Wise Only [Post-training]")
    if a_d:
        ax3.plot(a_d, a_v, "o-", color="crimson", linewidth=2, label="Fixed Combined [Post-training]")
    if b_d:
        ax3.plot(b_d, b_v, "x--", color="lightgray", linewidth=1.5, label="Baseline (Reference floor)")
    ax3.set_yscale("log")
    ax3.set_xlabel("Circuit Depth (Layers)")
    ax3.set_ylabel("Gradient Variance (log scale)")
    ax3.set_title("Panel C: Post-Training Measurements (2 GD Steps)\n(Layer-Wise vs Fixed Combined)")
    ax3.legend(loc="best", fontsize=8)
    ax3.grid(True, which="both", alpha=0.3)

    # PANEL 4: Adaptive Trigger Internal Mechanism vs Reported Statistics
    ax4 = axes[1, 1]
    if trigger_data:
        depths_trig = [int(r["depth"]) for r in trigger_data]
        curr_var = [float(r["current_variance"]) for r in trigger_data]
        thresh = [float(r["trigger_threshold"]) for r in trigger_data]
        pct = [float(r["mitigation_active_pct"]) for r in trigger_data]

        color1 = 'tab:blue'
        ax4.set_xlabel("Circuit Depth (Layers)")
        ax4.set_ylabel("Cross-Trial Variance (Current)", color=color1)
        l1 = ax4.plot(depths_trig, curr_var, "o-", color=color1, label="CSV Current Variance (Cross-Trial, θ0)")
        l2 = ax4.plot(depths_trig, thresh, "--", color="orange", label="Internal Threshold (0.05 * Intra-Layer Var)")
        ax4.tick_params(axis='y', labelcolor=color1)
        ax4.set_yscale("log")

        ax4_twin = ax4.twinx()
        color2 = 'tab:green'
        ax4_twin.set_ylabel("Mitigation Active Rate (%)", color=color2)
        l3 = ax4_twin.plot(depths_trig, pct, "s:", color=color2, linewidth=2, label="Mitigation Active %")
        ax4_twin.tick_params(axis='y', labelcolor=color2)
        ax4_twin.set_ylim(-5, 105)

        lines = l1 + l2 + l3
        labels = [l.get_label() for l in lines]
        ax4.legend(lines, labels, loc="center left", fontsize=8)
        ax4.set_title("Panel D: Adaptive Trigger Mechanism vs CSV Metric\n(Intra-Layer Trigger vs Cross-Trial Variance)")
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_png, dpi=150)
    print(f"[SUCCESS] Saved multi-panel benchmark visualization to: {output_png}")

def generate_text_report(records, output_path="benchmark_unified_comparison_report.txt"):
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("BENCHMARK UNIFIED COMPARISON & METHODOLOGY AUDIT REPORT")
    report_lines.append("Project: Demonstrating and Mitigating Barren Plateaus in Random Circuits")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append("1. EXECUTIVE SUMMARY & COMPARABILITY VERDICT")
    report_lines.append("-" * 80)
    report_lines.append("QUESTION: Does the current depth-experiment data permit a fair direct comparison?")
    report_lines.append("VERDICT: NO. The methods cannot be directly compared on a single unconditioned curve.")
    report_lines.append("")
    report_lines.append("RATIONALE:")
    report_lines.append("A rigorous audit reveals that each experiment used distinct protocols:")
    report_lines.append("  1. Measurement Timing:")
    report_lines.append("     - Baseline Depth: Measured at INITIALIZATION (0 training steps).")
    report_lines.append("     - Layer-Wise Corrected & Fixed Combined: Measured AFTER 2 gradient descent steps.")
    report_lines.append("     - Dynamic Adaptive Trigger: Measured BEFORE the 2 training steps.")
    report_lines.append("  2. Parameter Scope:")
    report_lines.append("     - Baseline: Measures parameter 0 of LAYER 0 (followed by L-1 scrambling layers).")
    report_lines.append("     - Layer-Wise, Combined, & Trigger: Measure parameter 0 of the ACTIVE LAYER l (at end).")
    report_lines.append("  3. Freezing & Training Scope:")
    report_lines.append("     - Baseline: Entire circuit initialized at once; no parameters trained or frozen.")
    report_lines.append("     - Layer-Wise & Combined: Earlier layers are permanently frozen from depth 1.")
    report_lines.append("     - Adaptive Trigger: All layers trained simultaneously before trigger; earlier layers")
    report_lines.append("       frozen only after trigger activates.")
    report_lines.append("  4. Sample Size Asymmetry:")
    report_lines.append("     - Baseline Depth uses N = 250 trials.")
    report_lines.append("     - Mitigations use N = 100 trials.")
    report_lines.append("")
    report_lines.append("2. ADAPTIVE TRIGGER STATISTIC CLARIFICATION")
    report_lines.append("-" * 80)
    report_lines.append("The adaptive trigger script (adaptive_trigger.py) evaluates a per-trial trigger condition:")
    report_lines.append("  v_curr < 0.05 * v_ref")
    report_lines.append("where:")
    report_lines.append("  - v_curr is the variance across all 18 gradient components of the active layer within trial k.")
    report_lines.append("  - v_ref is the intra-layer variance of layer 1 within trial k.")
    report_lines.append("")
    report_lines.append("However, the output CSV (results_adaptive_trigger.csv) reports:")
    report_lines.append("  - current_variance: Population variance across 100 trials of parameter 0 (0.30819 at depth 1).")
    report_lines.append("  - depth_1_reference: Mean of intra-layer variances across 100 trials (0.028605).")
    report_lines.append("  - trigger_threshold: 0.05 * depth_1_reference (0.001430).")
    report_lines.append("")
    report_lines.append("Therefore, 'current_variance' is NOT the statistic tested against 'trigger_threshold'.")
    report_lines.append("They represent two completely different variance metrics.")
    report_lines.append("")
    report_lines.append("3. NUMERICAL DATA COMPARISON TABLE")
    report_lines.append("-" * 80)
    report_lines.append(f"{'Method':<32} | {'Depth':<5} | {'Variance':<12} | {'Timing':<15} | {'Comparable?':<15}")
    report_lines.append("-" * 80)
    for r in records:
        m = r["method_name"][:32]
        d = r["circuit_depth"]
        v = f"{r['reported_gradient_variance']:.6e}"
        t = "Init (0 steps)" if "Init" in r["measurement_timing"] or "BEFORE" in r["measurement_timing"] else "Post-2 steps"
        c = "Reference" if "Self" in r["directly_comparable_to_baseline"] else "NO"
        report_lines.append(f"{m:<32} | {d:<5} | {v:<12} | {t:<15} | {c:<15}")
    report_lines.append("-" * 80)
    report_lines.append("")
    report_lines.append("4. VALID SCIENTIFIC CONCLUSIONS SUPPORTED BY DATA")
    report_lines.append("-" * 80)
    report_lines.append("1. Baseline barren plateau verified: As depth increases from L=1 to L=8, gradient variance")
    report_lines.append("   of layer 0 collapses from 0.2737 to 0.0099, saturating at ~0.0075 for L >= 8.")
    report_lines.append("2. Layer-wise training alone fails in deep circuits: When adding uniformly random layers,")
    report_lines.append("   gradient variance of the active layer drops from 0.1523 at L=1 to 0.0076 at L=20,")
    report_lines.append("   converging to the unmitigated baseline floor.")
    report_lines.append("3. Fixed combination prevents collapse: Appending Normal(0, 0.1) initialized layers keeps")
    report_lines.append("   post-training active layer variance above 0.0195 at all depths (0.0257 at L=20).")
    report_lines.append("4. Dynamic adaptive trigger activates as depth increases: Trigger rate rises from 0% at L=1")
    report_lines.append("   to 62% at L=20 (mean trigger depth = 9.98 layers). Active layer gradient variance at L=20")
    report_lines.append("   is recorded at 0.0535.")
    report_lines.append("")
    report_lines.append("5. REMAINING LIMITATIONS & FUTURE REQUIREMENTS")
    report_lines.append("-" * 80)
    report_lines.append("- To achieve a true apples-to-apples comparison, an experiment must be run where all methods")
    report_lines.append("  record BOTH pre-training and post-training gradients on the exact same parameter indices.")
    report_lines.append("- Higher gradient variance does not guarantee convergence to the global optimum.")
    report_lines.append("- Finite sampling noise (shot noise) has not been evaluated.")
    report_lines.append("=" * 80)

    with open(output_path, mode="w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"[SUCCESS] Saved detailed audit report to: {output_path}")

def main():
    print("=" * 60)
    print("RUNNING BENCHMARK UNIFIED COMPARISON")
    print("=" * 60)
    records, b_data, l_data, a_data, t_data = build_comparison_records()
    save_comparison_csv(records)
    generate_visualizations(b_data, l_data, a_data, t_data)
    generate_text_report(records)
    print("\nBenchmark Unified Comparison finished successfully.")

if __name__ == "__main__":
    main()
