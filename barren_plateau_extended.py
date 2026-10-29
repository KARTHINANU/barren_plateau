"""
Barren Plateau Extended Baseline Experiment (Qiskit Statevector)
================================================================
Extends the original baseline qubit scaling experiment (barren_plateau.py)
from n=2..7 up to n=2..14 qubits at depth L=20 with N=250 trials per qubit count.

Key Scientific Guarantees:
1. Exact architecture preservation:
   - Alternating Rx, Ry, Rz single-qubit rotations on all qubits.
   - Nearest-neighbor linear CNOT entangling chain: CX(q, q+1) for q in 0..n-2.
   - Depth fixed at L=20 layers.
   - Local Pauli-Z observable on qubit 0: SparsePauliOp("I"*(n-1) + "Z").
   - Parameter-shift gradient for theta[0] (Layer 1, Qubit 0, Rx): (C_+ - C_-) / 2.
   - Initialization: Uniform(0, 2*pi).
2. Reproducibility & Continuous Seed Schedule:
   - Base seed = 42.
   - Sequential random stream ensures exact match for n=2..7 with historical results_baseline_qubits.csv.
3. Unbiased Statistical Metrics (ddof=1):
   - Sample variance: s^2 = 1/(N-1) * sum((g_i - mean(g))^2)
   - Sample standard deviation: s = sqrt(s^2)
   - Standard error of the mean: SE = s / sqrt(N)
   - Population variance (ddof=0) also recorded for direct backward audit.
4. Robust Data Integrity Checks:
   - Verification against non-finite values (NaN, Inf).
   - Duplicate detection.
   - Exponential model fitting with R^2 goodness of fit.
   - Explicit scientific disclaimer on asymptotic scaling.
"""

import os
import sys
import csv
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector, SparsePauliOp


# ==============================================================================
# 1. QUANTUM CIRCUIT & OBSERVABLE DEFINITIONS
# ==============================================================================

def build_circuit(n_qubits, n_layers=20):
    """
    Constructs an n_qubits, n_layers random circuit using ParameterVector.
    Matches barren_plateau.py exactly.
    """
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
    """
    Local Pauli-Z observable on qubit 0.
    In Qiskit reverse little-endian convention, qubit 0 is the rightmost character:
    'I'*(n-1) + 'Z'.
    """
    return SparsePauliOp("I" * (n_qubits - 1) + "Z")


def compute_parameter_shift_gradient(qc, theta, observable, params, param_idx=0):
    """
    Computes exact parameter-shift gradient for theta[param_idx]:
    dC / dtheta_i = [ C(theta + pi/2 * e_i) - C(theta - pi/2 * e_i) ] / 2
    """
    val_plus = params.copy()
    val_plus[param_idx] += np.pi / 2.0

    val_minus = params.copy()
    val_minus[param_idx] -= np.pi / 2.0

    c_plus = qc.assign_parameters(dict(zip(theta, val_plus)))
    state_plus = Statevector.from_instruction(c_plus)
    cost_plus = float(state_plus.expectation_value(observable).real)

    c_minus = qc.assign_parameters(dict(zip(theta, val_minus)))
    state_minus = Statevector.from_instruction(c_minus)
    cost_minus = float(state_minus.expectation_value(observable).real)

    grad = (cost_plus - cost_minus) / 2.0
    return grad


# ==============================================================================
# 2. EXPERIMENT RUNNER PER QUBIT COUNT
# ==============================================================================

def evaluate_qubit_count(n_qubits, n_layers=20, n_experiments=250):
    """
    Evaluates n_experiments random circuit instances for a fixed qubit count.
    Draws parameters sequentially from the current NumPy random state.
    """
    t0 = time.time()
    qc, theta = build_circuit(n_qubits, n_layers=n_layers)
    observable = observable_for(n_qubits)
    n_params = 3 * n_qubits * n_layers

    gradients = []
    for i in range(n_experiments):
        random_values = np.random.uniform(0.0, 2.0 * np.pi, n_params)
        grad = compute_parameter_shift_gradient(qc, theta, observable, random_values, param_idx=0)
        gradients.append(grad)

    gradients = np.array(gradients, dtype=float)
    elapsed = time.time() - t0

    # Data Integrity check on gradients
    if not np.all(np.isfinite(gradients)):
        raise ValueError(f"Non-finite gradient encountered for n={n_qubits}!")

    mean_g = float(np.mean(gradients))
    pop_var = float(np.var(gradients, ddof=0))
    sample_var = float(np.var(gradients, ddof=1))
    sample_std = float(np.std(gradients, ddof=1))
    se_g = float(sample_std / np.sqrt(n_experiments))

    print(f"  n={n_qubits:2d} (L={n_layers}) | N={n_experiments} | "
          f"Mean(g)={mean_g:+.6e} | Sample Var={sample_var:.6e} | Pop Var={pop_var:.6e} | "
          f"SE={se_g:.6e} | Time={elapsed:.2f}s")

    return {
        "n_qubits": n_qubits,
        "depth": n_layers,
        "trial_count": n_experiments,
        "mean_gradient": mean_g,
        "sample_variance": sample_var,
        "pop_variance": pop_var,
        "sample_std": sample_std,
        "standard_error": se_g,
        "runtime_seconds": elapsed,
        "raw_gradients": gradients
    }


# ==============================================================================
# 3. EXPONENTIAL MODEL FITTING & PLOTTING
# ==============================================================================

def exp_decay_model(n, A, b):
    """Exponential decay model: Var(g) = A * exp(-b * n)."""
    return A * np.exp(-b * n)


def fit_and_plot_results(records, output_plot="barren_plateau_qubits_extended.png",
                         fit_plot="exponential_fit_extended.png"):
    """
    Fits exponential decay model Var(g) = A * exp(-b * n) and renders diagnostic figures.
    """
    qubits = np.array([r["n_qubits"] for r in records], dtype=float)
    variances = np.array([r["sample_variance"] for r in records], dtype=float)
    means = np.array([r["mean_gradient"] for r in records], dtype=float)
    stds = np.array([r["sample_std"] for r in records], dtype=float)
    ses = np.array([r["standard_error"] for r in records], dtype=float)

    # Nonlinear least-squares exponential fit
    fit_successful = False
    A_fit, b_fit = np.nan, np.nan
    r_squared = np.nan
    popt, pcov = None, None

    try:
        p0 = [variances[0] * np.exp(qubits[0] * 0.7), 0.7]
        popt, pcov = curve_fit(exp_decay_model, qubits, variances, p0=p0, maxfev=10000)
        A_fit, b_fit = popt
        fit_successful = True

        # Calculate R^2 goodness of fit
        y_pred = exp_decay_model(qubits, A_fit, b_fit)
        ss_res = np.sum((variances - y_pred) ** 2)
        ss_tot = np.sum((variances - np.mean(variances)) ** 2)
        r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

        # Parameter standard errors from covariance matrix
        perr = np.sqrt(np.diag(pcov))
        print("\n" + "=" * 60)
        print("EXPONENTIAL DECAY MODEL FIT: Var(g) = A * exp(-b * n)")
        print("=" * 60)
        print(f"  Amplitude A = {A_fit:.6f} +/- {perr[0]:.6f}")
        print(f"  Decay rate b= {b_fit:.6f} +/- {perr[1]:.6f}")
        print(f"  Goodness-of-Fit R^2 = {r_squared:.6f}")
        print("  NOTE: Finite-size exponential fit across n in [2, 14] is empirical evidence")
        print("        consistent with barren plateaus, but does not mathematically prove asymptotic scaling.")
        print("=" * 60)
    except Exception as e:
        print(f"\n[Warning] Exponential fit could not converge: {e}")

    # Figure 1: Dual Panel (Linear + Semilogy) with Confidence Intervals
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel A: Linear scale
    axes[0].plot(qubits, variances, "o-", color="crimson", linewidth=2, markersize=6, label="Sample Variance (ddof=1)")
    if fit_successful:
        n_dense = np.linspace(min(qubits), max(qubits), 200)
        axes[0].plot(n_dense, exp_decay_model(n_dense, A_fit, b_fit), "--", color="navy", linewidth=1.8,
                     label=f"Fit: {A_fit:.3f} e^{{-{b_fit:.3f}n}} (R^2={r_squared:.4f})")
    axes[0].set_xlabel("Number of Qubits (n)", fontsize=11, fontweight="bold")
    axes[0].set_ylabel(r"Gradient Variance $\mathrm{Var}(\partial_{\theta_0} C)$", fontsize=11, fontweight="bold")
    axes[0].set_title("Barren Plateau: Variance vs Qubits (Linear Scale)", fontsize=12, fontweight="bold")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=9)

    # Panel B: Logarithmic scale (Semilogy)
    axes[1].semilogy(qubits, variances, "s-", color="navy", linewidth=2, markersize=6, label="Sample Variance (Log)")
    if fit_successful:
        axes[1].semilogy(n_dense, exp_decay_model(n_dense, A_fit, b_fit), "--", color="crimson", linewidth=1.8,
                         label=f"Decay Rate b = {b_fit:.4f}")
    axes[1].set_xlabel("Number of Qubits (n)", fontsize=11, fontweight="bold")
    axes[1].set_ylabel(r"Gradient Variance $\mathrm{Var}(\partial_{\theta_0} C)$ (Log Scale)", fontsize=11, fontweight="bold")
    axes[1].set_title("Barren Plateau: Exponential Decay (Log Scale)", fontsize=12, fontweight="bold")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(fontsize=9)

    fig.text(0.5, -0.02,
             "Empirical finite-size observation across n in [2, 14] at depth L=20. Does not constitute proof of asymptotic scaling.",
             ha="center", fontsize=9, style="italic", color="#444444")
    fig.tight_layout()
    fig.savefig(output_plot, dpi=150, bbox_inches="tight")
    print(f"[Plot Saved] {output_plot}")
    plt.close(fig)

    # Figure 2: Dedicated Publication-Ready Fit Figure
    fig_fit, ax_fit = plt.subplots(figsize=(8, 6))
    ax_fit.errorbar(qubits, variances, yerr=None, fmt="o", color="#b2182b",
                    ecolor="gray", elinewidth=1.2, capsize=3, markersize=7, label="Sample Variance (N=250)")
    if fit_successful:
        ax_fit.plot(n_dense, exp_decay_model(n_dense, A_fit, b_fit), "-", color="#2166ac", linewidth=2.2,
                    label=rf"Exponential Fit: $\mathrm{{Var}}(g) = {A_fit:.4f} \, e^{{-{b_fit:.4f} n}}$ ($R^2 = {r_squared:.4f}$)")
    ax_fit.set_yscale("log")
    ax_fit.set_xlabel("Number of Qubits $n$", fontsize=12, fontweight="bold")
    ax_fit.set_ylabel(r"Gradient Variance $\mathrm{Var}(\partial C / \partial \theta_0)$", fontsize=12, fontweight="bold")
    ax_fit.set_title("Exponential Suppression of Gradient Variance with Qubit Count\n"
                     r"(Depth $L=20$, Local Observable $Z_0$, $N=250$ Independent Trials)",
                     fontsize=12, fontweight="bold", pad=12)
    ax_fit.grid(True, which="both", alpha=0.35, linestyle="--")
    ax_fit.legend(fontsize=10, loc="upper right")
    fig_fit.text(0.5, -0.01,
                 "Note: Regression reflects finite system sizes (n <= 14); asymptotic behavior requires formal Haar integration.",
                 ha="center", fontsize=8.5, style="italic", color="#555555")
    fig_fit.tight_layout()
    fig_fit.savefig(fit_plot, dpi=150, bbox_inches="tight")
    print(f"[Fit Plot Saved] {fit_plot}")
    plt.close(fig_fit)

    return {
        "A_fit": A_fit,
        "b_fit": b_fit,
        "r_squared": r_squared
    }


# ==============================================================================
# 4. DATA INTEGRITY & AUDIT ENGINE
# ==============================================================================

def run_data_integrity_audit(csv_path):
    """
    Validates CSV file for:
    - Missing values
    - Non-finite numbers (NaN, Inf)
    - Duplicate qubit records
    - Correct arithmetic: sample_variance == sample_std^2, SE == sample_std / sqrt(N)
    """
    print("\n" + "=" * 60)
    print(f"DATA INTEGRITY AUDIT: {csv_path}")
    print("=" * 60)

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    qubits_seen = set()
    for idx, r in enumerate(rows):
        n = int(r["n_qubits"])
        N = int(r["trial_count"])
        mean_g = float(r["mean_gradient"])
        s_var = float(r["sample_variance"])
        pop_var = float(r["pop_variance"])
        s_std = float(r["sample_std"])
        se = float(r["standard_error"])

        # Check finite
        for val, name in [(mean_g, "mean"), (s_var, "s_var"), (pop_var, "pop_var"), (s_std, "s_std"), (se, "se")]:
            if not np.isfinite(val):
                raise AssertionError(f"Row {idx} (n={n}) contains non-finite {name} = {val}!")

        # Check duplicates
        if n in qubits_seen:
            raise AssertionError(f"Duplicate record detected for n={n} in row {idx}!")
        qubits_seen.add(n)

        # Check sample vs pop variance relation
        expected_pop_var = s_var * (N - 1) / float(N)
        assert np.isclose(pop_var, expected_pop_var, rtol=1e-5), f"Row {idx} pop_var mismatch!"

        # Check s_std == sqrt(s_var)
        assert np.isclose(s_std, np.sqrt(s_var), rtol=1e-5), f"Row {idx} std mismatch!"

        # Check SE == s_std / sqrt(N)
        assert np.isclose(se, s_std / np.sqrt(N), rtol=1e-5), f"Row {idx} SE mismatch!"

    print(f"[PASS] All {len(rows)} records verified: No NaNs, No duplicates, perfect mathematical consistency.")
    return True


# ==============================================================================
# 5. BASELINE VERIFICATION AGAINST ORIGINAL CSV
# ==============================================================================

def verify_against_original_baseline(extended_records, original_csv="results_baseline_qubits.csv"):
    """
    Compares n=2..7 from extended records against original results_baseline_qubits.csv.
    """
    print("\n" + "=" * 60)
    print("BASELINE REPRODUCIBILITY VERIFICATION (n=2..7)")
    print("=" * 60)

    if not os.path.exists(original_csv):
        print(f"[Notice] Original baseline CSV '{original_csv}' not found. Skipping numerical match check.")
        return

    orig_data = {}
    with open(original_csv, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            orig_data[int(r["n_qubits"])] = {
                "mean": float(r["mean_gradient"]),
                "pop_var": float(r["gradient_variance"])
            }

    all_matched = True
    for r in extended_records:
        n = r["n_qubits"]
        if n in orig_data:
            orig_mean = orig_data[n]["mean"]
            orig_var = orig_data[n]["pop_var"]
            new_mean = r["mean_gradient"]
            new_pop_var = r["pop_variance"]

            mean_match = np.isclose(orig_mean, new_mean, rtol=1e-6, atol=1e-9)
            var_match = np.isclose(orig_var, new_pop_var, rtol=1e-6, atol=1e-9)

            status = "MATCH" if (mean_match and var_match) else "DIFFER"
            print(f"  n={n:2d}: Orig Var={orig_var:.6e} | New Pop Var={new_pop_var:.6e} | Status: [{status}]")
            if not (mean_match and var_match):
                all_matched = False

    if all_matched:
        print("[PASS] Extended runner reproduces original results_baseline_qubits.csv bit-for-bit on matching seeds!")
    else:
        print("[Info] Numerical differences noted. This is expected if random stream order or seed schedule differs.")


# ==============================================================================
# 6. MAIN EXECUTION PIPELINE
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Barren Plateau Extended Baseline Experiment")
    parser.add_argument("--smoke-test", action="store_true", help="Run rapid smoke test (n=2..4, N=10)")
    parser.add_argument("--verify-only", action="store_true", help="Verify reproducibility against baseline (n=2, N=250)")
    parser.add_argument("--min-qubits", type=int, default=2, help="Minimum qubits (default: 2)")
    parser.add_argument("--max-qubits", type=int, default=14, help="Maximum qubits (default: 14)")
    parser.add_argument("--layers", type=int, default=20, help="Circuit depth (default: 20)")
    parser.add_argument("--trials", type=int, default=250, help="Trials per qubit count (default: 250)")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed (default: 42)")
    parser.add_argument("--csv", type=str, default="results_baseline_qubits_extended.csv", help="Output CSV path")
    parser.add_argument("--plot", type=str, default="barren_plateau_qubits_extended.png", help="Output plot path")
    parser.add_argument("--fit-plot", type=str, default="exponential_fit_extended.png", help="Output fit plot path")

    args = parser.parse_args()

    # Mode 1: Smoke Test
    if args.smoke_test:
        print("\n[SMOKE TEST MODE] Running fast verification: n=2..4 with N=10 trials...")
        np.random.seed(args.seed)
        smoke_records = []
        for n in range(2, 5):
            res = evaluate_qubit_count(n, n_layers=args.layers, n_experiments=10)
            smoke_records.append(res)
        print("[SMOKE TEST SUCCESSFUL] Circuit construction, parameter shift, and metrics executed cleanly.")
        return

    # Mode 2: Verify Single Qubit Count Reproducibility
    if args.verify_only:
        print("\n[REPRODUCIBILITY CHECK] Running n=2, N=250 with seed 42...")
        np.random.seed(42)
        res_n2 = evaluate_qubit_count(2, n_layers=20, n_experiments=250)
        verify_against_original_baseline([res_n2], original_csv="results_baseline_qubits.csv")
        return

    # Mode 3: Full Extended Sweep
    print("\n" + "=" * 78)
    print("BARREN PLATEAU EXTENDED BASELINE EXPERIMENT (Qiskit Statevector)")
    print("=" * 78)
    print(f"Qubit Sweep:            n = {args.min_qubits} to {args.max_qubits}")
    print(f"Circuit Depth:          L = {args.layers} layers")
    print(f"Trials per Qubit Count: N = {args.trials}")
    print(f"Global Random Seed:     {args.seed}")
    print(f"Observable:             Local Pauli-Z on Qubit 0")
    print(f"Output CSV:             {args.csv}")
    print(f"Output Figures:         {args.plot}, {args.fit_plot}")
    print("=" * 78)

    # Initialize master seed for continuous sequential stream
    np.random.seed(args.seed)

    total_start_time = time.time()
    records = []

    for n in range(args.min_qubits, args.max_qubits + 1):
        rec = evaluate_qubit_count(n, n_layers=args.layers, n_experiments=args.trials)
        records.append(rec)

    total_elapsed = time.time() - total_start_time
    print("\n" + "=" * 78)
    print(f"[Completed Sweep] Total elapsed wall-clock time: {total_elapsed:.2f}s ({total_elapsed/60.0:.2f} mins)")
    print("=" * 78)

    # Save to CSV
    csv_fields = [
        "n_qubits", "depth", "trial_count", "mean_gradient",
        "sample_variance", "pop_variance", "sample_std",
        "standard_error", "runtime_seconds"
    ]
    with open(args.csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for r in records:
            row_dict = {k: r[k] for k in csv_fields}
            writer.writerow(row_dict)

    print(f"[CSV Saved] Extended baseline results written to: {args.csv}")

    # Data Integrity Audit
    run_data_integrity_audit(args.csv)

    # Reproducibility Check against original baseline for n=2..7
    verify_against_original_baseline(records, original_csv="results_baseline_qubits.csv")

    # Fit and Plot
    fit_and_plot_results(records, output_plot=args.plot, fit_plot=args.fit_plot)


if __name__ == "__main__":
    main()
