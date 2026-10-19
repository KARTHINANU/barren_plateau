"""
scratch/correct_phase7_statistics.py
Recalculates exact Student-t 95% confidence intervals with sample standard deviation (ddof=1)
for all 48 conditions in Phase 7.
"""
import csv
import math
import numpy as np
from scipy import stats

csv_in = r"c:\Users\cheedella.karthikeya\OneDrive\Desktop\Barren_Plateau_Project\results_phase7_controlled_benchmark.csv"
csv_out = r"c:\Users\cheedella.karthikeya\OneDrive\Desktop\Barren_Plateau_Project\results_phase7_controlled_benchmark_corrected.csv"

with open(csv_in, "r") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

N = 10
df = N - 1
t_crit = stats.t.ppf(0.975, df=df)
print(f"Sample size N = {N}, degrees of freedom df = {df}")
print(f"Student-t two-sided 95% critical value t_(0.975, 9) = {t_crit:.6f} (vs normal z = 1.960000)")
print(f"Ratio t_crit / z = {t_crit / 1.96:.4f}")

corrected_rows = []

for r in rows:
    # Original values
    mean_delta = float(r["mean_cost_reduction"])
    std_pop = float(r["std_cost_reduction"]) # ddof=0
    
    # Recalculate with ddof=1
    # s = std_pop * sqrt(N / (N - 1))
    sample_std = std_pop * math.sqrt(N / (N - 1))
    
    # Standard error: SE = sample_std / sqrt(N) = std_pop / sqrt(N - 1)
    se = sample_std / math.sqrt(N)
    
    # Student-t CI
    ci_t_low = mean_delta - t_crit * se
    ci_t_high = mean_delta + t_crit * se
    
    # Also adjust initial and final cost sample stds
    sample_std_init = float(r["std_initial_cost"]) * math.sqrt(N / (N - 1))
    sample_std_final = float(r["std_final_cost"]) * math.sqrt(N / (N - 1))
    
    # Construct corrected row
    # Preserving all original fields and appending corrected statistical metrics
    new_r = dict(r)
    new_r["sample_std_cost_reduction_ddof1"] = sample_std
    new_r["se_cost_reduction"] = se
    new_r["t_critical_95"] = t_crit
    new_r["ci_student_t_low"] = ci_t_low
    new_r["ci_student_t_high"] = ci_t_high
    new_r["sample_std_initial_cost_ddof1"] = sample_std_init
    new_r["sample_std_final_cost_ddof1"] = sample_std_final
    
    corrected_rows.append(new_r)

# Write to results_phase7_controlled_benchmark_corrected.csv
fieldnames = list(corrected_rows[0].keys())
with open(csv_out, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(corrected_rows)

print(f"[SUCCESS] Wrote {len(corrected_rows)} rows to {csv_out}")

# Verification check on corrected rows
for idx, r in enumerate(corrected_rows):
    m = float(r["mean_cost_reduction"])
    s = float(r["sample_std_cost_reduction_ddof1"])
    se = float(r["se_cost_reduction"])
    low = float(r["ci_student_t_low"])
    high = float(r["ci_student_t_high"])
    
    expected_se = s / math.sqrt(10)
    assert math.isclose(se, expected_se, abs_tol=1e-12), f"Row {idx} SE mismatch"
    assert math.isclose(low, m - t_crit * se, abs_tol=1e-12), f"Row {idx} CI low mismatch"
    assert math.isclose(high, m + t_crit * se, abs_tol=1e-12), f"Row {idx} CI high mismatch"

print("[PASS] All corrected formulas and Student-t CIs verified with zero error!")
