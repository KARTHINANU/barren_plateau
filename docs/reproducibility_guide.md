# Reproducibility Guide & Execution Protocol

**Project Title:** Demonstrating and Mitigating Barren Plateaus in Random Circuits  
**Document:** End-to-End Reproducibility Guide (`docs/reproducibility_guide.md`)  
**Status:** Verified Complete & Methodologically Audited  

---

## 1. Environment Setup & Dependency Requirements

This codebase was developed, audited, and verified using Python 3.12 on Windows 11.

### 1.1 Python Virtual Environment
To create and activate the recommended virtual environment:

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS (Bash):**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 1.2 Required Packages
Install the required dependencies via `pip`:
```bash
pip install qiskit==1.3.2 numpy scipy matplotlib
```

---

## 2. Verified Execution Commands by Module

All commands below have been cross-referenced against the actual `argparse` implementations in the codebase.

### Stage 1: Baseline Characterization
```powershell
# 1. Qubit scaling experiment (generates results_baseline_qubits.csv, barren_plateau_qiskit.png)
python barren_plateau.py

# 2. Non-linear exponential model fitting (generates exponential_fit_qubits.png)
python fit_exponential.py

# 3. Depth scaling experiment (generates results_baseline_depth.csv, barren_plateau_depth.png)
python depth_experiment.py
```

### Stage 2: Initial Mitigation Strategies
```powershell
# 1. Controlled near-zero initialization (generates results_mitigation_controlled_init.csv)
python mitigation_controlled_init.py

# 2. Corrected sequential layerwise training (generates results_mitigation_layerwise_corrected.csv)
python mitigation_layerwise_corrected.py
```

### Stage 3: Original Adaptive Trigger
```powershell
# 1. Original adaptive heuristic experiment (generates results_adaptive_mitigation.csv)
python adaptive_mitigation.py

# 2. Original adaptive trigger rate evaluation (generates results_adaptive_trigger.csv)
python adaptive_trigger.py
```

### Stage 4: Extended Diagnostics & Bias Analysis
```powershell
# 1. Expectation-value and gradient statistics diagnostics
python expectation_value_diagnostics.py --qubits 4 6 --depths 1 4 8 --trials 20 --seed 42

# 2. Quantum measurement shot noise vs statevector simulation
python measurement_shot_noise_experiment.py --qubits 4 --depth 2 --trials 30 --shots 100 1000 10000 --seed 42

# 3. Parameter-shift vs classical backpropagation comparison
python quantum_classical_gradient_comparison.py --trials 30 --seed 42

# 4. Controlled inductive bias experiment
python bias_analysis_experiment.py --qubits 4 --depths 1 4 8 --trials 30 --seed 42
```

### Stage 5: Enhanced Multi-Metric Plateau Detector
```powershell
# Multi-metric consensus detection sweep (qubits=6, depths 1..20)
python enhanced_plateau_detector.py --qubits 6 --depths 1 2 4 8 12 16 20 --trials 30 --seed 42
```

### Stage 6: Enhanced Adaptive Mitigation
```powershell
# Enhanced adaptive mitigation benchmark (qubits=4, depths 2 4 8, trials=25)
python adaptive_mitigation_enhanced.py --qubits 4 --depths 2 4 8 --trials 25 --seed 42
```

### Stage 7: Controlled CPU Benchmark Sweeps
```powershell
# 1. Rapid Smoke Test (Fast verification: n=4, L=2, N=2 trials, ~4.5 seconds)
python phase7_controlled_benchmark.py --smoke-test

# 2. Complete Pilot Benchmark Sweep (n=4 and 6, L=2,4,8,12, N=10 trials, 6 arms, ~45.3 minutes)
python phase7_controlled_benchmark.py --qubits 4 6 --depths 2 4 8 12 --trials 10 --seed 42
```

---

## 3. Computational Budget & Runtime Warnings

> [!WARNING]
> **Runtime Notice for Phase 7:**
> - The Phase 7 Pilot Benchmark (`--trials 10`) trains 480 quantum circuits and requires approximately **45 minutes** of CPU execution time.
> - An expanded full-scale benchmark (`--trials 100`, 4,800 circuits) is estimated to require approximately **7.5 hours** of execution time.
> - **Do not rerun the 480-circuit benchmark unnecessarily.** All raw trial records and summary metrics are preserved in `results_phase7_controlled_benchmark.csv` and `results_phase7_controlled_benchmark_corrected.csv`.

---

## 4. Random-Seed Policy & Bit-Exact Reproducibility

Every experiment accepts an explicit `--seed` parameter (default: 42).
Within benchmark loops, trial $i$ instantiates:
$$\text{seed}_{\text{trial}} = \text{base\_seed} + i$$
with paired independent streams for uniform draws ($\text{seed}_{\text{trial}}$) and normal draws ($\text{seed}_{\text{trial}} + 100000$).
This guarantees that any individual trial can be reproduced independently with bit-exact numerical parity down to machine precision.
