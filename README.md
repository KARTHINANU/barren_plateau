# Demonstrating and Mitigating Barren Plateaus in Random Quantum Circuits

**A simulation-based investigation of gradient concentration, trainability, and mitigation strategies in parameterised quantum circuits.**

## Overview

Barren plateaus are a potential trainability challenge in variational quantum algorithms (VQAs). They occur when gradients become increasingly concentrated around zero, making gradient-based optimisation difficult as the circuit or quantum system grows.

This project investigates the gradient behaviour of parameterised quantum circuits and evaluates different approaches for improving their optimisation performance. Using Qiskit's statevector simulator, the study examines how gradient variance changes with qubit count and circuit depth, investigates the effect of finite-shot sampling, and compares parameter initialisation, layer-wise optimisation, and gradient-assessment strategies.

The investigation progresses from establishing an extended baseline to developing and evaluating mitigation methods through controlled experiments and a scaled benchmark.

**All experiments were conducted using classical quantum-circuit simulation. No physical quantum hardware was used.**

## Research Objectives

The primary objectives of this project are:

1. Construct parameterised quantum circuits using single-qubit rotation gates and nearest-neighbour CNOT entanglement.
2. Evaluate objective functions and their gradients using the parameter-shift rule.
3. Investigate the relationship between gradient variance, qubit count, and circuit depth.
4. Examine the effect of finite-shot sampling on gradient estimation accuracy.
5. Compare uniform and controlled parameter initialisation strategies.
6. Investigate standard layer-wise optimisation and combined mitigation approaches.
7. Develop and evaluate fixed and variance-relative gradient-assessment criteria.
8. Compare Gradient Descent, Adam, and COBYLA as optimisation algorithms.
9. Evaluate optimisation performance using initial objective values, final objective values, objective-function reduction, statistical uncertainty, and computational requirements.
10. Conduct a scaled benchmark using matched random seeds and document the actual computational budgets.

## Quantum Circuit Architecture

The investigated circuits consist of parameterised single-qubit rotations and nearest-neighbour entangling operations.

Each qubit is initialised in the computational basis state:

$$
|\psi_0\rangle = |0\rangle^{\otimes n}
$$

The single-qubit rotation gates are:

- **Rx(θ):** Rotation around the X axis.
- **Ry(θ):** Rotation around the Y axis.
- **Rz(θ):** Rotation around the Z axis.

Nearest-neighbour CNOT gates are applied to introduce entanglement between adjacent qubits.

The parameterised circuit is repeated across $L$ layers. Each rotation gate has an associated trainable parameter.

### Objective Function

The objective function is the expectation value of the Pauli-Z observable on the first qubit:

$$
C(\boldsymbol{\theta}) =
\langle\psi(\boldsymbol{\theta})|
Z_0
|\psi(\boldsymbol{\theta})\rangle
$$

where:

- $\boldsymbol{\theta}$ represents the collection of circuit parameters.
- $|\psi(\boldsymbol{\theta})\rangle$ is the resulting quantum state.
- $Z_0$ is the Pauli-Z observable acting on the first qubit.

The objective-function value lies within:

$$
-1 \leq C(\boldsymbol{\theta}) \leq 1
$$

The optimisation task is to minimise this objective function towards the target value of $-1$.

## Gradient Evaluation

The gradient describes how the objective function changes with respect to an individual circuit parameter.

The parameter-shift rule is used to calculate the gradients:

$$
\frac{\partial C}{\partial \theta_i} = \frac{C(\theta_i+\pi/2)-C(\theta_i-\pi/2)}{2}
$$

This allows the gradients to be evaluated using shifted circuit parameters.

Gradient variance is calculated over multiple independent random parameter initialisations:

$$
\mathrm{Var}(g) = \frac{1}{N}\sum_{i=1}^{N}(g_i-\bar{g})^2
$$

where:

- $N$ is the number of independent trials.
- $g_i$ is the gradient value from an individual trial.
- $\bar{g}$ is the mean gradient.

A reduction in gradient variance towards zero indicates increasing gradient concentration in the investigated circuit configuration.

## Experimental Stages

### Stage 1 — Extended Baseline Investigation

The initial investigation was extended to examine gradient variance across a wider qubit range.

**Configuration:**

- Qubit count: $n=2$ to $14$.
- Independent random trials: 250 per qubit count.
- Circuit depth: $L=20$.
- Total trials: 3,250.
- Gradient variance calculated using the sampled gradient values.

The measured gradient variance decreased from approximately:

| Qubit count | Gradient variance |
|---:|---:|
| 2 | 0.146522 |
| 3 | 0.060230 |
| 4 | 0.039354 |
| 5 | 0.016599 |
| 6 | 0.008231 |
| 7 | 0.003428 |
| 8 | 0.002098 |
| 9 | 0.001123 |
| 10 | 0.000578 |
| 11 | 0.000256 |
| 12 | 0.000157 |
| 13 | 0.000113 |
| 14 | 0.0000875 |

An empirical exponential fit to the extended data was reported as:

$$
\mathrm{Var}(g)
\approx
0.6486e^{-0.7489n}
$$

with a coefficient of determination:

$$
R^2=0.9942
$$

The results provide empirical evidence of gradient concentration over the investigated finite range. The fitted relationship is not claimed to establish a universal asymptotic scaling law.

### Stage 2 — Preliminary Optimiser Comparison

Gradient Descent, Adam, and COBYLA were compared using a preliminary five-seed experiment.

The comparison considered:

- Objective-function convergence.
- Objective-function reduction.
- Circuit evaluation requirements.
- Wall-clock execution time.
- Consistency across random initialisations.

COBYLA was selected for subsequent experiments as a practical optimisation method based on its lower evaluation requirements in the preliminary comparison. This selection does not imply universal superiority over the other optimisers.

### Stage 3 — Pilot Benchmark

A pilot benchmark was conducted to examine the behaviour of different gradient-assessment and layer-wise optimisation approaches.

The investigated methods included:

- Uniform baseline initialisation.
- Fixed layer-wise assessment.
- Variance-relative assessment.

**Configuration:**

- Qubit counts: $n=4,6$.
- Circuit depths: $L=2,4$.
- Five random seeds.
- Exploratory comparisons of objective-function reduction and assessment activation rates.

The pilot results were used to inform the subsequent scaled benchmark. They were not treated as conclusive evidence of mitigation effectiveness.

### Stage 4 — Scaled Benchmark

The scaled benchmark evaluated three approaches:

1. **Baseline Uniform:** Full-circuit optimisation using uniform parameter initialisation.
2. **Layerwise FixedDetector:** Sequential layer-wise optimisation using a fixed multi-metric assessment criterion.
3. **Layerwise DynamicDetector:** Sequential layer-wise optimisation using a variance-relative assessment criterion.

**Experimental configuration:**

| Parameter | Configuration |
|---|---|
| Qubit counts | 4 and 6 |
| Circuit depths | 2, 4, 8 and 12 |
| Matched random seeds | 50 (Seeds 42–91) |
| Methods | 3 |
| Experimental conditions | 24 |
| Total trial records | 1,200 |
| Optimiser | COBYLA |

#### Objective-Evaluation Budgets

The baseline received 50 COBYLA objective-function evaluations per trial, independent of circuit depth.

The layer-wise methods used the following total objective-evaluation budgets:

| Circuit depth | Baseline | Layer-wise methods |
|---:|---:|---:|
| $L=2$ | 50 | 50 |
| $L=4$ | 50 | 48 |
| $L=8$ | 50 | 80 |
| $L=12$ | 50 | 120 |

The layer-wise budgets arise from the allocation of a minimum of 10 evaluations per layer, with the configured per-layer budget.

These are not equal-budget comparisons at every circuit depth. Diagnostic gradient evaluations were also recorded separately from COBYLA objective-function evaluations.

#### Evaluation Metrics

The benchmark considered:

- Initial objective-function values.
- Final objective-function values.
- Objective-function reduction.
- Assessment activation rates.
- Paired comparisons between methods.
- Statistical uncertainty.
- Objective-function evaluation counts.
- Diagnostic evaluation requirements.
- Computational execution time.

The results were interpreted with particular attention to the differences in computational budgets and initial objective-function values.

## Mitigation and Assessment Methods

The project investigates multiple strategies rather than assuming that a single mitigation method is effective in every circuit configuration.

### 1. Uniform Initialisation

Circuit parameters are independently initialised from a uniform distribution:

$$
\theta_i \sim U(0,2\pi)
$$

This serves as the reference initialisation strategy.

### 2. Controlled Initialisation

A normal distribution is used for parameter initialisation:

$$
\theta_i \sim \mathcal{N}(0,0.1^2)
$$

This provides a near-identity initialisation strategy for comparison with uniform initialisation.

### 3. Layer-Wise Optimisation

Circuit layers are optimised sequentially. The parameters associated with the current layer are updated while previously optimised layers are held fixed.

This approach investigates whether sequential optimisation can improve the behaviour of deeper parameterised circuits.

### 4. Fixed Multi-Metric Assessment

The fixed assessment method considers gradient magnitude, gradient variance, and objective-function stagnation.

The documented fixed criterion is:

$$
(\mathrm{MAG}<0.06 \land V<0.01)
$$

or

$$
(\Delta C_{\mathrm{progress}}<0.002
\land \mathrm{MAG}<0.10)
$$

where the quantities represent the gradient magnitude, gradient variance, and objective-function progress, respectively.

### 5. Variance-Relative Assessment

The enhanced variance-relative method compares the gradient variance of the current layer with a reference variance.

The relative gradient variance is calculated as:

$$
\rho_V(l)=
\frac{V_l}{V_{\mathrm{reference}}}
$$

The first assessment condition is:

$$
\rho_V(l)<0.10
$$

An additional condition considers objective-function stagnation:

$$
\Delta C_{\mathrm{progress}}<0.005|C_1|
\quad\land\quad
\rho_V(l)<0.25
$$

where:

- $V_l$ represents the gradient variance of the current layer.
- $V_{\mathrm{reference}}$ represents the reference gradient variance.
- $C_1$ represents the initial objective-function value.
- $\Delta C_{\mathrm{progress}}$ represents the objective-function progress.

A numerical safeguard is used when the reference variance is extremely small to avoid unstable relative comparisons.

The thresholds are empirical choices used in this study. They are not universal criteria for barren plateau detection.

## Key Findings

The principal findings from the reported experiments are:

- Gradient variance decreases substantially as qubit count increases in the investigated baseline configuration.
- The extended baseline exhibits a strong empirical exponential trend over the tested qubit range.
- Gradient Descent, Adam, and COBYLA all achieve objective-function reductions in the preliminary optimiser comparison.
- Lower gradient variance does not automatically imply better objective-function optimisation.
- A larger objective-function reduction must be interpreted alongside the initial and final objective-function values.
- Assessment activation frequency alone does not establish mitigation effectiveness.
- The Adaptive Enhanced method achieved a lower final objective-function value than the uniform baseline in the reported initial controlled benchmark condition at $n=6,\ L=12$.
- The scaled benchmark revealed differences in activation frequency and computational evaluation requirements between the fixed and variance-relative layer-wise methods.
- Layer-wise methods incurred different objective-evaluation budgets at different circuit depths.

These findings are specific to the investigated circuit configurations, observables, initialisation distributions, optimisation settings, and simulation conditions.

## Technology Stack

- **Python** — experiment implementation and analysis.
- **Qiskit** — quantum-circuit construction and simulation.
- **Qiskit Statevector Simulator** — exact statevector-based circuit evaluation.
- **NumPy** — numerical computation.
- **Pandas** — experimental data handling.
- **Matplotlib** — graph generation and result visualisation.
- **SciPy** — numerical fitting and statistical analysis.


## Installation

Clone the repository:

```bash
git clone <https://github.com/KARTHINANU/barren_plateau>
cd Barren-Plateau-Project
```

Create a Python virtual environment:

```bash
python -m venv .venv
```

On Windows PowerShell, activate it using:

```powershell
.venv\Scripts\Activate.ps1
```

Install the project dependencies:

```bash
pip install -r requirements.txt
```

The dependency file should contain the versions used in the actual experimental environment.

## Running the Experiments

Run the experiment scripts from the repository root using the corresponding Python commands.

For example:

```bash
python experiments/stage1_extended_baseline.py
python experiments/stage2_optimizer_smoke_test.py
python experiments/stage3_pilot_benchmark.py
python experiments/stage4_scaled_benchmark.py
```

The filenames above are illustrative. Replace them with the actual script names in the repository before publishing the README.

## Reproducibility

The experiments use recorded random seeds and preserve generated result files for analysis.

The reported benchmark configurations include:

- 250 independent trials per qubit count for the extended baseline.
- Five random seeds for the preliminary optimiser comparison.
- Five seeds for the pilot benchmark.
- Fifty matched random seeds per condition for the Stage 4 scaled benchmark.

The raw CSV files and generated plots should be retained alongside the experiment scripts to support reproducibility and result verification.

## Limitations

The following limitations apply to the investigation:

1. **Classical simulation:** All experiments were performed using classical quantum-circuit simulation. Results do not represent execution on physical quantum hardware.
2. **Finite qubit range:** The extended baseline investigates up to 14 qubits. The results do not establish universal asymptotic scaling.
3. **Circuit dependence:** Barren plateau behaviour depends on circuit architecture, depth, parameter initialisation, observable selection, and other experimental conditions.
4. **Finite-shot approximation:** Simulated measurement sampling does not reproduce every source of noise present in physical quantum processors.
5. **Empirical assessment thresholds:** The fixed and variance-relative criteria use experimentally selected thresholds rather than universally established boundaries.
6. **Unequal evaluation budgets:** The Stage 4 baseline and layer-wise methods do not receive equal objective-function evaluation budgets at every circuit depth.
7. **Statistical scope:** The initial controlled benchmark and scaled benchmark have different sample sizes and experimental configurations. Their results should not be treated as directly interchangeable.

## Conclusion

This project provides a simulation-based investigation of barren plateau behaviour and mitigation strategies in parameterised quantum circuits.

The extended baseline experiments provide empirical evidence of decreasing gradient variance with increasing qubit count in the investigated circuit configuration. Subsequent experiments examine optimiser behaviour, controlled initialisation, layer-wise optimisation, and fixed and variance-relative assessment strategies.

The results indicate that gradient behaviour, optimisation performance, and computational expenditure must be evaluated together. In particular, preserving gradient variance or increasing assessment activation frequency does not necessarily lead to better optimisation outcomes.

The study therefore emphasises careful experimental comparison, transparent computational budgets, statistical interpretation, and reproducibility when investigating trainability in variational quantum circuits.

**Project Focus:** Quantum Circuit Trainability | Gradient Concentration | Parameter-Shift Differentiation | Layer-Wise Optimisation | Mitigation Assessment | Reproducible Simulation

## References

1. McClean, J. R., Boixo, S., Smelyanskiy, V. N., Babbush, R., and Neven, H. (2018). *Barren Plateaus in Quantum Neural Network Training Landscapes*. Nature Communications, 9, 4812. https://doi.org/10.1038/s41467-018-07090-4
2. Cerezo, M., et al. (2021). *Cost Function Dependent Barren Plateaus in Shallow Parametrized Quantum Circuits*. Nature Communications, 12, 1791. https://doi.org/10.1038/s41467-021-21728-w

---

**Project focus:** Quantum circuit trainability | Gradient concentration | Parameter-shift differentiation | Layer-wise optimisation | Mitigation assessment | Reproducible simulation
