# Methodological Limitations & Future Research Directions

**Project Title:** Demonstrating and Mitigating Barren Plateaus in Random Circuits  
**Document:** Limitations & Open Questions (`docs/limitations_and_future_work.md`)  
**Status:** Verified Complete & Methodologically Audited  

---

## 1. Methodological Limitations

### 1.1 Finite-Size Scaling & Small Qubit Registers ($n \le 7$)
- **Empirical Boundary:** All simulations in this repository evaluate registers between $n=2$ and $n=7$ qubits ($n \le 6$ for comprehensive benchmarking).
- **Extrapolation Risk:** While exponential decay in gradient variance $\text{Var}(g) \approx A e^{-b n}$ is clearly observed, fitting an exponential to $n \le 7$ does not constitute formal proof of asymptotic scaling as $n \to \infty$. Finite-size effects, non-asymptotic boundary terms, and simulator-specific artifacts can influence measured decay rates.

### 1.2 Exploratory Pilot Sample Size ($N = 10$)
- In Phase 7, each cell was evaluated over $N = 10$ independent random parameter initializations to keep execution feasible on classical desktop CPUs (~45 minutes).
- As demonstrated in the independent audit, $N = 10$ yields wide confidence intervals for marginal cost reductions in deep circuits ($L \ge 8$), spanning zero. These 10-trial results must be treated as exploratory findings rather than high-precision asymptotic benchmarks.

### 1.3 Heuristic Detection Thresholds
- The enhanced consensus detector uses fixed thresholds:
  - $\tau_{\text{var}} = 0.01$
  - $\tau_{\text{mag}} = 0.06$
  - $\tau_{\text{stagnation}} = 0.002$
- These thresholds are empirical engineering rules calibrated to local observables on small registers. They are **not universal mathematical constants** and will fail or require recalibration on larger registers or different ansatz topologies.

### 1.4 Initial State Energy Asymmetry
- Uniform random initialization places circuits in Haar-distributed states where $\langle Z_0 \rangle \approx 0.0$, capping the maximum achievable reduction at $\Delta C \le 1.0$.
- Controlled near-zero initialization places states near $|00\dots0\rangle$ with $\langle Z_0 \rangle \approx +0.95$, enabling $\Delta C \le 2.0$.
- Claiming that Controlled Initialization is superior based solely on $\Delta C$ is scientifically misleading. Absolute final cost $C(T_{\text{fin}})$ demonstrates that Baseline Uniform and Adaptive Enhanced achieve substantially lower ground-state energies.

### 1.5 Marginal vs Cumulative Layerwise Optimization
- In layerwise training, `c_init` at depth $L$ is measured after layers $1 \dots L-1$ have already been trained.
- Marginal cost reduction $\Delta C$ during layer $L$ training reflects only fine-tuning gains, whereas in full-circuit methods, $\Delta C$ reflects optimization from scratch. Both $C(T_{\text{fin}})$ and $\Delta C$ must be reported together.

### 1.6 Classical Simulation vs Physical Quantum Hardware
- All experiments rely on exact numerical statevector linear algebra or simulated binomial shot sampling.
- Real quantum processors (NISQ hardware) incur physical noise sources not modeled here:
  - Coherence relaxation ($T_1$) and dephasing ($T_2$).
  - Two-qubit gate cross-talk and calibration drift.
  - Asymmetric readout measurement errors.
  Noise-induced barren plateaus (Wang et al., 2021) cannot be eliminated by initialization strategies alone.

---

## 2. Plausible Future Research Directions

*(Note: The following represent open research opportunities and are not presented as completed work.)*

1. **Tensor-Network & Matrix Product State (MPS) Scaling:**
   Simulating 1D quantum circuits with Matrix Product States (MPS) up to $n = 30$–$50$ qubits to probe asymptotic scaling beyond dense statevector memory limits ($n \le 20$).
2. **Quantum Natural Gradient (QNG) Integration:**
   Evaluating whether preconditioned geometric optimizers utilizing the Quantum Fisher Information Matrix (QFIM) can navigate the affine subspace trapping observed in Fixed Combined mitigation.
3. **Global vs Local Observable Transitions:**
   Formulating continuous interpolation between strictly local ($Z_0$) and strictly global ($Z^{\otimes n}$) observables to map the critical locality threshold where barren plateaus emerge.
4. **Adaptive Threshold Learning:**
   Employing online statistical hypothesis testing (e.g. Sequential Probability Ratio Tests) to dynamically calibrate detection thresholds rather than using fixed constants.
5. **NISQ Hardware Execution with Error Mitigation:**
   Testing the Enhanced Adaptive Trigger on physical superconducting or trapped-ion hardware under Zero-Noise Extrapolation (ZNE) and Readout Error Mitigation (M3).
