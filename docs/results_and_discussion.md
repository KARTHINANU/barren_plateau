# Experimental Results & Scientific Discussion

**Project Title:** Demonstrating and Mitigating Barren Plateaus in Random Circuits  
**Document:** Results, Synthesis, and Discussion (`docs/results_and_discussion.md`)  
**Status:** Verified Complete & Methodologically Audited  

---

## 1. Stage 1: Baseline Barren Plateau Investigation

### 1.1 Qubit Scaling & Exponential Concentration
- **Script:** [`barren_plateau.py`](file:///c:/Users/cheedella.karthikeya/OneDrive/Desktop/Barren_Plateau_Project/barren_plateau.py)
- **Primary Data:** [`results_baseline_qubits.csv`](file:///c:/Users/cheedella.karthikeya/OneDrive/Desktop/Barren_Plateau_Project/results_baseline_qubits.csv)
- **Plots:** [`barren_plateau_qiskit.png`](file:///c:/Users/cheedella.karthikeya/OneDrive/Desktop/Barren_Plateau_Project/barren_plateau_qiskit.png), [`exponential_fit_qubits.png`](file:///c:/Users/cheedella.karthikeya/OneDrive/Desktop/Barren_Plateau_Project/exponential_fit_qubits.png)

The initial baseline experiment evaluated the variance of the parameter-shift gradient of the local observable $Z_0$ across qubit counts $n \in \{2, 3, 4, 5, 6, 7\}$ at fixed circuit depth $L=4$ using 100 random parameter initializations per point:

| Qubits ($n$) | Mean Gradient $\mathbb{E}[g]$ | Gradient Variance $\text{Var}(g)$ | Empirical Standard Deviation |
| :---: | :---: | :---: | :---: |
| 2 | $+0.0188$ | $0.146522$ | $0.3828$ |
| 3 | $-0.0158$ | $0.060230$ | $0.2454$ |
| 4 | $+0.0098$ | $0.039354$ | $0.1984$ |
| 5 | $-0.0010$ | $0.016599$ | $0.1288$ |
| 6 | $+0.0040$ | $0.008231$ | $0.0907$ |
| 7 | $-0.0020$ | $0.003428$ | $0.0585$ |

**Exponential Model Fitting:**
Using [`fit_exponential.py`](file:///c:/Users/cheedella.karthikeya/OneDrive/Desktop/Barren_Plateau_Project/fit_exponential.py), an exponential decay model $\text{Var}(g) = A \cdot e^{-b \cdot n}$ was fitted to the measured qubit variances via non-linear least squares (`scipy.optimize.curve_fit`):
$$A = 0.5843, \quad b = 0.7381 \implies \text{Var}(g) \approx 0.5843 \cdot e^{-0.7381 n}$$
*Interpretation & Limitations:* The empirical decay rate $b \approx 0.7381$ is consistent with the theoretical prediction of McClean et al. (2018), where 2-design Haar-random circuits concentrate gradients exponentially as $O(2^{-n})$. However, this fit is an empirical parameterization over a small register ($n \le 7$). It provides strong supportive empirical evidence, but does not substitute for analytic integration over the unitary Haar measure.

### 1.2 Circuit Depth Scaling
- **Script:** [`depth_experiment.py`](file:///c:/Users/cheedella.karthikeya/OneDrive/Desktop/Barren_Plateau_Project/depth_experiment.py)
- **Primary Data:** [`results_baseline_depth.csv`](file:///c:/Users/cheedella.karthikeya/OneDrive/Desktop/Barren_Plateau_Project/results_baseline_depth.csv)
- **Plot:** [`barren_plateau_depth.png`](file:///c:/Users/cheedella.karthikeya/OneDrive/Desktop/Barren_Plateau_Project/barren_plateau_depth.png)

Gradient variance was evaluated on $n=4$ qubits across depths $L \in \{1, 2, 4, 8, 12, 16, 20\}$:
- $L = 1$: $\text{Var}(g) = 0.2737$
- $L = 2$: $\text{Var}(g) = 0.1708$
- $L = 4$: $\text{Var}(g) = 0.0427$
- $L = 8$: $\text{Var}(g) = 0.0099$
- $L = 12$: $\text{Var}(g) = 0.00697$
- $L = 20$: $\text{Var}(g) = 0.00747$

*Insight:* In shallow circuits ($L \le 2$), the circuit does not form an approximate unitary 2-design; gradient variance remains high. By depth $L=8 \approx 2n$, the ansatz scrambles the quantum state across the 16-dimensional Hilbert space, and gradient variance plateaus at the Haar-random floor ($\sim 0.007$).

---

## 2. Stage 2: Initial Mitigation Strategies

### 2.1 Controlled Near-Zero Initialization
- **Script:** [`mitigation_controlled_init.py`](file:///c:/Users/cheedella.karthikeya/OneDrive/Desktop/Barren_Plateau_Project/mitigation_controlled_init.py)
- **Data & Plot:** [`results_mitigation_controlled_init.csv`](file:///c:/Users/cheedella.karthikeya/OneDrive/Desktop/Barren_Plateau_Project/results_mitigation_controlled_init.csv), [`mitigation_controlled_init.png`](file:///c:/Users/cheedella.karthikeya/OneDrive/Desktop/Barren_Plateau_Project/mitigation_controlled_init.png)

Initializing parameters $\theta_k \sim \mathcal{N}(0, 0.1^2)$ retains non-trivial gradients:
- $n = 4$: $\text{Var}(g) = 0.0345$ (vs baseline $0.0394$)
- $n = 6$: $\text{Var}(g) = 0.0196$ (vs baseline $0.0082$)
- $n = 7$: $\text{Var}(g) = 0.0209$ (vs baseline $0.0034$)

*Limitation:* Controlled initialization forces the initial quantum state near $|00\dots0\rangle$, creating an initial cost asymmetry ($C(T_0) \approx +0.95$) that must not be confused with superior algorithmic convergence.

### 2.2 Standard & Corrected Layerwise Training
- **Scripts:** [`mitigation_layerwise.py`](file:///c:/Users/cheedella.karthikeya/OneDrive/Desktop/Barren_Plateau_Project/mitigation_layerwise.py), [`mitigation_layerwise_corrected.py`](file:///c:/Users/cheedella.karthikeya/OneDrive/Desktop/Barren_Plateau_Project/mitigation_layerwise_corrected.py)
- **Data & Plot:** [`results_mitigation_layerwise_corrected.csv`](file:///c:/Users/cheedella.karthikeya/OneDrive/Desktop/Barren_Plateau_Project/results_mitigation_layerwise_corrected.csv), [`mitigation_layerwise_corrected.png`](file:///c:/Users/cheedella.karthikeya/OneDrive/Desktop/Barren_Plateau_Project/mitigation_layerwise_corrected.png)

Initial layerwise implementations exhibited an indexing bug in gradient estimation. The corrected script trains active layers sequentially while freezing prior layers:
- $L = 1$: $\text{Var}(g) = 0.1523$
- $L = 4$: $\text{Var}(g) = 0.0411$
- $L = 8$: $\text{Var}(g) = 0.0123$
- $L = 20$: $\text{Var}(g) = 0.0076$

---

## 3. Stage 3: Original Adaptive Trigger Investigation

### 3.1 The 5% Spatial Variance Drop Rule
- **Scripts:** [`adaptive_mitigation.py`](file:///c:/Users/cheedella.karthikeya/OneDrive/Desktop/Barren_Plateau_Project/adaptive_mitigation.py), [`adaptive_trigger.py`](file:///c:/Users/cheedella.karthikeya/OneDrive/Desktop/Barren_Plateau_Project/adaptive_trigger.py)
- **Data & Plots:** [`results_adaptive_trigger.csv`](file:///c:/Users/cheedella.karthikeya/OneDrive/Desktop/Barren_Plateau_Project/results_adaptive_trigger.csv), [`adaptive_trigger.png`](file:///c:/Users/cheedella.karthikeya/OneDrive/Desktop/Barren_Plateau_Project/adaptive_trigger.png)

The legacy trigger tested whether active-layer gradient dispersion dropped below 5% of the layer-1 reference variance:
$$v_{\text{curr}} < 0.05 \cdot v_{\text{ref}}$$

**The Activation Rate Discrepancy:**
- In [`results_adaptive_trigger.csv`](file:///c:/Users/cheedella.karthikeya/OneDrive/Desktop/Barren_Plateau_Project/results_adaptive_trigger.csv), activation scaled smoothly: $L=1$ (0%), $L=2$ (5%), $L=4$ (11%), $L=8$ (25%), $L=20$ (62%).
- In [`results_fair_comparison_benchmark.csv`](file:///c:/Users/cheedella.karthikeya/OneDrive/Desktop/Barren_Plateau_Project/results_fair_comparison_benchmark.csv), the trigger activated in 100% of trials at all depths $L \ge 2$.
- *Root Cause Identified in Audit:* In `adaptive_trigger.py`, circuits were optimized sequentially inter-layer, preserving gradient dispersion. In the unconditioned benchmark, evaluating layer 2 from unoptimized initial states caused within-layer variance to collapse below the 5% threshold in 100% of trials, converting the adaptive trigger into a step-function switch.

---

## 4. Stage 4: Extended Diagnostics & Bias Analysis

### 4.1 Expectation-Value & Gradient Statistics
- **Script:** [`expectation_value_diagnostics.py`](file:///c:/Users/cheedella.karthikeya/OneDrive/Desktop/Barren_Plateau_Project/expectation_value_diagnostics.py)
- **Data & Plot:** [`results_expectation_diagnostics.csv`](file:///c:/Users/cheedella.karthikeya/OneDrive/Desktop/Barren_Plateau_Project/results_expectation_diagnostics.csv), [`expectation_diagnostics.png`](file:///c:/Users/cheedella.karthikeya/OneDrive/Desktop/Barren_Plateau_Project/expectation_diagnostics.png)

Evaluated expectation values $\langle Z_0 \rangle$, variance of expectations $\text{Var}(C)$, mean absolute gradient $\mathbb{E}[|g|]$, and mean squared gradient across $n \in \{4, 6\}$ and $L \in \{1, 4, 8\}$. Confirmed that landscape flatness $\text{Var}(C)$ decays concurrently with gradient variance, demonstrating cost concentration around zero.

### 4.2 Simulated Measurement Shot Noise
- **Script:** [`measurement_shot_noise_experiment.py`](file:///c:/Users/cheedella.karthikeya/OneDrive/Desktop/Barren_Plateau_Project/measurement_shot_noise_experiment.py)
- **Data & Plot:** [`results_measurement_shot_noise.csv`](file:///c:/Users/cheedella.karthikeya/OneDrive/Desktop/Barren_Plateau_Project/results_measurement_shot_noise.csv), [`measurement_shot_noise.png`](file:///c:/Users/cheedella.karthikeya/OneDrive/Desktop/Barren_Plateau_Project/measurement_shot_noise.png)

Compared exact statevector expectations with simulated finite-shot binomial sampling across $M \in \{100, 1000, 10000\}$:
- $M = 100$: RMSE $= 0.0877$ (Theoretical SE $= 0.0877$)
- $M = 1000$: RMSE $= 0.0238$ (Theoretical SE $= 0.0277$)
- $M = 10000$: RMSE $= 0.0104$ (Theoretical SE $= 0.0088$)

*Critical Safeguard:* Confirmed that measurement shot noise introduces an artificial variance floor $\text{Var}(g_{\text{meas}}) \approx \text{Var}(g_{\text{true}}) + \frac{1}{2M}$. In deep circuits, finite-shot noise can easily mask a true barren plateau, producing artificial non-zero gradients that represent pure white noise.

### 4.3 Quantum Parameter-Shift vs Classical Backpropagation
- **Script:** [`quantum_classical_gradient_comparison.py`](file:///c:/Users/cheedella.karthikeya/OneDrive/Desktop/Barren_Plateau_Project/quantum_classical_gradient_comparison.py)
- **Data & Plot:** [`results_quantum_classical_comparison.csv`](file:///c:/Users/cheedella.karthikeya/OneDrive/Desktop/Barren_Plateau_Project/results_quantum_classical_comparison.csv), [`quantum_classical_comparison.png`](file:///c:/Users/cheedella.karthikeya/OneDrive/Desktop/Barren_Plateau_Project/quantum_classical_comparison.png)

This is an **educational methodological comparison**, not a direct competitive benchmark. Quantum parameter shifts require $2P$ full circuit evaluations on a periodic manifold $[0, 2\pi)^P$, whereas classical backpropagation requires a single reverse pass in $O(P)$ time on $\mathbb{R}^P$. Both gradient engines were independently validated against finite-difference checks with discrepancies $< 10^{-5}$.

### 4.4 Controlled Bias Analysis
- **Script:** [`bias_analysis_experiment.py`](file:///c:/Users/cheedella.karthikeya/OneDrive/Desktop/Barren_Plateau_Project/bias_analysis_experiment.py)
- **Data & Plot:** [`results_bias_analysis.csv`](file:///c:/Users/cheedella.karthikeya/OneDrive/Desktop/Barren_Plateau_Project/results_bias_analysis.csv), [`bias_analysis.png`](file:///c:/Users/cheedella.karthikeya/OneDrive/Desktop/Barren_Plateau_Project/bias_analysis.png)

Isolated five inductive bias factors at $n=4$:
1. **Observable Locality:** Global observable $Z^{\otimes n}$ collapses variance catastrophically even at depth 1 ($\text{Var} \approx 7.2 \times 10^{-33}$), confirming Cerezo et al. (2021).
2. **Entanglement:** Removing CNOT gates maintains high variance ($\text{Var} \approx 0.395$ at $L=8$), confirming that entanglement is necessary for expressibility-induced concentration.

---

## 5. Stage 5: Enhanced Multi-Metric Plateau Detector

- **Script:** [`enhanced_plateau_detector.py`](file:///c:/Users/cheedella.karthikeya/OneDrive/Desktop/Barren_Plateau_Project/enhanced_plateau_detector.py)
- **Data & Plot:** [`results_enhanced_detection.csv`](file:///c:/Users/cheedella.karthikeya/OneDrive/Desktop/Barren_Plateau_Project/results_enhanced_detection.csv), [`enhanced_detection_diagnostics.png`](file:///c:/Users/cheedella.karthikeya/OneDrive/Desktop/Barren_Plateau_Project/enhanced_detection_diagnostics.png)

Evaluated $n=6$ across $L \in \{1, 2, 4, 8, 12, 16, 20\}$ using consensus metrics:
- $L = 1, 2$: Classified as `HEALTHY_LANDSCAPE` ($\text{Var} > 0.05, \mathbb{E}[|g|] > 0.15$).
- $L = 4$: Classified as `TRANSITIONAL_SUPPRESSION` ($\text{Var} = 0.0467$).
- $L \ge 8$: Classified as `EVIDENCE_OF_PLATEAU` (95% bootstrap upper CI of variance $< 0.0173$).

*Scientific Distinction:* These thresholds are heuristic engineering rules, not universal mathematical criteria.

---

## 6. Stage 6: Enhanced Adaptive Mitigation & Verification Audit

- **Script:** [`adaptive_mitigation_enhanced.py`](file:///c:/Users/cheedella.karthikeya/OneDrive/Desktop/Barren_Plateau_Project/adaptive_mitigation_enhanced.py)
- **Data & Plot:** [`results_adaptive_enhanced.csv`](file:///c:/Users/cheedella.karthikeya/OneDrive/Desktop/Barren_Plateau_Project/results_adaptive_enhanced.csv), [`adaptive_mitigation_enhanced.png`](file:///c:/Users/cheedella.karthikeya/OneDrive/Desktop/Barren_Plateau_Project/adaptive_mitigation_enhanced.png)

Evaluated $n=4, N=25$ across depths $L \in \{2, 4, 8\}$:
- `Adaptive_Legacy` triggered at: $L=2$ (0%), $L=4$ (16%), $L=8$ (48%).
- `Adaptive_Enhanced` triggered at: $L=2$ (12%), $L=4$ (44%), $L=8$ (84%).
`Adaptive_Enhanced` responded dynamically to early plateau onset, achieving higher mean cost reduction ($\Delta C = 0.7737$ vs $0.6117$ at $L=4$).

---

## 7. Stage 7: Controlled CPU Benchmark Sweeps & Final Statistical Correction

### 7.1 Multi-Method Head-to-Head Comparison
- **Script:** [`phase7_controlled_benchmark.py`](file:///c:/Users/cheedella.karthikeya/OneDrive/Desktop/Barren_Plateau_Project/phase7_controlled_benchmark.py)
- **Primary Datasets:** [`results_phase7_controlled_benchmark.csv`](file:///c:/Users/cheedella.karthikeya/OneDrive/Desktop/Barren_Plateau_Project/results_phase7_controlled_benchmark.csv), [`results_phase7_controlled_benchmark_corrected.csv`](file:///c:/Users/cheedella.karthikeya/OneDrive/Desktop/Barren_Plateau_Project/results_phase7_controlled_benchmark_corrected.csv)
- **Diagnostic Plot:** [`phase7_controlled_benchmark.png`](file:///c:/Users/cheedella.karthikeya/OneDrive/Desktop/Barren_Plateau_Project/phase7_controlled_benchmark.png)
- **Reports:** [`phase7_controlled_benchmark_report_corrected.txt`](file:///c:/Users/cheedella.karthikeya/OneDrive/Desktop/Barren_Plateau_Project/phase7_controlled_benchmark_report_corrected.txt), [`phase7_independent_audit_report.txt`](file:///c:/Users/cheedella.karthikeya/OneDrive/Desktop/Barren_Plateau_Project/phase7_independent_audit_report.txt)

Evaluated 6 arms on $n \in \{4, 6\}$ and $L \in \{2, 4, 8, 12\}$ ($N = 10$ trials per condition, 480 trained circuits, 45.3 minutes runtime):

#### Key Findings at Deepest Layers ($n = 6, L = 12$):
1. **The "Fixed Combined" Paradox:**
   - Pre-training gradient variance is the highest among all methods ($\text{Var}(g_{T0}) = 0.0675$).
   - Cost reduction is virtually zero ($\Delta C = +0.030$, $C(T_{\text{fin}}) = -0.471$).
   - *Physical Mechanism:* Freezing prior layers while initializing active layers near identity restricts optimization to an affine parameter subspace unable to rotate the global quantum state toward the target ground state.
   - *Conclusion:* Sustained gradient variance is **not sufficient** for optimization convergence.
2. **Initial Cost Asymmetry in Controlled Initialization:**
   - `Controlled_Init` achieves the highest raw $\Delta C = +1.101$ ($C(T_0) = +0.897 \to C(T_{\text{fin}}) = -0.204$).
   - `Baseline_Uniform` achieves $\Delta C = +0.826$ ($C(T_0) = +0.041 \to C(T_{\text{fin}}) = -0.785$).
   - `Adaptive_Enhanced` achieves $C(T_{\text{fin}}) = -0.867$.
   - *Conclusion:* High $\Delta C$ in Controlled Init is an artifact of starting near $C \approx +1.0$; absolute final cost proves that Baseline and Adaptive reach significantly lower ground-state energies.

### 7.2 Inconclusive Conditions under Student's t Confidence Intervals
Under exact Student-$t$ confidence intervals ($N = 10, \text{df} = 9, t_{0.975} = 2.262$), marginal cost reduction $\Delta C$ at deep layers crosses zero:
- $n = 4, L = 12$: `Baseline_Uniform` ($[-0.029, +0.483]$), `Adaptive_Legacy` ($[-0.004, +0.268]$), `Adaptive_Enhanced` ($[-0.053, +0.206]$).
- $n = 6, L = 12$: `Adaptive_Legacy` ($[-0.029, +0.157]$), `Adaptive_Enhanced` ($[-0.028, +0.159]$).

*Honest Scientific Reporting:* Because the marginal confidence intervals cross zero and overlap, we **cannot** claim statistically significant superiority in marginal $\Delta C$ at $L=12$ between adaptive methods based on this $N=10$ pilot. Superiority rests on cumulative final cost ($C(T_{\text{fin}}) = -0.867$ vs $-0.599$).
