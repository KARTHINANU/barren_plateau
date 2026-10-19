# Experimental Methodology & Implementation Standards

**Project Title:** Demonstrating and Mitigating Barren Plateaus in Random Circuits  
**Document:** Methodology, Protocols, and Mathematical Formulations (`docs/experimental_methodology.md`)  
**Status:** Verified Complete & Methodologically Audited  

---

## 1. Quantum Circuit Architecture & Ansatz Specification

All quantum circuits in this project are constructed using a parameterized hardware-efficient ansatz designed to model typical variational quantum algorithms (VQAs) and quantum neural networks (QNNs):

### 1.1 Gate Sequence
For a circuit with $n$ qubits and depth $L$ (layers):
1. **Parameterized Rotation Layer:** Each layer $l \in \{1, \dots, L\}$ applies single-qubit rotations on every qubit $q \in \{0, \dots, n-1\}$:
   $$U_l(\boldsymbol{\theta}_l) = \bigotimes_{q=0}^{n-1} R_z(\theta_{l, q, 3}) R_y(\theta_{l, q, 2}) R_x(\theta_{l, q, 1})$$
   where each qubit receives 3 independent rotational parameters, yielding $3n$ parameters per layer.
2. **Entangling Block:** Following each rotation layer, a nearest-neighbor linear entangling cascade of Controlled-NOT (CNOT) gates is applied:
   $$W = \prod_{q=0}^{n-2} \text{CNOT}(q, q+1)$$
   This generates entanglement across the register in $O(n)$ depth without wrap-around periodic boundary conditions.
3. **Total Parameter Dimension:** A circuit of $n$ qubits and depth $L$ contains $P = 3nL$ trainable parameters.

### 1.2 Simulation Paradigm: Exact Statevector vs Finite-Shot Sampling
- **Primary Simulation Engine:** Exact statevector evolution via Qiskit's `Statevector.from_instruction(qc)`. This computes the full $2^n$-dimensional complex state vector $|\psi(\boldsymbol{\theta})\rangle$ numerically through dense linear algebra, completely removing physical hardware decoherence, gate infidelities, and readout errors.
- **Finite-Shot Diagnostic Simulation:** In `measurement_shot_noise_experiment.py`, measurement sampling is simulated on classical CPUs using a multinomial pseudo-random draw according to Born's rule probabilities $p_i = |\langle i | \psi \rangle|^2$. **No physical quantum hardware was accessed or claimed.**

---

## 2. Observable, Cost Function & Minimization Objective

### 2.1 Local Pauli Observable
Throughout all benchmarking and diagnostic modules, the target observable is the local single-qubit Pauli-Z operator acting on qubit 0:
$$O = Z_0 = Z \otimes I^{\otimes (n-1)}$$
In Qiskit's reverse little-endian representation (where qubit 0 corresponds to the rightmost character in the string), the operator is represented as:
$$O = \text{SparsePauliOp}("I" \times (n-1) + "Z")$$

### 2.2 Objective Function
The cost function $C(\boldsymbol{\theta})$ is the exact quantum expectation value:
$$C(\boldsymbol{\theta}) = \langle \psi(\boldsymbol{\theta}) | Z_0 | \psi(\boldsymbol{\theta}) \rangle = \text{Re}\left( \langle \psi(\boldsymbol{\theta}) | Z_0 | \psi(\boldsymbol{\theta}) \rangle \right)$$
Bounded within $[-1.0, +1.0]$:
- Theoretical maximum: $+1.0$ (when qubit 0 is in state $|0\rangle$).
- Theoretical minimum (ground state): $-1.0$ (when qubit 0 is in state $|1\rangle$).
- Haar-random expectation: $\mathbb{E}_{\boldsymbol{\theta}}[C(\boldsymbol{\theta})] = 0.0$.

Minimization drives $C(\boldsymbol{\theta}) \to -1.0$.

---

## 3. Gradient Evaluation & Parameter Updates

### 3.1 Exact Parameter-Shift Rule
Because the generator of each single-qubit rotation $R_{\sigma}(\theta) = \exp(-i \theta \sigma / 2)$ has eigenvalues $\pm 1/2$, the partial derivative with respect to any parameter $\theta_k$ is computed analytically without finite-difference discretization error:
$$\frac{\partial C}{\partial \theta_k} = \frac{C\left(\boldsymbol{\theta} + \frac{\pi}{2}\mathbf{e}_k\right) - C\left(\boldsymbol{\theta} - \frac{\pi}{2}\mathbf{e}_k\right)}{2}$$
where $\mathbf{e}_k$ is the standard unit basis vector.

### 3.2 Optimization Update Rule
All trained arms utilize first-order gradient descent:
$$\boldsymbol{\theta}^{(t+1)} = \boldsymbol{\theta}^{(t)} - \eta \, \nabla C(\boldsymbol{\theta}^{(t)})$$
with a fixed learning rate of $\eta = 0.5$.
Because $\nabla C$ is subtracted, the update direction strictly minimizes $C(\boldsymbol{\theta})$.

### 3.3 Training Budget & Evaluation Counting
- **Full-Circuit Methods:** Receive 5 optimization steps updating all $3nL$ parameters concurrently.
- **Layerwise Methods:** Receive 5 optimization steps on the active layer ($3n$ parameters) across each of the $L$ stages, while freezing prior layers.
- **Circuit Call Parity:**
  $$\text{Full-Circuit:} \quad 5 \text{ steps} \times (2 \times 3nL) = 30nL \text{ gradient circuit calls}$$
  $$\text{Layerwise:} \quad L \text{ layers} \times [5 \text{ steps} \times (2 \times 3n)] = 30nL \text{ gradient circuit calls}$$
  Both optimization paradigms consume an identical budget of parameter-shift gradient evaluations.

---

## 4. Parameter Distributions & Paired Random Streams

To isolate algorithmic mitigation from random sampling variation, paired deterministic pseudo-random number generators (PRNGs) were established:

1. **Uniform Initialization ($\mathcal{U}[0, 2\pi)$):**
   Parameters are drawn uniformly from $[0, 2\pi)$, generating Haar-distributed states with exponentially vanishing gradient variance.
2. **Controlled Near-Zero Initialization ($\mathcal{N}(0, 0.1^2)$):**
   Parameters are drawn from a normal distribution centered at zero with standard deviation $\sigma = 0.1$, placing the circuit near the identity operator and the initial state near $|00\dots0\rangle$.
3. **Paired Seed Protocol:**
   For trial $i \in \{0, \dots, N-1\}$, deterministic seeds are instantiated:
   $$\text{seed}_u = \text{base\_seed} + i, \quad \text{seed}_n = \text{base\_seed} + 100000 + i$$
   Layer $l$ receives identical master slices across all corresponding benchmark arms, guaranteeing identical random angles across comparable arms.

---

## 5. Mitigation & Adaptive Detection Logic

### 5.1 Original 5% Variance Trigger (`Adaptive_Legacy`)
Evaluates the spatial sample variance of the active layer gradients at layer $l$:
$$v_{\text{curr}} = \text{Var}\left(\left\{ \frac{\partial C}{\partial \theta_{l, k}} \right\}_{k=1}^{3n}\right)$$
- Reference statistic: $v_{\text{ref}} = v_{\text{curr}}$ at layer $l=1$.
- Trigger condition: If $l > 1$ and $v_{\text{curr}} < 0.05 \cdot v_{\text{ref}}$, mitigation activates.
- Response: Active layer parameters are discarded and re-initialized from the paired normal stream $\mathcal{N}(0, 0.1^2)$, and subsequent layers are initialized normally with prior layers frozen.

### 5.2 Enhanced Multi-Metric Consensus Detector (`Adaptive_Enhanced`)
Replaces the single spatial variance check with a multi-metric heuristic evaluating gradient magnitude, variance, and loss stagnation:
$$\text{Trigger} = \left(\mathbb{E}[|g|] < 0.06 \land \text{Var}(g) < 0.01\right) \lor \left(|C_{\text{prior}} - C_{\text{curr}}| < 0.002 \land \mathbb{E}[|g|] < 0.10\right)$$
- Overcomes the limitation of the 5% legacy rule, which triggered prematurely on small statistical fluctuations in unconditioned circuits.

---

## 6. Statistical Formulations & Corrected Confidence Intervals

### 6.1 Sample Statistics ($N = 10$)
Because Phase 7 operates with an exploratory pilot sample of $N = 10$ trials per condition, statistical calculations were strictly corrected to eliminate small-sample bias:

1. **Sample Standard Deviation ($\text{ddof} = 1$):**
   $$s = \sqrt{\frac{1}{N - 1} \sum_{i=1}^N (x_i - \bar{x})^2} = \sigma_{\text{pop}} \sqrt{\frac{N}{N-1}} = \sigma_{\text{pop}} \sqrt{\frac{10}{9}}$$
2. **Standard Error of the Mean:**
   $$\text{SE} = \frac{s}{\sqrt{N}} = \frac{s}{\sqrt{10}}$$
3. **Exact Student's $t$ Multiplier:**
   For degrees of freedom $\nu = N - 1 = 9$ at two-sided $\alpha = 0.05$:
   $$t_{0.975, 9} = 2.262157 \quad (\text{vs. large-sample } z = 1.960000)$$
4. **Corrected 95% Confidence Interval:**
   $$\text{CI}_{95\%} = \left[ \bar{x} - t_{0.975, 9} \cdot \text{SE}, \quad \bar{x} + t_{0.975, 9} \cdot \text{SE} \right]$$

### 6.2 Distinction of Optimization Metrics
- **Initial Cost $C(T_0)$:** Cost before optimization. For full-circuit methods, measured at depth $L$. For layerwise methods, measured at the entrance of layer $L$ after layers $1 \dots L-1$ have been trained.
- **Final Cost $C(T_{\text{fin}})$:** Absolute state expectation value after optimization terminates (Target = $-1.0$).
- **Cost Reduction $\Delta C$:** Marginal cost reduction during the final training round:
  $$\Delta C = C(T_0) - C(T_{\text{fin}})$$
- **Reporting Rule:** Initial cost, final absolute cost, and cost reduction must always be reported together to prevent misinterpreting marginal progress as cumulative optimization.
