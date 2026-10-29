"""
Modular Optimizer Engine for Variational Quantum Circuits (Stage 2)
===================================================================
Provides standardized, modular optimization interfaces for:
1. Vanilla Gradient Descent (Reference parameter-shift optimizer)
2. Adam (Adaptive moment estimation using parameter-shift gradients)
3. COBYLA (Constrained Optimization BY Linear Approximations, derivative-free)

Key Design Guarantees:
- Common execution protocol: optimize(cost_fn, grad_fn, init_params, ...)
- Strict circuit evaluation counting:
  * cost_fn evaluation = 1 circuit evaluation
  * full grad_fn evaluation for P parameters = 2P circuit evaluations
- Step-by-step trajectory logging:
  * iteration, cost, cost_delta (|C_t - C_{t-1}|), gradient norm, evaluations count
  * evaluation of provisional tolerance (1e-4) vs professor stated tolerance (1e4)
- Identical parameter vector inputs for paired comparisons
- Zero modifications to Phase 7 files
"""

import time
import numpy as np
from scipy.optimize import minimize


class OptimizationResult:
    """Standardized result container for all optimization algorithms."""
    def __init__(self, optimizer_name, init_params, final_params, init_cost, final_cost,
                 iterations, circuit_evals, runtime_sec, stopping_reason,
                 budget_exhausted, trajectory, final_grad_norm=None):
        self.optimizer_name = optimizer_name
        self.init_params = np.array(init_params, dtype=float)
        self.final_params = np.array(final_params, dtype=float)
        self.init_cost = float(init_cost)
        self.final_cost = float(final_cost)
        self.cost_reduction = float(init_cost - final_cost)  # positive = improvement
        self.is_improved = bool(final_cost < init_cost)
        self.iterations = int(iterations)
        self.circuit_evals = int(circuit_evals)
        self.runtime_sec = float(runtime_sec)
        self.stopping_reason = str(stopping_reason)
        self.budget_exhausted = bool(budget_exhausted)
        self.trajectory = trajectory  # list of step dicts
        self.final_grad_norm = float(final_grad_norm) if final_grad_norm is not None else np.nan

    def to_dict(self):
        return {
            "optimizer_name": self.optimizer_name,
            "init_cost": self.init_cost,
            "final_cost": self.final_cost,
            "cost_reduction": self.cost_reduction,
            "is_improved": self.is_improved,
            "iterations": self.iterations,
            "circuit_evals": self.circuit_evals,
            "runtime_sec": self.runtime_sec,
            "final_grad_norm": self.final_grad_norm,
            "stopping_reason": self.stopping_reason,
            "budget_exhausted": self.budget_exhausted
        }


# ==============================================================================
# 1. VANILLA GRADIENT DESCENT OPTIMIZER
# ==============================================================================

def run_vanilla_gd(cost_fn, grad_fn, init_params, max_iterations=50, lr=0.2,
                   target_grad_norm=1e-6):
    """
    Executes standard first-order gradient descent:
    theta_{t+1} = theta_t - lr * grad_fn(theta_t)
    """
    t_start = time.time()
    params = np.array(init_params, dtype=float).copy()
    n_params = len(params)

    # Initial cost evaluation (1 eval)
    c_current = cost_fn(params)
    trajectory = [{
        "iteration": 0,
        "cost": c_current,
        "cost_delta": 0.0,
        "grad_norm": np.nan,
        "provisional_1e4_met": False,
        "evals_so_far": 1
    }]

    c_init = c_current
    circuit_evals = 1  # For initial cost
    stopping_reason = "MAX_BUDGET_REACHED"
    budget_exhausted = True

    for it in range(1, max_iterations + 1):
        # Compute gradient (2 * n_params circuit evals)
        grads = grad_fn(params)
        circuit_evals += 2 * n_params
        grad_norm = float(np.linalg.norm(grads))

        # Update parameters
        params = params - lr * grads

        # Evaluate new cost (1 eval)
        c_new = cost_fn(params)
        circuit_evals += 1

        cost_delta = abs(c_current - c_new)
        prov_tol_met = bool(cost_delta < 1e-4)

        trajectory.append({
            "iteration": it,
            "cost": c_new,
            "cost_delta": cost_delta,
            "grad_norm": grad_norm,
            "provisional_1e4_met": prov_tol_met,
            "evals_so_far": circuit_evals
        })

        c_current = c_new

        if grad_norm < target_grad_norm:
            stopping_reason = f"GRADIENT_CONVERGED (||g|| = {grad_norm:.2e} < {target_grad_norm:.2e})"
            budget_exhausted = False
            break

    t_elapsed = time.time() - t_start

    # Final gradient norm for diagnostics
    final_grads = grad_fn(params)
    circuit_evals += 2 * n_params
    final_grad_norm = float(np.linalg.norm(final_grads))

    return OptimizationResult(
        optimizer_name="Vanilla_GD",
        init_params=init_params,
        final_params=params,
        init_cost=c_init,
        final_cost=c_current,
        iterations=len(trajectory) - 1,
        circuit_evals=circuit_evals,
        runtime_sec=t_elapsed,
        stopping_reason=stopping_reason,
        budget_exhausted=budget_exhausted,
        trajectory=trajectory,
        final_grad_norm=final_grad_norm
    )


# ==============================================================================
# 2. ADAM OPTIMIZER
# ==============================================================================

def run_adam(cost_fn, grad_fn, init_params, max_iterations=50, lr=0.05,
             beta1=0.9, beta2=0.999, eps=1e-8, target_grad_norm=1e-6):
    """
    Executes Adaptive Moment Estimation (Adam):
    m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
    v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
    m_hat = m_t / (1 - beta1^t)
    v_hat = v_t / (1 - beta2^t)
    theta_t = theta_{t-1} - lr * m_hat / (sqrt(v_hat) + eps)
    """
    t_start = time.time()
    params = np.array(init_params, dtype=float).copy()
    n_params = len(params)

    m = np.zeros(n_params, dtype=float)
    v = np.zeros(n_params, dtype=float)

    c_current = cost_fn(params)
    trajectory = [{
        "iteration": 0,
        "cost": c_current,
        "cost_delta": 0.0,
        "grad_norm": np.nan,
        "provisional_1e4_met": False,
        "evals_so_far": 1
    }]

    c_init = c_current
    circuit_evals = 1
    stopping_reason = "MAX_BUDGET_REACHED"
    budget_exhausted = True

    for it in range(1, max_iterations + 1):
        grads = grad_fn(params)
        circuit_evals += 2 * n_params
        grad_norm = float(np.linalg.norm(grads))

        # Adam moment updates
        m = beta1 * m + (1.0 - beta1) * grads
        v = beta2 * v + (1.0 - beta2) * (grads ** 2)

        m_hat = m / (1.0 - (beta1 ** it))
        v_hat = v / (1.0 - (beta2 ** it))

        # Parameter update
        params = params - lr * m_hat / (np.sqrt(v_hat) + eps)

        c_new = cost_fn(params)
        circuit_evals += 1

        cost_delta = abs(c_current - c_new)
        prov_tol_met = bool(cost_delta < 1e-4)

        trajectory.append({
            "iteration": it,
            "cost": c_new,
            "cost_delta": cost_delta,
            "grad_norm": grad_norm,
            "provisional_1e4_met": prov_tol_met,
            "evals_so_far": circuit_evals
        })

        c_current = c_new

        if grad_norm < target_grad_norm:
            stopping_reason = f"GRADIENT_CONVERGED (||g|| = {grad_norm:.2e} < {target_grad_norm:.2e})"
            budget_exhausted = False
            break

    t_elapsed = time.time() - t_start

    final_grads = grad_fn(params)
    circuit_evals += 2 * n_params
    final_grad_norm = float(np.linalg.norm(final_grads))

    return OptimizationResult(
        optimizer_name="Adam",
        init_params=init_params,
        final_params=params,
        init_cost=c_init,
        final_cost=c_current,
        iterations=len(trajectory) - 1,
        circuit_evals=circuit_evals,
        runtime_sec=t_elapsed,
        stopping_reason=stopping_reason,
        budget_exhausted=budget_exhausted,
        trajectory=trajectory,
        final_grad_norm=final_grad_norm
    )


# ==============================================================================
# 3. COBYLA OPTIMIZER (DERIVATIVE-FREE)
# ==============================================================================

def run_cobyla(cost_fn, grad_fn, init_params, max_evals=50, rhobeg=0.5, tol=1e-4):
    """
    Executes COBYLA (Constrained Optimization BY Linear Approximations):
    Black-box derivative-free optimization evaluating solely the objective function.
    """
    t_start = time.time()
    params = np.array(init_params, dtype=float).copy()
    n_params = len(params)

    c_init = cost_fn(params)
    eval_counter = [1]
    trajectory = [{
        "iteration": 0,
        "cost": c_init,
        "cost_delta": 0.0,
        "grad_norm": np.nan,
        "provisional_1e4_met": False,
        "evals_so_far": 1
    }]

    last_cost = [c_init]

    def cobyla_wrapper(p):
        c = cost_fn(p)
        eval_counter[0] += 1
        delta = abs(c - last_cost[0])
        prov_tol_met = bool(delta < 1e-4)

        trajectory.append({
            "iteration": len(trajectory),
            "cost": c,
            "cost_delta": delta,
            "grad_norm": np.nan,  # COBYLA does not compute gradients
            "provisional_1e4_met": prov_tol_met,
            "evals_so_far": eval_counter[0]
        })
        last_cost[0] = c
        return c

    opt_res = minimize(
        fun=cobyla_wrapper,
        x0=params,
        method="COBYLA",
        options={"maxiter": max_evals, "rhobeg": rhobeg, "tol": tol}
    )

    t_elapsed = time.time() - t_start
    final_params = opt_res.x
    final_cost = float(opt_res.fun)

    # Compute final gradient norm for diagnostic comparison (2 * n_params evals)
    final_grads = grad_fn(final_params)
    eval_counter[0] += 2 * n_params
    final_grad_norm = float(np.linalg.norm(final_grads))

    budget_exhausted = bool(eval_counter[0] - (2 * n_params) >= max_evals)
    stopping_reason = opt_res.message if hasattr(opt_res, "message") else "COMPLETED"

    return OptimizationResult(
        optimizer_name="COBYLA",
        init_params=init_params,
        final_params=final_params,
        init_cost=c_init,
        final_cost=final_cost,
        iterations=opt_res.nfev,
        circuit_evals=eval_counter[0],
        runtime_sec=t_elapsed,
        stopping_reason=stopping_reason,
        budget_exhausted=budget_exhausted,
        trajectory=trajectory,
        final_grad_norm=final_grad_norm
    )
