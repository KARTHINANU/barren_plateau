import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def exp_decay(n, A, b):
    return A * np.exp(-b * n)

def main():
    qubits = []
    variances = []
    
    with open("results_baseline_qubits.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qubits.append(int(row["n_qubits"]))
            variances.append(float(row["gradient_variance"]))
            
    qubits = np.array(qubits)
    variances = np.array(variances)
    
    # Fit the data
    # Provide an initial guess p0 to help the optimizer
    p0 = [variances[0] * np.exp(qubits[0]), 1.0]
    popt, pcov = curve_fit(exp_decay, qubits, variances, p0=p0)
    
    A, b = popt
    
    print("Exponential Fit Results:")
    print("Model: Var = A * exp(-b * n)")
    print(f"A = {A:.6f}")
    print(f"b = {b:.6f}")
    
    # Generate points for smooth curve
    n_smooth = np.linspace(min(qubits), max(qubits), 100)
    var_smooth = exp_decay(n_smooth, A, b)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(qubits, variances, 'o', label='Measured Variance', color='crimson')
    ax.plot(n_smooth, var_smooth, '-', label=f'Fit: {A:.4f} * exp(-{b:.4f}n)', color='navy')
    
    ax.set_xlabel("Number of Qubits (n)")
    ax.set_ylabel("Gradient Variance")
    ax.set_title("Barren Plateau: Exponential Fit of Variance vs Qubits")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("exponential_fit_qubits.png", dpi=150)
    print("Saved plot to exponential_fit_qubits.png")

if __name__ == "__main__":
    main()
