import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit.circuit.library import RealAmplitudes
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2 as AerEstimator

# ==========================================
# 1. BASELINE DATA & MODEL GENERATION
# ==========================================
def generate_synthetic_market_data(num_assets, seed=101):
    """Generates unique, reproducible return/risk metrics for the baseline."""
    np.random.seed(seed)
    mu = np.random.uniform(0.02, 0.15, num_assets)
    shapes = np.random.uniform(0.01, 0.10, (num_assets, num_assets))
    sigma = np.dot(shapes, shapes.T)
    return mu, sigma

def build_portfolio_model(mu, sigma, risk_factor=0.5, budget=5):
    """Constructs the raw model using a local portfolio variable mapping."""
    num_assets = len(mu)
    portfolio = QuadraticProgram("Barren_Plateau_Model")
    
    for i in range(num_assets):
        portfolio.binary_var(name=f"x_{i}")
        
    linear_coeff = {f"x_{i}": -mu[i] for i in range(num_assets)}
    quadratic_coeff = {}
    for i in range(num_assets):
        for j in range(num_assets):
            quadratic_coeff[(f"x_{i}", f"x_{j}")] = risk_factor * sigma[i, j]
            
    portfolio.minimize(linear=linear_coeff, quadratic=quadratic_coeff)
    portfolio.linear_constraint(linear={f"x_{i}": 1 for i in range(num_assets)}, sense="==", rhs=budget, name="budget")
    return portfolio

# ==========================================
# 2. PARAMETER-SHIFT RULE GRADIENT ENGINE
# ==========================================
def compute_gradient_variance(qubit_op, num_qubits, reps, num_samples=10):
    """
    Computes the statistical variance of gradients for a given circuit depth
    using the formal Parameter-Shift Rule.
    """
    ansatz = RealAmplitudes(num_qubits=num_qubits, reps=reps, entanglement='linear').decompose()
    num_parameters = ansatz.num_parameters
    from qiskit_aer import AerSimulator
    backend=AerSimulator(method="matrix_product_state")
    estimator = AerEstimator.from_backend(backend)
    
    # Store the partial derivatives collected across random parameter initializations
    collected_gradients = []
    
    # Mathematical shift parameter for the shift rule: pi / 2
    shift_val = np.pi / 2
    
    for _ in range(num_samples):
        # Sample completely random parameters across the Bloch Sphere
        random_theta = np.random.uniform(-np.pi, np.pi, num_parameters)
        
        # Pick a target parameter index near the middle of the circuit to check
        target_param_idx = num_parameters // 2
        
        # Parameter Shift Rule calculation: dE/dTheta = (E(θ + π/2) - E(θ - π/2)) / 2
        theta_plus = random_theta.copy()
        theta_plus[target_param_idx] += shift_val
        
        theta_minus = random_theta.copy()
        theta_minus[target_param_idx] -= shift_val
        
        # Evaluate expectaton values
        job_plus = estimator.run([(ansatz, qubit_op, [theta_plus])])
        energy_plus = job_plus.result()[0].data.evs[0]        
        job_minus = estimator.run([(ansatz, qubit_op, [theta_minus])])
        energy_minus = job_minus.result()[0].data.evs[0]
        
        # Formula gradient evaluation
        gradient = (energy_plus - energy_minus) / 2
        collected_gradients.append(gradient)
        
    # Return statistical variance: Var[∂E / ∂θ]
    return np.var(collected_gradients), np.mean(np.abs(collected_gradients))

# ==========================================
# 3. CORE INVESTIGATION LOOP
# ==========================================
# Scale testing parameters across Qubits (10, 20, 30) and Depths (Shallow, Med, Deep)
qubit_dimensions = [10, 20, 30]
ansatz_configurations = {
    "Shallow Ansatz (Reps=2)": 2,
    "Medium Ansatz (Reps=4)": 4,
    "Deep Ansatz (Reps=8)": 8
}

investigation_results = []

print("🚀 Starting Barren Plateau Gradient Sampling Experiment...")

for n in qubit_dimensions:
    print(f"\n--- Analyzing Qubit Universe Size: {n} Qubits ---")
    # 1. Keep the complex financial asset math lightweight (max 10)
    active_assets = 10 if n > 10 else n
    mu, sigma = generate_synthetic_market_data(num_assets=active_assets)
    portfolio_qp = build_portfolio_model(mu, sigma, budget=5)

    converter = QuadraticProgramToQubo()
    qubo = converter.convert(portfolio_qp)
    qubit_op, _ = qubo.to_ising()
        
        # 2. NEW: Pad the operator with identity gates up to the full size 'n'
    if n > active_assets:
        from qiskit.quantum_info import SparsePauliOp
            # This appends identity (I) gates to the end of the operator to match circuit size
        qubit_op = qubit_op.expand(SparsePauliOp.from_list([("I" * (n - active_assets), 1.0)]))
    
    for label, reps in ansatz_configurations.items():
        if n>=20 and reps==8:
            print(f"|Skipping {label} for {n} Qubits to avoid long wait times...")
            continue
        start_time = time.time()
        current_samples=5 if n>=20 else 25
        variance, mean_mag = compute_gradient_variance(qubit_op, num_qubits=n, reps=reps, num_samples=current_samples)
        elapsed = time.time() - start_time
        
        investigation_results.append({
            "Qubits": n,
            "Ansatz Type": label,
            "Circuit Depth (Reps)": reps,
            "Gradient Variance": variance,
            "Mean Gradient Magnitude": mean_mag,
            "Compute Time (s)": round(elapsed, 2)
        })
        print(f"│ {label} -> Variance: {variance:.2e} | Mean Magnitude: {mean_mag:.6f} | Time: {elapsed:.1f}s")

# ==========================================
# 4. GENERATE GRADIENT ANALYSIS VISUALIZATIONS
# ==========================================
plt.figure(figsize=(10, 6))
df_res = pd.DataFrame(investigation_results)

for label in ansatz_configurations.keys():
    subset = df_res[df_res["Ansatz Type"] == label]
    # Log scale is vital because Barren Plateau variances decay exponentially
    plt.yscale('log')
    plt.plot(subset["Qubits"], subset["Gradient Variance"], marker='o', linewidth=2, label=label)

plt.title("Barren Plateau Analysis: Gradient Variance Decay vs Qubit Count", fontsize=12, fontweight='bold')
plt.xlabel("Number of Qubits (Asset Count)", fontsize=10)
plt.ylabel("Variance of Partial Derivatives $Var[\partial_i E]$ (Log Scale)", fontsize=10)
plt.xticks(qubit_dimensions)
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.savefig("gradient_variance_decay.png")
plt.show()

# ==========================================
# 5. PRINT FORMAL REPORT TABLE
# ==========================================
print("\n" + "="*75 + "\n BARREN PLATEAU RESEARCH METRICS SUMMARY (SHRIYA PATHAK - TASK 2) \n" + "="*75)
print(df_res[["Qubits", "Ansatz Type", "Gradient Variance", "Mean Gradient Magnitude", "Compute Time (s)"]].to_markdown(index=False))