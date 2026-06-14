import time
import numpy as np
import mlflow

# Initialize or connect to an MLflow experiment
mlflow.set_experiment("Quantum_Portfolio_Optimization")

def log_quantum_experiment(model_name, asset_size, noise_level, circuit_depth):
    """Logs quantum metrics to MLflow server runs."""
    run_name = f"{model_name}_Assets_{asset_size}_Noise_{noise_level}"
    
    with mlflow.start_run(run_name=run_name):
        # 1. Log System Parameters (Tags & Params)
        mlflow.set_tag("Algorithm", model_name)
        mlflow.log_param("Asset Universe Size", asset_size)
        mlflow.log_param("Noise Level", noise_level)
        mlflow.log_param("Circuit Depth", circuit_depth)
        
        print(f"Logging metrics for {run_name}...")
        
        # 2. Simulate Optimization Iterations (Convergence Curve)
        iterations = 20
        base_loss = 0.8 if model_name == "QSVC" else 15.5
        
        for step in range(iterations):
            time.sleep(0.05)  # Simulate computing time
            
            # Simulated convergence decay + random noise penalty
            decay = np.exp(-step / 5.0)
            noise_penalty = np.random.normal(0, noise_level * 0.1)
            current_loss = base_loss * decay + abs(noise_penalty)
            
            # Log metrics per iteration step to construct convergence curves
            mlflow.log_metric("loss_convergence", current_loss, step=step)
            
        # 3. Log final evaluation summaries
        final_accuracy = 0.5 + (0.4 / (noise_level + 1)) if model_name == "QSVC" else None
        if final_accuracy:
            mlflow.log_metric("Final_Accuracy", final_accuracy)
            
        print(f"Successfully logged {model_name} metrics to MLflow.")

# Run tracking passes to generate dashboards for QAOA, VQE, and QSVC
if __name__ == "__main__":
    # Dashboard Config 1: QAOA Scaling Check
    log_quantum_experiment(model_name="QAOA", asset_size=10, noise_level=0.05, circuit_depth=4)
    log_quantum_experiment(model_name="QAOA", asset_size=20, noise_level=0.10, circuit_depth=8)
    
    # Dashboard Config 2: VQE Scaling Check
    log_quantum_experiment(model_name="VQE", asset_size=10, noise_level=0.01, circuit_depth=6)
    log_quantum_experiment(model_name="VQE", asset_size=30, noise_level=0.15, circuit_depth=12)
    
    # Dashboard Config 3: QSVC Machine Learning Check
    log_quantum_experiment(model_name="QSVC", asset_size=2, noise_level=0.00, circuit_depth=2)