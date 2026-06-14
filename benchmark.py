import os
from pathlib import Path
import mlflow
import numpy as np

# 1. DEFINE PATHS STRIP-CLEANED FOR WINDOWS / CROSS-PLATFORM
# Using pathlib ensures robust URI generation for SQLite across Windows and Linux
current_dir = Path(__file__).resolve().parent
db_path = current_dir / "mlflow.db"
LOCAL_URI = f"sqlite:///{db_path.as_posix()}"

# 2. DATA ENGINEERING: GENERATE FINANCIAL DATA
def generate_market_data(num_assets=4):
    np.random.seed(42)
    returns = np.random.uniform(0.05, 0.20, num_assets)
    A = np.random.randn(num_assets, num_assets)
    cov_matrix = np.dot(A, A.T) * 0.05
    return returns, cov_matrix

# 3. METRICS EVALUATION ENGINE
def calculate_metrics(weights, returns, cov_matrix, execution_time, is_feasible, optimal_value, baseline_value):
    weights = np.array(weights)
    weights /= np.sum(weights)
    
    portfolio_return = np.dot(weights, returns)
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    risk_free_rate = 0.02
    sharpe_ratio = (portfolio_return - risk_free_rate) / np.sqrt(portfolio_variance)
    
    feasibility_rate = 1.0 if is_feasible else 0.0
    approximation_ratio = optimal_value / baseline_value if baseline_value != 0 else 1.0
    
    return {
        "Runtime": execution_time,
        "Sharpe Ratio": sharpe_ratio,
        "Portfolio Variance": portfolio_variance,
        "Feasibility Rate": feasibility_rate,
        "Approximation Ratio": approximation_ratio
    }

# 4. RUN AND LOG EXPERIMENT
def run_benchmark_run(algo_name, weights, returns, cov_matrix, exec_time, is_feasible, score, baseline_score):
    # Safe cleanup: End any lingering active runs before setting tracking URI
    if mlflow.active_run():
        mlflow.end_run()
        
    # Set configurations directly before running
    mlflow.set_tracking_uri(LOCAL_URI)
    
    # Automatically finds or creates the experiment safely
    experiment_name = "Quantum_Portfolio_Optimization_Benchmark"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        exp_id = mlflow.create_experiment(experiment_name)
    else:
        exp_id = experiment.experiment_id

    # Start run explicitly attached to our safe experiment ID
    with mlflow.start_run(experiment_id=exp_id, run_name=algo_name):
        mlflow.log_param("algorithm", algo_name)
        mlflow.log_param("num_assets", len(returns))
        
        metrics = calculate_metrics(weights, returns, cov_matrix, exec_time, is_feasible, score, baseline_score)
        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name, value)
            
        print(f"✅ Successfully logged metrics for {algo_name}")

# MAIN PIPELINE EXECUTION
if __name__ == "__main__":
    returns, cov = generate_market_data(num_assets=4)
    exact_classical_score = 0.85 
    
    run_benchmark_run(
        algo_name="Classical_BFGS",
        weights=[0.4, 0.3, 0.2, 0.1],
        returns=returns, cov_matrix=cov,
        exec_time=0.05, is_feasible=True, score=0.85, baseline_score=exact_classical_score
    )
    
    run_benchmark_run(
        algo_name="VQE_Simulated",
        weights=[0.35, 0.35, 0.15, 0.15],
        returns=returns, cov_matrix=cov,
        exec_time=12.4, is_feasible=True, score=0.81, baseline_score=exact_classical_score
    )