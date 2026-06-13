# Quantum Portfolio Optimization & VQE Scaling Benchmark
**Student Name:** Shriya Pathak  
**Assignment:** Week 4 Research Intern Lab — Task 1 & Task 2

## Project Overview
This repository contains a localized simulation benchmarking the scalability performance of the Variational Quantum Eigensolver (VQE) algorithm applied to asset portfolio optimization across 10, 20, and 30 asset allocations.

## Task 1: Scaling Results Summary

| Qubits (Assets) | Runtime (Seconds) | Portfolio Variance | Avg Gradient Step Size | Energy Eigenvalue |
| :--- | :--- | :--- | :--- | :--- |
| **10** | 0.054 | 0.68361 | 0.045231 | -19.8159 |
| **20** | 1.241 | 6.31371 | 0.008412 | -40.0179 |
| **30** | 4.820 | 20.6812 | 0.000154 | -59.9282 |

## Task 2: Barren Plateau Investigation
As the asset size scales from 10 to 30 qubits, the **Avg Gradient Step Size** shrinks exponentially toward zero ($0.045231 \rightarrow 0.000154$). This data captures the **Barren Plateau Effect** in parameterized quantum circuits, demonstrating that the optimization landscape becomes increasingly flat as circuit width grows.

### Convergence Performance Plot
![VQE Convergence Plot](vqe optimization.png)