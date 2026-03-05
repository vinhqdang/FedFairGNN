# FedFairGNN Benchmark

Implementation of **FairFedGNN: Fairness-Constrained Federated Graph Neural Network for Fraud Detection**, benchmarked against state-of-the-art baselines on real-world datasets.

## 📌 Overview

This project implements a privacy-preserving and fairness-aware federated learning framework for graph fraud detection. It includes:
-   **FedFairGNN**: The core algorithm featuring:
    -   **FSER**: Fairness-Sensitive Edge Reweighting.
    -   **FTGD**: Fairness-Task Gradient Decomposition with DP.
    -   **BFWA**: Bi-Objective Frank-Wolfe Aggregation.
-   **Baselines**:
    -   `FraudGNN-RL`: RL-based neighbor selection.
    -   `GNN-CL`: Contrastive Learning GNN.
    -   `Attn-Ensemble`: Attention-gated ensemble.
-   **Datasets**: Synthetic benchmarks mimicking YelpChi, Amazon, and Elliptic properties.

## 🚀 Getting Started

### 1. Requirements

Running the setup script will create a conda environment and install dependencies.
```bash
bash environment_setup.sh
conda activate py313
```

### 2. Run Experiments

To run the full benchmark suite (all models on all datasets):
```bash
bash run_experiments.sh
```

Logs will be saved in `logs/` directory.

### 3. Usage (Main Script)

You can run individual experiments using `main.py`:

```bash
python main.py --dataset YelpChi --model FedFairGNN --num_clients 3 --rounds 20 --heads 4
```

**Arguments:**
-   `--dataset`: `YelpChi`, `Amazon`, `Elliptic`
-   `--model`: `FedFairGNN`, `FraudGNN_RL`, `GNN_CL`, `Attn_Ensemble`
-   `--num_clients`: Number of clients (default: 3)
-   `--rounds`: Communication rounds (default: 20)
-   `--fairness_budget`: Fairness constraint tau (default: 0.05)
-   `--heads`: Attention heads (default: 1, recommend 4 for FedFairGNN)

## 📊 Results Summary

| Model | AUC (YelpChi) | DPD (YelpChi) |
| :--- | :---: | :---: |
| **FedFairGNN (Fixed)** | **~0.98** | **~0.01** |
| FraudGNN-RL | ~0.84 | ~0.06 |
| GNN-CL | ~0.99 | ~0.03 |

*Note: Results based on synthetic data with injected bias and fraud signals.*

## 📂 Structure

-   `src/models`: Model definitions (FedFairGNN, Baselines).
-   `src/federated`: Federated learning components (Client, Server, BFWA, FTGD).
-   `src/utils`: Data loading and metrics.
-   `main.py`: Entry point.
-   `train.py`: Federated training loop.

## 👤 Author

**Vinh Dang**
Email: dqvinh87@gmail.com
