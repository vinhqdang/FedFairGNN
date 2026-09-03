# Remote GPU Execution Workflow via Google Colab

This document details the remote execution workflow for offloading compute-heavy graph neural network experiments (such as Pokec-z with 1.24M edges, Elliptic with 203k transactions, and ogbn-products with 2.4M nodes) to Google Colab GPU instances.

---

## 1. Overview & Directory Structure

The `colab/` directory within `FedFairGNN` contains the automation scripts for packaging, transferring, and running experiments remotely:

```
colab/
├── 00_pack.sh                 # Packages the codebase into a clean tarball
├── 01_setup.py                # VM environment bootstrap (PyTorch, PyG, dependencies)
├── 11_stage4_1_smoke.py       # Fast remote smoke test (Synthetic N=160)
├── 15_stage4_remediation.py   # Multi-seed baseline and ablation execution script
├── run_local.sh               # Local helper script to trigger remote execution
└── archived/                  # Historical stage scripts (stages 4.2 - 4.5)
```

---

## 2. Remote Workflow Steps

### Step 1: Package the Codebase
Run the packaging script to create a standalone bundle of `FedFairGNN`, excluding `.git`, `__pycache__`, and temporary artifacts:
```bash
bash colab/00_pack.sh
```
This produces `bundle.tar.gz` ready for upload.

### Step 2: Bootstrap Remote Environment
On the remote Colab GPU VM:
```bash
python colab/01_setup.py
```
This script installs PyTorch Geometric wheels, checks CUDA availability (`torch.cuda.is_available()`), and prepares the directory layout.

### Step 3: Launch Training Runs
Execute the designated training workload:
```bash
# Example: Run 5-seed baseline matrix
python colab/15_stage4_remediation.py
```

### Step 4: Sync Results Back
Download the generated output files from `/content/results/` back to the local `FedFairGNN/results/` directory for analysis and report generation.

---

## 3. Academic Presentation Principle

> [!NOTE]
> In academic papers and journal submissions, we describe the computing hardware objectively (e.g., *"Commodity cloud instances equipped with an NVIDIA T4 / A100 GPU and 16GB VRAM"*). We do not reference internal orchestration scripts or platform-specific plumbing in the formal manuscript text.
