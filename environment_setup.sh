#!/bin/bash
# Environment setup script for FedFairGNN

# Ensure we are in the right conda environment
# (User specified py313)

echo "Installing dependencies..."

# Install PyTorch (assuming CPU for now, can be adjusted for CUDA/MPS)
# Using pip within the conda environment
pip install torch torchvision torchaudio

# Install PyTorch Geometric and dependencies
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.0+cpu.html

# Install other requirements
pip install pandas numpy scikit-learn matplotlib networkx scipy

# Install autodp for differential privacy
pip install autodp

echo "Dependencies installed."
