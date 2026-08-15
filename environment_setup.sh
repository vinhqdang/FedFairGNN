#!/bin/bash
# Environment setup for TrustFedGNN.
#
# Everything in the paper runs on CPU; GPU wheels are only worth it for the
# ogbn-products scalability study. Run from the repository root.

set -e

echo "Installing PyTorch (CPU wheels)..."
pip install torch --index-url https://download.pytorch.org/whl/cpu

echo "Installing PyTorch Geometric..."
pip install torch_geometric

echo "Installing project requirements..."
pip install -r requirements.txt

# The neighbour-sampling path used by the million-node study (--sampling) needs
# the compiled PyG extensions. Match the wheel URL to your torch version:
#   pip install torch-scatter torch-sparse \
#     -f https://data.pyg.org/whl/torch-$(python -c 'import torch;print(torch.__version__)')+cpu.html

echo
echo "Done. Verify with:  python -m pytest -q"
