#!/bin/bash
# Script to run FedFairGNN and Baselines Experiments

MODELS=("FedFairGNN" "FraudGNN_RL" "GNN_CL" "Attn_Ensemble")
DATASETS=("YelpChi" "Amazon" "Elliptic")

# Output directory for logs
mkdir -p logs

echo "Starting Experiments..."

for dataset in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
        echo "----------------------------------------------------------------"
        echo "Running Model: $model on Dataset: $dataset"
        echo "----------------------------------------------------------------"
        
        # Run python script and pipe output to log file
        python main.py \
            --dataset $dataset \
            --model $model \
            --rounds 5 \
            --local_epochs 1 \
            --num_clients 3 \
            --heads 4 \
             > "logs/${model}_${dataset}.log" 2>&1
             
        # Print last few lines of log to see result
        tail -n 5 "logs/${model}_${dataset}.log"
    done
done

echo "All experiments finished. Check logs/ directory."
