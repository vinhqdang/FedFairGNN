import argparse
import torch
from train import FederatedTrainer
from src.utils.data_utils import DataUtils

def main():
    parser = argparse.ArgumentParser(description='FedFairGNN Benchmark')
    
    # Experiment Settings
    parser.add_argument('--dataset', type=str, default='YelpChi', 
                        choices=['YelpChi', 'Amazon', 'Elliptic'],
                        help='Dataset name')
    parser.add_argument('--model', type=str, default='FedFairGNN', 
                        choices=['FedFairGNN', 'FraudGNN_RL', 'GNN_CL', 'Attn_Ensemble'],
                        help='Model name')
    parser.add_argument('--num_clients', type=int, default=3, help='Number of federated clients')
    parser.add_argument('--rounds', type=int, default=20, help='Communication rounds')
    parser.add_argument('--local_epochs', type=int, default=2, help='Local epochs per round')
    
    # FedFairGNN Hyperparams
    parser.add_argument('--fairness_budget', type=float, default=0.05, help='Tau: Fairness budget')
    parser.add_argument('--fairness_weight', type=float, default=1.0, help='Lambda: Fairness loss weight')
    parser.add_argument('--dp_epsilon', type=float, default=10.0, help='Privacy budget (higher=less noise)')
    parser.add_argument('--global_lr', type=float, default=0.1, help='Global learning rate')
    parser.add_argument('--heads', type=int, default=1, help='Number of attention heads')
    
    args = parser.parse_args()
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Data Utils
    data_utils = DataUtils()
    
    # Trainer
    trainer = FederatedTrainer(args, data_utils, device)
    trainer.run()

if __name__ == '__main__':
    main()
