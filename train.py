import torch
import copy
from src.federated.client import Client
from src.federated.server import Server
from src.federated.baselines import SimpleClient, SimpleServer

class FederatedTrainer:
    def __init__(self, args, data_utils, device='cpu'):
        self.args = args
        self.device = device
        self.data_utils = data_utils
        
        # Load Data
        self.full_data = self.data_utils.load_dataset(args.dataset)
        self.client_data_list = self.data_utils.split_data_for_clients(self.full_data, args.num_clients)
        
        # Initialize Architecture
        if args.model == 'FedFairGNN':
            # Server with BFWA
            self.server = Server(in_channels=self.full_data.num_features,
                                 device=device,
                                 fairness_budget=args.fairness_budget,
                                 global_lr=args.global_lr,
                                 heads=args.heads)
            # Clients with FTGD
            self.clients = [Client(k, data, device, 
                                   fairness_weight=args.fairness_weight, 
                                   dp_epsilon=args.dp_epsilon,
                                   heads=args.heads) 
                            for k, data in enumerate(self.client_data_list)]
        else:
            # Baseline Mode
            self.server = SimpleServer(args, self.full_data.num_features, device)
            self.clients = [SimpleClient(k, data, device, args.model) 
                            for k, data in enumerate(self.client_data_list)]

    def run(self):
        print(f"Starting Training: {self.args.model} on {self.args.dataset}")
        
        results = []
        
        for round in range(self.args.rounds):
            print(f"\n--- Round {round+1}/{self.args.rounds} ---")
            
            # 1. Server broadcasts weights
            global_weights = self.server.get_global_weights()
            for client in self.clients:
                client.set_weights(global_weights)
            
            # 2. Clients Train locally
            updates = []
            for client in self.clients:
                # Train
                client.train_epoch(local_epochs=self.args.local_epochs)
                
                # Compute gradients/metrics
                metrics = client.get_gradients_and_metrics()
                updates.append(metrics)
                
                print(f"Client {client.client_id}: AUC={metrics['Perf']:.4f}, DPD={metrics['DPD']:.4f}")
            
            # 3. Server Aggregates
            agg_metrics = self.server.aggregate(updates)
            
            # Log results
            results.append(agg_metrics)
            
        print("Training Completed.")
        return results[-1] if len(results) > 0 else {}
