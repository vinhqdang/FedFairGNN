import torch
import torch.optim as optim
import copy
import torch.nn.functional as F
from src.models.baselines import FraudGNN_RL, GNN_CL, Attn_Ensemble

class SimpleServer:
    def __init__(self, args, in_channels, device):
        self.device = device
        self.args = args
        
        if args.model == 'FraudGNN_RL':
            self.model = FraudGNN_RL(in_channels, 128, 1).to(device)
        elif args.model == 'GNN_CL':
            self.model = GNN_CL(in_channels, 128, 1).to(device)
        elif args.model == 'Attn_Ensemble':
            self.model = Attn_Ensemble(in_channels, 128, 1).to(device)
        else:
             raise ValueError(f"Unknown model {args.model}")
             
        self.global_weights = self.model.state_dict()
        self.lr = args.global_lr

    def get_global_weights(self):
        return copy.deepcopy(self.global_weights)

    def aggregate(self, updates):
        # Standard FedAvg
        K = len(updates)
        new_state_dict = copy.deepcopy(updates[0]['weights'])
        
        # Initialize with first client's weights * (1/K) ? 
        # Or sum then divide.
        # Deepcopy first, then add others? No.
        # Correct FedAvg: sum(w_i) / K
        
        # Init accumulator with 0
        for key in new_state_dict.keys():
            new_state_dict[key] = torch.zeros_like(new_state_dict[key])
            
        for update in updates:
            for key in new_state_dict.keys():
                new_state_dict[key] += update['weights'][key]
                
        # Divide by K
        for key in new_state_dict.keys():
            new_state_dict[key] = new_state_dict[key] / K
            
        self.model.load_state_dict(new_state_dict)
        self.global_weights = copy.deepcopy(new_state_dict)
        
        # Avg Metrics
        avg_auc = sum([u['Perf'] for u in updates]) / K
        avg_dpd = sum([u['DPD'] for u in updates]) / K
        
        # If any tensor, item()
        if isinstance(avg_auc, torch.Tensor): avg_auc = avg_auc.item()
        if isinstance(avg_dpd, torch.Tensor): avg_dpd = avg_dpd.item()
        
        print(f"Aggregation (FedAvg): AUC={avg_auc:.4f}, DPD={avg_dpd:.4f}")
        return {'Global_Perf': avg_auc, 'Global_DPD': avg_dpd}


class SimpleClient:
    def __init__(self, client_id, data, device, model_name):
        self.client_id = client_id
        self.data = data.to(device)
        self.device = device
        
        in_channels = data.x.shape[1] # Use data.x feature dim
        
        if model_name == 'FraudGNN_RL':
            self.model = FraudGNN_RL(in_channels, 128, 1).to(device)
        elif model_name == 'GNN_CL':
            self.model = GNN_CL(in_channels, 128, 1).to(device)
        elif model_name == 'Attn_Ensemble':
            self.model = Attn_Ensemble(in_channels, 128, 1).to(device)
        else:
            raise ValueError(f"Unknown model {model_name}")
            
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def set_weights(self, weights):
        self.model.load_state_dict(weights)

    def train_epoch(self, local_epochs=5):
        self.model.train()
        mask = self.data.train_mask
        
        for epoch in range(local_epochs):
            self.optimizer.zero_grad()
            
            # Forward
            # Check model signature
            # All baselines take (x, edge_index, sensitive_attr=None)
            out = self.model(self.data.x, self.data.edge_index).squeeze()
            
            # Loss
            loss = F.binary_cross_entropy(out[mask], self.data.y[mask].float())
            
            # GNN-CL Aux Loss
            if hasattr(self.model, 'cl_loss'):
                 loss += 0.1 * self.model.cl_loss(self.data.edge_index)
                 
            loss.backward()
            self.optimizer.step()

    def get_gradients_and_metrics(self):
        self.model.eval()
        mask = self.data.val_mask
        
        # Forward
        out = self.model(self.data.x, self.data.edge_index).squeeze()
        
        pred = out[mask]
        y = self.data.y[mask].float()
        s = self.data.sensitive_attr[mask]
        
        # Metrics
        mean_0 = pred[s==0].mean() if (s==0).any() else 0.0
        mean_1 = pred[s==1].mean() if (s==1).any() else 0.0
        dpd = torch.abs(torch.tensor(mean_0) - torch.tensor(mean_1)).item()
        
        from sklearn.metrics import roc_auc_score
        try:
             perf = roc_auc_score(y.detach().cpu().numpy(), pred.detach().cpu().numpy())
        except:
             perf = 0.5
             
        # Return weights for FedAvg
        return {
            'Perf': perf,
            'DPD': dpd,
            'weights': copy.deepcopy(self.model.state_dict())
        }
