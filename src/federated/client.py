import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
from src.models.fedfairgnn import FedFairGNN

class Client:
    def __init__(self, client_id, data, device='cpu', 
                 fairness_weight=0.5, dp_epsilon=1.0, dp_delta=1e-5, dp_clip=1.0, heads=1):
        """
        Args:
            client_id (int): Unique ID of the client.
            data: Local Pytorch Geometric Data object.
            device: 'cpu' or 'cuda'.
            fairness_weight (lambda): Weight for fairness loss.
            dp_epsilon: Differential privacy epsilon.
            dp_delta: Differential privacy delta.
            dp_clip: Gradient clipping bound C.
            heads: Number of attention heads.
        """
        self.client_id = client_id
        self.data = data.to(device)
        self.device = device
        self.fairness_weight = fairness_weight
        self.dp_epsilon = dp_epsilon
        self.dp_delta = dp_delta
        self.dp_clip = dp_clip
        
        # Model placeholder - checking input dimensions from data
        # Assuming data.x is [N, d]
        in_channels = data.x.shape[1]
        self.model = FedFairGNN(in_channels=in_channels, 
                                hidden_channels=128, 
                                out_channels=1,
                                heads=heads).to(device)
        
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.001)

    def set_weights(self, global_weights):
        self.model.load_state_dict(global_weights)

    def get_weights(self, as_numpy=False):
        if as_numpy:
             return {k: v.cpu().numpy() for k, v in self.model.state_dict().items()}
        return copy.deepcopy(self.model.state_dict())
        
    def train_epoch(self, local_epochs=5):
        self.model.train()
        
        # For full-batch training (if graph is small enough)
        # Using a mask if available, otherwise use all nodes?
        # Algorithm doc mentions mini-batch subgraph sampling. 
        # For simplicity in this implementation, we will assume full graph batch 
        # but masked by train_mask if it fits in memory (typical for Elliptic/YelpChi on modern machines).
        # If OOM, we'd need NeighborLoader.
        
        train_mask = self.data.train_mask if hasattr(self.data, 'train_mask') else torch.ones(self.data.num_nodes, dtype=bool)
        
        for epoch in range(local_epochs):
            self.optimizer.zero_grad()
            
            # Forward pass
            out = self.model(self.data.x, self.data.edge_index, self.data.sensitive_attr)
            out = out.squeeze()
            
            # Predictions and labels on training set
            pred_train = out[train_mask]
            y_train = self.data.y[train_mask].float()
            s_train = self.data.sensitive_attr[train_mask]
            
            # 1. Task Loss (Weighted BCE)
            # Calculate class weights for this batch/epoch
            num_pos = y_train.sum()
            num_neg = (len(y_train) - num_pos) + 1e-6
            pos_weight = (len(y_train) / 2) / (num_pos + 1e-6)
            neg_weight = (len(y_train) / 2) / num_neg
            
            # Manual weighted BCE
            loss_task = - (pos_weight * y_train * torch.log(pred_train + 1e-7) + 
                           neg_weight * (1 - y_train) * torch.log(1 - pred_train + 1e-7)).mean()
            
            # 2. Fairness Loss (Soft DPD)
            mean_0 = pred_train[s_train == 0].mean() if (s_train == 0).any() else torch.tensor(0.0).to(self.device)
            mean_1 = pred_train[s_train == 1].mean() if (s_train == 1).any() else torch.tensor(0.0).to(self.device)
            dpd_soft = torch.abs(mean_0 - mean_1)
            
            # Loss definition
            loss_total = loss_task + self.fairness_weight * dpd_soft
            
            # --- FTGD Implementation ---
            self.optimizer.zero_grad()
            
            # Retain graph for double backward
            # Re-compute or use existing loss_total if graph not freed?
            # loss_total assumes graph is alive.
            loss_total.backward(retain_graph=True)
            
            # Collect total gradients
            grads_total = []
            for param in self.model.parameters():
                if param.grad is not None:
                    grads_total.append(param.grad.clone())
                else:
                    grads_total.append(torch.zeros_like(param))
            
            self.optimizer.zero_grad()
            
            # Backward for Fairness only
            (self.fairness_weight * dpd_soft).backward()
            
            grads_fair = []
            for param in self.model.parameters():
                if param.grad is not None:
                    grads_fair.append(param.grad.clone())
                else:
                    grads_fair.append(torch.zeros_like(param))
            
            # Gradient Decomposition and Noise Addition
            self.ftgd_update(grads_total, grads_fair)
            
            # Clip Beta
            self.model.clamp_beta()
            
    def ftgd_update(self, grads_total, grads_fair):
        """
        Implements Algorithm 2: FTGD
        """
        # Flatten gradients for projection
        g_vec = torch.cat([g.view(-1) for g in grads_total])
        g_fair_vec = torch.cat([g.view(-1) for g in grads_fair])
        
        # Orthogonal Decomposition
        dot = torch.dot(g_vec, g_fair_vec)
        norm_sq = torch.dot(g_fair_vec, g_fair_vec) + 1e-12
        
        proj = (dot / norm_sq) * g_fair_vec
        g_task_vec = g_vec - proj
        
        # Clip Fairness Gradient
        norm_fair = g_fair_vec.norm(2)
        clip_factor = max(1.0, norm_fair / self.dp_clip)
        g_fair_clipped = g_fair_vec / clip_factor
        
        # Add DP Noise
        sigma = self.dp_clip * torch.sqrt(torch.tensor(2.0 * torch.log(torch.tensor(1.25 / self.dp_delta)))) / self.dp_epsilon
        noise = torch.randn_like(g_fair_clipped) * sigma
        g_fair_noisy = g_fair_clipped + noise
        
        # Reconstruct final gradient: g_final = g_task + g_fair_noisy
        g_final_vec = g_task_vec + g_fair_noisy
        
        # Apply gradients back to model parameters
        idx = 0
        for param in self.model.parameters():
            numel = param.numel()
            if param.grad is not None:
                # Update param.grad with the computed noisy gradient
                param.grad.copy_(g_final_vec[idx:idx+numel].view(param.shape))
            idx += numel
            
        # Step optimizer
        self.optimizer.step()

    def get_gradients_and_metrics(self):
        """
        Prepare payload for server: (g_task, g_fair, DPD, Perf)
        Computed on Validation set.
        """
        self.model.eval()
        mask_val = self.data.val_mask if hasattr(self.data, 'val_mask') else torch.zeros(self.data.num_nodes, dtype=bool) # Fallback?
        
        # Forward on Full Graph
        out = self.model(self.data.x, self.data.edge_index, self.data.sensitive_attr).squeeze()
        
        # Metrics on Valid
        pred_val = out[mask_val]
        y_val = self.data.y[mask_val].float()
        s_val = self.data.sensitive_attr[mask_val]
        
        # DPD
        mean_0 = pred_val[s_val == 0].mean() if (s_val == 0).any() else 0.0
        mean_1 = pred_val[s_val == 1].mean() if (s_val == 1).any() else 0.0
        dpd = abs(mean_0 - mean_1)
        if isinstance(dpd, torch.Tensor): dpd = dpd.item()
        
        # Perf (AUC)
        try:
            from sklearn.metrics import roc_auc_score
            perf = roc_auc_score(y_val.cpu().detach().numpy(), pred_val.cpu().detach().numpy())
        except:
            perf = 0.5
            
        # Gradients (One pass on validation set for aggregation)
        self.optimizer.zero_grad()
        
        # Re-compute losses for gradients
        # Note: Algorithm requires gradients.
        # We need to run backward again on validation set to get the gradients to send
        self.model.train() # Set to train mode briefly to enable gran
        out = self.model(self.data.x, self.data.edge_index, self.data.sensitive_attr).squeeze()
        pred_val = out[mask_val]
        
        # Weights for validation loss (optional, usually unweighted or same valid weights)
        loss_task = F.binary_cross_entropy(pred_val, y_val) # Simple BCE for gradient vector signal
        
        mean_0 = pred_val[s_val == 0].mean()
        mean_1 = pred_val[s_val == 1].mean()
        dpd_soft = torch.abs(mean_0 - mean_1)
        
        # Double backward for separation
        loss_total = loss_task + self.fairness_weight * dpd_soft
        loss_total.backward(retain_graph=True)
        
        g_all_vec = torch.cat([p.grad.view(-1) for p in self.model.parameters() if p.grad is not None])
        
        self.optimizer.zero_grad()
        (self.fairness_weight * dpd_soft).backward()
        g_fair_vec = torch.cat([p.grad.view(-1) for p in self.model.parameters() if p.grad is not None])
        
        # Decompose
        dot = torch.dot(g_all_vec, g_fair_vec)
        norm_sq = torch.dot(g_fair_vec, g_fair_vec) + 1e-12
        proj = (dot / norm_sq) * g_fair_vec
        g_task_vec = g_all_vec - proj
        
        # Noise Fair Gradient
        norm_fair = g_fair_vec.norm(2)
        clip_factor = max(1.0, norm_fair / self.dp_clip)
        g_fair_clipped = g_fair_vec / clip_factor
        sigma = self.dp_clip * torch.sqrt(torch.tensor(2.0 * torch.log(torch.tensor(1.25 / self.dp_delta)))) / self.dp_epsilon
        g_fair_noisy = g_fair_clipped + torch.randn_like(g_fair_clipped) * sigma
        
        return {
            'g_task': g_task_vec.detach().cpu(),
            'g_fair': g_fair_noisy.detach().cpu(),
            'DPD': dpd,
            'Perf': perf,
            'n_samples': mask_val.sum().item(),
            'weights': self.get_weights() # Also send weights if needed for simpler aggregation, but algo uses gradients
        }
