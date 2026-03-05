import torch
import numpy as np
import copy
from src.models.fedfairgnn import FedFairGNN

class Server:
    def __init__(self, in_channels, device='cpu', fairness_budget=0.05, 
                 global_lr=0.1, dual_step_size=0.1, fw_iterations=10, heads=1):
        """
        Args:
            in_channels: Input feature dimension.
            device: 'cpu' or 'cuda'.
            fairness_budget: Tensor or float (tau).
            global_lr: Global learning rate.
            dual_step_size: Step size for dual variable update.
            fw_iterations: Number of Frank-Wolfe iterations.
            heads: Number of attention heads.
        """
        self.device = device
        self.fairness_budget = fairness_budget
        self.global_lr = global_lr
        self.dual_step_size = dual_step_size
        self.fw_iterations = fw_iterations
        
        # Initialize Global Model
        self.model = FedFairGNN(in_channels=in_channels, 
                                hidden_channels=128, 
                                out_channels=1,
                                heads=heads).to(device)
        self.global_weights = self.model.state_dict()
        
        # Dual variable mu (Lagrange multiplier)
        self.mu = 0.0

    def get_global_weights(self):
        return copy.deepcopy(self.global_weights)

    def aggregate(self, updates):
        """
        BFWA: Bi-Objective Frank-Wolfe Aggregation.
        
        Args:
            updates: List of dicts from clients containing:
                     'g_task', 'g_fair', 'DPD', 'Perf'
        """
        K = len(updates)
        if K == 0:
            return
            
        # Extract metrics
        # shape: [K]
        dpd_vals = torch.tensor([u['DPD'] for u in updates], dtype=torch.float32)
        perf_vals = torch.tensor([u['Perf'] for u in updates], dtype=torch.float32)
        
        # Initialize weights (uniform)
        w = torch.ones(K, dtype=torch.float32) / K
        
        # Frank-Wolfe Optimization
        for t in range(self.fw_iterations):
            # 1. Compute Constraint Violation
            # Global DPD = sum(w_k * DPD_k)
            dpd_global = torch.dot(w, dpd_vals)
            violation = dpd_global - self.fairness_budget
            
            # 2. Gradient of Lagrangian w.r.t w
            # L(w, mu) = - sum(w_k * Perf_k) + mu * (sum(w_k * DPD_k) - tau)
            # grad_k = -Perf_k + mu * DPD_k
            grad_F = -perf_vals + self.mu * dpd_vals
            
            # 3. LMO (Linear Minimization Oracle)
            # Minimize s @ grad_F over simplex
            # Solution: s is one-hot at argmin(grad_F)
            min_idx = torch.argmin(grad_F)
            s = torch.zeros(K)
            s[min_idx] = 1.0
            
            # 4. Update
            gamma = 0.1 # Fixed step size for stability, prevents extreme weights
            w = (1 - gamma) * w + gamma * s
            
            # 5. Dual Update
            # Ascent on mu: mu = max(0, mu + lr * violation)
            self.mu = max(0.0, self.mu + self.dual_step_size * violation.item())
            
        # Ensure minimum weight floor (optional, for stability)
        w_min = 1.0 / (5 * K)
        w = torch.clamp(w, min=w_min)
        w = w / w.sum()
        
        print(f"Aggregation Weights: {w.numpy()}")
        print(f"Dual variable mu: {self.mu}")
        
        # Apply Aggregation to Weights directly
        # Delta_theta = sum(w_k * (theta_k))
        
        state_dict = self.model.state_dict()
        new_state_dict = copy.deepcopy(state_dict)
        
        for name in new_state_dict.keys():
            if torch.is_floating_point(new_state_dict[name]):
                new_state_dict[name] = torch.zeros_like(new_state_dict[name])
                for k in range(K):
                    new_state_dict[name] += w[k].item() * updates[k]['weights'][name]

        self.model.load_state_dict(new_state_dict)
        self.global_weights = copy.deepcopy(new_state_dict)
        
        return {
            'Global_DPD': torch.dot(w, dpd_vals).item(),
            'Global_Perf': torch.dot(w, perf_vals).item()
        }
