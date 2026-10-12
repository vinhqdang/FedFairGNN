import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv
from torch_geometric.nn import MessagePassing

# Ref: FraudGNN-RL (Simplified)
# Uses a policy network (MLP) to select/weight neighbors, then GNN.
class FraudGNN_RL(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(FraudGNN_RL, self).__init__()
        # Policy Network: Selects "useful" neighbors based on features
        self.policy_net = nn.Sequential(
            nn.Linear(in_channels * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Main GNN (Using GCNConv to support edge weights via normalization)
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, sensitive_attr=None): # sensitive_attr ignored
        # RL Step (Approximated by soft weighting)
        src, dst = edge_index
        
        # Features of edge endpoints
        x_src = x[src]
        x_dst = x[dst]
        cat_feat = torch.cat([x_src, x_dst], dim=1)
        
        # Edge weights from policy
        edge_weight = self.policy_net(cat_feat).squeeze()
        
        # GCN Conv with learned edge weights
        x1 = F.relu(self.conv1(x, edge_index, edge_weight))
        x1 = F.dropout(x1, p=0.2, training=self.training)
        
        # Reuse weights or learn new ones? 
        # For simplicity reuse the same policy or just standard GCN in second layer?
        # Let's use standard GCN for second layer or pass same weights.
        # Usually policy is per-layer or global. Let's pass same weights.
        x2 = F.relu(self.conv2(x1, edge_index, edge_weight))
        x2 = F.dropout(x2, p=0.2, training=self.training)
        
        out = self.lin(x2)
        return torch.sigmoid(out)

# Ref: GNN-CL (Contrastive Learning)
class GNN_CL(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNN_CL, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=2, concat=True)
        self.conv2 = GATConv(hidden_channels * 2, hidden_channels, heads=1, concat=False)
        self.lin = nn.Linear(hidden_channels, out_channels)
        
        self.embedding = None # Store for CL loss
        
    def forward(self, x, edge_index, sensitive_attr=None):
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.elu(self.conv2(x, edge_index))
        self.embedding = x 
        
        out = self.lin(x)
        return torch.sigmoid(out)
        
    def cl_loss(self, edge_index):
        if self.embedding is None: return 0.0
        
        src, dst = edge_index
        pos_score = (self.embedding[src] * self.embedding[dst]).sum(dim=1)
        
        # Negative samples
        neg_dst = dst[torch.randperm(len(dst))]
        neg_score = (self.embedding[src] * self.embedding[neg_dst]).sum(dim=1)
        
        loss = -torch.log(torch.sigmoid(pos_score) + 1e-15) - torch.log(1 - torch.sigmoid(neg_score) + 1e-15)
        return loss.mean()

# Ref: Attn-Ensemble
class Attn_Ensemble(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(Attn_Ensemble, self).__init__()
        
        # Branch 1: GAT
        self.gat1 = GATConv(in_channels, hidden_channels, heads=4, concat=True)
        self.gat2 = GATConv(hidden_channels * 4, hidden_channels, heads=1, concat=False)
        
        # Branch 2: MLP
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels * 2),
            nn.ReLU(),
            nn.Linear(hidden_channels * 2, hidden_channels)
        )
        
        # Attention Gate
        self.gate = nn.Linear(hidden_channels * 2, 2)
        
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, sensitive_attr=None):
        # GAT Branch
        x_gat = F.elu(self.gat1(x, edge_index))
        x_gat = F.elu(self.gat2(x_gat, edge_index))
        
        # MLP Branch
        x_mlp = self.mlp(x)
        
        # Attention Gating
        combined = torch.cat([x_gat, x_mlp], dim=1)
        attn_scores = F.softmax(self.gate(combined), dim=1)
        
        x_final = attn_scores[:, 0:1] * x_gat + attn_scores[:, 1:2] * x_mlp
        
        out = self.classifier(x_final)
        return torch.sigmoid(out)
