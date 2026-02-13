import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from torch.nn import Parameter

class FSERLayer(MessagePassing):
    """
    Fairness-Sensitive Edge Reweighting (FSER) Layer.
    Enhances GAT with a fairness constraint.
    """
    def __init__(self, in_channels, out_channels, heads=1, concat=True, dropout=0.0):
        super(FSERLayer, self).__init__(node_dim=0, aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout = dropout

        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))
        
        # Learnable fairness coefficient beta
        # Initialize to 0.5, will be clipped to [0, 5] during training
        self.beta = Parameter(torch.tensor(0.5))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att)
        nn.init.constant_(self.beta, 0.5)

    def forward(self, x, edge_index, sensitive_attr):
        """
        Args:
            x (Tensor): Node features [N, in_channels]
            edge_index (LongTensor): Graph edges [2, E]
            sensitive_attr (Tensor): Sensitive attribute for each node [N]
        """
        x_lin = self.lin(x).view(-1, self.heads, self.out_channels)
        
        # Propagate messages
        # sensitive_attr needs to be lifted to edge level during message passing
        # We pass it as a kwarg to propagate, which handles lifting to x_i, x_j
        return self.propagate(edge_index, x=x_lin, sensitive_attr=sensitive_attr)

    def message(self, x_i, x_j, index, ptr, size_i, sensitive_attr_i, sensitive_attr_j):
        # x_i, x_j: [E, heads, out_channels]
        # sensitive_attr_i, sensitive_attr_j: [E] (lifted by propagate)
        
        # 1. Standard GAT attention scores
        # Concatenate features of source and target
        # [E, heads, 2 * out_channels]
        cat_feat = torch.cat([x_i, x_j], dim=-1)
        # alpha: [E, heads]
        alpha = (cat_feat * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, 0.2)
        
        # 2. FSER Fairness Risk Score
        # Delta_s: 1 if cross-group edge, 0 otherwise
        # sensitive_attr_i/j are [E]
        # Ensure shape matches [E, 1] for broadcasting with heads if needed
        # Or [E, heads] if we want per-head logic (though attr is same for all heads)
        delta_s = (sensitive_attr_i != sensitive_attr_j).float().unsqueeze(-1) # [E, 1]
        
        # Cosine similarity between embeddings
        # Normalize for cosine similarity
        x_i_norm = F.normalize(x_i, p=2, dim=-1)
        x_j_norm = F.normalize(x_j, p=2, dim=-1)
        # [E, heads]
        cos_sim = (x_i_norm * x_j_norm).sum(dim=-1)
        
        # Phi: Fairness penalty
        # Penalize if cross-group (delta_s=1) AND similar (cos_sim > 0)
        phi = delta_s * torch.clamp(cos_sim, min=0)
        
        # 3. Corrected Attention
        # beta is a scalar parameter
        alpha = alpha - self.beta * phi
        
        # 4. Normalize (Softmax)
        alpha = softmax(alpha, index, ptr, size_i)
        
        self._alpha = alpha # Save for inspection/viz if needed
        
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        # Broadcast alpha to [E, heads, out_channels]
        return x_j * alpha.unsqueeze(-1)

    def update(self, aggr_out):
        # aggr_out: [N, heads, out_channels]
        if self.concat:
            return aggr_out.view(-1, self.heads * self.out_channels)
        else:
            return aggr_out.mean(dim=1)


class FedFairGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, heads=1, dropout=0.2):
        super(FedFairGNN, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.heads = heads
        self.hidden_channels = hidden_channels
        
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.bn_input = nn.BatchNorm1d(hidden_channels)
        
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        curr_dim = hidden_channels
        
        for i in range(num_layers):
            self.layers.append(FSERLayer(curr_dim, hidden_channels // heads, heads=heads, concat=True, dropout=dropout))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
            # Output of FSERLayer with concat=True is heads * (hidden // heads) = hidden
            curr_dim = hidden_channels 
            
        # Classifier Head
        # Skip connection concatenation: Input + L layers
        # Dimensions: hidden (input) + num_layers * hidden
        total_dim = hidden_channels * (num_layers + 1)
        
        self.final_lin = nn.Linear(total_dim, hidden_channels)
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, sensitive_attr):
        # Layer 0: Input Projection
        h0 = self.input_proj(x)
        h0 = self.bn_input(h0)
        h0 = F.elu(h0)
        h0 = F.dropout(h0, p=self.dropout, training=self.training)
        
        hidden_reps = [h0]
        h = h0
        
        # FSER Layers
        for i in range(self.num_layers):
            h_next = self.layers[i](h, edge_index, sensitive_attr)
            h_next = self.bns[i](h_next)
            h_next = F.elu(h_next)
            h_next = F.dropout(h_next, p=self.dropout, training=self.training)
            
            # Residual connection
            if i >= 1:
                h_next = h_next + h 
            
            h = h_next
            hidden_reps.append(h)
            
        # Skip connection concatenation
        h_skip = torch.cat(hidden_reps, dim=1)
        
        # Output Head
        h_out = self.final_lin(h_skip)
        h_out = F.elu(h_out)
        logits = self.classifier(h_out)
        
        # Return probability
        return torch.sigmoid(logits)

    def clamp_beta(self):
        # Enforce beta \in [0, 5]
        for layer in self.layers:
            if hasattr(layer, 'beta'):
                layer.beta.data.clamp_(0.0, 5.0)
