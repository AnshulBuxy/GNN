import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool

class GCNModel(nn.Module):
    """
    Graph Convolutional Network for traffic prediction.
    Uses GCN layers to process graph-structured traffic data.
    """
    def __init__(self, node_features=1, hidden_channels=[64, 32, 16], output_features=1):
        super().__init__()
        
        # Build GCN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # Input layer
        self.convs.append(GCNConv(node_features, hidden_channels[0]))
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels[0]))
        
        # Hidden layers
        for i in range(len(hidden_channels) - 1):
            self.convs.append(GCNConv(hidden_channels[i], hidden_channels[i+1]))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels[i+1]))
        
        # Output layer
        self.convs.append(GCNConv(hidden_channels[-1], output_features))
        
        self.dropout = nn.Dropout(0.2)
        self.activation = nn.ReLU()
    
    def forward(self, x, edge_index, edge_attr, u, batch):
        # GCN doesn't use edge_attr and u, but accepts them for consistency
        
        # Apply GCN layers with activation and normalization
        for i, (conv, bn) in enumerate(zip(self.convs[:-1], self.batch_norms)):
            x = conv(x, edge_index)
            x = bn(x)
            x = self.activation(x)
            x = self.dropout(x)
        
        # Final layer without activation
        x = self.convs[-1](x, edge_index)
        
        return x.view(-1)
