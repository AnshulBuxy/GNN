import torch
import torch.nn as nn
from torch_geometric.nn import Sequential, MetaLayer
from torch_geometric.utils import scatter

class EdgeModel(nn.Module):
    def __init__(self, node_features, edge_features, graph_features, hidden_dim):
        super().__init__()
        self.MLP = nn.Sequential(
                    nn.Linear(node_features[0] * 2 + edge_features[0] + graph_features[0], hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, edge_features[1])
                )

    def forward(self, src, dest, edge_attr, u, batch):
        out = torch.cat([src, dest, edge_attr, u[batch]], dim=1)
        out = self.MLP(out)
        return out 

class NodeModel(nn.Module):
    def __init__(self, node_features, edge_features, graph_features, hidden_dim):
        super().__init__()
        self.MLP1 = nn.Sequential(
                    nn.Linear(node_features[0] + edge_features[1] + graph_features[0], hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, node_features[1])
                )

    def forward(self, x, edge_index, edge_attr, u, batch):
        row, col = edge_index
        e_aggr = scatter(edge_attr, col, dim=0, reduce="mean")
        out = torch.cat([x, e_aggr, u[batch]], dim=1) 
        out = self.MLP1(out)
        return out

class GlobalModel(nn.Module):
    def __init__(self, node_features, edge_features, graph_features, hidden_dim):
        super().__init__()
        self.MLP = nn.Sequential(
                    nn.Linear(node_features[1] + edge_features[1] + graph_features[0], hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, graph_features[1])
                )

    def forward(self, x, edge_index, edge_attr, u, batch):
        row, col = edge_index
        x_aggr = scatter(x, batch, dim=0, reduce="mean")
        e_aggr = scatter(edge_attr, batch[col], dim=0, reduce="mean")
        out = torch.cat([u, x_aggr, e_aggr], dim=1)
        out = self.MLP(out)
        return out

class GNNLayer(nn.Module):
    def __init__(self, node_features=(1, 1), edge_features=(1, 1), graph_features=(3, 3), hidden_dims=(32, 32, 32), use_edge_model=True):
        super().__init__()
        if use_edge_model:
            self.edge_model = EdgeModel(node_features, edge_features, graph_features, hidden_dims[0])
        else:
            self.edge_model = None
        self.node_model = NodeModel(node_features, edge_features, graph_features, hidden_dims[1])
        self.global_model = GlobalModel(node_features, edge_features, graph_features, hidden_dims[2])
        self.layers = MetaLayer(self.edge_model, self.node_model, self.global_model)

    def forward(self, x, edge_index, edge_attr, u, batch):
        out = self.layers(x, edge_index, edge_attr, u, batch)
        return out

class GRUGNNModel(nn.Module):
    """
    Graph Recurrent Neural Network combining GRU with GNN (MetaLayer).
    Uses GRU to capture temporal dependencies and GNN for spatial relationships.
    Now properly handles sequences of past hours with improved capacity.
    """
    def __init__(self, node_features=1, hidden_dim=64, gru_layers=2, output_features=1, sequence_length=12, use_edge_model=True):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.gru_layers = gru_layers
        self.sequence_length = sequence_length
        self.use_edge_model = use_edge_model
        
        # GRU for temporal modeling - processes sequence of past hours
        self.gru = nn.GRU(
            input_size=node_features,
            hidden_size=hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            dropout=0.3 if gru_layers > 1 else 0  # Increased dropout for regularization
        )
        
        # Additional dropout layer after GRU
        self.dropout = nn.Dropout(0.2)
        
        # GNN layers for spatial modeling using MetaLayer
        # Increased capacity for better learning
        node_feature_sizes = [hidden_dim, 64, 32, 1]  # More layers with gradual reduction
        edge_feature_sizes = [1, 1, 1, 1]
        graph_feature_sizes = [3, 32, 16, 1]  # Increased capacity
        mlp_hidden_dims = (64, 64, 64)  # Increased from 32
        
        if not use_edge_model:
            edge_feature_sizes = [1] * len(node_feature_sizes)
        
        layers = []
        layer_header = "x, edge_index, edge_attr, u, batch -> x, edge_attr, u"
        
        # Add GNN layers
        for j in range(len(node_feature_sizes) - 1):
            layer = GNNLayer(
                node_feature_sizes[j:j+2], 
                edge_feature_sizes[j:j+2], 
                graph_feature_sizes[j:j+2], 
                mlp_hidden_dims, 
                use_edge_model
            )
            layers.append((layer, layer_header))
        
        self.gnn_layers = Sequential("x, edge_index, edge_attr, u, batch", layers)
    
    def forward(self, x, edge_index, edge_attr, u, batch):
        # x shape: (num_nodes_in_batch, sequence_length)
        # Each node has a sequence of past hours
        
        # Reshape for GRU: (num_nodes, sequence_length, features=1)
        x_gru = x.unsqueeze(-1)  # Add feature dimension: (num_nodes, seq_len, 1)
        
        # Apply GRU - processes temporal sequence for each node
        # gru_out: (num_nodes, seq_len, hidden_dim)
        # hidden: (num_layers, num_nodes, hidden_dim)
        gru_out, hidden = self.gru(x_gru)
        
        # Use the last output (most recent time step with all context)
        x = gru_out[:, -1, :]  # Shape: (num_nodes, hidden_dim)
        
        # Apply dropout for regularization
        x = self.dropout(x)
        
        # Apply GNN layers (Edge + Node + Global models)
        x, edge_attr, u = self.gnn_layers(x, edge_index, edge_attr, u, batch)
        
        return x.view(-1)
