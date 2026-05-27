"""
Graph WaveNet Model for Traffic Forecasting
Based on: "Graph WaveNet for Deep Spatial-Temporal Graph Modeling" (IJCAI 2019)

Key features:
- Adaptive adjacency matrix (learns hidden spatial dependencies)
- Temporal convolution with dilated causal convolutions
- Spatial graph convolution on both predefined and learned graphs
- Skip connections for deep architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveAdjacency(nn.Module):
    """
    Learns an adaptive adjacency matrix from node embeddings.
    This captures hidden spatial dependencies not evident from geography.
    Supports dynamic node counts by using a maximum size and slicing.
    """
    def __init__(self, num_nodes, embedding_dim=10):
        super().__init__()
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        # Two sets of learnable node embeddings with Xavier initialization
        self.node_embedding1 = nn.Parameter(torch.randn(num_nodes, embedding_dim) * 0.01)
        self.node_embedding2 = nn.Parameter(torch.randn(num_nodes, embedding_dim) * 0.01)
        
    def forward(self, actual_num_nodes=None):
        """
        Compute adaptive adjacency matrix.
        Args:
            actual_num_nodes: If provided, slice embeddings to this size
        Returns:
            Adjacency matrix of shape [actual_num_nodes, actual_num_nodes]
        """
        # Use subset of embeddings if actual nodes is less than initialized
        if actual_num_nodes is not None and actual_num_nodes < self.num_nodes:
            emb1 = self.node_embedding1[:actual_num_nodes]
            emb2 = self.node_embedding2[:actual_num_nodes]
        else:
            emb1 = self.node_embedding1
            emb2 = self.node_embedding2
        
        # Compute adaptive adjacency: A = softmax(ReLU(E1 @ E2.T))
        adj = F.relu(torch.mm(emb1, emb2.transpose(0, 1)))
        adj = F.softmax(adj, dim=1)  # Row-wise softmax
        return adj


class GraphConvolution(nn.Module):
    """
    Graph convolution operation: H' = A @ H @ W
    where A is adjacency matrix, H is node features, W is weight matrix
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)
        
    def forward(self, x, adj):
        """
        Args:
            x: [batch, num_nodes, in_features]
            adj: [num_nodes, num_nodes] or [batch, num_nodes, num_nodes]
        Returns:
            [batch, num_nodes, out_features]
        """
        # x @ W
        support = torch.matmul(x, self.weight)
        # A @ (x @ W)
        if adj.dim() == 2:
            output = torch.matmul(adj, support)
        else:
            output = torch.bmm(adj, support)
        return output


class TemporalConvLayer(nn.Module):
    """
    Dilated causal convolution for temporal modeling.
    Uses gated activation like WaveNet.
    """
    def __init__(self, in_channels, out_channels, kernel_size=2, dilation=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        
        # Calculate padding for causal convolution (to maintain sequence length)
        self.padding = (kernel_size - 1) * dilation
        
        # Two convolutions for gated activation
        self.filter_conv = nn.Conv2d(in_channels, out_channels, (1, kernel_size), dilation=(1, dilation))
        self.gate_conv = nn.Conv2d(in_channels, out_channels, (1, kernel_size), dilation=(1, dilation))
        
    def forward(self, x):
        """
        Args:
            x: [batch, channels, num_nodes, seq_len]
        Returns:
            [batch, channels, num_nodes, seq_len]
        """
        # Apply causal padding (pad on the left side only)
        x_padded = F.pad(x, (self.padding, 0, 0, 0))
        
        # Gated activation: tanh(filter) * sigmoid(gate)
        filter_out = torch.tanh(self.filter_conv(x_padded))
        gate_out = torch.sigmoid(self.gate_conv(x_padded))
        output = filter_out * gate_out
        return output


class SpatialTemporalBlock(nn.Module):
    """
    One Graph WaveNet block combining temporal and spatial convolutions.
    Structure: TCN -> GCN (predefined) -> GCN (adaptive) -> TCN
    """
    def __init__(self, in_channels, out_channels, num_nodes, kernel_size=2, dilation=1):
        super().__init__()
        
        # Temporal convolutions
        self.temporal1 = TemporalConvLayer(in_channels, out_channels, kernel_size, dilation)
        self.temporal2 = TemporalConvLayer(out_channels, out_channels, kernel_size, dilation)
        
        # Spatial graph convolutions (will use both predefined and adaptive graphs)
        self.graph_conv1 = GraphConvolution(out_channels, out_channels)
        self.graph_conv2 = GraphConvolution(out_channels, out_channels)
        
        # Batch norm and residual connection
        self.batch_norm = nn.BatchNorm2d(out_channels)
        
        # Residual connection (if needed)
        self.residual_conv = nn.Conv2d(in_channels, out_channels, (1, 1)) if in_channels != out_channels else None
        
    def forward(self, x, adj_predefined, adj_adaptive):
        """
        Args:
            x: [batch, in_channels, num_nodes, seq_len]
            adj_predefined: [num_nodes, num_nodes] predefined adjacency
            adj_adaptive: [num_nodes, num_nodes] learned adjacency
        Returns:
            [batch, out_channels, num_nodes, new_seq_len]
        """
        residual = x
        
        # Temporal convolution 1
        x = self.temporal1(x)
        
        # Spatial graph convolutions
        # Reshape: [batch, channels, nodes, seq] -> [batch, seq, nodes, channels]
        batch, channels, nodes, seq_len = x.shape
        x_spatial = x.permute(0, 3, 2, 1).contiguous()  # [batch, seq, nodes, channels]
        x_spatial = x_spatial.view(batch * seq_len, nodes, channels)  # [batch*seq, nodes, channels]
        
        # Apply GCN on predefined graph
        x_pred = self.graph_conv1(x_spatial, adj_predefined)
        
        # Apply GCN on adaptive graph
        x_adap = self.graph_conv2(x_spatial, adj_adaptive)
        
        # Combine outputs
        x_spatial = x_pred + x_adap
        
        # Reshape back
        x_spatial = x_spatial.view(batch, seq_len, nodes, channels)
        x = x_spatial.permute(0, 3, 2, 1).contiguous()  # [batch, channels, nodes, seq]
        
        # Temporal convolution 2
        x = self.temporal2(x)
        
        # Residual connection (pad if needed due to temporal reduction)
        if self.residual_conv:
            residual = self.residual_conv(residual)
        
        # Match temporal dimension if needed
        if residual.shape[-1] > x.shape[-1]:
            residual = residual[..., :x.shape[-1]]
        
        # Apply batch norm after adding residual (more stable)
        x = x + residual
        x = self.batch_norm(x) if hasattr(self, 'batch_norm') else x
        return F.relu(x)


class GraphWaveNet(nn.Module):
    """
    Graph WaveNet for traffic forecasting.
    Combines temporal dilated convolutions with spatial graph convolutions.
    Learns adaptive adjacency matrix for hidden spatial dependencies.
    """
    def __init__(self, num_nodes, in_channels=1, out_channels=1, residual_channels=32, 
                 dilation_channels=32, skip_channels=64, end_channels=128, 
                 kernel_size=2, num_blocks=4, num_layers=2, seq_length=12, 
                 embedding_dim=10):
        """
        Args:
            num_nodes: Number of nodes/stations in the graph
            in_channels: Number of input features per node (default 1 for traffic volume)
            out_channels: Number of output features (default 1 for traffic prediction)
            residual_channels: Channels in residual connections
            dilation_channels: Channels in dilated convolutions
            skip_channels: Channels in skip connections
            end_channels: Channels before output layer
            kernel_size: Kernel size for temporal convolutions
            num_blocks: Number of spatial-temporal blocks
            num_layers: Number of layers per block
            seq_length: Input sequence length
            embedding_dim: Dimension of node embeddings for adaptive graph
        """
        super().__init__()
        
        self.num_nodes = num_nodes
        self.seq_length = seq_length
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        
        # Learnable adaptive adjacency matrix
        self.adaptive_adj = AdaptiveAdjacency(num_nodes, embedding_dim)
        
        # Input projection
        self.start_conv = nn.Conv2d(in_channels, residual_channels, (1, 1))
        
        # Spatial-temporal blocks with increasing dilation
        self.st_blocks = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        
        for block in range(num_blocks):
            for layer in range(num_layers):
                dilation = 2 ** layer
                self.st_blocks.append(
                    SpatialTemporalBlock(
                        residual_channels, dilation_channels, num_nodes, 
                        kernel_size, dilation
                    )
                )
                # Skip connection
                self.skip_convs.append(nn.Conv2d(dilation_channels, skip_channels, (1, 1)))
        
        # Output layers
        self.end_conv1 = nn.Conv2d(skip_channels, end_channels, (1, 1))
        self.end_conv2 = nn.Conv2d(end_channels, out_channels, (1, 1))
        
        # Global feature embedding (for month, weekday, hour)
        self.global_emb = nn.Linear(3, residual_channels)
        
        # Placeholder for predefined adjacency matrix (will be set on first forward pass)
        self.register_buffer('adj_predefined', None)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights for better training stability"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, x, edge_index, edge_weight, u, batch):
        """
        Args:
            x: Node features [batch_size * num_nodes, 1, seq_length]
            edge_index: Edge connectivity [2, num_edges]
            edge_weight: Edge weights [num_edges, 1]
            u: Global features [batch_size, 3] (month, weekday, hour)
            batch: Batch assignment for nodes [batch_size * num_nodes]
        Returns:
            Predictions [batch_size * num_nodes, 1]
        """
        batch_size = u.shape[0]
        
        # Calculate actual number of nodes dynamically from input
        # x has shape [batch_size * num_nodes, 1, seq_length]
        total_nodes = x.shape[0]
        actual_num_nodes = total_nodes // batch_size
        
        # Build predefined adjacency matrix once (only on first call)
        if self.adj_predefined is None or self.adj_predefined.shape[0] != actual_num_nodes:
            adj_pred = torch.zeros(actual_num_nodes, actual_num_nodes, device=x.device)
            # edge_index in first batch contains indices 0 to num_nodes-1
            # We only need edges from the base graph (first batch)
            for i in range(edge_index.shape[1]):
                src, dst = edge_index[0, i].item(), edge_index[1, i].item()
                # Only use edges within first graph (indices < actual_num_nodes)
                if src < actual_num_nodes and dst < actual_num_nodes:
                    weight_val = edge_weight[i, 0].item() if edge_weight is not None else 1.0
                    adj_pred[dst, src] = weight_val
            
            # Row-normalize predefined adjacency
            row_sum = adj_pred.sum(1, keepdim=True)
            row_sum[row_sum == 0] = 1  # Avoid division by zero
            adj_pred = adj_pred / row_sum
            self.adj_predefined = adj_pred
        
        # Reshape x to [batch, channels, nodes, seq_len]
        x = x.view(batch_size, actual_num_nodes, 1, self.seq_length)
        x = x.permute(0, 2, 1, 3)  # [batch, 1, nodes, seq]
        
        # Get adaptive adjacency matrix (use actual node count)
        adj_adaptive = self.adaptive_adj(actual_num_nodes)
        
        # Add global features to initial representation
        global_emb = self.global_emb(u)  # [batch, residual_channels]
        global_emb = global_emb.view(batch_size, -1, 1, 1)  # [batch, channels, 1, 1]
        
        # Start convolution
        x = self.start_conv(x)  # [batch, residual_channels, nodes, seq]
        
        # Add global embedding
        x = x + global_emb
        
        # Pass through spatial-temporal blocks with skip connections
        skip_outputs = []
        for i, (st_block, skip_conv) in enumerate(zip(self.st_blocks, self.skip_convs)):
            x = st_block(x, self.adj_predefined, adj_adaptive)
            skip = skip_conv(x)
            skip_outputs.append(skip)
        
        # Sum all skip connections
        x = sum(skip_outputs)
        x = F.relu(x)
        
        # Output layers
        x = F.relu(self.end_conv1(x))
        x = self.end_conv2(x)  # [batch, 1, nodes, seq]
        
        # Take the last time step and reshape to [batch * nodes, 1]
        x = x[..., -1]  # [batch, 1, nodes]
        x = x.permute(0, 2, 1).contiguous()  # [batch, nodes, 1]
        x = x.view(batch_size * actual_num_nodes, 1)
        
        return x
