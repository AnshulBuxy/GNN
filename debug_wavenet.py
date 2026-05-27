"""
Debug script to check GraphWaveNet for dimension mismatches and gradient flow.
"""
import torch
import torch.nn as nn
from models.graph_wavenet import GraphWaveNet
from utils import create_edge_index_and_features, TrafficVolumeSequenceGraphDataLoader
from config import *

print("="*70)
print("GraphWaveNet Debugging")
print("="*70)

# Create a small model for testing
num_nodes = 10
batch_size = 4
sequence_length = 12

model = GraphWaveNet(
    num_nodes=num_nodes,
    in_channels=1,
    out_channels=1,
    residual_channels=16,
    dilation_channels=16,
    skip_channels=32,
    end_channels=64,
    kernel_size=2,
    num_blocks=3,
    num_layers=2,
    seq_length=sequence_length,
    embedding_dim=8
)

print(f"\nModel created with:")
print(f"  - num_nodes: {num_nodes}")
print(f"  - sequence_length: {sequence_length}")
print(f"  - num_blocks: 3, num_layers: 2")
print(f"  - Total blocks: {len(model.st_blocks)}")

# Create dummy input
x = torch.randn(batch_size * num_nodes, 1, sequence_length)
edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))
edge_weight = torch.randn(num_nodes * 2, 1)
u = torch.randn(batch_size, 3)
batch_tensor = torch.arange(batch_size).repeat_interleave(num_nodes)

print(f"\nInput shapes:")
print(f"  - x: {x.shape}")
print(f"  - edge_index: {edge_index.shape}")
print(f"  - edge_weight: {edge_weight.shape}")
print(f"  - u: {u.shape}")
print(f"  - batch: {batch_tensor.shape}")

# Forward pass with shape tracking
print(f"\n{'='*70}")
print("Forward Pass Shape Tracking:")
print(f"{'='*70}")

try:
    # Enable gradient computation
    x.requires_grad = True
    
    with torch.no_grad():
        output = model(x, edge_index, edge_weight, u, batch_tensor)
    
    print(f"\n[OK] Forward pass successful!")
    print(f"  Output shape: {output.shape}")
    print(f"  Expected: {(batch_size * num_nodes, 1)}")
    
    # Check if dimensions match
    if output.shape == (batch_size * num_nodes, 1):
        print(f"  [OK] Output shape is correct!")
    else:
        print(f"  [FAIL] Output shape mismatch!")
        
except Exception as e:
    print(f"\n[FAIL] Forward pass failed!")
    print(f"  Error: {e}")
    import traceback
    traceback.print_exc()

# Test backward pass
print(f"\n{'='*70}")
print("Backward Pass (Gradient Flow):")
print(f"{'='*70}")

try:
    x = torch.randn(batch_size * num_nodes, 1, sequence_length, requires_grad=True)
    output = model(x, edge_index, edge_weight, u, batch_tensor)
    
    # Create dummy target and loss
    target = torch.randn(batch_size * num_nodes, 1)
    loss = nn.MSELoss()(output, target)
    
    print(f"  Loss value: {loss.item():.4f}")
    
    # Backward pass
    loss.backward()
    
    print(f"  [OK] Backward pass successful!")
    
    # Check for NaN or zero gradients
    zero_grad_params = 0
    nan_grad_params = 0
    total_params = 0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            total_params += 1
            if torch.isnan(param.grad).any():
                nan_grad_params += 1
                print(f"    [FAIL] NaN gradient in: {name}")
            elif (param.grad == 0).all():
                zero_grad_params += 1
                print(f"    [WARN] Zero gradient in: {name}")
    
    print(f"\n  Gradient statistics:")
    print(f"    Total parameters: {total_params}")
    print(f"    Parameters with zero gradients: {zero_grad_params}")
    print(f"    Parameters with NaN gradients: {nan_grad_params}")
    
    if nan_grad_params == 0 and zero_grad_params < total_params * 0.5:
        print(f"  [OK] Gradients are flowing properly!")
    else:
        print(f"  [FAIL] Gradient flow issues detected!")
        
except Exception as e:
    print(f"\n[FAIL] Backward pass failed!")
    print(f"  Error: {e}")
    import traceback
    traceback.print_exc()

# Testing with Real Data:
print(f"\n{'='*70}")
print("Testing with Real Data:")
print(f"{'='*70}")

try:
    edge_index, edge_weight = create_edge_index_and_features(
        stations_included_file, 
        stations_data_file, 
        graph_file
    )
    
    # Get actual number of nodes
    import pandas as pd
    stations = pd.read_csv(stations_included_file)
    real_num_nodes = len(stations)
    print(f"  Actual number of nodes in data: {real_num_nodes}")
    
    # Create model with correct node count
    real_model = GraphWaveNet(
        num_nodes=real_num_nodes,
        in_channels=1,
        out_channels=1,
        residual_channels=32,
        dilation_channels=32,
        skip_channels=128,
        end_channels=128,
        kernel_size=2,
        num_blocks=4,
        num_layers=2,
        seq_length=sequence_length,
        embedding_dim=10
    )
    
    test_dataloader = TrafficVolumeSequenceGraphDataLoader(
        test_data_file, 
        edge_index, 
        edge_weight, 
        sequence_length=12, 
        batch_size=16, 
        num_workers=0
    )
    
    # Get one batch
    data = next(iter(test_dataloader))
    print(f"\n  Real data batch shapes:")
    print(f"    x: {data.x.shape}")
    print(f"    y: {data.y.shape}")
    print(f"    u: {data.u.shape}")
    print(f"    edge_index: {data.edge_index.shape}")
    
    # Forward pass with correctly sized model
    output = real_model(data.x, data.edge_index, data.edge_attr, data.u, data.batch)
    print(f"    output: {output.shape}")
    
    # Check shape match with target
    if output.shape[0] == data.y.shape[0]:
        print(f"  [OK] Output matches target shape!")
    else:
        print(f"  [FAIL] Shape mismatch! Output: {output.shape[0]}, Target: {data.y.shape[0]}")
    
except Exception as e:
    print(f"\n[FAIL] Real data test failed!")
    print(f"  Error: {e}")
    import traceback
    traceback.print_exc()

print(f"\n{'='*70}")
print("Debugging Complete")
print(f"{'='*70}")
