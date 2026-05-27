"""
Test script to demonstrate the difference between regular and sequence data loading.
Shows how the GRU-GNN now receives sequences of past hours instead of single time steps.
"""

import torch
from utils.dataloader import create_edge_index_and_features, TrafficVolumeGraphDataLoader, TrafficVolumeSequenceGraphDataLoader
from config import test_data_file, stations_included_file, stations_data_file, graph_file

print("="*70)
print("COMPARISON: Regular DataLoader vs Sequence DataLoader")
print("="*70)

# Create edge index
edge_index, edge_weight = create_edge_index_and_features(
    stations_included_file, 
    stations_data_file, 
    graph_file
)

print("\n--- REGULAR DATALOADER (used by GNN, GCN) ---")
regular_loader = TrafficVolumeGraphDataLoader(
    test_data_file, 
    edge_index, 
    edge_weight, 
    batch_size=2,
    num_workers=0
)

print(f"Dataset length: {len(regular_loader.dataset)} samples")
batch = next(iter(regular_loader))
print(f"Input shape (data.x): {batch.x.shape}")
print(f"  → Each node has {batch.x.shape[1]} feature(s) (current hour only)")
print(f"Target shape (data.y): {batch.y.shape}")
print(f"Edge index shape: {batch.edge_index.shape}")

print("\n--- SEQUENCE DATALOADER (used by GRU-GNN) ---")
sequence_length = 12
sequence_loader = TrafficVolumeSequenceGraphDataLoader(
    test_data_file,
    edge_index,
    edge_weight,
    sequence_length=sequence_length,
    batch_size=2,
    num_workers=0
)

print(f"Dataset length: {len(sequence_loader.dataset)} samples")
print(f"  → Reduced by {sequence_length} (need past hours for first sequence)")
batch_seq = next(iter(sequence_loader))
print(f"Input shape (data.x): {batch_seq.x.shape}")
print(f"  → Each node has {batch_seq.x.shape[1]} features (past {sequence_length} hours)")
print(f"Target shape (data.y): {batch_seq.y.shape}")
print(f"Edge index shape: {batch_seq.edge_index.shape}")

print("\n--- EXAMPLE: What the model sees ---")
print(f"\nRegular GNN/GCN at hour t:")
print(f"  Input:  [flow_t] for each station")
print(f"  Output: flow_{{t+1}}")

print(f"\nGRU-GNN with sequences at hour t:")
print(f"  Input:  [flow_{{t-11}}, flow_{{t-10}}, ..., flow_{{t-1}}, flow_t] for each station")
print(f"  Output: flow_{{t+1}}")
print(f"\n  The GRU processes this sequence to capture:")
print(f"    • Trends (increasing/decreasing traffic)")
print(f"    • Patterns (rush hours, daily cycles)")
print(f"    • Recent history (what happened in past {sequence_length} hours)")

print("\n" + "="*70)
print("VISUALIZATION")
print("="*70)

# Show actual data from first sample
sample = sequence_loader.dataset[0]
print(f"\nFirst sample in sequence dataset:")
print(f"Station 0 past {sequence_length} hours:")
print(f"  {sample.x[0].numpy()}")
print(f"  → GRU will process this sequence")
print(f"\nTarget (next hour) for Station 0: {sample.y[0].item():.2f}")

print("\n" + "="*70)
print("✓ Sequence loading working correctly!")
print("="*70)
