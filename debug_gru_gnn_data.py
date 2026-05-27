"""
Debug script to check if GRU-GNN is receiving data in correct format
"""

import torch
from utils.dataloader import create_edge_index_and_features, TrafficVolumeSequenceGraphDataLoader
from models import GRUGNNModel
from config import test_data_file, stations_included_file, stations_data_file, graph_file

print("="*70)
print("DEBUGGING GRU-GNN DATA FORMAT")
print("="*70)

# Create edge index
edge_index, edge_weight = create_edge_index_and_features(
    stations_included_file, 
    stations_data_file, 
    graph_file
)

# Create sequence dataloader
sequence_length = 12
sequence_loader = TrafficVolumeSequenceGraphDataLoader(
    test_data_file,
    edge_index,
    edge_weight,
    sequence_length=sequence_length,
    batch_size=16,
    num_workers=0
)

print(f"\n--- DATALOADER INFO ---")
print(f"Dataset length: {len(sequence_loader.dataset)}")
print(f"Sequence length: {sequence_length}")
print(f"Batch size: 16")

# Get a batch
batch = next(iter(sequence_loader))

print(f"\n--- BATCH STRUCTURE ---")
print(f"batch.x shape: {batch.x.shape}")
print(f"  Expected: (num_nodes_in_batch, sequence_length)")
print(f"  Got: ({batch.x.shape[0]}, {batch.x.shape[1]})")
print(f"  ✓ Correct!" if batch.x.shape[1] == sequence_length else f"  ✗ WRONG! Expected {sequence_length} features")

print(f"\nbatch.y shape: {batch.y.shape}")
print(f"  Expected: (num_nodes_in_batch,)")
print(f"  Got: ({batch.y.shape[0]},)")

print(f"\nbatch.edge_index shape: {batch.edge_index.shape}")
print(f"batch.batch shape: {batch.batch.shape}")

print(f"\n--- SAMPLE DATA ---")
print(f"First node's sequence (past {sequence_length} hours):")
print(f"  {batch.x[0].numpy()}")
print(f"First node's target (next hour): {batch.y[0].item():.2f}")

print(f"\n--- MODEL FORWARD PASS TEST ---")
model = GRUGNNModel(sequence_length=sequence_length)
model.eval()

try:
    with torch.no_grad():
        output = model(batch)
    
    print(f"✓ Model forward pass successful!")
    print(f"  Input shape: {batch.x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Expected output shape: ({batch.x.shape[0]},)")
    
    if output.shape[0] == batch.x.shape[0]:
        print(f"  ✓ Output shape is CORRECT!")
    else:
        print(f"  ✗ Output shape is WRONG!")
    
    print(f"\n  Sample predictions:")
    print(f"    First 5 predictions: {output[:5].numpy()}")
    print(f"    First 5 targets:     {batch.y[:5].numpy()}")
    
    # Check if predictions are varied or all same
    pred_std = output.std().item()
    pred_mean = output.mean().item()
    print(f"\n  Prediction statistics:")
    print(f"    Mean: {pred_mean:.2f}")
    print(f"    Std:  {pred_std:.2f}")
    print(f"    Min:  {output.min().item():.2f}")
    print(f"    Max:  {output.max().item():.2f}")
    
    if pred_std < 10:
        print(f"  ⚠️  WARNING: Very low std deviation - model may be predicting near-constant values!")
    
    # Check target statistics
    target_mean = batch.y.mean().item()
    target_std = batch.y.std().item()
    print(f"\n  Target statistics:")
    print(f"    Mean: {target_mean:.2f}")
    print(f"    Std:  {target_std:.2f}")
    print(f"    Min:  {batch.y.min().item():.2f}")
    print(f"    Max:  {batch.y.max().item():.2f}")
    
except Exception as e:
    print(f"✗ Model forward pass FAILED!")
    print(f"  Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("CHECKING MODEL ARCHITECTURE")
print("="*70)

print(f"\nGRU-GNN Model:")
print(f"  Sequence length: {model.sequence_length}")
print(f"  Hidden dim: {model.hidden_dim}")
print(f"  GRU layers: {model.gru_layers}")
print(f"  GRU input size: {model.gru.input_size}")
print(f"  GRU hidden size: {model.gru.hidden_size}")

print(f"\n  GCN layers:")
for i, layer in enumerate(model.gcn_layers):
    print(f"    Layer {i}: {layer}")

print("\n" + "="*70)

# Test with actual data processing step by step
print("STEP-BY-STEP DATA PROCESSING")
print("="*70)

print(f"\n1. Input data (batch.x):")
print(f"   Shape: {batch.x.shape}")
print(f"   Sample: {batch.x[0]}")

print(f"\n2. Add feature dimension (unsqueeze):")
x_gru = batch.x.unsqueeze(-1)
print(f"   Shape: {x_gru.shape}")
print(f"   Expected: (num_nodes, sequence_length, 1)")

print(f"\n3. Pass through GRU:")
gru_out, hidden = model.gru(x_gru)
print(f"   GRU output shape: {gru_out.shape}")
print(f"   Hidden shape: {hidden.shape}")
print(f"   Expected output: (num_nodes, sequence_length, hidden_dim)")

print(f"\n4. Extract last time step:")
x_temporal = gru_out[:, -1, :]
print(f"   Shape: {x_temporal.shape}")
print(f"   Expected: (num_nodes, hidden_dim={model.hidden_dim})")

print(f"\n5. Pass through GCN layers:")
x = x_temporal
for i, (gcn, bn) in enumerate(zip(model.gcn_layers[:-1], model.batch_norms)):
    x = gcn(x, batch.edge_index)
    x = bn(x)
    x = model.activation(x)
    x = model.dropout(x)
    print(f"   After GCN layer {i}: {x.shape}")

print(f"\n6. Final GCN layer:")
x = model.gcn_layers[-1](x, batch.edge_index)
print(f"   Shape: {x.shape}")

print(f"\n7. Final output (view(-1)):")
output_final = x.view(-1)
print(f"   Shape: {output_final.shape}")

print("\n" + "="*70)
print("✅ DEBUGGING COMPLETE")
print("="*70)
