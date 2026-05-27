"""
Diagnose why GRU-GNN predicts constant values
"""
import torch
import pandas as pd
from models import GRUGNNModel
from utils.dataloader import create_edge_index_and_features, TrafficVolumeSequenceGraphDataLoader
from config import *

print("="*70)
print("DIAGNOSING GRU-GNN MODEL")
print("="*70)

# Load data
print("\n1. Checking Training Data...")
train_df = pd.read_pickle(train_data_file)
print(f"   Shape: {train_df.shape}")
print(f"   Mean: {train_df.mean().mean():.2f}")
print(f"   Std: {train_df.std().mean():.2f}")
print(f"   Min: {train_df.min().min():.2f}")
print(f"   Max: {train_df.max().max():.2f}")

# Check test data
test_df = pd.read_pickle(test_data_file)
print(f"\n2. Checking Test Data...")
print(f"   Shape: {test_df.shape}")
print(f"   Mean: {test_df.mean().mean():.2f}")
print(f"   Std: {test_df.std().mean():.2f}")

# Create dataloader
print(f"\n3. Creating Dataloader with sequence_length=12...")
edge_index, edge_weight = create_edge_index_and_features(
    stations_included_file, 
    stations_data_file, 
    graph_file
)

sequence_length = config_gru_gnn.get("sequence_length", 12)
test_loader = TrafficVolumeSequenceGraphDataLoader(
    test_data_file,
    edge_index,
    edge_weight,
    sequence_length=sequence_length,
    batch_size=16,
    num_workers=0
)

print(f"   Dataset length: {len(test_loader.dataset)}")

# Get a batch
batch = next(iter(test_loader))
print(f"\n4. Batch Structure...")
print(f"   batch.x shape: {batch.x.shape} (should be [num_nodes, {sequence_length}])")
print(f"   batch.y shape: {batch.y.shape}")
print(f"   batch.u shape: {batch.u.shape} (global features)")
print(f"   batch.edge_index shape: {batch.edge_index.shape}")
print(f"   batch.edge_attr shape: {batch.edge_attr.shape}")

print(f"\n   Sample input sequence (first node):")
print(f"   {batch.x[0].numpy()}")
print(f"   Target: {batch.y[0].item():.2f}")

# Initialize model
print(f"\n5. Initializing GRU-GNN Model...")
model = GRUGNNModel(sequence_length=sequence_length, use_edge_model=True)
print(f"   Model created successfully")

# Try forward pass
print(f"\n6. Testing Forward Pass (untrained model)...")
model.eval()
with torch.no_grad():
    try:
        output = model(batch)
        print(f"   ✓ Forward pass successful")
        print(f"   Output shape: {output.shape}")
        print(f"   Output mean: {output.mean().item():.2f}")
        print(f"   Output std: {output.std().item():.2f}")
        print(f"   Output min: {output.min().item():.2f}")
        print(f"   Output max: {output.max().item():.2f}")
        print(f"   Sample predictions: {output[:5].numpy()}")
        print(f"   Sample targets:     {batch.y[:5].numpy()}")
    except Exception as e:
        print(f"   ✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()

# Load trained checkpoint
print(f"\n7. Loading Trained Checkpoint...")
checkpoint_file = config_gru_gnn["checkpoint_file"]
try:
    model.load_state_dict(torch.load(checkpoint_file, map_location=device))
    print(f"   ✓ Checkpoint loaded: {checkpoint_file}")
    
    # Test with trained model
    print(f"\n8. Testing with Trained Model...")
    model.eval()
    with torch.no_grad():
        output_trained = model(batch)
        print(f"   Output mean: {output_trained.mean().item():.2f}")
        print(f"   Output std: {output_trained.std().item():.2f}")
        print(f"   Output min: {output_trained.min().item():.2f}")
        print(f"   Output max: {output_trained.max().item():.2f}")
        
        if output_trained.std().item() < 10:
            print(f"   ⚠️  WARNING: Very low std - model predicting near-constant!")
        
        print(f"\n   Sample predictions: {output_trained[:10].numpy()}")
        print(f"   Sample targets:     {batch.y[:10].numpy()}")
        
        # Check if predictions are clustered
        unique_vals = torch.unique(output_trained.round())
        print(f"\n   Number of unique rounded predictions: {len(unique_vals)}")
        if len(unique_vals) < 5:
            print(f"   ⚠️  Model is predicting very few distinct values!")
    
    # Test on more batches
    print(f"\n9. Testing on Multiple Batches...")
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= 5:  # Test on 5 batches
                break
            preds = model(batch)
            all_preds.append(preds)
            all_targets.append(batch.y)
    
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    
    print(f"   Predictions across {len(all_preds)} samples:")
    print(f"     Mean: {all_preds.mean().item():.2f}")
    print(f"     Std: {all_preds.std().item():.2f}")
    print(f"     Range: [{all_preds.min().item():.2f}, {all_preds.max().item():.2f}]")
    
    print(f"   Targets across {len(all_targets)} samples:")
    print(f"     Mean: {all_targets.mean().item():.2f}")
    print(f"     Std: {all_targets.std().item():.2f}")
    print(f"     Range: [{all_targets.min().item():.2f}, {all_targets.max().item():.2f}]")
    
    # Calculate error
    mae = (all_preds - all_targets).abs().mean().item()
    print(f"\n   MAE: {mae:.2f}")
    
    # Check if model learned anything
    baseline_mae = (all_targets - all_targets.mean()).abs().mean().item()
    print(f"   Baseline MAE (predicting mean): {baseline_mae:.2f}")
    
    if mae >= baseline_mae * 0.95:
        print(f"   ⚠️  Model performs similar to or worse than baseline!")
    
except FileNotFoundError:
    print(f"   ✗ Checkpoint not found: {checkpoint_file}")
    print(f"   Train the model first!")
except Exception as e:
    print(f"   ✗ Error loading checkpoint: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("DIAGNOSIS COMPLETE")
print("="*70)
