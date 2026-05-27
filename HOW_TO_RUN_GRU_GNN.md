# Complete Pipeline to Run GRU-GNN Model

## Step-by-Step Instructions

### Step 1: Ensure You Have the Correct Data Files

Make sure these files exist in your directory:

- `traffic_flow_data_filtered.csv` (your processed CSV file)
- `data/graph.pkl` (graph structure)
- `data/traffic_stations.csv` (station GPS coordinates)

### Step 2: Preprocess Data for GRU-GNN

Run the preprocessing script to create pickle files:

```bash
python preprocess_csv_data.py
```

**Expected Output:**

- `data/time_series_train.pkl`
- `data/time_series_val.pkl`
- `data/time_series_test.pkl`
- `data/stations_included.csv`

**Verify:** Check that training data shape is correct:

```bash
python -c "import pandas as pd; df = pd.read_pickle('data/time_series_train.pkl'); print(f'Train shape: {df.shape}'); print(f'Mean: {df.mean().mean():.1f}')"
```

### Step 3: Delete Old Checkpoint (Important!)

The old checkpoint was trained with wrong settings. Delete it:

```bash
del checkpoints\gru_gnn.pth
```

Or on Linux/Mac:

```bash
rm checkpoints/gru_gnn.pth
```

### Step 4: Train the GRU-GNN Model

```bash
python train_and_evaluate.py
```

When prompted:

1. **Select model:** Enter `5` (for GRU_GNN)
2. **Checkpoint option:** Enter `3` (delete and retrain from scratch)

**Training Configuration:**

- Sequence length: 12 hours (captures temporal patterns)
- GRU layers: 2 layers with 64 hidden units
- GNN layers: 2 layers (simplified for better learning)
- Batch size: 16
- Learning rate: 0.001
- Epochs: 150
- Early stopping: Disabled

**What to Watch:**

- Training loss should decrease steadily
- Validation loss should follow training loss
- If loss plateaus early, the model may need more capacity or better initialization

### Step 5: Evaluate Results

After training completes, check the evaluation results:

1. **Comprehensive metrics** in `evaluation_results_GRU_GNN/`

   - `metrics_summary.csv` - All metrics per station and overall
   - `predicted_vs_actual.png` - Scatter plot of predictions
   - `percentage_mae_by_station.png` - Error distribution
   - `percentage_rmse_by_station.png` - RMSE distribution

2. **Time series plots** in `figs/gru_gnn_predictions/`
   - Plots showing predicted vs actual over time

### Expected Performance

**Good model should have:**

- R² > 0.5 (explains >50% of variance)
- MAE < 300 (average error less than 300 vehicles/hour)
- Predictions with good variance (not constant)
- Training loss < 200

**If model still predicts constant values:**

The issue is likely one of these:

1. **Data leakage in preprocessing:**

   - Check `preprocess_csv_data.py` line 30
   - Should be: `train_df = time_series_data.iloc[0:train_size]`
   - NOT: `train_df = time_series_data.iloc[:]`

2. **Learning rate too high/low:**

   - Try lr = 0.0001 (lower) if loss oscillates
   - Try lr = 0.005 (higher) if loss decreases too slowly

3. **Gradient issues:**

   - Add gradient clipping in trainer
   - Check if weights are updating (print model parameters before/after training)

4. **Data scaling:**
   - Current: No normalization (raw flow values ~2000-2800)
   - Try: MinMax scaling in config.py: `normalize_data = "minmax"`

### Quick Test After Training

```bash
python diagnose_gru_gnn.py
```

This will show:

- Data statistics
- Model predictions vs targets
- Whether model learned anything (MAE vs baseline)

---

## Troubleshooting Common Issues

### Issue: "Model performs worse than baseline"

**Solution:**

1. Check if data preprocessing is correct (train/val/test split)
2. Increase epochs to 200-300
3. Try simpler model first (reduce GNN layers)
4. Add data normalization

### Issue: "Predictions still constant"

**Solution:**

1. Verify preprocessing split is correct
2. Check model is actually training (losses decreasing)
3. Try different initialization: `torch.manual_seed(42)` before training
4. Reduce model complexity
5. Check for NaN in data

### Issue: "Training loss not decreasing"

**Solution:**

1. Lower learning rate to 0.0001
2. Check data has variation (not all same values)
3. Try different loss function (MSE instead of L1Loss)
4. Add gradient clipping

### Issue: "Out of memory"

**Solution:**

1. Reduce batch_size to 8 or 4
2. Reduce sequence_length to 6
3. Reduce GRU hidden_dim to 32

---

## Understanding the Model

### GRU-GNN Architecture:

```
Input: Past 12 hours of traffic flow for each station
  ↓
[GRU] - Processes temporal sequences
  - 2 layers, 64 hidden units
  - Captures: trends, patterns over 12 hours
  ↓
[GNN MetaLayer] - Processes spatial relationships
  - Edge Model: Updates edge features
  - Node Model: Aggregates neighbor information
  - Global Model: Creates graph-level representation
  ↓
Output: Predicted flow for next hour
```

### Data Flow:

1. **Input:** (8 stations × 12 hours) = Sequence of past hours
2. **GRU Output:** (8 stations × 64 features) = Temporal encoding
3. **GNN Output:** (8 stations × 1) = Final predictions

---

## Alternative: Try Simpler GCN Version First

If GRU-GNN is too complex, try GCN first:

```bash
python train_and_evaluate.py
# Select 4 (GCN)
```

This uses simpler spatial aggregation without temporal sequences. If GCN works well, then GRU-GNN should work better with sequences.

---

## Files Modified for GRU-GNN Support

1. `config.py` - Added sequence_length parameter
2. `utils/dataloader.py` - Added TrafficVolumeSequenceGraphDataLoader
3. `models/gru_gnn.py` - GRU + GNN MetaLayer architecture
4. `train_and_evaluate.py` - Uses sequence dataloader for GRU_GNN
5. `utils/__init__.py` - Exports sequence dataloader

---

## Quick Command Reference

```bash
# 1. Preprocess data
python preprocess_csv_data.py

# 2. Delete old checkpoint
del checkpoints\gru_gnn.pth

# 3. Train model
python train_and_evaluate.py
# Select: 5 (GRU_GNN)
# Option: 3 (delete and retrain)

# 4. Diagnose issues
python diagnose_gru_gnn.py

# 5. View results
cd evaluation_results_GRU_GNN
# Check metrics_summary.csv and PNG plots
```

---

## Expected Training Time

- CPU: ~20-30 minutes for 150 epochs
- GPU: ~3-5 minutes for 150 epochs

Monitor the progress bar to see:

- Training loss decreasing
- Validation loss staying reasonable
- Early stopping counter (if enabled)
