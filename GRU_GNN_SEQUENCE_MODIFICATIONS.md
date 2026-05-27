# GRU-GNN Sequence Modifications

## Overview

Modified the GRU-GNN model to use **temporal sequences** of past hours instead of just the current hour. This allows the GRU to fully utilize its recurrent capabilities to capture time-series patterns, trends, and temporal dependencies.

---

## What Changed

### 1. **Configuration** (`config.py`)

Added `sequence_length` parameter:

```python
config_gru_gnn["sequence_length"] = 12  # Uses past 12 hours for prediction
```

### 2. **Data Loading** (`utils/dataloader.py`)

Created new dataset and dataloader classes:

#### `TrafficVolumeSequenceGraphDataSet`

- Returns sequences of past hours instead of single time step
- Input shape: `(num_nodes, sequence_length)`
- Each node has a sequence of past 12 hours of traffic data
- Target: Next hour (hour 13) traffic flow

#### `TrafficVolumeSequenceGraphDataLoader`

- Wrapper for sequence dataset
- Automatically adjusts dataset length (requires `sequence_length` previous rows)

### 3. **GRU-GNN Model** (`models/gru_gnn.py`)

#### Before (Single Time Step):

```python
# Input: (num_nodes, 1) - only current hour
x_gru = x.unsqueeze(1)  # Add dummy sequence dimension
gru_out, _ = self.gru(x_gru)
x = gru_out.squeeze(1)  # GRU doesn't really learn anything
```

#### After (Sequence Processing):

```python
# Input: (num_nodes, sequence_length) - past 12 hours
x_gru = x.unsqueeze(-1)  # Shape: (num_nodes, 12, 1)
gru_out, hidden = self.gru(x_gru)  # GRU processes temporal patterns
x = gru_out[:, -1, :]  # Use last output with full temporal context
```

### 4. **Training Script** (`train_and_evaluate.py`)

Updated GRU_GNN case to use sequence dataloader:

```python
sequence_length = config.get("sequence_length", 12)
model = GRUGNNModel(sequence_length=sequence_length)
train_dataloader = TrafficVolumeSequenceGraphDataLoader(...)
```

### 5. **Exports** (`utils/__init__.py`)

Added export for new dataloader:

```python
from .dataloader import TrafficVolumeSequenceGraphDataLoader
```

---

## How It Works Now

### Data Flow with Sequences:

```
Hour indices: [0, 1, 2, ..., 11] → Predict hour 12
              [1, 2, 3, ..., 12] → Predict hour 13
              [2, 3, 4, ..., 13] → Predict hour 14
              ...

For Station A at time t:
  Input:  [flow_{t-11}, flow_{t-10}, ..., flow_{t-1}, flow_t]  (12 values)
  Target: flow_{t+1}  (1 value)
```

### Model Processing:

1. **GRU Stage (Temporal):**

   - Input: Past 12 hours of flow data per station
   - GRU processes sequence, maintains hidden state
   - Output: 64-dimensional embedding with temporal context
   - **Captures:** Trends, cycles, time-of-day patterns

2. **GCN Stage (Spatial):**
   - Input: GRU's temporal embeddings for all stations
   - Aggregates information from neighbor stations
   - Output: Final prediction for next hour
   - **Captures:** Spatial correlations between connected stations

### Example Prediction:

To predict traffic at Station A for **hour 100**:

- GRU processes: hours [88, 89, ..., 98, 99] from Station A
- Learns: "Traffic has been increasing steadily"
- GCN aggregates: Info from neighbors (Stations B, C, D) at hour 99
- Combines: Temporal trend + Spatial context → Prediction for hour 100

---

## Benefits

1. **Better Temporal Modeling:**

   - GRU now sees past patterns (was seeing only current hour)
   - Can detect trends, rushes, seasonal patterns
   - Memory mechanism utilizes previous hours

2. **More Informative Features:**

   - 12 data points per station vs. 1 before
   - Richer input for predictions

3. **Realistic Traffic Prediction:**
   - Real traffic depends on recent history
   - Captures morning/evening rush patterns
   - Detects gradual increases/decreases

---

## Key Parameters

| Parameter         | Value    | Description                  |
| ----------------- | -------- | ---------------------------- |
| `sequence_length` | 12       | Number of past hours to use  |
| `hidden_dim`      | 64       | GRU hidden state dimension   |
| `gru_layers`      | 2        | Number of stacked GRU layers |
| `gcn_hidden`      | [32, 16] | GCN layer dimensions         |

---

## Training

Simply run the training script and select GRU_GNN:

```bash
python train_and_evaluate.py
# Select: [5] GRU_GNN
```

The model will automatically:

- Use past 12 hours of data
- Process sequences through GRU
- Aggregate spatial information through GCN
- Predict the next hour

---

## Notes

- **Dataset Size:** Reduced by `sequence_length` samples (need 12 previous hours for first sample)
- **Memory Usage:** Slightly higher due to sequence storage
- **Training Time:** May be longer due to sequence processing
- **Compatibility:** Other models (Baseline, GNN, GCN) unchanged and still work

---

## Comparison with Other Models

| Model             | Temporal                 | Spatial          | Input Data        |
| ----------------- | ------------------------ | ---------------- | ----------------- |
| Baseline          | ❌                       | ❌               | Current hour only |
| GNN               | ❌                       | ✅ (MetaLayer)   | Current hour only |
| GCN               | ❌                       | ✅ (GCNConv)     | Current hour only |
| **GRU_GNN (NEW)** | **✅ (GRU on 12 hours)** | **✅ (GCNConv)** | **Past 12 hours** |

The GRU-GNN is now the **only model** that uses temporal sequences!
