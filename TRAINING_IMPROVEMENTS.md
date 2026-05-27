# Improvements to Fix Training Issues

## Problems Identified

1. **Training loss oscillating heavily** (308 → 383 → 315)
2. **Validation loss plateaued** around 270-300
3. **High test error** (MAE: 298, RMSE: 459)
4. **Model not converging smoothly**

## Changes Made

### 1. **Enable Data Normalization** ✅

```python
normalize_data = "minmax"  # Scale to [0,1]
```

**Why:** Raw traffic values (2000-2800) make learning difficult. Normalized [0,1] helps model converge faster.

### 2. **Lower Learning Rate** ✅

```python
config_gru_gnn["lr"] = 0.0005  # Was 0.001
config_gnn["lr"] = 0.0005      # Was 0.001
```

**Why:** High learning rate causes oscillation. Lower LR = smoother convergence.

### 3. **Add Gradient Clipping** ✅

```python
config_gru_gnn["gradient_clip"] = 1.0
config_gnn["gradient_clip"] = 1.0
```

**Why:** Prevents exploding gradients that cause training instability.

### 4. **Enable Adaptive Learning Rate Scheduler** ✅

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,    # Reduce LR by 50%
    patience=10,   # Wait 10 epochs without improvement
    verbose=True
)
```

**Why:** Automatically reduces LR when validation loss plateaus.

### 5. **Enable Early Stopping** ✅

```python
config_gru_gnn["earlystop_limit"] = 50  # Was None
```

**Why:** Stops training when model stops improving, prevents overfitting.

### 6. **Increase Epochs** ✅

```python
config_gru_gnn["epochs"] = 200  # Was 150
```

**Why:** With lower LR and scheduler, model needs more epochs to converge.

---

## Step-by-Step: What to Do Now

### Step 1: Reprocess Data with Normalization

```bash
python preprocess_csv_data.py
```

This will normalize all data to [0, 1] range.

### Step 2: Delete Old Checkpoints

```bash
del checkpoints\*.pth
```

Old checkpoints trained on unnormalized data won't work.

### Step 3: Train with New Settings

```bash
python train_and_evaluate.py
# Select model (e.g., 5 for GRU_GNN)
# Option 3: Delete and retrain
```

### Step 4: Monitor Training

**Look for:**

- ✅ Training loss **decreasing smoothly** (not oscillating)
- ✅ Validation loss **following training loss**
- ✅ Learning rate **reducing automatically** when loss plateaus
- ✅ Gradient clipping **preventing explosions**

**Example of good training:**

```
Epoch 10/200 | Loss (Train): 0.1234 | Loss (Val): 0.1456 | LR: 0.000500
Epoch 20/200 | Loss (Train): 0.0987 | Loss (Val): 0.1123 | LR: 0.000500
Epoch 30/200 | Loss (Train): 0.0876 | Loss (Val): 0.1034 | LR: 0.000250  ← LR reduced
```

---

## Expected Results After Changes

### Before (Your Current Results):

- MAE: 298
- RMSE: 459
- Training: Oscillating, not converging
- R²: Likely negative

### After (Expected):

- MAE: < 150 (50% improvement)
- RMSE: < 250 (45% improvement)
- Training: Smooth decrease
- R²: > 0.6 (explains 60%+ variance)

---

## If Still Not Working

### Problem: Loss still oscillating

**Solution:**

- Lower LR further to 0.0001
- Increase gradient_clip to 5.0
- Try different optimizer: `torch.optim.AdamW`

### Problem: Loss decreasing too slowly

**Solution:**

- Increase LR to 0.001
- Reduce batch_size to 8 (more updates per epoch)
- Try warmup: start with low LR, gradually increase

### Problem: Validation loss higher than training

**Solution:**

- Add dropout to model (reduce overfitting)
- Increase training data (reduce test_fraction)
- Use data augmentation

### Problem: Model still predicts constants

**Solution:**

- Check normalization worked: `python -c "import pandas as pd; df = pd.read_pickle('data/time_series_train.pkl'); print(df.min().min(), df.max().max())"`
  Should show: `0.0 1.0`
- Verify preprocessing split is correct (not using all data for training)
- Try MSELoss instead of L1Loss: `loss_function = nn.MSELoss()`

---

## Understanding the Changes

### Normalization Impact:

**Before:** Model tries to predict values like 2500.5
**After:** Model predicts 0.85, then denormalized back to 2500.5

This makes:

- Gradients more stable
- Learning faster
- Weights easier to optimize

### Gradient Clipping Impact:

**Before:** Gradient = 1000 → Weight update too large → Model diverges
**After:** Gradient clipped to 1.0 → Stable updates → Smooth learning

### Adaptive LR Impact:

**Before:** LR stays at 0.001 even when loss plateaus
**After:** LR reduces to 0.0005, 0.00025, etc. → Fine-tunes around minimum

---

## Quick Diagnostic Commands

### Check normalized data:

```bash
python -c "import pandas as pd; df = pd.read_pickle('data/time_series_train.pkl'); print(f'Min: {df.min().min()}, Max: {df.max().max()}, Mean: {df.mean().mean()}')"
```

Should show: `Min: 0.0, Max: 1.0, Mean: ~0.5`

### Check model is learning:

```bash
python diagnose_gru_gnn.py
```

### View training progress:

Check `figs/gru_gnn_loss_plot.png` - should show smooth decrease, not oscillation.

---

## Summary of Files Modified

1. `config.py` - Learning rates, normalization, gradient clipping
2. `utils/trainer.py` - Gradient clipping, adaptive scheduler support
3. `train_and_evaluate.py` - ReduceLROnPlateau scheduler enabled

---

## Expected Training Time

- With normalized data: **Faster convergence**
- With lower LR: **More epochs needed** (~200 instead of 100)
- With early stopping: **Auto-stops when converged**

**Total time:** ~15-25 minutes on CPU (vs 30+ minutes before)

---

## Key Takeaway

The main issue was **unnormalized data + high learning rate** causing:

- Large gradients
- Oscillating loss
- Poor convergence

The fixes normalize data and stabilize training for much better results! 🚀
