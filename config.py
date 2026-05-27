from os.path import join
from torch_geometric.seed import seed_everything
import torch

# Seed random generators (python, torch, numpy)
seed_everything(0)

# Paths
data_path = "data"
figs_path = "figs"
docs_path = "docs"
checkpoints_path = "checkpoints"

# Filenames
data_file_zip = join(data_path, "traffic_data.zip")
data_file_pkl = join(data_path, "traffic_data.pkl")
stations_data_file = join(data_path, "traffic_stations.csv")
summary_table_file = join(docs_path, "data_summary_table.md")
time_series_file = join(data_path, "time_series_data.pkl")
train_data_file = join(data_path, "time_series_train.pkl")
val_data_file = join(data_path, "time_series_val.pkl")
test_data_file = join(data_path, "time_series_test.pkl")
graph_file = join(data_path, "graph.pkl")
stations_included_file = join(data_path, "stations_included.csv")

# Pre-processing
min_number_of_observations = 1500   # Drop stations having too few observations
val_fraction = 0.15                 # Fraction of data to use for validation data
test_fraction = 0.15                # Fraction of data to use for test data
normalize_data = "normal"           # "minmax" : scale to [0,1], "normal" : use z-scores, None : no normalization

# Training 
num_workers = 0                     # Number of workers to use with dataloader (set to 0 for Colab)
# Automatically select device: use CUDA if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Global Features Toggle
use_global_features = True          # Enable/disable temporal features (month, weekday, hour)
print(f"Global features (temporal): {'ENABLED' if use_global_features else 'DISABLED'}")

# Baseline model config
config_baseline = {}
config_baseline["name"] = "Baseline"
config_baseline["batch_size"] = 16  # Increased to avoid BatchNorm errors 
config_baseline["lr"] = 0.001
config_baseline["epochs"] =50
config_baseline["val_per_epoch"] = 4
config_baseline["checkpoint_file"] = join(checkpoints_path, "baseline.pth") 
config_baseline["prediction_plot_dir"] = join(figs_path, "baseline_predictions")
config_baseline["loss_plot_file"] = join(figs_path, "baseline_loss_plot.png")
config_baseline["earlystop_limit"] = 50  # Disabled early stopping 
1

# GNN model config
config_gnn = {}
config_gnn["name"] = "GNN"
config_gnn["batch_size"] = 16  # Increased to avoid BatchNorm errors 
config_gnn["lr"] = 0.001  # Lower learning rate
config_gnn["epochs"] = 100
config_gnn["val_per_epoch"] = 4
config_gnn["gradient_clip"] = 1.0
config_gnn["checkpoint_file"] = join(checkpoints_path, "gnn.pth") 
config_gnn["prediction_plot_dir"] = join(figs_path, "gnn_predictions")
config_gnn["loss_plot_file"] = join(figs_path, "gnn_loss_plot.png")
config_gnn["earlystop_limit"] = 200  # Disabled early stopping

# GNN model with no edge model
config_gnn_ne = {}
config_gnn_ne["name"] = "GNN_NE"
config_gnn_ne["batch_size"] = 16  # Increased to avoid BatchNorm errors 
config_gnn_ne["lr"] = 0.001
config_gnn_ne["epochs"] = 100
config_gnn_ne["val_per_epoch"] = 4
config_gnn_ne["checkpoint_file"] = join(checkpoints_path, "gnn_ne.pth") 
config_gnn_ne["prediction_plot_dir"] = join(figs_path, "gnn_ne_predictions")
config_gnn_ne["loss_plot_file"] = join(figs_path, "gnn_ne_loss_plot.png")
config_gnn_ne["earlystop_limit"] = None  # Disabled early stopping 

# GNN model with kNN generated graph
config_gnn_knn = {}
config_gnn_knn["name"] = "GNN_KNN"
config_gnn_knn["batch_size"] = 16  # Increased to avoid BatchNorm errors 
config_gnn_knn["lr"] = 0.001
config_gnn_knn["epochs"] = 100
config_gnn_knn["val_per_epoch"] = 4
config_gnn_knn["checkpoint_file"] = join(checkpoints_path, "gnn_knn.pth") 
config_gnn_knn["prediction_plot_dir"] = join(figs_path, "gnn_knn_predictions")
config_gnn_knn["loss_plot_file"] = join(figs_path, "gnn_knn_loss_plot.png")
config_gnn_knn["earlystop_limit"] = None  # Disabled early stopping 

# GCN model (Graph Convolutional Network)
config_gcn = {}
config_gcn["name"] = "GCN"
config_gcn["batch_size"] = 16
config_gcn["lr"] = 0.001
config_gcn["epochs"] = 100
config_gcn["val_per_epoch"] = 4
config_gcn["checkpoint_file"] = join(checkpoints_path, "gcn.pth") 
config_gcn["prediction_plot_dir"] = join(figs_path, "gcn_predictions")
config_gcn["loss_plot_file"] = join(figs_path, "gcn_loss_plot.png")
config_gcn["earlystop_limit"] = None

# GRU-GNN model (Graph Recurrent Neural Network)
config_gru_gnn = {}
config_gru_gnn["name"] = "GRU_GNN"
config_gru_gnn["batch_size"] = 16
config_gru_gnn["lr"] = 0.001  # Increased from 0.0005 for better initial learning
config_gru_gnn["weight_decay"] = 1e-5  # L2 regularization
config_gru_gnn["lr_patience"] = 20  # Increased patience to prevent premature LR decay
config_gru_gnn["lr_factor"] = 0.7  # Less aggressive decay (was 0.5)
config_gru_gnn["lr_min"] = 1e-6  # Minimum learning rate to prevent collapse to 0
config_gru_gnn["epochs"] = 50  # More epochs since LR doesn't collapse now
config_gru_gnn["val_per_epoch"] = 4
config_gru_gnn["checkpoint_file"] = join(checkpoints_path, "gru_gnn.pth") 
config_gru_gnn["prediction_plot_dir"] = join(figs_path, "gru_gnn_predictions")
config_gru_gnn["loss_plot_file"] = join(figs_path, "gru_gnn_loss_plot.png")
config_gru_gnn["earlystop_limit"] = 80  # Increased patience for early stopping
config_gru_gnn["sequence_length"] = 3  # Use shorter sequences for small test CSVs
config_gru_gnn["gradient_clip"] = 2.0  # Increased from 1.0 for more stable training
config_gru_gnn["hidden_dim"] = 128  # Increased model capacity (was 64)
config_gru_gnn["gru_layers"] = 3  # Increased from 2 for better temporal modeling

# Graph WaveNet model (Adaptive graph with temporal convolutions)
config_graph_wavenet = {}
config_graph_wavenet["name"] = "GraphWaveNet"
config_graph_wavenet["batch_size"] = 32  # Increased for more stable gradients
config_graph_wavenet["lr"] = 0.003  # Increased initial LR
config_graph_wavenet["weight_decay"] = 1e-5  # Reduced regularization
config_graph_wavenet["lr_patience"] = 10  # Reduced patience
config_graph_wavenet["lr_factor"] = 0.5  # More aggressive LR decay
config_graph_wavenet["epochs"] = 200
config_graph_wavenet["val_per_epoch"] = 4
config_graph_wavenet["checkpoint_file"] = join(checkpoints_path, "graph_wavenet.pth") 
config_graph_wavenet["prediction_plot_dir"] = join(figs_path, "graph_wavenet_predictions")
config_graph_wavenet["loss_plot_file"] = join(figs_path, "graph_wavenet_loss_plot.png")
config_graph_wavenet["earlystop_limit"] = 30  # Reduced - model converges faster
config_graph_wavenet["sequence_length"] = 3  # Use shorter sequences for small test CSVs
config_graph_wavenet["gradient_clip"] = 5.0  # Increased for more aggressive updates
config_graph_wavenet["residual_channels"] = 64  # Increased capacity
config_graph_wavenet["dilation_channels"] = 64  # Increased capacity
config_graph_wavenet["skip_channels"] = 128     # Reduced (was too large)
config_graph_wavenet["end_channels"] = 128      # Reduced
config_graph_wavenet["num_blocks"] = 2          # Reduced to 2 - simpler model
config_graph_wavenet["num_layers"] = 2
config_graph_wavenet["embedding_dim"] = 8       # Reduced - fewer nodes need less embedding

# List of all models
configs = [config_baseline, config_gnn, config_gnn_ne, config_gnn_knn, config_gcn, config_gru_gnn, config_graph_wavenet]
