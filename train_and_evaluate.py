import torch
import torch.nn as nn
from os.path import isfile

from config import *
from utils import choose_model, BaselineTrainer, GNNTrainer
from models import BaseLineModel, GNNModel, GCNModel, GRUGNNModel, GraphWaveNet
from utils import TrafficVolumeDataLoader, TrafficVolumeGraphDataLoader, TrafficVolumeSequenceGraphDataLoader, create_edge_index_and_features
from evaluate_comprehensive import evaluate_model_comprehensive

if __name__ == "__main__":  
    config = choose_model(configs)
    name = config["name"]
    lr = config["lr"]
    batch_size = config["batch_size"]
    # Use MSE for better gradient signals, but report both MAE and RMSE
    loss_function = nn.MSELoss()  # Changed from L1Loss


    if name == "GNN": 
        # Graph NN with edge, node and graph models (using pre-defined graph)
        model = GNNModel()
        edge_index, edge_weight = create_edge_index_and_features(stations_included_file, stations_data_file, graph_file)
        train_dataloader = TrafficVolumeGraphDataLoader(train_data_file, edge_index, edge_weight, batch_size, num_workers, shuffle=True)
        val_dataloader = TrafficVolumeGraphDataLoader(val_data_file, edge_index, edge_weight, batch_size, num_workers)
        test_dataloader = TrafficVolumeGraphDataLoader(test_data_file, edge_index, edge_weight, batch_size, num_workers)
        trainer = GNNTrainer(model, train_dataloader, val_dataloader, test_dataloader, config, loss_function, device)
    elif name == "GNN_NE":
        # Graph NN with node and graph model (using pre-defined graph)
        model = GNNModel(use_edge_model=False)
        edge_index, edge_weight = create_edge_index_and_features(stations_included_file, stations_data_file, graph_file, compute_edge_features=False)
        train_dataloader = TrafficVolumeGraphDataLoader(train_data_file, edge_index, edge_weight, batch_size, num_workers, shuffle=True)
        val_dataloader = TrafficVolumeGraphDataLoader(val_data_file, edge_index, edge_weight, batch_size, num_workers)
        test_dataloader = TrafficVolumeGraphDataLoader(test_data_file, edge_index, edge_weight, batch_size, num_workers)
        trainer = GNNTrainer(model, train_dataloader, val_dataloader, test_dataloader, config, loss_function, device)
    elif name == "GNN_KNN":
        # Graph NN using graph generated with kNN
        model = GNNModel()
        edge_index, edge_weight = create_edge_index_and_features(stations_included_file, stations_data_file)
        train_dataloader = TrafficVolumeGraphDataLoader(train_data_file, edge_index, edge_weight, batch_size, num_workers, shuffle=True)
        val_dataloader = TrafficVolumeGraphDataLoader(val_data_file, edge_index, edge_weight, batch_size, num_workers)
        test_dataloader = TrafficVolumeGraphDataLoader(test_data_file, edge_index, edge_weight, batch_size, num_workers)
        trainer = GNNTrainer(model, train_dataloader, val_dataloader, test_dataloader, config, loss_function, device)
    elif name == "GCN":
        # Graph Convolutional Network
        model = GCNModel()
        edge_index, edge_weight = create_edge_index_and_features(stations_included_file, stations_data_file, graph_file)
        train_dataloader = TrafficVolumeGraphDataLoader(train_data_file, edge_index, edge_weight, batch_size, num_workers, shuffle=True)
        val_dataloader = TrafficVolumeGraphDataLoader(val_data_file, edge_index, edge_weight, batch_size, num_workers)
        test_dataloader = TrafficVolumeGraphDataLoader(test_data_file, edge_index, edge_weight, batch_size, num_workers)
        trainer = GNNTrainer(model, train_dataloader, val_dataloader, test_dataloader, config, loss_function, device)
    elif name == "GRU_GNN":
        # Graph Recurrent Neural Network (GRU + GNN) with sequence support
        sequence_length = config.get("sequence_length", 4)
        hidden_dim = config.get("hidden_dim", 64)  # Model capacity
        gru_layers = config.get("gru_layers", 2)  # Number of GRU layers
        model = GRUGNNModel(sequence_length=sequence_length, hidden_dim=hidden_dim, gru_layers=gru_layers)
        edge_index, edge_weight = create_edge_index_and_features(stations_included_file, stations_data_file, graph_file)
        train_dataloader = TrafficVolumeSequenceGraphDataLoader(train_data_file, edge_index, edge_weight, sequence_length, batch_size, num_workers, shuffle=True)
        val_dataloader = TrafficVolumeSequenceGraphDataLoader(val_data_file, edge_index, edge_weight, sequence_length, batch_size, num_workers)
        test_dataloader = TrafficVolumeSequenceGraphDataLoader(test_data_file, edge_index, edge_weight, sequence_length, batch_size, num_workers)
        trainer = GNNTrainer(model, train_dataloader, val_dataloader, test_dataloader, config, loss_function, device)
    elif name == "GraphWaveNet":
        # Graph WaveNet with adaptive adjacency matrix and temporal convolutions
        sequence_length = config.get("sequence_length", 12)
        
        # Count number of nodes from stations file
        import pandas as pd
        stations = pd.read_csv(stations_included_file)
        num_nodes = len(stations)
        
        model = GraphWaveNet(
            num_nodes=num_nodes,
            in_channels=1,
            out_channels=1,
            residual_channels=config.get("residual_channels", 32),
            dilation_channels=config.get("dilation_channels", 32),
            skip_channels=config.get("skip_channels", 64),
            end_channels=config.get("end_channels", 128),
            kernel_size=2,
            num_blocks=config.get("num_blocks", 4),
            num_layers=config.get("num_layers", 2),
            seq_length=sequence_length,
            embedding_dim=config.get("embedding_dim", 10)
        )
        
        edge_index, edge_weight = create_edge_index_and_features(stations_included_file, stations_data_file, graph_file)
        train_dataloader = TrafficVolumeSequenceGraphDataLoader(train_data_file, edge_index, edge_weight, sequence_length, batch_size, num_workers, shuffle=True)
        val_dataloader = TrafficVolumeSequenceGraphDataLoader(val_data_file, edge_index, edge_weight, sequence_length, batch_size, num_workers)
        test_dataloader = TrafficVolumeSequenceGraphDataLoader(test_data_file, edge_index, edge_weight, sequence_length, batch_size, num_workers)
        trainer = GNNTrainer(model, train_dataloader, val_dataloader, test_dataloader, config, loss_function, device)
    elif name == "Baseline":
        # Baseline fully connected NN model
        model = BaseLineModel()
        train_dataloader = TrafficVolumeDataLoader(train_data_file, batch_size, num_workers, shuffle=True)
        val_dataloader = TrafficVolumeDataLoader(val_data_file, batch_size, num_workers)
        test_dataloader = TrafficVolumeDataLoader(test_data_file, batch_size, num_workers)
        trainer = BaselineTrainer(model, train_dataloader, val_dataloader, test_dataloader, config, loss_function, device)
    
    trainer.print_model_size()

    # Check if checkpoint exists
    if isfile(config["checkpoint_file"]):
        print("\n" + "="*60)
        print(f"Checkpoint file found: {config['checkpoint_file']}")
        print("="*60)
        choice = input("\nOptions:\n[1] Use existing checkpoint (evaluate only)\n[2] Train new model and keep existing checkpoint (save new checkpoint)\n[3] Delete existing checkpoint and retrain from scratch\n\nSelect option (1/2/3): ").strip()
        
        if choice == "2":
            # Keep existing checkpoint; save new training checkpoints to a new file
            import os, time
            orig_ckpt = config["checkpoint_file"]
            base_dir = os.path.dirname(orig_ckpt)
            base_name = os.path.splitext(os.path.basename(orig_ckpt))[0]
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            new_ckpt_name = f"{base_name}.new_{timestamp}.pth"
            new_ckpt_path = os.path.join(base_dir, new_ckpt_name) if base_dir else new_ckpt_name
            print(f"✓ Keeping existing checkpoint at {orig_ckpt}")
            print(f"✓ New training checkpoints will be saved to: {new_ckpt_path}\n")
            # Update checkpoint path in config for this training run
            config["checkpoint_file"] = new_ckpt_path
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=config.get("weight_decay", 1e-5)) 
            lr_patience = config.get("lr_patience", 5)
            lr_factor = config.get("lr_factor", 0.5)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=lr_factor, patience=lr_patience)
            trainer.train(optimizer, scheduler)
            trainer.summarize_training()
        elif choice == "3":
            import os
            os.remove(config["checkpoint_file"])
            print(f"✓ Deleted checkpoint file. Starting training from scratch...\n")
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=config.get("weight_decay", 1e-5)) 
            lr_patience = config.get("lr_patience", 5)
            lr_factor = config.get("lr_factor", 0.5)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=lr_factor, patience=lr_patience)
            trainer.train(optimizer, scheduler)
            trainer.summarize_training()
        elif choice == "1":
            print("✓ Using existing checkpoint for evaluation...\n")
        else:
            print("Invalid option. Using existing checkpoint by default...\n")
    else:
        print("\nNo checkpoint found. Starting training from scratch...\n")
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=config.get("weight_decay", 1e-5)) 
        # Enable learning rate scheduler for better convergence
        lr_patience = config.get("lr_patience", 5)
        lr_factor = config.get("lr_factor", 0.5)
        lr_min = config.get("lr_min", 0)  # Minimum learning rate
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=lr_factor, patience=lr_patience, min_lr=lr_min)
        trainer.train(optimizer, scheduler)
        trainer.summarize_training()

    # Evaluate model on test data and compute test loss
    trainer.evaluate()

    # Comprehensive evaluation with all metrics and plots
    print("\n" + "="*70)
    print("Starting comprehensive evaluation...")
    print("="*70)
    results = evaluate_model_comprehensive(trainer, output_dir=f'evaluation_results_{name}')

    # Make some prediction and save plot (original time series plots)
    # Plot 10 days of predictions (10 days * 24 hours = 240 samples)
    from_index = 0 
    length = min(240, len(trainer.test_dataloader.dataset))  # 10 days of hourly data
    trainer.save_prediction_plot(from_index, length)
