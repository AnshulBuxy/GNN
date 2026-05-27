#!/usr/bin/env python3
"""Test-only evaluation runner that uses existing checkpoints.

This script is based on train_and_evaluate.py but skips training and only:
- loads the selected model
- loads the existing checkpoint
- evaluates on the test split
- saves evaluation artifacts with a "test_" prefix

Example:
    python test_only_evaluate.py --model GRU_GNN

If --model is omitted, an interactive menu is shown.
"""

import argparse
from copy import deepcopy
from os.path import isfile
from pathlib import Path
import pickle
import re
import time

import torch
import torch.nn as nn
import pandas as pd

from config import *
from utils import choose_model, BaselineTrainer, GNNTrainer
from models import BaseLineModel, GNNModel, GCNModel, GRUGNNModel, GraphWaveNet
from utils import (
    TrafficVolumeDataLoader,
    TrafficVolumeGraphDataLoader,
    TrafficVolumeSequenceGraphDataLoader,
    create_edge_index_and_features,
)
from evaluate_comprehensive import evaluate_model_comprehensive, denormalize_data


def parse_args():
    parser = argparse.ArgumentParser(description="Test-only evaluation using checkpoints")
    parser.add_argument(
        "--model",
        default=None,
        help="Model name (e.g., Baseline, GNN, GNN_NE, GNN_KNN, GCN, GRU_GNN, GraphWaveNet). If omitted, prompts interactively.",
    )
    parser.add_argument(
        "--test-csv",
        default=None,
        help="CSV file with a timestamp column and 8 station columns for external testing.",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=None,
        help="Override sequence length for sequence models (GRU_GNN/GraphWaveNet).",
    )
    return parser.parse_args()


def add_test_prefix(path_str: str, prefix: str = "test_") -> str:
    path = Path(path_str)
    return str(path.with_name(prefix + path.name))


def apply_test_prefix(config: dict) -> dict:
    cfg = deepcopy(config)
    if "loss_plot_file" in cfg:
        cfg["loss_plot_file"] = add_test_prefix(cfg["loss_plot_file"])
    if "prediction_plot_dir" in cfg:
        cfg["prediction_plot_dir"] = add_test_prefix(cfg["prediction_plot_dir"])
    return cfg


def load_station_ids():
    stations_df = pd.read_csv(stations_included_file)
    ids = stations_df.iloc[:, -1].tolist()
    return [int(x) for x in ids]


def station_id_from_col(col_name: str) -> int | None:
    s = str(col_name).strip().lower()
    if s.isdigit():
        return int(s)
    match = re.search(r"(\d+)", s)
    if match:
        return int(match.group(1))
    return None


def reorder_station_columns(df: pd.DataFrame, station_ids: list[int]) -> pd.DataFrame:
    col_map = {}
    for col in df.columns:
        sid = station_id_from_col(col)
        if sid is not None and sid not in col_map:
            col_map[sid] = col
    missing = [sid for sid in station_ids if sid not in col_map]
    if missing:
        expected = ", ".join(str(s) for s in station_ids)
        raise SystemExit(
            f"Missing station columns for IDs: {missing}. Expected station columns for IDs: {expected}."
        )
    ordered_cols = [col_map[sid] for sid in station_ids]
    df = df[ordered_cols]
    df.columns = station_ids
    return df


def apply_normalization(df: pd.DataFrame) -> pd.DataFrame:
    params = load_normalization_params()
    if params is None:
        if normalize_data:
            print("⚠ Normalization parameters not found; using raw values.")
        return df
    mean = params["mean"]
    std = params["std"]
    return (df - mean) / std


def load_normalization_params():
    norm_file = Path("data") / "normalization_params.pkl"
    if not norm_file.exists():
        return None
    with open(norm_file, "rb") as f:
        params = pickle.load(f)
    return params


def prepare_test_pkl(csv_path: Path) -> Path:
    print("[1/6] Loading test CSV...")
    t0 = time.time()
    df = pd.read_csv(csv_path)
    if df.shape[1] < 2:
        raise SystemExit("CSV must have a timestamp column and station columns.")
    timestamp_col = df.columns[0]
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.sort_values(timestamp_col)
    df = df.set_index(timestamp_col)
    station_ids = load_station_ids()
    df = reorder_station_columns(df, station_ids)
    df = apply_normalization(df)
    print(f"  -> Loaded and normalized {len(df)} rows in {time.time() - t0:.2f}s")

    out_path = csv_path.with_name(f"test_{csv_path.stem}_preprocessed.pkl")
    df.to_pickle(out_path)
    print(f"Saved preprocessed test data to: {out_path}")
    return out_path


def get_prediction_timestamps(dataset, n_rows: int):
    if hasattr(dataset, "valid_indices") and hasattr(dataset, "sequence_length"):
        idx = [i + dataset.sequence_length for i in dataset.valid_indices]
        ts = dataset.timestamps[idx]
        return ts[:n_rows]
    if hasattr(dataset, "timestamps"):
        ts = dataset.timestamps[1:]
        return ts[:n_rows]
    return None


def save_predictions_csv(trainer, out_path: str):
    preds = trainer.test_results["predictions"]
    truth = trainer.test_results["ground_truth"]
    if preds.size == 0 or truth.size == 0:
        print("No predictions to save.")
        return

    dataset = trainer.test_dataloader.dataset
    station_names = [str(s) for s in getattr(dataset, "column_names", range(preds.shape[1]))]
    timestamps = get_prediction_timestamps(dataset, len(preds))

    data = {}
    if timestamps is not None:
        data["timestamp"] = pd.to_datetime(timestamps)

    # Always save normalized values
    for i, name in enumerate(station_names):
        data[f"pred_{name}"] = preds[:, i]
    for i, name in enumerate(station_names):
        data[f"truth_{name}"] = truth[:, i]

    # Also save denormalized values (veh/hr) if params exist
    params = load_normalization_params()
    if params is not None:
        preds_denorm = denormalize_data(preds, params)
        truth_denorm = denormalize_data(truth, params)
        for i, name in enumerate(station_names):
            data[f"pred_{name}_vehph"] = preds_denorm[:, i]
        for i, name in enumerate(station_names):
            data[f"truth_{name}_vehph"] = truth_denorm[:, i]
        print("Saved denormalized columns with *_vehph suffix.")

    pd.DataFrame(data).to_csv(out_path, index=False)
    print(f"Saved predictions to: {out_path}")


def choose_config(model_name: str | None) -> dict:
    if model_name:
        for cfg in configs:
            if cfg.get("name", "").lower() == model_name.lower():
                return apply_test_prefix(cfg)
        raise SystemExit(f"Unknown model name: {model_name}")
    cfg = choose_model(configs)
    return apply_test_prefix(cfg)


def build_trainer(
    name: str,
    config: dict,
    loss_function,
    test_data_override: Path | None = None,
    sequence_length_override: int | None = None,
):
    batch_size = config["batch_size"]
    train_file = test_data_override or train_data_file
    val_file = test_data_override or val_data_file
    test_file = test_data_override or test_data_file

    if name == "GNN":
        model = GNNModel()
        edge_index, edge_weight = create_edge_index_and_features(
            stations_included_file, stations_data_file, graph_file
        )
        train_dataloader = TrafficVolumeGraphDataLoader(
            train_file, edge_index, edge_weight, batch_size, num_workers, shuffle=False
        )
        val_dataloader = TrafficVolumeGraphDataLoader(
            val_file, edge_index, edge_weight, batch_size, num_workers
        )
        test_dataloader = TrafficVolumeGraphDataLoader(
            test_file, edge_index, edge_weight, batch_size, num_workers
        )
        return GNNTrainer(model, train_dataloader, val_dataloader, test_dataloader, config, loss_function, device)

    if name == "GNN_NE":
        model = GNNModel(use_edge_model=False)
        edge_index, edge_weight = create_edge_index_and_features(
            stations_included_file, stations_data_file, graph_file, compute_edge_features=False
        )
        train_dataloader = TrafficVolumeGraphDataLoader(
            train_file, edge_index, edge_weight, batch_size, num_workers, shuffle=False
        )
        val_dataloader = TrafficVolumeGraphDataLoader(
            val_file, edge_index, edge_weight, batch_size, num_workers
        )
        test_dataloader = TrafficVolumeGraphDataLoader(
            test_file, edge_index, edge_weight, batch_size, num_workers
        )
        return GNNTrainer(model, train_dataloader, val_dataloader, test_dataloader, config, loss_function, device)

    if name == "GNN_KNN":
        model = GNNModel()
        edge_index, edge_weight = create_edge_index_and_features(stations_included_file, stations_data_file)
        train_dataloader = TrafficVolumeGraphDataLoader(
            train_file, edge_index, edge_weight, batch_size, num_workers, shuffle=False
        )
        val_dataloader = TrafficVolumeGraphDataLoader(
            val_file, edge_index, edge_weight, batch_size, num_workers
        )
        test_dataloader = TrafficVolumeGraphDataLoader(
            test_file, edge_index, edge_weight, batch_size, num_workers
        )
        return GNNTrainer(model, train_dataloader, val_dataloader, test_dataloader, config, loss_function, device)

    if name == "GCN":
        model = GCNModel()
        edge_index, edge_weight = create_edge_index_and_features(
            stations_included_file, stations_data_file, graph_file
        )
        train_dataloader = TrafficVolumeGraphDataLoader(
            train_file, edge_index, edge_weight, batch_size, num_workers, shuffle=False
        )
        val_dataloader = TrafficVolumeGraphDataLoader(
            val_file, edge_index, edge_weight, batch_size, num_workers
        )
        test_dataloader = TrafficVolumeGraphDataLoader(
            test_file, edge_index, edge_weight, batch_size, num_workers
        )
        return GNNTrainer(model, train_dataloader, val_dataloader, test_dataloader, config, loss_function, device)

    if name == "GRU_GNN":
        sequence_length = sequence_length_override or config.get("sequence_length", 4)
        hidden_dim = config.get("hidden_dim", 64)
        gru_layers = config.get("gru_layers", 2)
        model = GRUGNNModel(sequence_length=sequence_length, hidden_dim=hidden_dim, gru_layers=gru_layers)
        edge_index, edge_weight = create_edge_index_and_features(
            stations_included_file, stations_data_file, graph_file
        )
        train_dataloader = TrafficVolumeSequenceGraphDataLoader(
            train_file, edge_index, edge_weight, sequence_length, batch_size, num_workers, shuffle=False
        )
        val_dataloader = TrafficVolumeSequenceGraphDataLoader(
            val_file, edge_index, edge_weight, sequence_length, batch_size, num_workers
        )
        test_dataloader = TrafficVolumeSequenceGraphDataLoader(
            test_file, edge_index, edge_weight, sequence_length, batch_size, num_workers
        )
        return GNNTrainer(model, train_dataloader, val_dataloader, test_dataloader, config, loss_function, device)

    if name == "GraphWaveNet":
        sequence_length = sequence_length_override or config.get("sequence_length", 12)
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
            embedding_dim=config.get("embedding_dim", 10),
        )
        edge_index, edge_weight = create_edge_index_and_features(
            stations_included_file, stations_data_file, graph_file
        )
        train_dataloader = TrafficVolumeSequenceGraphDataLoader(
            train_file, edge_index, edge_weight, sequence_length, batch_size, num_workers, shuffle=False
        )
        val_dataloader = TrafficVolumeSequenceGraphDataLoader(
            val_file, edge_index, edge_weight, sequence_length, batch_size, num_workers
        )
        test_dataloader = TrafficVolumeSequenceGraphDataLoader(
            test_file, edge_index, edge_weight, sequence_length, batch_size, num_workers
        )
        return GNNTrainer(model, train_dataloader, val_dataloader, test_dataloader, config, loss_function, device)

    if name == "Baseline":
        model = BaseLineModel()
        train_dataloader = TrafficVolumeDataLoader(train_file, batch_size, num_workers, shuffle=False)
        val_dataloader = TrafficVolumeDataLoader(val_file, batch_size, num_workers)
        test_dataloader = TrafficVolumeDataLoader(test_file, batch_size, num_workers)
        return BaselineTrainer(model, train_dataloader, val_dataloader, test_dataloader, config, loss_function, device)

    raise SystemExit(f"Unsupported model: {name}")


def main():
    print("[0/6] Entering main()", flush=True)
    args = parse_args()
    print(f"[0/6] Parsed args: model={args.model}, test_csv={args.test_csv}", flush=True)
    config = choose_config(args.model)
    name = config["name"]
    print("Hii", flush=True)
    test_data_override = None
    if args.test_csv:
        csv_path = Path(args.test_csv)
        if not csv_path.exists():
            raise SystemExit(f"Test CSV not found: {csv_path}")
        test_data_override = prepare_test_pkl(csv_path)

    loss_function = nn.MSELoss()
    print("[2/6] Building model and dataloaders...")
    t0 = time.time()
    trainer = build_trainer(
        name,
        config,
        loss_function,
        test_data_override=test_data_override,
        sequence_length_override=args.sequence_length,
    )
    print(f"  -> Built in {time.time() - t0:.2f}s")
    trainer.print_model_size()

    # Dataset size checks for tiny CSVs
    test_len = len(trainer.test_dataloader.dataset)
    print(f"[3/6] Test dataset size: {test_len}")
    if test_len == 0:
        raise SystemExit(
            "Test dataset has 0 samples. For GRU_GNN/GraphWaveNet, you need at least sequence_length + 1 rows."
        )

    if not isfile(config["checkpoint_file"]):
        raise SystemExit(f"Checkpoint not found: {config['checkpoint_file']}")

    print("[4/6] Loading checkpoint and evaluating test set...")
    t0 = time.time()
    trainer.evaluate()
    print(f"  -> Evaluation completed in {time.time() - t0:.2f}s")

    print("\n" + "=" * 70)
    print("[5/6] Starting comprehensive evaluation...")
    print("=" * 70)
    output_dir = f"test_evaluation_results_{name}"
    t0 = time.time()
    evaluate_model_comprehensive(trainer, output_dir=output_dir)
    print(f"  -> Comprehensive evaluation completed in {time.time() - t0:.2f}s")

    print("[6/6] Saving predictions CSV and plots...")
    t0 = time.time()
    test_preds_path = f"test_predictions_{name}.csv"
    save_predictions_csv(trainer, test_preds_path)

    from_index = 0
    length = min(240, len(trainer.test_dataloader.dataset))
    trainer.save_prediction_plot(from_index, length)
    print(f"  -> Saved outputs in {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
