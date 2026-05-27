import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from pathlib import Path
import seaborn as sns
import pickle
from os.path import join

def denormalize_data(data, normalization_params):
    """
    Denormalize data back to original scale.
    
    Parameters:
    -----------
    data : numpy array
        Normalized data
    normalization_params : dict or None
        Dictionary with 'method', 'mean', and 'std'
    
    Returns:
    --------
    Denormalized data
    """
    if normalization_params is None:
        return data
    
    method = normalization_params['method']
    mean = normalization_params['mean'].values if hasattr(normalization_params['mean'], 'values') else normalization_params['mean']
    std = normalization_params['std'].values if hasattr(normalization_params['std'], 'values') else normalization_params['std']
    
    # Denormalize: original = normalized * std + mean
    return data * std + mean

def calculate_metrics(predictions, ground_truth, station_names):
    """
    Calculate MAE, RMSE, %MAE, %RMSE, and R2 score (total and per station).
    
    Parameters:
    -----------
    predictions : numpy array (n_samples, n_stations)
    ground_truth : numpy array (n_samples, n_stations)
    station_names : list of station IDs
    
    Returns:
    --------
    dict with all metrics
    """
    n_stations = predictions.shape[1]
    
    # Initialize results dictionary
    results = {
        'total': {},
        'per_station': {}
    }
    
    # Calculate total metrics (across all stations and time steps)
    preds_flat = predictions.flatten()
    truth_flat = ground_truth.flatten()
    
    # Remove NaN values
    mask = ~(np.isnan(preds_flat) | np.isnan(truth_flat))
    preds_clean = preds_flat[mask]
    truth_clean = truth_flat[mask]
    
    # Total metrics
    mae_total = mean_absolute_error(truth_clean, preds_clean)
    rmse_total = np.sqrt(mean_squared_error(truth_clean, preds_clean))
    mean_truth = np.mean(truth_clean)
    
    # Percentage errors (based on mean of actual values)
    mae_percent_total = (mae_total / mean_truth) * 100 if mean_truth != 0 else 0
    rmse_percent_total = (rmse_total / mean_truth) * 100 if mean_truth != 0 else 0
    r2_total = r2_score(truth_clean, preds_clean)
    
    results['total'] = {
        'MAE': mae_total,
        'RMSE': rmse_total,
        '%MAE': mae_percent_total,
        '%RMSE': rmse_percent_total,
        'R2': r2_total,
        'Mean_Truth': mean_truth
    }
    
    print("\n" + "="*70)
    print("TOTAL METRICS (All Stations)")
    print("="*70)
    print(f"MAE:        {mae_total:.4f}")
    print(f"RMSE:       {rmse_total:.4f}")
    print(f"%MAE:       {mae_percent_total:.2f}%")
    print(f"%RMSE:      {rmse_percent_total:.2f}%")
    print(f"R² Score:   {r2_total:.4f}")
    print(f"Mean Truth: {mean_truth:.4f}")
    
    # Per-station metrics
    print("\n" + "="*70)
    print("PER-STATION METRICS")
    print("="*70)
    print(f"{'Station':<10} {'MAE':<10} {'RMSE':<10} {'%MAE':<10} {'%RMSE':<10} {'R²':<10}")
    print("-"*70)
    
    for i, station_id in enumerate(station_names):
        preds_station = predictions[:, i]
        truth_station = ground_truth[:, i]
        
        # Remove NaN values for this station
        mask_station = ~(np.isnan(preds_station) | np.isnan(truth_station))
        preds_station_clean = preds_station[mask_station]
        truth_station_clean = truth_station[mask_station]
        
        if len(preds_station_clean) > 0:
            mae = mean_absolute_error(truth_station_clean, preds_station_clean)
            rmse = np.sqrt(mean_squared_error(truth_station_clean, preds_station_clean))
            mean_truth_station = np.mean(truth_station_clean)
            
            mae_percent = (mae / mean_truth_station) * 100 if mean_truth_station != 0 else 0
            rmse_percent = (rmse / mean_truth_station) * 100 if mean_truth_station != 0 else 0
            r2 = r2_score(truth_station_clean, preds_station_clean)
            
            results['per_station'][station_id] = {
                'MAE': mae,
                'RMSE': rmse,
                '%MAE': mae_percent,
                '%RMSE': rmse_percent,
                'R2': r2,
                'Mean_Truth': mean_truth_station
            }
            
            print(f"{station_id:<10} {mae:<10.4f} {rmse:<10.4f} {mae_percent:<10.2f} {rmse_percent:<10.2f} {r2:<10.4f}")
        else:
            print(f"{station_id:<10} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10}")
    
    return results

def plot_predicted_vs_actual(predictions, ground_truth, output_file):
    """
    Create scatter plot of predicted vs actual values.
    """
    print(f"\nGenerating predicted vs actual scatter plot...")
    
    preds_flat = predictions.flatten()
    truth_flat = ground_truth.flatten()
    
    # Remove NaN values
    mask = ~(np.isnan(preds_flat) | np.isnan(truth_flat))
    preds_clean = preds_flat[mask]
    truth_clean = truth_flat[mask]
    
    # Create figure
    plt.figure(figsize=(10, 10))
    
    # Scatter plot with transparency
    plt.scatter(truth_clean, preds_clean, alpha=0.3, s=10, label='Predictions')
    
    # Perfect prediction line (y=x)
    min_val = min(truth_clean.min(), preds_clean.min())
    max_val = max(truth_clean.max(), preds_clean.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    # Calculate R²
    r2 = r2_score(truth_clean, preds_clean)
    
    plt.xlabel('Actual Values', fontsize=14)
    plt.ylabel('Predicted Values', fontsize=14)
    plt.title(f'Predicted vs Actual Values\n(R² = {r2:.4f})', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_file}")

def plot_percentage_errors_by_station(results, output_dir):
    """
    Generate bar plots for %MAE and %RMSE by station.
    """
    print(f"\nGenerating percentage error plots by station...")
    
    stations = list(results['per_station'].keys())
    mae_percents = [results['per_station'][s]['%MAE'] for s in stations]
    rmse_percents = [results['per_station'][s]['%RMSE'] for s in stations]
    
    # Set up the plot style
    sns.set_style("whitegrid")
    
    # Plot %MAE
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(stations, mae_percents, color='steelblue', alpha=0.8, edgecolor='black')
    ax.axhline(y=results['total']['%MAE'], color='red', linestyle='--', 
               linewidth=2, label=f"Total %MAE: {results['total']['%MAE']:.2f}%")
    ax.set_xlabel('Station ID', fontsize=14)
    ax.set_ylabel('%MAE (Percentage)', fontsize=14)
    ax.set_title('Percentage Mean Absolute Error by Station', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    mae_file = Path(output_dir) / 'percentage_mae_by_station.png'
    plt.savefig(mae_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {mae_file}")
    
    # Plot %RMSE
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(stations, rmse_percents, color='coral', alpha=0.8, edgecolor='black')
    ax.axhline(y=results['total']['%RMSE'], color='red', linestyle='--', 
               linewidth=2, label=f"Total %RMSE: {results['total']['%RMSE']:.2f}%")
    ax.set_xlabel('Station ID', fontsize=14)
    ax.set_ylabel('%RMSE (Percentage)', fontsize=14)
    ax.set_title('Percentage Root Mean Squared Error by Station', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    rmse_file = Path(output_dir) / 'percentage_rmse_by_station.png'
    plt.savefig(rmse_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {rmse_file}")

def save_metrics_to_csv(results, output_file):
    """
    Save all metrics to a CSV file.
    """
    print(f"\nSaving metrics to CSV...")
    
    # Create DataFrame for per-station metrics
    data = []
    for station_id, metrics in results['per_station'].items():
        row = {'Station': station_id}
        row.update(metrics)
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Add total row
    total_row = {'Station': 'TOTAL'}
    total_row.update(results['total'])
    df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)
    
    df.to_csv(output_file, index=False, float_format='%.4f')
    print(f"✓ Saved: {output_file}")
    print(f"\nMetrics summary:\n{df.to_string(index=False)}")

def evaluate_model_comprehensive(trainer, output_dir='evaluation_results'):
    """
    Comprehensive evaluation with all requested metrics and plots.
    Handles denormalization if data was normalized during preprocessing.
    """
    print("\n" + "="*70)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("="*70)
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Load normalization parameters if they exist
    normalization_params = None
    norm_file = 'data/normalization_params.pkl'
    try:
        with open(norm_file, 'rb') as f:
            normalization_params = pickle.load(f)
        if normalization_params is not None:
            print(f"✓ Loaded normalization parameters (method: {normalization_params['method']})")
            print(f"  Will denormalize predictions to original scale for evaluation")
    except FileNotFoundError:
        print(f"⚠ No normalization parameters found - assuming unnormalized data")
    
    # Get predictions and ground truth
    predictions = trainer.test_results["predictions"]
    ground_truth = trainer.test_results["ground_truth"]
    
    # Denormalize if needed
    if normalization_params is not None:
        print(f"\nDenormalizing predictions and ground truth...")
        predictions = denormalize_data(predictions, normalization_params)
        ground_truth = denormalize_data(ground_truth, normalization_params)
        print(f"✓ Denormalized to original scale")
        print(f"  Prediction range: [{predictions.min():.1f}, {predictions.max():.1f}]")
        print(f"  Ground truth range: [{ground_truth.min():.1f}, {ground_truth.max():.1f}]")
    
    station_names = trainer.test_dataloader.dataset.column_names.tolist()
    
    # Debug: Check shapes before calculate_metrics
    print(f"\nDEBUG: predictions.shape = {predictions.shape}")
    print(f"DEBUG: ground_truth.shape = {ground_truth.shape}")
    print(f"DEBUG: len(station_names) = {len(station_names)}")
    
    # 1. Calculate all metrics
    results = calculate_metrics(predictions, ground_truth, station_names)
    
    # 2. Save metrics to CSV
    csv_file = Path(output_dir) / 'metrics_summary.csv'
    save_metrics_to_csv(results, csv_file)
    
    # 3. Generate predicted vs actual scatter plot
    scatter_file = Path(output_dir) / 'predicted_vs_actual.png'
    plot_predicted_vs_actual(predictions, ground_truth, scatter_file)
    
    # 4. Generate percentage error plots by station
    plot_percentage_errors_by_station(results, output_dir)
    
    print("\n" + "="*70)
    print(f"✅ EVALUATION COMPLETE!")
    print(f"All results saved to: {output_dir}/")
    print("="*70)
    
    return results


if __name__ == "__main__":
    print("This script should be imported and used with a trained model.")
    print("\nUsage:")
    print("  from evaluate_comprehensive import evaluate_model_comprehensive")
    print("  results = evaluate_model_comprehensive(trainer)")
