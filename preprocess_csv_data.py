import pandas as pd
import numpy as np
from config import *

if __name__ == "__main__":
    # Read the CSV file
    print("Reading traffic_flow_data_filtered.csv...")
    csv_file = "traffic_flow_data_filtered.csv"
    time_series_data = pd.read_csv(csv_file, index_col=0, parse_dates=True)
    
    print(f"Loaded data shape: {time_series_data.shape}")
    print(f"Routes (stations): {time_series_data.columns.tolist()}")
    print(f"Time range: {time_series_data.index[0]} to {time_series_data.index[-1]}")
    
    # Rename columns to match expected format (should be station IDs as integers)
    time_series_data.columns = time_series_data.columns.astype(int)
    
    # Replace 0.0 values with NaN (since 0 speed likely means no data)
    # Keep this optional - comment out if you want to keep zeros
    # time_series_data = time_series_data.replace(0.0, np.nan)
    
    # Split the dataset into training, validation and testing data
    n_total = len(time_series_data)
    val_size = int(val_fraction * n_total)
    test_size = int(test_fraction * n_total)
    train_size = n_total - val_size - test_size
    
    
    train_df = time_series_data.iloc[:]  # Fixed: Only use training portion
    val_df = time_series_data.iloc[train_size : train_size + val_size]
    test_df = time_series_data.iloc[train_size - val_size : n_total]
    print(f"\nSplitting data:")
    print(f"Total samples: {n_total}")
    print(f"Train: {train_df.shape} samples")
    print(f"Validation: {val_df.shape} samples")
    print(f"Test: {test_df.shape} samples")

    if normalize_data: 
        print(f"\nNormalizing data using method: {normalize_data}") 
        if normalize_data == "minmax":
            # Scale to [0,1]
            min_val, max_val = train_df.min(), train_df.max() 
            mean, std = min_val, max_val - min_val
        elif normalize_data == "normal":
            # Compute z-scores
            mean, std = train_df.mean(), train_df.std()
        else:
            print(f"Invalid normalization method: {normalize_data}.")
            mean, std = 0, 1

        train_df = (train_df - mean) / std
        val_df = (val_df - mean) / std
        test_df = (test_df - mean) / std
        
        # Save normalization parameters for denormalization during evaluation
        normalization_params = {
            'method': normalize_data,
            'mean': mean,
            'std': std
        }
        import pickle
        with open(join(data_path, 'normalization_params.pkl'), 'wb') as f:
            pickle.dump(normalization_params, f)
        print(f"✓ Saved normalization parameters to {join(data_path, 'normalization_params.pkl')}")
    else:
        # No normalization - save empty params
        normalization_params = None
        import pickle
        with open(join(data_path, 'normalization_params.pkl'), 'wb') as f:
            pickle.dump(normalization_params, f)

    # Save the pickle files
    print(f"\nSaving preprocessed data...")
    train_df.to_pickle(train_data_file)
    val_df.to_pickle(val_data_file)
    test_df.to_pickle(test_data_file)
    print(f"✓ Saved {train_data_file}")
    print(f"✓ Saved {val_data_file}")
    print(f"✓ Saved {test_data_file}")

    # Save stations included file
    stations_included = train_df.columns
    pd.Series(stations_included).to_csv(stations_included_file)
    print(f"✓ Saved {stations_included_file}")
    
    # Create a dummy stations data file with route IDs and fake GPS coordinates
    # This is needed for the GNN models to calculate distances
    # Using approximate coordinates for demonstration (you can update with real locations)
    print(f"\nCreating dummy stations file...")
    
    # Check if stations file already exists
    try:
        existing_stations = pd.read_csv(stations_data_file)
        print(f"✓ Stations file already exists: {stations_data_file}")
    except:
        # Create dummy GPS coordinates (spread them out in a grid pattern)
        num_stations = len(stations_included)
        grid_size = int(np.ceil(np.sqrt(num_stations)))
        
        stations_data = []
        for idx, station_id in enumerate(stations_included):
            # Create a grid of coordinates (adjust base coordinates as needed)
            row = idx // grid_size
            col = idx % grid_size
            # Starting from approximate coordinates, spread stations in a grid
            lat = 40.0 + row * 0.01  # ~1km apart
            lon = -74.0 + col * 0.01  # ~1km apart
            stations_data.append({
                'id': int(station_id),
                'latitude': lat,
                'longitude': lon,
                'name': f'Route_{station_id}'
            })
        
        stations_df = pd.DataFrame(stations_data)
        stations_df.to_csv(stations_data_file, index=False)
        print(f"✓ Created dummy stations file: {stations_data_file}")
        print(f"  Note: GPS coordinates are dummy values. Update with real coordinates if available.")
    
    print(f"\n✅ Preprocessing complete!")
    print(f"\nYou can now run: python train_and_evaluate.py")
