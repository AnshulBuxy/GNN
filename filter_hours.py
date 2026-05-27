import pandas as pd

def remove_early_hours(input_file, output_file, start_hour=0, end_hour=6):
    """
    Remove rows from CSV where the hour is between start_hour and end_hour (exclusive of end_hour).
    Also replace zero values with the average of preceding and following values in the same column.
    
    Parameters:
    -----------
    input_file : str
        Path to input CSV file
    output_file : str
        Path to save filtered CSV file
    start_hour : int
        Start hour to remove (default: 0)
    end_hour : int
        End hour to remove (default: 6, meaning 0-5 hours are removed)
    """
    print(f"Reading {input_file}...")
    df = pd.read_csv(input_file, index_col=0, parse_dates=True)
    
    print(f"Original data shape: {df.shape}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    
    # Count zeros before replacement
    zero_count_before = (df == 0).sum().sum()
    print(f"\nZero values found: {zero_count_before}")
    
    # Replace zeros with average of preceding and following values
    print("Replacing zero values with average of neighboring values...")
    for col in df.columns:
        # Find indices where values are zero
        zero_indices = df[col] == 0
        zero_positions = df.index[zero_indices]
        
        for idx in zero_positions:
            row_position = df.index.get_loc(idx)
            
            # Get preceding value
            prev_val = df[col].iloc[row_position - 1] if row_position > 0 else None
            # Get following value
            next_val = df[col].iloc[row_position + 1] if row_position < len(df) - 1 else None
            
            # Calculate average of non-zero neighbors
            neighbors = [v for v in [prev_val, next_val] if v is not None and v != 0]
            
            if neighbors:
                avg_val = sum(neighbors) / len(neighbors)
                df.loc[idx, col] = avg_val
    
    zero_count_after = (df == 0).sum().sum()
    print(f"Zero values remaining: {zero_count_after}")
    print(f"Replaced {zero_count_before - zero_count_after} zero values")
    
    # Get the hour from the timestamp index
    hours = df.index.hour
    
    # Count rows to be removed
    rows_to_remove = ((hours >= start_hour) & (hours < end_hour)).sum()
    print(f"\nRows with hours {start_hour}-{end_hour-1}: {rows_to_remove}")
    
    # Keep rows where hour is NOT in the range [start_hour, end_hour)
    df_filtered = df[(hours < start_hour) | (hours >= end_hour)]
    
    print(f"Filtered data shape: {df_filtered.shape}")
    print(f"Removed {rows_to_remove} rows ({100*rows_to_remove/len(df):.2f}%)")
    print(f"\nRemaining date range: {df_filtered.index[0]} to {df_filtered.index[-1]}")
    
    # Save filtered data
    df_filtered.to_csv(output_file)
    print(f"\n✓ Filtered data saved to: {output_file}")
    
    return df_filtered


if __name__ == "__main__":
    input_file = "traffic_flow_data.csv"
    output_file = "traffic_flow_data_filtered.csv"
    
    try:
        filtered_df = remove_early_hours(input_file, output_file, start_hour=0, end_hour=6)
        print("\n✅ Successfully processed data!")
        print("   - Replaced zero values with averages")
        print("   - Removed rows with hours 0-5")
    except FileNotFoundError:
        print(f"\n❌ Error: Could not find input file '{input_file}'")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
