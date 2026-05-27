import pandas as pd
import numpy as np

def speed_to_flow(speed):
    """
    Convert speed to flow using the equation: y = -8.6462x² + 488.78x - 4082.6
    where x = speed (kph) and y = flow (vehicles/hour)
    
    If the result is negative, set it to 0.-5.4086x2 + 248.51x
    """
    flow = -5.4086 * (speed ** 2) + 248.51 * speed
    # Set negative values to 0
    return max(0, flow)

def convert_csv_speed_to_flow(input_file, output_file):
    """
    Convert speed values in CSV to flow values.
    
    Parameters:
    -----------
    input_file : str
        Path to input CSV with speed data
    output_file : str
        Path to save output CSV with flow data
    """
    print(f"Reading {input_file}...")
    df = pd.read_csv(input_file, index_col=0, parse_dates=True)
    
    print(f"Original data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nSample speed data (first 5 rows):")
    print(df.head())
    
    print("\nConverting speed to flow using: y = -8.6462x² + 488.78x - 4082.6")
    print("(Negative values will be set to 0)")
    
    # Apply the conversion formula to all values
    df_flow = df.applymap(lambda speed: speed_to_flow(speed) if not pd.isna(speed) else speed)
    
    print(f"\nSample flow data (first 5 rows):")
    print(df_flow.head())
    
    # Print statistics
    print("\n" + "="*60)
    print("CONVERSION STATISTICS")
    print("="*60)
    print(f"\nOriginal Speed Statistics:")
    print(df.describe())
    print(f"\nConverted Flow Statistics:")
    print(df_flow.describe())
    
    # Count how many values were set to 0 (negative flow)
    zero_count = (df_flow == 0).sum().sum()
    total_values = df_flow.notna().sum().sum()
    print(f"\nValues set to 0 (negative flow): {zero_count} / {total_values} ({100*zero_count/total_values:.2f}%)")
    
    # Save to CSV
    print(f"\nSaving flow data to {output_file}...")
    df_flow.to_csv(output_file)
    
    print(f"✓ Conversion complete!")
    print(f"✓ Flow data saved to: {output_file}")
    
    return df_flow


if __name__ == "__main__":
    input_file = "traffic_data_converted.csv"
    output_file = "traffic_flow_full_data.csv"
    
    try:
        flow_df = convert_csv_speed_to_flow(input_file, output_file)
    except FileNotFoundError:
        print(f"\n❌ Error: Could not find input file '{input_file}'")
        print("Please make sure the file exists in the current directory.")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
