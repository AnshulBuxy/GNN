import pandas as pd
import numpy as np
from datetime import datetime

def convert_traffic_data(input_excel_path, output_csv_path):
    """
    Convert traffic data from Excel format to CSV with routes as columns and timestamps as rows.
    Only extracts Harmonic Average Speed data.
    
    Parameters:
    -----------
    input_excel_path : str
        Path to the input Excel file
    output_csv_path : str
        Path to save the output CSV file
    """
    
    # Read the Excel file - need to skip metadata rows and find the actual header
    print("Reading Excel file...")
    # First, read without headers to inspect the structure
    df_raw = pd.read_excel(input_excel_path, header=None)
    
    # Find the row with 'Route Id' (the actual header row)
    header_row = None
    for idx, row in df_raw.iterrows():
        if 'Route Id' in str(row.values):
            header_row = idx
            print(f"Found header row at index: {idx}")
            break
    
    if header_row is None:
        raise ValueError("Could not find 'Route Id' header in the Excel file")
    
    # Now read the file with the correct header row
    df = pd.read_excel(input_excel_path, header=header_row)
    
    # Display original columns for verification
    print(f"\nOriginal columns: {df.columns.tolist()}")
    print(f"Original shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    # Extract only the required columns
    df_filtered = df[['Route Id', 'Date Range', 'Time Set', 'Harmonic Average Speed [kph]']].copy()
    
    # Rename columns for easier processing
    df_filtered.columns = ['RouteId', 'DateRange', 'TimeSet', 'HarmonicSpeed']
    
    print(f"\nProcessing {len(df_filtered)} records...")
    print(f"Columns: {df_filtered.columns.tolist()}")
    
    # Create a combined datetime column
    # Assuming Date Range is like "Aug1" and Time Set is like "3:00-4:00"
    def parse_datetime(row):
        date_str = str(row['DateRange'])
        time_str = str(row['TimeSet'])
        
        # Extract start time from time range (e.g., "3:00-4:00" -> "3:00")
        start_time = time_str.split('-')[0].strip()
        
        # Parse date (assuming format like "Aug1" means August 1, 2015)
        # Adjust the year as needed
        try:
            # Extract month and day
            month_map = {
                'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
            }
            
            month_str = ''.join([c for c in date_str.lower() if c.isalpha()])
            day_str = ''.join([c for c in date_str if c.isdigit()])
            
            if month_str in month_map and day_str:
                month = month_map[month_str]
                day = int(day_str)
                
                # Combine with time
                hour, minute = map(int, start_time.split(':'))
                
                # Create datetime (using 2015 as base year, adjust if needed)
                dt = pd.Timestamp(year=2015, month=month, day=day, hour=hour, minute=minute)
                return dt
        except:
            pass
        
        return pd.NaT
    
    print("\nCreating timestamps...")
    df_filtered['Timestamp'] = df_filtered.apply(parse_datetime, axis=1)
    
    # Remove rows with invalid timestamps
    df_filtered = df_filtered.dropna(subset=['Timestamp'])
    
    # Create pivot table with Timestamp as index and RouteId as columns
    print("Pivoting data...")
    
    # Use RouteId as the column identifier
    pivot_df = df_filtered.pivot_table(
        index='Timestamp',
        columns='RouteId',
        values='HarmonicSpeed',
        aggfunc='mean'  # In case there are duplicates, take the mean
    )
    
    # Sort by timestamp
    pivot_df = pivot_df.sort_index()
    
    # Format the index as shown in the example image
    pivot_df.index = pivot_df.index.strftime('%Y-%m-%d %H:%M:%S+00:00')
    
    print(f"\nFinal shape: {pivot_df.shape}")
    print(f"Date range: {pivot_df.index[0]} to {pivot_df.index[-1]}")
    print(f"Number of routes: {len(pivot_df.columns)}")
    
    # Save to CSV
    print(f"\nSaving to {output_csv_path}...")
    pivot_df.to_csv(output_csv_path)
    
    print("✓ Conversion complete!")
    print(f"\nPreview of output:")
    print(pivot_df.head())
    
    return pivot_df


if __name__ == "__main__":
    # Update these paths to match your file locations
    input_file = "jobs_8218491_results_24-30.xlsx"  # Your input Excel file
    output_file = "traffic_data_converted_24_30.csv"  # Desired output CSV file
    
    try:
        result = convert_traffic_data(input_file, output_file)
    except FileNotFoundError:
        print(f"\n❌ Error: Could not find input file '{input_file}'")
        print("Please update the 'input_file' path in the script to point to your Excel file.")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\nPlease check:")
        print("1. The Excel file path is correct")
        print("2. The column names match your data")
        print("3. You have the required packages installed (pandas, openpyxl)")
