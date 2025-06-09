import pandas as pd
import os
import numpy as np

# Construct the absolute path to the data file
# __file__ is the path to the current script (make_features.py)
# os.path.dirname(__file__) is FeatureEngineering directory
# os.path.dirname(os.path.dirname(__file__)) is Solar-Power-generation directory (project root)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_file_path = os.path.join(project_root, 'DATA', 'merged_solar_data.pkl')

# Load the merged dataset
try:
    df = pd.read_pickle(data_file_path)
    print(f"Successfully loaded {data_file_path}")
except FileNotFoundError:
    print(f"Error: The file {data_file_path} was not found. Please ensure the file exists at this location.")
    exit()
except Exception as e:
    print(f"An error occurred while loading the pickle file: {e}")
    exit()

# Verify the 'DATE_TIME' column exists
if 'DATE_TIME' not in df.columns:
    print("Error: 'DATE_TIME' column not found in the DataFrame.")
    print(f"Available columns are: {df.columns.tolist()}")
    exit()

# Convert 'DATE_TIME' column to datetime objects and sort
try:
    df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'])
    # Sort by DATE_TIME to ensure proper time series split
    df = df.sort_values('DATE_TIME').reset_index(drop=True)
except Exception as e:
    print(f"Error processing 'DATE_TIME' column: {e}")
    print("Please ensure the 'DATE_TIME' column has a consistent and parseable format.")
    exit()

# --- Time-based train/test split (80/20) ---
print("\nPerforming time-based train/test split...")

# Calculate the split index for 80% training data
split_idx = int(len(df) * 0.8)

# Split the data
train_df = df.iloc[:split_idx].copy()
test_df = df.iloc[split_idx:].copy()

print(f"  Training set: {len(train_df)} rows ({len(train_df)/len(df)*100:.1f}%)")
print(f"  Testing set: {len(test_df)} rows ({len(test_df)/len(df)*100:.1f}%)")
print(f"  Training date range: {train_df['DATE_TIME'].min()} to {train_df['DATE_TIME'].max()}")
print(f"  Testing date range: {test_df['DATE_TIME'].min()} to {test_df['DATE_TIME'].max()}")

# Function to apply feature engineering to a DataFrame
def apply_feature_engineering(df):
    """Apply all feature engineering steps to the given DataFrame."""
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # --- Apply all the feature engineering steps to the DataFrame ---
    
    # Ensure PLANT_ID in DataFrame is of a numeric type for comparison
    df['PLANT_ID'] = pd.to_numeric(df['PLANT_ID'], errors='coerce')
    
    # --- Categorical Encoding ---
    print("\nPerforming categorical encoding...")
    columns_to_encode = ['SOURCE_KEY_gen', 'SOURCE_KEY_weather', 'PLANT_ID']
    all_mappings = {}  # To store all mappings for later reference
    
    # Create directory for saving encodings if it doesn't exist
    import os
    import json
    encodings_dir = os.path.join(project_root, 'models', 'encodings')
    os.makedirs(encodings_dir, exist_ok=True)

    for col_name in columns_to_encode:
        if col_name not in df.columns:
            print(f"Warning: Column '{col_name}' not found for encoding. Skipping.")
            continue

        print(f"  Encoding column: {col_name}")
        
        # Convert to string and then to category
        df[col_name] = df[col_name].astype(str)
        df[col_name] = df[col_name].astype('category')
        
        # Create mapping from category to code
        current_mapping = {str(category): int(code) for code, category in enumerate(df[col_name].cat.categories)}
        all_mappings[col_name] = current_mapping
        
        # Apply the encoding
        df[col_name] = df[col_name].cat.codes
        df[col_name] = df[col_name].astype('category')
        
        # Save the mapping to a JSON file
        mapping_file = os.path.join(encodings_dir, f'{col_name}_mapping.json')
        with open(mapping_file, 'w') as f:
            json.dump(current_mapping, f, indent=2)
        print(f"    Saved {col_name} mapping to {mapping_file}")
        print(f"    Mapping: {current_mapping}")
    
    # Create new time-based features
    df['month'] = df['DATE_TIME'].dt.month
    df['day'] = df['DATE_TIME'].dt.day
    df['hour'] = df['DATE_TIME'].dt.hour
    df['minute'] = df['DATE_TIME'].dt.minute
    
    # --- Add new features ---
    print("\nAdding new features...")
    
    # 1. is_peak_hour
    df['is_peak_hour'] = ((df['hour'] >= 6) & (df['hour'] < 18)).astype(int)
    
    # 2. Sort once for all subsequent operations
    df = df.sort_values(by=['PLANT_ID', 'DATE_TIME']).copy()
    
    # 3. Create date column for grouping (temporary)
    date_group = df['DATE_TIME'].dt.normalize()  # Faster than dt.date
    
    # 4. Calculate cumulative sums using vectorized operations
    df['cumsum_dc_power'] = df.groupby(['PLANT_ID', date_group])['DC_POWER'].cumsum()
    df['cumsum_ac_power'] = df.groupby(['PLANT_ID', date_group])['AC_POWER'].cumsum()
    
    # 5. Inverter efficiency using numpy where for vectorized conditional logic
    df['inverter_efficiency'] = np.where(
        df['DC_POWER'] != 0,
        (df['AC_POWER'] / df['DC_POWER']) * 100,
        0.0
    )
    
    # 6. Inverter input flag: 1 if DC_POWER == 0, else 0
    df['inverter_input'] = (df['DC_POWER'] == 0).astype('uint8')  # Binary flag (0 or 1)
    
    # --- Add rolling averages for key metrics ---
    print("Adding rolling averages...")
    metrics = ['AC_POWER', 'DC_POWER', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION']
    window_sizes = [24, 72]  # 24 hours and 72 hours
    
    # Pre-compute the groupby object once for better performance
    group_by_plant = df.groupby('PLANT_ID')
    
    # Process each metric and window size
    for metric in metrics:
        if metric in df.columns:
            # Pre-compute the rolling windows for each plant
            for window in window_sizes:
                col_name = f'{metric.lower()}_rolling_avg_{window}'
                # Use simple integer window for rolling mean
                # Convert window from hours to number of samples (assuming 15-minute intervals)
                samples_per_hour = 4  # 4 samples per hour (15-min intervals)
                window_samples = window * samples_per_hour
                df[col_name] = group_by_plant[metric].transform(
                    lambda x: x.rolling(window=window_samples, min_periods=1).mean()
                )
    
    # Clear the groupby object to free memory
    del group_by_plant
    
    # Remove the original DATETIME column as we've extracted all time-based features
    if 'DATE_TIME' in df.columns:
        df = df.drop(columns=['DATE_TIME'])
    
    # --- Optimize data types ---
    print("Optimizing data types...")
    
    # Handle DAILY_YIELD separately
    if 'DAILY_YIELD' in df.columns:
        df['DAILY_YIELD'] = df['DAILY_YIELD'].astype('float32')
    
    # Convert other float columns to float32
    float_columns = df.select_dtypes(include=['float64', 'float16']).columns.tolist()
    if 'DAILY_YIELD' in float_columns:
        float_columns.remove('DAILY_YIELD')  # Already float32
    
    for col in float_columns:
        try:
            df[col] = df[col].astype('float32')
        except Exception as e:
            print(f"  Could not convert '{col}' to float32: {e}")
    
    # Optimize integer columns
    for col in df.select_dtypes(include=['int64', 'int32']).columns:
        try:
            if df[col].min() >= 0:  # Unsigned int
                if df[col].max() < 256:
                    df[col] = df[col].astype('uint8')
                elif df[col].max() < 65536:
                    df[col] = df[col].astype('uint16')
                elif df[col].max() < 4294967295:
                    df[col] = df[col].astype('uint32')
            else:  # Signed int
                if df[col].between(-128, 127).all():
                    df[col] = df[col].astype('int8')
                elif df[col].between(-32768, 32767).all():
                    df[col] = df[col].astype('int16')
        except Exception as e:
            print(f"  Could not optimize '{col}': {e}")
    
    return df

# --- Main Execution ---
if __name__ == "__main__":
    # Apply feature engineering to both train and test sets
    print("\nProcessing training data...")
    train_df = apply_feature_engineering(train_df)
    
    print("\nProcessing testing data...")
    test_df = apply_feature_engineering(test_df)

    # Save the processed datasets
    output_dir = os.path.join(project_root, 'DATA', 'processed')
    os.makedirs(output_dir, exist_ok=True)
    
    train_output_path = os.path.join(output_dir, 'train_data.pkl')
    test_output_path = os.path.join(output_dir, 'test_data.pkl')

    try:
        train_df.to_pickle(train_output_path)
        test_df.to_pickle(test_output_path)
        print(f"\nSuccessfully saved training data to {train_output_path}")
        print(f"Successfully saved testing data to {test_output_path}")
    except Exception as e:
        print(f"\nError saving processed data: {e}")
        exit()

    # Display final information
    print("\n" + "="*50)
    print("Feature Engineering Complete!")
    print("="*50)
    print(f"Final training set shape: {train_df.shape}")
    print(f"Final testing set shape: {test_df.shape}")
    print("\nColumns in the final datasets:")
    print("-" * 30)
    for col in train_df.columns:
        print(f"- {col} ({train_df[col].dtype})")
    
    print("\nScript 'make_features.py' finished processing.")
