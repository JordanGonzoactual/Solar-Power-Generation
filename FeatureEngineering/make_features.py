import pandas as pd
import os
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils import check_array
from sklearn.preprocessing import LabelEncoder
import json
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)  # Suppress FutureWarnings

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

# --- Time-based train/test split using TimeSeriesSplit ---
print("\nPerforming time-based train/test split using TimeSeriesSplit...")

# Prepare raw X and y for TimeSeriesSplit
# Ensure 'DAILY_YIELD' exists before dropping
if 'DAILY_YIELD' not in df.columns:
    print("Error: 'DAILY_YIELD' target column not found in the DataFrame before splitting.")
    exit()

X_raw = df.drop(columns=['DAILY_YIELD'])
y_raw = df['DAILY_YIELD']

n_samples = len(X_raw)

# Use 5-fold TimeSeriesSplit
n_splits_val = 5
print(f"  Using TimeSeriesSplit with {n_splits_val} folds")

# Initialize TimeSeriesSplit with 5 folds
tscv = TimeSeriesSplit(n_splits=n_splits_val, test_size=None, gap=0)

# Get all splits for analysis
splits = list(tscv.split(X_raw))
print(f"  Total samples: {len(X_raw)}")

# Log the size of each fold
for i, (tr_idx, te_idx) in enumerate(splits):
    print(f"  Fold {i+1}: Train size={len(tr_idx)}, Test size={len(te_idx)}", end="")
    if len(te_idx) > 0:
        print(f" ({(len(te_idx)/len(X_raw))*100:.1f}% of data)")
    else:
        print()

# Use the last split for final train/test
train_index, test_index = splits[-1]
print(f"\n  Using fold {n_splits_val} as test set with {len(test_index)} samples ({(len(test_index)/len(X_raw))*100:.1f}% of data)")
print(f"  Training set size: {len(train_index)} samples ({(len(train_index)/len(X_raw))*100:.1f}% of data)")

# Verify we have enough samples
if len(train_index) == 0 or len(test_index) == 0:
    print("Error: One of the splits is empty. Check your data and split parameters.")
    exit()

# Create raw training and testing DataFrames using the original df to keep all columns initially
train_df_raw = df.iloc[train_index].copy()
test_df_raw = df.iloc[test_index].copy()

# Store original DATE_TIME series for metadata, as apply_feature_engineering might alter/drop it
if 'DATE_TIME' in train_df_raw.columns:
    train_dates_original = train_df_raw['DATE_TIME'].copy()
else:
    print("Error: 'DATE_TIME' column missing from train_df_raw before feature engineering.")
    exit()
if 'DATE_TIME' in test_df_raw.columns:
    test_dates_original = test_df_raw['DATE_TIME'].copy()
else:
    print("Error: 'DATE_TIME' column missing from test_df_raw before feature engineering.")
    exit()

print(f"  Raw training set size (before feature engineering): {len(train_df_raw)} rows ({len(train_df_raw)/len(df)*100:.1f}%)")
print(f"  Raw testing set size (before feature engineering): {len(test_df_raw)} rows ({len(test_df_raw)/len(df)*100:.1f}%)")
print(f"  Raw training y-target shape: {y_raw.iloc[train_index].shape}")
print(f"  Raw testing y-target shape: {y_raw.iloc[test_index].shape}")
print(f"  Raw training date range: {train_df_raw['DATE_TIME'].min()} to {train_df_raw['DATE_TIME'].max()}")
print(f"  Raw testing date range: {test_df_raw['DATE_TIME'].min()} to {test_df_raw['DATE_TIME'].max()}")
# Note: The 'split_data' dictionary is removed from here; it will be redefined later with processed data info.

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
    # The merge operation creates SOURCE_KEY_x and SOURCE_KEY_y from the generator and weather data respectively
    columns_to_encode = ['SOURCE_KEY_x', 'SOURCE_KEY_y', 'PLANT_ID']
    
    # Define a mapping from column names to the desired JSON file names
    mapping_filenames = {
        'SOURCE_KEY_x': 'SOURCE_KEY_gen_mapping.json',
        'SOURCE_KEY_y': 'SOURCE_KEY_weather_mapping.json'
    }

    encodings_dir = os.path.join(project_root, 'models', 'encodings')
    os.makedirs(encodings_dir, exist_ok=True)

    for col_name in columns_to_encode:
        if col_name not in df.columns:
            print(f"Warning: Column '{col_name}' not found for encoding. Skipping.")
            continue

        print(f"  Encoding column: {col_name}")
        
        try:
            # Ensure column is of string type before encoding for the LabelEncoder
            df[col_name] = df[col_name].astype(str)
            
            le = LabelEncoder()
            df[col_name] = le.fit_transform(df[col_name])
            
            # Convert the transformed integer column to a 'category' dtype as requested
            df[col_name] = df[col_name].astype('category')
            
            # Create and save the mapping from original labels to integer codes
            mapping = {label: int(code) for label, code in zip(le.classes_, le.transform(le.classes_))}
            
            # Use the predefined filename for the mapping file, or default to the column name
            filename = mapping_filenames.get(col_name, f'{col_name}_mapping.json')
            mapping_file = os.path.join(encodings_dir, filename)

            with open(mapping_file, 'w') as f:
                json.dump(mapping, f, indent=2)
            print(f"    Saved {col_name} mapping to {mapping_file}")
            print(f"    Mapping: {mapping}")
            
        except Exception as e:
            print(f"    Error encoding column {col_name}: {str(e)}")
            continue
    
    # Create new time-based features with explicit categorical types
    try:
        df['month'] = df['DATE_TIME'].dt.month.astype('int8')
        df['day'] = df['DATE_TIME'].dt.day.astype('int8')
        df['hour'] = df['DATE_TIME'].dt.hour.astype('int8')
        df['minute'] = df['DATE_TIME'].dt.minute.astype('int8')
        
        # Convert to categorical with explicit categories
        df['month'] = pd.Categorical(df['month'], categories=range(1, 13), ordered=True)
        df['day'] = pd.Categorical(df['day'], categories=range(1, 32), ordered=True)
        df['hour'] = pd.Categorical(df['hour'], categories=range(0, 24), ordered=True)
        df['minute'] = pd.Categorical(df['minute'], categories=range(0, 60), ordered=True)
        
        print("  Created time-based categorical features: 'month', 'day', 'hour', 'minute'")
    except Exception as e:
        print(f"  Error creating time-based features: {str(e)}")
    
    # --- Add new features ---
    print("\nAdding new features...")
    
    # 1. is_peak_hour - convert hour to int for comparison
    try:
        hour_as_int = df['hour'].astype(int)
        df['is_peak_hour'] = ((hour_as_int >= 6) & (hour_as_int < 18)).astype('int8')
        df['is_peak_hour'] = df['is_peak_hour'].astype('category')
        print("  Created 'is_peak_hour' feature")
    except Exception as e:
        print(f"  Error creating 'is_peak_hour' feature: {str(e)}")
    
    # 2. Sort once for all subsequent operations
    try:
        df = df.sort_values(by=['PLANT_ID', 'DATE_TIME']).copy()
    except Exception as e:
        print(f"  Error sorting DataFrame: {str(e)}")
    
    # 3. Create date column for grouping (temporary) - CUMSUM REMOVED
    # date_group = df['DATE_TIME'].dt.normalize()  # Faster than dt.date
    
    # 4. Calculate cumulative sums using vectorized operations - CUMSUM REMOVED
    # df['cumsum_dc_power'] = df.groupby(['PLANT_ID', date_group])['DC_POWER'].cumsum()
    # df['cumsum_ac_power'] = df.groupby(['PLANT_ID', date_group])['AC_POWER'].cumsum()
    # print("  Removed 'cumsum_dc_power' and 'cumsum_ac_power'.")
    
    # Inverter efficiency using numpy where for vectorized conditional logic
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

    # --- Calculate daytime statistics (6 AM to 6 PM) ---
    print("\nCalculating daytime statistics (6 AM to 6 PM)...")
    if 'DATE_TIME' not in df.columns:
        print("  Error: DATE_TIME column required for daytime stats, but not found.")
    elif 'hour' not in df.columns:
        print("  Error: 'hour' column required for daytime stats, but not found.")
    else:
        df['date_only_temp'] = df['DATE_TIME'].dt.date
        
        daytime_mask = (df['hour'].astype(int) >= 6) & (df['hour'].astype(int) < 18)
        df_daytime_only = df[daytime_mask].copy()

        metrics_for_daytime_stats = ['AC_POWER', 'DC_POWER', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION']
        aggregations_map = {}
        for metric in metrics_for_daytime_stats:
            if metric in df_daytime_only.columns:
                aggregations_map[metric] = ['mean', 'min', 'max', 'std']
            else:
                print(f"  Warning: Metric {metric} not found in daytime data for stats calculation. Skipping.")
        
        if df_daytime_only.empty or not aggregations_map:
            print("  No daytime data or no valid metrics to aggregate. Skipping daytime stats calculation.")
        else:
            # --- Daytime statistics (mean, min, max, std) calculation REMOVED ---
            # try:
            #     daytime_aggregated_stats = df_daytime_only.groupby(['PLANT_ID', 'date_only_temp'], observed=True).agg(aggregations_map)
            #     
            #     new_col_names = []
            #     for col_level0, col_level1 in daytime_aggregated_stats.columns:
            #         new_col_names.append(f"daytime_{col_level0.lower()}_{col_level1}")
            #     daytime_aggregated_stats.columns = new_col_names
            #     daytime_aggregated_stats.reset_index(inplace=True)
            #     
            #     df = pd.merge(df, daytime_aggregated_stats, on=['PLANT_ID', 'date_only_temp'], how='left')
            #     print(f"  Daytime statistics (mean, min, max, std) features REMOVED.") # Adjusted print message
            # except Exception as e:
            #     print(f"  Error during (now removed) daytime stats calculation: {e}")
            print("  Daytime statistics (mean, min, max, std) features REMOVED.")

        if 'date_only_temp' in df.columns:
            df = df.drop(columns=['date_only_temp'])

    # --- Add lag features ---
    print("\nAdding lag features...")
    metrics_for_lag = ['AC_POWER', 'DC_POWER', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION']
    # Lags in terms of number of 15-minute periods
    lag_periods_map = {
        '1h': 4,    # 1 hour * 4 samples/hour
        '24h': 96,  # 24 hours * 4 samples/hour
        '72h': 288  # 72 hours * 4 samples/hour
    }

    # Ensure DataFrame is sorted by PLANT_ID and DATE_TIME for correct lag calculation
    df = df.sort_values(by=['PLANT_ID', 'DATE_TIME']).copy()

    # Store original index to restore later
    original_index = df.index
    
    for metric in metrics_for_lag:
        if metric in df.columns:
            for lag_name, periods in lag_periods_map.items():
                col_name = f"{metric.lower()}_lag_{lag_name}"
                # Calculate lagged values within each PLANT_ID group
                df[col_name] = df.groupby('PLANT_ID')[metric].transform(
                    lambda x: x.shift(periods).fillna(method='ffill').fillna(0)
                )
                print(f"  Added lag feature: {col_name} (NaN values forward filled within each PLANT_ID group)")
        else:
            print(f"  Warning: Metric {metric} not found for lag feature calculation. Skipping.")
    
    # Restore original index
    df = df.reindex(original_index)
    
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
    # Apply feature engineering to the raw training and testing DataFrames
    # Pass copies to apply_feature_engineering to avoid modifying raw DFs if function modifies inplace
    print("\nProcessing raw training data for feature engineering...")
    train_df_processed = apply_feature_engineering(train_df_raw.copy())
    
    print("\nProcessing raw testing data for feature engineering...")
    test_df_processed = apply_feature_engineering(test_df_raw.copy())

    # Create final X_train and X_test from processed dataframes by dropping the target column
    # errors='ignore' ensures no error if 'DAILY_YIELD' was already removed or not present
    if 'DAILY_YIELD' not in train_df_processed.columns or 'DAILY_YIELD' not in test_df_processed.columns:
        print("Warning: 'DAILY_YIELD' not found in one or both processed dataframes when creating final X splits.")
    # Ensure DATE_TIME is available in processed dataframes for metadata generation
    # Re-attach if apply_feature_engineering dropped it. This uses the original DATE_TIME values.
    if 'DATE_TIME' not in train_df_processed.columns and 'train_dates_original' in locals():
        print("Re-attaching original DATE_TIME to train_df_processed for metadata.")
        train_df_processed['DATE_TIME'] = train_dates_original
    if 'DATE_TIME' not in test_df_processed.columns and 'test_dates_original' in locals():
        print("Re-attaching original DATE_TIME to test_df_processed for metadata.")
        test_df_processed['DATE_TIME'] = test_dates_original

    # Create final X_train and X_test. 
    # DAILY_YIELD is the target. DATE_TIME is used for metadata and to derive time features;
    # it's assumed not to be a direct model input feature itself here.
    cols_to_drop_for_X = ['DAILY_YIELD']
    if 'DATE_TIME' in train_df_processed.columns: # Check if it was preserved or re-added
        cols_to_drop_for_X.append('DATE_TIME')
    
    X_train_final = train_df_processed.drop(columns=cols_to_drop_for_X, errors='ignore')
    X_test_final = test_df_processed.drop(columns=cols_to_drop_for_X, errors='ignore')

    # Define y_train_final and y_test_final from the processed dataframes to ensure index alignment
    if 'DAILY_YIELD' in train_df_processed.columns:
        y_train_final = train_df_processed['DAILY_YIELD'].copy()
        print(f"  y_train_final created from processed data, shape: {y_train_final.shape}")
    else:
        print("CRITICAL ERROR: 'DAILY_YIELD' not found in train_df_processed. Cannot create y_train_final.")
        exit("Exiting due to missing DAILY_YIELD in train_df_processed.")

    if 'DAILY_YIELD' in test_df_processed.columns:
        y_test_final = test_df_processed['DAILY_YIELD'].copy()
        print(f"  y_test_final created from processed data, shape: {y_test_final.shape}")
    else:
        print("CRITICAL ERROR: 'DAILY_YIELD' not found in test_df_processed. Cannot create y_test_final.")
        exit("Exiting due to missing DAILY_YIELD in test_df_processed.")

    # Define directory for processed data
    output_dir = os.path.join(project_root, 'DATA', 'processed')
    os.makedirs(output_dir, exist_ok=True)

    # Define comprehensive metadata for the splits using processed data information
    split_metadata = {
        'original_train_indices': train_index.tolist(), # Original indices from TimeSeriesSplit, saved as list
        'original_test_indices': test_index.tolist(),   # Original indices from TimeSeriesSplit, saved as list
        'feature_names': X_train_final.columns.tolist(),
        'target_variable_name': 'DAILY_YIELD', # Key expected by view_processed.py
        'train_date_range': {
            'start': str(train_df_processed['DATE_TIME'].min()), 
            'end': str(train_df_processed['DATE_TIME'].max())
        },
        'test_date_range': {
            'start': str(test_df_processed['DATE_TIME'].min()), 
            'end': str(test_df_processed['DATE_TIME'].max())
        }
        # Add other relevant info like split type, parameters, etc. if needed
    }

    # Define file paths for final X/y splits and metadata
    x_train_path = os.path.join(output_dir, 'X_train.pkl')
    x_test_path = os.path.join(output_dir, 'X_test.pkl')
    y_train_path = os.path.join(output_dir, 'y_train.pkl')
    y_test_path = os.path.join(output_dir, 'y_test.pkl')
    metadata_path = os.path.join(output_dir, 'time_series_splits.pkl')

    try:
        # Save final X/y splits
        X_train_final.to_pickle(x_train_path)
        X_test_final.to_pickle(x_test_path)
        y_train_final.to_pickle(y_train_path)
        y_test_final.to_pickle(y_test_path)
        
        # Save metadata (using pd.to_pickle for consistency if all elements are pickle-friendly)
        pd.to_pickle(split_metadata, metadata_path)
        
        print(f"\nSuccessfully saved X_train_final to {x_train_path}")
        print(f"Successfully saved X_test_final to {x_test_path}")
        print(f"Successfully saved y_train_final to {y_train_path}")
        print(f"Successfully saved y_test_final to {y_test_path}")
        print(f"Successfully saved split metadata to {metadata_path}")
        
        # Print final split information
        print("\nFinal Split Information (Post-Feature Engineering):")
        print(f"  X_train_final shape: {X_train_final.shape}")
        print(f"  X_test_final shape: {X_test_final.shape}")
        print(f"  y_train_final shape: {y_train_final.shape}")
        print(f"  y_test_final shape: {y_test_final.shape}")
        print(f"  Number of features in X_train_final: {len(split_metadata['feature_names'])}")

    except Exception as e:
        print(f"\nError saving processed X/y splits and metadata: {e}")
        exit()

    # Display final information
    print("\n" + "="*50)
    print("Feature Engineering and Data Saving Complete!")
    print("="*50)
    # Print shapes of the final saved X matrices, which represent the features for the model
    print(f"Final X_train_final (features for training) shape: {X_train_final.shape}")
    print(f"Final X_test_final (features for testing) shape: {X_test_final.shape}")
    print("\nColumns in the final X_train_final (features for training):")
    print("-" * 30)
    for col in X_train_final.columns:
        print(f"- {col} ({X_train_final[col].dtype})")
    
    print("\nScript 'make_features.py' finished processing.")
