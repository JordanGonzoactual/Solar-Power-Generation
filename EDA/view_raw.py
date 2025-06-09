import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys # For getting script name for plot saving

# Define the path to the RawData directory
# Assuming the script is in EDA directory, and RawData is in the parent directory
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
raw_data_dir = os.path.join(base_dir, 'DATA') # Changed from 'RawData' to 'DATA'

# List of CSV files to process
files_to_process = [
    'Plant_1_Generation_Data.csv',
    'Plant_1_Weather_Sensor_Data.csv',
    'Plant_2_Generation_Data.csv',
    'Plant_2_Weather_Sensor_Data.csv'
]


def plot_correlation_heatmap(df, title):
    """Generate and save a correlation heatmap for the given DataFrame."""
    print(f"\nPlotting correlation heatmap for: {title}")
    
    # Select only numeric columns for correlation
    numeric_df = df.select_dtypes(include=np.number)

    # Exclude 'PLANT_ID' if it's present among numeric columns, as it's an identifier
    if 'PLANT_ID' in numeric_df.columns:
        print("  Excluding 'PLANT_ID' from correlation heatmap as it's an identifier.")
        numeric_df = numeric_df.drop(columns=['PLANT_ID'])
    
    if numeric_df.empty:
        print("  No numeric columns found to plot correlation heatmap.")
        return

    # Drop columns with no variance (all same value) as they result in NaN correlations
    no_variance_cols = numeric_df.columns[numeric_df.nunique() <= 1]
    if not no_variance_cols.empty:
        print(f"  Dropping columns with no variance before correlation: {list(no_variance_cols)}")
        numeric_df = numeric_df.drop(columns=no_variance_cols)
        if numeric_df.empty:
            print("  No numeric columns with variance left. Skipping heatmap.")
            return

    correlation_matrix = numeric_df.corr()
    
    plt.figure(figsize=(max(12, len(numeric_df.columns)*0.5), max(10, len(numeric_df.columns)*0.4)))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, annot_kws={"size": 8})
    plt.title(f'Correlation Heatmap: {title}', fontsize=15)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    
    # Save the plot
    script_name_prefix = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    # Get project root assuming script is in EDA folder: .. (parent) / .. (project root)
    project_root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_plot_dir = os.path.join(project_root_path, 'reports', 'figures', script_name_prefix)
    os.makedirs(output_plot_dir, exist_ok=True)
    
    plot_filename = f"{script_name_prefix}_{title.replace(' ', '_').replace(':', '').lower()}_correlation_heatmap.png"
    output_plot_path = os.path.join(output_plot_dir, plot_filename)
    
    try:
        plt.savefig(output_plot_path, dpi=300)
        print(f"  Correlation heatmap saved to: {output_plot_path}")
    except Exception as e:
        print(f"  Error saving correlation heatmap: {e}")
    plt.close() # Close the plot to free memory

print(f"Looking for data in: {raw_data_dir}\n")
# Check if essential variables for the initial exploration are defined
if 'raw_data_dir' not in globals() and 'raw_data_dir' not in locals():
    print("Error: 'raw_data_dir' is not defined for the initial exploration loop.")
    print("Please ensure the cell(s) defining 'base_dir' and 'raw_data_dir' (usually at the top of the script/notebook) have been executed.")
elif 'files_to_process' not in globals() and 'files_to_process' not in locals():
    print("Error: 'files_to_process' is not defined for the initial exploration loop.")
    print("Please ensure the cell defining 'files_to_process' (usually at the top of the script/notebook) has been executed.")
else:
    for file_name in files_to_process:
        file_path = os.path.join(raw_data_dir, file_name)
        header_text = f"Processing file: {file_name}"
        print(header_text)
        print("=" * len(header_text))
        # print(f"Full file path: {file_path}") # Uncomment for debugging path issues
        
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                print(f"\nShape: {df.shape}")
                
                print("\nFirst 10 rows:")
                print(df.head(10))
                
                print("\nMissing values per column:")
                missing_values = df.isnull().sum()
                print(missing_values[missing_values > 0] if not missing_values[missing_values > 0].empty else "No missing values.")
                
                print("\nNumber of duplicate rows:")
                num_duplicates = df.duplicated().sum()
                print(f"{num_duplicates} duplicate rows found." if num_duplicates > 0 else "No duplicate rows.")
                
                print("\nDataFrame info:")
                df.info()
                
                print("\nDescriptive statistics:")
                print(df.describe(include='all'))
                
            except Exception as e:
                print(f"Error processing file {file_name}: {e}")
        else:
            print(f"File not found: {file_path}")
        print("\n" + "="*80 + "\n") # Separator after each file's processing

print("Data exploration complete.")

# <<< START OF MERGE CODE >>>
# Merging Plant Generation Data with Weather Sensor Data

print("\n" + "="*80)
print("Starting data merging process...")

# Ensure necessary imports are available (pandas, os are at top of file)
# from IPython.display import display # Will be imported conditionally later

# Check if raw_data_dir is defined, as it's crucial for file paths.
# The original script defines raw_data_dir near the top.
# If this script is run in cells (e.g., Jupyter), ensure the cell defining raw_data_dir is executed first.
if 'raw_data_dir' not in globals() and 'raw_data_dir' not in locals():
    print("Error: 'raw_data_dir' is not defined. Please ensure the initial path setup cells (lines defining base_dir and raw_data_dir) have been run.")
    print("Skipping merging process due to missing raw_data_dir.")
    all_files_present_for_merge = False
else:
    plant1_gen_path = os.path.join(raw_data_dir, 'Plant_1_Generation_Data.csv')
    plant1_weather_path = os.path.join(raw_data_dir, 'Plant_1_Weather_Sensor_Data.csv')
    plant2_gen_path = os.path.join(raw_data_dir, 'Plant_2_Generation_Data.csv')
    plant2_weather_path = os.path.join(raw_data_dir, 'Plant_2_Weather_Sensor_Data.csv')

    all_files_present_for_merge = True
    required_files_for_merge = {
        "Plant 1 Generation Data": plant1_gen_path,
        "Plant 1 Weather Data": plant1_weather_path,
        "Plant 2 Generation Data": plant2_gen_path,
        "Plant 2 Weather Data": plant2_weather_path
    }

    for name, pth in required_files_for_merge.items():
        if not os.path.exists(pth):
            print(f"Error: Required data file for merging not found: {name} at {pth}")
            all_files_present_for_merge = False
            break

if all_files_present_for_merge:
    try:
        print("\nLoading datasets for merging...")
        df_plant1_gen = pd.read_csv(plant1_gen_path)
        df_plant1_weather = pd.read_csv(plant1_weather_path)
        df_plant2_gen = pd.read_csv(plant2_gen_path)
        df_plant2_weather = pd.read_csv(plant2_weather_path)

        datasets_for_merge = {
            "Plant 1 Generation": df_plant1_gen,
            "Plant 1 Weather": df_plant1_weather,
            "Plant 2 Generation": df_plant2_gen,
            "Plant 2 Weather": df_plant2_weather
        }

        print("\nConverting DATE_TIME columns to datetime objects...")
        for name, df_item in datasets_for_merge.items():
            if 'DATE_TIME' in df_item.columns:
                df_item['DATE_TIME'] = pd.to_datetime(df_item['DATE_TIME'], errors='coerce')
            else:
                print(f"Warning: 'DATE_TIME' column not found in {name} dataframe.")
            # Ensure PLANT_ID is suitable for merging (e.g., consistent type)
            # if 'PLANT_ID' in df_item.columns:
            #     df_item['PLANT_ID'] = df_item['PLANT_ID'].astype(int) # Or str, depending on data

        # Perform merges for each plant
        print("\nMerging Plant 1 Generation and Weather data...")
        merged_plant1 = pd.merge(
            df_plant1_gen,
            df_plant1_weather,
            on=['DATE_TIME', 'PLANT_ID'],
            how='inner', # Use 'inner' to keep only records with matching DATE_TIME and PLANT_ID in both datasets
            suffixes=('_gen', '_weather') # Appends suffix to overlapping column names (excluding keys)
        )
        print(f"Plant 1 merged data shape: {merged_plant1.shape}")

        print("\nMerging Plant 2 Generation and Weather data...")
        merged_plant2 = pd.merge(
            df_plant2_gen,
            df_plant2_weather,
            on=['DATE_TIME', 'PLANT_ID'],
            how='inner',
            suffixes=('_gen', '_weather')
        )
        print(f"Plant 2 merged data shape: {merged_plant2.shape}")

        # Concatenate the two plant-specific merged dataframes
        print("\nConcatenating Plant 1 and Plant 2 merged data into 'merged_df'...")
        merged_df = pd.concat([merged_plant1, merged_plant2], ignore_index=True)
        print(f"Final merged_df shape: {merged_df.shape}")

        # Display the new dataset
        header_merged_df = "\nDisplaying details for the final merged_df:"
        print(header_merged_df)
        print("=" * len(header_merged_df.strip())) # .strip() to remove leading newline for length calculation
        print("\nFirst 5 rows:")
        # Removed IPython.display dependency
        print(merged_df.head())

        print("\nInfo:")
        merged_df.info(verbose=True, show_counts=True)

        print("\nDescriptive statistics:")
        print(merged_df.describe(include='all'))

        print("\nMissing values per column:")
        missing_values_merged = merged_df.isnull().sum()
        if not missing_values_merged[missing_values_merged > 0].empty:
            print(missing_values_merged[missing_values_merged > 0])
        else:
            print("No missing values in merged_df.")

        # Plot correlation heatmap for the merged data
        plot_correlation_heatmap(merged_df, "Raw Merged Data")

        # Pickle the merged_df to the DATA folder
        data_dir_path = os.path.join(base_dir, 'DATA') # base_dir is defined at the top of the script
        if not os.path.exists(data_dir_path):
            os.makedirs(data_dir_path)
            print(f"\nCreated directory: {data_dir_path}")
        
        pickle_file_path = os.path.join(data_dir_path, 'merged_solar_data.pkl')
        try:
            merged_df.to_pickle(pickle_file_path)
            print(f"\nSuccessfully saved merged_df to: {pickle_file_path}")
        except Exception as e:
            print(f"\nError saving merged_df to pickle file: {e}")

    except FileNotFoundError as e:
        print(f"Error during merge: A data file was not found. This might be due to 'raw_data_dir' not being set correctly.")
        print(f"Details: {e}")
    except KeyError as e:
        print(f"Error during merge: A key column (e.g., 'DATE_TIME' or 'PLANT_ID') was not found in one of the dataframes.")
        print(f"Please check column names. Details: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during merging: {e}")
        import traceback
        traceback.print_exc()
elif 'raw_data_dir' in globals() or 'raw_data_dir' in locals(): # only print if raw_data_dir was defined but files were missing
    print("Skipping merging process due to missing data files.")

print("\nData merging process complete.")
print("="*80 + "\n")
# <<< END OF MERGE CODE >>>


