import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys

def plot_correlation_heatmap(df, title, output_dir):
    """Generate and save a correlation heatmap for the given DataFrame."""
    print(f"\nPlotting correlation heatmap for: {title}")
    
    numeric_df = df.select_dtypes(include=np.number)

    if 'PLANT_ID' in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=['PLANT_ID'])
    
    if numeric_df.empty:
        print("  No numeric columns to plot.")
        return

    correlation_matrix = numeric_df.corr()
    
    plt.figure(figsize=(max(12, len(numeric_df.columns) * 0.5), max(10, len(numeric_df.columns) * 0.4)))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, annot_kws={"size": 8})
    plt.title(f'Correlation Heatmap: {title}', fontsize=15)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    
    plot_filename = f"{title.replace(' ', '_').replace(':', '').lower()}_correlation_heatmap.png"
    output_plot_path = os.path.join(output_dir, plot_filename)
    
    try:
        plt.savefig(output_plot_path, dpi=300)
        print(f"  Correlation heatmap saved to: {output_plot_path}")
    except Exception as e:
        print(f"  Error saving correlation heatmap: {e}")
    plt.close()

def load_and_profile_data(file_path, output_dir):
    """Load data, print profile, and generate heatmap."""
    file_name = os.path.basename(file_path)
    header_text = f"Processing file: {file_name}"
    print(header_text)
    print("=" * len(header_text))

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None

    try:
        df = pd.read_csv(file_path)
        
        # Convert DATE_TIME column, handling potential format issues
        if 'DATE_TIME' in df.columns:
            df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'], dayfirst=True, errors='coerce')

        print(f"\nShape: {df.shape}")
        print("\nDataFrame Info:")
        df.info()
        
        print("\nFirst 5 rows:")
        print(df.head())
        
        print("\nMissing values:")
        missing_values = df.isnull().sum()
        print(missing_values[missing_values > 0] if missing_values.sum() > 0 else "No missing values.")

        plot_correlation_heatmap(df, file_name, output_dir)
        return df

    except Exception as e:
        print(f"Error processing file {file_name}: {e}")
        return None

def merge_plant_data(gen_df, weather_df):
    """Merge generation and weather data for a single plant."""
    try:
        merged_df = pd.merge(
            gen_df, 
            weather_df, 
            on=['DATE_TIME', 'PLANT_ID'], 
            how='inner'
        )
        return merged_df
    except Exception as e:
        print(f"Error during merge: {e}")
        return None

def main():
    """Main function to run the EDA script."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_data_dir = os.path.join(base_dir, 'DATA')
    
    script_name_prefix = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    output_plot_dir = os.path.join(base_dir, 'reports', 'figures', script_name_prefix)
    os.makedirs(output_plot_dir, exist_ok=True)

    print(f"Looking for data in: {raw_data_dir}")
    print(f"Saving plots to: {output_plot_dir}\n")

    # --- 1. Load all data first ---
    plant1_gen_path = os.path.join(raw_data_dir, 'Plant_1_Generation_Data.csv')
    plant1_weather_path = os.path.join(raw_data_dir, 'Plant_1_Weather_Sensor_Data.csv')
    plant2_gen_path = os.path.join(raw_data_dir, 'Plant_2_Generation_Data.csv')
    plant2_weather_path = os.path.join(raw_data_dir, 'Plant_2_Weather_Sensor_Data.csv')

    df_p1_gen = load_and_profile_data(plant1_gen_path, output_plot_dir)
    print("\n" + "="*80 + "\n")
    df_p1_weather = load_and_profile_data(plant1_weather_path, output_plot_dir)
    print("\n" + "="*80 + "\n")
    df_p2_gen = load_and_profile_data(plant2_gen_path, output_plot_dir)
    print("\n" + "="*80 + "\n")
    df_p2_weather = load_and_profile_data(plant2_weather_path, output_plot_dir)
    print("\n" + "="*80 + "\n")

    # --- 2. Merging and Final Analysis ---
    print("Starting data merging process...")
    if all(df is not None for df in [df_p1_gen, df_p1_weather, df_p2_gen, df_p2_weather]):
        merged_plant1 = merge_plant_data(df_p1_gen, df_p1_weather)
        merged_plant2 = merge_plant_data(df_p2_gen, df_p2_weather)

        if merged_plant1 is not None and merged_plant2 is not None:
            print("\n--- Final Combined DataFrame --- ")
            combined_df = pd.concat([merged_plant1, merged_plant2], ignore_index=True)
            print("Shape:", combined_df.shape)
            print("\nFirst 5 rows:")
            print(combined_df.head())
            print("\nDescriptive Statistics:")
            print(combined_df.describe(include='all'))
            plot_correlation_heatmap(combined_df, "Final_Combined_Dataset", output_plot_dir)

            # --- 3. Save the final combined DataFrame ---
            pickle_file_path = os.path.join(raw_data_dir, 'merged_solar_data.pkl')
            try:
                combined_df.to_pickle(pickle_file_path)
                print(f"\nSuccessfully saved combined data to: {pickle_file_path}")
            except Exception as e:
                print(f"\nError saving combined data to pickle file: {e}")
    else:
        print("Skipping merge process: One or more data files failed to load.")

    print("\nScript finished.")

if __name__ == "__main__":
    main()
