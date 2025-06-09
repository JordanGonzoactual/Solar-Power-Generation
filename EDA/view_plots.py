import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_daily_metrics_comparison():
    # Construct the absolute path to the data file
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_file_path = os.path.join(project_root, 'DATA', 'merged_solar_data.pkl')
    eda_folder = os.path.dirname(os.path.abspath(__file__))

    # Load the merged dataset
    try:
        df = pd.read_pickle(data_file_path)
        print(f"Successfully loaded {data_file_path}")
    except FileNotFoundError:
        print(f"Error: The file {data_file_path} was not found. Please ensure the file exists.")
        return
    except Exception as e:
        print(f"An error occurred while loading the pickle file: {e}")
        return

    # Verify essential columns
    required_cols = ['DATE_TIME', 'PLANT_ID', 'DC_POWER', 'AC_POWER', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}. Available: {df.columns.tolist()}")
        return

    # Convert 'DATE_TIME' column to datetime objects and set as index
    try:
        df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'])
    except Exception as e:
        print(f"Error converting 'DATE_TIME' column to datetime: {e}")
        return
    df = df.set_index('DATE_TIME')

    plant_ids = sorted(df['PLANT_ID'].unique())

    metrics_to_plot = {
        'DC_POWER': 'DC Power (kW)',
        'AC_POWER': 'AC Power (kW)',
        'AMBIENT_TEMPERATURE': 'Ambient Temperature (°C)',
        'MODULE_TEMPERATURE': 'Module Temperature (°C)',
        'IRRADIATION': 'Irradiation'
    }

    if not plant_ids:
        print("No plant IDs found in the data.")
        return

    for metric_col, metric_title_base in metrics_to_plot.items():
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 7), squeeze=False)
        fig.suptitle(f'{metric_title_base} - Daily Averages Comparison', fontsize=16, y=1.0)

        # Subplot 1: All-Day Averages
        ax1 = axes[0, 0]
        for plant_id in plant_ids:
            plant_data = df[df['PLANT_ID'] == plant_id]
            if not plant_data.empty and metric_col in plant_data.columns:
                daily_avg = plant_data[metric_col].resample('D').mean()
                if not daily_avg.empty:
                    daily_avg.plot(ax=ax1, label=f'Plant {plant_id}', marker='.', linestyle='-')
        
        ax1.set_title('All Day (24 hours)', fontsize=12)
        ax1.set_xlabel('Date', fontsize=10)
        ax1.set_ylabel(f'Daily Avg {metric_title_base}', fontsize=10)
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.tick_params(axis='x', rotation=45, labelsize=8)
        ax1.tick_params(axis='y', labelsize=8)

        # Subplot 2: Average Hourly Profile
        ax2 = axes[0, 1]
        for plant_id in plant_ids:
            plant_data = df[df['PLANT_ID'] == plant_id]
            if not plant_data.empty and metric_col in plant_data.columns:
                # Group by hour of the day and calculate the mean for that hour across all days
                # Ensure the index is sorted for a proper line plot (0-23 hours)
                hourly_avg_profile = plant_data[metric_col].groupby(plant_data.index.hour).mean().sort_index()
                if not hourly_avg_profile.empty:
                    hourly_avg_profile.plot(ax=ax2, label=f'Plant {plant_id}', marker='.', linestyle='-')
        
        ax2.set_title('Average Hourly Profile', fontsize=12)
        ax2.set_xlabel('Hour of Day (0-23)', fontsize=10)
        ax2.set_ylabel(f'Average Hourly {metric_title_base}', fontsize=10)
        ax2.set_xticks(range(0, 24, 2)) # Set x-ticks to be every 2 hours for clarity
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.tick_params(axis='x', rotation=45, labelsize=8)
        ax2.tick_params(axis='y', labelsize=8)

        plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout for suptitle
        
        save_filename = f"{metric_col.lower().replace(' ', '_')}_comparison_plot.png"
        save_path = os.path.join(eda_folder, save_filename)
        
        try:
            plt.savefig(save_path)
            print(f"Successfully saved plot: {save_path}")
        except Exception as e:
            print(f"Error saving plot {save_path}: {e}")
        
        plt.close(fig) # Close the figure to free up memory

if __name__ == '__main__':
    plot_daily_metrics_comparison()
