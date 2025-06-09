import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import warnings
from tabulate import tabulate

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set display options for better readability
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)  # Fixed width for consistent display
pd.set_option('display.float_format', '{:10.4f}'.format)  # Fixed width for numbers
pd.set_option('display.max_colwidth', 15)  # Shorter column width
pd.set_option('display.expand_frame_repr', True)  # Allow wrapping to prevent horizontal scrolling
pd.set_option('display.max_rows', 10)  # Limit number of rows displayed

# Set seaborn style
sns.set_style("whitegrid")
sns.set_palette('viridis')
plt.rcParams['figure.facecolor'] = 'white'  # White background for plots

# Set up paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
processed_dir = os.path.join(project_root, 'DATA', 'processed')

def load_data():
    """Load X/y splits and metadata, then reconstruct full train/test DataFrames."""
    try:
        # Load the X/y splits
        X_train_loaded = pd.read_pickle(os.path.join(processed_dir, 'X_train.pkl'))
        X_test_loaded = pd.read_pickle(os.path.join(processed_dir, 'X_test.pkl'))
        y_train_loaded = pd.read_pickle(os.path.join(processed_dir, 'y_train.pkl'))
        y_test_loaded = pd.read_pickle(os.path.join(processed_dir, 'y_test.pkl'))
        
        # Load split metadata
        with open(os.path.join(processed_dir, 'time_series_splits.pkl'), 'rb') as f:
            import pickle # Keep import here as it was, or move to top
            split_info = pickle.load(f)

        target_name = split_info.get('target_variable_name', 'DAILY_YIELD') # Default if not in metadata

        # Combine X_train and y_train
        X_train_processed = X_train_loaded.reset_index(drop=True)
        y_train_processed = y_train_loaded.reset_index(drop=True).rename(target_name)
        # Ensure y_train_processed is a DataFrame for concat, if it's a Series
        if isinstance(y_train_processed, pd.Series):
            y_train_processed = y_train_processed.to_frame()
        train_df_reconstructed = pd.concat([X_train_processed, y_train_processed], axis=1)

        # Combine X_test and y_test
        X_test_processed = X_test_loaded.reset_index(drop=True)
        y_test_processed = y_test_loaded.reset_index(drop=True).rename(target_name)
        # Ensure y_test_processed is a DataFrame for concat, if it's a Series
        if isinstance(y_test_processed, pd.Series):
            y_test_processed = y_test_processed.to_frame()
        test_df_reconstructed = pd.concat([X_test_processed, y_test_processed], axis=1)

        # Sort DataFrames by engineered time features to ensure .head() shows earliest records
        time_sort_columns = ['month', 'day', 'hour', 'minute']
        # Check if all sort columns exist before attempting to sort
        if all(col in train_df_reconstructed.columns for col in time_sort_columns):
            train_df_reconstructed = train_df_reconstructed.sort_values(by=time_sort_columns).reset_index(drop=True)
            print("  Sorted training data by month, day, hour, minute.")
        else:
            print(f"  Warning: Could not sort training data by {time_sort_columns} as one or more columns are missing.")
        
        if all(col in test_df_reconstructed.columns for col in time_sort_columns):
            test_df_reconstructed = test_df_reconstructed.sort_values(by=time_sort_columns).reset_index(drop=True)
            print("  Sorted testing data by month, day, hour, minute.")
        else:
            print(f"  Warning: Could not sort testing data by {time_sort_columns} as one or more columns are missing.")
        
        print("✅ Successfully loaded and reconstructed processed datasets using time series splits")
        print(f"  Training date range (from metadata): {split_info['train_date_range']['start']} to {split_info['train_date_range']['end']}")
        print(f"  Testing date range (from metadata):  {split_info['test_date_range']['start']} to {split_info['test_date_range']['end']}")
        print(f"  Reconstructed train_df shape: {train_df_reconstructed.shape}, columns: {train_df_reconstructed.columns.tolist()[:5]}...")
        print(f"  Reconstructed test_df shape: {test_df_reconstructed.shape}, columns: {test_df_reconstructed.columns.tolist()[:5]}...")

        return {
            'train_df': train_df_reconstructed,
            'test_df': test_df_reconstructed,
            'metadata': split_info
        }
        
    except FileNotFoundError as fnf_error:
        print(f"❌ Error loading processed datasets: File not found - {fnf_error}. Ensure make_features.py has run successfully.")
        return None
    except Exception as e:
        print(f"❌ Error loading and reconstructing processed datasets: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        return None

def load_legacy_data():
    """Legacy function to load data in the old format (for backward compatibility)."""
    try:
        train_df = pd.read_pickle(os.path.join(processed_dir, 'train_data.pkl'))
        test_df = pd.read_pickle(os.path.join(processed_dir, 'test_data.pkl'))
        print("✅ Successfully loaded legacy processed datasets")
        return train_df, test_df
    except Exception as e:
        print(f"❌ Error loading legacy datasets: {e}")
        return None, None

def display_data_preview(df, title, n_rows=5):
    """Display a clean preview of the dataset with horizontal scrolling."""
    print(f"\n{title}")
    print("=" * 80)
    
    # Create a copy to avoid modifying the original
    df_preview = df.head(n_rows).copy()
    
    # Format float columns to 4 decimal places
    float_cols = df_preview.select_dtypes(include=['float32', 'float64']).columns
    for col in float_cols:
        df_preview[col] = df_preview[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "NaN")
    
    # Reset index to show row numbers
    df_preview = df_preview.reset_index(drop=True)
    
    if df_preview.shape[1] > 20: # If more than 20 columns, transpose for better readability
        print("DataFrame has many columns. First, showing key time features for the earliest records (non-transposed):")
        time_features = ['month', 'day', 'hour', 'minute']
        # Ensure all time features are present before trying to display them
        actual_time_features_present = [col for col in time_features if col in df.columns]
        if actual_time_features_present:
            print(df[actual_time_features_present].head(n_rows).to_string())
        else:
            print("(Could not display key time features as columns 'month', 'day', 'hour', 'minute' were not all found.)")
        print("\nNow, displaying .head() transposed (all original columns shown as rows):")
        # When transposing, ensure all rows (original columns) are shown.
        # The number of columns will be small (n_rows).
        with pd.option_context('display.width', 200, 
                               'display.max_rows', None, 
                               'display.max_columns', n_rows + 1, # For index + n_rows 
                               'display.max_colwidth', 50, 
                               'display.precision', 4):
            print(df_preview.T)
    else: # For DataFrames with fewer columns, display horizontally
        # display.width=1000 allows for more horizontal space before wrapping.
        # display.max_columns=None ensures all columns are shown.
        with pd.option_context('display.width', 1000, 
                               'display.max_columns', None, 
                               'display.max_colwidth', 35, 
                               'display.precision', 4):
            print(df_preview.to_string())
    
    print(f"\nShape: {df.shape}")
    print("-" * 80)

def display_dataframe_summary(df, title):
    """Display .info() and transposed .describe() for a DataFrame."""
    print(f"\n{title} - Summary Information")
    print("=" * 80)
    print(f"\n--- {title}: .info() ---")
    df.info(verbose=True, show_counts=True)
    print(f"\n--- {title}: .describe() (transposed) ---")
    with pd.option_context('display.width', 200, 'display.max_columns', None, 'display.precision', 4):
        # Transposing describe() output is often more readable for many features
        print(df.describe(include='all').T)
    print("-" * 80)

def plot_correlation_heatmap(df, title):
    """Plot and save a correlation heatmap for numerical features."""
    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'reports', 'figures')
    os.makedirs(output_dir, exist_ok=True)
    
    # Select only numerical columns
    numeric_cols = df.select_dtypes(include=['float32', 'float64', 'int8', 'int16', 'int32', 'int64']).columns.tolist()
    
    # Skip if no numeric columns
    if not numeric_cols:
        print(f"No numeric columns found for correlation heatmap in {title}")
        return
    
    # Calculate correlation matrix
    corr = df[numeric_cols].corr()
    
    # Create a larger figure for better readability
    plt.figure(figsize=(18, 16))
    
    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # Use a diverging color palette that clearly shows -1 to 1
    cmap = sns.diverging_palette(10, 240, as_cmap=True)
    
    # Create the heatmap
    sns.heatmap(corr, 
                mask=mask,
                cmap=cmap,
                vmin=-1,  # Full range from -1
                vmax=1,   # to 1
                center=0,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"})
    
    # Improve the plot appearance
    plt.title(f'Correlation Heatmap - {title}', pad=20, fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    
    # Save the figure
    filename = f"correlation_heatmap_{title.lower().replace(' ', '_')}.png"
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved correlation plot to: {save_path}")
    
    # Show the plot
    plt.show()

def main():
    print("\n" + "="*80)
    print("PROCESSED DATA EXPLORATION".center(80))
    print("="*80)
    
    # Load the data with time series splits
    data = load_data()
    
    # Fall back to legacy data loading if new format fails
    if data is None:
        print("\n⚠️  Could not load data with time series splits. Trying legacy format...")
        train_df, test_df = load_legacy_data()
        if train_df is None or test_df is None:
            return
    else:
        train_df = data['train_df']
        test_df = data['test_df']
    
    # Display data previews with a subset of rows
    print("\n" + " LOADING DATA PREVIEWS ".center(80, "#"))
    display_data_preview(train_df, "TRAINING DATA (First 5 Rows)", n_rows=5)
    display_data_preview(test_df, "TESTING DATA (First 5 Rows)", n_rows=5)

    # Display DataFrame summaries (.info() and .describe())
    print("\n" + " DATAFRAME SUMMARIES (.info() & .describe()) ".center(80, "#"))
    display_dataframe_summary(train_df, "Training Data")
    display_dataframe_summary(test_df, "Testing Data")
    
    # Ask user if they want to see more rows
    show_more = input("\nShow more rows? (y/n): ").lower()
    if show_more == 'y':
        n_rows = int(input("How many rows to display? (suggested: 10-20): ") or "10")
        display_data_preview(train_df, f"TRAINING DATA (First {n_rows} Rows)", n_rows=n_rows)
        display_data_preview(test_df, f"TESTING DATA (First {n_rows} Rows)", n_rows=n_rows)
    
    # Plot correlation heatmaps
    print("\n" + " GENERATING CORRELATION HEATMAPS ".center(80, "#"))
    plot_correlation_heatmap(train_df, "Training Data")
    plot_correlation_heatmap(test_df, "Testing Data")
    
    # The train/test split visualization that relied on DATE_TIME has been removed.
    # We can still report the size of the train and test sets.
    print("\n" + " DATA SPLIT INFORMATION ".center(80, "#"))
    if train_df is not None and test_df is not None:
        print(f"Number of samples in Training set: {len(train_df)}")
        print(f"Number of samples in Testing set:  {len(test_df)}")
        if len(train_df) + len(test_df) > 0:
            test_percentage = (len(test_df) / (len(train_df) + len(test_df))) * 100
            print(f"Test set constitutes {test_percentage:.2f}% of the total processed data.")
    else:
        print("Train/Test data not available to show split information.")

if __name__ == "__main__":
    main()
