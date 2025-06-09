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
train_path = os.path.join(processed_dir, 'train_data.pkl')
test_path = os.path.join(processed_dir, 'test_data.pkl')

def load_data():
    """Load the processed train and test datasets."""
    try:
        train_df = pd.read_pickle(train_path)
        test_df = pd.read_pickle(test_path)
        print("✅ Successfully loaded processed datasets")
        return train_df, test_df
    except Exception as e:
        print(f"❌ Error loading processed datasets: {e}")
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
    
    # Display the table with horizontal scrolling
    with pd.option_context('display.width', None, 'display.max_columns', None, 'display.max_colwidth', 30):
        print(df_preview.to_string())
    
    print(f"\nShape: {df.shape}")
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
    
    # Load the data
    train_df, test_df = load_data()
    
    if train_df is None or test_df is None:
        return
    
    # Display data previews with a subset of rows
    print("\n" + " LOADING DATA PREVIEWS ".center(80, "#"))
    display_data_preview(train_df, "TRAINING DATA (First 5 Rows)", n_rows=5)
    display_data_preview(test_df, "TESTING DATA (First 5 Rows)", n_rows=5)
    
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

if __name__ == "__main__":
    main()
