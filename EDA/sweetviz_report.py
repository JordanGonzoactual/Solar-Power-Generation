import os
import sys
import warnings
import pandas as pd
import numpy as np
from datetime import datetime

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning, module='pkg_resources')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Workaround for numpy deprecation warning
import numpy
if not hasattr(numpy, 'VisibleDeprecationWarning'):
    numpy.VisibleDeprecationWarning = type('VisibleDeprecationWarning', (Warning,), {})

def check_versions():
    """Check and print package versions for debugging."""
    import pkg_resources
    packages = ['pandas', 'numpy', 'sweetviz', 'matplotlib']
    print("\nPackage Versions:")
    for pkg in packages:
        try:
            version = pkg_resources.get_distribution(pkg).version
            print(f"{pkg}: {version}")
        except Exception as e:
            print(f"Could not get version for {pkg}: {e}")

def generate_sweetviz_report():
    """
    Generate a Sweetviz EDA report for the merged solar dataset.
    The report will be saved as an HTML file in the reports directory.
    """
    print("\n" + "="*80)
    print("GENERATING SWEETVIZ EDA REPORT".center(80))
    print("="*80)
    
    # Import sweetviz here to handle any potential import errors
    try:
        import sweetviz as sv
    except ImportError as e:
        print(f"Error importing sweetviz: {e}")
        print("Please install sweetviz using: pip install sweetviz")
        return
    
    # Check versions first
    check_versions()
    
    try:
        import sweetviz as sv
    except ImportError:
        print("\nERROR: Sweetviz is not installed. Installing now...")
        try:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "sweetviz"])
            import sweetviz as sv
            print("Sweetviz installed successfully!")
        except Exception as e:
            print(f"Failed to install sweetviz: {e}")
            return

    # Construct the absolute path to the data files
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_data_dir = os.path.join(project_root, 'DATA', 'processed')
    output_dir = os.path.join(project_root, 'reports', 'sweetviz')
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f'training_data_sweetviz_report_{timestamp}.html')
    
    print(f"\nLoading processed training data (X_train, y_train) and metadata...")
    
    try:
        import pickle # Ensure pickle is imported
        X_train = pd.read_pickle(os.path.join(processed_data_dir, 'X_train.pkl'))
        y_train = pd.read_pickle(os.path.join(processed_data_dir, 'y_train.pkl'))
        with open(os.path.join(processed_data_dir, 'time_series_splits.pkl'), 'rb') as f:
            metadata = pickle.load(f)
        
        target_name = metadata.get('target_variable_name', 'DAILY_YIELD') # Default if not found
        
        print(f"Successfully loaded X_train (shape: {X_train.shape}) and y_train (shape: {y_train.shape}).")
        print(f"Target variable for Sweetviz: {target_name}")

        # Reconstruct the training DataFrame for Sweetviz
        # Ensure y_train is a Series and then rename it to the target_name
        if isinstance(y_train, pd.DataFrame) and y_train.shape[1] == 1:
            y_train_series = y_train.iloc[:, 0].copy()
        elif isinstance(y_train, pd.Series):
            y_train_series = y_train.copy()
        else:
            raise ValueError("y_train is not a Series or a single-column DataFrame.")
        
        y_train_series.name = target_name
        df_for_sweetviz = pd.concat([X_train.reset_index(drop=True), y_train_series.reset_index(drop=True)], axis=1)
        print(f"Reconstructed DataFrame for Sweetviz with shape: {df_for_sweetviz.shape}")

        # Check if target column exists in the reconstructed DataFrame
        if target_name not in df_for_sweetviz.columns:
            raise ValueError(f"'{target_name}' column not found in the reconstructed training data for Sweetviz.")
        
        # Generate the Sweetviz report
        print("\nGenerating Sweetviz report. This may take a moment...")
        
        try:
            # Generate the Sweetviz report with the correct target feature
            report = sv.analyze(
                df_for_sweetviz, 
                target_feat=target_name, 
                pairwise_analysis='on' # Keep pairwise analysis if desired
            )
            report.show_html(output_file, open_browser=False)
            print(f"\n✅ Sweetviz report saved to: {output_file}")
            print("Open the HTML file in your web browser to view the report.")
            print(f"Note: Using '{target_name}' as the target variable for analysis.")
            
        except Exception as e:
            print(f"\n⚠️  Error generating full report: {str(e)}")
            print("Trying with a smaller sample of the data...")
            
            try:
                # Try with a smaller sample
                sample_df = df.sample(min(5000, len(df)), random_state=42)
                report = sv.analyze(sample_df, target_feat='DAILY_YIELD', pairwise_analysis='on')
                output_file = output_file.replace('.html', '_SAMPLE.html')
                report.show_html(output_file, open_browser=False)
                print(f"\n✅ Generated report with sample data: {output_file}")
            except Exception as e2:
                print(f"\n❌ Failed to generate report: {str(e2)}")
                print("\nTroubleshooting tips:")
                print("1. Try updating packages: pip install --upgrade pandas numpy sweetviz")
                print("2. Try with a smaller dataset first")
                print("3. Check the Sweetviz documentation for known issues")
                    
    except FileNotFoundError:
        print(f"\n❌ Error: Data file not found at {data_file_path}")
        print("Please make sure you've run the data merging step first.")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    generate_sweetviz_report()