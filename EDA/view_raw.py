import pandas as pd
import os

# Define the path to the RawData directory
# Assuming the script is in EDA directory, and RawData is in the parent directory
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
raw_data_dir = os.path.join(base_dir, 'RawData')

# List of CSV files to process
files_to_process = [
    'Plant_1_Generation_Data.csv',
    'Plant_1_Weather_Sensor_Data.csv',
    'Plant_2_Generation_Data.csv',
    'Plant_2_Weather_Sensor_Data.csv'
]

print(f"Looking for data in: {raw_data_dir}\n")

for file_name in files_to_process:
    file_path = os.path.join(raw_data_dir, file_name)
    print(f"Processing file: {file_name}")
    # print(f"Full file path: {file_path}") # Uncomment for debugging path issues
    
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            
            print("\nFirst 10 rows:")
            # Using display() for better output in Jupyter
            from IPython.display import display
            display(df.head(10))
            
            print("\nMissing values per column:")
            missing_values = df.isnull().sum()
            display(missing_values[missing_values > 0] if not missing_values[missing_values > 0].empty else "No missing values.")
            
            print("\nNumber of duplicate rows:")
            num_duplicates = df.duplicated().sum()
            display(f"{num_duplicates} duplicate rows found." if num_duplicates > 0 else "No duplicate rows.")
            
            print("\nDataFrame info:")
            df.info()
            
            print("\nDescriptive statistics:")
            display(df.describe(include='all'))
            
        except Exception as e:
            print(f"Error processing file {file_name}: {e}")
    else:
        print(f"File not found: {file_path}")
    print("\n" + "="*80 + "\n")

print("Data exploration complete.")

