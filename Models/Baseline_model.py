import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats

# Define paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, 'DATA', 'processed')
FIGURES_OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'reports', 'figures', 'baseline_model')

# Ensure output directory exists
os.makedirs(FIGURES_OUTPUT_DIR, exist_ok=True)

def load_data():
    """Loads preprocessed train and test data and metadata."""
    try:
        X_train = pd.read_pickle(os.path.join(PROCESSED_DATA_DIR, 'X_train.pkl'))
        X_test = pd.read_pickle(os.path.join(PROCESSED_DATA_DIR, 'X_test.pkl'))
        y_train = pd.read_pickle(os.path.join(PROCESSED_DATA_DIR, 'y_train.pkl'))
        y_test = pd.read_pickle(os.path.join(PROCESSED_DATA_DIR, 'y_test.pkl'))
        
        with open(os.path.join(PROCESSED_DATA_DIR, 'time_series_splits.pkl'), 'rb') as f:
            metadata = pickle.load(f)
        
        feature_names = metadata.get('feature_names', X_train.columns.tolist())
        target_name = metadata.get('target_variable_name', 'DAILY_YIELD')
        
        print("Data loaded successfully.")
        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
        return X_train, X_test, y_train, y_test, feature_names, target_name
    except FileNotFoundError as e:
        print(f"Error loading data: {e}. Please ensure 'make_features.py' has been run.")
        return None, None, None, None, None, None

def plot_feature_importances(importances, feature_names, output_path):
    """Plots feature importances."""
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(12, max(6, len(feature_names) // 2)))
    plt.title("Feature Importances")
    plt.bar(range(len(feature_names)), importances[indices], align="center")
    plt.xticks(range(len(feature_names)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Feature importances plot saved to {output_path}")

def plot_residuals_vs_predicted(y_true, y_pred, title, output_path):
    """Plots residuals vs. predicted values."""
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals (Actual - Predicted)")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Residuals vs. Predicted plot saved to {output_path}")

def plot_qq(residuals, title, output_path):
    """Generates a Q-Q plot for residuals."""
    plt.figure(figsize=(8, 6))
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Q-Q plot saved to {output_path}")

def plot_actual_vs_predicted(y_true, y_pred, title, output_path):
    """Plots actual vs. predicted values."""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, y_true, alpha=0.5, label='Data points')
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal y=x line')
    plt.xlabel("Predicted Values")
    plt.ylabel("Actual Values")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.axis('equal') # Ensure aspect ratio is equal for y=x line to look correct
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Actual vs. Predicted plot saved to {output_path}")

def main():
    X_train, X_test, y_train, y_test, feature_names, target_name = load_data()

    if X_train is None:
        return

    # Ensure y_train and y_test are 1D arrays
    y_train_arr = y_train.values.ravel() if hasattr(y_train, 'values') else y_train
    y_test_arr = y_test.values.ravel() if hasattr(y_test, 'values') else y_test

    # Initialize and train Random Forest Regressor
    print("\nTraining Random Forest Regressor...")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=20, min_samples_split=10, min_samples_leaf=5)
    rf_model.fit(X_train, y_train_arr)
    print("Model training complete.")

    # Make predictions
    y_train_pred = rf_model.predict(X_train)
    y_test_pred = rf_model.predict(X_test)

    # Evaluate metrics
    mse_train = mean_squared_error(y_train_arr, y_train_pred)
    r2_train = r2_score(y_train_arr, y_train_pred)
    mse_test = mean_squared_error(y_test_arr, y_test_pred)
    r2_test = r2_score(y_test_arr, y_test_pred)

    print("\n" + " MODEL EVALUATION METRICS ".center(50, "="))

    print("\n-- Training Set Metrics --")
    print(f"  MSE (Train):          {mse_train:.4f}")
    print(f"  R-squared (Train):    {r2_train:.4f}")

    print("\n-- Testing Set Metrics --")
    print(f"  MSE (Test):           {mse_test:.4f}")
    print(f"  R-squared (Test):     {r2_test:.4f}")

    print("\n-- Metric Differences --")
    print(f"  MSE Difference (Test - Train):     {mse_test - mse_train:.4f}")
    print(f"  R-squared Difference (Train - Test): {r2_train - r2_test:.4f}")
    print("=" * 50)

    # Calculate residuals
    residuals_train = y_train_arr - y_train_pred
    residuals_test = y_test_arr - y_test_pred

    # Generate and save plots
    print("\nGenerating and saving plots...")
    plot_feature_importances(rf_model.feature_importances_, feature_names, 
                             os.path.join(FIGURES_OUTPUT_DIR, 'feature_importances.png'))
    
    plot_residuals_vs_predicted(y_train_arr, y_train_pred, 'Residuals vs. Predicted (Train Set)', 
                                os.path.join(FIGURES_OUTPUT_DIR, 'residuals_vs_predicted_train.png'))
    plot_residuals_vs_predicted(y_test_arr, y_test_pred, 'Residuals vs. Predicted (Test Set)', 
                                os.path.join(FIGURES_OUTPUT_DIR, 'residuals_vs_predicted_test.png'))

    plot_qq(residuals_train, 'Q-Q Plot of Residuals (Train Set)', 
            os.path.join(FIGURES_OUTPUT_DIR, 'qq_plot_residuals_train.png'))
    plot_qq(residuals_test, 'Q-Q Plot of Residuals (Test Set)', 
            os.path.join(FIGURES_OUTPUT_DIR, 'qq_plot_residuals_test.png'))

    plot_actual_vs_predicted(y_train_arr, y_train_pred, 'Actual vs. Predicted (Train Set)',
                             os.path.join(FIGURES_OUTPUT_DIR, 'actual_vs_predicted_train.png'))
    plot_actual_vs_predicted(y_test_arr, y_test_pred, 'Actual vs. Predicted (Test Set)',
                             os.path.join(FIGURES_OUTPUT_DIR, 'actual_vs_predicted_test.png'))
    
    print("\nAll plots generated and saved.")

if __name__ == "__main__":
    main()
