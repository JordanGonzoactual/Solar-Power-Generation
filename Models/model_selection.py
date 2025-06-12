import os
import pandas as pd
import joblib
import numpy as np
import json
import time
import uuid
import logging
from datetime import datetime

# Machine learning imports
import xgboost as xgb
import mlflow
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from sklearn.model_selection import TimeSeriesSplit

# Visualization imports
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
matplotlib.use('Agg')  # Use non-interactive backend for saving figures

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set random seed for reproducibility
NP_RANDOM_SEED = 42
XGB_RANDOM_SEED = 42
np.random.seed(NP_RANDOM_SEED)

# Define MLflow experiment name
MLFLOW_EXPERIMENT_NAME = "Solar_Power_XGBoost_StepWise"

# Define a variable to track if MLflow is available
MLFLOW_AVAILABLE = True

# Create a local directory for MLflow tracking if server is unavailable
mlflow_local_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'mlruns')
os.makedirs(mlflow_local_dir, exist_ok=True)

try:
    # First try to connect to the server
    try:
        mlflow.set_tracking_uri("http://127.0.0.1:8080")
        # Test the connection with a simple API call
        mlflow.search_experiments()
        logger.info(f"MLflow tracking URI set to: http://127.0.0.1:8080")
    except Exception as e:
        # If server connection fails, fall back to local file-based tracking
        logger.warning(f"Could not connect to MLflow server: {e}")
        logger.warning(f"Falling back to local file-based tracking in {mlflow_local_dir}")
        mlflow.set_tracking_uri(f"file://{mlflow_local_dir}")
        logger.info(f"MLflow tracking URI set to: file://{mlflow_local_dir}")
    
    # Set up the experiment
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    logger.info(f"MLflow experiment set to: {MLFLOW_EXPERIMENT_NAME}")
except Exception as e:
    logger.error(f"Error setting MLflow experiment: {e}")
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow logging will be disabled for this run")

# Define safe MLflow helper functions
def safe_mlflow_start_run(run_id=None, nested=False, run_name=None):
    """Start an MLflow run if available, otherwise return a context manager."""
    if MLFLOW_AVAILABLE:
        return mlflow.start_run(run_id=run_id, nested=nested, run_name=run_name)
    else:
        # Return a dummy context manager when MLflow is not available
        class DummyContextManager:
            def __enter__(self):
                logger.info(f"[MLFLOW DISABLED] Would start run: {run_name if run_name else 'unnamed'}")
                return None
            def __exit__(self, exc_type, exc_val, exc_tb):
                return False
        return DummyContextManager()

def safe_mlflow_log_param(key, value):
    """Log parameter if MLflow is available."""
    if MLFLOW_AVAILABLE:
        try:
            mlflow.log_param(key, value)
        except Exception as e:
            logger.warning(f"Failed to log MLflow parameter {key}: {e}")
    else:
        logger.info(f"[MLFLOW DISABLED] Would log param: {key}={value}")

def safe_mlflow_log_params(params):
    """Log parameters if MLflow is available."""
    if MLFLOW_AVAILABLE:
        try:
            mlflow.log_params(params)
        except Exception as e:
            logger.warning(f"Failed to log MLflow parameters: {e}")
    else:
        logger.info(f"[MLFLOW DISABLED] Would log params: {params}")

def safe_mlflow_log_metric(key, value):
    """Log metric if MLflow is available."""
    if MLFLOW_AVAILABLE:
        try:
            mlflow.log_metric(key, value)
        except Exception as e:
            logger.warning(f"Failed to log MLflow metric {key}: {e}")
    else:
        logger.info(f"[MLFLOW DISABLED] Would log metric: {key}={value}")

def safe_mlflow_log_artifact(local_path):
    """Log artifact if MLflow is available."""
    if MLFLOW_AVAILABLE:
        try:
            mlflow.log_artifact(local_path)
        except Exception as e:
            logger.warning(f"Failed to log MLflow artifact {local_path}: {e}")
    else:
        logger.info(f"[MLFLOW DISABLED] Would log artifact: {local_path}")

def setup_logging():
    """Set up logging to file and return the log file path."""
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create a timestamped log file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(logs_dir, f'model_training_{timestamp}.log')
    
    # Add a file handler to the logger
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    return log_file

def load_data(data_path):
    """Loads preprocessed data from the data_path directory."""
    logger.info(f"Loading data from {data_path}")
    try:
        # Define file paths based on our memory about make_features.py output
        X_train_path = os.path.join(data_path, 'X_train.pkl')
        y_train_path = os.path.join(data_path, 'y_train.pkl')
        X_test_path = os.path.join(data_path, 'X_test.pkl')
        y_test_path = os.path.join(data_path, 'y_test.pkl')

        # Load the data
        logger.info(f"Loading training and test datasets...")
        X_train = joblib.load(X_train_path)
        y_train = joblib.load(y_train_path)
        X_test = joblib.load(X_test_path)
        y_test = joblib.load(y_test_path)
        logger.info("Data loaded successfully.")

        # Ensure y_train and y_test are the right format
        if isinstance(y_train, pd.DataFrame) or isinstance(y_train, pd.Series):
            y_train = y_train.values
        if isinstance(y_test, pd.DataFrame) or isinstance(y_test, pd.Series):
            y_test = y_test.values
            
        # Flatten if needed
        if y_train.ndim > 1 and y_train.shape[1] == 1:
            y_train = y_train.ravel()
        if y_test.ndim > 1 and y_test.shape[1] == 1:
            y_test = y_test.ravel()
        
        # Log data dimensions
        logger.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        logger.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

        # Ensure X data is in DataFrame format
        if not isinstance(X_train, pd.DataFrame):
            logger.warning(f"X_train is not a pandas DataFrame. Converting...")
            X_train = pd.DataFrame(X_train)
        if not isinstance(X_test, pd.DataFrame):
            logger.warning(f"X_test is not a pandas DataFrame. Converting...")
            X_test = pd.DataFrame(X_test)

        return X_train, y_train, X_test, y_test
    except FileNotFoundError as e:
        logger.error(f"Error loading data file: {e}. Please ensure make_features.py has been run.")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during data loading: {e}", exc_info=True)
        raise

def cross_validate_xgb(params, X, y, n_folds=5):
    """Perform time series cross-validation for XGBoost with given parameters.
    
    Args:
        params (dict): XGBoost parameters
        X (DataFrame): Features
        y (array): Target variable
        n_folds (int): Number of folds for time series CV
        
    Returns:
        tuple: Mean and std of the cross-validation scores (RMSE), training duration
    """
    tscv = TimeSeriesSplit(n_splits=n_folds)
    scores = []
    durations = []
    fold_info = []
    
    logger.info(f"Performing {n_folds}-fold TimeSeriesSplit cross-validation")
    
    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
        start_time = time.time()
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # Train the model - use params directly which already includes random_state
        # Enable categorical features support
        model = xgb.XGBRegressor(enable_categorical=True, **params)
        model.fit(
            X_train_fold, y_train_fold,
            eval_set=[(X_val_fold, y_val_fold)],
            verbose=False
        )
        
        # Calculate metrics
        y_pred = model.predict(X_val_fold)
        mse = mean_squared_error(y_val_fold, y_pred)
        rmse = np.sqrt(mse)
        fold_duration = time.time() - start_time
        
        scores.append(rmse)
        durations.append(fold_duration)
        fold_info.append({
            "fold": fold_idx + 1,
            "train_size": len(X_train_fold),
            "val_size": len(X_val_fold),
            "rmse": rmse,
            "duration": fold_duration
        })
    
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    mean_duration = np.mean(durations)
    
    logger.info(f"CV Results - Mean RMSE: {mean_score:.4f}, Std: {std_score:.4f}, Avg duration: {mean_duration:.2f}s")
    return mean_score, std_score, fold_info, mean_duration

def save_cv_results_plot(cv_results, title, filename):
    """Create and save a plot showing cross-validation results across folds.
    
    Args:
        cv_results (list): List of dictionaries with fold results
        title (str): Plot title
        filename (str): Output filename
        
    Returns:
        str: Path to saved plot
    """
    plt.figure(figsize=(12, 6))
    
    folds = [result['fold'] for result in cv_results]
    rmses = [result['rmse'] for result in cv_results]
    
    plt.bar(folds, rmses, color='skyblue', alpha=0.7)
    plt.axhline(y=np.mean(rmses), color='red', linestyle='--', label=f'Mean RMSE: {np.mean(rmses):.4f}')
    
    plt.xlabel('Fold')
    plt.ylabel('RMSE')
    plt.title(f'{title} - Cross-validation Results')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(folds)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filename

def create_actual_vs_predicted_plot(y_true, y_pred, title, filename):
    """Create a scatter plot of actual vs predicted values and return the path to the saved file.
    
    Parameters:
    -----------
    y_true : array-like
        The true target values.
    y_pred : array-like
        The predicted target values.
    title : str
        Title for the plot.
    filename : str
        Filename to save the plot to.
        
    Returns:
    --------
    str
        Path to the saved plot file.
    """
    plt.figure(figsize=(10, 6))
    
    # Create scatter plot
    plt.scatter(y_true, y_pred, alpha=0.5)
    
    # Add diagonal line for perfect prediction
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    
    # Add labels and title
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(title)
    
    # Add metrics to plot
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    plt.annotate(f'RMSE: {rmse:.4f}\nMAE: {mae:.4f}\nR²: {r2:.4f}', 
                 xy=(0.05, 0.95), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
                 fontsize=10)
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    # Save the plot to file
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filename

def log_feature_importance(model, feature_names, parent_run_id=None, step_name=""):
    """Plot feature importance and log as artifact in MLflow.
    
    Parameters:
    -----------
    model : XGBoost model
        The trained XGBoost model.
    feature_names : list
        Names of the features used for training.
    parent_run_id : str, optional
        Parent MLflow run ID for logging.
    step_name : str, optional
        Optional step identifier for naming the artifact.
    """
    # Create feature importance DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=importance_df)
    plt.title('XGBoost Feature Importance')
    plt.tight_layout()
    
    # Save plot to file
    plot_file = f"feature_importance_{step_name}.png"
    plt.savefig(plot_file)
    plt.close()
    
    # Log the feature importance plot as an artifact
    safe_mlflow_log_artifact(plot_file)
    
    # Save and log the feature importance as a CSV
    fi_csv = f"feature_importance_{step_name}.csv"
    importance_df.to_csv(fi_csv, index=False)
    safe_mlflow_log_artifact(fi_csv)
    
    # Clean up files
    os.remove(plot_file)
    os.remove(fi_csv)
    
    return plot_file

def evaluate_and_log_metrics(model, X_train, y_train, X_test, y_test):
    """Evaluate model on train and test data, log metrics to MLflow."""
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    
    train_rmse = np.sqrt(train_mse)
    test_rmse = np.sqrt(test_mse)
    
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # Additional metrics - explained variance for regression quality
    train_explained_variance = explained_variance_score(y_train, y_train_pred)
    test_explained_variance = explained_variance_score(y_test, y_test_pred)
    
    # Log metrics to MLflow
    safe_mlflow_log_metric("train_mse", train_mse)
    safe_mlflow_log_metric("test_mse", test_mse)
    safe_mlflow_log_metric("train_rmse", train_rmse)
    safe_mlflow_log_metric("test_rmse", test_rmse)
    safe_mlflow_log_metric("train_mae", train_mae)
    safe_mlflow_log_metric("test_mae", test_mae)
    safe_mlflow_log_metric("train_r2", train_r2)
    safe_mlflow_log_metric("test_r2", test_r2)
    safe_mlflow_log_metric("train_explained_variance", train_explained_variance)
    safe_mlflow_log_metric("test_explained_variance", test_explained_variance)
    
    # Calculate and log delta between train and test metrics (for overfitting assessment)
    rmse_delta = test_rmse - train_rmse
    r2_delta = train_r2 - test_r2
    
    safe_mlflow_log_metric("rmse_delta", rmse_delta)
    safe_mlflow_log_metric("r2_delta", r2_delta)
    
    # Return metrics as a dictionary
    metrics = {
        "train_mse": train_mse,
        "test_mse": test_mse,
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
        "train_mae": train_mae,
        "test_mae": test_mae,
        "train_r2": train_r2,
        "test_r2": test_r2,
        "train_explained_variance": train_explained_variance,
        "test_explained_variance": test_explained_variance,
        "rmse_delta": rmse_delta,
        "r2_delta": r2_delta
    }
    
    # Log summary to console
    logger.info(f"Model Evaluation:")
    logger.info(f"  Training RMSE: {train_rmse:.4f}, R²: {train_r2:.4f}")
    logger.info(f"  Test RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}")
    logger.info(f"  Overfitting Check - RMSE Delta: {rmse_delta:.4f}")
    logger.info(f"  Overfitting Check - R² Delta: {r2_delta:.4f}")
    
    return metrics

def create_actual_vs_predicted_plot(y_true, y_pred, title, filename):
    """Create and save a plot comparing actual vs predicted values.
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        title: Plot title
        filename: Output filename
    """
    plt.figure(figsize=(10, 7))
    plt.scatter(y_true, y_pred, alpha=0.5)
    
    # Add perfect prediction line
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    # Add metrics to plot
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    plt.figtext(0.15, 0.85, f'RMSE: {rmse:.4f}\nR²: {r2:.4f}', 
                bbox=dict(facecolor='white', alpha=0.8))
    
    # Save plot
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filename

def tune_tree_params(X_train, y_train, run_id, n_folds=5):
    """Step 1: Tune tree structure parameters (max_depth, min_child_weight).
    
    Args:
        X_train: Training features
        y_train: Training target
        run_id: Parent MLflow run ID
        n_folds: Number of cross-validation folds
        
    Returns:
        dict: Best tree parameters
    """
    logger.info("Step 1: Tuning tree structure parameters...")
    
    # Define parameter grid for tree structure
    max_depth_values = [3, 4, 5, 6, 7, 8]
    min_child_weight_values = [1, 3, 5, 7]
    
    # Fixed parameters during this tuning step
    base_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'tree_method': 'hist',  # Modern XGBoost parameter
        'device': 'cuda',       # Use GPU acceleration
        'learning_rate': 0.1,
        'n_estimators': 100,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0,
        'reg_alpha': 0,
        'reg_lambda': 1,
        'random_state': XGB_RANDOM_SEED
    }
    
    # Track best parameters and score
    best_params = {}
    best_score = float('inf')
    best_std = float('inf')
    
    # Start nested run for tree parameter tuning
    with safe_mlflow_start_run(run_id=run_id, nested=True, run_name="Step 1: Tree Structure Tuning"):
        safe_mlflow_log_params(base_params)
        safe_mlflow_log_param("tuning_step", "tree_structure")
        
        results = []
        
        # Grid search
        for max_depth in max_depth_values:
            for min_child_weight in min_child_weight_values:
                # Current parameters
                params = base_params.copy()
                params['max_depth'] = max_depth
                params['min_child_weight'] = min_child_weight
                
                # Log this combination
                with safe_mlflow_start_run(nested=True, run_name=f"max_depth={max_depth}, min_child_weight={min_child_weight}"):
                    safe_mlflow_log_param("max_depth", max_depth)
                    safe_mlflow_log_param("min_child_weight", min_child_weight)
                    
                    # Cross-validate
                    mean_rmse, std_rmse, cv_results, duration = cross_validate_xgb(params, X_train, y_train, n_folds)
                    
                    # Log metrics
                    safe_mlflow_log_metric("mean_rmse", mean_rmse)
                    safe_mlflow_log_metric("std_rmse", std_rmse)
                    safe_mlflow_log_metric("training_duration", duration)
                    
                    # Save CV results per fold
                    for fold_result in cv_results:
                        safe_mlflow_log_metric(f"fold_{fold_result['fold']}_rmse", fold_result['rmse'])
                    
                    # Track this combination
                    results.append({
                        "max_depth": max_depth,
                        "min_child_weight": min_child_weight,
                        "mean_rmse": mean_rmse,
                        "std_rmse": std_rmse
                    })
                    
                    # Update best parameters if better score
                    if mean_rmse < best_score:
                        best_score = mean_rmse
                        best_std = std_rmse
                        best_params = {
                            "max_depth": max_depth,
                            "min_child_weight": min_child_weight
                        }
                        logger.info(f"New best: max_depth={max_depth}, min_child_weight={min_child_weight}, RMSE={mean_rmse:.4f}")
        
        # Log the best parameters and performance
        safe_mlflow_log_params(best_params)
        safe_mlflow_log_metric("best_rmse", best_score)
        safe_mlflow_log_metric("best_std", best_std)
        
        # Create and log results visualization
        results_df = pd.DataFrame(results)
        plt.figure(figsize=(12, 8))
        pivot_table = results_df.pivot(index="max_depth", columns="min_child_weight", values="mean_rmse")
        sns_plot = sns.heatmap(pivot_table, annot=True, fmt=".4f", cmap="YlGnBu")
        plt.title('RMSE for max_depth and min_child_weight combinations')
        plt.tight_layout()
        
        # Save and log the heatmap
        heatmap_path = "tree_params_heatmap.png"
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        safe_mlflow_log_artifact(heatmap_path)
        os.remove(heatmap_path) # Clean up
        
        # Log best parameters summary
        with open("best_tree_params.json", 'w') as f:
            json.dump(best_params, f, indent=2)
        safe_mlflow_log_artifact("best_tree_params.json")
        os.remove("best_tree_params.json") # Clean up
        
    return best_params

def tune_regularization_params(X_train, y_train, run_id, tree_params, n_folds=5):
    """Step 2: Tune regularization parameters (gamma, reg_alpha, reg_lambda).
    
    Args:
        X_train: Training features
        y_train: Training target
        run_id: Parent MLflow run ID
        tree_params: Best tree parameters from step 1
        n_folds: Number of cross-validation folds
        
    Returns:
        dict: Best regularization parameters
    """
    logger.info("Step 2: Tuning regularization parameters...")
    
    # Define parameter grid for regularization
    gamma_values = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    reg_alpha_values = [0, 0.001, 0.01, 0.1, 1]
    reg_lambda_values = [0.1, 0.5, 1, 1.5, 2]
    
    # Fixed parameters during this tuning step (including best tree params)
    base_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'tree_method': 'hist',  # Modern XGBoost parameter
        'device': 'cuda',       # Use GPU acceleration
        'learning_rate': 0.1,
        'n_estimators': 100,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'max_depth': tree_params['max_depth'],
        'min_child_weight': tree_params['min_child_weight'],
        'random_state': XGB_RANDOM_SEED
    }
    
    # Track best parameters and score
    best_params = {}
    best_score = float('inf')
    
    # First round: tune gamma
    with safe_mlflow_start_run(run_id=run_id, nested=True, run_name="Step 2.1: Gamma Tuning"):
        safe_mlflow_log_params(base_params)
        gamma_results = []
        
        for gamma in gamma_values:
            params = base_params.copy()
            params['gamma'] = gamma
            
            with safe_mlflow_start_run(nested=True, run_name=f"gamma={gamma}"):
                safe_mlflow_log_param("gamma", gamma)
                
                # Cross-validate
                mean_rmse, std_rmse, cv_results, duration = cross_validate_xgb(params, X_train, y_train, n_folds)
                
                # Log metrics
                mlflow.log_metric("mean_rmse", mean_rmse)
                mlflow.log_metric("std_rmse", std_rmse)
                mlflow.log_metric("training_duration", duration)
                
                gamma_results.append({
                    "gamma": gamma,
                    "mean_rmse": mean_rmse,
                    "std_rmse": std_rmse
                })
                
                if mean_rmse < best_score:
                    best_score = mean_rmse
                    best_gamma = gamma
                    logger.info(f"New best gamma: {gamma}, RMSE={mean_rmse:.4f}")
        
        # Update base parameters with best gamma
        base_params['gamma'] = best_gamma
        safe_mlflow_log_param("best_gamma", best_gamma)
        
        # Plot gamma results
        plt.figure(figsize=(10, 6))
        gamma_df = pd.DataFrame(gamma_results)
        sns.lineplot(x="gamma", y="mean_rmse", data=gamma_df, marker='o')
        plt.title('RMSE vs Gamma Value')
        plt.grid(True)
        plt.savefig("gamma_tuning.png", dpi=300, bbox_inches='tight')
        plt.close()
        safe_mlflow_log_artifact("gamma_tuning.png")
        os.remove("gamma_tuning.png")
    
    # Reset best score for next round
    best_score = float('inf')
    
    # Second round: tune reg_alpha and reg_lambda
    with safe_mlflow_start_run(run_id=run_id, nested=True, run_name="Step 2.2: Alpha/Lambda Tuning"):
        safe_mlflow_log_params(base_params)
        reg_results = []
        
        for reg_alpha in reg_alpha_values:
            for reg_lambda in reg_lambda_values:
                params = base_params.copy()
                params['reg_alpha'] = reg_alpha
                params['reg_lambda'] = reg_lambda
                
                with safe_mlflow_start_run(nested=True, run_name=f"alpha={reg_alpha}, lambda={reg_lambda}"):
                    safe_mlflow_log_param("reg_alpha", reg_alpha)
                    safe_mlflow_log_param("reg_lambda", reg_lambda)
                    
                    # Cross-validate
                    mean_rmse, std_rmse, cv_results, duration = cross_validate_xgb(params, X_train, y_train, n_folds)
                    
                    # Log metrics
                    safe_mlflow_log_metric("mean_rmse", mean_rmse)
                    safe_mlflow_log_metric("std_rmse", std_rmse)
                    safe_mlflow_log_metric("training_duration", duration)
                    
                    reg_results.append({
                        "reg_alpha": reg_alpha,
                        "reg_lambda": reg_lambda,
                        "mean_rmse": mean_rmse,
                        "std_rmse": std_rmse
                    })
                    
                    if mean_rmse < best_score:
                        best_score = mean_rmse
                        best_params = {
                            "gamma": best_gamma,
                            "reg_alpha": reg_alpha,
                            "reg_lambda": reg_lambda
                        }
                        logger.info(f"New best: alpha={reg_alpha}, lambda={reg_lambda}, RMSE={mean_rmse:.4f}")
        
        # Log best parameters
        safe_mlflow_log_params(best_params)
        safe_mlflow_log_metric("best_rmse", best_score)
        
        # Create heatmap for alpha/lambda
        reg_df = pd.DataFrame(reg_results)
        plt.figure(figsize=(12, 8))
        pivot = reg_df.pivot_table(index="reg_alpha", columns="reg_lambda", values="mean_rmse")
        sns.heatmap(pivot, annot=True, fmt=".4f", cmap="YlGnBu")
        plt.title('RMSE for reg_alpha and reg_lambda combinations')
        plt.tight_layout()
        plt.savefig("reg_params_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
        mlflow.log_artifact("reg_params_heatmap.png")
        os.remove("reg_params_heatmap.png")
        
        # Log best parameters summary
        with open("best_reg_params.json", 'w') as f:
            json.dump(best_params, f, indent=2)
        mlflow.log_artifact("best_reg_params.json")
        os.remove("best_reg_params.json")
    
    return best_params

def tune_lr_n_estimators(X_train, y_train, run_id, tree_params, reg_params, n_folds=5):
    """Step 3: Tune learning rate and number of estimators.
    
    Args:
        X_train: Training features
        y_train: Training target
        run_id: Parent MLflow run ID
        tree_params: Best tree parameters from step 1
        reg_params: Best regularization parameters from step 2
        n_folds: Number of cross-validation folds
        
    Returns:
        dict: Best learning rate and n_estimators parameters
    """
    logger.info("Step 3: Tuning learning rate and n_estimators...")
    
    # Define parameter grid for learning rate and estimators
    # Note: Often use an inverse relationship between learning_rate and n_estimators
    learning_rates = [0.01, 0.03, 0.05, 0.07, 0.1, 0.2]
    n_estimators_values = [100, 200, 300, 500, 750, 1000]
    
    # Fixed parameters from previous tuning steps
    base_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'tree_method': 'hist',  # Modern XGBoost parameter
        'device': 'cuda',       # Use GPU acceleration
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'max_depth': tree_params['max_depth'],
        'min_child_weight': tree_params['min_child_weight'],
        'gamma': reg_params['gamma'],
        'reg_alpha': reg_params['reg_alpha'],
        'reg_lambda': reg_params['reg_lambda'],
        'random_state': XGB_RANDOM_SEED
    }
    
    # Track best parameters and score
    best_params = {}
    best_score = float('inf')
    best_std = float('inf')
    
    with safe_mlflow_start_run(run_id=run_id, nested=True, run_name="Step 3: Learning Rate/Estimators Tuning"):
        safe_mlflow_log_params(base_params)
        safe_mlflow_log_param("tuning_step", "learning_rate_n_estimators")
        
        results = []
        
        # Grid search
        for learning_rate in learning_rates:
            for n_estimators in n_estimators_values:
                # Current parameters
                params = base_params.copy()
                params['learning_rate'] = learning_rate
                params['n_estimators'] = n_estimators
                
                # Log this combination
                with safe_mlflow_start_run(nested=True, run_name=f"lr={learning_rate}, n_est={n_estimators}"):
                    safe_mlflow_log_param("learning_rate", learning_rate)
                    safe_mlflow_log_param("n_estimators", n_estimators)
                    
                    # Cross-validate
                    mean_rmse, std_rmse, cv_results, duration = cross_validate_xgb(params, X_train, y_train, n_folds)
                    
                    # Log metrics
                    safe_mlflow_log_metric("mean_rmse", mean_rmse)
                    safe_mlflow_log_metric("std_rmse", std_rmse)
                    safe_mlflow_log_metric("training_duration", duration)
                    
                    # Track this combination
                    results.append({
                        "learning_rate": learning_rate,
                        "n_estimators": n_estimators,
                        "mean_rmse": mean_rmse,
                        "std_rmse": std_rmse,
                        "duration": duration
                    })
                    
                    # Update best parameters if better score
                    if mean_rmse < best_score:
                        best_score = mean_rmse
                        best_std = std_rmse
                        best_params = {
                            "learning_rate": learning_rate,
                            "n_estimators": n_estimators
                        }
                        logger.info(f"New best: lr={learning_rate}, n_est={n_estimators}, RMSE={mean_rmse:.4f}")
        
        # Log the best parameters and performance
        safe_mlflow_log_params(best_params)
        safe_mlflow_log_metric("best_rmse", best_score)
        safe_mlflow_log_metric("best_std", best_std)
        
        # Create and log visualization
        results_df = pd.DataFrame(results)
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        pivot_table = results_df.pivot_table(index="learning_rate", columns="n_estimators", values="mean_rmse")
        sns.heatmap(pivot_table, annot=True, fmt=".4f", cmap="YlGnBu")
        plt.title('RMSE for learning rate and n_estimators combinations')
        plt.tight_layout()
        
        heatmap_path = "lr_estimators_heatmap.png"
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        safe_mlflow_log_artifact(heatmap_path)
        os.remove(heatmap_path)
        
        # Create training time heatmap
        plt.figure(figsize=(12, 8))
        duration_pivot = results_df.pivot("learning_rate", "n_estimators", "duration")
        sns.heatmap(duration_pivot, annot=True, fmt=".1f", cmap="Reds")
        plt.title('Training Duration (seconds) for learning rate and n_estimators combinations')
        plt.tight_layout()
        
        duration_path = "training_duration_heatmap.png"
        plt.savefig(duration_path, dpi=300, bbox_inches='tight')
        plt.close()
        safe_mlflow_log_artifact(duration_path)
        os.remove(duration_path)
        
        # Log best parameters summary
        with open("best_lr_params.json", 'w') as f:
            json.dump(best_params, f, indent=2)
        safe_mlflow_log_artifact("best_lr_params.json")
        os.remove("best_lr_params.json")
        
    return best_params

def train_and_log_model(X_train, y_train, X_test, y_test, tmp_folder=None):
    """Implements the four-step XGBoost hyperparameter tuning process with MLflow tracking.
    
    Steps:
    1. Optimize tree structure parameters (max_depth, min_child_weight)
    2. Optimize regularization parameters (gamma, reg_alpha, reg_lambda)
    3. Optimize learning rate and n_estimators
    4. Train final model with optimized parameters
    """
    # Setup logging to file
    log_file = setup_logging()
    
    # Generate unique ID for this run
    run_uuid = str(uuid.uuid4())[:8]
    logger.info(f"Starting new training run with ID: {run_uuid}")
    logger.info("Implementing four-step XGBoost hyperparameter tuning with MLflow tracking")
    logger.info("Configuring XGBoost for GPU training (tree_method='gpu_hist')")
    
    # Import seaborn for heatmaps if not already imported
    try:
        import seaborn as sns
    except ImportError:
        logger.warning("Seaborn not found. Installing seaborn for visualization...")
        import sys, subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "seaborn"])
        import seaborn as sns
    
    # Start parent MLflow run
    with safe_mlflow_start_run(run_name=f"XGBoost_StepWise_Tuning_{run_uuid}") as parent_run:
        parent_run_id = parent_run.info.run_id
        
        # Log dataset info
        safe_mlflow_log_param("num_train_samples", X_train.shape[0])
        safe_mlflow_log_param("num_test_samples", X_test.shape[0])
        safe_mlflow_log_param("num_features", X_train.shape[1])
        safe_mlflow_log_param("feature_names", list(X_train.columns))
        
        # STEP 1: Tune tree structure parameters
        logger.info("Starting Step 1: Tree structure parameter tuning")
        tree_params = tune_tree_params(X_train, y_train, parent_run_id)
        
        # STEP 2: Tune regularization parameters using best tree params
        logger.info("Starting Step 2: Regularization parameter tuning")
        reg_params = tune_regularization_params(X_train, y_train, parent_run_id, tree_params)
        
        # STEP 3: Tune learning rate and n_estimators
        logger.info("Starting Step 3: Learning rate and n_estimators tuning")
        lr_params = tune_lr_n_estimators(X_train, y_train, parent_run_id, tree_params, reg_params)
        
        # STEP 4: Train final model with all optimized parameters
        logger.info("Starting Step 4: Final model training with optimized parameters")
        
        # Combine all best parameters
        final_params = {
            **tree_params,
            **reg_params,
            **lr_params,
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'tree_method': 'hist',  # Modern XGBoost parameter
            'device': 'cuda',       # Use GPU acceleration
            'subsample': 0.8,  # Could also be tuned but we're using default
            'colsample_bytree': 0.8,  # Could also be tuned but we're using default
            'random_state': XGB_RANDOM_SEED
        }
        
        with safe_mlflow_start_run(run_id=parent_run_id, nested=True, run_name="Step 4: Final Model Training"):
            # Log final parameters
            safe_mlflow_log_params(final_params)
            
            start_time = time.time()
            
            # Train final model with categorical features enabled
            final_model = xgb.XGBRegressor(enable_categorical=True, **final_params)
            final_model.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train), (X_test, y_test)],
                verbose=False
            )
            
            training_time = time.time() - start_time
            mlflow.log_metric("training_time", training_time)
            logger.info(f"Final model training completed in {training_time:.2f} seconds")
            
            # Evaluate model and log metrics
            metrics = evaluate_and_log_metrics(final_model, X_train, y_train, X_test, y_test)
            
            # Log feature importances
            log_feature_importance(final_model, X_train.columns, parent_run_id, "final_model")
            
            # Create and log actual vs predicted plots
            y_train_pred = final_model.predict(X_train)
            y_test_pred = final_model.predict(X_test)
            
            train_plot = create_actual_vs_predicted_plot(
                y_train, y_train_pred, 
                "Training Set: Actual vs Predicted", 
                "train_actual_vs_predicted.png"
            )
            
            test_plot = create_actual_vs_predicted_plot(
                y_test, y_test_pred, 
                "Test Set: Actual vs Predicted", 
                "test_actual_vs_predicted.png"
            )
            
            mlflow.log_artifact(train_plot)
            mlflow.log_artifact(test_plot)
            
            # Clean up plot files
            os.remove(train_plot)
            os.remove(test_plot)
            
            # Log the model with example input
            mlflow.xgboost.log_model(
                final_model, 
                "xgboost_model", 
                input_example=X_train.iloc[0:5]
            )
            
            # Save model locally as well
            models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
            os.makedirs(models_dir, exist_ok=True)
            model_path = os.path.join(models_dir, f"xgboost_model_{run_uuid}.json")
            final_model.save_model(model_path)
            logger.info(f"Model saved locally to: {model_path}")
            
            # Also log the model file as an artifact
            mlflow.log_artifact(model_path)
            
            # Log run summary information
            mlflow.log_param("best_iteration", final_model.best_iteration)
            mlflow.log_metric("best_score", final_model.best_score)
            
            # Log the log file as an artifact
            mlflow.log_artifact(log_file)
            
        logger.info("Step-wise XGBoost hyperparameter tuning completed successfully")
        logger.info(f"Final model test RMSE: {metrics['test_rmse']:.4f}, R²: {metrics['test_r2']:.4f}")
        
        return final_model, metrics
            
def main():
    """Main function to run the step-wise XGBoost model tuning process."""
    # Define data path
    processed_data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'processed')
    
    # Load the preprocessed data
    X_train, y_train, X_test, y_test = load_data(processed_data_path)
    if X_train is None:
        logger.error("Failed to load data. Exiting.")
        return
    
    # Check for GPU availability
    try:
        import cupy
        logger.info("GPU is available for XGBoost training.")
    except ImportError:
        logger.warning("cupy not found - GPU may not be available.")
        logger.warning("Running with 'gpu_hist' might fail if no CUDA environment is set up.")
        logger.warning("If training fails, modify the tree_method parameter to 'hist' instead.")
    
    try:
        # Train and evaluate model using step-wise optimization approach
        model, metrics = train_and_log_model(X_train, y_train, X_test, y_test)
        
        # Print final summary
        logger.info("\n==== Step-wise XGBoost Model Training Complete ====\n")
        logger.info(f"Trained XGBoost model with step-wise optimized hyperparameters")
        logger.info(f"Test RMSE: {metrics['test_rmse']:.4f}")
        logger.info(f"Test R²: {metrics['test_r2']:.4f}")
        logger.info(f"Train RMSE: {metrics['train_rmse']:.4f}")
        logger.info(f"Train R²: {metrics['train_r2']:.4f}")
        logger.info(f"Overfit Check - RMSE Delta: {metrics['rmse_delta']:.4f}")
        logger.info(f"Overfit Check - R² Delta: {metrics['r2_delta']:.4f}")
        
        # Display detailed evaluation metrics table
        logger.info("\n==== Detailed Metrics ====\n")
        logger.info(f"{'Metric':<20} {'Training':<15} {'Test':<15} {'Delta':<15}")
        logger.info(f"{'-'*65}")
        logger.info(f"{'RMSE':<20} {metrics['train_rmse']:<15.4f} {metrics['test_rmse']:<15.4f} {metrics['rmse_delta']:<15.4f}")
        logger.info(f"{'MAE':<20} {metrics['train_mae']:<15.4f} {metrics['test_mae']:<15.4f} {abs(metrics['train_mae'] - metrics['test_mae']):<15.4f}")
        logger.info(f"{'R²':<20} {metrics['train_r2']:<15.4f} {metrics['test_r2']:<15.4f} {metrics['r2_delta']:<15.4f}")
        logger.info(f"{'Expl. Variance':<20} {metrics['train_explained_variance']:<15.4f} {metrics['test_explained_variance']:<15.4f} {abs(metrics['train_explained_variance'] - metrics['test_explained_variance']):<15.4f}")
        
    except Exception as e:
        logger.error(f"Error during model training: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
