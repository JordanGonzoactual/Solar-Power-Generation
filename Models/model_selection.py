import os
import pandas as pd
import joblib
import autosklearn.regression
import autosklearn.metrics
import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define MLflow experiment name
MLFLOW_EXPERIMENT_NAME = "Solar_Power_AutoSklearn_Regression"
try:
    mlflow.set_tracking_uri("http://127.0.0.1:8080")
    logger.info(f"MLflow tracking URI set to: http://127.0.0.1:8080")
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    logger.info(f"MLflow experiment set to: {MLFLOW_EXPERIMENT_NAME}")
except Exception as e:
    logger.error(f"Error setting MLflow experiment: {e}")

def load_data(data_path):
    """Loads preprocessed data."""
    logger.info(f"Loading data from {data_path}")
    try:
        X_train_path = os.path.join(data_path, 'X_train.pkl')
        y_train_path = os.path.join(data_path, 'y_train.pkl')
        X_test_path = os.path.join(data_path, 'X_test.pkl')
        y_test_path = os.path.join(data_path, 'y_test.pkl')

        logger.info(f"Attempting to load X_train from: {X_train_path}")
        X_train = joblib.load(X_train_path)
        logger.info(f"Attempting to load y_train from: {y_train_path}")
        y_train = joblib.load(y_train_path)
        logger.info(f"Attempting to load X_test from: {X_test_path}")
        X_test = joblib.load(X_test_path)
        logger.info(f"Attempting to load y_test from: {y_test_path}")
        y_test = joblib.load(y_test_path)
        logger.info("Raw data loaded successfully.")

        # Ensure y_train and y_test are 1D arrays
        if isinstance(y_train, pd.DataFrame) or isinstance(y_train, pd.Series):
            y_train = y_train.values
        if isinstance(y_test, pd.DataFrame) or isinstance(y_test, pd.Series):
            y_test = y_test.values
            
        if y_train.ndim > 1 and y_train.shape[1] == 1:
            y_train = y_train.ravel()
        if y_test.ndim > 1 and y_test.shape[1] == 1:
            y_test = y_test.ravel()
        
        logger.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        logger.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
        logger.info("Data types: X_train: %s, y_train: %s, X_test: %s, y_test: %s", type(X_train), type(y_train), type(X_test), type(y_test))

        # Check for DataFrame type for X data
        if not isinstance(X_train, pd.DataFrame):
            logger.warning(f"X_train is not a pandas DataFrame (type: {type(X_train)}). Converting...")
            X_train = pd.DataFrame(X_train)
        if not isinstance(X_test, pd.DataFrame):
            logger.warning(f"X_test is not a pandas DataFrame (type: {type(X_test)}). Converting...")
            X_test = pd.DataFrame(X_test)

        return X_train, y_train, X_test, y_test
    except FileNotFoundError as e:
        logger.error(f"Error loading data file: {e}. Please ensure make_features.py has been run and files are in the correct location.")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during data loading: {e}", exc_info=True)
        raise

def train_and_log_model(X_train, y_train, X_test, y_test):
    """Trains an AutoSklearn model and logs with MLflow."""
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info(f"Started MLflow Run: {run_id} for experiment {MLFLOW_EXPERIMENT_NAME}")
        mlflow.log_param("ml_framework", "auto-sklearn")

        time_for_task_seconds = 600  # Reduced for quicker testing, e.g., 5 minutes
        per_run_time_limit_seconds = 180 # e.g., 1 minute per model
        automl_memory_limit_mb = 4096 # 4GB

        logger.info(f"Initializing AutoSklearnRegressor with time_left_for_this_task={time_for_task_seconds}s, "
                    f"per_run_time_limit={per_run_time_limit_seconds}s, memory_limit={automl_memory_limit_mb}MB")
        
        automl = autosklearn.regression.AutoSklearnRegressor(
            time_left_for_this_task=time_for_task_seconds,
            per_run_time_limit=per_run_time_limit_seconds,
            memory_limit=automl_memory_limit_mb,
            metric=autosklearn.metrics.r2, # Optimize for R2 score
            # n_jobs=1, # Consider setting n_jobs=1 if memory is an issue or for debugging
            # resampling_strategy='cv',
            # resampling_strategy_arguments={'folds': 3} # Smaller folds for quicker run
        )

        mlflow.log_param("time_left_for_this_task", time_for_task_seconds)
        mlflow.log_param("per_run_time_limit", per_run_time_limit_seconds)
        mlflow.log_param("memory_limit_mb", automl_memory_limit_mb)
        mlflow.log_param("metric", "r2")

        logger.info("Fitting AutoSklearnRegressor model...")
        automl.fit(X_train.copy(), y_train.copy(), dataset_name="solar_power_generation")
        logger.info("Model fitting complete.")

        logger.info("Evaluating model performance...")
        y_pred_train = automl.predict(X_train)
        y_pred_test = automl.predict(X_test)

        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)

        logger.info(f"Train R2: {train_r2:.4f}, Test R2: {test_r2:.4f}")
        logger.info(f"Train MSE: {train_mse:.4f}, Test MSE: {test_mse:.4f}")
        logger.info(f"Train MAE: {train_mae:.4f}, Test MAE: {test_mae:.4f}")

        mlflow.log_metric("train_r2", train_r2)
        mlflow.log_metric("test_r2", test_r2)
        mlflow.log_metric("train_mse", train_mse)
        mlflow.log_metric("test_mse", test_mse)
        mlflow.log_metric("train_mae", train_mae)
        mlflow.log_metric("test_mae", test_mae)

        logger.info("Logging auto-sklearn model to MLflow...")
        mlflow.sklearn.log_model(automl, "autosklearn-model")
        logger.info("Model logged.")

        leaderboard = automl.leaderboard(detailed=True, ensemble_only=False)
        if leaderboard is not None and not leaderboard.empty:
            logger.info("AutoSklearn Leaderboard:\n" + leaderboard.to_string())
            leaderboard_path = "leaderboard.csv"
            leaderboard.to_csv(leaderboard_path, index=False)
            mlflow.log_artifact(leaderboard_path, "leaderboard")
            try:
                os.remove(leaderboard_path)
            except OSError as e:
                logger.warning(f"Could not remove temporary leaderboard file {leaderboard_path}: {e}")
            logger.info("Leaderboard logged as artifact.")
        else:
            logger.warning("Leaderboard is empty or None.")

        sprint_statistics_str = str(automl.sprint_statistics())
        logger.info("AutoSklearn Sprint Statistics:\n" + sprint_statistics_str)
        sprint_stats_path = "sprint_statistics.txt"
        with open(sprint_stats_path, "w") as f:
            f.write(sprint_statistics_str)
        mlflow.log_artifact(sprint_stats_path, "sprint_statistics")
        try:
            os.remove(sprint_stats_path)
        except OSError as e:
            logger.warning(f"Could not remove temporary sprint_statistics file {sprint_stats_path}: {e}")
        logger.info("Sprint statistics logged as artifact.")

        logger.info(f"MLflow Run {run_id} finished. View at http://localhost:5000 or your MLflow tracking server.")
        return run_id

def main():
    try:
        # Determine base directory: Solar-Power-generation/
        # __file__ is Models/model_selection.py
        # os.path.dirname(__file__) is Models/
        # os.path.dirname(os.path.dirname(__file__)) is Solar-Power-generation/
        current_script_path = os.path.abspath(__file__)
        models_dir = os.path.dirname(current_script_path)
        base_dir = os.path.dirname(models_dir)
        data_path = os.path.join(base_dir, 'data', 'processed')
        
        logger.info(f"Script path: {current_script_path}")
        logger.info(f"Base project directory determined as: {base_dir}")
        logger.info(f"Data path set to: {data_path}")

        if not os.path.exists(data_path):
            logger.error(f"Data path does not exist: {data_path}. Please check the path or run feature engineering.")
            return

        X_train, y_train, X_test, y_test = load_data(data_path)
        
        if X_train is None or y_train is None or X_test is None or y_test is None:
            logger.error("Data loading failed and returned None for one or more datasets. Exiting.")
            return
        
        # Check for empty data after loading (X should be DataFrame, y should be numpy array)
        if X_train.empty or X_test.empty or len(y_train) == 0 or len(y_test) == 0:
             logger.warning("One or more datasets are empty after loading. This will likely cause errors.")
             # Optionally, exit if data is critical and empty
             # return 

        train_and_log_model(X_train, y_train, X_test, y_test)
    except FileNotFoundError:
        logger.error("Halting execution due to missing data files. Ensure 'make_features.py' has run successfully and data is in the expected 'data/processed' directory.")
    except Exception as e:
        logger.error(f"An error occurred in the main execution block: {e}", exc_info=True)

if __name__ == "__main__":
    main()
