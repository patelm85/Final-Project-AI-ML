"""
LightGBM Model Training Script

This script trains a LightGBM model for Walmart sales forecasting and logs
all metrics, parameters, and artifacts to MLflow.

Usage:
    python src/models/train_lightgbm.py
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.lightgbm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def load_datasets(data_dir: Path):
    """Load train, validation, and test datasets."""
    print("Loading datasets...")
    train_df = pd.read_csv(data_dir / 'train.csv')
    val_df = pd.read_csv(data_dir / 'validate.csv')
    test_df = pd.read_csv(data_dir / 'test.csv')
    
    print(f"  Training: {len(train_df)} rows")
    print(f"  Validation: {len(val_df)} rows")
    print(f"  Test: {len(test_df)} rows")
    
    return train_df, val_df, test_df


def prepare_features(df: pd.DataFrame, target_col: str = 'Weekly_Sales'):
    """Prepare features and target for model training."""
    # Convert Date to datetime if it's a string
    if 'Date' in df.columns:
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        # Extract date features
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Week'] = df['Date'].dt.isocalendar().week
        df['DayOfYear'] = df['Date'].dt.dayofyear
        df = df.drop('Date', axis=1)
    
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    return X, y


def train_lightgbm_model(X_train, y_train, X_val, y_val):
    """Train LightGBM model with hyperparameters."""
    print("\nTraining LightGBM model...")
    
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.1,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 20,
        'lambda_l1': 0.1,
        'lambda_l2': 1.0,
        'random_state': 42,
        'verbose': -1
    }
    
    print("Hyperparameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        num_boost_round=200,
        callbacks=[lgb.early_stopping(stopping_rounds=10), lgb.log_evaluation(0)]
    )
    
    print("Training completed!")
    return model, params


def evaluate_model(model, X, y, dataset_name: str):
    """Evaluate model and return metrics."""
    predictions = model.predict(X, num_iteration=model.best_iteration)
    
    rmse = np.sqrt(mean_squared_error(y, predictions))
    mae = mean_absolute_error(y, predictions)
    r2 = r2_score(y, predictions)
    
    print(f"\n{dataset_name} Set Metrics:")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAE: {mae:.2f}")
    print(f"  R²: {r2:.4f}")
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'predictions': predictions
    }


def plot_feature_importance(model, feature_names, output_path: Path):
    """Create and save feature importance plot."""
    importance = model.feature_importance(importance_type='gain')
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df, x='importance', y='feature')
    plt.title('LightGBM Feature Importance')
    plt.xlabel('Importance (Gain)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nFeature importance plot saved: {output_path}")


def plot_predictions_vs_actual(y_true, y_pred, dataset_name: str, output_path: Path):
    """Create predictions vs actual values scatter plot."""
    plt.figure(figsize=(10, 8))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual Weekly Sales')
    plt.ylabel('Predicted Weekly Sales')
    plt.title(f'Predictions vs Actual - {dataset_name} Set')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Predictions plot saved: {output_path}")


def main():
    """Main training function."""
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / 'data' / 'processed'
    
    # Check for AWS credentials (required for S3 artifact storage)
    aws_access_key = os.environ.get('AWS_ACCESS_KEY_ID')
    aws_secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
    if not aws_access_key or not aws_secret_key:
        print("Warning: AWS credentials not found in environment.")
        print("Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")
        print("You can load them from .env file: source .env")
    
    # MLflow configuration
    # Set tracking URI first, before setting experiment
    tracking_uri = os.environ.get('MLFLOW_TRACKING_URI')
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        print(f"MLflow Tracking URI: {tracking_uri}")
    else:
        print("Warning: MLFLOW_TRACKING_URI not set. Using default local tracking.")
        print("Please set: export MLFLOW_TRACKING_URI='http://56.228.2.60:5000'")
    
    experiment_name = "walmart-sales-forecast"
    mlflow.set_experiment(experiment_name)
    
    # Load datasets
    train_df, val_df, test_df = load_datasets(data_dir)
    
    # Prepare features
    X_train, y_train = prepare_features(train_df)
    X_val, y_val = prepare_features(val_df)
    X_test, y_test = prepare_features(test_df)
    
    print(f"\nFeatures: {list(X_train.columns)}")
    print(f"Number of features: {len(X_train.columns)}")
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"lightgbm-{datetime.now().strftime('%Y%m%d-%H%M%S')}"):
        # Train model
        model, params = train_lightgbm_model(X_train, y_train, X_val, y_val)
        
        # Evaluate on validation set
        val_metrics = evaluate_model(model, X_val, y_val, "Validation")
        
        # Evaluate on test set
        test_metrics = evaluate_model(model, X_test, y_test, "Test")
        
        # Log parameters
        mlflow.log_params(params)
        mlflow.log_param("model_type", "LightGBM")
        mlflow.log_param("best_iteration", model.best_iteration)
        mlflow.log_param("num_features", len(X_train.columns))
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("val_samples", len(X_val))
        mlflow.log_param("test_samples", len(X_test))
        
        # Log metrics
        mlflow.log_metric("val_rmse", val_metrics['rmse'])
        mlflow.log_metric("val_mae", val_metrics['mae'])
        mlflow.log_metric("val_r2", val_metrics['r2'])
        mlflow.log_metric("test_rmse", test_metrics['rmse'])
        mlflow.log_metric("test_mae", test_metrics['mae'])
        mlflow.log_metric("test_r2", test_metrics['r2'])
        
        # Create artifacts directory
        artifacts_dir = project_root / 'artifacts' / 'lightgbm'
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate and save plots
        plot_feature_importance(model, X_train.columns, artifacts_dir / 'feature_importance.png')
        plot_predictions_vs_actual(
            y_val, val_metrics['predictions'], 
            "Validation", 
            artifacts_dir / 'val_predictions.png'
        )
        plot_predictions_vs_actual(
            y_test, test_metrics['predictions'], 
            "Test", 
            artifacts_dir / 'test_predictions.png'
        )
        
        # Log artifacts (with error handling for S3 upload issues)
        try:
            mlflow.log_artifacts(str(artifacts_dir))
            print("\nArtifacts successfully uploaded to S3")
        except Exception as e:
            print(f"\nWarning: Failed to upload artifacts to S3: {str(e)}")
            print("Artifacts are saved locally in:", artifacts_dir)
            print("Metrics and model have been logged to MLflow successfully.")
            print("To fix S3 upload:")
            print("  1. Verify AWS credentials in .env file")
            print("  2. Check if access key is valid: aws s3 ls s3://mlflow-artifacts-patel/")
            print("  3. Ensure IAM user has S3 write permissions")
        
        # Log model
        mlflow.lightgbm.log_model(model, "model")
        
        # Register model
        try:
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
            mlflow.register_model(model_uri, "LightGBM_Sales_Forecast")
            print(f"Model registered in MLflow Model Registry as: LightGBM_Sales_Forecast")
        except Exception as e:
            print(f"\nWarning: Failed to register model: {str(e)}")
            print("Model has been logged successfully. You can register it manually in MLflow UI.")
        
        print(f"\n{'='*70}")
        print("Model training and logging completed!")
        print(f"{'='*70}")
        print(f"\nMLflow Run ID: {mlflow.active_run().info.run_id}")
        print(f"Experiment: {experiment_name}")
        print(f"View run in MLflow UI: {tracking_uri}/#/experiments/{mlflow.active_run().info.experiment_id}/runs/{mlflow.active_run().info.run_id}")
        print(f"{'='*70}\n")


if __name__ == '__main__':
    main()

