"""
H2O AutoML Analysis Script

This script runs H2O AutoML on the training dataset to identify the top-performing
model types. The leaderboard is saved and the top 3 model types are displayed.

Usage:
    python src/automl/run_h2o_automl.py
"""

import sys
from pathlib import Path
import pandas as pd
import h2o
from h2o.automl import H2OAutoML

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def load_training_data(data_path: Path) -> pd.DataFrame:
    """
    Load the training dataset.
    
    Args:
        data_path: Path to the training CSV file
        
    Returns:
        DataFrame containing the training data
    """
    print(f"Loading training data from {data_path}...")
    if not data_path.exists():
        raise FileNotFoundError(
            f"Training data not found at {data_path}\n"
            "Please run 'python src/data/make_dataset.py' first to generate the processed datasets."
        )
    
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    return df


def prepare_h2o_frame(df: pd.DataFrame) -> h2o.H2OFrame:
    """
    Convert pandas DataFrame to H2O Frame and prepare for AutoML.
    
    Args:
        df: Pandas DataFrame
        
    Returns:
        H2O Frame ready for AutoML
    """
    print("\nPreparing data for H2O AutoML...")
    
    # Initialize H2O
    h2o.init(nthreads=-1, max_mem_size="4G")
    
    # Convert to H2O frame
    hf = h2o.H2OFrame(df)
    
    # Identify target and features
    target = "Weekly_Sales"
    feature_cols = [col for col in hf.columns if col != target]
    
    print(f"Target variable: {target}")
    print(f"Feature columns: {len(feature_cols)}")
    print(f"Features: {', '.join(feature_cols)}")
    
    return hf, target, feature_cols


def run_automl(hf: h2o.H2OFrame, target: str, max_models: int = 20, 
               max_runtime_secs: int = 3600, seed: int = 42) -> H2OAutoML:
    """
    Run H2O AutoML on the training dataset.
    
    Args:
        hf: H2O Frame with training data
        target: Target variable name
        max_models: Maximum number of models to train
        max_runtime_secs: Maximum runtime in seconds
        seed: Random seed for reproducibility
        
    Returns:
        Trained H2O AutoML object
    """
    print(f"\nStarting H2O AutoML...")
    print(f"  Max models: {max_models}")
    print(f"  Max runtime: {max_runtime_secs} seconds")
    print(f"  Random seed: {seed}")
    print(f"  Sorting metric: RMSE")
    
    # Run AutoML
    aml = H2OAutoML(
        max_models=max_models,
        max_runtime_secs=max_runtime_secs,
        seed=seed,
        sort_metric="RMSE",
        stopping_metric="RMSE",
        stopping_tolerance=0.001,
        stopping_rounds=3,
        balance_classes=False,
        nfolds=5,
        keep_cross_validation_predictions=False,
        verbosity="info"
    )
    
    aml.train(x=[col for col in hf.columns if col != target], 
              y=target, 
              training_frame=hf)
    
    print("\nH2O AutoML training completed!")
    return aml


def save_leaderboard(aml: H2OAutoML, output_path: Path) -> None:
    """
    Save the AutoML leaderboard to CSV.
    
    Args:
        aml: Trained H2O AutoML object
        output_path: Path to save the leaderboard CSV
    """
    print(f"\nSaving leaderboard to {output_path}...")
    
    # Get leaderboard
    leaderboard = aml.leaderboard
    
    # Convert to pandas DataFrame
    lb_df = leaderboard.as_data_frame()
    
    # Save to CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lb_df.to_csv(output_path, index=False)
    
    print(f"Leaderboard saved with {len(lb_df)} models")


def display_top_models(aml: H2OAutoML, top_n: int = 3) -> list:
    """
    Display the top N models from the leaderboard.
    
    Args:
        aml: Trained H2O AutoML object
        top_n: Number of top models to display
        
    Returns:
        List of top model types
    """
    print(f"\n{'='*70}")
    print(f"TOP {top_n} MODELS FROM H2O AUTOML LEADERBOARD")
    print(f"{'='*70}")
    
    # Get leaderboard
    leaderboard = aml.leaderboard
    lb_df = leaderboard.as_data_frame()
    
    # Display top models
    print("\nLeaderboard:")
    print(lb_df.head(top_n).to_string(index=False))
    
    # Extract model types from the top N models
    top_models = lb_df.head(top_n)
    model_types = []
    
    for idx, row in top_models.iterrows():
        model_id = row['model_id']
        rmse = row['rmse']
        mae = row['mae']
        
        # Extract model type from model_id (e.g., "GBM_1_AutoML_1" -> "GBM")
        model_type = model_id.split('_')[0]
        if model_type == "StackedEnsemble":
            # For ensemble models, try to get more info
            if "BestOfFamily" in model_id:
                model_type = "StackedEnsemble_BestOfFamily"
            else:
                model_type = "StackedEnsemble_AllModels"
        
        model_types.append(model_type)
        
        print(f"\n  Rank {idx + 1}:")
        print(f"    Model ID: {model_id}")
        print(f"    Model Type: {model_type}")
        print(f"    RMSE: {rmse:.2f}")
        print(f"    MAE: {mae:.2f}")
    
    print(f"\n{'='*70}")
    print(f"TOP {top_n} MODEL TYPES (for manual training):")
    for i, model_type in enumerate(model_types, 1):
        print(f"  {i}. {model_type}")
    print(f"{'='*70}\n")
    
    return model_types


def main():
    """Main function to run H2O AutoML analysis."""
    project_root = Path(__file__).parent.parent.parent
    
    # Define paths
    train_data_path = project_root / 'data' / 'processed' / 'train.csv'
    leaderboard_path = project_root / 'leaderboard.csv'
    
    try:
        # Load training data
        train_df = load_training_data(train_data_path)
        
        # Prepare H2O frame
        hf, target, feature_cols = prepare_h2o_frame(train_df)
        
        # Run AutoML
        aml = run_automl(hf, target, max_models=20, seed=42)
        
        # Save leaderboard
        save_leaderboard(aml, leaderboard_path)
        
        # Display top models
        top_model_types = display_top_models(aml, top_n=3)
        
        print("\n" + "="*70)
        print("H2O AutoML Analysis Complete!")
        print("="*70)
        print(f"\nNext steps:")
        print(f"1. Review the leaderboard: {leaderboard_path}")
        print(f"2. Based on the top 3 model types above, create training scripts")
        print(f"3. Common model types: XGBoost, GBM, DRF (Random Forest), StackedEnsemble")
        print("="*70 + "\n")
        
        # Shutdown H2O
        h2o.cluster().shutdown()
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

