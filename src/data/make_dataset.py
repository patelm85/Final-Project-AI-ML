"""
Data cleaning and time-based splitting script for Walmart sales forecasting.

This script:
1. Loads the raw Walmart.csv dataset
2. Cleans the data (handles missing values, converts dates, removes outliers)
3. Splits the data chronologically: 35% train, 35% validate, 30% test
4. Saves the processed datasets to data/processed/
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Tuple

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def load_raw_data(data_path: Path) -> pd.DataFrame:
    """
    Load the raw Walmart dataset.
    
    Args:
        data_path: Path to the raw CSV file
        
    Returns:
        DataFrame containing the raw data
    """
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset: handle dates, missing values, outliers, and normalize text.
    
    Args:
        df: Raw DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    print("\nCleaning data...")
    df = df.copy()
    
    # Convert Date column to datetime
    # Date format appears to be DD-MM-YYYY
    print("Converting Date column to datetime...")
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')
    
    # Check for any failed date conversions
    invalid_dates = df['Date'].isna().sum()
    if invalid_dates > 0:
        print(f"Warning: {invalid_dates} rows have invalid dates and will be dropped")
        df = df.dropna(subset=['Date'])
    
    # Sort by Date in ascending order (chronological)
    print("Sorting data by date in ascending order...")
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Handle missing values
    print("\nChecking for missing values...")
    missing_counts = df.isnull().sum()
    if missing_counts.sum() > 0:
        print("Missing values found:")
        print(missing_counts[missing_counts > 0])
        
        # Handle missing values in numeric columns
        numeric_cols = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                # Use median for imputation (more robust to outliers)
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                print(f"  Imputed {df[col].isnull().sum()} missing values in {col} with median: {median_val:.2f}")
        
        # Handle missing values in categorical columns
        if df['Holiday_Flag'].isnull().sum() > 0:
            df['Holiday_Flag'] = df['Holiday_Flag'].fillna(0).astype(int)
            print(f"  Filled missing Holiday_Flag with 0")
    else:
        print("No missing values found.")
    
    # Normalize text fields (Store might have inconsistent formatting)
    print("\nNormalizing text fields...")
    if 'Store' in df.columns:
        df['Store'] = df['Store'].astype(str).str.strip()
    
    # Remove outliers in Weekly_Sales using IQR method
    print("\nChecking for outliers in Weekly_Sales...")
    Q1 = df['Weekly_Sales'].quantile(0.25)
    Q3 = df['Weekly_Sales'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 3 * IQR  # Use 3*IQR for less aggressive filtering
    upper_bound = Q3 + 3 * IQR
    
    outliers = ((df['Weekly_Sales'] < lower_bound) | (df['Weekly_Sales'] > upper_bound)).sum()
    if outliers > 0:
        print(f"  Found {outliers} outliers (outside [{lower_bound:.2f}, {upper_bound:.2f}])")
        print(f"  Removing outliers...")
        df = df[(df['Weekly_Sales'] >= lower_bound) & (df['Weekly_Sales'] <= upper_bound)].copy()
        print(f"  Remaining rows: {len(df)}")
    else:
        print("  No outliers detected using 3*IQR method")
    
    # Encode categorical variables (Store is already numeric, Holiday_Flag is binary)
    print("\nEncoding categorical variables...")
    # Store: already numeric, but ensure it's int
    df['Store'] = df['Store'].astype(int)
    # Holiday_Flag: already binary, ensure it's int
    df['Holiday_Flag'] = df['Holiday_Flag'].astype(int)
    
    # Ensure numeric columns are properly typed
    numeric_columns = ['Weekly_Sales', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Final check for missing values after conversions
    remaining_missing = df.isnull().sum().sum()
    if remaining_missing > 0:
        print(f"\nWarning: {remaining_missing} missing values remain after cleaning")
        print(df.isnull().sum()[df.isnull().sum() > 0])
        # Drop any remaining rows with missing values
        df = df.dropna()
        print(f"Dropped rows with missing values. Remaining: {len(df)} rows")
    
    print(f"\nCleaning complete. Final dataset shape: {df.shape}")
    return df


def time_based_split(df: pd.DataFrame, train_pct: float = 0.35, 
                     val_pct: float = 0.35, test_pct: float = 0.30) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data chronologically by time order.
    
    Args:
        df: Cleaned DataFrame (already sorted by date)
        train_pct: Percentage of data for training (default: 0.35)
        val_pct: Percentage of data for validation (default: 0.35)
        test_pct: Percentage of data for testing (default: 0.30)
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    assert abs(train_pct + val_pct + test_pct - 1.0) < 1e-6, "Percentages must sum to 1.0"
    
    print(f"\nPerforming time-based split...")
    print(f"  Training: {train_pct*100:.0f}%")
    print(f"  Validation: {val_pct*100:.0f}%")
    print(f"  Test: {test_pct*100:.0f}%")
    
    n_total = len(df)
    n_train = int(n_total * train_pct)
    n_val = int(n_total * val_pct)
    
    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train:n_train + n_val].copy()
    test_df = df.iloc[n_train + n_val:].copy()
    
    print(f"\nSplit complete:")
    print(f"  Training set: {len(train_df)} rows ({len(train_df)/n_total*100:.1f}%)")
    print(f"    Date range: {train_df['Date'].min().date()} to {train_df['Date'].max().date()}")
    print(f"  Validation set: {len(val_df)} rows ({len(val_df)/n_total*100:.1f}%)")
    print(f"    Date range: {val_df['Date'].min().date()} to {val_df['Date'].max().date()}")
    print(f"  Test set: {len(test_df)} rows ({len(test_df)/n_total*100:.1f}%)")
    print(f"    Date range: {test_df['Date'].min().date()} to {test_df['Date'].max().date()}")
    
    return train_df, val_df, test_df


def save_processed_data(train_df: pd.DataFrame, val_df: pd.DataFrame, 
                       test_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Save processed datasets to CSV files.
    
    Args:
        train_df: Training dataset
        val_df: Validation dataset
        test_df: Test dataset
        output_dir: Directory to save processed data
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving processed datasets to {output_dir}...")
    
    train_path = output_dir / 'train.csv'
    val_path = output_dir / 'validate.csv'
    test_path = output_dir / 'test.csv'
    
    train_df.to_csv(train_path, index=False)
    print(f"  Saved training set: {train_path} ({len(train_df)} rows)")
    
    val_df.to_csv(val_path, index=False)
    print(f"  Saved validation set: {val_path} ({len(val_df)} rows)")
    
    test_df.to_csv(test_path, index=False)
    print(f"  Saved test set: {test_path} ({len(test_df)} rows)")
    
    print("\nAll datasets saved successfully!")


def main():
    """Main function to run the data cleaning and splitting pipeline."""
    project_root = Path(__file__).parent.parent.parent
    
    # Define paths
    raw_data_path = project_root / 'data' / 'raw' / 'Walmart.csv'
    processed_data_dir = project_root / 'data' / 'processed'
    
    # Check if raw data exists
    if not raw_data_path.exists():
        print(f"Error: Raw data file not found at {raw_data_path}")
        print("Please ensure Walmart.csv is in data/raw/ directory")
        sys.exit(1)
    
    # Load raw data
    df = load_raw_data(raw_data_path)
    
    # Clean data
    df_cleaned = clean_data(df)
    
    # Perform time-based split
    train_df, val_df, test_df = time_based_split(df_cleaned)
    
    # Save processed data
    save_processed_data(train_df, val_df, test_df, processed_data_dir)
    
    print("\n" + "="*60)
    print("Data processing pipeline completed successfully!")
    print("="*60)


if __name__ == '__main__':
    main()

