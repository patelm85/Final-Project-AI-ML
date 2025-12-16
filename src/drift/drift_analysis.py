"""
Drift Analysis Script for Walmart Sales Forecasting

This script detects data drift and target drift between the reference (test) 
dataset and simulated production data using statistical tests.

Usage:
    python src/drift/drift_analysis.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from scipy import stats

# Add project root to path
project_root = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(project_root))


def load_test_data(data_dir: Path) -> pd.DataFrame:
    """Load the test dataset as reference data."""
    test_path = data_dir / 'test.csv'
    if not test_path.exists():
        raise FileNotFoundError(f"Test dataset not found at {test_path}")
    
    print(f"Loading reference (test) dataset from {test_path}...")
    df = pd.read_csv(test_path)
    
    # Ensure Date is datetime
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    
    print(f"  Loaded {len(df)} rows")
    print(f"  Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    
    return df


def simulate_production_data(df: pd.DataFrame, production_months: int = 3) -> pd.DataFrame:
    """
    Simulate production data by taking the last N months of data.
    
    This simulates a scenario where we're monitoring drift over time.
    In a real scenario, this would be actual production data.
    
    Args:
        df: Full dataset (e.g., test set or full dataset)
        production_months: Number of months to simulate as production data
        
    Returns:
        DataFrame with simulated production data
    """
    print(f"\nSimulating production data (last portion of test data)...")
    
    # Sort by date
    df = df.sort_values('Date').copy()
    
    # Use last portion of data as production
    production_size = max(int(len(df) * 0.3), 100)  # At least 100 rows or 30%
    production_df = df.tail(production_size).copy()
    
    print(f"  Production data: {len(production_df)} rows")
    if 'Date' in production_df.columns:
        print(f"  Date range: {production_df['Date'].min().date()} to {production_df['Date'].max().date()}")
    
    return production_df


def prepare_features_for_drift(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare features for drift analysis.
    Extracts the same features used in model training.
    """
    df = df.copy()
    
    # Extract date features if Date column exists
    if 'Date' in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df['Date']):
            df['Date'] = pd.to_datetime(df['Date'])
        
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Week'] = df['Date'].dt.isocalendar().week
        df['DayOfYear'] = df['Date'].dt.dayofyear
    
    return df


def kolmogorov_smirnov_test(ref_data: pd.Series, prod_data: pd.Series) -> dict:
    """
    Perform Kolmogorov-Smirnov test for distribution drift.
    
    Returns:
        dict with statistic, p-value, and drift detected flag
    """
    try:
        statistic, p_value = stats.ks_2samp(ref_data, prod_data)
        drift_detected = p_value < 0.05  # Significant if p < 0.05
        return {
            'statistic': statistic,
            'p_value': p_value,
            'drift_detected': drift_detected
        }
    except Exception as e:
        return {
            'statistic': np.nan,
            'p_value': np.nan,
            'drift_detected': False,
            'error': str(e)
        }


def mann_whitney_test(ref_data: pd.Series, prod_data: pd.Series) -> dict:
    """
    Perform Mann-Whitney U test (non-parametric test for distribution differences).
    
    Returns:
        dict with statistic, p-value, and drift detected flag
    """
    try:
        statistic, p_value = stats.mannwhitneyu(ref_data, prod_data, alternative='two-sided')
        drift_detected = p_value < 0.05
        return {
            'statistic': statistic,
            'p_value': p_value,
            'drift_detected': drift_detected
        }
    except Exception as e:
        return {
            'statistic': np.nan,
            'p_value': np.nan,
            'drift_detected': False,
            'error': str(e)
        }


def analyze_drift_statistical(ref_df: pd.DataFrame, prod_df: pd.DataFrame, 
                              output_dir: Path) -> dict:
    """
    Analyze drift using statistical tests (KS test, Mann-Whitney U test).
    
    Args:
        ref_df: Reference (baseline) dataset
        prod_df: Production dataset to compare
        output_dir: Directory to save reports
        
    Returns:
        dict with drift analysis results
    """
    print("\n" + "="*70)
    print("Running Statistical Drift Analysis")
    print("="*70)
    
    # Prepare features
    print("\nPreparing features for drift analysis...")
    ref_features = prepare_features_for_drift(ref_df)
    prod_features = prepare_features_for_drift(prod_df)
    
    print(f"\nReference dataset: {len(ref_features)} rows, {len(ref_features.columns)} columns")
    print(f"Production dataset: {len(prod_features)} rows, {len(prod_features.columns)} columns")
    
    # Select numeric columns for analysis
    numeric_cols = ref_features.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove date-derived columns that might cause issues
    exclude_cols = ['Year', 'Month', 'Week', 'DayOfYear']
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    print(f"\nAnalyzing {len(numeric_cols)} numeric features: {numeric_cols}")
    
    # Store results
    drift_results = {
        'features': {},
        'target': None,
        'summary': {
            'total_features': len(numeric_cols),
            'drifted_features': 0,
            'drift_percentage': 0.0
        }
    }
    
    # Separate target if it exists
    target_col = 'Weekly_Sales' if 'Weekly_Sales' in numeric_cols else None
    if target_col:
        feature_cols = [col for col in numeric_cols if col != target_col]
        print(f"\nTarget variable: {target_col}")
        print(f"Feature variables: {len(feature_cols)} features")
    else:
        feature_cols = numeric_cols
        print(f"\nNo target variable found. Analyzing all numeric features.")
    
    # Analyze each feature
    print("\n" + "-"*70)
    print("Feature Drift Analysis (Kolmogorov-Smirnov Test)")
    print("-"*70)
    
    drifted_features = []
    
    for col in feature_cols:
        if col not in ref_features.columns or col not in prod_features.columns:
            continue
        
        ref_data = ref_features[col].dropna()
        prod_data = prod_features[col].dropna()
        
        if len(ref_data) < 10 or len(prod_data) < 10:
            print(f"  {col:20s}: Skipped (insufficient data)")
            continue
        
        # KS test
        ks_result = kolmogorov_smirnov_test(ref_data, prod_data)
        
        # Mann-Whitney test
        mw_result = mann_whitney_test(ref_data, prod_data)
        
        drift_detected = ks_result['drift_detected'] or mw_result['drift_detected']
        
        if drift_detected:
            drifted_features.append(col)
        
        drift_results['features'][col] = {
            'ks_test': ks_result,
            'mann_whitney_test': mw_result,
            'drift_detected': drift_detected,
            'ref_mean': float(ref_data.mean()),
            'ref_std': float(ref_data.std()),
            'prod_mean': float(prod_data.mean()),
            'prod_std': float(prod_data.std())
        }
        
        status = "DRIFT DETECTED" if drift_detected else "No drift"
        print(f"  {col:20s}: {status} | KS p-value: {ks_result['p_value']:.4f} | MW p-value: {mw_result['p_value']:.4f}")
    
    # Analyze target drift if available
    if target_col and target_col in ref_features.columns and target_col in prod_features.columns:
        print("\n" + "-"*70)
        print("Target Drift Analysis (Weekly_Sales)")
        print("-"*70)
        
        ref_target = ref_features[target_col].dropna()
        prod_target = prod_features[target_col].dropna()
        
        if len(ref_target) >= 10 and len(prod_target) >= 10:
            ks_result = kolmogorov_smirnov_test(ref_target, prod_target)
            mw_result = mann_whitney_test(ref_target, prod_target)
            
            drift_detected = ks_result['drift_detected'] or mw_result['drift_detected']
            
            drift_results['target'] = {
                'ks_test': ks_result,
                'mann_whitney_test': mw_result,
                'drift_detected': drift_detected,
                'ref_mean': float(ref_target.mean()),
                'ref_std': float(ref_target.std()),
                'prod_mean': float(prod_target.mean()),
                'prod_std': float(prod_target.std()),
                'ref_median': float(ref_target.median()),
                'prod_median': float(prod_target.median())
            }
            
            status = "DRIFT DETECTED" if drift_detected else "No drift"
            print(f"  {target_col:20s}: {status}")
            print(f"  Reference mean: ${drift_results['target']['ref_mean']:,.2f}")
            print(f"  Production mean: ${drift_results['target']['prod_mean']:,.2f}")
            print(f"  Change: {((drift_results['target']['prod_mean'] / drift_results['target']['ref_mean']) - 1) * 100:.2f}%")
            print(f"  KS p-value: {ks_result['p_value']:.4f}")
            print(f"  MW p-value: {mw_result['p_value']:.4f}")
    
    # Update summary
    drift_results['summary']['drifted_features'] = len(drifted_features)
    drift_results['summary']['drift_percentage'] = (len(drifted_features) / len(feature_cols) * 100) if feature_cols else 0.0
    
    print("\n" + "="*70)
    print("Drift Analysis Summary")
    print("="*70)
    print(f"Total features analyzed: {drift_results['summary']['total_features']}")
    print(f"Features with drift: {drift_results['summary']['drifted_features']}")
    print(f"Drift percentage: {drift_results['summary']['drift_percentage']:.2f}%")
    
    if target_col and drift_results['target']:
        print(f"Target drift detected: {'Yes' if drift_results['target']['drift_detected'] else 'No'}")
    
    # Generate visualization
    print("\nGenerating drift visualization...")
    create_drift_visualization(ref_features, prod_features, drift_results, output_dir)
    
    # Save results to JSON
    import json
    results_path = output_dir / 'drift_results.json'
    with open(results_path, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            return obj
        
        json.dump(convert_to_serializable(drift_results), f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    
    # Generate HTML report
    html_path = output_dir / 'drift_report.html'
    create_html_report(drift_results, html_path, ref_df, prod_df)
    print(f"HTML report saved to: {html_path}")
    
    return drift_results


def create_html_report(drift_results: dict, output_path: Path, 
                      ref_df: pd.DataFrame, prod_df: pd.DataFrame):
    """Create an HTML report for drift analysis."""
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Walmart Sales Forecasting - Drift Analysis Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 5px;
        }}
        .summary-box {{
            background-color: #34495e;
            color: white;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }}
        .summary-box h3 {{
            margin-top: 0;
        }}
        .summary-stats {{
            display: flex;
            justify-content: space-around;
            margin-top: 15px;
        }}
        .stat-item {{
            text-align: center;
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
        }}
        .stat-label {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
        .drift-badge {{
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 0.9em;
        }}
        .drift-yes {{
            background-color: #e74c3c;
            color: white;
        }}
        .drift-no {{
            background-color: #27ae60;
            color: white;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .p-value {{
            font-family: monospace;
        }}
        .p-value.significant {{
            color: #e74c3c;
            font-weight: bold;
        }}
        .p-value.not-significant {{
            color: #27ae60;
        }}
        .target-section {{
            background-color: #ecf0f1;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }}
        .target-metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}
        .metric-card {{
            background-color: white;
            padding: 15px;
            border-radius: 5px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .metric-label {{
            font-size: 0.9em;
            color: #7f8c8d;
            margin-bottom: 5px;
        }}
        .metric-value {{
            font-size: 1.5em;
            font-weight: bold;
            color: #2c3e50;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            text-align: center;
            color: #7f8c8d;
            font-size: 0.9em;
        }}
        img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin: 20px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Walmart Sales Forecasting - Drift Analysis Report</h1>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="summary-box">
            <h3>Executive Summary</h3>
            <div class="summary-stats">
                <div class="stat-item">
                    <div class="stat-value">{drift_results['summary']['total_features']}</div>
                    <div class="stat-label">Features Analyzed</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{drift_results['summary']['drifted_features']}</div>
                    <div class="stat-label">Features with Drift</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{drift_results['summary']['drift_percentage']:.1f}%</div>
                    <div class="stat-label">Drift Percentage</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{'Yes' if (drift_results.get('target') and drift_results['target']['drift_detected']) else 'No'}</div>
                    <div class="stat-label">Target Drift</div>
                </div>
            </div>
        </div>
        
        <h2>Target Variable Analysis (Weekly_Sales)</h2>
"""
    
    if drift_results.get('target'):
        target = drift_results['target']
        drift_status = 'drift-yes' if target['drift_detected'] else 'drift-no'
        drift_text = 'DRIFT DETECTED' if target['drift_detected'] else 'NO DRIFT'
        
        change_pct = ((target['prod_mean'] / target['ref_mean']) - 1) * 100
        
        html_content += f"""
        <div class="target-section">
            <p><span class="drift-badge {drift_status}">{drift_text}</span></p>
            <div class="target-metrics">
                <div class="metric-card">
                    <div class="metric-label">Reference Mean</div>
                    <div class="metric-value">${target['ref_mean']:,.2f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Production Mean</div>
                    <div class="metric-value">${target['prod_mean']:,.2f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Change</div>
                    <div class="metric-value">{change_pct:+.2f}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">KS Test p-value</div>
                    <div class="metric-value">{target['ks_test']['p_value']:.4f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">MW Test p-value</div>
                    <div class="metric-value">{target['mann_whitney_test']['p_value']:.4f}</div>
                </div>
            </div>
        </div>
"""
    
    html_content += """
        <h2>Feature Drift Analysis</h2>
        <table>
            <thead>
                <tr>
                    <th>Feature</th>
                    <th>Drift Status</th>
                    <th>KS Test p-value</th>
                    <th>MW Test p-value</th>
                    <th>Ref Mean</th>
                    <th>Prod Mean</th>
                    <th>Change</th>
                </tr>
            </thead>
            <tbody>
"""
    
    for feature_name, feature_data in drift_results['features'].items():
        drift_status = 'drift-yes' if feature_data['drift_detected'] else 'drift-no'
        drift_text = 'DRIFT' if feature_data['drift_detected'] else 'OK'
        
        ks_p = feature_data['ks_test']['p_value']
        mw_p = feature_data['mann_whitney_test']['p_value']
        ks_class = 'significant' if ks_p < 0.05 else 'not-significant'
        mw_class = 'significant' if mw_p < 0.05 else 'not-significant'
        
        ref_mean = feature_data['ref_mean']
        prod_mean = feature_data['prod_mean']
        change_pct = ((prod_mean / ref_mean) - 1) * 100 if ref_mean != 0 else 0
        
        html_content += f"""
                <tr>
                    <td><strong>{feature_name}</strong></td>
                    <td><span class="drift-badge {drift_status}">{drift_text}</span></td>
                    <td><span class="p-value {ks_class}">{ks_p:.4f}</span></td>
                    <td><span class="p-value {mw_class}">{mw_p:.4f}</span></td>
                    <td>{ref_mean:.2f}</td>
                    <td>{prod_mean:.2f}</td>
                    <td>{change_pct:+.2f}%</td>
                </tr>
"""
    
    html_content += """
            </tbody>
        </table>
        
        <h2>Visualization</h2>
        <p>The following visualization shows the distribution comparison between reference and production data:</p>
        <img src="drift_analysis.png" alt="Drift Analysis Visualization">
        
        <h2>Interpretation</h2>
        <ul>
            <li><strong>KS Test (Kolmogorov-Smirnov):</strong> Tests if two samples come from the same distribution. p-value &lt; 0.05 indicates significant drift.</li>
            <li><strong>MW Test (Mann-Whitney U):</strong> Non-parametric test for distribution differences. p-value &lt; 0.05 indicates significant drift.</li>
            <li><strong>Drift Detected:</strong> If either test shows p-value &lt; 0.05, drift is considered detected.</li>
            <li><strong>Target Drift:</strong> Changes in the target variable (Weekly_Sales) distribution indicate potential model performance degradation.</li>
        </ul>
        
        <div class="footer">
            <p>Generated by Walmart Sales Forecasting MLOps Pipeline</p>
            <p>Statistical tests: Kolmogorov-Smirnov and Mann-Whitney U</p>
        </div>
    </div>
</body>
</html>
"""
    
    with open(output_path, 'w') as f:
        f.write(html_content)


def create_drift_visualization(ref_df: pd.DataFrame, prod_df: pd.DataFrame, 
                               drift_results: dict, output_dir: Path):
    """Create visualization plots for drift analysis."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Select features to plot (top drifted features + target)
    feature_cols = [col for col in drift_results['features'].keys() 
                    if drift_results['features'][col]['drift_detected']]
    
    # Limit to top 6 features for readability
    if len(feature_cols) > 6:
        # Sort by KS statistic (higher = more drift)
        feature_cols = sorted(feature_cols, 
                             key=lambda x: drift_results['features'][x]['ks_test']['statistic'],
                             reverse=True)[:6]
    
    # Add target if available
    target_col = 'Weekly_Sales' if 'Weekly_Sales' in ref_df.columns else None
    if target_col:
        feature_cols.insert(0, target_col)
    
    if not feature_cols:
        print("  No features to visualize")
        return
    
    # Create subplots
    n_features = len(feature_cols)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for idx, col in enumerate(feature_cols):
        ax = axes[idx]
        
        if col not in ref_df.columns or col not in prod_df.columns:
            continue
        
        ref_data = ref_df[col].dropna()
        prod_data = prod_df[col].dropna()
        
        # Plot histograms
        ax.hist(ref_data, bins=30, alpha=0.6, label='Reference', color='blue', density=True)
        ax.hist(prod_data, bins=30, alpha=0.6, label='Production', color='red', density=True)
        
        # Add drift indicator
        if col in drift_results['features']:
            drift_status = "DRIFT" if drift_results['features'][col]['drift_detected'] else "OK"
            ax.set_title(f"{col}\n({drift_status})", fontweight='bold')
        elif col == target_col and drift_results['target']:
            drift_status = "DRIFT" if drift_results['target']['drift_detected'] else "OK"
            ax.set_title(f"{col} (Target)\n({drift_status})", fontweight='bold')
        else:
            ax.set_title(col)
        
        ax.set_xlabel(col)
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / 'drift_analysis.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"  Visualization saved to: {plot_path}")
    plt.close()


def main():
    """Main function to run drift analysis."""
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / 'data' / 'processed'
    output_dir = project_root / 'reports'
    
    print("="*70)
    print("Walmart Sales Forecasting - Drift Analysis")
    print("="*70)
    
    # Load test data as reference
    reference_df = load_test_data(data_dir)
    
    # Simulate production data (last portion of test data)
    production_df = simulate_production_data(reference_df, production_months=3)
    
    # Run drift analysis
    drift_results = analyze_drift_statistical(reference_df, production_df, output_dir)
    
    print(f"\n{'='*70}")
    print("Drift analysis pipeline completed successfully!")
    print(f"{'='*70}\n")
    
    print(f"\nResults saved to:")
    print(f"  - {output_dir / 'drift_results.json'}")
    print(f"  - {output_dir / 'drift_analysis.png'}")
    print(f"  - {output_dir / 'drift_report.html'}")
    print(f"\nTo view the HTML report:")
    print(f"  open {output_dir / 'drift_report.html'}")
    print(f"  # or")
    print(f"  xdg-open {output_dir / 'drift_report.html'}  # Linux\n")


if __name__ == '__main__':
    main()
