"""
Clean and fix peru_rainfall.csv to remove NaN/Inf/zero-variance issues
This script should be run BEFORE training if debug_data_quality.py finds issues
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

def clean_data(input_path, output_path, verbose=True):
    """
    Clean ERA5 processed data to remove issues causing NaN in training
    
    Fixes:
    1. Fill NaN values with forward fill + backward fill
    2. Replace Inf with max/min finite values
    3. Remove zero-variance features
    4. Clip extreme outliers (beyond 5 sigma)
    5. Ensure all features have reasonable scale
    
    Args:
        input_path: Path to peru_rainfall.csv
        output_path: Path to save cleaned data
        verbose: Print detailed info
    """
    
    if verbose:
        print("=" * 80)
        print("🧹 CLEANING PERU RAINFALL DATA")
        print("=" * 80)
    
    # Load data
    df = pd.read_csv(input_path)
    original_shape = df.shape
    
    if verbose:
        print(f"\n📊 Original data: {original_shape}")
    
    # Identify feature columns
    protected_cols = ['rain_24h', 'timestamp', 'region']
    feature_cols = [col for col in df.columns if col not in protected_cols]
    
    # 1. Handle NaN values
    nan_before = df[feature_cols].isnull().sum().sum()
    if nan_before > 0:
        if verbose:
            print(f"\n1️⃣ Fixing {nan_before} NaN values...")
        
        # Forward fill then backward fill
        df[feature_cols] = df[feature_cols].fillna(method='ffill').fillna(method='bfill')
        
        # If still NaN (all values in column are NaN), fill with 0
        df[feature_cols] = df[feature_cols].fillna(0)
        
        nan_after = df[feature_cols].isnull().sum().sum()
        if verbose:
            print(f"   ✅ Reduced to {nan_after} NaN values")
    else:
        if verbose:
            print("\n1️⃣ ✅ No NaN values")
    
    # 2. Handle Inf values
    inf_counts = {}
    for col in feature_cols:
        inf_mask = np.isinf(df[col])
        if inf_mask.sum() > 0:
            inf_counts[col] = inf_mask.sum()
            
            # Replace +Inf with 99th percentile
            # Replace -Inf with 1st percentile
            finite_values = df.loc[~inf_mask, col]
            if len(finite_values) > 0:
                p99 = finite_values.quantile(0.99)
                p01 = finite_values.quantile(0.01)
                
                df.loc[df[col] == np.inf, col] = p99
                df.loc[df[col] == -np.inf, col] = p01
    
    if len(inf_counts) > 0:
        if verbose:
            print(f"\n2️⃣ Fixed Inf values in {len(inf_counts)} columns:")
            for col, count in list(inf_counts.items())[:5]:
                print(f"      {col}: {count}")
    else:
        if verbose:
            print("\n2️⃣ ✅ No Inf values")
    
    # 3. Remove zero-variance features
    zero_var_cols = []
    for col in feature_cols:
        if df[col].std() < 1e-8:  # Essentially zero
            zero_var_cols.append(col)
    
    if len(zero_var_cols) > 0:
        if verbose:
            print(f"\n3️⃣ Removing {len(zero_var_cols)} zero-variance features:")
            for col in zero_var_cols[:5]:
                print(f"      {col} (value={df[col].iloc[0]:.6f})")
        
        df = df.drop(columns=zero_var_cols)
        feature_cols = [col for col in df.columns if col not in protected_cols]
    else:
        if verbose:
            print("\n3️⃣ ✅ All features have variance")
    
    # 4. Clip extreme outliers (beyond 5 sigma)
    clipped_counts = {}
    for col in feature_cols:
        mean = df[col].mean()
        std = df[col].std()
        
        if std > 0:
            # Clip at mean ± 5 sigma
            lower_bound = mean - 5 * std
            upper_bound = mean + 5 * std
            
            outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
            if outliers.sum() > 0:
                clipped_counts[col] = outliers.sum()
                df[col] = df[col].clip(lower_bound, upper_bound)
    
    if len(clipped_counts) > 0:
        if verbose:
            print(f"\n4️⃣ Clipped outliers in {len(clipped_counts)} columns:")
            for col, count in list(clipped_counts.items())[:5]:
                print(f"      {col}: {count} values")
    else:
        if verbose:
            print("\n4️⃣ ✅ No extreme outliers")
    
    # 5. Final validation
    if verbose:
        print(f"\n5️⃣ Final validation:")
    
    final_nan = df[feature_cols].isnull().sum().sum()
    final_inf = sum([np.isinf(df[col]).sum() for col in feature_cols])
    
    if verbose:
        print(f"   NaN values: {final_nan}")
        print(f"   Inf values: {final_inf}")
        print(f"   Feature columns: {len(feature_cols)}")
        print(f"   Final shape: {df.shape}")
    
    # Save cleaned data
    df.to_csv(output_path, index=False)
    
    if verbose:
        print(f"\n💾 Saved cleaned data to: {output_path}")
    
    # Save cleaning report
    report = {
        'original_shape': original_shape,
        'final_shape': df.shape,
        'nan_fixed': int(nan_before),
        'inf_fixed': int(sum(inf_counts.values())) if inf_counts else 0,
        'zero_var_removed': len(zero_var_cols),
        'outliers_clipped': int(sum(clipped_counts.values())) if clipped_counts else 0,
        'final_features': len(feature_cols),
        'removed_columns': zero_var_cols
    }
    
    report_path = Path(output_path).parent / 'cleaning_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, indent=2, fp=f)
    
    if verbose:
        print(f"📄 Cleaning report saved to: {report_path}")
        print("=" * 80)
    
    return report

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Clean Peru rainfall data')
    parser.add_argument('--input', type=str, default='datasets/processed/peru_rainfall.csv',
                        help='Input CSV file')
    parser.add_argument('--output', type=str, default='datasets/processed/peru_rainfall_cleaned.csv',
                        help='Output CSV file')
    parser.add_argument('--inplace', action='store_true',
                        help='Overwrite input file')
    
    args = parser.parse_args()
    
    output_path = args.input if args.inplace else args.output
    
    report = clean_data(args.input, output_path, verbose=True)
    
    print("\n✅ DATA CLEANING COMPLETE!")
    print(f"   Use: --data_path {Path(output_path).name}")
