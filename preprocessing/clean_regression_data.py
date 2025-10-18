"""
Data Cleaning Script for REGRESSION TASK
Handles NaN values, outliers, and data quality issues differently than classification

⚠️ KEY DIFFERENCES vs clean_classification_data.py:
   - MORE AGGRESSIVE NaN handling (regression is sensitive)
   - Outlier detection (extreme precipitation values)
   - Temporal interpolation (better for continuous data)
   - Feature scaling preparation

Usage:
    python preprocessing/clean_regression_data.py \
        --input_path datasets/processed/peru_rainfall_regression.csv \
        --output_path datasets/processed/peru_rainfall_regression_cleaned.csv
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime


class RegressionDataCleaner:
    """
    Data cleaner optimized for REGRESSION task
    Handles continuous precipitation values (mm/24h)
    """
    
    def __init__(self, max_precip_threshold=200.0):
        """
        Args:
            max_precip_threshold: Max valid precipitation (mm/24h)
                                  Values > threshold considered outliers
                                  (200mm = extreme El Niño event)
        """
        self.max_precip_threshold = max_precip_threshold
        
        print("="*80)
        print("🧹 REGRESSION DATA CLEANER")
        print("="*80)
        print(f"⚠️  REGRESSION MODE:")
        print(f"   ✅ Aggressive NaN removal (MSE sensitive)")
        print(f"   ✅ Outlier detection (extreme events)")
        print(f"   ✅ Temporal interpolation (continuous data)")
        print(f"   ✅ Feature quality checks")
        print(f"\n🌧️  Precipitation outlier threshold: {max_precip_threshold} mm/24h")
        print("="*80)
    
    def load_data(self, input_path):
        """Load raw regression data"""
        print(f"\n[1/7] Loading data from {input_path}...")
        
        df = pd.read_csv(input_path)
        
        print(f"  📊 Shape: {df.shape}")
        print(f"  📅 Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        # Check critical columns
        required_cols = ['timestamp', 'region', 'target_precip_24h', 'total_precipitation']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        return df
    
    def analyze_missing_values(self, df):
        """Analyze NaN patterns"""
        print("\n[2/7] Analyzing missing values...")
        
        total_rows = len(df)
        
        # Count NaN per column
        nan_counts = df.isnull().sum()
        nan_cols = nan_counts[nan_counts > 0].sort_values(ascending=False)
        
        print(f"\n  📊 Columns with NaN values:")
        for col, count in nan_cols.items():
            pct = 100 * count / total_rows
            print(f"     {col}: {count} ({pct:.2f}%)")
        
        # Rows with ANY NaN
        rows_with_nan = df.isnull().any(axis=1).sum()
        print(f"\n  🔍 Rows with ANY NaN: {rows_with_nan} ({100*rows_with_nan/total_rows:.2f}%)")
        
        return nan_cols
    
    def remove_extreme_outliers(self, df):
        """
        Remove EXTREME outliers in precipitation
        
        Keep El Niño extremes (up to 200mm) but remove impossible values
        """
        print(f"\n[3/7] Detecting extreme outliers...")
        
        initial_size = len(df)
        
        # Check target variable
        print(f"\n  🎯 Target variable (target_precip_24h):")
        print(f"     Min: {df['target_precip_24h'].min():.3f} mm")
        print(f"     Max: {df['target_precip_24h'].max():.3f} mm")
        print(f"     Mean: {df['target_precip_24h'].mean():.3f} mm")
        print(f"     Std: {df['target_precip_24h'].std():.3f} mm")
        
        # Remove negative precipitation (impossible)
        negative_mask = df['target_precip_24h'] < 0
        n_negative = negative_mask.sum()
        if n_negative > 0:
            print(f"  ⚠️  Found {n_negative} negative precipitation values (removing)")
            df = df[~negative_mask]
        
        # Remove extreme outliers (> threshold)
        outlier_mask = df['target_precip_24h'] > self.max_precip_threshold
        n_outliers = outlier_mask.sum()
        if n_outliers > 0:
            print(f"  ⚠️  Found {n_outliers} extreme outliers (>{self.max_precip_threshold}mm)")
            print(f"     Max outlier value: {df.loc[outlier_mask, 'target_precip_24h'].max():.1f} mm")
            print(f"     Action: REMOVING (likely data errors)")
            df = df[~outlier_mask]
        
        final_size = len(df)
        removed = initial_size - final_size
        
        if removed > 0:
            print(f"\n  ✅ Removed {removed} rows with extreme outliers ({100*removed/initial_size:.2f}%)")
        else:
            print(f"\n  ✅ No extreme outliers detected")
        
        return df
    
    def handle_feature_nans(self, df):
        """
        Handle NaN in feature columns
        
        STRATEGY for regression:
        1. Temporal interpolation (better than median for time series)
        2. Forward/backward fill (preserve temporal patterns)
        3. Remove rows if still NaN (aggressive but safe for regression)
        """
        print("\n[4/7] Handling NaN in features...")
        
        # Get feature columns (exclude metadata and target)
        metadata_cols = ['timestamp', 'region', 'target_precip_24h']
        feature_cols = [col for col in df.columns if col not in metadata_cols]
        
        print(f"  📊 Feature columns: {len(feature_cols)}")
        
        # Convert timestamp to datetime for interpolation
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(['region', 'timestamp'])
        
        initial_rows = len(df)
        
        # Process each region separately (interpolation within region)
        cleaned_dfs = []
        
        for region in df['region'].unique():
            region_df = df[df['region'] == region].copy()
            
            # Method 1: Linear interpolation (time-aware)
            region_df[feature_cols] = region_df[feature_cols].interpolate(
                method='linear',
                limit_direction='both',
                limit=5  # Max 5 timesteps gap (2.5 days for 12h data)
            )
            
            # Method 2: Forward/Backward fill (for remaining NaN)
            region_df[feature_cols] = region_df[feature_cols].fillna(method='ffill', limit=3)
            region_df[feature_cols] = region_df[feature_cols].fillna(method='bfill', limit=3)
            
            cleaned_dfs.append(region_df)
        
        df = pd.concat(cleaned_dfs, ignore_index=True)
        
        # Method 3: Remove rows with STILL NaN (aggressive)
        rows_before = len(df)
        df = df.dropna(subset=feature_cols)
        rows_after = len(df)
        removed = rows_before - rows_after
        
        total_removed = initial_rows - rows_after
        
        print(f"\n  ✅ Cleaning summary:")
        print(f"     Interpolated: {initial_rows - rows_before} NaN values")
        print(f"     Removed rows: {removed} ({100*removed/rows_before:.2f}%)")
        print(f"     Total removed: {total_removed} ({100*total_removed/initial_rows:.2f}%)")
        print(f"     Remaining: {rows_after} rows")
        
        return df
    
    def validate_data_quality(self, df):
        """Final data quality checks"""
        print("\n[5/7] Validating data quality...")
        
        # Check 1: No NaN in features
        nan_count = df.isnull().sum().sum()
        if nan_count > 0:
            print(f"  ⚠️  WARNING: Still {nan_count} NaN values found!")
            print(df.isnull().sum()[df.isnull().sum() > 0])
        else:
            print(f"  ✅ No NaN values in features")
        
        # Check 2: Target variable statistics
        print(f"\n  🎯 Target variable (target_precip_24h):")
        print(f"     Range: [{df['target_precip_24h'].min():.3f}, {df['target_precip_24h'].max():.3f}] mm")
        print(f"     Mean: {df['target_precip_24h'].mean():.3f} mm")
        print(f"     Median: {df['target_precip_24h'].median():.3f} mm")
        print(f"     Std: {df['target_precip_24h'].std():.3f} mm")
        
        # Check 3: Unique values (should be thousands, not 2)
        unique_target = df['target_precip_24h'].nunique()
        print(f"     Unique values: {unique_target}")
        
        if unique_target < 100:
            print(f"  ⚠️  WARNING: Only {unique_target} unique values (expected thousands)")
            print(f"     This suggests binary data, not continuous!")
            raise ValueError("Data appears to be binary, not continuous regression data")
        else:
            print(f"  ✅ CONTINUOUS data confirmed ({unique_target} unique values)")
        
        # Check 4: Distribution
        rainy_days = (df['target_precip_24h'] > 0.1).sum()
        heavy_rain = (df['target_precip_24h'] > 10.0).sum()
        extreme_rain = (df['target_precip_24h'] > 50.0).sum()
        
        total = len(df)
        print(f"\n  🌧️  Precipitation distribution:")
        print(f"     Rainy (>0.1mm): {rainy_days} ({100*rainy_days/total:.1f}%)")
        print(f"     Heavy (>10mm): {heavy_rain} ({100*heavy_rain/total:.1f}%)")
        print(f"     Extreme (>50mm): {extreme_rain} ({100*extreme_rain/total:.1f}%)")
        
        # Check 5: Regional balance
        print(f"\n  🗺️  Regional distribution:")
        region_counts = df['region'].value_counts()
        for region, count in region_counts.items():
            print(f"     {region}: {count} ({100*count/total:.1f}%)")
        
        return True
    
    def prepare_feature_list(self, df):
        """Generate feature list for Timer-XL"""
        print("\n[6/7] Preparing feature list...")
        
        # Exclude metadata columns
        metadata_cols = ['timestamp', 'region', 'target_precip_24h']
        feature_cols = [col for col in df.columns if col not in metadata_cols]
        
        print(f"  📊 Total features: {len(feature_cols)}")
        print(f"\n  Feature categories:")
        
        # Categorize features
        base_features = [col for col in feature_cols if not any(x in col for x in ['lag', 'rolling'])]
        lag_features = [col for col in feature_cols if 'lag' in col]
        rolling_features = [col for col in feature_cols if 'rolling' in col]
        
        print(f"     Base features: {len(base_features)}")
        print(f"     Lag features: {len(lag_features)}")
        print(f"     Rolling features: {len(rolling_features)}")
        
        return feature_cols
    
    def save_cleaned_data(self, df, output_path, stats):
        """Save cleaned data and statistics"""
        print(f"\n[7/7] Saving cleaned data to {output_path}...")
        
        df.to_csv(output_path, index=False)
        
        # Save statistics
        stats_path = output_path.parent / 'regression_cleaning_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"  ✅ Data saved: {output_path}")
        print(f"  📊 Stats saved: {stats_path}")
        
        return output_path, stats_path


def main():
    parser = argparse.ArgumentParser(description='Clean regression data')
    parser.add_argument('--input_path', type=str, required=True,
                       help='Input CSV file (raw regression data)')
    parser.add_argument('--output_path', type=str, required=True,
                       help='Output CSV file (cleaned data)')
    parser.add_argument('--max_precip', type=float, default=200.0,
                       help='Maximum valid precipitation (mm/24h)')
    
    args = parser.parse_args()
    
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize cleaner
    cleaner = RegressionDataCleaner(max_precip_threshold=args.max_precip)
    
    # Load data
    df = cleaner.load_data(input_path)
    initial_shape = df.shape
    
    # Analyze missing values
    nan_cols = cleaner.analyze_missing_values(df)
    
    # Remove extreme outliers
    df = cleaner.remove_extreme_outliers(df)
    
    # Handle feature NaN
    df = cleaner.handle_feature_nans(df)
    
    # Validate quality
    cleaner.validate_data_quality(df)
    
    # Prepare feature list
    feature_cols = cleaner.prepare_feature_list(df)
    
    # Collect statistics
    stats = {
        'cleaning_date': datetime.now().isoformat(),
        'initial_shape': list(initial_shape),
        'final_shape': list(df.shape),
        'rows_removed': initial_shape[0] - df.shape[0],
        'rows_removed_pct': 100 * (initial_shape[0] - df.shape[0]) / initial_shape[0],
        'target_statistics': {
            'mean': float(df['target_precip_24h'].mean()),
            'median': float(df['target_precip_24h'].median()),
            'std': float(df['target_precip_24h'].std()),
            'min': float(df['target_precip_24h'].min()),
            'max': float(df['target_precip_24h'].max()),
            'unique_values': int(df['target_precip_24h'].nunique())
        },
        'precipitation_distribution': {
            'rainy_days_pct': float(100 * (df['target_precip_24h'] > 0.1).sum() / len(df)),
            'heavy_rain_pct': float(100 * (df['target_precip_24h'] > 10.0).sum() / len(df)),
            'extreme_rain_pct': float(100 * (df['target_precip_24h'] > 50.0).sum() / len(df))
        },
        'feature_count': len(feature_cols),
        'features': feature_cols,
        'regions': df['region'].unique().tolist(),
        'date_range': {
            'start': str(df['timestamp'].min()),
            'end': str(df['timestamp'].max())
        }
    }
    
    # Save cleaned data
    cleaner.save_cleaned_data(df, output_path, stats)
    
    print("\n" + "="*80)
    print("✅ CLEANING COMPLETED")
    print("="*80)
    print(f"\n📊 SUMMARY:")
    print(f"   Initial rows: {initial_shape[0]}")
    print(f"   Final rows: {df.shape[0]}")
    print(f"   Removed: {stats['rows_removed']} ({stats['rows_removed_pct']:.2f}%)")
    print(f"   Features: {len(feature_cols)}")
    print(f"   Target range: [{stats['target_statistics']['min']:.3f}, {stats['target_statistics']['max']:.3f}] mm")
    print(f"   Unique target values: {stats['target_statistics']['unique_values']}")
    print(f"\n🎯 READY FOR REGRESSION TRAINING")
    print("="*80)


if __name__ == '__main__':
    main()
