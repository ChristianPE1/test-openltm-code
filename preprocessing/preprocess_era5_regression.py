"""
Preprocessing script for ERA5 data - REGRESSION TASK (Rainfall Forecasting)
Converts ERA5 NetCDF files to format for CONTINUOUS precipitation prediction

⚠️ KEY DIFFERENCES vs preprocess_era5_peru.py:
   - NO binarization (keeps continuous mm values)
   - Target: 'target_precip_24h' in MM (not binary 0/1)
   - Optimized for MSE/MAE metrics (not F1-score)

Usage:
    python preprocessing/preprocess_era5_regression.py \
        --input_dir datasets/raw_era5 \
        --output_dir datasets/processed \
        --years 2014,2015,2016,2017,2018,2019,2020,2021,2022,2023,2024 \
        --target_horizon 24
"""

import os
import argparse
import numpy as np
import pandas as pd
import xarray as xr
import zipfile
from pathlib import Path
from datetime import datetime, timedelta
import json


class ERA5RegressionPreprocessor:
    """
    Preprocessor for ERA5 data - REGRESSION TASK
    Preserves continuous precipitation values (mm/24h)
    """
    
    def __init__(self, input_dir, output_dir, target_horizon=24):
        """
        Args:
            input_dir: Directory containing raw ERA5 .nc files
            output_dir: Directory to save processed data
            target_horizon: Hours ahead to predict (default: 24)
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.target_horizon = target_horizon
        
        # Peru coastal region (where ENSO impacts most)
        self.regions = {
            'costa_norte': {'lat': slice(-4, -8), 'lon': slice(-82, -78)},
            'costa_centro': {'lat': slice(-8, -14), 'lon': slice(-82, -76)},
            'costa_sur': {'lat': slice(-14, -18), 'lon': slice(-80, -70)},
        }
        
        # ERA5 variables
        self.variables = [
            'tp',      # total_precipitation (ACCUMULATED in METERS)
            't2m',     # temperature_2m
            'd2m',     # dewpoint_2m
            'sp',      # surface_pressure
            'msl',     # mean_sea_level_pressure
            'u10',     # u_component_of_wind_10m
            'v10',     # v_component_of_wind_10m
            'tcwv',    # total_column_water_vapour
            'cape'     # convective_available_potential_energy
        ]
        
        print("="*80)
        print("🌊 ERA5 REGRESSION PREPROCESSOR (Rainfall Forecasting)")
        print("="*80)
        print(f"📁 Input directory: {self.input_dir}")
        print(f"💾 Output directory: {self.output_dir}")
        print(f"⏱️  Target horizon: {self.target_horizon} hours")
        print(f"🗺️  Regions: {list(self.regions.keys())}")
        print(f"\n⚠️  REGRESSION MODE:")
        print(f"   ✅ Preserves CONTINUOUS precipitation values")
        print(f"   ✅ NO binarization (keeps mm/24h)")
        print(f"   ✅ Converts METERS → MILLIMETERS (×1000)")
        print(f"   ✅ Target: 'target_precip_24h' (continuous)")
        print("="*80)
    
    def load_nc_files(self, years):
        """Load .nc files for specified years"""
        print("\n[1/6] Loading ERA5 NetCDF files...")
        
        datasets = []
        for year in years:
            nc_file = self.input_dir / f"era5_peru_{year}.nc"
            
            if not nc_file.exists():
                print(f"  ⚠️  File not found: {nc_file}")
                continue
            
            print(f"  📂 Loading {nc_file.name}...")
            ds = xr.open_dataset(nc_file)
            
            # Standardize coordinate names
            if 'latitude' in ds.coords:
                ds = ds.rename({'latitude': 'lat', 'longitude': 'lon'})
            
            # Verificar variables
            available_vars = list(ds.data_vars)
            print(f"     Variables: {available_vars}")
            
            # Check if 'tp' is in METERS (typical ERA5)
            if 'tp' in ds:
                tp_sample = float(ds['tp'].isel(time=0, lat=0, lon=0).values)
                if tp_sample < 0.1:  # Likely in meters
                    print(f"     💧 'tp' detected in METERS (sample: {tp_sample:.6f} m)")
                    print(f"        Will convert to MILLIMETERS (×1000)")
                else:
                    print(f"     💧 'tp' appears to be in MM (sample: {tp_sample:.3f} mm)")
            
            datasets.append(ds)
        
        # Concatenate along time
        if len(datasets) > 1:
            print("\n  🔗 Concatenating datasets...")
            combined_ds = xr.concat(datasets, dim='time')
        else:
            combined_ds = datasets[0]
        
        combined_ds = combined_ds.sortby('time')
        
        print(f"\n✅ Combined dataset:")
        print(f"   ⏰ Time range: {combined_ds.time.values[0]} to {combined_ds.time.values[-1]}")
        print(f"   📊 Timesteps: {len(combined_ds.time)}")
        print(f"   🌍 Spatial: {len(combined_ds.lat)} × {len(combined_ds.lon)}")
        
        return combined_ds
    
    def spatial_aggregation(self, ds):
        """Aggregate by regions (mean over spatial dimensions)"""
        print("\n[2/6] Spatial aggregation by regions...")
        
        regional_data = {}
        
        for region_name, bounds in self.regions.items():
            print(f"  🗺️  Processing {region_name}...")
            
            # Select region
            ds_region = ds.sel(lat=bounds['lat'], lon=bounds['lon'])
            
            # Aggregate (mean over lat/lon)
            regional_means = {}
            for var in self.variables:
                if var in ds_region:
                    regional_means[var] = ds_region[var].mean(dim=['lat', 'lon'])
                else:
                    print(f"      ⚠️  Variable {var} not found")
            
            regional_data[region_name] = regional_means
            print(f"      ✅ {len(regional_means)} variables aggregated")
        
        print(f"\n✅ Regional aggregation complete")
        return regional_data
    
    def feature_engineering(self, regional_data):
        """Create derived features"""
        print("\n[3/6] Feature engineering...")
        
        engineered_data = {}
        
        for region_name, data in regional_data.items():
            print(f"  🔧 {region_name}...")
            
            features = {}
            
            # Original variables (convert 'tp' from METERS to MILLIMETERS)
            for var_name, var_data in data.items():
                if var_name == 'tp':
                    # ⚠️ CRITICAL: Convert meters → millimeters
                    values = var_data.values * 1000.0  # m → mm
                    features['total_precipitation'] = values
                    print(f"      💧 'tp' converted to MM (mean: {np.nanmean(values):.3f} mm)")
                else:
                    features[var_name] = var_data.values
            
            # Derived features
            if 'u10' in data and 'v10' in data:
                features['wind_speed'] = np.sqrt(
                    data['u10'].values**2 + data['v10'].values**2
                )
                features['wind_direction'] = np.arctan2(
                    data['v10'].values, data['u10'].values
                )
            
            if 't2m' in data and 'd2m' in data:
                features['td_spread'] = data['t2m'].values - data['d2m'].values
                features['relative_humidity'] = 100 * np.exp(
                    (17.625 * data['d2m'].values) / (243.04 + data['d2m'].values)
                ) / np.exp(
                    (17.625 * data['t2m'].values) / (243.04 + data['t2m'].values)
                )
            
            # Time-lagged features (precipitation history)
            if 'total_precipitation' in features:
                precip = features['total_precipitation']
                for lag_days in [1, 2, 3]:
                    lag_steps = lag_days * 2  # 12-hourly data
                    lagged = np.concatenate([
                        np.full(lag_steps, np.nan),
                        precip[:-lag_steps]
                    ])
                    features[f'precip_lag_{lag_days}d'] = lagged
                
                # Rolling statistics (7-day window)
                window = 14  # 7 days * 2
                rolling_mean = pd.Series(precip).rolling(window=window, min_periods=1).mean().values
                rolling_std = pd.Series(precip).rolling(window=window, min_periods=1).std().values
                rolling_max = pd.Series(precip).rolling(window=window, min_periods=1).max().values
                
                features['precip_rolling_mean_7d'] = rolling_mean
                features['precip_rolling_std_7d'] = rolling_std
                features['precip_rolling_max_7d'] = rolling_max
            
            engineered_data[region_name] = features
            print(f"      ✅ {len(features)} features created")
        
        return engineered_data
    
    def create_target_variable(self, regional_data):
        """
        Create CONTINUOUS target: total precipitation in next 24 hours (MM)
        
        NO binarization - preserves actual mm values
        
        ⚠️ CRITICAL: ERA5 'tp' is ACCUMULATED precipitation
        For 24h forecast, we need: precip(t+24h) - precip(t)
        """
        print("\n[4/6] Creating CONTINUOUS target variable...")
        
        targets = {}
        
        for region_name, features in regional_data.items():
            if 'total_precipitation' not in features:
                print(f"  ⚠️  Precipitation not found for {region_name}")
                continue
            
            precip = features['total_precipitation']  # Already in MM
            horizon_steps = self.target_horizon // 12  # 12-hourly resolution (24h = 2 steps)
            
            # ⚠️ CRITICAL FIX: ERA5 'tp' is ACCUMULATED
            # Target = difference between t+24h and t (accumulated precip in 24h window)
            # For 12-hourly data: target_24h[t] = precip[t+2] - precip[t]
            
            # Method: Calculate 24h accumulated precipitation
            # For each timestep t, we want total precip from t to t+24h
            target_24h = np.full(len(precip), np.nan)
            
            for i in range(len(precip) - horizon_steps):
                # Sum of precipitation in next horizon_steps (24h)
                # Since data is 12-hourly accumulated, we sum next 2 values
                window_precip = precip[i:i+horizon_steps]
                target_24h[i] = np.sum(window_precip)
            
            targets[region_name] = target_24h
            
            # Statistics (excluding NaN)
            valid_target = target_24h[~np.isnan(target_24h)]
            n_samples = len(valid_target)
            mean_precip = np.mean(valid_target)
            median_precip = np.median(valid_target)
            max_precip = np.max(valid_target)
            min_precip = np.min(valid_target)
            rainy_days = np.sum(valid_target > 0.1)  # >0.1mm considered rain
            heavy_rain = np.sum(valid_target > 10.0)  # >10mm heavy rain
            extreme_rain = np.sum(valid_target > 50.0)  # >50mm extreme
            
            print(f"  📊 {region_name}:")
            print(f"     Samples: {n_samples}")
            print(f"     Mean: {mean_precip:.3f} mm/24h")
            print(f"     Median: {median_precip:.3f} mm/24h")
            print(f"     Range: [{min_precip:.3f}, {max_precip:.3f}] mm")
            print(f"     Rainy days (>0.1mm): {rainy_days} ({100*rainy_days/n_samples:.1f}%)")
            print(f"     Heavy rain (>10mm): {heavy_rain} ({100*heavy_rain/n_samples:.2f}%)")
            print(f"     Extreme rain (>50mm): {extreme_rain} ({100*extreme_rain/n_samples:.2f}%)")
        
        print(f"\n✅ CONTINUOUS target created (NO binarization)")
        return targets
    
    def create_dataframe(self, engineered_data, targets, time_index):
        """Combine features and targets into DataFrame"""
        print("\n[5/6] Creating DataFrame...")
        
        all_data = []
        
        for region_name in engineered_data.keys():
            if region_name not in targets:
                continue
            
            features = engineered_data[region_name]
            target = targets[region_name]
            
            # Create dataframe for this region
            region_df = pd.DataFrame(features)
            region_df['target_precip_24h'] = target  # CONTINUOUS (mm)
            region_df['timestamp'] = time_index
            region_df['region'] = region_name
            
            all_data.append(region_df)
        
        # Combine all regions
        df = pd.concat(all_data, ignore_index=True)
        
        # Remove rows with NaN target
        df = df.dropna(subset=['target_precip_24h'])
        
        print(f"\n✅ DataFrame created:")
        print(f"   📊 Shape: {df.shape}")
        print(f"   📅 Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"   🗺️  Regions: {df['region'].unique()}")
        
        # Statistics
        print(f"\n📊 TARGET STATISTICS (target_precip_24h):")
        print(f"   Mean: {df['target_precip_24h'].mean():.3f} mm")
        print(f"   Median: {df['target_precip_24h'].median():.3f} mm")
        print(f"   Std: {df['target_precip_24h'].std():.3f} mm")
        print(f"   Min: {df['target_precip_24h'].min():.3f} mm")
        print(f"   Max: {df['target_precip_24h'].max():.3f} mm")
        print(f"   25%ile: {df['target_precip_24h'].quantile(0.25):.3f} mm")
        print(f"   75%ile: {df['target_precip_24h'].quantile(0.75):.3f} mm")
        
        rainy_pct = 100 * (df['target_precip_24h'] > 0.1).sum() / len(df)
        heavy_pct = 100 * (df['target_precip_24h'] > 10.0).sum() / len(df)
        extreme_pct = 100 * (df['target_precip_24h'] > 50.0).sum() / len(df)
        
        print(f"\n📊 RAINFALL DISTRIBUTION:")
        print(f"   Rainy days (>0.1mm): {rainy_pct:.1f}%")
        print(f"   Heavy rain (>10mm): {heavy_pct:.1f}%")
        print(f"   Extreme rain (>50mm): {extreme_pct:.1f}%")
        
        return df
    
    def save_data(self, df, output_name='peru_rainfall_regression.csv'):
        """Save processed data"""
        print("\n[6/6] Saving processed data...")
        
        output_path = self.output_dir / output_name
        df.to_csv(output_path, index=False)
        
        print(f"✅ Data saved to: {output_path}")
        
        # Save statistics
        stats = {
            'total_samples': len(df),
            'date_range': {
                'start': str(df['timestamp'].min()),
                'end': str(df['timestamp'].max())
            },
            'regions': df['region'].unique().tolist(),
            'target_statistics': {
                'mean_mm': float(df['target_precip_24h'].mean()),
                'median_mm': float(df['target_precip_24h'].median()),
                'std_mm': float(df['target_precip_24h'].std()),
                'min_mm': float(df['target_precip_24h'].min()),
                'max_mm': float(df['target_precip_24h'].max()),
                'q25_mm': float(df['target_precip_24h'].quantile(0.25)),
                'q75_mm': float(df['target_precip_24h'].quantile(0.75))
            },
            'rainfall_distribution': {
                'rainy_days_pct': float(100 * (df['target_precip_24h'] > 0.1).sum() / len(df)),
                'heavy_rain_pct': float(100 * (df['target_precip_24h'] > 10.0).sum() / len(df)),
                'extreme_rain_pct': float(100 * (df['target_precip_24h'] > 50.0).sum() / len(df))
            },
            'features': df.columns.tolist()
        }
        
        stats_path = self.output_dir / 'regression_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"✅ Statistics saved to: {stats_path}")
        
        return output_path
    
    def process(self, years):
        """Main processing pipeline"""
        print("\n" + "="*80)
        print("🚀 STARTING REGRESSION PREPROCESSING PIPELINE")
        print("="*80)
        
        # Load data
        ds = self.load_nc_files(years)
        
        # Spatial aggregation
        regional_data = self.spatial_aggregation(ds)
        
        # Feature engineering
        engineered_data = self.feature_engineering(regional_data)
        
        # Create target
        targets = self.create_target_variable(engineered_data)
        
        # Get time index (use first region's time)
        first_region = list(regional_data.keys())[0]
        time_index = regional_data[first_region]['tp'].time.values
        
        # Create dataframe
        df = self.create_dataframe(engineered_data, targets, time_index)
        
        # Save
        output_path = self.save_data(df)
        
        print("\n" + "="*80)
        print("✅ REGRESSION PREPROCESSING COMPLETE!")
        print("="*80)
        print(f"\n💡 NEXT STEPS:")
        print(f"   1. Verify data: !head -n 5 {output_path}")
        print(f"   2. Check stats: !cat {self.output_dir / 'regression_stats.json'}")
        print(f"   3. Train regression model (see notebook cells)")
        print(f"\n🎯 TARGET METRICS:")
        print(f"   • RMSE < 3.5 mm (good)")
        print(f"   • MAE < 2.5 mm (good)")
        print(f"   • RMSE < 3.0 mm (excellent)")
        print("="*80)
        
        return df


def main():
    parser = argparse.ArgumentParser(description='Preprocess ERA5 data for REGRESSION task')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory with ERA5 .nc files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--years', type=str, required=True,
                        help='Comma-separated years (e.g., 2014,2015,2016)')
    parser.add_argument('--target_horizon', type=int, default=24,
                        help='Hours ahead to predict (default: 24)')
    
    args = parser.parse_args()
    
    # Parse years
    years = [int(y.strip()) for y in args.years.split(',')]
    
    # Create preprocessor
    preprocessor = ERA5RegressionPreprocessor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        target_horizon=args.target_horizon
    )
    
    # Process
    df = preprocessor.process(years)
    
    print(f"\n✅ Done! Processed {len(df)} samples from {len(years)} years")


if __name__ == '__main__':
    main()
