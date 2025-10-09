"""
Preprocessing script for ERA5 data - Peru Rainfall Prediction
Converts ERA5 NetCDF files to format compatible with Timer-XL

Usage:
    python preprocessing/preprocess_era5_peru.py \
        --input_dir datasets/raw_era5 \
        --output_dir datasets/processed \
        --years 2023,2024 \
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


class ERA5PreprocessorPeru:
    """
    Preprocessor for ERA5 data from Peru
    Handles spatial aggregation, feature engineering, and target creation
    """
    
    def __init__(self, input_dir, output_dir, target_horizon=24, threshold=0.1):
        """
        Args:
            input_dir: Directory containing raw ERA5 .zip files
            output_dir: Directory to save processed data
            target_horizon: Hours ahead to predict (default: 24)
            threshold: Precipitation threshold in mm to define "rain" (default: 0.1)
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.target_horizon = target_horizon
        self.threshold = threshold
        
        # Peru regions (approximate boundaries)
        self.regions = {
            'costa_norte': {'lat': slice(0, -5), 'lon': slice(-82, -78)},
            'costa_centro': {'lat': slice(-5, -10), 'lon': slice(-82, -76)},
            'costa_sur': {'lat': slice(-10, -18), 'lon': slice(-80, -70)},
            'sierra_norte': {'lat': slice(0, -10), 'lon': slice(-78, -73)},
            'sierra_sur': {'lat': slice(-10, -18), 'lon': slice(-76, -68)}
        }
        
        # ERA5 variables (CORE - 9 variables obligatorias)
        self.variables = [
            'tp',      # total_precipitation (ACCUMULATED)
            't2m',     # temperature_2m
            'd2m',     # dewpoint_2m
            'sp',      # surface_pressure
            'msl',     # mean_sea_level_pressure
            'u10',     # u_component_of_wind_10m
            'v10',     # v_component_of_wind_10m
            'tcwv',    # total_column_water_vapour
            'cape'     # convective_available_potential_energy
        ]
        
        # Variables OPCIONALES (si las tienes, se usarán)
        self.optional_variables = [
            'tcc',     # total_cloud_cover
            'sst',     # sea_surface_temperature
            'tp_mean'  # mean_total_precipitation_rate
        ]
        
        print("="*80)
        print("ERA5 Preprocessor for Peru Rainfall Prediction")
        print("="*80)
        print(f"Input directory: {self.input_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Target horizon: {self.target_horizon} hours")
        print(f"Rain threshold: {self.threshold} mm")
        print(f"Regions: {list(self.regions.keys())}")
        print("="*80)
    
    def extract_zip_files(self, years):
        """
        Extract .zip files for specified years OR use .nc files directly
        Maneja archivos CDS que contienen múltiples .nc (accum + instant)
        O archivos .nc ya combinados/extraídos
        """
        print("\n[1/6] Locating data files...")
        
        extracted_files = []
        for year in years:
            # Primero buscar archivos .nc directos (ya combinados)
            nc_direct = self.input_dir / f"era5_peru_{year}.nc"
            if nc_direct.exists():
                print(f"  ✅ Found .nc file: {nc_direct.name}")
                extracted_files.append(nc_direct)
                continue
            
            # Si no hay .nc, buscar .zip para extraer
            possible_names = [
                f"era5_peru_{year}.zip",  # Formato esperado
                f"cds_{year}.zip"          # Formato CDS original
            ]
            
            zip_path = None
            for name in possible_names:
                candidate = self.input_dir / name
                if candidate.exists():
                    zip_path = candidate
                    break
            
            if zip_path is None:
                print(f"⚠️  Warning: No ZIP file found for year {year}")
                print(f"    Tried: {', '.join(possible_names)}")
                continue
            
            print(f"  Extracting {zip_path.name}...")
            
            # Extraer contenido
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                file_list = zip_ref.namelist()
                print(f"    Archivos en ZIP: {len(file_list)}")
                for f in file_list:
                    print(f"      - {f}")
                zip_ref.extractall(self.input_dir)
            
            # Buscar archivos .nc extraídos
            nc_files_in_zip = [f for f in file_list if f.endswith('.nc')]
            
            if len(nc_files_in_zip) == 0:
                print(f"  ⚠️  No .nc files found in {zip_path.name}")
                continue
            
            # Identificar archivo 'accum' (contiene precipitación)
            accum_file = None
            for nc_file in nc_files_in_zip:
                nc_path = self.input_dir / nc_file
                if 'accum' in nc_file.lower():
                    accum_file = nc_path
                    print(f"  ✅ Found ACCUMULATED file: {nc_file}")
                    break
            
            # Si no hay archivo 'accum', buscar cualquier .nc con 'tp'
            if accum_file is None:
                print(f"  🔍 No 'accum' file found. Checking all .nc files for 'tp'...")
                for nc_file in nc_files_in_zip:
                    nc_path = self.input_dir / nc_file
                    try:
                        ds_test = xr.open_dataset(nc_path)
                        if 'tp' in ds_test.data_vars:
                            accum_file = nc_path
                            print(f"  ✅ Found file with 'tp': {nc_file}")
                            ds_test.close()
                            break
                        ds_test.close()
                    except:
                        continue
            
            if accum_file and accum_file.exists():
                extracted_files.append(accum_file)
                print(f"  ✅ Will use: {accum_file.name}")
            else:
                print(f"  ❌ No suitable .nc file found for {year}")
                print(f"      Make sure your ZIP contains 'tp' (precipitation) variable")
        
        print(f"\n✅ Extracted {len(extracted_files)} usable files")
        return extracted_files
    
    def load_netcdf_files(self, nc_files):
        """
        Load and combine multiple NetCDF files
        Filtra solo horas 06 y 18 UTC (ignora 00 y 12 si existen)
        """
        print("\n[2/6] Loading NetCDF files...")
        
        datasets = []
        for nc_file in nc_files:
            print(f"  Loading {nc_file.name}...")
            ds = xr.open_dataset(nc_file)
            
            # Información inicial
            print(f"    Time range: {ds.time.values[0]} to {ds.time.values[-1]}")
            print(f"    Total timesteps: {len(ds.time)}")
            print(f"    Variables: {list(ds.data_vars)}")
            
            # Detectar horas disponibles
            times = pd.to_datetime(ds.time.values)
            unique_hours = sorted(set(times.hour))
            print(f"    Hours (UTC): {unique_hours}")
            
            # Filtrar solo 06 y 18 UTC
            if len(unique_hours) > 2:
                print(f"    🔧 Filtering to keep only 06 and 18 UTC...")
                mask = (times.hour == 6) | (times.hour == 18)
                ds = ds.sel(time=mask)
                times_filtered = pd.to_datetime(ds.time.values)
                print(f"    ✅ Filtered: {len(ds.time)} timesteps (from {len(times)})")
                print(f"    New hours: {sorted(set(times_filtered.hour))}")
            elif 6 in unique_hours and 18 in unique_hours:
                print(f"    ✅ Already has only 06 and 18 UTC")
            else:
                print(f"    ⚠️  WARNING: Does not contain 06 and 18 UTC!")
                print(f"    Available: {unique_hours}")
            
            datasets.append(ds)
        
        # Combine datasets along time dimension
        if len(datasets) > 1:
            print("\n  Concatenating datasets...")
            combined_ds = xr.concat(datasets, dim='time')
        else:
            combined_ds = datasets[0]
        
        # Sort by time
        combined_ds = combined_ds.sortby('time')
        
        print(f"\n✅ Combined dataset:")
        print(f"   Time range: {combined_ds.time.values[0]} to {combined_ds.time.values[-1]}")
        print(f"   Total timesteps: {len(combined_ds.time)}")
        
        # Verificar resolución espacial
        lat_name = 'latitude' if 'latitude' in combined_ds.coords else 'lat'
        lon_name = 'longitude' if 'longitude' in combined_ds.coords else 'lon'
        print(f"   Spatial resolution: {len(combined_ds[lat_name])} x {len(combined_ds[lon_name])}")
        lat_min = float(combined_ds[lat_name].min())
        lat_max = float(combined_ds[lat_name].max())
        lon_min = float(combined_ds[lon_name].min())
        lon_max = float(combined_ds[lon_name].max())
        print(f"   Lat range: {lat_min:.2f} to {lat_max:.2f}")
        print(f"   Lon range: {lon_min:.2f} to {lon_max:.2f}")
        
        # Verificar variables disponibles
        available_vars = list(combined_ds.data_vars)
        required_present = [v for v in self.variables if v in available_vars]
        optional_present = [v for v in self.optional_variables if v in available_vars]
        
        print(f"\n   📊 Variables check:")
        print(f"      Required: {len(required_present)}/{len(self.variables)}")
        print(f"      Optional: {len(optional_present)}/{len(self.optional_variables)}")
        
        if len(required_present) < len(self.variables):
            missing = [v for v in self.variables if v not in available_vars]
            print(f"      ⚠️  Missing required: {missing}")
        
        return combined_ds
    
    def spatial_aggregation(self, ds):
        """Aggregate data by regions"""
        print("\n[3/6] Spatial aggregation by regions...")
        
        regional_data = {}
        
        for region_name, bounds in self.regions.items():
            print(f"  Processing {region_name}...")
            
            # Select region
            ds_region = ds.sel(
                latitude=bounds['lat'],
                longitude=bounds['lon']
            )
            
            # Calculate mean over spatial dimensions
            regional_means = {}
            for var in self.variables:
                if var in ds_region:
                    regional_means[var] = ds_region[var].mean(dim=['latitude', 'longitude'])
                else:
                    print(f"    ⚠️  Variable {var} not found")
            
            regional_data[region_name] = regional_means
            print(f"    ✅ {len(regional_means)} variables aggregated")
        
        print(f"\n✅ Regional aggregation complete for {len(regional_data)} regions")
        return regional_data
    
    def feature_engineering(self, regional_data):
        """Create derived features"""
        print("\n[4/6] Feature engineering...")
        
        engineered_data = {}
        
        for region_name, data in regional_data.items():
            print(f"  Engineering features for {region_name}...")
            
            features = {}
            
            # Original variables
            for var_name, var_data in data.items():
                features[f'{var_name}'] = var_data.values
            
            # Derived features
            if 'u10' in data and 'v10' in data:
                # Wind speed
                features['wind_speed'] = np.sqrt(
                    data['u10'].values**2 + data['v10'].values**2
                )
                # Wind direction
                features['wind_direction'] = np.arctan2(
                    data['v10'].values, data['u10'].values
                )
            
            if 't2m' in data and 'd2m' in data:
                # Temperature - Dewpoint spread
                features['td_spread'] = data['t2m'].values - data['d2m'].values
                # Relative humidity (approximation)
                features['relative_humidity'] = 100 * np.exp(
                    (17.625 * data['d2m'].values) / (243.04 + data['d2m'].values)
                ) / np.exp(
                    (17.625 * data['t2m'].values) / (243.04 + data['t2m'].values)
                )
            
            if 'sp' in data:
                # Pressure tendency (change over 12h)
                pressure = data['sp'].values
                features['pressure_tendency_12h'] = np.concatenate([
                    [0],  # First value
                    np.diff(pressure)
                ])
            
            # Time-lagged features (1, 2, 3 days)
            for var_name in ['tp', 't2m', 'tcwv']:
                if var_name in data:
                    values = data[var_name].values
                    for lag_days in [1, 2, 3]:
                        lag_steps = lag_days * 2  # 12-hourly data
                        lagged = np.concatenate([
                            np.full(lag_steps, np.nan),
                            values[:-lag_steps]
                        ])
                        features[f'{var_name}_lag_{lag_days}d'] = lagged
            
            # Rolling statistics (7-day window)
            for var_name in ['tp', 't2m']:
                if var_name in data:
                    values = data[var_name].values
                    window = 14  # 7 days * 2 (12h resolution)
                    
                    rolling_mean = pd.Series(values).rolling(window=window, min_periods=1).mean().values
                    rolling_std = pd.Series(values).rolling(window=window, min_periods=1).std().values
                    
                    features[f'{var_name}_rolling_mean_7d'] = rolling_mean
                    features[f'{var_name}_rolling_std_7d'] = rolling_std
            
            engineered_data[region_name] = features
            print(f"    ✅ Created {len(features)} features")
        
        print(f"\n✅ Feature engineering complete")
        return engineered_data
    
    def create_target_variable(self, regional_data):
        """Create binary target: will it rain in next 24 hours?"""
        print("\n[5/6] Creating target variable...")
        
        targets = {}
        
        for region_name, data in regional_data.items():
            if 'tp' not in data:
                print(f"  ⚠️  Precipitation not found for {region_name}")
                continue
            
            # Precipitation in next 24 hours
            precip = data['tp'].values
            horizon_steps = self.target_horizon // 12  # 12-hourly resolution
            
            # Shift precipitation forward
            rain_24h = np.concatenate([
                precip[horizon_steps:],
                np.full(horizon_steps, np.nan)
            ])
            
            # Binary target: 1 if rain > threshold, 0 otherwise
            target_binary = (rain_24h >= self.threshold).astype(int)
            
            # Handle NaN (end of series)
            target_binary = np.where(np.isnan(rain_24h), -1, target_binary)
            
            targets[region_name] = {
                'rain_24h_continuous': rain_24h,
                'rain_24h_binary': target_binary
            }
            
            n_rain = np.sum(target_binary == 1)
            n_no_rain = np.sum(target_binary == 0)
            print(f"  {region_name}: {n_rain} rain days, {n_no_rain} no-rain days")
        
        print(f"\n✅ Target creation complete")
        return targets
    
    def create_dataframe(self, engineered_data, targets, time_index):
        """Combine all data into a single DataFrame"""
        print("\n[6/6] Creating final DataFrame...")
        
        all_data = []
        
        for region_name in engineered_data.keys():
            print(f"  Processing {region_name}...")
            
            # Features
            features_df = pd.DataFrame(engineered_data[region_name])
            features_df['timestamp'] = time_index
            features_df['region'] = region_name
            
            # Targets
            if region_name in targets:
                features_df['precipitation'] = targets[region_name]['rain_24h_continuous']
                features_df['rain_24h'] = targets[region_name]['rain_24h_binary']
            else:
                features_df['precipitation'] = np.nan
                features_df['rain_24h'] = -1
            
            all_data.append(features_df)
        
        # Concatenate all regions
        final_df = pd.concat(all_data, ignore_index=True)
        
        # Remove rows with NaN target
        final_df = final_df[final_df['rain_24h'] != -1]
        
        # Remove rows with too many NaN features
        threshold_nan = 0.5 * len(final_df.columns)
        final_df = final_df.dropna(thresh=threshold_nan)
        
        print(f"\n✅ Final DataFrame:")
        print(f"   Total rows: {len(final_df)}")
        print(f"   Total features: {len(final_df.columns) - 4}")  # Exclude timestamp, region, precipitation, rain_24h
        print(f"   Class distribution:")
        print(final_df['rain_24h'].value_counts())
        
        return final_df
    
    def save_processed_data(self, df):
        """Save processed data to CSV"""
        output_file = self.output_dir / 'peru_rainfall.csv'
        df.to_csv(output_file, index=False)
        print(f"\n✅ Data saved to: {output_file}")
        
        # Save statistics
        stats = {
            'n_samples': len(df),
            'n_features': len(df.columns) - 4,
            'n_regions': df['region'].nunique(),
            'time_range': {
                'start': str(df['timestamp'].min()),
                'end': str(df['timestamp'].max())
            },
            'class_distribution': df['rain_24h'].value_counts().to_dict(),
            'feature_names': [col for col in df.columns if col not in ['timestamp', 'region', 'precipitation', 'rain_24h']]
        }
        
        stats_file = self.output_dir / 'preprocessing_stats.json'
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"✅ Statistics saved to: {stats_file}")
        
        return output_file
    
    def run_pipeline(self, years):
        """Run complete preprocessing pipeline"""
        print("\n" + "="*80)
        print("STARTING PREPROCESSING PIPELINE")
        print("="*80)
        
        # Step 1: Extract ZIP files
        nc_files = self.extract_zip_files(years)
        
        if not nc_files:
            print("\n❌ No NetCDF files found. Please check your input directory.")
            return None
        
        # Step 2: Load NetCDF
        ds = self.load_netcdf_files(nc_files)
        
        # Step 3: Spatial aggregation
        regional_data = self.spatial_aggregation(ds)
        
        # Step 4: Feature engineering
        engineered_data = self.feature_engineering(regional_data)
        
        # Step 5: Create target
        targets = self.create_target_variable(regional_data)
        
        # Step 6: Create DataFrame
        df = self.create_dataframe(engineered_data, targets, ds.time.values)
        
        # Save
        output_file = self.save_processed_data(df)
        
        print("\n" + "="*80)
        print("PREPROCESSING COMPLETE")
        print("="*80)
        print(f"\nOutput file: {output_file}")
        print(f"Ready for Timer-XL training!")
        
        return output_file


def main():
    parser = argparse.ArgumentParser(description='Preprocess ERA5 data for Peru rainfall prediction')
    
    parser.add_argument('--input_dir', type=str, default='datasets/raw_era5',
                       help='Directory containing ERA5 .zip files')
    parser.add_argument('--output_dir', type=str, default='datasets/processed',
                       help='Directory to save processed data')
    parser.add_argument('--years', type=str, default='2023,2024',
                       help='Comma-separated list of years to process (e.g., 2023,2024)')
    parser.add_argument('--target_horizon', type=int, default=24,
                       help='Hours ahead to predict (default: 24)')
    parser.add_argument('--threshold', type=float, default=0.1,
                       help='Precipitation threshold in mm to define rain (default: 0.1)')
    
    args = parser.parse_args()
    
    # Parse years
    years = [int(y.strip()) for y in args.years.split(',')]
    
    # Create preprocessor
    preprocessor = ERA5PreprocessorPeru(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        target_horizon=args.target_horizon,
        threshold=args.threshold
    )
    
    # Run pipeline
    preprocessor.run_pipeline(years)


if __name__ == '__main__':
    main()
