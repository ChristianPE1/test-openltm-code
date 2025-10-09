"""
Script de prueba para verificar archivos CDS descargados
Verifica que tus archivos cds_2023.zip y cds_2024.zip sean compatibles

Usage:
    python preprocessing/test_cds_files.py --zip_file cds_2023.zip
"""

import os
import argparse
import xarray as xr
import zipfile
from pathlib import Path
import pandas as pd


def test_cds_zip(zip_path):
    """
    Prueba un archivo ZIP de CDS y verifica compatibilidad
    """
    print("="*80)
    print("TEST DE ARCHIVO CDS")
    print("="*80)
    print(f"Archivo: {zip_path}")
    print()
    
    if not os.path.exists(zip_path):
        print(f"❌ ERROR: Archivo no encontrado: {zip_path}")
        return
    
    # 1. Listar contenido del ZIP
    print("[1/5] Contenido del ZIP:")
    print("-" * 80)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        file_list = zip_ref.namelist()
        for f in file_list:
            file_info = zip_ref.getinfo(f)
            size_mb = file_info.file_size / (1024 * 1024)
            print(f"  📁 {f}")
            print(f"     Tamaño: {size_mb:.2f} MB")
        print()
    
    # 2. Identificar archivos .nc
    nc_files = [f for f in file_list if f.endswith('.nc')]
    print(f"[2/5] Archivos NetCDF encontrados: {len(nc_files)}")
    print("-" * 80)
    for nc_file in nc_files:
        print(f"  📄 {nc_file}")
    print()
    
    if len(nc_files) == 0:
        print("❌ ERROR: No se encontraron archivos .nc en el ZIP")
        return
    
    # 3. Extraer temporalmente para análisis
    temp_dir = Path("temp_test_cds")
    temp_dir.mkdir(exist_ok=True)
    
    print(f"[3/5] Extrayendo archivos a carpeta temporal: {temp_dir}")
    print("-" * 80)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    print("  ✅ Archivos extraídos\n")
    
    # 4. Analizar cada archivo .nc
    print("[4/5] Análisis de archivos NetCDF:")
    print("-" * 80)
    
    compatible_files = {}
    
    for nc_file in nc_files:
        nc_path = temp_dir / nc_file
        
        print(f"\n📊 Analizando: {nc_file}")
        print("  " + "-" * 76)
        
        try:
            ds = xr.open_dataset(nc_path)
            
            # Info básica
            print(f"  ✅ Archivo cargado exitosamente")
            print(f"  📏 Dimensiones: {dict(ds.dims)}")
            print(f"  📍 Coordenadas: {list(ds.coords)}")
            print(f"  📊 Variables de datos: {list(ds.data_vars)}")
            
            # Variables que necesitamos
            required_vars = {
                'tp': 'Total Precipitation',
                't2m': 'Temperature at 2m',
                'd2m': 'Dewpoint at 2m',
                'sp': 'Surface Pressure',
                'msl': 'Mean Sea Level Pressure',
                'u10': 'U-component of wind at 10m',
                'v10': 'V-component of wind at 10m',
                'tcwv': 'Total Column Water Vapour',
                'cape': 'Convective Available Potential Energy'
            }
            
            # Verificar variables disponibles
            print(f"\n  🔍 Variables requeridas:")
            found_vars = []
            missing_vars = []
            
            for var_code, var_name in required_vars.items():
                if var_code in ds.data_vars or var_code in ds.coords:
                    print(f"     ✅ {var_code:6s} - {var_name}")
                    found_vars.append(var_code)
                else:
                    print(f"     ❌ {var_code:6s} - {var_name} (NO ENCONTRADA)")
                    missing_vars.append(var_code)
            
            print(f"\n  📊 Resumen: {len(found_vars)}/9 variables encontradas")
            
            # Analizar dimensión temporal
            if 'time' in ds.coords:
                times = pd.to_datetime(ds.time.values)
                print(f"\n  🕐 Información temporal:")
                print(f"     Inicio:     {times[0]}")
                print(f"     Fin:        {times[-1]}")
                print(f"     Timesteps:  {len(times)}")
                
                # Detectar frecuencia
                if len(times) > 1:
                    time_diffs = pd.Series(times).diff().dropna()
                    freq_hours = time_diffs.dt.total_seconds().mode()[0] / 3600
                    print(f"     Frecuencia: {freq_hours:.1f} horas")
                    
                    # Contar horas únicas
                    unique_hours = sorted(set(times.hour))
                    print(f"     Horas UTC:  {unique_hours}")
                    
                    # Verificar si tenemos 06 y 18 UTC
                    has_06 = 6 in unique_hours
                    has_18 = 18 in unique_hours
                    
                    if has_06 and has_18:
                        print(f"     ✅ Contiene horas 06 y 18 UTC (necesarias)")
                    else:
                        print(f"     ⚠️  Horas disponibles: {unique_hours}")
                        if not has_06:
                            print(f"     ⚠️  Falta hora 06 UTC")
                        if not has_18:
                            print(f"     ⚠️  Falta hora 18 UTC")
            
            # Analizar dimensión espacial
            lat_name = 'latitude' if 'latitude' in ds.coords else 'lat'
            lon_name = 'longitude' if 'longitude' in ds.coords else 'lon'
            
            if lat_name in ds.coords and lon_name in ds.coords:
                lats = ds[lat_name].values
                lons = ds[lon_name].values
                print(f"\n  🌍 Información espacial:")
                print(f"     Latitud:  {lats.min():.2f}° a {lats.max():.2f}° ({len(lats)} puntos)")
                print(f"     Longitud: {lons.min():.2f}° a {lons.max():.2f}° ({len(lons)} puntos)")
                print(f"     Resolución: ~{abs(lats[1] - lats[0]):.3f}°")
                
                # Verificar si cubre Perú (aprox: 0° a -18° lat, -82° a -68° lon)
                covers_peru = (lats.min() <= 0 and lats.max() >= -18 and 
                              lons.min() <= -68 and lons.max() >= -82)
                if covers_peru:
                    print(f"     ✅ Cubre región de Perú")
                else:
                    print(f"     ⚠️  Podría NO cubrir completamente Perú")
            
            # Mostrar sample de datos (primera variable con datos)
            if len(found_vars) > 0:
                sample_var = found_vars[0]
                print(f"\n  📈 Muestra de datos ({sample_var}):")
                sample_data = ds[sample_var].values
                print(f"     Shape:  {sample_data.shape}")
                print(f"     Min:    {sample_data.min():.6f}")
                print(f"     Max:    {sample_data.max():.6f}")
                print(f"     Mean:   {sample_data.mean():.6f}")
                print(f"     Std:    {sample_data.std():.6f}")
                print(f"     NaNs:   {pd.isna(sample_data).sum()} ({pd.isna(sample_data).sum()/sample_data.size*100:.2f}%)")
            
            # Determinar compatibilidad
            is_compatible = (
                len(found_vars) >= 7 and  # Al menos 7/9 variables
                'time' in ds.coords and
                (lat_name in ds.coords) and
                (lon_name in ds.coords)
            )
            
            if is_compatible:
                print(f"\n  ✅ ARCHIVO COMPATIBLE")
                compatible_files[nc_file] = {
                    'found_vars': found_vars,
                    'missing_vars': missing_vars,
                    'timesteps': len(ds.time) if 'time' in ds.coords else 0
                }
            else:
                print(f"\n  ❌ ARCHIVO NO COMPATIBLE")
                print(f"     Razón: Faltan variables o dimensiones críticas")
            
            ds.close()
            
        except Exception as e:
            print(f"  ❌ ERROR al leer archivo: {e}")
    
    # 5. Recomendaciones
    print("\n" + "="*80)
    print("[5/5] RECOMENDACIONES")
    print("="*80)
    
    if len(compatible_files) == 0:
        print("❌ NINGÚN ARCHIVO ES COMPATIBLE")
        print("\n🔧 Acciones recomendadas:")
        print("  1. Verifica que descargaste las variables correctas de CDS")
        print("  2. Revisa la guía de descarga actualizada")
        print("  3. Si necesitas re-descargar, usa las variables listadas abajo")
    
    elif len(compatible_files) == 1:
        print("✅ ENCONTRADO 1 ARCHIVO COMPATIBLE")
        compatible_file = list(compatible_files.keys())[0]
        print(f"\n📄 Archivo a usar: {compatible_file}")
        
        info = compatible_files[compatible_file]
        if len(info['missing_vars']) > 0:
            print(f"\n⚠️  Variables faltantes: {', '.join(info['missing_vars'])}")
            print(f"   El modelo funcionará, pero con menos features")
        
        # Determinar si es accum o instant
        if 'accum' in compatible_file.lower():
            print(f"\n📊 Tipo: ACCUMULATED (acumulado)")
            print(f"   ✅ Este archivo contiene 'tp' (precipitación acumulada)")
            print(f"   ✅ Úsalo para entrenar el modelo")
        elif 'instant' in compatible_file.lower():
            print(f"\n📊 Tipo: INSTANTANEOUS (instantáneo)")
            print(f"   ⚠️  Este archivo NO contiene 'tp' (precipitación)")
            print(f"   ❌ NO usar para entrenamiento (falta variable objetivo)")
    
    elif len(compatible_files) == 2:
        print("✅ ENCONTRADOS 2 ARCHIVOS COMPATIBLES")
        print("\n🔍 Análisis:")
        
        accum_file = None
        instant_file = None
        
        for filename in compatible_files.keys():
            if 'accum' in filename.lower():
                accum_file = filename
                print(f"  📊 {filename} → ACCUMULATED (acumulado)")
            elif 'instant' in filename.lower():
                instant_file = filename
                print(f"  📊 {filename} → INSTANTANEOUS (instantáneo)")
        
        print("\n💡 RECOMENDACIÓN:")
        if accum_file and instant_file:
            print(f"  ✅ Usa: {accum_file}")
            print(f"     Razón: Contiene 'tp' (precipitación acumulada)")
            print(f"     Este es el archivo que necesitas para predecir lluvia")
            print()
            print(f"  ❌ NO uses: {instant_file}")
            print(f"     Razón: Valores instantáneos, no incluye precipitación")
        else:
            print(f"  ✅ Puedes combinar ambos archivos para más variables")
            print(f"     Pero probablemente solo necesites el archivo 'accum'")
    
    # Resumen de variables necesarias
    print("\n" + "="*80)
    print("📋 VARIABLES REQUERIDAS PARA TIMER-XL")
    print("="*80)
    print("""
Variables OBLIGATORIAS (9 en total):
  1. tp    - Total Precipitation (ACUMULADA, no instantánea)
  2. t2m   - 2m Temperature
  3. d2m   - 2m Dewpoint Temperature
  4. sp    - Surface Pressure
  5. msl   - Mean Sea Level Pressure
  6. u10   - 10m U Wind Component
  7. v10   - 10m V Wind Component
  8. tcwv  - Total Column Water Vapour
  9. cape  - Convective Available Potential Energy

Configuración temporal recomendada:
  - Horas: 06:00 y 18:00 UTC (12-hourly)
  - Si descargaste 00, 06, 12, 18 → Filtraremos solo 06 y 18

Configuración espacial:
  - Área: Perú (0°N, -82°W, -18°S, -68°W)
  - Resolución: 0.25° (default ERA5)
""")
    
    # Limpiar carpeta temporal
    print("\n🧹 Limpiando archivos temporales...")
    import shutil
    shutil.rmtree(temp_dir)
    print("  ✅ Limpieza completa")
    
    print("\n" + "="*80)
    print("TEST COMPLETADO")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test CDS ZIP files')
    parser.add_argument('--zip_file', type=str, required=True,
                       help='Path to CDS ZIP file (e.g., cds_2023.zip)')
    
    args = parser.parse_args()
    
    test_cds_zip(args.zip_file)
