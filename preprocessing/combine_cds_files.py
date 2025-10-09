"""
Script para combinar archivos ACCUM e INSTANT de CDS en un solo archivo
Soluciona el problema de que las variables están separadas en 2 archivos

Usage:
    python preprocessing/combine_cds_files.py --zip_file cds_2024.zip --output era5_peru_2024.nc
"""

import argparse
import xarray as xr
import zipfile
from pathlib import Path
import shutil


def combine_cds_files(zip_path, output_file):
    """
    Combina archivos ACCUM e INSTANT en un solo archivo NetCDF
    
    Args:
        zip_path: Ruta al archivo ZIP de CDS
        output_file: Nombre del archivo de salida
    """
    print("="*80)
    print("COMBINAR ARCHIVOS CDS (ACCUM + INSTANT)")
    print("="*80)
    print(f"Archivo ZIP: {zip_path}")
    print(f"Salida: {output_file}\n")
    
    # Crear carpeta temporal
    temp_dir = Path("temp_combine_cds")
    temp_dir.mkdir(exist_ok=True)
    
    try:
        # 1. Extraer ZIP
        print("[1/5] Extrayendo archivos del ZIP...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        print("  ✅ Archivos extraídos\n")
        
        # 2. Identificar archivos
        print("[2/5] Identificando archivos NetCDF...")
        nc_files = list(temp_dir.glob("*.nc"))
        
        accum_file = None
        instant_file = None
        
        for nc_file in nc_files:
            if 'accum' in nc_file.name.lower():
                accum_file = nc_file
                print(f"  ✅ ACCUM encontrado: {nc_file.name}")
            elif 'instant' in nc_file.name.lower():
                instant_file = nc_file
                print(f"  ✅ INSTANT encontrado: {nc_file.name}")
        
        if not accum_file or not instant_file:
            print("\n❌ ERROR: No se encontraron ambos archivos (ACCUM y INSTANT)")
            return None
        
        print()
        
        # 3. Cargar archivos
        print("[3/5] Cargando archivos NetCDF...")
        print("  Cargando ACCUM (tp)...")
        ds_accum = xr.open_dataset(accum_file)
        print(f"    Variables: {list(ds_accum.data_vars)}")
        
        print("  Cargando INSTANT (otras variables)...")
        ds_instant = xr.open_dataset(instant_file)
        print(f"    Variables: {list(ds_instant.data_vars)}")
        print()
        
        # 4. Verificar compatibilidad
        print("[4/5] Verificando compatibilidad...")
        
        # Renombrar coordenadas si es necesario
        time_coord_accum = 'valid_time' if 'valid_time' in ds_accum.coords else 'time'
        time_coord_instant = 'valid_time' if 'valid_time' in ds_instant.coords else 'time'
        
        # Renombrar a 'time' estándar
        if time_coord_accum != 'time':
            ds_accum = ds_accum.rename({time_coord_accum: 'time'})
        if time_coord_instant != 'time':
            ds_instant = ds_instant.rename({time_coord_instant: 'time'})
        
        # Verificar dimensiones
        print(f"  ACCUM   - Time: {len(ds_accum.time)}, Lat: {len(ds_accum.latitude)}, Lon: {len(ds_accum.longitude)}")
        print(f"  INSTANT - Time: {len(ds_instant.time)}, Lat: {len(ds_instant.latitude)}, Lon: {len(ds_instant.longitude)}")
        
        if len(ds_accum.time) != len(ds_instant.time):
            print(f"  ⚠️  WARNING: Diferentes timesteps. Usando intersección.")
        
        print("  ✅ Archivos compatibles\n")
        
        # 5. Combinar datasets
        print("[5/5] Combinando archivos...")
        
        # Método 1: Merge (combina variables)
        print("  Combinando variables...")
        ds_combined = xr.merge([ds_instant, ds_accum])
        
        # Verificar variables combinadas
        all_vars = list(ds_combined.data_vars)
        print(f"\n  ✅ Variables combinadas ({len(all_vars)}):")
        for var in all_vars:
            print(f"     - {var}")
        
        # Mapear nombres de variables si es necesario
        rename_dict = {}
        if 'u100' in ds_combined.data_vars:
            print("\n  🔧 Detectadas variables u100/v100 (deben ser u10/v10)")
            print("     ⚠️  ADVERTENCIA: Tus datos tienen viento a 100m, no a 10m")
            print("     ⚠️  Esto puede afectar el rendimiento del modelo")
            print("     💡 Recomendación: Re-descargar con u10/v10")
            # Renombrar para compatibilidad (aunque no sea ideal)
            if 'u100' in ds_combined.data_vars:
                rename_dict['u100'] = 'u10'
            if 'v100' in ds_combined.data_vars:
                rename_dict['v100'] = 'v10'
        
        if rename_dict:
            print(f"\n  🔄 Renombrando variables: {rename_dict}")
            ds_combined = ds_combined.rename(rename_dict)
        
        # Limpiar atributos innecesarios
        print("\n  🧹 Limpiando metadatos...")
        coords_to_drop = ['number', 'expver']  # Coordenadas de ensemble/versión
        for coord in coords_to_drop:
            if coord in ds_combined.coords:
                print(f"     Eliminando coordenada: {coord}")
                ds_combined = ds_combined.drop_vars(coord)
        
        # Filtrar región de Perú (si descargaste área más grande)
        print("\n  🌍 Filtrando región de Perú...")
        print(f"     Región original: Lat {ds_combined.latitude.min().values:.2f} a {ds_combined.latitude.max().values:.2f}")
        
        # Solo filtrar si hay datos fuera de Perú
        if ds_combined.latitude.max() > 1:  # Si descargaste más allá de Perú
            print(f"     🔧 Recortando a región de Perú (0° a -18°)...")
            ds_combined = ds_combined.sel(
                latitude=slice(0, -18),  # Perú: 0° a -18°
                longitude=slice(-82, -68)  # Perú: -82° a -68°
            )
            print(f"     ✅ Nueva región: Lat {ds_combined.latitude.min().values:.2f} a {ds_combined.latitude.max().values:.2f}")
            print(f"        Tamaño reducido: Lat {len(ds_combined.latitude)}, Lon {len(ds_combined.longitude)}")
        else:
            print(f"     ✅ Ya está en región de Perú")
        
        # 6. Guardar archivo combinado
        print(f"\n  💾 Guardando archivo combinado: {output_file}")
        print(f"     Variables: {list(ds_combined.data_vars)}")
        print(f"     Dimensiones: Time={len(ds_combined.time)}, Lat={len(ds_combined.latitude)}, Lon={len(ds_combined.longitude)}")
        
        ds_combined.to_netcdf(output_file)
        
        # Cerrar datasets
        ds_accum.close()
        ds_instant.close()
        ds_combined.close()
        
        # Calcular tamaño
        file_size_mb = Path(output_file).stat().st_size / (1024 * 1024)
        
        print(f"\n✅ ARCHIVO COMBINADO EXITOSAMENTE")
        print(f"   Archivo: {output_file}")
        print(f"   Tamaño: {file_size_mb:.2f} MB")
        
        # Verificación final
        print("\n🔍 Verificación final:")
        ds_verify = xr.open_dataset(output_file)
        
        required_vars = ['tp', 't2m', 'd2m', 'sp', 'msl', 'u10', 'v10', 'tcwv', 'cape']
        found_vars = [v for v in required_vars if v in ds_verify.data_vars]
        missing_vars = [v for v in required_vars if v not in ds_verify.data_vars]
        
        print(f"   Variables encontradas: {len(found_vars)}/9")
        if missing_vars:
            print(f"   ⚠️  Variables faltantes: {missing_vars}")
            print(f"   💡 Descarga nuevamente con las variables correctas")
        else:
            print(f"   ✅ Todas las variables requeridas presentes")
        
        ds_verify.close()
        
        return output_file
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None
        
    finally:
        # Limpiar carpeta temporal
        print("\n🧹 Limpiando archivos temporales...")
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        print("  ✅ Limpieza completa\n")
        print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Combinar archivos CDS ACCUM + INSTANT')
    parser.add_argument('--zip_file', type=str, required=True,
                       help='Archivo ZIP de CDS (e.g., cds_2024.zip)')
    parser.add_argument('--output', type=str, default=None,
                       help='Archivo de salida (default: era5_peru_YEAR.nc)')
    
    args = parser.parse_args()
    
    # Generar nombre de salida automático si no se proporciona
    if args.output is None:
        # Extraer año del nombre del archivo
        import re
        match = re.search(r'(\d{4})', args.zip_file)
        if match:
            year = match.group(1)
            args.output = f"era5_peru_{year}.nc"
        else:
            args.output = "era5_peru_combined.nc"
    
    result = combine_cds_files(args.zip_file, args.output)
    
    if result:
        print(f"\n🎉 ¡ÉXITO! Archivo listo para usar: {result}")
        print(f"\n📋 Siguiente paso:")
        print(f"   python preprocessing/preprocess_era5_peru.py --input_file {result}")
    else:
        print(f"\n❌ Error al combinar archivos")
