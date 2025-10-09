"""
Script simple para verificar contenido de archivos CDS (sin dependencias externas)
Solo verifica contenido del ZIP sin procesar NetCDF

Usage:
    python preprocessing/test_cds_simple.py cds_2023.zip
"""

import sys
import zipfile
from pathlib import Path


def test_cds_zip_simple(zip_path):
    """
    Prueba simple de archivo ZIP de CDS
    No requiere xarray/pandas, solo zipfile
    """
    print("="*80)
    print("TEST SIMPLE DE ARCHIVO CDS")
    print("="*80)
    print(f"Archivo: {zip_path}\n")
    
    if not Path(zip_path).exists():
        print(f"❌ ERROR: Archivo no encontrado: {zip_path}")
        return
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            
            print(f"📦 Contenido del ZIP ({len(file_list)} archivos):")
            print("-" * 80)
            
            nc_files = []
            accum_file = None
            instant_file = None
            
            for f in file_list:
                file_info = zip_ref.getinfo(f)
                size_mb = file_info.file_size / (1024 * 1024)
                
                print(f"  📁 {f}")
                print(f"     Tamaño: {size_mb:.2f} MB")
                
                if f.endswith('.nc'):
                    nc_files.append(f)
                    if 'accum' in f.lower():
                        accum_file = f
                    elif 'instant' in f.lower():
                        instant_file = f
                print()
            
            # Análisis
            print("="*80)
            print("ANÁLISIS")
            print("="*80)
            
            if len(nc_files) == 0:
                print("❌ No se encontraron archivos .nc")
                print("   Tu archivo ZIP no contiene datos NetCDF")
                return
            
            print(f"✅ Archivos NetCDF encontrados: {len(nc_files)}\n")
            
            # Identificar archivos
            if accum_file:
                print(f"✅ ARCHIVO ACCUM ENCONTRADO: {accum_file}")
                print(f"   Este archivo contiene 'tp' (precipitación)")
                print(f"   ✅ USA ESTE ARCHIVO para entrenar\n")
            else:
                print("⚠️  No se encontró archivo 'accum'")
                print("   Buscando archivo con nombre diferente...\n")
                for nc in nc_files:
                    print(f"   📄 {nc}")
                print()
            
            if instant_file:
                print(f"ℹ️  ARCHIVO INSTANT ENCONTRADO: {instant_file}")
                print(f"   Este archivo contiene variables instantáneas")
                print(f"   ❌ NO lo necesitas (no tiene precipitación)\n")
            
            # Resumen
            print("="*80)
            print("RESUMEN Y RECOMENDACIÓN")
            print("="*80)
            
            if accum_file:
                print(f"✅ Tu archivo CDS es COMPATIBLE")
                print(f"✅ Archivo a usar: {accum_file}")
                print(f"\n📋 Próximos pasos:")
                print(f"   1. Renombrar (opcional): {Path(zip_path).name} → era5_peru_YYYY.zip")
                print(f"   2. Mover a: AdaptationOpenLTM/datasets/raw_era5/")
                print(f"   3. Ejecutar preprocesamiento")
                print(f"\n💡 El código automáticamente extraerá y usará el archivo 'accum'")
            else:
                print(f"⚠️  VERIFICACIÓN NECESARIA")
                print(f"   No se identificó automáticamente el archivo 'accum'")
                print(f"   Archivos disponibles: {nc_files}")
                print(f"\n   Necesitas verificar manualmente cuál contiene 'tp'")
                print(f"   (Requiere instalar xarray para análisis completo)")
            
            print("\n" + "="*80)
            
    except Exception as e:
        print(f"❌ ERROR al leer archivo: {e}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_cds_simple.py <archivo.zip>")
        print("Ejemplo: python test_cds_simple.py cds_2023.zip")
        sys.exit(1)
    
    zip_path = sys.argv[1]
    test_cds_zip_simple(zip_path)
