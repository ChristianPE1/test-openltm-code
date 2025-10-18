"""
Quick validation script for regression data after bug fix
Run this BEFORE training to confirm data is correct

Usage:
    python validate_regression_data.py --data_path datasets/processed/peru_rainfall_regression_cleaned.csv
"""

import argparse
import pandas as pd
import numpy as np
import sys


def validate_regression_data(data_path):
    """Validate that regression data has realistic precipitation values"""
    
    print("="*80)
    print("🔍 VALIDACIÓN DE DATOS DE REGRESIÓN")
    print("="*80)
    print(f"\n📂 Archivo: {data_path}\n")
    
    # Load data
    try:
        df = pd.read_csv(data_path)
        print(f"✅ Datos cargados: {df.shape}")
    except Exception as e:
        print(f"❌ ERROR al cargar datos: {e}")
        return False
    
    # Check target column exists
    if 'target_precip_24h' not in df.columns:
        print("❌ ERROR: Columna 'target_precip_24h' no encontrada")
        print(f"   Columnas disponibles: {df.columns.tolist()}")
        return False
    
    target = df['target_precip_24h']
    
    # Basic statistics
    print("\n📊 ESTADÍSTICAS BÁSICAS:")
    print(f"   Samples: {len(target)}")
    print(f"   Min: {target.min():.3f} mm")
    print(f"   Max: {target.max():.3f} mm")
    print(f"   Mean: {target.mean():.3f} mm")
    print(f"   Median: {target.median():.3f} mm")
    print(f"   Std: {target.std():.3f} mm")
    print(f"   25%: {target.quantile(0.25):.3f} mm")
    print(f"   75%: {target.quantile(0.75):.3f} mm")
    print(f"   95%: {target.quantile(0.95):.3f} mm")
    print(f"   99%: {target.quantile(0.99):.3f} mm")
    
    # Rainfall distribution
    rainy_pct = 100 * (target > 0.1).sum() / len(target)
    heavy_pct = 100 * (target > 10.0).sum() / len(target)
    extreme_pct = 100 * (target > 50.0).sum() / len(target)
    
    print("\n📊 DISTRIBUCIÓN DE LLUVIA:")
    print(f"   Rainy (>0.1mm): {rainy_pct:.1f}%")
    print(f"   Heavy (>10mm): {heavy_pct:.2f}%")
    print(f"   Extreme (>50mm): {extreme_pct:.2f}%")
    
    # Check for NaN
    nan_count = target.isna().sum()
    print(f"\n🔍 VALORES FALTANTES:")
    print(f"   NaN count: {nan_count} ({100*nan_count/len(target):.2f}%)")
    
    # VALIDATION CHECKS
    print("\n" + "="*80)
    print("✅ VERIFICACIÓN DE CORRECCIONES")
    print("="*80)
    
    all_passed = True
    
    # CHECK 1: Max value should be realistic (>20mm)
    print("\n[1/6] Verificando valor máximo...")
    if target.max() > 20:
        print(f"   ✅ PASS: Max = {target.max():.1f} mm (realista)")
    else:
        print(f"   ❌ FAIL: Max = {target.max():.1f} mm (muy bajo, bug no corregido)")
        print(f"        Esperado: >20mm (eventos de lluvia intensa)")
        all_passed = False
    
    # CHECK 2: Mean should be reasonable (1-10mm)
    print("\n[2/6] Verificando valor medio...")
    if 1.0 < target.mean() < 15.0:
        print(f"   ✅ PASS: Mean = {target.mean():.1f} mm (realista)")
    else:
        print(f"   ⚠️  WARNING: Mean = {target.mean():.1f} mm")
        if target.mean() < 1.0:
            print(f"        Muy bajo - posible bug en acumulación")
            all_passed = False
        else:
            print(f"        Alto pero posible para región con mucha lluvia")
    
    # CHECK 3: Should have heavy rain events (>10mm)
    print("\n[3/6] Verificando eventos de lluvia intensa...")
    if heavy_pct > 1.0:
        print(f"   ✅ PASS: Heavy rain = {heavy_pct:.2f}% (eventos capturados)")
    else:
        print(f"   ❌ FAIL: Heavy rain = {heavy_pct:.2f}% (muy bajo)")
        print(f"        Costa de Perú tiene eventos ENSO con >10mm")
        all_passed = False
    
    # CHECK 4: Should have some extreme events (>50mm) for El Niño
    print("\n[4/6] Verificando eventos extremos ENSO...")
    if extreme_pct > 0.1:
        print(f"   ✅ PASS: Extreme rain = {extreme_pct:.2f}% (eventos ENSO capturados)")
    elif extreme_pct > 0:
        print(f"   ⚠️  WARNING: Extreme rain = {extreme_pct:.2f}% (bajo pero presente)")
    else:
        print(f"   ⚠️  WARNING: Extreme rain = 0% (sin eventos >50mm)")
        print(f"        Puede ser normal si datos no incluyen El Niño fuerte")
    
    # CHECK 5: No NaN values
    print("\n[5/6] Verificando valores faltantes...")
    if nan_count == 0:
        print(f"   ✅ PASS: No NaN values")
    else:
        print(f"   ❌ FAIL: {nan_count} NaN values ({100*nan_count/len(target):.2f}%)")
        print(f"        Ejecutar clean_regression_data.py")
        all_passed = False
    
    # CHECK 6: Data is continuous (not binary)
    print("\n[6/6] Verificando que datos son continuos...")
    unique_values = target.nunique()
    if unique_values > 100:
        print(f"   ✅ PASS: {unique_values} valores únicos (datos continuos)")
    else:
        print(f"   ❌ FAIL: {unique_values} valores únicos (datos binarios/categóricos)")
        print(f"        Datos de regresión deben tener muchos valores únicos")
        all_passed = False
    
    # FINAL VERDICT
    print("\n" + "="*80)
    if all_passed:
        print("🎉 ¡VALIDACIÓN EXITOSA!")
        print("="*80)
        print("\n✅ Los datos están correctos y listos para entrenamiento")
        print("\n📊 MÉTRICAS ESPERADAS:")
        print("   • RMSE: 2-5 mm (NASA IMERG: 3.4mm)")
        print("   • MAE: 1.5-3.5 mm")
        print("   • R²: 0.40-0.70")
        print("\n💡 PRÓXIMO PASO:")
        print("   Ejecutar celda 'Entrenamiento Regresión' en el notebook")
        print("="*80)
        return True
    else:
        print("❌ VALIDACIÓN FALLIDA")
        print("="*80)
        print("\n⚠️  Los datos tienen problemas. Posibles causas:")
        print("   1. Bug en preprocess_era5_regression.py no corregido")
        print("   2. Datos de entrada (ERA5 .nc) corruptos")
        print("   3. Limpieza de datos (clean_regression_data.py) no ejecutada")
        print("\n🔧 SOLUCIÓN:")
        print("   1. Verificar que archivos .py tienen correcciones (ver CORRECCION_REGRESION_2025_01_18.md)")
        print("   2. Re-ejecutar preprocessing:")
        print("      !python preprocessing/preprocess_era5_regression.py ...")
        print("   3. Re-ejecutar limpieza:")
        print("      !python preprocessing/clean_regression_data.py ...")
        print("   4. Re-ejecutar este script de validación")
        print("="*80)
        return False


def main():
    parser = argparse.ArgumentParser(description='Validate regression data after bug fix')
    parser.add_argument('--data_path', type=str, 
                        default='datasets/processed/peru_rainfall_regression_cleaned.csv',
                        help='Path to cleaned regression CSV')
    
    args = parser.parse_args()
    
    success = validate_regression_data(args.data_path)
    
    # Exit code (for automation)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
