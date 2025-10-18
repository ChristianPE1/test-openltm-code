# 🧹 Limpieza de Datos para REGRESIÓN vs CLASIFICACIÓN

## ¿Por qué es DIFERENTE?

### TL;DR
- **Clasificación**: Puede tolerar NaN (rellenar con median/0)
- **Regresión**: Muy sensible a NaN (causa problemas en MSE/MAE)

---

## 📊 Comparación de Estrategias

| Aspecto | Clasificación | Regresión |
|---------|---------------|-----------|
| **NaN Handling** | Forward fill + Median | Interpolación temporal + Remoción agresiva |
| **Outliers** | Menos crítico | CRÍTICO (detectar valores imposibles) |
| **Missing Data** | Tolera hasta 10% | Tolera hasta 5% |
| **Interpolación** | Simple (ffill/bfill) | Temporal (time-aware linear) |
| **Validación** | Chequeo básico | Exhaustiva (unique values, range, distribution) |

---

## 🔍 Diferencias Técnicas

### 1️⃣ **Manejo de NaN**

#### CLASIFICACIÓN (`clean_classification_data.py`)
```python
# Estrategia: Forward fill + Median (simple)
df[feature_cols] = df[feature_cols].fillna(method='ffill')
df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())
```

**Problema para regresión**: Median global puede distorsionar patrones temporales

#### REGRESIÓN (`clean_regression_data.py`)
```python
# Estrategia: Interpolación temporal por región
for region in df['region'].unique():
    region_df = df[df['region'] == region]
    
    # Interpolación lineal (time-aware)
    region_df[feature_cols] = region_df[feature_cols].interpolate(
        method='linear',
        limit_direction='both',
        limit=5  # Máximo 5 pasos (2.5 días)
    )
    
    # Forward/Backward fill (para gaps pequeños)
    region_df[feature_cols] = region_df[feature_cols].fillna(method='ffill', limit=3)
    region_df[feature_cols] = region_df[feature_cols].fillna(method='bfill', limit=3)

# Remoción agresiva (si aún quedan NaN)
df = df.dropna(subset=feature_cols)
```

**Por qué mejor**: Preserva patrones temporales, respeta estructura regional

---

### 2️⃣ **Detección de Outliers**

#### CLASIFICACIÓN
No tiene detección específica (binario 0/1 no tiene outliers)

#### REGRESIÓN
```python
# Remover valores negativos (imposibles)
df = df[df['target_precip_24h'] >= 0]

# Remover valores extremos (> 200mm/día)
# 200mm = evento El Niño extremo (límite razonable)
df = df[df['target_precip_24h'] <= 200.0]
```

**Por qué crítico**: Valores imposibles (negativos) o errores de medición (>500mm) distorsionan MSE

---

### 3️⃣ **Validación de Calidad**

#### CLASIFICACIÓN
```python
# Validación básica
assert df['target'].nunique() == 2  # Binary check
assert df['target'].min() == 0
assert df['target'].max() == 1
```

#### REGRESIÓN
```python
# Validación exhaustiva
unique_count = df['target_precip_24h'].nunique()

# CRÍTICO: Detectar si datos son binarios (ERROR)
if unique_count < 100:
    raise ValueError("Data appears to be binary, not continuous!")

# Verificar rango razonable
assert df['target_precip_24h'].min() >= 0, "Negative precipitation!"
assert df['target_precip_24h'].max() < 200, "Extreme outlier!"

# Verificar distribución
rainy_pct = (df['target_precip_24h'] > 0.1).sum() / len(df)
assert 0.10 < rainy_pct < 0.40, f"Unusual rainy percentage: {rainy_pct}"
```

**Por qué exhaustiva**: Un error en los datos (binario vs continuo) invalida TODO el entrenamiento

---

## 🚨 Casos Problemáticos

### PROBLEMA 1: Usar CSV de Clasificación para Regresión

```python
# ❌ MAL: Usar peru_rainfall_cleaned.csv (binario)
df = pd.read_csv('peru_rainfall_cleaned.csv')
print(df['rain_24h'].unique())  # [0, 1]  ← BINARIO

# Entrenar "regresión" en datos binarios
# RMSE = 0.45 ← No tiene sentido comparar con NASA IMERG (3.4mm)
```

**Solución**: Usar `peru_rainfall_regression_cleaned.csv` (continuo)

---

### PROBLEMA 2: NaN en Features

```python
# Ejemplo: Temperatura tiene NaN en 3 días consecutivos

# ❌ MAL (Clasificación): Rellenar con median
df['temperature'].fillna(df['temperature'].median())
# Problema: 3 días con EXACTAMENTE el mismo valor (irreal)

# ✅ BIEN (Regresión): Interpolación temporal
df['temperature'].interpolate(method='linear', limit=5)
# Resultado: 20.1 → 20.3 → 20.5 → 20.7 (gradiente natural)
```

---

### PROBLEMA 3: Outliers Extremos

```python
# Caso real: Error en sensor reportó 15000 mm (15 metros) de lluvia

# ❌ Clasificación: No lo detecta (se binariza a 1 de todas formas)

# ✅ Regresión: Detecta y elimina
if df['target_precip_24h'].max() > 200:
    print(f"Outlier detectado: {df['target_precip_24h'].max()} mm")
    df = df[df['target_precip_24h'] <= 200]
```

---

## 📝 Pipeline Completo: REGRESIÓN

```bash
# PASO 1: Preprocesar desde .nc (METERS → MM)
python preprocessing/preprocess_era5_regression.py \
    --input_dir datasets/raw_era5 \
    --output_dir datasets/processed \
    --years 2014,2015,...,2024

# Output: peru_rainfall_regression.csv
# Contenido: valores continuos [0.0, 100.0] mm

# PASO 2: Limpiar datos (NaN, outliers, validación)
python preprocessing/clean_regression_data.py \
    --input_path datasets/processed/peru_rainfall_regression.csv \
    --output_path datasets/processed/peru_rainfall_regression_cleaned.csv \
    --max_precip 200.0

# Output: peru_rainfall_regression_cleaned.csv
# Garantías:
#   ✅ Sin NaN en features
#   ✅ Sin outliers extremos
#   ✅ Valores continuos (unique > 100)
#   ✅ Rango válido [0, 200] mm

# PASO 3: Entrenar Timer-XL
python run.py \
    --task_name long_term_forecast \
    --data_path peru_rainfall_regression_cleaned.csv \
    --loss MSE \
    ...
```

---

## 🎯 Checklist de Validación

Antes de entrenar regresión, verificar:

```python
import pandas as pd

df = pd.read_csv('datasets/processed/peru_rainfall_regression_cleaned.csv')

# ✅ Check 1: Sin NaN
assert df.isnull().sum().sum() == 0, "Found NaN values!"

# ✅ Check 2: Valores continuos (NO binarios)
unique_count = df['target_precip_24h'].nunique()
assert unique_count > 100, f"Only {unique_count} unique values (binary?)"

# ✅ Check 3: Rango válido
assert df['target_precip_24h'].min() >= 0, "Negative precipitation!"
assert df['target_precip_24h'].max() <= 200, "Extreme outlier!"

# ✅ Check 4: Distribución razonable
mean_precip = df['target_precip_24h'].mean()
assert 2.0 < mean_precip < 6.0, f"Unusual mean: {mean_precip}"

rainy_pct = (df['target_precip_24h'] > 0.1).sum() / len(df) * 100
assert 15 < rainy_pct < 35, f"Unusual rainy %: {rainy_pct}"

print("✅ Todos los checks pasados - listo para regresión")
```

---

## 📊 Estadísticas Esperadas

### Clasificación (Binary)
```json
{
  "target_column": "rain_24h",
  "unique_values": 2,
  "values": [0, 1],
  "class_0_pct": 71.3,
  "class_1_pct": 28.7
}
```

### Regresión (Continuous)
```json
{
  "target_column": "target_precip_24h",
  "unique_values": 2847,
  "range": [0.0, 87.3],
  "mean": 3.54,
  "median": 0.82,
  "std": 6.21,
  "rainy_pct": 24.6,
  "heavy_rain_pct": 8.3,
  "extreme_rain_pct": 1.2
}
```

**Diferencia clave**: 
- Clasificación: **2 valores únicos** (binario)
- Regresión: **2847 valores únicos** (continuo)

---

## 💡 Recomendaciones Finales

### Para tu Tesis

1. **Documentar diferencias**: Explica en Capítulo 3 (Metodología) por qué la limpieza es diferente

2. **Justificar umbral 200mm**: 
   - Citar eventos El Niño históricos
   - Mostrar que > 200mm son errores de sensor (no eventos reales)

3. **Validar interpolación**:
   - Comparar RMSE con/sin interpolación
   - Mostrar que interpolación mejora resultados (preserva patrones)

4. **Tabla comparativa**:
   ```markdown
   | Aspecto | Clasificación | Regresión | Justificación |
   |---------|---------------|-----------|---------------|
   | NaN handling | Median fill | Temporal interpolation | Preserva patrones |
   | Outliers | No aplica | Detección activa | Evita distorsión MSE |
   | Target | Binary {0,1} | Continuous [0,200]mm | Domain-aligned |
   ```

---

## 🔗 Referencias

- **ERA5 Data Quality**: ECMWF (2023) - Precipitation validation
- **Interpolation Methods**: Pandas documentation - Time series interpolation
- **Outlier Detection**: IQR method + domain knowledge (200mm threshold)
- **Literature Benchmarks**: 
  - NASA IMERG: RMSE ~3.4 mm (after quality control)
  - ERA5-Land: MAE ~2.5 mm (with outlier removal)

---

## ✅ Resumen Ejecutivo

**CLASIFICACIÓN**:
- Datos binarios (0/1)
- Limpieza simple (forward fill + median)
- Tolera NaN moderado
- No requiere detección de outliers

**REGRESIÓN**:
- Datos continuos (0-200 mm)
- Limpieza sofisticada (interpolación temporal)
- Muy sensible a NaN (remoción agresiva)
- Requiere validación exhaustiva (unique values, range, distribution)
- Detección de outliers CRÍTICA

**CRÍTICO**: Nunca usar CSV de clasificación para regresión → datos binarios hacen RMSE no comparable con literatura.
