# 🔄 FLUJO COMPLETO: Clasificación vs Regresión

## 📊 Comparación Visual

```mermaid
flowchart TD
    A[Raw ERA5 .nc files] --> B{Tarea?}
    
    B -->|CLASIFICACIÓN| C1[preprocess_era5_peru.py]
    B -->|REGRESIÓN| C2[preprocess_era5_regression.py]
    
    C1 --> D1[peru_rainfall.csv]
    C2 --> D2[peru_rainfall_regression.csv]
    
    D1 --> E1[Limpieza Clasificación]
    D2 --> E2[Limpieza Regresión]
    
    E1 --> F1[peru_rainfall_cleaned.csv]
    E2 --> F2[peru_rainfall_regression_cleaned.csv]
    
    F1 --> G1[Target: rain_24h = {0, 1}]
    F2 --> G2[Target: target_precip_24h = [0, 200] mm]
    
    G1 --> H1[Timer-XL Classifier]
    G2 --> H2[Timer-XL Regressor]
    
    H1 --> I1[Métricas: F1, Precision, Recall]
    H2 --> I2[Métricas: RMSE, MAE, R²]
```

---

## 🔍 PASO a PASO

### 📁 **Input**: Archivos .nc (idénticos para ambos)

```
datasets/raw_era5/
├── era5_peru_2014.nc
├── era5_peru_2015.nc
...
└── era5_peru_2024.nc

Variables en .nc:
- tp (total_precipitation) → METROS
- t2m, d2m, sp, msl, u10, v10, tcwv, cape
```

---

## 🔀 CLASIFICACIÓN

### PASO 1: Preprocesamiento
```bash
python preprocessing/preprocess_era5_peru.py \
    --input_dir datasets/raw_era5 \
    --years 2014,...,2024 \
    --threshold 0.0001  # ← BINARIZACIÓN
```

**Transformaciones**:
1. Leer .nc files
2. Agregar espacialmente (por región)
3. **Convertir METROS → MM** (×1000)
4. **BINARIZAR**: `rain_24h = 1 if precip > 0.1mm else 0`
5. Features engineering (lags, rolling stats)

**Output**: `peru_rainfall.csv`
```python
{
  'timestamp': '2014-01-01',
  'region': 'costa_norte',
  'total_precipitation': 2.5,  # mm (convertido)
  'temperature_2m': 25.3,
  'rain_24h': 1  # ← BINARIO {0, 1}
}
```

---

### PASO 2: Limpieza (Clasificación)
```bash
# ⚠️ Actualmente NO existe script de limpieza para clasificación
# Se asume que preprocess_era5_peru.py ya hace limpieza básica
```

**Estrategia** (si se implementara):
- Forward fill para NaN temporales
- Median fill para NaN persistentes
- Sin detección de outliers (binario 0/1)

---

### PASO 3: Entrenamiento
```bash
python run.py \
    --task_name classification \
    --data_path peru_rainfall_cleaned.csv \
    --loss CE \
    --use_focal_loss \
    --n_classes 2
```

**Métricas**:
- F1-Score: 83.24%
- Recall Rain: 83%
- Recall No Rain: 71%

---

## 🌊 REGRESIÓN

### PASO 1: Preprocesamiento
```bash
python preprocessing/preprocess_era5_regression.py \
    --input_dir datasets/raw_era5 \
    --years 2014,...,2024
    # ⚠️ SIN --threshold (NO binarización)
```

**Transformaciones**:
1. Leer .nc files (idéntico a clasificación)
2. Agregar espacialmente (idéntico)
3. **Convertir METROS → MM** (×1000) (idéntico)
4. **NO BINARIZAR**: Preservar valores continuos
5. Features engineering (idéntico)
6. **Target continuo**: `target_precip_24h = precip_mm` (NO umbral)

**Output**: `peru_rainfall_regression.csv`
```python
{
  'timestamp': '2014-01-01',
  'region': 'costa_norte',
  'total_precipitation': 2.5,  # mm (convertido)
  'temperature_2m': 25.3,
  'target_precip_24h': 2.5  # ← CONTINUO [0.0, 200.0] mm
}
```

---

### PASO 2: Limpieza (Regresión) ✨ NUEVO
```bash
python preprocessing/clean_regression_data.py \
    --input_path peru_rainfall_regression.csv \
    --output_path peru_rainfall_regression_cleaned.csv \
    --max_precip 200.0
```

**Estrategia**:
1. **Interpolación temporal** (mejor que median)
   ```python
   df.interpolate(method='linear', limit=5)
   ```

2. **Detección de outliers**
   ```python
   df = df[df['target_precip_24h'] >= 0]  # Sin negativos
   df = df[df['target_precip_24h'] <= 200]  # Sin extremos
   ```

3. **Remoción agresiva de NaN**
   ```python
   df = df.dropna(subset=feature_cols)
   ```

4. **Validación exhaustiva**
   ```python
   assert df['target_precip_24h'].nunique() > 100  # NO binario
   ```

**Output**: `peru_rainfall_regression_cleaned.csv`
```python
{
  'timestamp': '2014-01-01',
  'region': 'costa_norte',
  'total_precipitation': 2.5,
  'temperature_2m': 25.3,
  'target_precip_24h': 2.5,
  # ✅ SIN NaN
  # ✅ SIN outliers
  # ✅ Valores continuos
}
```

---

### PASO 3: Entrenamiento
```bash
python run.py \
    --task_name long_term_forecast \
    --data_path peru_rainfall_regression_cleaned.csv \
    --loss MSE \
    --pred_len 24
```

**Métricas** (esperadas):
- RMSE: 2.5-3.5 mm (vs NASA IMERG: 3.4 mm)
- MAE: 1.8-2.5 mm
- R²: 0.55-0.70

---

## 📊 Comparación de Archivos

| Archivo | Tarea | Target | Valores Únicos | Limpieza |
|---------|-------|--------|----------------|----------|
| `peru_rainfall.csv` | Clasificación | `rain_24h` | 2 | Básica (en preproceso) |
| `peru_rainfall_cleaned.csv` | Clasificación | `rain_24h` | 2 | N/A (no hay script) |
| `peru_rainfall_regression.csv` | Regresión | `target_precip_24h` | ~3000 | Pendiente |
| `peru_rainfall_regression_cleaned.csv` | Regresión | `target_precip_24h` | ~2800 | ✅ Completa |

---

## 🚨 ERRORES COMUNES

### ❌ ERROR 1: Usar CSV equivocado
```python
# MAL: Regresión con CSV de clasificación
python run.py --task_name long_term_forecast \
    --data_path peru_rainfall_cleaned.csv  # ← BINARIO {0,1}

# Resultado: RMSE = 0.45 (sin sentido)
```

✅ **Correcto**:
```python
python run.py --task_name long_term_forecast \
    --data_path peru_rainfall_regression_cleaned.csv  # ← CONTINUO [0,200]mm
```

---

### ❌ ERROR 2: Olvidar limpieza
```python
# MAL: Entrenar con CSV RAW (tiene NaN)
python run.py --task_name long_term_forecast \
    --data_path peru_rainfall_regression.csv  # ← Tiene NaN, outliers

# Error: "RuntimeError: Input contains NaN"
```

✅ **Correcto**:
```python
# Primero limpiar
python preprocessing/clean_regression_data.py ...

# Luego entrenar
python run.py --task_name long_term_forecast \
    --data_path peru_rainfall_regression_cleaned.csv
```

---

### ❌ ERROR 3: No validar datos
```python
# MAL: Asumir que está correcto
df = pd.read_csv('peru_rainfall_regression_cleaned.csv')
# Entrenar sin verificar

# Problema oculto: Datos binarios en "regresión"
```

✅ **Correcto**:
```python
df = pd.read_csv('peru_rainfall_regression_cleaned.csv')

# Validar SIEMPRE
assert df['target_precip_24h'].nunique() > 100, "Datos binarios!"
assert df.isnull().sum().sum() == 0, "Tiene NaN!"
assert df['target_precip_24h'].max() <= 200, "Outliers extremos!"

print("✅ Validación pasada - listo para entrenar")
```

---

## 📝 Checklist Completo

### CLASIFICACIÓN
- [ ] Archivos .nc en `datasets/raw_era5/`
- [ ] Ejecutar `preprocess_era5_peru.py` con `--threshold 0.0001`
- [ ] Verificar `peru_rainfall.csv` tiene columna `rain_24h` con {0, 1}
- [ ] (Opcional) Crear script de limpieza si hay NaN
- [ ] Entrenar con `--task_name classification`
- [ ] Evaluar F1-Score, Precision, Recall

### REGRESIÓN ✨
- [ ] Archivos .nc en `datasets/raw_era5/` (mismos que clasificación)
- [ ] Ejecutar `preprocess_era5_regression.py` SIN `--threshold`
- [ ] Verificar `peru_rainfall_regression.csv` tiene `target_precip_24h` continuo
- [ ] ✅ **Ejecutar `clean_regression_data.py`** (NUEVO, CRÍTICO)
- [ ] Verificar `peru_rainfall_regression_cleaned.csv`:
  - [ ] Sin NaN (`df.isnull().sum().sum() == 0`)
  - [ ] Valores continuos (`df['target_precip_24h'].nunique() > 100`)
  - [ ] Rango válido (`0 <= target <= 200`)
  - [ ] Distribución razonable (`mean ≈ 3-5 mm`)
- [ ] Entrenar con `--task_name long_term_forecast`
- [ ] Evaluar RMSE, MAE, R²

---

## 🎯 Archivos Creados (NUEVOS)

1. **`preprocessing/clean_regression_data.py`** ✨
   - Limpieza específica para regresión
   - Interpolación temporal (time-aware)
   - Detección de outliers
   - Validación exhaustiva

2. **`LIMPIEZA_DATOS_REGRESION.md`** ✨
   - Explica diferencias clasificación vs regresión
   - Justifica estrategias de limpieza
   - Ejemplos de validación

3. **`FLUJO_COMPLETO_CLASIFICACION_REGRESION.md`** (este archivo) ✨
   - Comparación visual completa
   - Checklist paso a paso
   - Errores comunes y soluciones

---

## 📞 Contacto Rápido

**Duda**: ¿Cuál CSV usar para regresión?
**Respuesta**: `peru_rainfall_regression_cleaned.csv`

**Duda**: ¿Por qué crear CSV diferente? ¿No puedo usar el de clasificación?
**Respuesta**: NO. Clasificación tiene target binario {0,1}, regresión necesita continuo [0,200]mm

**Duda**: ¿Es necesario limpiar datos de regresión?
**Respuesta**: SÍ. Regresión es MUY sensible a NaN y outliers (MSE se dispara). Clasificación puede tolerar más.

**Duda**: ¿Qué validar antes de entrenar regresión?
**Respuesta**: 
```python
df['target_precip_24h'].nunique() > 100  # Continuo, NO binario
df.isnull().sum().sum() == 0  # Sin NaN
df['target_precip_24h'].max() <= 200  # Sin outliers
```

---

## ✅ Resumen Ejecutivo

**CLASIFICACIÓN**:
- Preproceso → Binarización → Entrenamiento
- Target: `rain_24h = {0, 1}`
- Limpieza: Básica (en preproceso)

**REGRESIÓN**: ✨
- Preproceso → **Limpieza (NUEVO)** → Entrenamiento
- Target: `target_precip_24h = [0, 200] mm`
- Limpieza: Sofisticada (interpolación, outliers, validación)

**CRÍTICO**: 
- ❌ NO mezclar CSVs (clasificación ≠ regresión)
- ✅ SIEMPRE limpiar datos de regresión antes de entrenar
- ✅ VALIDAR datos continuos (unique > 100)
