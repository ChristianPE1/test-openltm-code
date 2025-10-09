# 🚨 DIAGNÓSTICO CRÍTICO: Problema de Clases Desbalanceadas

## ❌ Problema Identificado

El entrenamiento falla con **NaN loss** y **100% accuracy** porque:

### Causa Raíz
**Todos los datos pertenecen a una sola clase** (probablemente clase 0 = "No Rain")

```
[TRAIN] Class distribution: [7664]  # Solo clase 0
[VAL] Class distribution: [3082]     # Solo clase 0  
[TEST] Class distribution: [3082]    # Solo clase 0
```

### Síntomas
1. **NaN Loss**: CrossEntropy no puede calcular gradientes cuando solo hay una clase
2. **100% Accuracy**: El modelo siempre predice la clase mayoritaria (correctamente)
3. **Skipping all batches**: Todos los batches tienen NaN/Inf loss

---

## 🔍 Análisis del Problema

### Por qué sucede esto

El **threshold de precipitación = 0.1 mm** puede ser:
- **Demasiado alto** para datos ERA5 en Perú
- **Mal calibrado** para la resolución temporal (12-hourly)
- **Incompatible** con los datos ERA5 descargados

### Datos ERA5
- **Resolución temporal**: 12-hourly (00:00 y 12:00 UTC)
- **Variable**: Total Precipitation (`tp`)
- **Unidades**: metros → convertido a mm (multiplicado por 1000)
- **Acumulación**: 12 horas

### Posibles causas
1. **Threshold muy alto**: 0.1 mm/12h es ~2 mm/día, que puede ser demasiado para eventos de lluvia en Perú
2. **Datos con ruido**: ERA5 puede tener valores muy bajos (< 0.01 mm)
3. **Región árida**: Algunas regiones costeras de Perú tienen muy poca lluvia
4. **Error en conversión**: Unidades mal convertidas en el preprocesamiento

---

## ✅ Solución Propuesta

### Opción 1: Ajustar threshold automáticamente (RECOMENDADO)

Ejecuta el **notebook cell de verificación** que agregué:

```python
# Check class distribution
df = pd.read_csv('datasets/processed/peru_rainfall.csv')
print(df['rain_24h'].value_counts())
print(df['precipitation'].describe())
```

Luego **re-procesa con threshold ajustado**:

```python
# Calculate threshold for ~35% rain events
suggested_threshold = df['precipitation'].quantile(0.65)

!python preprocessing/preprocess_era5_peru.py \
    --input_dir datasets/raw_era5 \
    --output_dir datasets/processed \
    --years 2022,2023,2024 \
    --target_horizon 24 \
    --threshold {suggested_threshold:.4f}
```

### Opción 2: Usar percentiles en lugar de threshold absoluto

Modificar `preprocess_era5_peru.py`:

```python
# En lugar de:
target_binary = (rain_24h >= self.threshold).astype(int)

# Usar:
threshold_percentile = np.nanpercentile(rain_24h, 65)  # Top 35% = rain
target_binary = (rain_24h >= threshold_percentile).astype(int)
```

### Opción 3: Threshold muy bajo (0.001 mm)

Si los datos tienen mucho ruido, usa threshold muy bajo:

```bash
!python preprocessing/preprocess_era5_peru.py \
    --threshold 0.001
```

---

## 📊 Target Class Distribution

Para entrenamiento exitoso, necesitas:

- **Mínimo**: 10% de la clase minoritaria (900+ samples de "Rain")
- **Óptimo**: 30-40% de la clase minoritaria (~3000 samples)
- **Ideal**: 40-60% balance (clase minoritaria)

### Distribución actual (estimado)
```
No Rain (0): 100% (9510 samples)
Rain (1):      0% (0 samples)  ← PROBLEMA
```

### Distribución objetivo
```
No Rain (0): 65% (~6200 samples)
Rain (1):    35% (~3300 samples)  ← OBJETIVO
```

---

## 🔧 Pasos para Corregir

### 1. Verificar datos en Colab

```python
# Cell nuevo en el notebook
import pandas as pd
import numpy as np

df = pd.read_csv('datasets/processed/peru_rainfall.csv')

print("📊 Current Data:")
print(f"   Total samples: {len(df)}")
print(f"   Rain events: {(df['rain_24h'] == 1).sum()}")
print(f"   No-rain events: {(df['rain_24h'] == 0).sum()}")
print(f"\n🌧️ Precipitation:")
print(f"   Min: {df['precipitation'].min():.6f} mm")
print(f"   Max: {df['precipitation'].max():.6f} mm")
print(f"   Mean: {df['precipitation'].mean():.6f} mm")
print(f"   Median: {df['precipitation'].median():.6f} mm")
print(f"   95th percentile: {df['precipitation'].quantile(0.95):.6f} mm")

# Suggested new threshold
new_threshold = df['precipitation'].quantile(0.65)
print(f"\n💡 Suggested threshold: {new_threshold:.6f} mm")
print(f"   This would give ~{((df['precipitation'] >= new_threshold).sum() / len(df) * 100):.1f}% rain events")
```

### 2. Re-procesar con threshold correcto

Usa el threshold sugerido del paso anterior.

### 3. Verificar distribución después

```python
df_new = pd.read_csv('datasets/processed/peru_rainfall.csv')
print("✅ New distribution:")
print(df_new['rain_24h'].value_counts())
print(df_new['rain_24h'].value_counts(normalize=True) * 100)
```

### 4. Entrenar con datos balanceados

Solo después de verificar que **ambas clases existen**, ejecuta el training cell.

---

## 🎯 Checklist de Verificación

Antes de entrenar, verifica:

- [ ] `df['rain_24h'].value_counts()` muestra AMBAS clases (0 y 1)
- [ ] Clase minoritaria (Rain) tiene al menos 10% de los datos
- [ ] No hay valores NaN en `rain_24h`
- [ ] Precipitation tiene rango razonable (> 0, < 100 mm/día)
- [ ] DataLoader imprime: `No Rain=XXXX, Rain=YYYY` con ambos > 0

---

## 📝 Notas Adicionales

### Por qué 0.1 mm puede ser problemático

- ERA5 `tp` es **acumulación en 12 horas**
- 0.1 mm/12h = 0.2 mm/día = 2 mm/10 días
- En regiones áridas de Perú, esto puede ser demasiado
- Muchos eventos de "lluvia ligera" (< 0.1 mm) se pierden

### Alternativa: Clasificación de intensidad

En lugar de binario (Rain/No Rain), podrías usar:
- Clase 0: No Rain (< 0.01 mm)
- Clase 1: Light Rain (0.01 - 1 mm)
- Clase 2: Moderate Rain (1 - 10 mm)
- Clase 3: Heavy Rain (> 10 mm)

---

## 🚀 Próximos Pasos

1. **EJECUTA** las cells de verificación en el notebook
2. **ANALIZA** la distribución de precipitación
3. **RE-PROCESA** con threshold ajustado
4. **VERIFICA** que ambas clases existen
5. **ENTRENA** solo cuando veas distribución balanceada

No intentes entrenar hasta que veas algo como:
```
[TRAIN] Class distribution: No Rain=4500, Rain=1724
```

---

**⚠️ IMPORTANTE**: Este no es un problema del modelo Timer-XL ni del transfer learning. Es un problema de **calidad de datos** que debe corregirse en el preprocesamiento.
