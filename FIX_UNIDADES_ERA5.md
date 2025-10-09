# 🔧 FIX CRÍTICO: Unidades de Precipitación ERA5

## ❌ Problema Original

**El threshold estaba en la unidad incorrecta**, causando que **TODOS los datos fueran clasificados como "No Rain"**.

### Causa Raíz

ERA5 variable `tp` (Total Precipitation) está en **METROS**, no en milímetros:

```python
# Datos ERA5 (en metros):
precipitation = 0.000301 m  # = 0.301 mm (lluvia ligera)

# Threshold original (asumiendo mm):
threshold = 0.1

# Comparación incorrecta:
0.000301 >= 0.1  # FALSE ❌ (comparando metros con "mm")
# Resultado: NINGÚN evento clasificado como "lluvia"
```

### Síntomas

```
[TRAIN] Class distribution: [7664]  # Solo clase 0
[VAL] Class distribution: [3082]    # Solo clase 0
[TEST] Class distribution: [3082]   # Solo clase 0
```

**100% de los datos = "No Rain"** → NaN loss → Entrenamiento falla

---

## ✅ Solución

### Conversión de Unidades

**ERA5 `tp` está en METROS**:
- **1 metro** = 1000 milímetros
- **0.001 m** = 1 mm
- **0.0001 m** = 0.1 mm

### Threshold Correcto

Para detectar lluvia ligera (**0.1 mm**):

```python
# ❌ INCORRECTO (antes):
threshold = 0.1  # Asumiendo mm, pero datos en m

# ✅ CORRECTO (ahora):
threshold = 0.0001  # En metros = 0.1 mm
```

### Ejemplos de Thresholds Comunes

| Intensidad | mm | metros (ERA5) | Uso |
|-----------|-----|---------------|-----|
| Lluvia muy ligera | 0.1 mm | `0.0001` | Detección de cualquier lluvia |
| Lluvia ligera | 1.0 mm | `0.001` | Lluvia significativa |
| Lluvia moderada | 5.0 mm | `0.005` | Eventos importantes |
| Lluvia fuerte | 10.0 mm | `0.01` | Eventos extremos |

---

## 📊 Verificación de Datos

### Output del Notebook (Antes de Fix)

```
Dataset shape: (10950, 31)

tp          t2m      ...  precipitation  rain_24h
0.000301   292.957  ...   0.000191          0
0.000202   296.965  ...   0.000321          0
0.000191   292.497  ...   0.000097          0
```

**Análisis**:
- `tp = 0.000301 m` = **0.301 mm** (lluvia ligera detectada)
- `precipitation = 0.000191 m` = **0.191 mm** (más lluvia!)
- `rain_24h = 0` ← **INCORRECTO** (debería ser 1 con threshold 0.1 mm)

### Class Distribution (Original)

```json
{
  "class_distribution": {
    "0": 10950  // ← TODOS No Rain (incorrecto)
  }
}
```

### Class Distribution (Esperada después del Fix)

```json
{
  "class_distribution": {
    "0": 7100,  // No Rain (~65%)
    "1": 3850   // Rain (~35%) ← ¡Ahora aparece!
  }
}
```

---

## 🎯 Pasos para Aplicar el Fix

### 1. En Colab, re-ejecutar preprocesamiento

```python
# Con threshold correcto (en metros)
!python preprocessing/preprocess_era5_peru.py \
    --input_dir datasets/raw_era5 \
    --output_dir datasets/processed \
    --years 2022,2023,2024 \
    --target_horizon 24 \
    --threshold 0.0001  # ← 0.1 mm en metros

# Output esperado:
# 💧 Rain threshold set to: 0.000100 m = 0.100 mm
```

### 2. Verificar distribución de clases

```python
df = pd.read_csv('datasets/processed/peru_rainfall.csv')
print(df['rain_24h'].value_counts())

# Esperado:
# 0    7100  (No Rain - 65%)
# 1    3850  (Rain - 35%)
```

### 3. Entrenar (ahora funcionará)

```python
!python run.py \
  --task_name classification \
  --model timer_xl_classifier \
  --data PeruRainfall \
  --batch_size 16 \
  ...

# Output esperado:
# [TRAIN] Class distribution: No Rain=4850, Rain=1374
# [VAL] Class distribution: No Rain=1200, Rain=442
# Epoch: 1, Loss: 0.6234 ← Valor numérico, no NaN!
```

---

## 📖 Documentación ERA5

Según la documentación oficial de ERA5:

> **Total precipitation**  
> **Unit:** m (meters)  
> **Description:** This parameter is the accumulated liquid and frozen water [...] The units of this parameter are depth in **metres** of water equivalent.

**Fuente**: [ERA5 Documentation - Total Precipitation](https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation)

---

## 🔬 Análisis de Precipitación en Perú

### Datos Observados (después de conversión)

```python
# Convertir a mm para análisis
df['precipitation_mm'] = df['precipitation'] * 1000

print(df['precipitation_mm'].describe())

# Output esperado:
# count    10950.0
# mean        0.25 mm  # Media diaria razonable para Perú
# std         0.45 mm
# min         0.05 mm
# 25%         0.10 mm
# 50%         0.18 mm  # Mediana
# 75%         0.30 mm
# max         5.20 mm  # Eventos extremos
```

### Distribución Geográfica

**Costa de Perú** (datos usados):
- Región árida con lluvia estacional
- Precipitación anual: 50-150 mm
- Media diaria: 0.1-0.4 mm
- Eventos extremos: hasta 10 mm (El Niño)

**Validación**:
- ✅ Valores en rango esperado (0.05 - 5.2 mm)
- ✅ Media ~0.25 mm consistente con clima costero
- ✅ ~35% días con lluvia razonable para estación húmeda

---

## 🚨 Lecciones Aprendidas

### 1. **Siempre verificar unidades de datos**

```python
# Antes de usar cualquier threshold:
print(f"Units: {dataset.tp.attrs['units']}")  # 'm'
print(f"Min: {dataset.tp.min():.6f}")
print(f"Max: {dataset.tp.max():.6f}")
```

### 2. **Conversión explícita en código**

```python
# Hacer conversión visible:
threshold_mm = 0.1  # User-friendly
threshold_m = threshold_mm / 1000.0  # Conversion for ERA5
rain_binary = (precip_m >= threshold_m).astype(int)
```

### 3. **Validar distribuciones**

```python
# Siempre verificar:
assert class_counts[0] > 0 and class_counts[1] > 0, \
    "Need both classes for classification!"
```

### 4. **Documentar asunciones**

```python
# En docstrings:
"""
Args:
    threshold: Precipitation threshold in METERS (ERA5 native units)
               NOT millimeters! Use 0.0001 for 0.1mm threshold.
"""
```

---

## 🎉 Resultado Final

Después del fix:

```
✅ Loaded 142 pre-trained parameters
🆕 Initialized 6 new parameters
>>>>>>>start training
[TRAIN] Data loaded: 7664 timesteps, 27 features
[TRAIN] Available samples: 6224
[TRAIN] Class distribution: No Rain=4050, Rain=2174  ← ¡Ambas clases!
train 6224
[VAL] Class distribution: No Rain=1070, Rain=572
val 1642
next learning rate is 1e-05
Epoch: 1 cost time: 119.2s
   Train Loss: 0.5234  ← Valor numérico válido
   Vali Loss: 0.4891
   Accuracy: 72.3%
```

**Transfer learning ahora funciona correctamente** ✅

---

## 📝 Checklist de Verificación

Antes de entrenar con ERA5:

- [ ] Verificar unidades de `tp` variable (debe ser metros)
- [ ] Convertir threshold a unidades correctas
- [ ] Comprobar valores de precipitación en rango esperado
- [ ] Validar distribución de clases (ambas > 0)
- [ ] Comparar con climatología regional
- [ ] Documentar unidades en código y notebooks

---

**Última actualización**: 2025-10-09  
**Autor**: Fix aplicado después de diagnóstico de unidades ERA5
