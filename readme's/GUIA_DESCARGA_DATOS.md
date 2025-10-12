# 📊 Guía de Descarga de Datos ERA5 para Timer-XL

## 🎯 Resumen Ejecutivo

**Recomendación**: Comienza con **3 años (2022-2024)** para pruebas iniciales, luego escala a **10 años (2014-2024)** para el modelo final.

---

## 📈 Estrategia de Descarga por Fases

### **Fase 1: Pruebas y Validación (RECOMENDADO INICIAR AQUÍ)**

```python
años_a_descargar = [2022, 2023, 2024]  # 3 años
```

**Ventajas:**
- ✅ Descarga rápida (~300 MB por año = ~900 MB total)
- ✅ Preprocesamiento rápido (~15 minutos)
- ✅ Entrenamiento rápido en T4 (~3-4 horas)
- ✅ Suficiente para validar pipeline completo
- ✅ Incluye datos recientes (más relevantes)
- ✅ Cubre al menos 1 ciclo ENSO parcial

**Desventajas:**
- ⚠️ No captura ciclo ENSO completo (3-7 años)
- ⚠️ Menor robustez estadística
- ⚠️ Puede no generalizar bien a eventos extremos

**Métricas esperadas (3 años):**
```
F1-Score: 0.65-0.72
AUC-ROC: 0.75-0.82
Recall: 0.60-0.70
```

**¿Cuándo usar?**
- Primera iteración del proyecto
- Debuggear pipeline
- Ajustar hiperparámetros
- Validar metodología
- Iteración rápida

---

### **Fase 2: Entrenamiento Intermedio (OPCIONAL)**

```python
años_a_descargar = [2018, 2019, 2020, 2021, 2022, 2023, 2024]  # 7 años
```

**Ventajas:**
- ✅ Captura al menos 1 ciclo ENSO completo
- ✅ Mejor generalización
- ✅ Mayor diversidad climática
- ✅ ~2.1 GB total (manejable)


**Métricas esperadas (7 años):**
```
F1-Score: 0.70-0.77
AUC-ROC: 0.80-0.87
Recall: 0.68-0.76
```

**¿Cuándo usar?**
- Después de validar pipeline con 3 años
- Antes del modelo final
- Si quieres balance tiempo/desempeño

---

### **Fase 3: Modelo Final de Producción**

```python
años_a_descargar = list(range(2014, 2025))  # 11 años (2014-2024)
```

**Ventajas:**
- ✅ Captura 2-3 ciclos ENSO completos
- ✅ Máxima robustez estadística
- ✅ Mejor generalización a eventos extremos
- ✅ Datos suficientes para publicación
- ✅ Cubre El Niño 2015-2016 (muy fuerte)
- ✅ Cubre La Niña 2020-2022 (triple-dip)


**Métricas esperadas (10+ años):**
```
F1-Score: 0.75-0.82
AUC-ROC: 0.85-0.92
Recall: 0.72-0.82
```

**¿Cuándo usar?**
- Modelo final para tesis
- Resultados para publicación
- Después de validar con 3-7 años
- Experimentos de ablación de contexto

---

## 📅 Años Clave de Eventos ENSO

### **El Niño (Años Cálidos)**
```
2014-2015: Débil
2015-2016: MUY FUERTE ⭐ (uno de los más fuertes registrados)
2018-2019: Débil a moderado
2023-2024: Moderado
```

### **La Niña (Años Fríos)**
```
2016-2017: Débil
2017-2018: Moderado
2020-2021: Moderado
2021-2022: Moderado (triple-dip)
2022-2023: Débil
```

### **Neutral**
```
2014: Neutral
2019-2020: Neutral a débil
```

**Implicación**: Para capturar ambas fases ENSO, necesitas **mínimo 5-7 años**.

---

## 💡 Recomendación Detallada por Objetivo

### **Si tu objetivo es:** Validar pipeline y metodología
```python
años = [2022, 2023, 2024]  # 3 años
tiempo_descarga = "~30 min"
tiempo_preprocesamiento = "~15 min"
tiempo_entrenamiento = "~3-4 horas (T4)"
```

### **Si tu objetivo es:** Resultados preliminares de tesis
```python
años = [2018, 2019, 2020, 2021, 2022, 2023, 2024]  # 7 años
tiempo_descarga = "~1 hora"
tiempo_preprocesamiento = "~45 min"
tiempo_entrenamiento = "~6-8 horas (T4)"
```

### **Si tu objetivo es:** Modelo final para publicación
```python
años = list(range(2014, 2025))  # 11 años
tiempo_descarga = "~2 horas"
tiempo_preprocesamiento = "~1.5 horas"
tiempo_entrenamiento = "~12-15 horas (T4)"
```

---

## 📦 Archivos a Descargar

### **Opción 1: Descarga Manual (CDS Web Interface)**

Para cada año:
1. Ir a: https://cds.climate.copernicus.eu/
2. Seleccionar "ERA5 hourly data on single levels from 1940 to present"
3. Configurar:
   - **Product**: Reanalysis
   - **Variable**: (seleccionar las 10 variables necesarias)
   - **Year**: [año específico]
   - **Month**: All
   - **Day**: All
   - **Time**: 06:00, 18:00
   - **Area**: 0°N, -82°W, -18°S, -68°W (Perú)
   - **Format**: NetCDF
4. Submit form → Download

**Nombrar archivos:**
```
era5_peru_2022.nc
era5_peru_2023.nc
era5_peru_2024.nc
...
```

**Luego comprimir:**
```bash
zip era5_peru_2022.zip era5_peru_2022.nc
zip era5_peru_2023.zip era5_peru_2023.nc
zip era5_peru_2024.zip era5_peru_2024.nc
```

### **Opción 2: Descarga Automática (CDS API)**

```python
# Crear script de descarga (próximamente)
# preprocessing/download_era5.py --years 2022,2023,2024
```

---

## 💾 Gestión de Almacenamiento en Google Colab

### **Google Drive**
```python
# Montar Drive
from google.colab import drive
drive.mount('/content/drive')

# Estructura recomendada:
/content/drive/MyDrive/
├── ERA5_Data/
│   ├── era5_peru_2022.zip
│   ├── era5_peru_2023.zip
│   └── era5_peru_2024.zip
├── timer_xl_checkpoints/
│   └── peru_rainfall_best.pth
└── results/
    └── (outputs del entrenamiento)
```
