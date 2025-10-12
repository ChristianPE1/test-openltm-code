# 🎯 AdaptationOpenLTM - Resumen de Implementación

## ✅ Cambios Implementados

### 📁 Estructura de Carpetas Creada

```
AdaptationOpenLTM/
├── datasets/
│   ├── raw_era5/          ← SUBIR ARCHIVOS .zip AQUÍ
│   │   └── README.md
│   └── processed/         ← Datos procesados (generado automáticamente)
│
├── preprocessing/
│   └── preprocess_era5_peru.py  ← Script de preprocesamiento
│
├── data_provider/
│   ├── data_loader_peru.py      ← DataLoader para clasificación binaria
│   └── data_factory.py           ← Actualizado con PeruRainfall
│
├── models/
│   └── timer_xl_classifier.py   ← Timer-XL adaptado para clasificación
│
├── exp/
│   ├── exp_basic.py              ← Actualizado con timer_xl_classifier
│   └── exp_forecast.py           ← Modificado para soportar clasificación
│
├── utils/
│   └── tools.py                  ← Agregado FocalLoss
│
├── scripts/
│   └── adaptation/
│       └── peru_rainfall/
│           └── train_timerxl_peru.sh  ← Script de entrenamiento
│
├── notebooks/
│   └── colab_training_demo.ipynb     ← Demo para Google Colab
│
├── results/                           ← Resultados (generado automáticamente)
│
├── checkpoints/
│   └── timer_xl/
│       └── checkpoint.pth            ← Modelo pre-entrenado (descargar)
│
├── README_PERU_RAINFALL.md           ← README principal del proyecto
├── GUIA_DESCARGA_DATOS.md            ← Guía de descarga de ERA5
└── CAMBIOS_RUN_PY.md                 ← Cambios necesarios en run.py
```

---

## 🚀 Pasos para Ejecutar (Quick Start)

### **Paso 1: Preparar Repositorio**

```bash
# 1. Subir la carpeta AdaptationOpenLTM a GitHub
cd AdaptationOpenLTM
git init
git add .
git commit -m "Initial commit: Timer-XL adaptation for Peru rainfall"
git remote add origin https://github.com/TU_USUARIO/AdaptationOpenLTM.git
git push -u origin main
```

### **Paso 2: Descargar Checkpoint Pre-entrenado**

```bash
# Descargar de: https://cloud.tsinghua.edu.cn/f/01c35ca13f474176be7b/
# Guardar en: checkpoints/timer_xl/checkpoint.pth
```

### **Paso 3: Subir Datos ERA5**

**Opción A: Descargar primero (Recomendado)**
- Ir a CDS Copernicus: https://cds.climate.copernicus.eu/
- Descargar años 2022, 2023, 2024 (ver `GUIA_DESCARGA_DATOS.md`)
- Guardar como: `era5_peru_2022.zip`, `era5_peru_2023.zip`, `era5_peru_2024.zip`
- Subir a `datasets/raw_era5/` en tu repositorio o Google Drive

**Opción B: Usar datos existentes**
- Si ya tienes `cds_2023.zip` y `cds_2024.zip`
- Renombrar a `era5_peru_2023.zip` y `era5_peru_2024.zip`
- Subir a `datasets/raw_era5/`

### **Paso 4: Modificar run.py**

Seguir instrucciones en `CAMBIOS_RUN_PY.md` para agregar:
- `--n_classes`
- `--use_focal_loss`
- `--class_weights`

### **Paso 5: Ejecutar en Google Colab**

```python
# En Google Colab
!git clone https://github.com/TU_USUARIO/AdaptationOpenLTM.git
%cd AdaptationOpenLTM

# Instalar dependencias
!pip install -r requirements.txt

# Subir datos ERA5 (o copiar desde Drive)
from google.colab import drive
drive.mount('/content/drive')
!cp /content/drive/MyDrive/ERA5_Data/*.zip datasets/raw_era5/

# Preprocesar
!python preprocessing/preprocess_era5_peru.py \
    --input_dir datasets/raw_era5 \
    --output_dir datasets/processed \
    --years 2023,2024 \
    --target_horizon 24

# Entrenar
!bash scripts/adaptation/peru_rainfall/train_timerxl_peru.sh
```

O usar el notebook: `notebooks/colab_training_demo.ipynb`

---

## 📊 Configuración Recomendada

### **Para Pruebas Iniciales (3 años)**

```bash
# Datos
años = [2022, 2023, 2024]

# Modelo
seq_len = 1440        # 720 días de contexto
batch_size = 256      # Ajustar según GPU
learning_rate = 1e-5
epochs = 50

# Tiempo esperado en T4
preprocesamiento: ~15 min
entrenamiento: ~3-4 horas
```

### **Para Modelo Final (10 años)**

```bash
# Datos
años = [2014-2024]

# Modelo
seq_len = 2880        # 1440 días (2 años) de contexto
batch_size = 128      # Reducido por más contexto
learning_rate = 1e-5
epochs = 50

# Tiempo esperado en T4
preprocesamiento: ~1.5 horas
entrenamiento: ~12-15 horas
```

---

## 🎯 Experimentos de Ablación (Context Length)

Una vez validado el pipeline con 3 años, ejecutar:

```bash
# Experimento 1: 90 días de contexto
seq_len=180  # 90 días * 2 (12h)
python run.py --seq_len 180 --model_id "context_90d" ...

# Experimento 2: 180 días
seq_len=360
python run.py --seq_len 360 --model_id "context_180d" ...

# Experimento 3: 1 año
seq_len=730
python run.py --seq_len 730 --model_id "context_1y" ...

# Experimento 4: 2 años
seq_len=1460
python run.py --seq_len 1460 --model_id "context_2y" ...

# Experimento 5: 3 años
seq_len=2190
python run.py --seq_len 2190 --model_id "context_3y" ...
```

---

## 📈 Métricas Esperadas

### **Con 3 años de datos**
```
F1-Score: 0.65-0.72
AUC-ROC: 0.75-0.82
Recall: 0.60-0.70
```

### **Con 10 años de datos**
```
F1-Score: 0.75-0.82
AUC-ROC: 0.85-0.92
Recall: 0.72-0.82
```

### **Objetivo de Tesis (Mínimo Aceptable)**
```
F1-Score: > 0.70
AUC-ROC: > 0.80
Recall: > 0.65
```

---

## 🔧 Troubleshooting Común

### **Error: Import "torch" could not be resolved**
- Esto es solo un warning de VSCode
- No afecta la ejecución en Colab
- Instalar torch localmente si quieres eliminar el warning: `pip install torch`

### **Error: CUDA Out of Memory**
```bash
# Reducir batch_size
--batch_size 128  # o 64

# Reducir seq_len
--seq_len 720  # en lugar de 1440
```

### **Error: File not found después de extraction**
```bash
# Verificar estructura dentro del .zip
unzip -l datasets/raw_era5/era5_peru_2023.zip

# Si el .nc está en subdirectorio, extraer manualmente
unzip datasets/raw_era5/era5_peru_2023.zip -d datasets/raw_era5/
mv datasets/raw_era5/subdirectory/*.nc datasets/raw_era5/era5_peru_2023.nc
```

### **Error: Variable not found in NetCDF**
- Verificar que el archivo ERA5 contiene las 10 variables necesarias
- Ver `datasets/raw_era5/README.md` para lista de variables
- Re-descargar con las variables correctas si es necesario

---

## 📚 Archivos de Documentación

1. **README_PERU_RAINFALL.md** - Visión general del proyecto
2. **GUIA_DESCARGA_DATOS.md** - Guía detallada de descarga ERA5
3. **CAMBIOS_RUN_PY.md** - Modificaciones necesarias en run.py
4. **datasets/raw_era5/README.md** - Instrucciones para subir datos
5. **ESTE ARCHIVO (RESUMEN_IMPLEMENTACION.md)** - Resumen ejecutivo

---

## 🎓 Recomendación Final

### **Semana 1-2: Validación**
1. ✅ Descargar 3 años (2022-2024)
2. ✅ Ejecutar pipeline completo
3. ✅ Obtener métricas preliminares
4. ✅ Ajustar hiperparámetros si es necesario

### **Semana 3-4: Modelo Final**
1. ✅ Si resultados son prometedores, descargar 10 años
2. ✅ Entrenar modelo final
3. ✅ Ejecutar experimentos de ablación
4. ✅ Analizar por fase ENSO

### **Semana 5+: Análisis**
1. ✅ Generar figuras para tesis
2. ✅ Análisis de errores
3. ✅ Comparación por regiones
4. ✅ Escribir resultados

---

## 📞 Próximos Pasos Inmediatos

1. **Subir AdaptationOpenLTM a GitHub**
   ```bash
   git init
   git add .
   git commit -m "Timer-XL adaptation for Peru rainfall"
   git remote add origin https://github.com/TU_USUARIO/AdaptationOpenLTM.git
   git push -u origin main
   ```

2. **Descargar checkpoint pre-entrenado**
   - URL: https://cloud.tsinghua.edu.cn/f/01c35ca13f474176be7b/
   - Guardar en: `checkpoints/timer_xl/checkpoint.pth`

3. **Decidir cantidad de años**
   - Opción A: 3 años (2022-2024) para pruebas rápidas
   - Opción B: 10 años (2014-2024) directamente (más tiempo)
   - **Recomendación**: Empezar con 3 años

4. **Descargar ERA5**
   - Seguir `GUIA_DESCARGA_DATOS.md`
   - Subir a `datasets/raw_era5/`

5. **Modificar run.py**
   - Seguir `CAMBIOS_RUN_PY.md`
   - Agregar argumentos de clasificación

6. **Ejecutar en Colab**
   - Usar `notebooks/colab_training_demo.ipynb`
   - O seguir pasos en `README_PERU_RAINFALL.md`

---

## ✨ Todo Listo!

Tu repositorio está completamente preparado para:
- ✅ Preprocesamiento de ERA5
- ✅ Transfer learning con Timer-XL
- ✅ Clasificación binaria de lluvias
- ✅ Experimentos de contexto largo
- ✅ Análisis por fase ENSO
- ✅ Ejecución en Google Colab con GPU T4

**¡Hora de comenzar los experimentos! 🚀**
