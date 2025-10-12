# 🎯 RESPUESTAS A TUS PREGUNTAS

## ❓ Pregunta 1: ¿Cuántos años de datos debo descargar?

### **Respuesta Directa:**

**Para PRUEBAS INICIALES:** Descarga **3 años (2022-2024)**

**Para MODELO FINAL:** Descarga **10 años (2014-2024)**

### **Recomendación Específica:**

```
FASE 1 (Semana 1-2): 
  └─→ 3 años: 2022, 2023, 2024
      ├─ Tamaño: ~900 MB
      ├─ Tiempo: 4-5 horas total
      └─ Objetivo: Validar que todo funciona

FASE 2 (Semana 3-4): 
  └─→ 10 años: 2014-2024
      ├─ Tamaño: ~3.3 GB
      ├─ Tiempo: 16-19 horas total
      └─ Objetivo: Modelo final para tesis
```

### **¿Por qué empezar con 3 años?**

✅ **Ventajas:**
- Descarga rápida (30 minutos)
- Puedes empezar HOY
- Suficiente para validar pipeline
- Identificar errores rápido
- Ajustar hiperparámetros

❌ **Si empiezas directo con 10 años:**
- Si hay un error, pierdes 1 día entero
- No puedes iterar rápido
- Mayor riesgo de frustración

### **Mi Recomendación Personal:**

```
DÍA 1: Descargar 2023 y 2024 (ya los tienes!)
       └─→ Renombrar cds_2023.zip → era5_peru_2023.zip
       └─→ Renombrar cds_2024.zip → era5_peru_2024.zip
       └─→ Descargar 2022 adicional

DÍA 2: Ejecutar pipeline completo con 3 años

DÍA 3-4: Entrenar y validar

DÍA 5: SI TODO FUNCIONA → Descargar 10 años completos

SEMANA 2+: Entrenar modelo final
```

---

## ❓ Pregunta 2: ¿Qué hacer con los datos CDS 2023 y 2024 que ya tengo?

### **Respuesta:**

**¡Úsalos! No necesitas re-descargarlos.**

### **Pasos:**

```bash
# 1. Renombrar archivos
mv cds_2023.zip era5_peru_2023.zip
mv cds_2024.zip era5_peru_2024.zip

# 2. Copiar a datasets/raw_era5/
cp era5_peru_2023.zip AdaptationOpenLTM/datasets/raw_era5/
cp era5_peru_2024.zip AdaptationOpenLTM/datasets/raw_era5/

# 3. Descargar solo 2022 adicional (para 3 años)
# (seguir GUIA_DESCARGA_DATOS.md)

# 4. Ejecutar preprocesamiento
python preprocessing/preprocess_era5_peru.py \
    --years 2022,2023,2024
```

### **Verificar que sean los correctos:**

Tus archivos CDS deben contener:
- ✅ Resolución: 12-hourly (06:00, 18:00 UTC)
- ✅ 10 variables atmosféricas
- ✅ Región: Perú (0°N a -18°S, -82°W a -68°W)
- ✅ Formato: NetCDF (.nc)

Si no estás seguro, abre uno y verifica:

```python
import xarray as xr
ds = xr.open_dataset('era5_peru_2023.nc')
print(ds)  # Ver variables y dimensiones
```

---

## ❓ Pregunta 3: ¿Objetivo es predecir lluvia en 24 horas (no 3 horas)?

### **Respuesta:**

**✅ YA ESTÁ IMPLEMENTADO ASÍ**

El código que creé ya predice lluvia en las próximas **24 horas** (RainTomorrow), NO 3 horas.

### **Configuración:**

En `preprocess_era5_peru.py`:

```python
# Línea de comando
--target_horizon 24  # 24 horas (default)

# Dentro del código
target_horizon = 24  # horas
horizon_steps = 24 // 12  # = 2 timesteps (porque resolución es 12h)
```

### **Cómo funciona:**

```
Timestep actual: t
  ├─ Observaciones: [t-720d ... t]  (contexto)
  └─ Target: ¿Lluvia en t+24h?

Ejemplo concreto:
  ├─ Ahora: 2024-01-15 06:00
  ├─ Contexto: 2022-01-15 hasta 2024-01-15
  └─ Predicción: ¿Lloverá el 2024-01-16 06:00? (24h después)
```

### **Si quisieras cambiar a otro horizonte:**

```python
# Para predecir lluvia en 48 horas
--target_horizon 48

# Para predecir lluvia en 12 horas
--target_horizon 12

# Para predecir lluvia en 1 semana
--target_horizon 168  # 7 días * 24 horas
```

**Pero para tu tesis, quédate con 24 horas** (es el estándar).

---

## ❓ Pregunta 4: ¿Cómo ejecuto esto en Google Colab?

### **Respuesta:**

### **Opción A: Usar el Notebook (MÁS FÁCIL)**

1. Subir `AdaptationOpenLTM` a GitHub
2. Abrir Google Colab: https://colab.research.google.com/
3. File → Upload notebook
4. Seleccionar: `notebooks/colab_training_demo.ipynb`
5. Ejecutar celdas una por una
6. ¡Listo!

### **Opción B: Comandos Manuales**

```python
# ====== CELDA 1: Setup ======
!git clone https://github.com/TU_USUARIO/AdaptationOpenLTM.git
%cd AdaptationOpenLTM
!pip install -r requirements.txt

# ====== CELDA 2: Datos ======
from google.colab import drive
drive.mount('/content/drive')

# Subir archivos ERA5 (opción 1: desde Drive)
!cp /content/drive/MyDrive/ERA5_Data/*.zip datasets/raw_era5/

# O subir manualmente (opción 2)
from google.colab import files
uploaded = files.upload()  # Seleccionar era5_peru_*.zip
!mv era5_peru_*.zip datasets/raw_era5/

# ====== CELDA 3: Preprocesar ======
!python preprocessing/preprocess_era5_peru.py \
    --input_dir datasets/raw_era5 \
    --output_dir datasets/processed \
    --years 2023,2024 \
    --target_horizon 24

# ====== CELDA 4: Entrenar ======
!python run.py \
  --task_name forecast \
  --is_training 1 \
  --model timer_xl_classifier \
  --data PeruRainfall \
  --root_path datasets/processed/ \
  --data_path peru_rainfall.csv \
  --seq_len 1440 \
  --input_token_len 96 \
  --output_token_len 96 \
  --batch_size 128 \
  --learning_rate 1e-5 \
  --train_epochs 50 \
  --patience 10 \
  --use_focal_loss \
  --adaptation \
  --pretrain_model_path checkpoints/timer_xl/checkpoint.pth \
  --gpu 0 \
  --use_norm \
  --cosine \
  --checkpoints results/peru_rainfall/

# ====== CELDA 5: Guardar Resultados ======
!cp results/peru_rainfall/*/checkpoint.pth \
   /content/drive/MyDrive/timer_xl_checkpoints/best_model.pth
```

---

## ❓ Pregunta 5: ¿GPU T4 de Colab es suficiente?

### **Respuesta:**

**✅ SÍ, perfectamente suficiente**

### **Especificaciones T4:**

```
GPU: NVIDIA Tesla T4
VRAM: 16 GB
Compute: 8.1 TFLOPS (FP32)

Comparación:
  RTX 3060:  12 GB VRAM  ❌ Menor
  RTX 3090:  24 GB VRAM  ✅ Mayor (pero cara)
  T4:        16 GB VRAM  ✅ Perfecto para este proyecto
```

### **Ajustes recomendados para T4:**

```python
# Para 3 años de datos
batch_size = 256  # ✅ Funciona perfecto
seq_len = 1440    # ✅ 720 días de contexto

# Para 10 años de datos
batch_size = 128  # ✅ Reduce un poco
seq_len = 1440    # ✅ Mantener igual

# Si tienes OOM (Out of Memory):
batch_size = 64   # Reducir más
seq_len = 720     # O reducir contexto
```

### **Tiempos estimados en T4:**

```
3 años:
  └─ Preprocesamiento: ~15 min
  └─ Entrenamiento (50 epochs): ~3-4 horas
  └─ Total: ~4-5 horas

10 años:
  └─ Preprocesamiento: ~1.5 horas
  └─ Entrenamiento (50 epochs): ~12-15 horas
  └─ Total: ~14-17 horas
```

**IMPORTANTE:** Colab desconecta después de 12 horas. Para 10 años:

```python
# Guardar checkpoint cada época
# Ya implementado en el código
# Se guarda en results/peru_rainfall/

# Y copiar a Drive frecuentemente
!cp results/peru_rainfall/*/checkpoint.pth \
   /content/drive/MyDrive/timer_xl_checkpoints/backup_epoch_XX.pth
```

---

## ❓ Pregunta 6: ¿Qué archivos subo a GitHub?

### **Respuesta:**

**SÍ subes:**
- ✅ Todo el código (`models/`, `preprocessing/`, etc.)
- ✅ Scripts de entrenamiento (`scripts/`)
- ✅ Notebooks (`notebooks/`)
- ✅ Documentación (`*.md`)
- ✅ Configuración (`requirements.txt`, `.gitignore`)

**NO subes:**
- ❌ Datos ERA5 (archivos .zip, .nc) - muy grandes
- ❌ Checkpoints (.pth) - muy grandes
- ❌ Resultados (CSV, NPZ) - muy grandes

### **Estructura en GitHub:**

```
AdaptationOpenLTM/  (repo público)
├── código/         ✅ SUBIR
├── documentación/  ✅ SUBIR
├── datasets/raw_era5/     ❌ NO SUBIR (solo README.md)
├── checkpoints/           ❌ NO SUBIR (solo .gitkeep)
└── results/               ❌ NO SUBIR (solo .gitkeep)
```

### **¿Dónde guardar datos grandes?**

```
Google Drive:
└─ ERA5_Data/
   ├─ era5_peru_2023.zip       (subir aquí)
   ├─ era5_peru_2024.zip
   └─ ...

└─ timer_xl_checkpoints/
   ├─ pretrained.pth           (descargar y subir aquí)
   └─ best_model.pth
```

### **En Colab, copiar desde Drive:**

```python
!cp /content/drive/MyDrive/ERA5_Data/*.zip datasets/raw_era5/
!cp /content/drive/MyDrive/timer_xl_checkpoints/pretrained.pth checkpoints/timer_xl/checkpoint.pth
```

---

## 📋 CHECKLIST FINAL

### **Antes de empezar:**

- [ ] AdaptationOpenLTM subido a GitHub
- [ ] Checkpoint pre-entrenado descargado
- [ ] Archivos ERA5 2023, 2024 renombrados
- [ ] (Opcional) ERA5 2022 descargado
- [ ] Google Colab abierto
- [ ] Google Drive montado

### **Durante ejecución:**

- [ ] Datos preprocesados exitosamente
- [ ] Entrenamiento iniciado sin errores
- [ ] Loss decrece normalmente
- [ ] Checkpoints se guardan en Drive
- [ ] Métricas monitoreadas

### **Después de resultados:**

- [ ] F1-Score > 0.70 (mínimo)
- [ ] Confusion matrix analizada
- [ ] Resultados por región guardados
- [ ] Si exitoso → Descargar 10 años
- [ ] Experimentar con context lengths

---

## 🎓 CONCLUSIÓN

### **Tu Plan de Acción Inmediato:**

```
HOY (Día 1):
  ✓ Subir AdaptationOpenLTM a GitHub
  ✓ Descargar checkpoint pre-entrenado
  ✓ Renombrar cds_2023.zip y cds_2024.zip
  ✓ Descargar ERA5 2022

MAÑANA (Día 2):
  ✓ Ejecutar preprocesamiento
  ✓ Iniciar entrenamiento en Colab

PASADO (Día 3-4):
  ✓ Monitorear entrenamiento
  ✓ Analizar primeros resultados

SIGUIENTE SEMANA:
  ✓ SI EXITOSO → Descargar 10 años
  ✓ Entrenar modelo final
```

### **Contacto para Dudas:**

Si tienes problemas, revisa:
1. `IMPLEMENTACION_COMPLETA.md` - Guía paso a paso
2. `GUIA_DESCARGA_DATOS.md` - Detalles de descarga
3. `README_PERU_RAINFALL.md` - Overview completo
4. `notebooks/colab_training_demo.ipynb` - Demo ejecutable

---

**¡TODO CLARO! 🚀**

Tu siguiente paso es:
1. Subir a GitHub
2. Descargar checkpoint
3. Ejecutar en Colab

**Tiempo estimado hasta primeros resultados: 1 día**

¡Éxitos con tu tesis! 🎓
