# 📊 RESUMEN VISUAL - Timer-XL Peru Rainfall

## 🎯 Objetivo del Proyecto

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  INPUT: ERA5 Data (2023-2024 o 2014-2024)                     │
│  ├── 10 variables atmosféricas                                 │
│  ├── Resolución: 12-hourly                                     │
│  └── Región: Perú (5 subregiones)                             │
│                                                                 │
│  MODELO: Timer-XL (Transfer Learning)                          │
│  ├── Pre-entrenado en 260B time points                        │
│  ├── Arquitectura: 8 layers, d_model=1024                     │
│  └── Context: hasta 3 años de lookback                        │
│                                                                 │
│  OUTPUT: Predicción Binaria                                    │
│  ├── Target: RainTomorrow (próximas 24h)                      │
│  ├── Métricas: F1, AUC, Recall                                │
│  └── Análisis por fase ENSO y región                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Estructura del Repositorio

```
AdaptationOpenLTM/
│
├── 📦 datasets/
│   ├── raw_era5/              ← SUBIR .zip AQUÍ (Step 1)
│   │   ├── era5_peru_2023.zip
│   │   └── era5_peru_2024.zip
│   └── processed/             ← Generado automáticamente
│       └── peru_rainfall.csv
│
├── 🔧 preprocessing/
│   └── preprocess_era5_peru.py    ← Script principal (Step 2)
│
├── 🧠 models/
│   └── timer_xl_classifier.py     ← Modelo adaptado
│
├── 📊 data_provider/
│   ├── data_loader_peru.py        ← DataLoader custom
│   └── data_factory.py            ← Actualizado
│
├── 🚀 scripts/
│   └── adaptation/peru_rainfall/
│       └── train_timerxl_peru.sh  ← Script de entrenamiento (Step 3)
│
├── 💾 checkpoints/
│   └── timer_xl/
│       └── checkpoint.pth         ← Pre-trained (descargar)
│
├── 📓 notebooks/
│   └── colab_training_demo.ipynb  ← Google Colab demo
│
└── 📚 Documentación/
    ├── README_PERU_RAINFALL.md
    ├── GUIA_DESCARGA_DATOS.md
    ├── RESUMEN_IMPLEMENTACION.md
    └── IMPLEMENTACION_COMPLETA.md
```

---

## 🔄 Pipeline Completo

```
┌──────────────────────────────────────────────────────────────────┐
│  FASE 1: PREPARACIÓN DE DATOS                                    │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  [1] Descargar ERA5 (CDS Copernicus)                           │
│       │                                                          │
│       ├─→ Opción A: 3 años (2022-2024) ~900 MB                 │
│       └─→ Opción B: 10 años (2014-2024) ~3.3 GB               │
│                                                                  │
│  [2] Subir a datasets/raw_era5/                                │
│       │                                                          │
│       └─→ era5_peru_YYYY.zip                                   │
│                                                                  │
│  [3] Ejecutar preprocesamiento                                  │
│       │                                                          │
│       └─→ python preprocessing/preprocess_era5_peru.py         │
│                                                                  │
│  [4] Output: peru_rainfall.csv                                 │
│       │                                                          │
│       └─→ Shape: (N_samples, N_features + 4)                   │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│  FASE 2: TRANSFER LEARNING                                       │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  [1] Descargar checkpoint pre-entrenado                         │
│       │                                                          │
│       └─→ https://cloud.tsinghua.edu.cn/f/01c35ca13f474176be7b/│
│                                                                  │
│  [2] Cargar Timer-XL                                            │
│       │                                                          │
│       ├─→ Load pretrained encoder                              │
│       ├─→ Add classification head                              │
│       └─→ Initialize with checkpoint                           │
│                                                                  │
│  [3] Entrenar                                                   │
│       │                                                          │
│       ├─→ Fine-tune all layers                                 │
│       ├─→ Use Focal Loss (desbalance)                          │
│       ├─→ Early stopping (patience=10)                         │
│       └─→ Cosine annealing LR                                  │
│                                                                  │
│  [4] Guardar best model                                         │
│       │                                                          │
│       └─→ results/peru_rainfall/checkpoint.pth                 │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│  FASE 3: EVALUACIÓN Y ANÁLISIS                                   │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  [1] Métricas generales                                         │
│       │                                                          │
│       ├─→ F1-Score, AUC, Recall, Precision                     │
│       └─→ Confusion Matrix                                     │
│                                                                  │
│  [2] Análisis por fase ENSO                                     │
│       │                                                          │
│       ├─→ El Niño (años cálidos)                               │
│       ├─→ La Niña (años fríos)                                 │
│       └─→ Neutral                                              │
│                                                                  │
│  [3] Análisis por región                                        │
│       │                                                          │
│       ├─→ Costa Norte, Centro, Sur                             │
│       └─→ Sierra Norte, Sur                                    │
│                                                                  │
│  [4] Ablación de contexto                                       │
│       │                                                          │
│       ├─→ 90d vs 180d vs 1y vs 2y vs 3y                       │
│       └─→ Demostrar ventaja de contexto largo                 │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## ⏱️ Timeline Estimado

### **Semana 1: Validación Rápida**
```
Día 1: [████████░░] 80%
  ✓ Subir repo a GitHub
  ✓ Descargar checkpoint
  ✓ Descargar 3 años ERA5

Día 2: [██████████] 100%
  ✓ Preprocesar datos
  ✓ Verificar pipeline

Día 3-4: [██████████] 100%
  ✓ Entrenar Timer-XL (3 años)
  ✓ Primeros resultados

Día 5: [██████████] 100%
  ✓ Analizar métricas
  ✓ Ajustar hiperparámetros
```

### **Semana 2-3: Modelo Final**
```
Día 6-7: [████████░░] 80%
  ✓ Descargar 10 años ERA5
  ✓ Preprocesar

Día 8-10: [██████████] 100%
  ✓ Entrenar modelo final
  ✓ Guardar checkpoints

Día 11-14: [██████████] 100%
  ✓ Experimentos de ablación
  ✓ Análisis ENSO
```

### **Semana 4: Análisis y Escritura**
```
Día 15-20: [██████████] 100%
  ✓ Generar figuras
  ✓ Análisis estadístico
  ✓ Escribir resultados
```

---

## 📊 Experimentos Planificados

### **Experimento 1: Context Length Ablation**
```python
contextos = {
    '90d':   seq_len=180,   # Baseline corto
    '180d':  seq_len=360,   # Medio
    '1year': seq_len=730,   # Largo
    '2year': seq_len=1460,  # Muy largo ⭐
    '3year': seq_len=2190   # Extremo
}

# Hipótesis: 2 años es óptimo para ENSO
```

### **Experimento 2: Transfer Learning vs From Scratch**
```python
configs = {
    'pretrained': {
        'adaptation': True,
        'pretrain_path': 'checkpoints/timer_xl/checkpoint.pth'
    },
    'from_scratch': {
        'adaptation': False
    }
}

# Hipótesis: Transfer learning > From scratch
```

### **Experimento 3: Loss Function Comparison**
```python
losses = {
    'focal_loss': {'use_focal_loss': True},
    'weighted_ce': {'class_weights': '0.3,0.7'},
    'standard_ce': {}
}

# Hipótesis: Focal Loss > CE para desbalance
```

### **Experimento 4: Regional Analysis**
```python
regions = [
    'costa_norte',   # Tumbes, Piura
    'costa_centro',  # Lambayeque, La Libertad
    'costa_sur',     # Lima, Ica, Arequipa
    'sierra_norte',  # Andes norte
    'sierra_sur'     # Andes sur
]

# Objetivo: Identificar regiones más predecibles
```

---

## 🎓 Contribuciones de la Tesis

### **1. Técnica**
```
✓ Adaptación de Timer-XL para clasificación climática
✓ Pipeline completo de preprocesamiento ERA5
✓ Estrategia de transfer learning para series temporales climáticas
```

### **2. Metodológica**
```
✓ Análisis sistemático de contexto temporal (90d - 3 años)
✓ Evaluación por fase ENSO
✓ Comparación regional en Perú
```

### **3. Aplicada**
```
✓ Predicción de lluvias con horizonte de 24 horas
✓ Modelo reproducible para autoridades climáticas
✓ Base para sistemas de alerta temprana
```

---

## 📈 Métricas de Éxito

### **Mínimo Aceptable** ✅
```
F1-Score:  > 0.70
AUC-ROC:   > 0.80
Recall:    > 0.65
```

### **Deseable** ⭐
```
F1-Score:  > 0.75
AUC-ROC:   > 0.85
Recall:    > 0.70
```

### **Excelente** 🏆
```
F1-Score:  > 0.80
AUC-ROC:   > 0.90
Recall:    > 0.75
```

---

## 🚀 Comandos Rápidos

### **Preprocesamiento**
```bash
python preprocessing/preprocess_era5_peru.py \
    --input_dir datasets/raw_era5 \
    --output_dir datasets/processed \
    --years 2023,2024 \
    --target_horizon 24
```

### **Entrenamiento**
```bash
bash scripts/adaptation/peru_rainfall/train_timerxl_peru.sh
```

### **Entrenamiento Custom**
```bash
python run.py \
  --model timer_xl_classifier \
  --data PeruRainfall \
  --seq_len 1440 \
  --batch_size 128 \
  --learning_rate 1e-5 \
  --train_epochs 50 \
  --use_focal_loss \
  --adaptation \
  --pretrain_model_path checkpoints/timer_xl/checkpoint.pth
```

---

## 💡 Recomendaciones Finales

### **✅ DO (Hacer)**
- Empezar con 3 años para validar pipeline
- Usar Focal Loss para desbalance
- Guardar checkpoints en Google Drive frecuentemente
- Monitorear loss y métricas cada época
- Probar diferentes context lengths

### **❌ DON'T (No Hacer)**
- Entrenar sin GPU (muy lento)
- Usar batch_size muy grande (OOM)
- Ignorar el desbalance de clases
- Olvidar guardar resultados en Drive
- Empezar directo con 10 años sin probar

---

## 📞 Recursos

### **Documentación**
- Timer-XL Paper: https://arxiv.org/abs/2410.04803
- OpenLTM Repo: https://github.com/thuml/OpenLTM
- ERA5 Docs: https://cds.climate.copernicus.eu/

### **Checkpoint**
- Pre-trained weights: https://cloud.tsinghua.edu.cn/f/01c35ca13f474176be7b/

### **Soporte**
- Google Colab: https://colab.research.google.com/
- CDS API: https://cds.climate.copernicus.eu/api-how-to

---

**Status:** ✅ IMPLEMENTACIÓN COMPLETA  
**Fecha:** Octubre 8, 2025  
**Listo para:** Entrenamiento en Google Colab  

🚀 **¡TODO LISTO PARA EMPEZAR!** 🎓
