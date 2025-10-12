# 📊 Estrategia para Dataset Extendido (10-20 Años)

**Fecha**: 2025-01-11  
**Dataset**: ERA5 2014-2024 (11 años) → Expandible a 2005-2024 (20 años)

---

## 🎯 Cambio de Estrategia: 5 años → 10-20 años

### ¿Por qué más años es crítico para ENSO?

**Ciclos ENSO completos**:
- **5 años (2020-2024)**: Solo 1-2 ciclos ENSO completos
  - El Niño 2023-2024 (moderado/fuerte)
  - La Niña 2020-2022 (débil/moderado)
  - Datos insuficientes para patrones complejos

- **11 años (2014-2024)**: 3-4 ciclos ENSO completos ✅
  - El Niño 2023-2024 (fuerte)
  - La Niña 2020-2022 (moderado)
  - El Niño 2015-2016 (SUPER FUERTE) ⭐ **Crítico**
  - Neutral 2013-2014, 2017-2019

- **20 años (2005-2024)**: 6-8 ciclos ENSO completos ✅✅
  - El Niño 2023-2024 (fuerte)
  - La Niña 2020-2022 (moderado)
  - El Niño 2015-2016 (super fuerte)
  - La Niña 2010-2012 (fuerte)
  - El Niño 2009-2010 (moderado)
  - La Niña 2007-2008 (moderado)
  - El Niño 2006-2007 (débil)

**Ventajas de más años**:
1. ✅ **Diversidad de eventos**: Captura El Niño débil, moderado, fuerte y SUPER FUERTE
2. ✅ **Transiciones**: Aprende patrones de transición La Niña → El Niño → Neutral
3. ✅ **Robustez**: Reduce overfitting al tener más ejemplos de cada fase
4. ✅ **Eventos raros**: Incluye eventos extremos como El Niño 2015-2016 (devastador en Perú)

---

## 🔬 Experimentos Recomendados (Dataset Grande)

### Experimento 1: Transfer Learning vs From Scratch (11 años)

#### Hipótesis
> "Con 11 años de datos, Transfer Learning sigue siendo superior porque el checkpoint ERA5 preentrenado contiene conocimiento general de patrones atmosféricos que acelera convergencia"

#### Configuración A: Transfer Learning (Recomendado)
```bash
python run.py \
  --model timer_xl_classifier \
  --seq_len 2880 \      # 120 días (captura ciclo ENSO completo)
  --e_layers 8 \
  --d_model 1024 \
  --batch_size 12 \     # Reducido por secuencia larga
  --learning_rate 5e-5 \  # Más alto que 1e-5 (más datos = más LR)
  --dropout 0.2 \       # Mayor regularización
  --train_epochs 30 \
  --adaptation \
  --pretrain_model_path checkpoints/timer_xl/checkpoint.pth
```

**Expectativa**: F1 = 0.82-0.85 (mejor que 0.79 con 5 años)

#### Configuración B: From Scratch (Comparación)
```bash
python run.py \
  --model timer_xl_classifier \
  --seq_len 2880 \
  --e_layers 8 \
  --d_model 1024 \
  --batch_size 12 \
  --learning_rate 1e-4 \  # Más alto para convergencia desde cero
  --dropout 0.2 \
  --train_epochs 50       # Más épocas sin pretrain
```

**Expectativa**: F1 = 0.80-0.83 (puede competir con Transfer Learning)

#### Análisis Comparativo
| Aspecto | Transfer Learning | From Scratch | Ganador |
|---------|-------------------|--------------|---------|
| **Convergencia** | Rápida (30 épocas) | Lenta (50 épocas) | Transfer Learning |
| **F1-Score esperado** | 0.82-0.85 | 0.80-0.83 | Transfer Learning |
| **Tiempo total** | 15-20 horas | 25-35 horas | Transfer Learning |
| **Uso de memoria** | 6 GB | 6 GB | Empate |
| **Robustez** | Alta (conocimiento previo) | Media | Transfer Learning |

**Conclusión esperada**: Transfer Learning sigue siendo superior, pero la diferencia se reduce con más datos (ΔF1 ≈ 0.02-0.03 vs ΔF1 ≈ 0.10 con 5 años)

---

### Experimento 2: Small Model Mejorado (11 años)

#### Motivación
> "El Small Model original (4 layers, 512 dim) funcionó bien con 5 años (F1=0.78), pero con 11 años puede necesitar más capacidad"

#### Configuración Mejorada
```bash
python run.py \
  --model timer_xl_classifier \
  --seq_len 1440 \      # 60 días (más que 720h = 30 días original)
  --e_layers 6 \        # MÁS capas (antes 4)
  --d_model 768 \       # MÁS dimensiones (antes 512)
  --d_ff 1536 \         # Proporcional a d_model
  --batch_size 24 \     # Optimizado para 2-3 GB VRAM
  --learning_rate 8e-5 \
  --dropout 0.15 \      # Mayor regularización
  --train_epochs 25
```

**Mejoras vs versión original**:
1. ✅ +50% capacidad (6 layers vs 4 layers)
2. ✅ +50% dimensionalidad (768 vs 512)
3. ✅ +100% contexto (1440h vs 720h)
4. ✅ Sigue siendo eficiente (2-3 GB vs 1.5 GB)

**Expectativa**: F1 = 0.80-0.82 (mejora de +2-4% vs F1=0.78 con 5 años)

---

### Experimento 3: Análisis de Longitud de Contexto (11 años)

#### Hipótesis
> "Con 11 años de datos, contextos más largos (90-120 días) capturan mejor los ciclos ENSO que contextos cortos (30-60 días)"

#### Configuraciones a Probar
```python
context_experiments = [
    {"seq_len": 720,  "dias": 30,  "descripcion": "Corto - Captura variabilidad mensual"},
    {"seq_len": 1440, "dias": 60,  "descripcion": "Medio - Captura 2 meses"},
    {"seq_len": 2160, "dias": 90,  "descripcion": "Largo - Mínimo ciclo ENSO (3 meses)"},
    {"seq_len": 2880, "dias": 120, "descripcion": "Muy Largo - Ciclo ENSO completo"},
]
```

**Estrategia de evaluación**:
1. Entrenar mismo modelo (8 layers, 1024 dim) con 4 contextos diferentes
2. Medir F1-Score en 3 fases ENSO (El Niño, La Niña, Neutral)
3. Identificar punto de saturación (donde ΔF1 < 0.02 con más contexto)

**Resultados esperados**:
```
Contexto | F1 Global | F1 El Niño | F1 La Niña | F1 Neutral | Tiempo/Época
---------|-----------|------------|------------|------------|-------------
30 días  | 0.78      | 0.75       | 0.79       | 0.80       | 20 min
60 días  | 0.82      | 0.80       | 0.83       | 0.83       | 30 min
90 días  | 0.84      | 0.83       | 0.85       | 0.84       | 40 min
120 días | 0.85      | 0.84       | 0.86       | 0.85       | 50 min ⭐ Óptimo
```

**Conclusión esperada**: 90-120 días es óptimo (captura ciclo ENSO completo sin exceso computacional)

---

## 📈 Configuraciones Optimizadas para Dataset Grande

### Transfer Learning Timer-XL (8 layers, 1024 dim)

```python
config_transfer_learning = {
    # Arquitectura
    "e_layers": 8,
    "d_model": 1024,
    "d_ff": 2048,
    "n_heads": 8,
    
    # Contexto (120 días para capturar ciclo ENSO)
    "seq_len": 2880,  # 120 días × 24 horas
    "input_token_len": 96,
    "output_token_len": 96,
    
    # Optimización
    "batch_size": 12,  # Reducido por secuencia larga
    "learning_rate": 5e-5,  # Más alto con más datos
    "dropout": 0.2,  # Mayor regularización
    "train_epochs": 30,
    "patience": 8,
    
    # Transfer Learning
    "adaptation": True,
    "pretrain_model_path": "checkpoints/timer_xl/checkpoint.pth",
    
    # Pérdida
    "use_focal_loss": True,  # Para desbalance de clases
    "loss": "CE",
    
    # Scheduler
    "cosine": True,
    "tmax": 30
}
```

**Recursos**:
- VRAM: ~6 GB
- Tiempo por época: ~30-45 min (depende de GPU)
- Tiempo total: 15-20 horas (30 épocas)

**F1-Score esperado**: 0.82-0.85

---

### Small Model Mejorado (6 layers, 768 dim)

```python
config_small_improved = {
    # Arquitectura (MEJORADA vs versión original)
    "e_layers": 6,  # Antes: 4
    "d_model": 768,  # Antes: 512
    "d_ff": 1536,  # Antes: 1024
    "n_heads": 8,
    
    # Contexto (60 días, más que 30 días original)
    "seq_len": 1440,  # 60 días × 24 horas
    "input_token_len": 96,
    "output_token_len": 96,
    
    # Optimización
    "batch_size": 24,  # Optimizado para 2-3 GB VRAM
    "learning_rate": 8e-5,  # Adaptativo
    "dropout": 0.15,  # Mayor regularización
    "train_epochs": 25,
    "patience": 8,
    
    # Pérdida
    "use_focal_loss": True,
    "loss": "CE",
    
    # Scheduler
    "cosine": True,
    "tmax": 25
}
```

**Recursos**:
- VRAM: ~2-3 GB
- Tiempo por época: ~15-20 min
- Tiempo total: 6-8 horas (25 épocas)

**F1-Score esperado**: 0.80-0.82

---

## 🎯 Plan de Ejecución (11 Años de Datos)

### Fase 1: Preprocesamiento (1-2 horas)
```bash
# Descargar datos ERA5 2014-2024 (11 años)
python preprocessing/preprocess_era5_peru.py \
    --input_dir datasets/raw_era5 \
    --output_dir datasets/processed \
    --years 2014,2015,2016,2017,2018,2019,2020,2021,2022,2023,2024 \
    --target_horizon 24 \
    --threshold 0.0001  # 0.1 mm
```

**Datos esperados**:
- Total timesteps: ~96,360 (11 años × 365 días × 24 horas)
- Total samples (con seq_len=2880): ~93,480
- Distribución ENSO: ~40% Neutral, ~30% La Niña, ~30% El Niño

### Fase 2: Entrenamiento Transfer Learning (15-20 horas)
```bash
python run.py \
  --task_name classification \
  --model timer_xl_classifier \
  # ... configuración Transfer Learning arriba ...
  --des 'Peru_Rainfall_Transfer_Learning_11Years_2014_2024'
```

**Monitoreo**:
- Guardar checkpoint cada época
- Backup a Drive cada 5 épocas (usar celda de backup)
- Early stopping si no mejora por 8 épocas

**Expectativa**: F1 > 0.82

### Fase 3: Entrenamiento Small Model Mejorado (6-8 horas)
```bash
python run.py \
  --task_name classification \
  --model timer_xl_classifier \
  # ... configuración Small Model Mejorado arriba ...
  --des 'Peru_Rainfall_Small_Improved_11Years_2014_2024'
```

**Expectativa**: F1 > 0.80

### Fase 4: Comparación Transfer Learning vs From Scratch (Opcional, 25-35 horas)
```bash
# Solo si Transfer Learning NO supera F1 = 0.83
python run.py \
  --task_name classification \
  --model timer_xl_classifier \
  --seq_len 2880 \
  --e_layers 8 \
  --d_model 1024 \
  --batch_size 12 \
  --learning_rate 1e-4 \  # Sin pretrain necesita más LR
  --train_epochs 50 \     # Más épocas
  --des 'Peru_Rainfall_From_Scratch_11Years_2014_2024'
```

### Fase 5: Validación ENSO (2-3 horas)
```python
# Etiquetar datos por fase ENSO
python scripts/label_enso_phases.py \
    --oni_file data/oni_index_2014_2024.csv \
    --data_file datasets/processed/peru_rainfall_cleaned.csv

# Evaluar por fase
python scripts/evaluate_by_enso_phase.py \
    --checkpoint checkpoints/.../checkpoint.pth \
    --enso_phases data/enso_labeled_data.csv
```

**Métricas objetivo**:
- F1 El Niño > 0.80
- F1 La Niña > 0.82
- F1 Neutral > 0.83
- Consistencia: |ΔF1| < 0.08

---

## 📊 Resultados Esperados (11 Años vs 5 Años)

### Comparación Transfer Learning

| Métrica | 5 Años (2020-2024) | 11 Años (2014-2024) | Mejora |
|---------|-------------------|---------------------|--------|
| **F1-Score Global** | 0.79 | **0.82-0.85** | +3-6% ✅ |
| **Precision** | 0.71 | **0.78-0.82** | +7-11% ✅ |
| **Recall** | 0.89 | **0.86-0.88** | -1-3% (aceptable) |
| **Accuracy** | 72.87% | **78-82%** | +5-9% ✅ |
| **False Positives** | 49% | **30-35%** | -14-19% ✅ |
| **F1 El Niño** | ~0.76* | **0.80-0.83** | +4-7% ✅ |
| **F1 La Niña** | ~0.80* | **0.82-0.85** | +2-5% ✅ |
| **F1 Neutral** | ~0.80* | **0.83-0.86** | +3-6% ✅ |

*Valores estimados, no medidos directamente en 5 años

### Comparación Small Model

| Métrica | 5 Años (4L, 512D) | 11 Años (6L, 768D) | Mejora |
|---------|-------------------|-------------------|--------|
| **F1-Score** | 0.78 | **0.80-0.82** | +2-4% ✅ |
| **Precision** | 0.87 | **0.84-0.86** | -1-3% (trade-off) |
| **Recall** | 0.70 | **0.76-0.78** | +6-8% ✅ |
| **Accuracy** | 76.93% | **78-80%** | +1-3% ✅ |

---

## 🎯 Decisiones Estratégicas Basadas en Resultados (11 Años)

### Escenario 1: Transfer Learning F1 > 0.83 ✅

**Interpretación**: Transfer Learning + más datos es la mejor estrategia

**Próximo paso**: Mejoras arquitectónicas
1. ENSO-aware TimeAttention
2. Máscara Kronecker adaptativa
3. Multi-scale temporal features

**Objetivo**: F1 > 0.87

---

### Escenario 2: Transfer Learning F1 = 0.80-0.83 (Zona intermedia)

**Interpretación**: Transfer Learning funciona pero puede mejorar

**Próximo paso (Opción A)**: Probar From Scratch
- Si From Scratch > Transfer Learning: Usar From Scratch de base
- Si Transfer Learning ≥ From Scratch: Continuar con Transfer Learning

**Próximo paso (Opción B)**: Optimización Transfer Learning
- Entrenar 15-20 épocas adicionales
- Ajustar class weights
- Fine-tuning learning rate

---

### Escenario 3: Small Model F1 ≥ Transfer Learning F1

**Interpretación**: Modelo más pequeño es suficiente (sorprendente pero posible)

**Próximo paso**: Usar Small Model como baseline
- Más rápido para experimentos
- Permite probar más configuraciones
- Menos recursos computacionales

---

## 💡 Recomendaciones Prácticas

### 1. Gestión de Recursos en Colab

**Transfer Learning (6 GB VRAM)**:
- ✅ Google Colab Pro (15 GB VRAM disponible)
- ✅ Google Colab Gratis (12 GB VRAM, ajustado)
- ⚠️ Reducir batch_size si OOM: 12 → 8 → 6

**Small Model (2-3 GB VRAM)**:
- ✅ Google Colab Gratis
- ✅ Cualquier GPU moderna

### 2. Backup Estratégico

**Cada 5 épocas**: Ejecutar celda de backup
```python
# Automático cada 5 épocas
if epoch % 5 == 0:
    !python backup_checkpoint.py
```

**Antes de desconectar**: SIEMPRE ejecutar celda final de backup

### 3. Monitoreo de Convergencia

**Señales de buen entrenamiento**:
- ✅ Train loss decrece consistentemente
- ✅ Val loss decrece (puede estancarse después de época 15-20)
- ✅ F1-Score sube en val cada 2-3 épocas
- ✅ Early stopping actúa después de época 20+

**Señales de problemas**:
- ❌ Train loss crece o se estanca desde inicio
- ❌ Val loss sube consistentemente (overfitting)
- ❌ F1-Score no supera 0.70 después de 10 épocas
- ❌ Early stopping actúa antes de época 10

---

## 🚀 Siguiente Ejecución (Inmediata)

### Paso 1: Descargar datos 2014-2024 (1-2 horas)

```bash
# Descargar 11 años de ERA5
# Usar CDS API o manual download
```

### Paso 2: Preprocesar (30 min)

```bash
python preprocessing/preprocess_era5_peru.py \
    --years 2014,2015,2016,2017,2018,2019,2020,2021,2022,2023,2024
```

### Paso 3: Entrenar Transfer Learning (15-20 horas)

**Usar configuración optimizada en celda del notebook**

**Monitoreo**:
- Backup cada 5 épocas
- Detener manualmente si F1 > 0.83 en época 20-25 (ya es excelente)

### Paso 4: Entrenar Small Model Mejorado (6-8 horas)

**Comparar con Transfer Learning**

### Paso 5: Decidir siguiente paso según resultados

---

**Última Actualización**: 2025-01-11  
**Próxima Ejecución**: Dataset 2014-2024 (11 años)  
**Meta**: F1-Score > 0.82 (Transfer Learning) y > 0.80 (Small Model)
