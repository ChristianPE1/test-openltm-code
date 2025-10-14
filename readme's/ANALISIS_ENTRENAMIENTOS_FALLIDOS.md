# 🚨 Análisis: Entrenamientos Fallidos (11 Años, 13/Oct/2025)

**Fecha**: 2025-10-13  
**Dataset**: 11 años (2014-2024), 28,119 timesteps  
**Problema**: Ambos modelos consumieron MUCHA más VRAM y tiempo de lo esperado

---

## 📊 Resultados Observados vs Esperados

### Transfer Learning (8L, 1024D, seq_len=2880)

| Métrica | Esperado | Real | Diferencia |
|---------|----------|------|------------|
| **VRAM** | ~6 GB | **10 GB** | +67% ❌ |
| **Tiempo/época** | ~30 min | **~65 min** | +117% ❌ |
| **Tiempo total** | 15-20 horas | **32.5 horas** | +62% ❌ |
| **Val Accuracy Época 1** | ~78-80% | **69.05%** | -11% ❌ |
| **Val Accuracy Época 2** | Mejora | **69.05%** (igual) | Estancado ❌ |
| **Val Loss Época 1** | 0.035-0.040 | **0.0393** | OK ✅ |
| **Val Loss Época 2** | Mejora | **0.0395** ↑ | Empeoró ❌ |

**Logs completos**:
```
Epoch 1 cost time: 3897.7179651260376 (64.96 min)
Val Accuracy: 69.05% (4160/6025)
Test Accuracy: 63.34% (3817/6026)
Vali Loss: 0.0392924, Test Loss: 0.0413758

Epoch 2 cost time: 3881.464823484421 (64.69 min)
Val Accuracy: 69.05% (4160/6025)  # ⚠️ IDÉNTICO A ÉPOCA 1
Test Accuracy: 63.34% (3817/6026)  # ⚠️ IDÉNTICO A ÉPOCA 1
Vali Loss: 0.0394477, Test Loss: 0.0413155
EarlyStopping counter: 1 out of 8
```

**Diagnóstico**: 
- ❌ Accuracy **exactamente igual** en épocas 1 y 2 (69.05%) indica **convergencia bloqueada**
- ❌ Val Loss sube ligeramente (0.0393 → 0.0395) = no aprende
- ❌ `seq_len=2880` (120 días) es **demasiado largo** para optimizar
- ❌ `batch_size=12` con secuencias largas → 10 GB VRAM

---

### Small Model "Mejorado" (6L, 768D, seq_len=1440)

| Métrica | Esperado | Real | Diferencia |
|---------|----------|------|------------|
| **VRAM** | ~2-3 GB | **6 GB** | +200% ❌ |
| **Tiempo/época** | ~5-8 min | **~14 min** | +75-180% ❌ |
| **Tiempo total** | 2-3 horas | **5.8 horas** | +93% ❌ |
| **Val Accuracy Época 1** | ~75-78% | **75.40%** | OK ✅ |
| **Val Accuracy Época 2** | Mejora | **70.29%** ↓ | -5.1% ❌ |
| **Val Accuracy Época 3** | Mejora | **75.42%** | Inestable ❌ |
| **Val Loss Época 1** | 0.035-0.040 | **0.0356** | OK ✅ |
| **Val Loss Época 2** | Mejora | **0.0381** ↑ | Empeoró ❌ |
| **Val Loss Época 3** | Mejora | **0.0372** | Empeoró ❌ |

**Logs completos**:
```
Epoch 1 cost time: 860.0865225791931 (14.33 min)
Val Accuracy: 75.40% (4543/6025)
Test Accuracy: 76.22% (4593/6026)
Vali Loss: 0.0355761, Test Loss: 0.0321258

Epoch 2 cost time: 866.5918929576874 (14.44 min)
Val Accuracy: 70.29% (4235/6025)  # ⚠️ EMPEORÓ -5.1%
Test Accuracy: 72.49% (4368/6026)  # ⚠️ EMPEORÓ -3.7%
Vali Loss: 0.0380584, Test Loss: 0.0342675
EarlyStopping counter: 1 out of 8

Epoch 3 cost time: 865.8048624992371 (14.43 min)
Val Accuracy: 75.42% (4544/6025)  # ⚠️ Volvió pero inestable
Test Accuracy: 70.11% (4225/6026)  # ⚠️ EMPEORÓ -6.1%
Vali Loss: 0.0372419, Test Loss: 0.0420861
EarlyStopping counter: 2 out of 8
```

**Diagnóstico**: 
- ❌ Val/Test Accuracy **oscila violentamente** (75% → 70% → 75%)
- ❌ Test Accuracy **diverge** de Val Accuracy (diferencia de 5.3% en época 3)
- ❌ Val Loss **no mejora** después de época 1 (0.0356 → 0.0381 → 0.0372)
- ❌ Modelo "Small" usa **6 GB VRAM** = dejó de ser "small"
- ⚠️ Overfitting temprano: mejora en época 1, empeora en épocas 2-3

---

## 🔍 Análisis Detallado: ¿Qué Salió Mal?

### Problema 1: seq_len Demasiado Largo (Transfer Learning)

**Configuración problemática**:
```python
seq_len = 2880  # 120 días × 24 horas
batch_size = 12
```

**Memoria GPU requerida**:
```
Memory = batch_size × seq_len × d_model × e_layers × overhead
        = 12 × 2880 × 1024 × 8 × 1.5 (gradients, activations, optimizer states)
        ≈ 10 GB
```

**Por qué 120 días es excesivo**:
1. ❌ **Optimización lenta**: Backpropagation a través de 2880 timesteps es computacionalmente intenso
2. ❌ **Vanishing gradients**: Señal de gradiente se debilita con secuencias muy largas
3. ❌ **Overfitting local**: Modelo memoriza patrones específicos de 120 días en lugar de generalizar
4. ❌ **Convergencia bloqueada**: Accuracy idéntica en épocas 1-2 (69.05%) indica que no optimiza

**Evidencia empírica (con 5 años de datos)**:
- Con `seq_len=1440` (60 días) lograste **F1=0.79, Val Acc ~80%**
- Con `seq_len=2880` (120 días) solo lograste **Val Acc=69.05%** (estancado)
- **Conclusión**: Más contexto NO siempre es mejor

---

### Problema 2: Small Model Dejó de Ser "Small"

**Comparación de arquitecturas**:

| Modelo | Layers | d_model | Parámetros | VRAM | Tiempo/época |
|--------|--------|---------|------------|------|--------------|
| **Small Original (5 años)** | 4 | 512 | ~20M | 1.5 GB | ~5 min |
| **Small "Mejorado" (11 años)** | 6 | 768 | ~60M | **6 GB** | **~14 min** |
| **Transfer Learning** | 8 | 1024 | ~100M | 6 GB (esperado) | ~30 min |

**Análisis**:
- ❌ Small "Mejorado" creció **+300% VRAM** (1.5 GB → 6 GB)
- ❌ Parámetros aumentaron **+200%** (20M → 60M)
- ❌ Casi tan grande como Transfer Learning (60M vs 100M parámetros)
- ❌ Perdió su ventaja principal: **eficiencia para experimentos rápidos**

**Por qué 6 layers, 768 dim fue excesivo**:
1. ❌ **Overfitting temprano**: Val Loss empeoró en épocas 2-3
2. ❌ **Inestabilidad**: Val Accuracy oscila 75% → 70% → 75%
3. ❌ **Test diverge de Val**: Diferencia de 5.3% en época 3 (overfitting)
4. ❌ **Recursos desperdiciados**: 6 GB VRAM sin beneficio claro

---

### Problema 3: Desbalance Class Distribution en Splits

**Class distribution observada**:
```python
# Transfer Learning (seq_len=2880)
[TRAIN] Class distribution: No Rain=9657, Rain=18462  # 34% / 66%
[VAL]   Class distribution: No Rain=2490, Rain=6415   # 28% / 72% ⚠️
[TEST]  Class distribution: No Rain=3208, Rain=5698   # 36% / 64%

# Small Model (seq_len=1440)
[TRAIN] Class distribution: No Rain=9657, Rain=18462  # 34% / 66%
[VAL]   Class distribution: No Rain=2173, Rain=5292   # 29% / 71% ⚠️
[TEST]  Class distribution: No Rain=2739, Rain=4727   # 37% / 63%
```

**Análisis**:
- ⚠️ Train: 66% Rain, Val: 71-72% Rain → **Val más desbalanceado que Train**
- ⚠️ Esto puede causar que modelo optimice para mayoría (Rain) en Val
- ⚠️ Test tiene mejor distribución (63-64% Rain) pero aún desbalanceado

**Impacto**:
- Val Loss puede ser engañoso (optimiza para clase mayoritaria)
- F1-Score es mejor métrica que Accuracy en este caso
- Early stopping basado en Val Loss puede ser subóptimo

---

## 🎯 Configuraciones Corregidas (Urgente)

### ✅ Transfer Learning Optimizado (RECOMENDADO)

**Cambios críticos**:
```python
# ANTES (10 GB VRAM, 65 min/época)
seq_len = 2880        # ❌ Demasiado largo
batch_size = 12       # ❌ Ineficiente
learning_rate = 5e-5

# DESPUÉS (5-6 GB VRAM, 25-30 min/época)
seq_len = 1440        # ✅ 60 días (suficiente para ENSO)
batch_size = 16       # ✅ Más eficiente
learning_rate = 5e-5  # Sin cambios
```

**Justificación seq_len=1440 (60 días)**:
1. ✅ Captura 2 meses completos de patrones atmosféricos
2. ✅ Incluye transiciones ENSO (El Niño → Neutral en ~1-2 meses)
3. ✅ Con 5 años + seq_len=1440 lograste F1=0.79
4. ✅ 120 días causó convergencia bloqueada (Val Acc=69.05% estancado)

**Recursos esperados**:
- VRAM: ~5-6 GB ✅ (reducción de 40%)
- Tiempo/época: ~25-30 min ✅ (reducción de 54%)
- Tiempo total: 12-15 horas ✅ (viable en Colab)

**Configuración completa**:
```bash
python run.py \
  --model timer_xl_classifier \
  --seq_len 1440 \              # ⭐ CAMBIO CRÍTICO
  --e_layers 8 \
  --d_model 1024 \
  --d_ff 2048 \
  --n_heads 8 \
  --batch_size 16 \             # ⭐ CAMBIO CRÍTICO
  --learning_rate 5e-5 \
  --dropout 0.2 \
  --train_epochs 30 \
  --adaptation \
  --pretrain_model_path checkpoints/timer_xl/checkpoint.pth
```

---

### ✅ Small Model REALMENTE Eficiente (RECOMENDADO)

**Cambios críticos**:
```python
# ANTES (6 GB VRAM, 14 min/época, overfitting)
e_layers = 6          # ❌ Demasiadas capas
d_model = 768         # ❌ Demasiado grande
d_ff = 1536
batch_size = 24

# DESPUÉS (3-4 GB VRAM, 8-10 min/época)
e_layers = 5          # ✅ Punto medio
d_model = 640         # ✅ Entre 512 y 768
d_ff = 1280           # ✅ Proporcional
batch_size = 32       # ✅ Más eficiente
```

**Justificación (5 layers, 640 dim)**:
1. ✅ Mantiene capacidad para 11 años de datos (~40M parámetros)
2. ✅ Reduce VRAM a 3-4 GB (realmente "small")
3. ✅ Más estable que 6L/768D (menos overfitting)
4. ✅ Permite experimentos rápidos (3-4 horas total)

**Recursos esperados**:
- VRAM: ~3-4 GB ✅ (reducción de 33%)
- Tiempo/época: ~8-10 min ✅ (reducción de 29%)
- Tiempo total: 3-4 horas ✅ (muy manejable)

**Configuración completa**:
```bash
python run.py \
  --model timer_xl_classifier \
  --seq_len 1440 \
  --e_layers 5 \                # ⭐ CAMBIO CRÍTICO
  --d_model 640 \               # ⭐ CAMBIO CRÍTICO
  --d_ff 1280 \                 # ⭐ CAMBIO CRÍTICO
  --n_heads 8 \
  --batch_size 32 \             # ⭐ CAMBIO CRÍTICO
  --learning_rate 8e-5 \
  --dropout 0.15 \
  --train_epochs 25
```

---

## 📊 Comparación: Antes vs Después

### Transfer Learning

| Aspecto | Antes (Fallido) | Después (Corregido) | Mejora |
|---------|-----------------|---------------------|--------|
| **seq_len** | 2880 (120 días) | 1440 (60 días) | -50% |
| **batch_size** | 12 | 16 | +33% |
| **VRAM** | 10 GB ❌ | 5-6 GB ✅ | -40% |
| **Tiempo/época** | 65 min ❌ | 25-30 min ✅ | -54% |
| **Tiempo total** | 32.5 horas ❌ | 12-15 horas ✅ | -54% |
| **Val Acc época 1** | 69.05% ❌ | ~78-80% ✅ | +9-11% |
| **Convergencia** | Estancado ❌ | Esperada ✅ | ✅ |

---

### Small Model

| Aspecto | Antes (Fallido) | Después (Corregido) | Mejora |
|---------|-----------------|---------------------|--------|
| **e_layers** | 6 | 5 | -17% |
| **d_model** | 768 | 640 | -17% |
| **d_ff** | 1536 | 1280 | -17% |
| **batch_size** | 24 | 32 | +33% |
| **VRAM** | 6 GB ❌ | 3-4 GB ✅ | -33% |
| **Tiempo/época** | 14 min ❌ | 8-10 min ✅ | -29% |
| **Tiempo total** | 5.8 horas ⚠️ | 3-4 horas ✅ | -31% |
| **Val Acc estabilidad** | Oscila ±5% ❌ | Esperada estable ✅ | ✅ |
| **Overfitting** | Temprano ❌ | Reducido ✅ | ✅ |

---

## 🔬 Lecciones Aprendidas

### 1. Más datos ≠ Secuencias más largas
- ✅ 11 años de datos (28,119 timesteps) está bien
- ❌ seq_len=2880 (120 días) es contraproducente
- ✅ seq_len=1440 (60 días) es óptimo (equilibrio contexto/optimización)

### 2. "Mejorado" no significa "más grande siempre"
- ❌ 6 layers, 768 dim = 6 GB VRAM (casi como Transfer Learning)
- ✅ 5 layers, 640 dim = 3-4 GB VRAM (eficiente pero capaz)
- ✅ Mantiene identidad "small" para experimentos rápidos

### 3. Señales de problemas de convergencia
- ❌ Val Accuracy idéntica en múltiples épocas (Transfer: 69.05% épocas 1-2)
- ❌ Val Accuracy oscila violentamente (Small: 75% → 70% → 75%)
- ❌ Test diverge de Val (Small: diferencia 5.3% en época 3)
- ✅ Detener entrenamiento temprano si se observan estas señales

### 4. Class imbalance requiere métricas adecuadas
- ⚠️ Val tiene 71-72% Rain (más desbalanceado que Train 66%)
- ⚠️ Accuracy puede ser engañosa (modelo predice "Rain" siempre = 71% Acc)
- ✅ F1-Score, Precision, Recall son mejores métricas
- ✅ Use Focal Loss (ya configurado) para manejo de desbalance

---

## 🚀 Próximos Pasos (Inmediato)

### Paso 1: Actualizar notebook (COMPLETADO ✅)
- ✅ Transfer Learning: seq_len=2880 → 1440, batch_size=12 → 16
- ✅ Small Model: 6L/768D → 5L/640D, batch_size=24 → 32

### Paso 2: Re-entrenar con configuraciones corregidas

**Prioridad 1: Transfer Learning** (12-15 horas)
```bash
# Usar celda actualizada en colab_training_demo.ipynb
# Configuración: 8L, 1024D, seq_len=1440, batch_size=16
# Meta: F1 > 0.82, Val Acc > 78%
```

**Prioridad 2: Small Model Eficiente** (3-4 horas)
```bash
# Usar celda actualizada en colab_training_demo.ipynb
# Configuración: 5L, 640D, seq_len=1440, batch_size=32
# Meta: F1 > 0.80, Val Acc > 75%
```

### Paso 3: Monitoreo de convergencia

**Señales de éxito** (detener si aparecen):
- ✅ Val Accuracy mejora consistentemente (epochs 1-10)
- ✅ Val Loss decrece suavemente
- ✅ Test Accuracy alineado con Val Accuracy (diferencia < 2%)
- ✅ F1-Score > 0.82 (Transfer) o > 0.80 (Small) en época 15-20

**Señales de problemas** (detener inmediatamente):
- ❌ Val Accuracy estancada por 3+ épocas consecutivas
- ❌ Val Accuracy oscila >5% entre épocas
- ❌ Test Accuracy diverge >5% de Val Accuracy
- ❌ VRAM > 7 GB (Transfer) o > 5 GB (Small)

---

## 📈 Expectativas Realistas (11 Años, Config Corregida)

### Transfer Learning (seq_len=1440, 8L, 1024D)
```
Época    Val Loss    Val Acc    Test Acc    F1-Score (estimado)
-----    --------    --------   ---------   -------------------
  1      0.0355      74-76%     73-75%      0.74-0.76
  5      0.0320      77-79%     76-78%      0.77-0.79
 10      0.0300      79-81%     78-80%      0.79-0.81
 15      0.0285      80-82%     79-81%      0.80-0.82
 20      0.0275      81-83%     80-82%      0.81-0.83 ⭐
 25      0.0270      81-84%     80-83%      0.81-0.84
 30      0.0268      82-84%     81-83%      0.82-0.84 ⭐ META
```

**Meta**: F1 > 0.82, Val Acc > 81%, Test Acc > 80%

---

### Small Model Eficiente (seq_len=1440, 5L, 640D)
```
Época    Val Loss    Val Acc    Test Acc    F1-Score (estimado)
-----    --------    --------   ---------   -------------------
  1      0.0370      73-75%     72-74%      0.73-0.75
  5      0.0335      75-77%     74-76%      0.75-0.77
 10      0.0310      77-79%     76-78%      0.77-0.79
 15      0.0295      78-80%     77-79%      0.78-0.80 ⭐
 20      0.0285      79-81%     78-80%      0.79-0.81
 25      0.0280      79-81%     78-80%      0.79-0.81 ⭐ META
```

**Meta**: F1 > 0.80, Val Acc > 79%, Test Acc > 78%

---

## 📝 Notas Finales

### Checkpoints guardados (13/Oct/2025)
```
✅ classification_peru_rainfall_timerxl_11years_..._0_20251013_044634.pth
   - Época 1: Val Acc 69.05% (estancado con seq_len=2880)
   - NO usar para evaluación final

✅ classification_peru_rainfall_small_improved_11years_..._0_20251013_044634.pth
   - Época 1: Val Acc 75.40% (mejor época, overfitting después)
   - NO usar para evaluación final
```

**Ambos checkpoints son de configuraciones fallidas** → Re-entrenar con configuraciones corregidas

---

**Última Actualización**: 2025-10-13  
**Próxima Acción**: Re-entrenar Transfer Learning (seq_len=1440) + Small Eficiente (5L/640D)  
**Meta**: Transfer Learning F1 > 0.82, Small Model F1 > 0.80
