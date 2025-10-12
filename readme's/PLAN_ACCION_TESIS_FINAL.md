# 🎓 Plan de Acción Final para Tesis Timer-XL

**Fecha**: 2025-01-11  
**Autor**: Christian  
**Estado**: Fase de Decisión Estratégica

---

## 📊 Resultados Actuales (COMPLETADO)

### Modelos Entrenados

| Modelo | F1-Score | Precision | Recall | Accuracy | Tiempo | VRAM |
|--------|----------|-----------|--------|----------|--------|------|
| **Transfer Learning** | **0.79** ✅ | 0.71 | **0.89** ✅ | 72.87% | 60 min (5 épocas) | 8 GB |
| **Small Model** | **0.78** | **0.87** ✅ | 0.70 | **76.93%** ✅ | 15 min | 1.5 GB |
| Big Scratch | ~0.55 | - | - | 57.98% | 40 min | 6 GB |

### ✅ Hallazgos Clave

1. **Transfer Learning (F1=0.79)**: 
   - Mayor recall (89%) → mejor detección de lluvias
   - Menor precisión (71%) → más falsos positivos (49%)
   - Solo 5 épocas → **TIENE POTENCIAL DE MEJORA**

2. **Small Model (F1=0.78)**:
   - Mayor precisión (87%) → menos falsos positivos (14%)
   - Menor recall (70%) → pierde más eventos reales
   - Ultra eficiente → 4x más rápido

3. **Zona intermedia (0.75 < F1 < 0.80)**: 
   - No es "excelente" pero tampoco "malo"
   - Permite dos caminos válidos para la tesis

---

## 🎯 DECISIÓN ESTRATÉGICA: Dos Caminos Posibles

### ⭐ **OPCIÓN A: Mejoras Arquitectónicas (RECOMENDADO)**

**Razón**: F1=0.79 es mejorable con optimizaciones y modificaciones arquitectónicas

#### Objetivo de Tesis
> *"Optimización de Timer-XL mediante mejoras en TimeAttention y transfer learning para predicción de lluvias con influencia ENSO, logrando F1-Score > 0.85"*

### Plan de Trabajo (3-4 semanas)

### **Semana 1: Optimización de Transfer Learning Actual**

**Objetivo**: Mejorar F1 de 0.79 a 0.82+ con ajustes simples

**Experimentos**:

1. **Entrenar más épocas** (15-20 épocas totales)
   ```bash
   # Continuar desde checkpoint actual
   python run.py \
     --task_name classification \
     --model timer_xl_classifier \
     --train_epochs 20 \
     --learning_rate 5e-6 \  # Reducir LR para fine-tuning
     --adaptation \
     --pretrain_model_path checkpoints/.../checkpoint.pth
   ```
   - **Expectativa**: F1 → 0.81-0.82
   - **Tiempo**: 3 horas (15 épocas adicionales)

2. **Optimizar Class Weights** para balancear Precision-Recall
   ```python
   # En timer_xl_classifier.py
   class_weights = [1.5, 1.0]  # Penalizar más No Rain
   criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights))
   ```
   - **Expectativa**: Reducir falsos positivos de 49% → 35%
   - **F1 esperado**: 0.82-0.83

3. **Learning Rate Schedule optimizado**
   ```python
   # Cosine annealing con warmup
   scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
   ```
   - **Expectativa**: Convergencia más estable
   - **F1 esperado**: 0.83-0.84

### **Semana 2: Modificaciones Arquitectónicas en Timer-XL**

**Objetivo**: Mejorar captura de patrones ENSO con mejoras específicas

**Experimento 1: ENSO-Aware TimeAttention**

Modificar `layers/SelfAttention_Family.py` para incluir phase embeddings:

```python
class ENSOAwareTimeAttention(TimeAttention):
    def __init__(self, d_model, n_heads, enso_dim=3):
        super().__init__(d_model, n_heads)
        # Embeddings para fases ENSO (El Niño, La Niña, Neutral)
        self.enso_embedding = nn.Embedding(3, enso_dim)
        self.enso_projection = nn.Linear(d_model + enso_dim, d_model)
        
    def forward(self, x, enso_phase):
        # x: [B, L, D]
        # enso_phase: [B] (0=Neutral, 1=El Niño, 2=La Niña)
        
        # 1. Agregar ENSO embeddings
        enso_emb = self.enso_embedding(enso_phase).unsqueeze(1)  # [B, 1, enso_dim]
        enso_emb = enso_emb.expand(-1, x.size(1), -1)  # [B, L, enso_dim]
        
        # 2. Concatenar y proyectar
        x_enso = torch.cat([x, enso_emb], dim=-1)  # [B, L, D+enso_dim]
        x_enso = self.enso_projection(x_enso)  # [B, L, D]
        
        # 3. Aplicar TimeAttention original
        return super().forward(x_enso)
```

**Expectativa**: F1 → 0.84-0.85 (mejor en fases ENSO extremas)

**Experimento 2: Máscara Kronecker Adaptativa**

Modificar `layers/Attn_Bias.py` para ajustar máscara según fase ENSO:

```python
class AdaptiveKroneckerMask(KroneckerMask):
    def forward(self, x, enso_phase):
        # Generar máscara base
        mask = super().forward(x)
        
        # Ajustar según fase ENSO
        if enso_phase in [1, 2]:  # El Niño o La Niña
            # Aumentar receptive field para eventos extremos
            mask = self.expand_mask(mask, factor=1.5)
        
        return mask
```

**Expectativa**: Mejor captura de patrones de largo plazo → F1 → 0.85+

**Experimento 3: Multi-Scale Temporal Features**

Agregar módulo de extracción multi-escala antes de TimeAttention:

```python
class MultiScaleTemporalModule(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # Convoluciones con diferentes receptive fields
        self.conv_daily = nn.Conv1d(d_model, d_model//3, kernel_size=24)  # 1 día
        self.conv_weekly = nn.Conv1d(d_model, d_model//3, kernel_size=168)  # 7 días
        self.conv_monthly = nn.Conv1d(d_model, d_model//3, kernel_size=720)  # 30 días
        
    def forward(self, x):
        # x: [B, L, D]
        x_t = x.transpose(1, 2)  # [B, D, L]
        
        feat_daily = self.conv_daily(x_t)
        feat_weekly = self.conv_weekly(x_t)
        feat_monthly = self.conv_monthly(x_t)
        
        # Concatenar características multi-escala
        feat_multi = torch.cat([feat_daily, feat_weekly, feat_monthly], dim=1)
        return feat_multi.transpose(1, 2)  # [B, L', D]
```

**Expectativa**: Capturar mejor la estacionalidad → F1 → 0.86+

### **Semana 3: Ablation Studies y Validación**

**Objetivo**: Demostrar que cada mejora contribuye al rendimiento

**Experimentos**:

1. **Baseline**: Transfer Learning original (F1=0.79)
2. **+ Más épocas**: F1 esperado 0.82
3. **+ Class weights**: F1 esperado 0.83
4. **+ ENSO-aware attention**: F1 esperado 0.85
5. **+ Máscara adaptativa**: F1 esperado 0.85
6. **+ Multi-scale**: F1 esperado 0.86
7. **Todos juntos**: F1 esperado **0.87-0.88** ✅

**Tabla de Resultados**:

| Configuración | F1-Score | Precision | Recall | ΔF1 vs Baseline |
|---------------|----------|-----------|--------|-----------------|
| Baseline (Transfer Learning) | 0.79 | 0.71 | 0.89 | - |
| + 15 épocas | 0.82 | 0.75 | 0.90 | +0.03 |
| + Class weights | 0.83 | 0.80 | 0.86 | +0.04 |
| + ENSO-aware attention | 0.85 | 0.82 | 0.88 | +0.06 |
| + Máscara adaptativa | 0.85 | 0.83 | 0.87 | +0.06 |
| + Multi-scale | 0.86 | 0.84 | 0.88 | +0.07 |
| **TODO** | **0.87** | **0.85** | **0.89** | **+0.08** ✅ |

##### **Semana 4: Validación ENSO y Escritura**

**Objetivo**: Validar que mejoras funcionan en todas las fases ENSO

1. **Etiquetar datos por fase ENSO**:
   ```python
   # Descargar ONI index 2020-2024
   # Clasificar cada sample: El Niño (1), La Niña (2), Neutral (0)
   ```

2. **Evaluar modelo mejorado por fase**:
   ```
   F1 El Niño:   0.85-0.87
   F1 La Niña:   0.87-0.89
   F1 Neutral:   0.86-0.88
   ```

3. **Escribir Metodología + Resultados**:
   - Descripción de cada mejora arquitectónica
   - Ablation studies con gráficos
   - Validación ENSO
   - Conclusiones

---

### 🔬 **OPCIÓN B: Análisis ENSO + Contexto Temporal**

**Razón**: F1=0.79 es suficiente para análisis robusto de contexto óptimo

#### Objetivo de Tesis
> *"Determinación del contexto temporal óptimo para predicción de lluvias con influencia ENSO usando Timer-XL, validando rendimiento por fase climática"*

### Plan de Trabajo (2-3 semanas)

### **Semana 1: Etiquetado ENSO y Validación por Fase**

1. **Descargar índice ONI** (Oceanic Niño Index 2020-2024)
2. **Etiquetar cada muestra** con fase ENSO
3. **Evaluar Small Model por fase**:
   ```
   F1 El Niño:   ?
   F1 La Niña:   ?
   F1 Neutral:   ?
   ```

### **Semana 2: Experimentos de Contexto**

Entrenar Small Model con 5 longitudes de contexto:

```bash
for seq_len in 90 180 365 730 1095; do
  python run.py \
    --seq_len $((seq_len * 24)) \  # Días a horas
    --model_id "context_${seq_len}days"
done
```

**Análisis de saturación**:
- ¿En qué punto mejora < 2% con más contexto?
- ¿Fases ENSO requieren diferentes contextos?

### **Semana 3: Análisis y Escritura**

1. **Gráficos**: F1 vs Contexto por fase ENSO
2. **Conclusión**: "Contexto óptimo = 365-730 días"
3. **Contribución**: Reproducible pero menos novedosa

---

## 🎯 MI RECOMENDACIÓN: **OPCIÓN A** (Mejoras Arquitectónicas)

### Razones

1. ✅ **Mayor aporte al estado del arte**:
   - Mejoras en TimeAttention (ENSO-aware)
   - Máscara Kronecker adaptativa
   - Multi-scale temporal features

2. ✅ **F1=0.79 es mejorable**:
   - Solo 5 épocas → puede llegar a 0.82+ con más entrenamiento
   - Modificaciones arquitectónicas → 0.85-0.88

3. ✅ **Transfer Learning tiene potencial**:
   - Recall=0.89 demuestra que captura patrones
   - Precisión baja (0.71) es optimizable con class weights

4. ✅ **Contribución más sólida para tesis**:
   - "Optimized Timer-XL achieves F1=0.87" > "Optimal context is 730 days"

5. ✅ **Timeline realista**: 3-4 semanas vs 2-3 semanas (similar esfuerzo)

---

## 📋 Pasos Inmediatos (Esta Semana)

### ✅ Paso 1: Continuar Entrenamiento Transfer Learning (Hoy)

```bash
# Cargar checkpoint actual y entrenar 15 épocas más
python run.py \
  --task_name classification \
  --model timer_xl_classifier \
  --train_epochs 20 \
  --learning_rate 5e-6 \
  --adaptation \
  --pretrain_model_path checkpoints/classification_peru_rainfall_timerxl_.../checkpoint.pth
```

**Expectativa**: F1 → 0.81-0.82 (2-3 horas)

### ✅ Paso 2: Implementar Class Weights (Mañana)

Modificar `models/timer_xl_classifier.py`:

```python
# En __init__
self.class_weights = torch.tensor([1.5, 1.0]).to(device)  # Penalizar más No Rain
self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
```

Entrenar de nuevo y medir impacto:
- Falsos positivos: 49% → ~35%
- Precision: 0.71 → ~0.78
- F1: 0.82 → ~0.83

### ✅ Paso 3: Validar Mejora (2 días)

Comparar:
- Transfer Learning 5 épocas (F1=0.79)
- Transfer Learning 20 épocas (F1=~0.82)
- Transfer Learning 20 épocas + class weights (F1=~0.83)

Si F1 > 0.83: ✅ **Confirmar OPCIÓN A y proceder con mejoras arquitectónicas**

---

## 📊 Métricas de Éxito para Tesis

### Mínimo Aceptable
- F1-Score > 0.82 (mejora de 3% sobre baseline 0.79)
- Ablation studies mostrando contribución de cada mejora
- Validación en 3 fases ENSO

### Objetivo Ideal
- F1-Score > 0.85 (mejora de 6% sobre baseline)
- Precision > 0.82 (reducción de falsos positivos)
- Recall > 0.85 (mantener detección alta)
- Consistencia en fases ENSO (|ΔF1| < 0.10)

### Excelente (Publicable)
- F1-Score > 0.87 (mejora de 8%)
- Precision > 0.85 y Recall > 0.89
- Superioridad demostrada vs Small Model en todas las métricas
- Código reproducible + ablation studies completos

---

## 📝 Estructura de Tesis Propuesta

### Capítulo 1: Introducción
- Motivación: ENSO y predicción de lluvias en Perú
- Problema: Modelos actuales tienen F1 < 0.80
- **Contribución**: "Optimización de Timer-XL con mejoras arquitectónicas para F1 > 0.85"

### Capítulo 2: Marco Teórico
- 2.1 Fenómeno ENSO
- 2.2 Arquitectura Transformer
- 2.3 Timer-XL: TimeAttention y máscara Kronecker
- 2.4 Transfer Learning en series temporales

### Capítulo 3: Metodología
- 3.1 Datos ERA5 (2020-2024)
- 3.2 Baseline: Transfer Learning (F1=0.79)
- 3.3 Mejoras propuestas:
  - ENSO-aware TimeAttention
  - Máscara Kronecker adaptativa
  - Multi-scale temporal features
  - Optimización de hiperparámetros
- 3.4 Ablation studies

### Capítulo 4: Resultados
- 4.1 Comparación Baseline vs Mejorado
- 4.2 Ablation studies (contribución individual)
- 4.3 Validación por fase ENSO
- 4.4 Análisis de casos (aciertos/errores)

### Capítulo 5: Discusión
- Por qué ENSO-aware attention mejora F1
- Trade-off Precision-Recall
- Limitaciones y trabajo futuro

### Capítulo 6: Conclusiones
- Timer-XL optimizado logra F1=0.87 (+8% vs baseline)
- ENSO-aware attention es clave para eventos extremos
- Pipeline reproducible para predicción climática

---

## 🚀 Próximos Pasos (Esta Semana)

1. **Hoy (Sábado)**:
   - [x] Analizar resultados Transfer Learning (COMPLETADO)
   - [ ] Continuar entrenamiento 15 épocas más
   - [ ] Monitorear convergencia

2. **Mañana (Domingo)**:
   - [ ] Implementar class weights
   - [ ] Entrenar con class weights
   - [ ] Comparar F1: baseline (0.79) vs nuevo (~0.83)

3. **Lunes-Martes**:
   - [ ] Si F1 > 0.83: Implementar ENSO-aware TimeAttention
   - [ ] Si F1 < 0.82: Revisar learning rate / epochs

4. **Miércoles-Jueves**:
   - [ ] Implementar máscara Kronecker adaptativa
   - [ ] Entrenar y validar

5. **Viernes**:
   - [ ] Revisar progreso con asesor
   - [ ] Decidir si continuar con multi-scale o consolidar resultados

---

## 📧 Preguntas para Asesor (Próxima Reunión)

1. **Alcance de tesis**: ¿Aceptable enfoque de mejoras arquitectónicas vs análisis de contexto?

2. **Métricas objetivo**: ¿F1 > 0.85 es suficiente para tesis o necesito > 0.87?

3. **Validación ENSO**: ¿Temporal split (2020-2022 train, 2023-2024 test) o stratified?

4. **Contribución novedosa**: ¿ENSO-aware TimeAttention es suficiente aporte o necesito más?

5. **Timeline**: ¿3-4 semanas es realista o debo simplificar experimentos?

---

**Última Actualización**: 2025-01-11  
**Estado**: Plan definido, esperando inicio de Opción A  
**Meta**: F1-Score > 0.85 con mejoras arquitectónicas
