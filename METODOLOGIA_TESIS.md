# 📘 Metodología de Tesis: Predicción de Lluvias con Timer-XL y Análisis ENSO

**Autor**: Christian  
**Fecha**: Octubre 2025  
**Objetivo**: Evaluar Timer-XL para predicción de lluvias en Perú considerando fases ENSO y variabilidad regional

---

## 🎯 Pregunta de Investigación Central

**"¿Cómo varía el desempeño de Timer-XL en la predicción de lluvias en la costa peruana según las fases ENSO (El Niño, La Niña, Neutral) y la ubicación geográfica (Norte, Centro, Sur)?"**

### Sub-preguntas:
1. ¿Timer-XL mantiene consistencia (F1 > 0.75) en todas las fases ENSO?
2. ¿El rendimiento es mejor en regiones con mayor influencia ENSO (Costa Norte)?
3. ¿Qué configuración de Focal Loss optimiza el balance entre clases minoritarias?

---

## 📊 Metodología de 3 Fases

### **FASE 1: Baseline con Focal Loss Optimizado**

#### Objetivo
Establecer un modelo baseline con **F1 > 0.80** y **Recall No Rain > 60%**.

#### Configuración del Modelo
```python
model_config = {
    "arquitectura": "Timer-XL Small (5 layers, 640 dim)",
    "seq_len": 1440,  # 60 días de contexto
    "batch_size": 32,
    "learning_rate": 5e-5,
    "dropout": 0.25,  # Regularización fuerte
    "train_epochs": 20,
    "patience": 6
}

focal_loss_config = {
    "alpha": 0.70,  # Favorece clase "No Rain" (minoritaria)
    "gamma": 3.0,   # Enfoque fuerte en ejemplos difíciles
    "justificacion": """
        Ratio No Rain:Rain = 1:1.91
        Alpha = 0.70 compensa el desbalance
        Gamma = 3.0 penaliza fuertemente errores en minoritaria
    """
}
```

#### Métricas de Éxito
- ✅ **F1-Score > 0.80** (weighted average)
- ✅ **Recall No Rain > 60%** (clase minoritaria crítica)
- ✅ **Recall Rain ~80-85%** (balance con mayoría)
- ✅ **|Recall No Rain - Recall Rain| < 25%** (balance general)

#### Baseline de Comparación
- Regresión Logística con features de último mes
- Timer-XL debe superar por **> 15% en F1-Score**

---

### **FASE 2: Validación ENSO-aware**

#### Objetivo
Evaluar consistencia del modelo a través de fases ENSO.

#### Definición de Fases (basado en ONI Index)
```python
ENSO_PHASES = {
    "El_Niño": [
        ("2015-01-01", "2016-06-30"),  # El Niño fuerte 2015-16
        ("2018-10-01", "2019-06-30"),  # El Niño débil 2018-19
        ("2023-06-01", "2024-05-31"),  # El Niño 2023-24
    ],
    "La_Niña": [
        ("2020-08-01", "2021-05-31"),
        ("2021-09-01", "2022-03-31"),
        ("2022-09-01", "2023-02-28"),
    ],
    "Neutral": [
        ("2017-01-01", "2017-12-31"),
        ("2019-07-01", "2020-07-31"),
        ("2024-06-01", "2024-12-31"),
    ]
}
```

#### Hipótesis a Validar
1. **H1 (Consistencia)**: F1-Score > 0.75 en TODAS las fases ENSO
2. **H2 (Diferencia El Niño vs La Niña)**: |F1_ElNiño - F1_LaNiña| < 0.15
3. **H3 (Mejor rendimiento en extremos)**: F1_ElNiño ≥ F1_Neutral AND F1_LaNiña ≥ F1_Neutral

#### Métricas por Fase
- F1-Score (weighted, No Rain, Rain)
- Precision y Recall por clase
- AUC-ROC
- Confusion Matrix

#### Visualizaciones
- Gráfico de barras comparativo de F1-Score por fase
- Confusion matrices lado a lado
- Curvas ROC por fase

---

### **FASE 3: Análisis Regional**

#### Objetivo
Verificar si el modelo captura el gradiente de influencia ENSO (Norte → Sur).

#### Definición de Regiones
```python
REGIONS = {
    "Costa_Norte": (-8.0, -4.0),    # Piura, Tumbes, Lambayeque
    "Costa_Centro": (-14.0, -8.0),  # Lima, Callao, Ica
    "Costa_Sur": (-18.0, -14.0),    # Arequipa, Moquegua, Tacna
}
```

#### Hipótesis a Validar
1. **H4 (Gradiente ENSO)**: F1_Norte > F1_Centro > F1_Sur
   - Justificación: Costa Norte tiene mayor influencia ENSO (mayor señal → más predecible)
   
2. **H5 (Prevalencia de lluvia)**: Rain_prevalence_Norte > Rain_prevalence_Sur
   - El Niño trae más lluvias al norte

#### Métricas Regionales
- F1-Score por región
- Rain prevalence (proporción de timesteps con lluvia)
- Correlation entre "influencia ENSO regional" y "F1-Score"

#### Visualizaciones
- Mapa de Perú con F1-Score por región (heatmap)
- Gráfico de barras: F1 por región
- Scatter plot: Rain prevalence vs F1-Score

---

## 🔬 Comparación con Papers de ENSO

### **Paper 1: ENSO-Former** (Predicción ENSO con Transformers)

#### Ideas Aplicables a tu Tesis
1. **Atención espacial dual**:
   - ENSO-Former usa atención local + global para capturar patrones multi-escala
   - **Aplicación**: Puedes añadir capas de atención espacial en Timer-XL para diferenciar regiones

2. **Preentrenamiento + Transfer Learning**:
   - ENSO-Former preentrenó con CMIP6 (simulaciones) y fine-tuned con reanálisis
   - **Tu approach**: Ya usaste checkpoint pretrained → similar a transfer learning

3. **Validación por fases**:
   - ENSO-Former evalúa en eventos específicos de El Niño
   - **Tu contribución**: Extiendes a 3 fases (El Niño, La Niña, Neutral) + análisis regional

#### Diferencias Clave
| Aspecto | ENSO-Former | Tu Tesis |
|---------|-------------|----------|
| **Target** | Índice Niño 3.4 (temperatura oceánica) | Lluvia en tierra (Perú) |
| **Horizonte** | 20 meses adelante | 24 horas adelante |
| **Variables** | SST, vientos oceánicos | Variables atmosféricas (ERA5) |
| **Validación** | Solo eventos El Niño | Todas las fases ENSO + regiones |

---

### **Paper 2: Predictabilidad Estacional del Monzón**

#### Ideas Aplicables
1. **Separación de componentes predecibles**:
   - Usan descomposición de covarianza para separar señal predecible vs ruido
   - **Aplicación potencial**: Añadir análisis EOF para identificar modos predecibles en tus datos

2. **Predictores regionales independientes**:
   - Identificaron 9 predictores (SST Atlántico, Índico, Ártico, tendencia)
   - **Tu mejora**: Añadir features ENSO (ONI, SOI, SST Niño 3.4) como inputs explícitos

3. **Validación cruzada temporal**:
   - Dejan fuera períodos de 3 años para evitar data leakage
   - **Tu approach**: Ya usas split temporal (train/val/test por años)

#### Diferencias Clave
| Aspecto | Season-Predictable | Tu Tesis |
|---------|-------------------|----------|
| **Región** | Asia Oriental (monzón) | Costa de Perú |
| **Horizonte** | Estacional (1-2 meses) | Corto plazo (24h) |
| **Predictores** | 9 índices climáticos | Variables ERA5 locales |
| **Enfoque** | Regresión lineal (PCR) | Deep Learning (Timer-XL) |

---

## 📈 Contribuciones Originales de tu Tesis

### **1. Primera Evaluación ENSO-aware de Timer-XL**
- No existe literatura previa de Timer-XL con validación por fases ENSO
- Demuestras si Timer-XL es "ENSO-aware" implícitamente

### **2. Análisis Regional Detallado**
- Validación del gradiente ENSO (Norte → Sur)
- Conexión entre influencia ENSO regional y predictibilidad

### **3. Optimización de Focal Loss para Desbalance Climático**
- Determinas configuración óptima (alpha=0.70, gamma=3.0)
- Aplicable a otros problemas de predicción de eventos extremos

### **4. Benchmark para Predicción de Lluvias en Perú**
- Primer estudio con 11 años de datos ERA5 (2014-2024)
- Estableces métricas baseline para futuros trabajos

---

## 🎯 Métricas de Éxito de la Tesis

### **Meta Mínima** (para aprobar)
- ✅ F1-Score > 0.75 (general)
- ✅ F1-Score > 0.70 en TODAS las fases ENSO
- ✅ Recall No Rain > 50%

### **Meta Objetivo** (para excelencia)
- ✅ F1-Score > 0.80 (general)
- ✅ F1-Score > 0.75 en TODAS las fases ENSO
- ✅ Recall No Rain > 60%
- ✅ Gradiente ENSO cumplido (F1_Norte > F1_Centro > F1_Sur)
- ✅ Consistencia ENSO: |F1_ElNiño - F1_LaNiña| < 0.15

### **Meta Aspiracional** (para publicación)
- ✅ F1-Score > 0.85 (general)
- ✅ Implementación de arquitectura ENSO-Former style (atención espacial)
- ✅ Incorporación de features ENSO explícitos (ONI, SOI)
- ✅ Comparación con modelos SOTA (LSTM-ENSO, ConvLSTM, etc.)

---

## 🛠️ Herramientas y Scripts Desarrollados

### **1. `validate_enso_phases.py`**
- Separa test set por fases ENSO
- Calcula métricas (F1, Precision, Recall, AUC-ROC) por fase
- Genera visualizaciones comparativas
- Verifica hipótesis de consistencia

### **2. `validate_regional.py`**
- Asigna región geográfica (Norte/Centro/Sur) por latitud
- Calcula métricas regionales
- Valida gradiente ENSO
- Genera mapas de rendimiento

### **3. Notebook Colab**
- Celda FASE 1: Entrenamiento con Focal Loss optimizado
- Celda FASE 2: (placeholder para integrar `validate_enso_phases.py`)
- Celda FASE 3: (placeholder para integrar `validate_regional.py`)

---

## 📝 Estructura de Resultados para Reporte

### **Capítulo 4: Resultados**

#### **4.1 Baseline con Focal Loss**
- Tabla 1: Métricas generales (Accuracy, F1, Precision, Recall)
- Figura 1: Confusion Matrix
- Figura 2: Curva de entrenamiento (Loss vs Epochs)
- Figura 3: ROC Curve

#### **4.2 Validación ENSO-aware**
- Tabla 2: Métricas por fase ENSO
- Figura 4: F1-Score comparativo por fase (gráfico de barras)
- Figura 5: Confusion matrices por fase (3 paneles)
- Tabla 3: Análisis de consistencia (F1 range, diferencia El Niño vs La Niña)

#### **4.3 Análisis Regional**
- Tabla 4: Métricas por región
- Figura 6: F1-Score por región (gráfico de barras)
- Figura 7: Mapa de Perú con F1-Score por región (heatmap)
- Figura 8: Scatter plot: Rain prevalence vs F1-Score

#### **4.4 Comparación con Baseline**
- Tabla 5: Timer-XL vs Regresión Logística
- Discusión: ¿Timer-XL supera por > 15%?

---

## 🔄 Flujo de Trabajo Completo

### **Semana 1-2: FASE 1 (Baseline)**
1. Ejecutar celda FASE 1 en Colab
2. Entrenar con Focal Loss (alpha=0.70, gamma=3.0)
3. Validar F1 > 0.80, Recall No Rain > 60%
4. Si NO cumple: ajustar alpha a 0.75, gamma a 3.5
5. Guardar mejor checkpoint

### **Semana 3: FASE 2 (Validación ENSO)**
1. Ejecutar `validate_enso_phases.py` con mejor checkpoint
2. Generar reporte y visualizaciones
3. Verificar hipótesis H1, H2, H3
4. Si consistencia baja: considerar features ENSO explícitos

### **Semana 4: FASE 3 (Análisis Regional)**
1. Ejecutar `validate_regional.py`
2. Generar reporte y visualizaciones
3. Verificar hipótesis H4, H5
4. Analizar correlación influencia ENSO vs F1

### **Semana 5: Redacción y Conclusiones**
1. Consolidar resultados en documento de tesis
2. Escribir sección de Metodología (basada en este documento)
3. Escribir sección de Resultados (tablas y figuras)
4. Discusión: Comparación con ENSO-Former y Season-Predictable
5. Conclusiones: Contribuciones y trabajo futuro

---

## 🎓 Conclusiones Esperadas

### **Si TODAS las hipótesis se cumplen**:
> "Timer-XL demuestra ser un modelo robusto para predicción de lluvias en Perú, manteniendo consistencia (F1 > 0.75) a través de todas las fases ENSO y capturando el gradiente de influencia ENSO esperado (mayor rendimiento en Costa Norte). La optimización de Focal Loss (alpha=0.70, gamma=3.0) fue crítica para balancear clases desbalanceadas. Este trabajo establece el primer benchmark de Timer-XL para predicción climática con validación ENSO-aware."

### **Si alguna hipótesis NO se cumple**:
> "Timer-XL alcanza F1 > 0.80 en condiciones generales, pero presenta limitaciones en [fase ENSO específica / región específica]. Esto sugiere que el modelo requiere [features ENSO explícitos / atención espacial / datos regionales adicionales] para capturar completamente la variabilidad ENSO-regional. Trabajos futuros deberían explorar [arquitecturas tipo ENSO-Former / predictores regionales del paper Season-Predictable / transfer learning desde modelos preentrenados en SST]."

---

## 📚 Referencias Clave para Discusión

1. **ENSO-Former**: SHA X., et al. (2024). "Multi-Head Spatiotemporal Convolutional Attention Block for ENSO Prediction." *Climate Modeling Journal*.

2. **Season-Predictable**: ZHENG F., et al. (2023). "Predictable Sources of East Asian Summer Monsoon Rainfall Beyond ENSO." *Atmospheric Science Letters*.

3. **Timer-XL**: Original paper (agregar referencia completa)

4. **Focal Loss**: LIN T., et al. (2017). "Focal Loss for Dense Object Detection." *ICCV*.

5. **ERA5**: HERSBACH H., et al. (2020). "The ERA5 global reanalysis." *Quarterly Journal of the Royal Meteorological Society*.

---

## 🚀 Trabajo Futuro (si quieres publicar)

### **Mejora 1: ENSO-Former Style Attention**
- Implementar atención espacial dual (local + global)
- Dividir Perú en grids de 2°x2° con atención independiente
- Comparar F1 con arquitectura base

### **Mejora 2: Features ENSO Explícitos**
- Añadir ONI index, SOI index, SST Niño 3.4 como inputs
- Entrenar modelo "ENSO-aware" vs baseline
- Validar si mejora consistencia entre fases

### **Mejora 3: Ensemble de Modelos Regionales**
- Entrenar 3 modelos especializados (Norte/Centro/Sur)
- Ensemble ponderado por confianza regional
- Comparar con modelo único global

### **Mejora 4: Predicción Multi-horizonte**
- Extender a 48h, 72h, 7 días
- Analizar degradación de F1 vs horizonte
- Comparar con papers de predicción estacional

---

**Este documento debe servir como tu guía metodológica completa. Actualízalo conforme avances en cada fase. ¡Éxito con tu tesis! 🎓**
