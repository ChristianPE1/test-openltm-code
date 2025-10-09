# 🔧 Troubleshooting: NaN Loss en Timer-XL Classification

## 🔴 Problema

**Síntoma**: Training produce NaN loss en casi todos los batches, incluso con datos balanceados correctamente.

```
✅ Class distribution: No Rain=2934 (38%), Rain=4730 (62%)
⚠️ Skipping batch 50 due to NaN/Inf loss
⚠️ Skipping batch 100 due to NaN/Inf loss
Epoch: 1, Steps: 389 | Vali Loss: nan Test Loss: nan
```

## 🔍 Causas Probables

### 1. **Gradientes Explosivos en Transfer Learning**

El modelo Timer-XL pre-entrenado tiene pesos optimizados para regresión (forecasting). Al adaptar para clasificación:
- Las escalas de activación son diferentes
- El clasificador nuevo puede generar gradientes muy grandes
- Los gradientes se propagan hacia el encoder pre-entrenado
- **Resultado**: Overflow numérico → NaN

### 2. **Inicialización Inadecuada del Classifier**

Xavier initialization puede generar pesos demasiado grandes para clasificación binaria:

```python
# ❌ Problema:
nn.init.xavier_uniform_(module.weight)  # Pesos en rango [-sqrt(6/n), sqrt(6/n)]

# Con output_token_len=96, n_in=96, n_out=256:
# Rango: [-0.25, 0.25] → Puede causar overflow en forward pass
```

### 3. **Learning Rate Demasiado Alto**

```python
# Para transfer learning desde Timer-XL:
learning_rate = 1e-5  # ❌ Demasiado alto
learning_rate = 1e-6  # ✅ Más estable
```

## ✅ Soluciones Aplicadas

### Fix 1: Inicialización Más Pequeña

```python
def _init_classifier_weights(self):
    for module in self.classifier.modules():
        if isinstance(module, nn.Linear):
            # Usar distribución normal con std muy pequeño
            nn.init.normal_(module.weight, mean=0.0, std=0.01)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
```

**Efecto**: Pesos iniciales en rango ~[-0.03, 0.03] → Menor probabilidad de overflow

### Fix 2: Classifier Más Simple con LayerNorm

```python
# ❌ ANTES (inestable):
self.classifier = nn.Sequential(
    nn.Linear(96, 256),  # Gran expansión
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(256, 64),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(64, 2)
)

# ✅ AHORA (estable):
self.classifier = nn.Sequential(
    nn.LayerNorm(96),        # ← Normalización para estabilidad
    nn.Linear(96, 128),      # Menor expansión
    nn.GELU(),               # ← Más suave que ReLU
    nn.Dropout(0.1),
    nn.Linear(128, 2)
)
```

**Ventajas**:
- LayerNorm previene activaciones explosivas
- GELU es más suave que ReLU (sin "muros")
- Menos capas = menos oportunidades para NaN

### Fix 3: Learning Rate Más Bajo

```bash
# Cambiar de:
--learning_rate 1e-5

# A:
--learning_rate 1e-6  # ← 10x más conservador
```

**Justificación**:
- Encoder pre-entrenado necesita ajustes muy finos
- Classifier nuevo no debe "tirar" del encoder demasiado rápido
- Convergencia más lenta pero más estable

### Fix 4: Gradient Clipping (Ya estaba, pero mejorado)

```python
# Ya existía:
torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

# Mejorado con debug info:
if torch.isnan(loss) or torch.isinf(loss):
    print(f"outputs: min={outputs.min()}, max={outputs.max()}")
    print(f"has_nan={torch.isnan(outputs).any()}")
    continue
```

## 🎯 Estrategia de Entrenamiento

### Fase 1: Congelar Encoder (Opcional)

Si el problema persiste, congelar temporalmente:

```python
# En el código de training:
model.freeze_encoder()  # Solo entrenar classifier

# Entrenar por 5-10 epochs

model.unfreeze_encoder()  # Luego ajustar todo
```

### Fase 2: Warm-up Gradual

```python
# Empezar con LR muy bajo, subir gradualmente:
for epoch in range(5):  # Warm-up
    lr = 1e-7 * (epoch + 1)
    
# Luego usar 1e-6 normal
```

## 📊 Métricas de Verificación

### Antes del Fix

```
Epoch: 1 | Vali Loss: nan | Test Loss: nan
Accuracy: 38.19% (casi aleatorio)
⚠️ 389/389 batches skipped
```

### Después del Fix (Esperado)

```
Epoch: 1 | Vali Loss: 0.6234 | Test Loss: 0.6511
Accuracy: 58.3%
✅ 0 batches skipped
```

## 🔍 Debug Checklist

Si el problema persiste:

1. **Verificar outputs del modelo**:
```python
with torch.no_grad():
    outputs = model(batch_x, batch_x_mark, batch_y_mark)
    print(f"Min: {outputs.min()}, Max: {outputs.max()}")
    print(f"Has NaN: {torch.isnan(outputs).any()}")
```

2. **Verificar gradientes**:
```python
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_norm={param.grad.norm()}")
```

3. **Verificar datos**:
```python
print(f"batch_x: min={batch_x.min()}, max={batch_x.max()}")
print(f"Has NaN in data: {torch.isnan(batch_x).any()}")
```

4. **Probar sin pre-trained weights**:
```python
# Comentar temporalmente:
# --adaptation \
# --pretrain_model_path checkpoints/timer_xl/checkpoint.pth \

# Ver si entrena desde cero (sin transfer learning)
```

## 🚀 Próximos Pasos

1. **Pull cambios del repo**:
```bash
!git pull origin main
```

2. **Re-entrenar con fixes**:
```bash
!python run.py \
  --learning_rate 1e-6 \  # ← Cambio clave
  --batch_size 16 \
  ...
```

3. **Monitorear**:
- Loss debe ser numérico (0.5 - 0.8 inicial)
- Accuracy debe mejorar desde ~50% baseline
- No debería haber "skipping batch" messages

## 📝 Referencias

- [PyTorch Numerical Stability](https://pytorch.org/docs/stable/notes/numerical_accuracy.html)
- [Transfer Learning Best Practices](https://arxiv.org/abs/1411.1792)
- [Gradient Clipping Guide](https://neptune.ai/blog/understanding-gradient-clipping-and-how-it-can-fix-exploding-gradients-problem)

---

**Última actualización**: 2025-10-09  
**Estado**: Fixes aplicados, esperando validación en Colab
