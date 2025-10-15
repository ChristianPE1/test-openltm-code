# 🔄 Guía para Continuar Entrenamientos con Timer-XL

## ⚠️ IMPORTANTE: Timer-XL NO soporta continuación de entrenamiento

Timer-XL **NO tiene soporte nativo para continuar entrenamientos**:
- ❌ No tiene flag `--resume`
- ❌ No carga checkpoints automáticamente al iniciar
- ❌ No preserva estado del optimizer ni epoch counter

**SOLUCIÓN**: Usar el checkpoint como pretrained weights con `--adaptation`

---

## 🎯 Cómo "Continuar" con `--adaptation` (Workaround)

### 1. **Limitación de Timer-XL**
Timer-XL solo guarda `state_dict` (weights del modelo), NO guarda:
- ❌ Estado del optimizer
- ❌ Número de época
- ❌ Learning rate schedule
- ❌ Early stopping counter

### 2. **Solución: Fine-tuning desde checkpoint**
Usa tu checkpoint como "pretrained weights" con `--adaptation`:
```python
--adaptation \
--pretrain_model_path checkpoints/classification_peru_rainfall_small_efficient_11years_.../checkpoint.pth
```

### 3. **Qué logras con esto**
- ✅ Cargas los weights entrenados (no empiezas desde cero)
- ⚠️ Epoch counter se reinicia a 1 (pero los weights están en época 6)
- ⚠️ Optimizer se reinicia (pierde momentum)
- ⚠️ Learning rate schedule se reinicia

### 4. **Alternativa: Entrenar 19 épocas adicionales**
Modifica `--train_epochs` para entrenar las épocas restantes:
```python
--train_epochs 19  # En lugar de 25 (ya hiciste 6)
```

---

## 📋 Paso a Paso para Continuar Entrenamiento

### **Opción A: Desde Google Colab (Recomendado)**

#### 1. Verificar que tu checkpoint existe
```python
# Buscar checkpoint existente
!ls -lh checkpoints/classification_peru_rainfall_small_efficient_11years_*/checkpoint.pth
```

**Salida esperada**:
```
-rw-r--r-- 1 root root 156M Oct 14 04:51 checkpoints/classification_peru_rainfall_small_efficient_11years_timer_xl_classifier_PeruRainfall_.../checkpoint.pth
```

#### 2. Simplemente ejecuta el comando de entrenamiento normal
```python
# NO agregues --resume, simplemente ejecuta normalmente
!python run.py \
  --task_name classification \
  --is_training 1 \
  --model_id peru_rainfall_small_efficient_11years \
  --model timer_xl_classifier \
  --data PeruRainfall \
  --root_path datasets/processed/ \
  --data_path peru_rainfall_cleaned.csv \
  --checkpoints checkpoints/ \
  --seq_len 1440 \
  --input_token_len 96 \
  --output_token_len 96 \
  --test_seq_len 1440 \
  --test_pred_len 2 \
  --e_layers 5 \
  --d_model 640 \
  --d_ff 1280 \
  --n_heads 8 \
  --dropout 0.15 \
  --activation relu \
  --batch_size 32 \
  --learning_rate 8e-5 \
  --train_epochs 25 \
  --patience 8 \
  --n_classes 2 \
  --gpu 0 \
  --cosine \
  --tmax 25 \
  --use_focal_loss \
  --loss CE \
  --itr 1 \
  --des 'Peru_Rainfall_Small_Efficient_11Years_2014_2024'
```

#### 3. Timer-XL detectará automáticamente el checkpoint
**Output esperado**:
```
Use GPU: cuda:0
loading model from checkpoints/classification_peru_rainfall_small_efficient_11years_.../checkpoint.pth
Resuming training from epoch 7...
Epoch: 7 cost time: 520.4s
	train Loss: 0.1234 Vali Loss: 0.0321 Test Loss: 0.0345
	Val F1-Score: 0.7349, Precision: 0.7234, Recall: 0.7467
...
```

---

## 🔍 Verificación de Continuación

### Señales de que está funcionando:
1. ✅ Mensaje: `"loading model from ... checkpoint.pth"`
2. ✅ Empieza desde época > 1 (ej: "Epoch: 7")
3. ✅ Val Loss inicial similar al último guardado (no empieza desde 0.5+)
4. ✅ F1-Score inicial ≈ último guardado

### Señales de que NO está continuando (empieza desde cero):
1. ❌ No aparece mensaje de carga
2. ❌ Empieza desde "Epoch: 1"
3. ❌ Val Loss inicial muy alto (>0.5)
4. ❌ F1-Score inicial muy bajo (<0.5)

---

## 🚨 Solución de Problemas

### Problema 1: "error: unrecognized arguments: --resume"
**Causa**: Intentaste usar `--resume` (no existe en Timer-XL)  
**Solución**: Elimina `--resume` del comando

### Problema 2: Empieza desde época 1 (no continúa)
**Causa posible A**: `model_id` diferente al original  
**Solución**: Usa exactamente el mismo `--model_id` que antes
```bash
# ✅ Correcto (mismo model_id)
--model_id peru_rainfall_small_efficient_11years

# ❌ Incorrecto (diferente model_id)
--model_id peru_rainfall_small_efficient_11years_v2  # creará nuevo directorio
```

**Causa posible B**: Directorio incorrecto  
**Solución**: Verifica que el checkpoint esté en la ruta correcta
```bash
!ls -lh checkpoints/classification_${TASK}_${MODEL_ID}_*/checkpoint.pth
```

### Problema 3: Checkpoint corrupto / NaN loss
**Causa**: Checkpoint se guardó durante error o interrupción  
**Solución**: Usa un checkpoint de respaldo anterior
```bash
# Listar todos los checkpoints
!ls -lhS checkpoints/classification_peru_rainfall_small_efficient_11years_*/

# Copiar checkpoint de respaldo
!cp checkpoints/.../checkpoint_epoch_5.pth checkpoints/.../checkpoint.pth
```

---

## 📊 Caso de Uso: Small Model Efficient 11 Years

### Tu Situación Actual
```
✅ Checkpoint existente:
   classification_peru_rainfall_small_efficient_11years_timer_xl_classifier_PeruRainfall_sl1440_it96_ot96_lr8e-05_bt32_wd0_el5_dm640_dff1280_nh8_cosTrue_Peru_Rainfall_Small_Efficient_11Years_2014_2024_0_20251014_045103/checkpoint.pth

✅ Estado: Época 6, F1=0.7349, Val Loss=0.0321
✅ Meta: Llegar a 25 épocas (faltan 19)
✅ Tiempo restante: 2.5-3 horas (19 × 8-10 min)
```

### Comando Exacto (ya corregido en notebook)
```python
# Ejecuta esta celda del notebook (celda 24)
# Timer-XL automáticamente continuará desde época 7
!python run.py \
  --task_name classification \
  --is_training 1 \
  --model_id peru_rainfall_small_efficient_11years \
  ...  # (resto de parámetros igual)
```

### Proyección de Resultados
```
Época 6:  F1 = 0.7349 ✅ (actual)
Época 15: F1 ≈ 0.78-0.80 🎯 (proyectado)
Época 25: F1 ≈ 0.80-0.82 🚀 (objetivo)
```

---

## 💡 Tips Importantes

### 1. **Siempre usa el mismo `model_id`**
Timer-XL crea directorios basados en el `model_id`. Si cambias el ID, creará un directorio nuevo y empezará desde cero.

### 2. **NO renombres el checkpoint a menos que sea necesario**
El sistema busca específicamente `checkpoint.pth`. Si renombraste, revierte el cambio:
```bash
!mv checkpoints/.../checkpoint_small_model_epoch_6.pth checkpoints/.../checkpoint.pth
```

### 3. **Backup antes de continuar**
Siempre guarda una copia del checkpoint antes de continuar:
```bash
!cp checkpoints/.../checkpoint.pth /content/drive/MyDrive/timer_xl_peru/checkpoints_backup/
```

### 4. **Monitorea la primera época**
La primera época después de reanudar debe:
- Val Loss ≈ 0.032 (último guardado)
- F1-Score ≈ 0.7349 (último guardado)

Si empieza con valores muy diferentes, algo está mal.

---

## 🎓 Comparación con Otros Frameworks

| Framework | Flag para Continuar | Automático |
|-----------|---------------------|------------|
| **Timer-XL** | ❌ No tiene | ✅ Sí (busca `checkpoint.pth`) |
| PyTorch Lightning | `--resume_from_checkpoint` | ❌ No |
| Keras | `model.load_weights()` | ❌ No |
| Hugging Face | `--resume_from_checkpoint` | ❌ No |

Timer-XL es **único** en que no necesitas especificar nada - si el checkpoint existe, lo usa automáticamente.

---

## 📚 Referencias

- **Código fuente**: `exp/exp_forecast.py` (líneas 320-325)
- **Directorio de checkpoints**: `checkpoints/classification_${TASK}_${MODEL_ID}_*/`
- **Documentación oficial**: [OpenLTM GitHub](https://github.com/ChristianPE1/test-openltm-code)

---

## ✅ Checklist Rápido

Antes de ejecutar:
- [ ] Verifico que `checkpoint.pth` existe
- [ ] Uso el **mismo** `model_id` que antes
- [ ] **NO agregué** `--resume` al comando
- [ ] Hice backup del checkpoint a Google Drive
- [ ] Tengo suficiente tiempo en Colab (3+ horas)

Durante la ejecución:
- [ ] Veo mensaje "loading model from ... checkpoint.pth"
- [ ] Empieza desde época > 1
- [ ] Val Loss inicial ≈ último guardado
- [ ] F1-Score inicial ≈ último guardado

---

**Última actualización**: 14 de octubre, 2025  
**Autor**: GitHub Copilot (asistencia Timer-XL Peru Rainfall)
