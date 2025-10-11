# 🚀 COMANDO DE ENTRENAMIENTO - GOOGLE COLAB

## ✅ Comando Completo (Copiar y Pegar)

```python
# Entrenar Timer-XL con Transfer Learning
!python run.py \
  --task_name classification \
  --is_training 1 \
  --model_id peru_rainfall_timerxl \
  --model timer_xl_classifier \
  --data PeruRainfall \
  --root_path datasets/processed/ \
  --data_path peru_rainfall.csv \
  --checkpoints checkpoints/ \
  --seq_len 1440 \
  --input_token_len 96 \
  --output_token_len 96 \
  --test_seq_len 1440 \
  --test_pred_len 2 \
  --e_layers 8 \
  --d_model 1024 \
  --d_ff 2048 \
  --n_heads 8 \
  --dropout 0.1 \
  --activation relu \
  --batch_size 128 \
  --learning_rate 1e-5 \
  --train_epochs 50 \
  --patience 10 \
  --n_classes 2 \
  --gpu 0 \
  --cosine \
  --tmax 50 \
  --use_norm \
  --adaptation \
  --pretrain_model_path checkpoints/timer_xl/checkpoint.pth \
  --use_focal_loss \
  --loss CE \
  --itr 1 \
  --des 'Peru_Rainfall_Transfer_Learning'
```

---

## 📋 Explicación de Argumentos Clave

### **Argumentos OBLIGATORIOS** (causaban el error)
```python
--task_name classification      # Tipo de tarea: clasificación (no forecast)
--is_training 1                 # Modo entrenamiento (1=train, 0=test)
--model_id peru_rainfall_timerxl # ID único del experimento
```

### **Argumentos del Modelo**
```python
--model timer_xl_classifier     # Modelo: Timer-XL adaptado a clasificación
--data PeruRainfall            # Dataset personalizado
--n_classes 2                  # Clasificación binaria (Rain/No Rain)
```

### **Rutas de Datos**
```python
--root_path datasets/processed/           # Carpeta con datos procesados
--data_path peru_rainfall.csv            # Archivo CSV generado por preprocesamiento
--checkpoints checkpoints/               # Carpeta para guardar checkpoints
```

### **Transfer Learning**
```python
--adaptation                                        # Activa transfer learning
--pretrain_model_path checkpoints/timer_xl/checkpoint.pth  # Checkpoint pre-entrenado
```

### **Arquitectura**
```python
--seq_len 1440              # Longitud de secuencia (60 días × 24h)
--input_token_len 96        # Tokens de entrada por ventana
--output_token_len 96       # Tokens de salida
--e_layers 8                # Capas del encoder
--d_model 1024              # Dimensión del modelo
--d_ff 2048                 # Dimensión feedforward
--n_heads 8                 # Attention heads
--dropout 0.1               # Dropout rate
```

### **Entrenamiento**
```python
--batch_size 128            # Tamaño del batch (ajustar si OOM)
--learning_rate 1e-5        # Learning rate (bajo para transfer learning)
--train_epochs 50           # Máximo de épocas
--patience 10               # Early stopping patience
--cosine                    # Cosine annealing scheduler
--tmax 50                   # T_max para cosine scheduler
```

### **Loss y Métricas**
```python
--loss CE                   # Cross-Entropy Loss
--use_focal_loss            # Focal Loss para desbalance de clases
```

### **Otros**
```python
--gpu 0                     # ID de GPU (0 para T4 en Colab)
--use_norm                  # Normalización de datos
--itr 1                     # Número de iteraciones del experimento
--des 'Peru_Rainfall_Transfer_Learning'  # Descripción del experimento
```

---

## 🔧 Ajustes para Optimizar

### **Si obtienes OOM (Out of Memory):**
```python
--batch_size 64    # Reducir a 64 o 32
```

### **Para entrenamiento más rápido (menos preciso):**
```python
--train_epochs 20   # Reducir épocas
--patience 5        # Reducir patience
```

### **Para mayor precisión (más lento):**
```python
--train_epochs 100  # Aumentar épocas
--learning_rate 5e-6  # Learning rate aún más bajo
```

---

## ✅ Verificación Pre-Entrenamiento

**Antes de ejecutar el comando, verifica:**

```python
# 1. GPU disponible
!nvidia-smi

# 2. Checkpoint pre-entrenado existe
!ls -lh checkpoints/timer_xl/checkpoint.pth

# 3. Datos procesados existen
!ls -lh datasets/processed/peru_rainfall.csv

# 4. Estructura de carpetas
!mkdir -p checkpoints/
```

---

## 📊 Salida Esperada Durante Entrenamiento

```
Loading pretrained model from checkpoints/timer_xl/checkpoint.pth
✅ Pretrained weights loaded successfully

Epoch 1/50:
  Train Loss: 0.6421
  Valid Loss: 0.5892
  Valid F1: 0.6234
  ⏱️  Time: 15m 30s

Epoch 2/50:
  Train Loss: 0.5234
  Valid Loss: 0.5123
  Valid F1: 0.6890
  ⏱️  Time: 15m 25s

...

Early Stopping at epoch 35
✅ Best model saved to: checkpoints/peru_rainfall_timerxl/checkpoint.pth
📊 Best F1-Score: 0.7245
```

---

## 🎯 Después del Entrenamiento

```python
# Verificar resultados
!ls -lh checkpoints/*/peru_rainfall_timerxl*/

# Copiar a Google Drive
!cp -r checkpoints/*/peru_rainfall_timerxl* /content/drive/MyDrive/timer_xl_peru/results/
```

---

## ❓ Solución de Problemas

### **Error: "the following arguments are required: --task_name, --is_training, --model_id"**
✅ **Solución:** Usar el comando completo de arriba (no omitir argumentos)

### **Error: "FileNotFoundError: checkpoint.pth"**
✅ **Solución:** Asegúrate de que el checkpoint esté en Drive:
```python
!ls -lh /content/drive/MyDrive/timer_xl_peru/checkpoints/checkpoint.pth
```

### **Error: "CUDA out of memory"**
✅ **Solución:** Reducir batch_size:
```python
--batch_size 64  # o 32
```

### **Error: "FileNotFoundError: peru_rainfall.csv"**
✅ **Solución:** Ejecutar preprocesamiento primero:
```python
!python preprocessing/preprocess_era5_peru.py --years 2022,2023,2024
```

---

## 🚀 Comando TODO-EN-UNO (para Copy-Paste directo)

```bash
# Verificar requisitos
nvidia-smi && \
ls -lh checkpoints/timer_xl/checkpoint.pth && \
ls -lh datasets/processed/peru_rainfall.csv && \

# Entrenar
python run.py \
  --task_name classification \
  --is_training 1 \
  --model_id peru_rainfall_timerxl \
  --model timer_xl_classifier \
  --data PeruRainfall \
  --root_path datasets/processed/ \
  --data_path peru_rainfall.csv \
  --checkpoints checkpoints/ \
  --seq_len 1440 \
  --input_token_len 96 \
  --output_token_len 96 \
  --test_seq_len 1440 \
  --test_pred_len 2 \
  --e_layers 8 \
  --d_model 1024 \
  --d_ff 2048 \
  --n_heads 8 \
  --dropout 0.1 \
  --activation relu \
  --batch_size 128 \
  --learning_rate 1e-5 \
  --train_epochs 50 \
  --patience 10 \
  --n_classes 2 \
  --gpu 0 \
  --cosine \
  --tmax 50 \
  --use_norm \
  --adaptation \
  --pretrain_model_path checkpoints/timer_xl/checkpoint.pth \
  --use_focal_loss \
  --loss CE \
  --itr 1 \
  --des 'Peru_Rainfall_Transfer_Learning'

echo "✅ Entrenamiento completado!"
```

---

## 📞 Siguiente Paso

**AHORA EN COLAB:**
1. Abre el notebook actualizado
2. Ejecuta todas las celdas hasta "Train Timer-XL"
3. Copia y pega el comando completo de arriba
4. Espera ~4-6 horas

**¡Listo!** 🎉
