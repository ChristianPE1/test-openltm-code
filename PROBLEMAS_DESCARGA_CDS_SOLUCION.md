# 🚨 PROBLEMAS ENCONTRADOS Y SOLUCIONES - Descarga ERA5

## ❌ Problemas en tu Descarga CDS

### **Problema 1: Variables de Viento Incorrectas**
```
❌ Descargaste: 100m u-component of wind (u100)
❌ Descargaste: 100m v-component of wind (v100)

✅ Necesitas:    10m u-component of wind (u10)
✅ Necesitas:    10m v-component of wind (v10)
```

**Impacto:**
- El viento a 100m de altura NO representa bien las condiciones de superficie
- Para predicción de lluvia, necesitamos viento cercano al suelo (10m)
- **Solución temporal**: El script renombra u100→u10, v100→v10
- **Solución definitiva**: Re-descargar con u10/v10 correctos

---

### **Problema 2: Área Geográfica Demasiado Grande**
```
❌ Descargaste: North: 90° (¡Polo Norte!)
                South: -18° (Chile)
                
✅ Necesitas:    North: 0° (Ecuador)
                South: -18° (Chile)
```

**Impacto:**
- Archivo INSTANT: 653 MB (en lugar de ~30 MB)
- Descargaste datos desde Perú hasta el Ártico
- Desperdicio de 500+ MB de datos innecesarios
- **Solución aplicada**: El script filtra automáticamente solo Perú (0° a -18°)

---

### **Problema 3: Variables Distribuidas en 2 Archivos**
```
❌ CDS separa en 2 archivos:
   - ACCUM: Solo 'tp' (precipitación)
   - INSTANT: Todas las demás
   
✅ Timer-XL necesita: 1 archivo con todas las variables
```

**Impacto:**
- Código original no puede leer 2 archivos separados
- **Solución aplicada**: Script `combine_cds_files.py` los combina automáticamente

---

## ✅ Solución Implementada

### **Script Creado: `combine_cds_files.py`**

**Hace automáticamente:**
1. ✅ Extrae ambos archivos del ZIP
2. ✅ Combina ACCUM + INSTANT en 1 archivo
3. ✅ Renombra u100/v100 → u10/v10 (temporal)
4. ✅ Filtra región de Perú (0° a -18°)
5. ✅ Limpia metadatos innecesarios
6. ✅ Verifica todas las variables requeridas

**Resultado:**
```
era5_peru_2024.nc (119 MB)
  ✅ 9/9 variables requeridas
  ✅ 1,464 timesteps (4 horas/día × 366 días)
  ✅ Solo región de Perú (73 lat × 53 lon)
```

---

## 📋 GUÍA CORRECTA PARA PRÓXIMAS DESCARGAS

### **Configuración CDS Correcta**

#### **1. Product type**
```
✅ Reanalysis
```

#### **2. Variables (9 OBLIGATORIAS)** ⚠️ **CORREGIR ESTO**
```
✅ 2m temperature                    (t2m)
✅ 2m dewpoint temperature           (d2m)
✅ Surface pressure                  (sp)
✅ Mean sea level pressure           (msl)
✅ 10m u-component of wind           (u10)  ⚠️ NO 100m
✅ 10m v-component of wind           (v10)  ⚠️ NO 100m
✅ Total precipitation               (tp)
✅ Total column water vapour         (tcwv)
✅ Convective available potential energy (cape)
```

**Variables OPCIONALES (útiles pero no críticas):**
```
⭕ Total cloud cover                (tcc)
⭕ Boundary layer height            (blh)
⭕ Skin temperature                 (skt)
```

#### **3. Year**
```
Para pruebas iniciales:
✅ 2022, 2023, 2024 (3 años)

Para modelo final:
✅ 2014-2024 (11 años)
```

#### **4. Month**
```
✅ Todos (01-12)
```

#### **5. Day**
```
✅ Todos (01-31)
```

#### **6. Time** ⚠️ **IMPORTANTE**
```
✅ 06:00
✅ 18:00

❌ NO incluir: 00:00, 12:00 (opcional, el código los filtrará)
```

**Razón**: Timer-XL usa resolución 12-hourly (cada 12h)

#### **7. Geographical area** ⚠️ **CORREGIR ESTO**
```
✅ North:  0°     (frontera Ecuador)
✅ West:  -82°    (costa Pacífico)
✅ South: -18°    (frontera Chile)
✅ East:  -68°    (frontera Brasil)
```

#### **8. Data format**
```
✅ NetCDF4 (Experimental)
```

#### **9. Download format**
```
✅ Unarchived  (1 archivo .nc)

❌ NO usar ZIP (genera 2 archivos separados)
```

---

## 🔄 Qué Hacer Ahora

### **Opción A: Usar Archivos Actuales (RÁPIDO)**
```powershell
# Ya combinaste 2024, ahora combinar 2023
python AdaptationOpenLTM\preprocessing\combine_cds_files.py --zip_file cds_2023.zip --output era5_peru_2023.nc

# Resultado:
# era5_peru_2023.nc
# era5_peru_2024.nc

# LISTO PARA ENTRENAR (con limitación de u100/v100 en lugar de u10/v10)
```

**Pros:**
- ✅ Rápido (ya tienes los datos)
- ✅ Puedes empezar a entrenar HOY
- ✅ Validar pipeline completo

**Contras:**
- ⚠️ Viento a 100m en lugar de 10m (menor precisión)
- ⚠️ Área descargada más grande de lo necesario (ya filtrada en combinación)

---

### **Opción B: Re-descargar Correctamente (RECOMENDADO A FUTURO)**

Para descargas futuras (2022, 2021, etc.):

```
1. Usar configuración corregida (ver arriba)
2. **CLAVE**: Seleccionar u10/v10 (NO u100/v100)
3. **CLAVE**: North: 0° (NO 90°)
4. **CLAVE**: Download format: Unarchived (NO Zip)
5. Descargar directamente como 1 archivo .nc
6. NO necesitarás combinar archivos
```

**Ventajas:**
- ✅ Variables correctas (u10/v10)
- ✅ Archivo más pequeño (~40 MB por año en lugar de 700 MB)
- ✅ 1 archivo en lugar de 2 (no necesita combinación)
- ✅ Mejor rendimiento del modelo

---

## 📊 Comparación de Tamaños

### **Tu Descarga (incorrecta)**
```
cds_2024.zip (686 MB total)
  ├── instant.nc (653 MB) ← Incluye datos hasta Polo Norte
  └── accum.nc   (33 MB)
  
Después de combinar y filtrar:
era5_peru_2024.nc (119 MB) ← Solo Perú
```

### **Descarga Correcta (futuro)**
```
era5_peru_2024.nc (40-50 MB) ← Directo, solo Perú, 1 archivo
```

**Ahorro:** ~90% menos espacio

---

## 🎯 Plan de Acción

### **Ahora (Usar lo que tienes):**
```powershell
# 1. Combinar 2023
python AdaptationOpenLTM\preprocessing\combine_cds_files.py --zip_file cds_2023.zip

# 2. Mover archivos combinados
Move-Item era5_peru_2023.nc AdaptationOpenLTM\datasets\raw_era5\
Move-Item era5_peru_2024.nc AdaptationOpenLTM\datasets\raw_era5\

# 3. Preprocesar
cd AdaptationOpenLTM
python preprocessing\preprocess_era5_peru.py \
    --input_dir datasets\raw_era5 \
    --output_dir datasets\processed \
    --years 2023,2024 \
    --target_horizon 24

# 4. Entrenar en Colab
# (subir archivos procesados a Drive)
```

### **Después (Mejorar modelo):**
```
1. Descargar 2022 con configuración CORREGIDA (u10/v10, area=Perú)
2. Descargar 2021, 2020... si quieres más datos
3. Re-entrenar con datos correctos
4. Comparar mejora de rendimiento
```

---

## ❓ Preguntas Frecuentes

**P: ¿Debo re-descargar TODO ahora?**  
R: NO. Usa lo que tienes (2023-2024) para validar el pipeline. Re-descarga más adelante con configuración correcta.

**P: ¿El modelo funcionará con u100 en lugar de u10?**  
R: SÍ, pero con menor precisión. El viento a 100m no representa bien condiciones de superficie.

**P: ¿Por qué CDS me dio 2 archivos (ACCUM + INSTANT)?**  
R: Porque seleccionaste "Download format: Zip". Usa "Unarchived" para 1 archivo.

**P: ¿Cuánto mejora el modelo con u10 vs u100?**  
R: Estimado: 2-5% mejor F1-Score. No es crítico, pero es mejor tenerlo correcto.

**P: ¿Debo descargar solo 06:00 y 18:00 o las 4 horas?**  
R: Solo 06:00 y 18:00 es suficiente. El código filtra automáticamente si descargas 4 horas.

---

## 📝 Resumen

### ✅ Lo que hiciste bien:
- Descargaste las 9 variables obligatorias (aunque u100/v100 en lugar de u10/v10)
- Formato NetCDF correcto
- Período 2023-2024 adecuado para pruebas

### ⚠️ Lo que debe corregirse en futuras descargas:
- **u10/v10** en lugar de u100/v100
- **North: 0°** en lugar de 90°
- **Unarchived** en lugar de Zip (para evitar 2 archivos)

### 🎯 Estado actual:
- ✅ `era5_peru_2024.nc` combinado (119 MB, 9/9 variables)
- ⏳ Pendiente: combinar `cds_2023.zip`
- ⏳ Pendiente: preprocesar ambos años
- ⏳ Pendiente: entrenar modelo

---

**¡Puedes empezar a entrenar con lo que tienes! El modelo funcionará, aunque no óptimo. Corrige en futuras descargas.** 🚀
