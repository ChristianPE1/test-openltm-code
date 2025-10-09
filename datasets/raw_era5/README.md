# 📁 ERA5 Raw Data Directory

## Purpose
This directory stores raw ERA5 NetCDF files (.zip format) before preprocessing.

## Expected Files

Upload your ERA5 data files here with the following naming convention:

```
era5_peru_2022.zip
era5_peru_2023.zip
era5_peru_2024.zip
...
era5_peru_2014.zip (for full 10-year dataset)
```

## File Requirements

Each .zip file should contain:
- **Format**: NetCDF (.nc)
- **Variables**: 10 atmospheric variables
  - `tp` - Total precipitation
  - `t2m` - Temperature at 2m
  - `d2m` - Dewpoint temperature at 2m
  - `sp` - Surface pressure
  - `msl` - Mean sea level pressure
  - `u10` - U-component of wind at 10m
  - `v10` - V-component of wind at 10m
  - `tcwv` - Total column water vapour
  - `cape` - Convective available potential energy
- **Temporal Resolution**: 12-hourly (06:00, 18:00 UTC)
- **Spatial Coverage**: Peru region
  - Latitude: 0°N to -18°S
  - Longitude: -82°W to -68°W
- **Spatial Resolution**: 0.25° × 0.25°

## How to Obtain Data

### Option 1: Manual Download (CDS Website)

1. Go to: https://cds.climate.copernicus.eu/
2. Register/Login
3. Navigate to: "ERA5 hourly data on single levels from 1940 to present"
4. Configure download:
   - Product: Reanalysis
   - Variables: Select the 10 required variables
   - Year: [Select specific year]
   - Month: All
   - Day: All  
   - Time: 06:00, 18:00
   - Geographical area: Custom
     - North: 0°
     - West: -82°
     - South: -18°
     - East: -68°
   - Format: NetCDF
5. Submit form and download
6. Rename to: `era5_peru_YYYY.nc`
7. Compress to: `era5_peru_YYYY.zip`
8. Upload here

### Option 2: Google Colab Upload

```python
# In Colab notebook
from google.colab import files

uploaded = files.upload()

# Move to raw_era5
!mv era5_peru_*.zip /content/AdaptationOpenLTM/datasets/raw_era5/
```

### Option 3: Google Drive

```python
# Copy from Drive
!cp '/content/drive/MyDrive/ERA5_Data/era5_peru_2023.zip' datasets/raw_era5/
```

## Recommended Download Order

### Phase 1: Testing (Start Here)
```
era5_peru_2022.zip
era5_peru_2023.zip
era5_peru_2024.zip
Total: ~900 MB
```

### Phase 2: Final Model (After validation)
```
era5_peru_2014.zip
era5_peru_2015.zip
...
era5_peru_2024.zip
Total: ~3.3 GB
```

## File Size Reference

- **Per file**: ~300 MB (compressed)
- **3 years**: ~900 MB
- **7 years**: ~2.1 GB
- **10 years**: ~3.0 GB
- **11 years (2014-2024)**: ~3.3 GB

## Next Steps

After uploading files here:

1. Run preprocessing:
   ```bash
   python preprocessing/preprocess_era5_peru.py \
       --input_dir datasets/raw_era5 \
       --output_dir datasets/processed \
       --years 2022,2023,2024
   ```

2. Check processed data:
   ```bash
   ls -lh datasets/processed/
   ```

3. Start training:
   ```bash
   bash scripts/adaptation/peru_rainfall/train_timerxl_peru.sh
   ```

## Troubleshooting

**Issue**: File not found after extraction
- Check if .nc file is inside another subdirectory
- Manually extract and rename if needed

**Issue**: CDS download fails
- Check API key configuration
- Try downloading smaller time ranges
- Use manual download as fallback

**Issue**: Storage full
- Download years incrementally
- Process and delete .nc files after creating .csv
- Use Google Drive for long-term storage

## Additional Resources

- CDS Tutorial: https://cds.climate.copernicus.eu/tutorial
- ERA5 Documentation: https://confluence.ecmwf.int/display/CKB/ERA5
- Data Guide: See `GUIA_DESCARGA_DATOS.md` in parent directory

---

**Status**: Empty (waiting for data upload)  
**Last Updated**: October 2025
