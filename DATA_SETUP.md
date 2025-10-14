# Data Setup Guide

This document explains how to set up and access the ERA5 weather data used in this project.

## 🌍 Data Source

The project uses **ERA5 reanalysis data** from the European Centre for Medium-Range Weather Forecasts (ECMWF), specifically:

- **Dataset**: ERA5 hourly data on single levels from 1959 to 2022
- **Source**: Google Cloud Public Datasets
- **Format**: Zarr format for efficient access
- **Variables**: 2m temperature, 10m U-wind, 10m V-wind components
- **Spatial Coverage**: Canada (filtered from global data)
- **Temporal Resolution**: 6-hourly data
- **Spatial Resolution**: 0.25° × 0.25° (approximately 25km)

## 🔗 Data Access

### Automatic Access (Recommended)

The code automatically handles data access through Google Cloud Storage:

```python
# Data is automatically loaded from:
ZARR_URL = 'gs://gcp-public-data-arco-era5/ar/1959-2022-6h-1440x721.zarr'
```

**No additional setup required** - the data is publicly accessible and the code handles authentication automatically.

### Manual Data Access (Optional)

If you need to access the data manually or want to understand the data structure:

1. **Browse the dataset**: [ERA5 on Google Cloud](https://console.cloud.google.com/marketplace/product/noaa-public/era5-pds)
2. **Documentation**: [ERA5 Documentation](https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation)
3. **API Access**: [CDS API](https://cds.climate.copernicus.eu/api-how-to) (for custom downloads)

## 📊 Data Processing

The code automatically:

1. **Filters to Canada**: Bounding box [-141.0, 41.7, -52.6, 83.1]
2. **Resizes to square**: Interpolates to 224×224 pixels (configurable)
3. **Normalizes**: Calculates mean/std from training data
4. **Creates clips**: Generates 8-frame video sequences

### Data Variables

| Variable | Description | Units |
|----------|-------------|-------|
| `2m_temperature` | Air temperature at 2m height | Kelvin |
| `10m_u_component_of_wind` | Eastward wind component at 10m | m/s |
| `10m_v_component_of_wind` | Northward wind component at 10m | m/s |

## 💾 Storage Requirements

- **Memory**: ~8-16GB RAM recommended for data loading
- **Storage**: No local storage required (streaming from cloud)
- **Network**: Stable internet connection for data streaming

## 🔧 Configuration

You can modify data parameters via command line arguments:

```bash
# Change date range
python main.py --date_start 2020-01-01 --date_end 2020-12-31

# Change spatial resolution
python main.py --image_size 512  # 512x512 pixels

# Change temporal sequence length
python main.py --num_frames 16  # 16 frames per clip

# All together
python main.py \
    --date_start 2020-01-01 \
    --date_end 2020-12-31 \
    --image_size 256 \
    --num_frames 12
```

## 🚨 Troubleshooting

### Common Issues

1. **Network Timeout**: 
   - Check internet connection
   - Try running with smaller batch size
   - Consider downloading subset locally

2. **Memory Issues**:
   - Reduce batch size
   - Use fewer workers
   - Process smaller time ranges

3. **Authentication Errors**:
   - Ensure you have internet access
   - Check if Google Cloud services are accessible in your region

### Performance Tips

- **Faster Loading**: Increase `num_workers` in DataLoader
- **Memory Efficient**: Use smaller `image_size` or `clip_length`
- **Local Caching**: Consider downloading data subset for offline use

## 📈 Data Statistics

Typical data characteristics:
- **Training samples**: ~1,400 clips (80% of year)
- **Validation samples**: ~350 clips (20% of year)
- **Clip duration**: 8 frames × 6 hours = 48 hours
- **Spatial coverage**: ~4,000 km × 4,000 km (Canada)

## 🔗 Additional Resources

- [ERA5 Documentation](https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation)
- [Google Cloud ERA5](https://console.cloud.google.com/marketplace/product/noaa-public/era5-pds)
- [Xarray Documentation](https://xarray.pydata.org/)
- [Zarr Format](https://zarr.readthedocs.io/)
