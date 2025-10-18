import torch
import xarray as xr
import gcsfs
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# --- Constants ---
CANADA_BOUNDS = [-141.0, 41.7, -52.6, 83.1] # [lon_min, lat_min, lon_max, lat_max]

# Temporal resolution options
TEMPORAL_RESOLUTIONS = {
    '1h': 'gs://gcp-public-data-arco-era5/ar/1959-2022-1h-1440x721.zarr',
    '6h': 'gs://gcp-public-data-arco-era5/ar/1959-2022-6h-1440x721.zarr', 
    '12h': 'gs://gcp-public-data-arco-era5/ar/1959-2022-12h-1440x721.zarr'
}

# Use 5 pressure levels for 4 variables: u, v, t, q (specific humidity)
PRESSURE_LEVELS = [1000, 850, 700, 500, 300]  # 5 pressure levels in hPa
PRESSURE_LEVEL_VARS = ['u_component_of_wind', 'v_component_of_wind', 'temperature', 'specific_humidity']  # u-wind, v-wind, temperature, specific humidity 

class ERA5ClipDataset(Dataset):
    """
    Custom PyTorch Dataset for loading clips of ERA5 data.
    """
    def __init__(self, data, clip_length, mean, std, pressure_levels, pressure_level_vars):
        """
        Args:
            data (xarray.Dataset): The pre-loaded and processed xarray dataset.
            clip_length (int): The number of time steps in each sample.
            mean (torch.Tensor): The mean of the dataset for normalization.
            std (torch.Tensor): The standard deviation of the dataset for normalization.
            pressure_levels (list): List of pressure levels to use.
            pressure_level_vars (list): List of variables to use at pressure levels.
        """
        self.data = data
        self.clip_length = clip_length
        self.pressure_levels = pressure_levels
        self.pressure_level_vars = pressure_level_vars
        self.mean = mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        self.std = std.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

    def __len__(self):
        length = len(self.data.time) - self.clip_length + 1
        return max(0, length)  # Ensure non-negative length

    def __getitem__(self, idx):
        time_slice = slice(idx, idx + self.clip_length)
        clip_data = self.data.isel(time=time_slice)

        # Extract data for each variable at each pressure level
        all_channels = []
        for var in self.pressure_level_vars:
            for level in self.pressure_levels:
                var_at_level = clip_data.sel(level=level)[var].values
                all_channels.append(var_at_level)
        
        # Stack all channels: 4 variables × 5 levels = 20 channels
        # (Time, Channels, Height, Width)
        clip_tensor = torch.from_numpy(
            np.stack(all_channels, axis=1)
        ).float()

        clip_tensor = (clip_tensor - self.mean) / self.std

        return clip_tensor

def calculate_mean_std(data, pressure_levels, pressure_level_vars):
    """
    Calculates the mean and standard deviation for each channel in the dataset.
    Handles 4 variables at multiple pressure levels.
    
    Args:
        data (xarray.Dataset): The dataset to calculate stats for.
        pressure_levels (list): List of pressure levels to use.
        pressure_level_vars (list): List of variables to use at pressure levels.

    Returns:
        A tuple of PyTorch tensors (mean, std).
    """
    print("Calculating dataset statistics (mean and std)...")
    
    # Calculate stats for each variable at each pressure level
    all_means = []
    all_stds = []
    
    for var in pressure_level_vars:
        for level in pressure_levels:
            var_data = data.sel(level=level)[var]
            all_means.append(var_data.mean().item())
            all_stds.append(var_data.std().item())
    
    mean_tensor = torch.tensor(all_means)
    std_tensor = torch.tensor(all_stds)
    
    return mean_tensor, std_tensor

def get_dataloaders(
    batch_size=4,
    clip_length=8,
    image_size=224,
    num_workers=4,
    temporal_resolution='6h',
    date_start='2021-01-01',
    date_end='2021-12-31'
):
    """
    Loads ERA5 data, calculates statistics, and creates train/validation DataLoaders.

    Args:
        temporal_resolution (str): Temporal resolution ('1h', '6h', or '12h')
        Other args same as before

    Returns:
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        mean (torch.Tensor): The calculated mean of the training data.
        std (torch.Tensor): The calculated standard deviation of the training data.
    """
    print(f"Connecting to Zarr store and loading data...")
    print(f"Temporal resolution: {temporal_resolution}")
    
    # Get the appropriate Zarr URL
    zarr_url = TEMPORAL_RESOLUTIONS[temporal_resolution]
    
    gcs = gcsfs.GCSFileSystem(token='anon')
    ds = xr.open_zarr(gcs.get_mapper(zarr_url), consolidated=False)

    # Load data for each variable at specified pressure levels
    data_vars = []
    for var in PRESSURE_LEVEL_VARS:
        var_data = ds[var].sel(level=PRESSURE_LEVELS, time=slice(date_start, date_end))
        data_vars.append(var_data)
    
    # Combine the datasets
    data_subset = xr.merge(data_vars)

    # Normalize longitudes to [-180, 180] and filter to Canada's bounding box
    data_subset = data_subset.assign_coords(
        longitude=((data_subset.longitude + 180) % 360) - 180
    ).sortby(['longitude', 'latitude'])
    mask = (data_subset.longitude >= CANADA_BOUNDS[0]) & (data_subset.longitude <= CANADA_BOUNDS[2]) & \
           (data_subset.latitude >= CANADA_BOUNDS[1]) & (data_subset.latitude <= CANADA_BOUNDS[3])
    data_canada = data_subset.where(mask, drop=True)

    lat_min = float(data_canada.latitude.min().values)
    lat_max = float(data_canada.latitude.max().values)
    lon_min = float(data_canada.longitude.min().values)
    lon_max = float(data_canada.longitude.max().values)

    data_resized = data_canada.interp(
        latitude=np.linspace(lat_min, lat_max, image_size),
        longitude=np.linspace(lon_min, lon_max, image_size),
        method="linear"
    )
    
    print("Loading data into memory...")
    data_resized.load()
    print("Data loaded.")
    
    time_len = len(data_resized.time)
    train_size = int(0.8 * time_len)
    
    train_data = data_resized.isel(time=slice(0, train_size))
    val_data = data_resized.isel(time=slice(train_size, time_len))
    
    mean, std = calculate_mean_std(train_data, PRESSURE_LEVELS, PRESSURE_LEVEL_VARS)
    print(f"\nCalculated Mean: {mean.tolist()}")
    print(f"Calculated Std: {std.tolist()}")

    train_dataset = ERA5ClipDataset(train_data, clip_length=clip_length, mean=mean, std=std, 
                                   pressure_levels=PRESSURE_LEVELS, pressure_level_vars=PRESSURE_LEVEL_VARS)
    val_dataset = ERA5ClipDataset(val_data, clip_length=clip_length, mean=mean, std=std, 
                                 pressure_levels=PRESSURE_LEVELS, pressure_level_vars=PRESSURE_LEVEL_VARS)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    print("\nDataloaders created successfully.")
    print(f"Data resolution: {temporal_resolution}")
    
    lat = data_resized.latitude.values
    lon = data_resized.longitude.values
    
    return train_loader, val_loader, mean, std, lat, lon