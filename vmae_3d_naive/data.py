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
ZARR_URL = 'gs://gcp-public-data-arco-era5/ar/1959-2022-6h-1440x721.zarr'
# Use 5 temperature pressure levels + 10m wind components
TEMP_PRESSURE_LEVELS = [1000, 850, 700, 500, 300]  # 5 pressure levels in hPa
SINGLE_LEVEL_VARS = ['10m_u_component_of_wind', '10m_v_component_of_wind'] 

class ERA5ClipDataset(Dataset):
    """
    Custom PyTorch Dataset for loading clips of ERA5 data.
    Now handles temperature at multiple pressure levels + single level variables.
    """
    def __init__(self, data, clip_length, mean, std, temp_levels):
        """
        Args:
            data (xarray.Dataset): The pre-loaded and processed xarray dataset.
            clip_length (int): The number of time steps in each sample.
            mean (torch.Tensor): The mean of the dataset for normalization.
            std (torch.Tensor): The standard deviation of the dataset for normalization.
            temp_levels (list): List of temperature pressure levels to use.
        """
        self.data = data
        self.clip_length = clip_length
        self.temp_levels = temp_levels
        self.mean = mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        self.std = std.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

    def __len__(self):
       
        return len(self.data.time) - self.clip_length + 1

    def __getitem__(self, idx):
        time_slice = slice(idx, idx + self.clip_length)
        clip_data = self.data.isel(time=time_slice)

        # Extract temperature data at specified pressure levels
        temp_channels = []
        for level in self.temp_levels:
            temp_at_level = clip_data.sel(level=level)['t'].values
            temp_channels.append(temp_at_level)
        
        # Extract single level variables
        single_level_channels = []
        for var in SINGLE_LEVEL_VARS:
            single_level_channels.append(clip_data[var].values)
        
        # Stack all channels: 5 temp levels + 2 wind components = 7 channels
        all_channels = temp_channels + single_level_channels
        
        # (Time, Channels, Height, Width)
        clip_tensor = torch.from_numpy(
            np.stack(all_channels, axis=1)
        ).float()

        clip_tensor = (clip_tensor - self.mean) / self.std

        return clip_tensor

def calculate_mean_std(data, temp_levels):
    """
    Calculates the mean and standard deviation for each channel in the dataset.
    Handles temperature at multiple pressure levels + single level variables.
    
    Args:
        data (xarray.Dataset): The dataset to calculate stats for.
        temp_levels (list): List of temperature pressure levels to use.

    Returns:
        A tuple of PyTorch tensors (mean, std).
    """
    print("Calculating dataset statistics (mean and std)...")
    
    # Calculate stats for temperature at each pressure level
    temp_means = []
    temp_stds = []
    for level in temp_levels:
        temp_data = data.sel(level=level)['t']
        temp_means.append(temp_data.mean().item())
        temp_stds.append(temp_data.std().item())
    
    # Calculate stats for single level variables
    single_level_means = []
    single_level_stds = []
    for var in SINGLE_LEVEL_VARS:
        single_level_means.append(data[var].mean().item())
        single_level_stds.append(data[var].std().item())
    
    # Combine all means and stds
    all_means = temp_means + single_level_means
    all_stds = temp_stds + single_level_stds
    
    mean_tensor = torch.tensor(all_means)
    std_tensor = torch.tensor(all_stds)
    
    return mean_tensor, std_tensor

def get_dataloaders(
    batch_size=4,
    clip_length=8,
    image_size=224,
    num_workers=4,
    date_start='2021-01-01',
    date_end='2021-12-31'
):
    """
    Loads ERA5 data, calculates statistics, and creates train/validation DataLoaders.
    Now handles temperature at multiple pressure levels + single level variables.

    Returns:
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        mean (torch.Tensor): The calculated mean of the training data.
        std (torch.Tensor): The calculated standard deviation of the training data.
    """
    print("Connecting to Zarr store and loading data...")
    gcs = gcsfs.GCSFileSystem(token='anon')
    ds = xr.open_zarr(gcs.get_mapper(ZARR_URL), consolidated=False)

    # Load temperature data at specified pressure levels + single level variables
    temp_data = ds['t'].sel(level=TEMP_PRESSURE_LEVELS, time=slice(date_start, date_end))
    single_level_data = ds[SINGLE_LEVEL_VARS].sel(time=slice(date_start, date_end))
    
    # Combine the datasets
    data_subset = xr.merge([temp_data, single_level_data])

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
    
    mean, std = calculate_mean_std(train_data, TEMP_PRESSURE_LEVELS)
    print(f"\nCalculated Mean: {mean.tolist()}")
    print(f"Calculated Std: {std.tolist()}")

    train_dataset = ERA5ClipDataset(train_data, clip_length=clip_length, mean=mean, std=std, temp_levels=TEMP_PRESSURE_LEVELS)
    val_dataset = ERA5ClipDataset(val_data, clip_length=clip_length, mean=mean, std=std, temp_levels=TEMP_PRESSURE_LEVELS)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    print("\nDataloaders created successfully.")
    
    lat = data_resized.latitude.values
    lon = data_resized.longitude.values
    
    return train_loader, val_loader, mean, std, lat, lon