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
VARIABLES = ['2m_temperature', '10m_u_component_of_wind', '10m_v_component_of_wind'] 

class ERA5ClipDataset(Dataset):
    """
    Custom PyTorch Dataset for loading clips of ERA5 data.
    """
    def __init__(self, data, clip_length, mean, std):
        """
        Args:
            data (xarray.Dataset): The pre-loaded and processed xarray dataset.
            clip_length (int): The number of time steps in each sample.
            mean (torch.Tensor): The mean of the dataset for normalization.
            std (torch.Tensor): The standard deviation of the dataset for normalization.
        """
        self.data = data
        self.clip_length = clip_length
        self.mean = mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        self.std = std.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

    def __len__(self):
       
        return len(self.data.time) - self.clip_length + 1

    def __getitem__(self, idx):
        time_slice = slice(idx, idx + self.clip_length)
        clip_data = self.data.isel(time=time_slice)

        # (Time, Channels, Height, Width)
        clip_tensor = torch.from_numpy(
            np.stack([clip_data[var].values for var in VARIABLES], axis=1)
        ).float()

        clip_tensor = (clip_tensor - self.mean) / self.std

        return clip_tensor

def calculate_mean_std(data):
    """
    Calculates the mean and standard deviation for each variable in the dataset.
    This is done efficiently using xarray's built-in functions.
    
    Args:
        data (xarray.Dataset): The dataset to calculate stats for.

    Returns:
        A tuple of PyTorch tensors (mean, std).
    """
    print("Calculating dataset statistics (mean and std)...")
    
    mean = data.mean()
    std = data.std()

    mean_tensor = torch.tensor([mean[var].item() for var in VARIABLES])
    std_tensor = torch.tensor([std[var].item() for var in VARIABLES])
    
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

    Returns:
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        mean (torch.Tensor): The calculated mean of the training data.
        std (torch.Tensor): The calculated standard deviation of the training data.
    """
    print("Connecting to Zarr store and loading data...")
    gcs = gcsfs.GCSFileSystem(token='anon')
    ds = xr.open_zarr(gcs.get_mapper(ZARR_URL), consolidated=False)

    # full year of data
    data_subset = ds[VARIABLES].sel(time=slice(date_start, date_end))

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
    
    mean, std = calculate_mean_std(train_data)
    print(f"\nCalculated Mean: {mean.tolist()}")
    print(f"Calculated Std: {std.tolist()}")

    train_dataset = ERA5ClipDataset(train_data, clip_length=clip_length, mean=mean, std=std)
    val_dataset = ERA5ClipDataset(val_data, clip_length=clip_length, mean=mean, std=std)

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