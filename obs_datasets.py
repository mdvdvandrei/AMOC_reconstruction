

import os
import numpy as np
import xarray as xr
import torch
from torch.utils.data import Dataset
from scipy.signal import butter, sosfilt

import matplotlib.pyplot as plt

import scipy.stats as st

import glob
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import get_original_cwd
from tqdm import tqdm
import zarr
from filelock import FileLock
import warnings
warnings.filterwarnings("ignore")
import dask
dask.config.set(scheduler='synchronous')
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
    

import os
import glob
import numpy as np
import xarray as xr
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ─── Your model and dataset imports ─────────────────────────────────────────
from models import ResidualCNNHet








def compute_slope_ci_with_uncertainty(y, sigma, alpha=0.05):
    """
    Linear trend slope + CI via Weighted Least Squares (WLS) with known uncertainties.
    
    Parameters:
    -----------
    y : array-like of observations
    sigma : array-like of standard deviations (uncertainty for each point)
    alpha : significance level (default 0.05 for 95% CI)
    
    Returns:
    --------
    slope : trend slope
    intercept : y-intercept
    lower_ci : lower confidence interval for slope
    upper_ci : upper confidence interval for slope
    slope_se : standard error of the slope
    trend_ci_lower : lower CI for entire trend line
    trend_ci_upper : upper CI for entire trend line
    """
    y = np.asarray(y)
    sigma = np.asarray(sigma)
    x = np.arange(len(y))
    
    # Weights (inverse variance)
    weights = 1.0 / (sigma**2)
    
    # Design matrix
    X = np.column_stack([np.ones(len(x)), x])
    W = np.diag(weights)
    
    # Weighted least squares: β = (X'WX)^(-1) X'Wy
    XtWX = X.T @ W @ X
    XtWy = X.T @ W @ y
    
    try:
        XtWX_inv = np.linalg.inv(XtWX)
        beta = XtWX_inv @ XtWy
    except np.linalg.LinAlgError:
        # Fallback to pseudoinverse if singular
        XtWX_inv = np.linalg.pinv(XtWX)
        beta = XtWX_inv @ XtWy
    
    intercept, slope = beta
    
    # Covariance matrix of coefficients
    # For WLS: Cov(β) = (X'WX)^(-1)
    cov_beta = XtWX_inv
    
    # Standard error of slope (second diagonal element)
    slope_se = np.sqrt(cov_beta[1, 1])
    
    # Degrees of freedom
    df = len(y) - 2
    
    # Critical t-value
    t_crit = st.t.ppf(1 - alpha/2, df)
    
    # Confidence interval for slope
    lower_ci = slope - t_crit * slope_se
    upper_ci = slope + t_crit * slope_se
    
    # Confidence intervals for the entire trend line
    y_pred = X @ beta
    
    # Standard errors for predictions at each point
    # se(ŷ) = sqrt(x'(X'WX)^(-1)x) for each point
    pred_se = np.sqrt(np.diag(X @ cov_beta @ X.T))
    
    trend_ci_lower = y_pred - t_crit * pred_se
    trend_ci_upper = y_pred + t_crit * pred_se
    
    return slope, intercept, lower_ci, upper_ci, slope_se, trend_ci_lower, trend_ci_upper


def compute_slope_ci(y, alpha=0.05):
    """
    DEPRECATED: Use compute_slope_ci_with_uncertainty instead.
    Linear trend slope + 95% CI via OLS (for backward compatibility).
    """
    x = np.arange(len(y))
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    resid = y - y_pred
    df    = len(x) - 2
    s_err = np.sqrt((resid**2).sum() / df)
    ssx   = ((x - x.mean())**2).sum()
    se_slope = s_err / np.sqrt(ssx)
    t_val = st.t.ppf(1 - alpha/2, df)
    lower = slope - t_val * se_slope
    upper = slope + t_val * se_slope
    return slope, intercept, lower, upper


# Updated main analysis code (replace the relevant section):
def analyze_trends_with_uncertainty(times, mu, sigma, start_date, end_date):
    """
    Analyze trends with uncertainty using WLS.
    
    Parameters:
    -----------
    times : array of datetime objects
    mu : mean predictions
    sigma : total uncertainty (aleatoric + epistemic)
    start_date, end_date : datetime64 objects for trend window
    
    Returns:
    --------
    Dictionary with trend analysis results
    """
    # Create mask for time window
    mask = (times >= start_date) & (times <= end_date)
    
    if mask.sum() < 3:  # Need at least 3 points
        raise ValueError(f"Not enough data points in window {start_date} to {end_date}")
    
    # Extract windowed data
    y_window = mu[mask]
    sigma_window = sigma[mask]  # Use total uncertainty
    times_window = times[mask]
    
    # Compute WLS trend with uncertainty
    (slope, intercept, slope_lower, slope_upper, 
     slope_se, trend_lower, trend_upper) = compute_slope_ci_with_uncertainty(
        y_window, sigma_window
    )
    
    # Convert slope to desired units (Sv/century)
    factor = 12 * 100  # monthly to century
    trend_slope = slope * factor
    trend_margin = (slope_upper - slope_lower) / 2 * factor
    
    return {
        'slope': slope,
        'intercept': intercept,
        'slope_se': slope_se,
        'trend_slope_sv_century': trend_slope,
        'trend_margin_sv_century': trend_margin,
        'trend_lower': trend_lower,
        'trend_upper': trend_upper,
        'times_window': times_window,
        'mask': mask
    }




def lowpass_filter(data, cutoff_freq, order=5, fs=1, pad=2):
    """
    Apply a Butterworth low-pass filter along the time axis.

    Args:
        data (ndarray): input data (1D, 2D, or 3D).
        cutoff_freq (float): cutoff frequency.
        order (int): filter order.
        fs (float): sampling frequency.
        pad (int): reflection padding size.

    Returns:
        Filtered data with NaNs replaced by zeros.
    """
    sos = butter(order, cutoff_freq, btype='low', output='sos', fs=fs)
    if data.ndim == 1:
        padded = np.pad(data, (pad, pad), mode='reflect')
        filtered = sosfilt(sos, padded)[pad:-pad]
    elif data.ndim == 2:
        padded = np.pad(data, ((pad, pad), (0, 0)), mode='reflect')
        filtered = sosfilt(sos, padded, axis=0)[pad:-pad, :]
    elif data.ndim == 3:
        padded = np.pad(data, ((pad, pad), (0, 0), (0, 0)), mode='reflect')
        filtered = sosfilt(sos, padded, axis=0)[pad:-pad, :, :]
    else:
        raise ValueError(f"Unsupported data dimensions: {data.ndim}")
    return np.nan_to_num(filtered)


def compute_slope_ci(y, alpha=0.05):
    """
    Linear trend slope + 95% CI via OLS.
    y : array-like of observations.
    Returns: slope, intercept, lower_ci, upper_ci.
    """
    x = np.arange(len(y))
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    resid = y - y_pred
    df    = len(x) - 2
    s_err = np.sqrt((resid**2).sum() / df)
    ssx   = ((x - x.mean())**2).sum()
    se_slope = s_err / np.sqrt(ssx)
    t_val = st.t.ppf(1 - alpha/2, df)
    lower = slope - t_val * se_slope
    upper = slope + t_val * se_slope
    return slope, intercept, lower, upper



class PreprocessedDataset(Dataset):
    def __init__(self, file_path, variable='sst', lpf='raw', order=5, fs=1, transform=None, monthly = True, minus_basin_mean = False):
        """
        Args:
            file_path (str): path to the NetCDF file.
            variable (str): variable name to load (e.g. 'sst' or 'tos').
            lpf (str): filter key; 'raw' means no filtering, otherwise LPF is applied.
            cutoff_freq (float): LPF cutoff frequency.
            order (int): Butterworth filter order.
            fs (float): sampling frequency.
            pad (int): padding length for the filter.
            transform (callable, optional): optional extra transform on the sample.
        """
        self.file_path = file_path
        self.variable = variable
        self.lpf = lpf
        
        self.cutoff_freq = 0
        if self.lpf == 'LPF120':
            self.cutoff_freq = 1/120
        elif self.lpf == 'LPF24':
            self.cutoff_freq = 1/24
        elif self.lpf == 'LPF10':
            # 10-year low pass = 120 months
            self.cutoff_freq = 1/120

            
        self.order = order
        self.fs = fs
        if lpf != 'raw':
            self.pad = int(2/self.cutoff_freq)
        else:
            self.pad = 0
        
        
        self.transform = transform
        
        # Load from NetCDF (not Zarr)
        self.ds = xr.open_dataset(file_path)

        # Extract variable (expected dims: time, lat, lon)
        self.data = self.ds[variable]

        open_water = (np.abs(self.data - (-1.8)) > 0.05) | ~np.isfinite(self.data)
        self.data = self.data.where(open_water, other=-1.8)




        
        if minus_basin_mean == True: 
            mean_NA = np.nanmean(self.data.values, axis=(1, 2))
            mean_repeated = np.repeat(mean_NA[:, np.newaxis, np.newaxis], self.data.shape[1], axis=1)
            mean_repeated = np.repeat(mean_repeated, self.data.shape[2], axis=2)
            self.data = self.data - mean_repeated
        
        

        if monthly == False:
            self.data_annual = self.data.resample(time='1Y').mean()
            #self.detrended_data = detrending(self.data[:])
            self.mean = self.data_annual[10:70].mean(dim='time')
        else:
            #self.detrended_data = detrending(self.data[:12*100])
            self.mean = self.data[10*12:70*12].mean(dim='time') # during training, anomalies are computed relative to the piControl mean; during inference on observations, anomalies are computed relative to the 1880–1940 mean
            

        
        # Optional LPF along time
        if lpf != 'raw':
            data_np = self.data.values  # shape: (time, lat, lon)
            data_filtered = lowpass_filter(data_np, self.cutoff_freq, order, fs, self.pad)
            # rebuild DataArray with same coords/dims
            self.data = xr.DataArray(data_filtered, dims=self.data.dims, coords=self.data.coords)
        if monthly == False:
            self.data = self.data.resample(time='1Y').mean()
            

        
        
        self.num_samples = self.data.shape[0]
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        sample = self.data.isel(time=idx)
        sample = (sample-self.mean) #/ 0.16530  #/self.std
        #sample = sample.where((sample >= -30) & (sample <= 40))

        sample_np = sample.values.astype(np.float32)
        sample_np = np.nan_to_num(sample_np)
        if self.transform:
            sample_np = self.transform(sample_np)
        return torch.from_numpy(sample_np)
    
    



class PreprocessedSSHDatasetFromZarr(Dataset):
    def __init__(self, zarr_path, variable='sossheig', lpf='raw', order=5, fs=1, transform=None, monthly = True):
        """
        Args:
            zarr_path (str): path to the Zarr store with SSH data.
            variable (str): variable name (e.g. 'sossheig' or 'ssh').
            lpf (str): 'raw' for no filter, else LPF ('LPF120' or 'LPF24').
            order (int): Butterworth filter order.
            fs (float): sampling frequency.
            transform (callable, optional): optional extra transform on the sample.

        Data are read from Zarr using the chunking defined when the store was written.
        """
        self.zarr_path = zarr_path
        self.variable = variable
        self.lpf = lpf
        self.order = order
        self.fs = fs
        self.transform = transform

        self.ds = xr.open_zarr(zarr_path)
        self.data = self.ds[variable]  # expected dims: (time, lat, lon)
        
        
        mean_NA = np.nanmean(self.data.values, axis=(1, 2))
        mean_repeated = np.repeat(mean_NA[:, np.newaxis, np.newaxis], self.data.shape[1], axis=1)
        mean_repeated = np.repeat(mean_repeated, self.data.shape[2], axis=2)
        self.data = self.data - mean_repeated
        
        
        
        # Normalization stats (see mean/std below)

        # Optional LPF
        if self.lpf != 'raw':
            if self.lpf == 'LPF120':
                cutoff_freq = 1/120
            elif self.lpf == 'LPF24':
                cutoff_freq = 1/24
            else:
                cutoff_freq = 0
            pad = int(2/cutoff_freq) if cutoff_freq > 0 else 0
            data_np = self.data.values
            data_filtered = lowpass_filter(data_np, cutoff_freq, order, fs, pad)
            self.data = xr.DataArray(data_filtered, dims=self.data.dims, coords=self.data.coords)
        
        if monthly == False:
            self.data = self.data.resample(time='1Y').mean()
            print(self.data.shape)
            
            
        if monthly == False:
            self.mean = self.data[-30:].mean(dim='time')
            self.std = self.data[-30:].std(dim='time')
        else:
            self.mean = self.data[-12*30:].mean(dim='time')
            self.std = self.data[-12*30:].std(dim='time')
        
        self.num_samples = self.data.sizes['time']

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        sample = self.data.isel(time=idx)
        sample = (sample - self.mean) #/ std_global #self.std
        #print(self.data.coords["time"][idx])
        sample_np = sample.values.astype(np.float32)
        sample_np = np.nan_to_num(sample_np)
        if self.transform:
            sample_np = self.transform(sample_np)
        return torch.from_numpy(sample_np)

