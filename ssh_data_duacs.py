import os
import glob

import numpy as np
import xarray as xr
import xesmf as xe
import regionmask


def regrid_da(da, target_grid):
    """
    Regrid a DataArray to the target grid using bilinear interpolation.
    """
    regridder = xe.Regridder(
        da, target_grid,
        method='bilinear',
        periodic=True,
        extrap_method='inverse_dist',
        extrap_num_src_pnts=8,
        reuse_weights=False,
        ignore_degenerate=False,
    )
    return regridder(da, output_chunks={'time': 240, 'y': 180, 'x': 360})


def apply_region_mask(da):
    """
    Apply a region mask based on Natural Earth's ocean basins.
    Only keeps the basins with codes in [2, 6, 60, 32, 31, 17, 55].
    """
    if 'lon' in da.coords:
        lon = da['lon']
    elif 'longitude' in da.coords:
        lon = da['longitude']
    else:
        print("No longitude coordinate found.")
        return da

    if 'lat' in da.coords:
        lat = da['lat']
    elif 'latitude' in da.coords:
        lat = da['latitude']
    else:
        print("No latitude coordinate found.")
        return da

    mask = regionmask.defined_regions.natural_earth_v5_0_0.ocean_basins_50.mask(lon, lat)
    mask = mask.isin([2, 6, 60, 32, 31, 17, 55])
    return da.where(mask, drop=True)


# Merge per-cycle NetCDF files into one dataset
input_pattern = '/dat1/smart1n/eke_trends/duacs_allsat/SEALEVEL_GLO_PHY_L4_MY_008_047/cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.25deg_P1D_202112/*/*/*.nc'
ds = xr.open_mfdataset(input_pattern, combine='by_coords', parallel=True)
print("Loaded merged dataset via open_mfdataset.")

if 'sla' not in ds:
    raise ValueError("Variable 'sla' (SSH) not found in merged dataset.")
ssh = ds['sla'].resample(time='1M').mean()

# Target 1° global lat/lon grid
target_grid = xe.util.grid_global(1, 1)

ssh_regridded = regrid_da(ssh, target_grid)
print("Regridding done.")

if 'lon' in ssh_regridded.coords:
    lon = ssh_regridded['lon']
elif 'longitude' in ssh_regridded.coords:
    lon = ssh_regridded['longitude']
else:
    raise ValueError("Longitude coordinate not found.")

if 'lat' in ssh_regridded.coords:
    lat = ssh_regridded['lat']
elif 'latitude' in ssh_regridded.coords:
    lat = ssh_regridded['latitude']
else:
    raise ValueError("Latitude coordinate not found.")
chunks = {'time': 460, 'y': 144, 'x': 108}

ssh_masked = apply_region_mask(ssh_regridded).chunk(chunks)
print("Regional mask applied.")

output_file = "duacs_data_6_may.nc"
ssh_masked.to_netcdf(output_file)
print(f"Wrote processed dataset to: {output_file}")
