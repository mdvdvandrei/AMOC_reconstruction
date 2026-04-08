#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  Pre‑process CMIP6 data:
#     • regrid + coastal mask
#     • low‑pass filters
#     • Ekman (mass) transport  +  y_no_ekman
#     • global‑mean‑removed zos / pbo / tos
#     • statistics for piControl
#
#  Andrey Medvedev · May 2025
# ----------------------------------------------------------------------

import os, glob, warnings
import numpy as np
import xarray as xr
import xesmf as xe
import regionmask
from scipy.signal import butter, sosfilt
from scipy.ndimage import binary_dilation
import zarr

warnings.filterwarnings("ignore")

# ───────────────────────────────────────────────────────────────────────
# Globals
# ───────────────────────────────────────────────────────────────────────
R_earth = 6_371_000.0  # m
WEIGHTS_PARENT_DIR = "regrid_weights"
os.makedirs(WEIGHTS_PARENT_DIR, exist_ok=True)

# ───────────────────────────────────────────────────────────────────────
# Helper functions
# ───────────────────────────────────────────────────────────────────────
def zonal_width(lat, lon):
    """Grid‑cell zonal width (m)."""
    dlon = np.deg2rad(float(lon.diff("lon").mean()))
    return R_earth * np.cos(np.deg2rad(lat)) * dlon

def lowpass_filter(data, cutoff, order=5, fs=1, pad=4):
    """Butterworth low‑pass along axis 0 (time)."""
    sos = butter(order, cutoff, "low", output="sos", fs=fs)
    # reflect‑pad so filter start/stop aren’t distorted
    if data.ndim == 1:
        arr = np.pad(data, (pad, pad), mode="reflect")
        out = sosfilt(sos, arr)[pad:-pad]
    elif data.ndim == 2:
        arr = np.pad(data, ((pad, pad), (0, 0)), mode="reflect")
        out = sosfilt(sos, arr, axis=0)[pad:-pad, :]
    elif data.ndim == 3:
        arr = np.pad(data, ((pad, pad), (0, 0), (0, 0)), mode="reflect")
        out = sosfilt(sos, arr, axis=0)[pad:-pad, :, :]
    else:
        raise ValueError(f"Unsupported ndim={data.ndim}")
    return np.nan_to_num(out)

def regrid_da(da, target_grid, weights_dir):
    """Bilinear regrid with cached weight files."""
    da = da.where(np.isfinite(da))
    fname = f"{da.name}_to_{target_grid['lon'].size}x{target_grid['lat'].size}.nc"
    wts = os.path.join(weights_dir, fname)
    regridder = xe.Regridder(
        da, target_grid, "b ilinear",
        periodic=True,
        extrap_method="inverse_dist",
        extrap_num_src_pnts=8,
        #filename=wts,  # <-- uncomment for on‑disk weights
        reuse_weights=False,
        ignore_degenerate=False,
    )
    return regridder(da)

def apply_region_mask(da):
    """Keep only Atlantic + Arctic etc. (Natural‑Earth basin ids)."""
    lon = da["lon"] if "lon" in da.coords else da["longitude"]
    lat = da["lat"] if "lat" in da.coords else da["latitude"]
    mask = regionmask.defined_regions.natural_earth_v5_0_0.ocean_basins_50.mask(lon, lat)
    keep = [2, 6, 60, 32, 31, 17, 55]  # Atlantic + sub‑basins
    return da.where(mask.isin(keep), drop=True)

def remove_coastal(da, n=2):
    """Optional – dilate NaNs then drop a couple grid cells inland."""
    m = xr.where(np.isnan(da), 0, 1)
    return da.where(binary_dilation(m == 0, iterations=n) == 0)


def compute_ekman_transport(tau_x: xr.DataArray) -> xr.DataArray:
    lat = tau_x["lat"] if "lat" in tau_x.coords else tau_x["latitude"]
    f   = 2 * OMEGA * np.sin(np.deg2rad(lat))
    return (-tau_x / (RHO_WATER * f)).rename("U_ek")    # m² s‑1


RHO_WATER = 1025.0                 # kg m‑3
OMEGA     = 7.2921e-5              # s‑1      (Earth rotation)
R_EARTH   = 6_371_000.0            # m

# ----------------------------------------------------------------------
#  2)  Zonal integral of Uₑₖ → volume transport profile
# ----------------------------------------------------------------------
def zonal_vol_transport_profile(ekman: xr.DataArray) -> xr.DataArray:
    """
    Integrate a surface Ekman transport field zonally to obtain a
    latitude‑profile volume transport (Sverdrups).

    Parameters
    ----------
    ekman : DataArray
        Units m² s⁻¹, dims (..., lat, lon)

    Returns
    -------
    Sv : DataArray
        Zonal‑integrated transport (10⁶ m³ s⁻¹), dims = (time, lat)
    """
    lat_dim, lon_dim = ekman.dims[-2], ekman.dims[-1]
    nlat, nlon       = ekman.sizes[lat_dim], ekman.sizes[lon_dim]

    # ------------------------------------------------------------------
    # 1.  Build 2‑D longitude array (same shape as data)
    # ------------------------------------------------------------------
    if ekman[lon_dim].ndim == 1:                     # regular grid
        lon_vec = ekman[lon_dim].values[:nlon]       # enforce length = nlon
        lon_2d  = np.tile(lon_vec, (nlat, 1))        # (lat, lon)
    else:                                            # curvilinear 2‑D
        lon_2d  = ekman[lon_dim].values[:nlat, :nlon]



    # Δλ (radians per cell) along lon dimension
    dlam_2d = np.gradient(np.deg2rad(lon_2d), axis=-1)

    # ------------------------------------------------------------------
    # 2.  Build 2‑D latitude array (same shape)
    # ------------------------------------------------------------------
    if ekman[lat_dim].ndim == 1:
        lat_vec = ekman[lat_dim].values[:nlat]       # length = nlat
        lat_2d  = np.tile(lat_vec[:, None], (1, nlon))
    else:
        lat_2d  = ekman[lat_dim].values[:nlat, :nlon]

    # ------------------------------------------------------------------
    # 3.  dx and zonal integration
    # ------------------------------------------------------------------
    coslat = np.cos(np.deg2rad(lat_2d))
    dx     = R_EARTH * coslat * dlam_2d              # (lat, lon) meters

    # wrap dx as DataArray to align with ekman
    dx_da = xr.DataArray(
        dx,
        coords={
            lat_dim: ekman[lat_dim].values[:nlat],
            lon_dim: ekman[lon_dim].values[:nlon] if ekman[lon_dim].ndim == 1
                      else ekman[lon_dim].values[:nlat, :nlon],
        },
        dims=[lat_dim, lon_dim],
    )

    vol = (ekman.isel({lat_dim: slice(None, nlat), lon_dim: slice(None, nlon)})
                 * dx_da).sum(dim=lon_dim, skipna=True)          # m³ s⁻¹


    Sv = (vol / 1e6).rename("Sv_ek")                             # Sverdrups

    # if latitude was 2‑D, collapse to 1‑D after integration
    if ekman[lat_dim].ndim == 2:
        lat_1d = ekman[lat_dim].values[:nlat, :nlon].mean(axis=1)
        Sv = Sv.assign_coords({lat_dim: lat_1d})

    return Sv


'''
def build_ekman_terms(
    model,
    y_da,                  # (time, lev, rlat)
    tauu_da,               # (time, lat, lon)  – may have 2‑D lat/lon coords
    target_lat=np.arange(-30, 70 + 2.5, 2.5),
    rho0=1025.0,
    omega=7.2921e-5,
):
    """
    Returns
      ekman_3d   – Ekman *mass* transport (kg s⁻¹) broadcast to y_da shape
      y_no_ekman – y_da minus ekman_3d

    Steps
      1.  Collapse 2‑D latitude coord to 1‑D
      2.  Compute Coriolis  f(lat)
      3.  Per‑cell Ekman transport   τu / (ρfρ₀)                [m² s⁻¹]
      4.  Multiply by cell width  dx(lat)                       [m³ s⁻¹]
      5.  **Sum across lon**  (skip NaNs)                       [m³ s⁻¹]
      6.  Convert to mass (×ρ₀)                                 [kg s⁻¹]
      7.  Interpolate to y_da latitude grid (2.5°)
      8.  Broadcast & subtract
    """

    # ---------------------------------------------------------------
    # 1) Ensure latitude is 1‑D (curvilinear → collapse)
    # ---------------------------------------------------------------
    if tauu_da["lat"].ndim == 2:
        lat1d = tauu_da["lat"].isel(lon=0, drop=True)
        tauu_da = tauu_da.assign_coords(lat=lat1d)

    # Aliases for dim names
    lat_dim = "lat"
    lon_dim = "lon"

    # ---------------------------------------------------------------
    # 2) Coriolis parameter  f(lat)
    # ---------------------------------------------------------------
    f_1d = 2 * omega * np.sin(np.deg2rad(tauu_da[lat_dim]))
    f_da = xr.DataArray(f_1d, coords={lat_dim: tauu_da[lat_dim]}, dims=[lat_dim])

    # ---------------------------------------------------------------
    # 3)  Per‑cell area transport   (m² s⁻¹)
    # ---------------------------------------------------------------
    TE_cell = tauu_da / (rho0 * f_da)

    # ---------------------------------------------------------------
    # 4)  Zonal‑cell width  dx(lat)  →  broadcast to (lat, lon)
    # ---------------------------------------------------------------
    dx_1d = zonal_width(tauu_da[lat_dim], tauu_da[lon_dim])       # np.array(len(lat))
    dx_da = xr.DataArray(dx_1d, coords={lat_dim: tauu_da[lat_dim]}, dims=[lat_dim])
    dx_2d = dx_da.broadcast_like(tauu_da.isel(time=0))            # (lat, lon)

    TE_vol_cell = TE_cell * dx_2d                                 # m³ s⁻¹

    # ---------------------------------------------------------------
    # 5)  **Integrate across longitudes**  (skipna=True)
    # ---------------------------------------------------------------
    TE_vol = TE_vol_cell.sum(dim=lon_dim, skipna=True)            # (time, lat)

    # ---------------------------------------------------------------
    # 6)  Convert to mass transport
    # ---------------------------------------------------------------
    TE_mass = (TE_vol * rho0).rename("ekman_mass")                # kg s⁻¹

    # ---------------------------------------------------------------
    # 7)  Interpolate to y‑data latitude grid (dim 'rlat')
    # ---------------------------------------------------------------
    y_lat_dim = [d for d in y_da.dims if "lat" in d][0]   # usually 'rlat'

    TE_mass_i = TE_mass.interp(
        {lat_dim: y_da[y_lat_dim]},
        kwargs={"fill_value": "extrapolate"},
    )

    # ❶ drop the helper coordinate that xarray added
    if y_lat_dim in TE_mass_i.coords:
        TE_mass_i = TE_mass_i.drop_vars(y_lat_dim)

    # ❷ now safely rename the dimension
    if lat_dim != y_lat_dim:
        TE_mass_i = TE_mass_i.rename({lat_dim: y_lat_dim})

    # ---------------------------------------------------------------
    # 8)  Broadcast to 3‑D & subtract
    # ---------------------------------------------------------------
    ekman_3d = TE_mass_i.broadcast_like(y_da).rename("ekman")     # kg s⁻¹
    y_no_ekman = (y_da - ekman_3d).rename("y_no_ekman")

    return ekman_3d, y_no_ekman
'''

def build_ekman_terms(
    model,
    y_da,            # (time, lev, rlat)
    tauu_da,         # (time, lat, lon)
    rho0=RHO_WATER,
):
    """
    Returns ekman_3d  (kg s‑1, broadcast to y_da)  and  y_no_ekman
    """

    # -------- a) collapse 2‑D lat to 1‑D if needed ----------------------
    if tauu_da["lat"].ndim == 2:
        tauu_da = tauu_da.assign_coords(lat=tauu_da["lat"].isel(lon=0, drop=True), lon=tauu_da["lon"].isel(lat=0, drop=True))

    # -------- b) per‑cell Ekman velocity (m² s‑1) -----------------------
    U_ek = compute_ekman_transport(tauu_da)            # m² s‑1

    # -------- c) zonal integral → volume transport (m³ s‑1) ------------
    Sv_da = zonal_vol_transport_profile(U_ek)          # Sverdrups (10⁶ m³ s‑1)
    vol_da = (Sv_da * 1e6).rename("vol_ek")            # m³ s‑1

    # -------- d) mass transport (kg s‑1) -------------------------------
    mass_da = (vol_da * rho0).rename("ekman_mass")     # kg s‑1

    # -------- e) interpolate to y_da latitude grid ---------------------
    y_lat_dim = [d for d in y_da.dims if "lat" in d][0]   # 'rlat'
    mass_i = (
        mass_da.interp({vol_da.dims[-1]: y_da[y_lat_dim]},
                       kwargs={"fill_value": "extrapolate"})
               .drop_vars(y_lat_dim, errors="ignore")      # avoid rename clash
               .rename({vol_da.dims[-1]: y_lat_dim})
    )

    # -------- f) broadcast & subtract ----------------------------------
    ekman_3d  = mass_i.broadcast_like(y_da).rename("ekman")
    y_no_ekman = (y_da - ekman_3d).rename("y_no_ekman")

    return ekman_3d, y_no_ekman


def compute_spatial_stats(da):
    return da.mean("time"), da.std("time")

# ───────────────────────────────────────────────────────────────────────
# Variable‑specific preprocessors
# ───────────────────────────────────────────────────────────────────────
def process_x_var(
    var,
    file_list,
    target_grid,
    monthly,
    lowpass_cutoffs,
    fs,
    weights_dir,
):
    """
    Returns dict whose KEYS are exactly the DataArray .name values, e.g.:

        {'tos'                       : <xarray.DataArray ...>,
         'tos_LPF12'                 : <...>,
         'tos_minus_basin_mean'      : <...>,
         'tos_minus_basin_mean_LPF12': <...>}
    """
    da_list = []

    # ---------- 1.  open + regrid + mask ---------------------------------
    for f in file_list:
        try:
            with xr.open_dataset(f, use_cftime=True) as ds:
                if var not in ds:
                    continue
                da = regrid_da(ds[var], target_grid, weights_dir)
                da = apply_region_mask(da)
                for old, new in [
                    ("rlat", "lat"), ("latitude", "lat"), ("y", "lat"),
                    ("rlon", "lon"), ("longitude", "lon"), ("x", "lon"),
                ]:
                    if old in da.dims:
                        da = da.rename({old: new})
                da_list.append(da)
        except Exception as e:
            print(f"[{var}] {f}: {e}")

    if not da_list:
        return None

    # ---------- 2.  concat / yearly mean ---------------------------------
    combined = xr.concat(da_list, "time")
    if not monthly:
        combined = combined.resample(time="1Y").mean()

    # ---------- 3.  build result dict ------------------------------------
    #result = {var: combined}
    result = {"raw": combined}
    #result = {}
    #result[f"{var}_raw"] = combined

    if var in ("zos", "pbo", "tos"):
        mean_t = np.nanmean(combined.values, axis=(1, 2))
        minus_vals = combined.values - mean_t[:, None, None]
        minus_da = xr.DataArray(
            minus_vals, coords=combined.coords, dims=combined.dims
        ).rename(f"minus_basin_mean_raw")
        result[minus_da.name] = minus_da

    # low‑pass every field in result → add new entries

    for base_name, base_da in list(result.items()):
        for lpf_key, cutoff in lowpass_cutoffs.items():
            pad = int(2 / cutoff)
            filt = lowpass_filter(base_da.values, cutoff, order=5, fs=fs, pad=pad)
            if base_name == "minus_basin_mean_raw":
                new_name = f"minus_basin_mean_{lpf_key}"
            else:
                new_name = f"{lpf_key}"
            result[new_name] = xr.DataArray(
                filt, coords=base_da.coords, dims=base_da.dims
            ).rename(new_name)

    return result

def process_y_var(model, y_var, files, monthly,
                  lowpass_cutoffs, fs, order=5):
    """Load MOC, vertical slice, interp depth/lat, low‑pass."""
    da_list = []
    for f in files:
        try:
            with xr.open_dataset(f, use_cftime=True) as ds:
                if y_var not in ds:
                    continue
                da = ds[y_var]

                # --- multi‑basin handling ----------------------------------
                if "3basin" in da.dims:
                    da = da.rename({"3basin": "basin"})
                if "basin" in da.dims:
                    basin_idx = 1 if model in [
                        "MPI-ESM1-2-HR", "CNRM-CM6-1-HR", "EC-Earth3",
                        "IPSL-CM6A-MR1", "IPSL-CM6A-LR", "E3SM-1-0", "CAS-ESM2-0"
                    ] else 0
                    da = da.isel(basin=basin_idx)

                # --- depth coord rename -----------------------------------
                if "depth" in da.dims and "lev" not in da.dims:
                    da = da.rename({"depth": "lev"})
                if "olevel" in da.dims and "lev" not in da.dims:
                    da = da.rename({"olevel": "lev"})
                if "lev" not in da.dims:
                    raise RuntimeError("No 'lev' coordinate in y-data.")

                if model == "IPSL-CM6A-LR":
                    da = da[:,:,:,0]
                #print(da)
                #print(da.nav_lat)
                #print()
                #print(1/0)
                


                # depth units → metres
                lev_units = da.lev.attrs.get("units", "").lower()
                if lev_units in ("cm", "centimeter", "centimeters"):
                    da = da.assign_coords(lev=da.lev / 100)
                elif lev_units in ("dbar", "db"):
                    pass  # assume ≈ m

                # --- lat rename -------------------------------------------

                if model == "CAS-ESM2-0":
                    da = da.where(np.abs(da) < 1e30)



                if model == "IPSL-CM6A-LR":
                    # IPSL gives nav_lat[y] as a 1-D array
                    # attach it to the remaining 'y' dim, then rename
                    nav = ds.nav_lat
                    if "x" in nav.dims:
                        nav = nav.isel(x=0)
                    nav[-1] = 90

                    da = da.rename({"y": "rlat"})
                    # this works because dim 'rlat' now exists
                    da.coords["rlat"] = ("rlat", nav.values)



                else:
                    # for all other models, just rename whatever lat dim they use
                    for old in ("y", "nav_lat", "lat", "latitude"):
                        if old in da.dims:
                            da = da.rename({old: "rlat"})
                            break
                                
                
                # --- select 500–2500 m, interp 2.5° lat grid --------------
                da = da.sel(lev=slice(0, 2500))
                new_lat = np.arange(-30, 70 + 2.5, 2.5)
                target_lev = np.arange(0, 2501, 100)
                da = da.interp(rlat=new_lat).interp(lev=target_lev).fillna(0)

                da_list.append(da)
                


        except Exception as e:
            print(f"[{y_var}] error opening {f}: {e}")



    if not da_list:
        return None

    combined = xr.concat(da_list, "time")
    if not monthly:
        combined = combined.resample(time="1Y").mean()

    result = {"raw": combined}
    for lpf_key, cutoff in lowpass_cutoffs.items():
        pad = int(2 / cutoff)
        filt = lowpass_filter(combined.values, cutoff, order, fs, pad)
        result[lpf_key] = xr.DataArray(
            filt, coords=combined.coords, dims=combined.dims
        )

    return result

# ───────────────────────────────────────────────────────────────────────
# Main driver
# ───────────────────────────────────────────────────────────────────────
def preprocess_and_save_zarr_by_model(
    base_dir, models, scenarios, x_vars, target_grid,
    monthly, lowpass_cutoffs, fs,
    output_zarr_template, chunk_dict,
):
    for model in models:
        out = {}
        model_wts = os.path.join(WEIGHTS_PARENT_DIR, model)
        os.makedirs(model_wts, exist_ok=True)

        for scenario in scenarios:
            scen = {}
            # ------------ X ------------------------------------------------
            x_group = {}
            for var in x_vars:
                var_dir = os.path.join(base_dir, model, scenario, "r1i1p1f1", var)
                files = sorted(glob.glob(f"{var_dir}/*.nc"))
                if not files:
                    print(f"[{model}/{scenario}] no {var}")
                    continue
                data_dict = process_x_var(
                    var, files, target_grid, monthly, lowpass_cutoffs, fs, model_wts
                )
                if not data_dict:
                    continue
                for lpf_key, da in data_dict.items():
                    da = da.assign_coords(var=var, lpf=lpf_key)
                    da = da.chunk({k: v for k, v in chunk_dict.items() if k in da.dims})
                    x_group[f"{var}_{lpf_key}"] = da # x_group[f"{var}_{lpf_key}"] = da


            if x_group:
                # ─── Compute dtauu_dy for raw + apply LPFs ─────────────────────────
                if "tauu_raw" in x_group:
                    da_tauu = x_group["tauu_raw"]


                    if da_tauu["lat"].ndim == 2:
                        # take the first column (all rows identical in a global grid)
                        lat1d = da_tauu["lat"].isel(lon=0)
                        da_tauu = da_tauu.assign_coords(lat=lat1d)


                    # 1) derivative with respect to latitude (Pa per degree)
                    dtaudlat = da_tauu.differentiate("lat")

                    # 2) convert from per-degree to per-meter:
                    #    1° latitude ≈ R_earth * π/180 meters
                    dtauu_dy_raw = dtaudlat / (R_earth * np.deg2rad(1))
                    dtauu_dy_raw.name = "dtauu_dy_raw"
                    dtauu_dy_raw = dtauu_dy_raw.chunk(
                        {dim: size for dim, size in chunk_dict.items() if dim in dtauu_dy_raw.dims}
                    )
                    x_group["dtauu_dy_raw"] = dtauu_dy_raw

                    # 3) apply each low-pass filter to dtauu_dy_raw
                    for lpf_key, cutoff in lowpass_cutoffs.items():
                        pad = int(2 / cutoff)
                        filtered_vals = lowpass_filter(
                            dtauu_dy_raw.values,
                            cutoff,
                            order=5,
                            fs=fs,
                            pad=pad
                        )
                        da_filt = xr.DataArray(
                            filtered_vals,
                            coords=dtauu_dy_raw.coords,
                            dims=dtauu_dy_raw.dims
                        )
                        da_filt.name = f"dtauu_dy_{lpf_key}"
                        da_filt = da_filt.chunk(
                            {dim: size for dim, size in chunk_dict.items() if dim in da_filt.dims}
                        )
                        da_filt = da_filt.assign_coords(lpf=lpf_key)
                        x_group[f"dtauu_dy_{lpf_key}"] = da_filt

                #scenario_dict["x"] = x_group
            if x_group:
                scen["x"] = x_group

            # ------------ Y ------------------------------------------------
            y_var = (
                "msftyz"
                if model
                in [
                    "CIESM",
                    "GFDL-CM4",
                    "GFDL-ESM4",
                    "HadGEM3-GC31-LL",
                    "CMCC-ESM2",
                    "CMCC-CM2-HR4",
                    "CNRM-CM6-1-HR",
                    "EC-Earth3",
                    "HadGEM3-GC31-MM",
                    "IPSL-CM6A-LR",
                    "IPSL-CM6A-MR1",
                    "UKESM1-1-LL",
                ]
                else "msftmz"
            )
            model_path = os.path.join(base_dir, model, scenario, "r1i1p1f1")
            y_dir = None
            if os.path.isdir(model_path):
                for cand in os.listdir(model_path):
                    cand_path = os.path.join(model_path, cand)
                    f0 = sorted(glob.glob(f"{cand_path}/*.nc"))[:1]
                    if f0:
                        with xr.open_dataset(f0[0], use_cftime=True) as test_ds:
                            if y_var in test_ds:
                                y_dir = cand_path
                                break
            if y_dir:
                y_files = sorted(glob.glob(f"{y_dir}/*.nc"))
                y_dict = process_y_var(
                    model, y_var, y_files, monthly, lowpass_cutoffs, fs
                )
                if y_dict:
                    y_group = {}
                    for lpf_key, da in y_dict.items():
                        da = da.assign_coords(lpf_y=lpf_key)
                        da = da.chunk(
                            {k: v for k, v in chunk_dict.items() if k in da.dims}
                        )
                        y_group[lpf_key] = da
                    scen["y"] = y_group
            else:
                print(f"[{model}/{scenario}] no y‑var")

            # --------- Ekman & y_no_ekman ----------------------------------
            if (
                "x" in scen
                and "y" in scen
                and "tauu_raw" in scen["x"]
                and "raw" in scen["y"]
            ):
                ekman_raw, y_no_ek_raw = build_ekman_terms(
                    model, scen["y"]["raw"], scen["x"]["tauu_raw"]
                )
                ekman_raw = ekman_raw.chunk(
                    {k: v for k, v in chunk_dict.items() if k in ekman_raw.dims}
                )
                y_no_ek_raw = y_no_ek_raw.chunk(
                    {k: v for k, v in chunk_dict.items() if k in y_no_ek_raw.dims}
                )

                scen.setdefault("ekman", {})["raw"] = ekman_raw
                scen.setdefault("y_no_ekman", {})["raw"] = y_no_ek_raw

                for lpf_key, cutoff in lowpass_cutoffs.items():
                    pad = int(2 / cutoff)
                    ek_filt = xr.DataArray(
                        lowpass_filter(np.nan_to_num(ekman_raw.values), cutoff, 5, fs, pad),
                        coords=ekman_raw.coords,
                        dims=ekman_raw.dims,
                        name=f"ekman_{lpf_key}",
                    )
                    yn_filt = xr.DataArray(
                        lowpass_filter(np.nan_to_num(y_no_ek_raw.values), cutoff, 5, fs, pad),
                        coords=y_no_ek_raw.coords,
                        dims=y_no_ek_raw.dims,
                        name=f"y_no_ekman_{lpf_key}",
                    )
                    scen["ekman"][lpf_key] = ek_filt.chunk(
                        {k: v for k, v in chunk_dict.items() if k in ek_filt.dims}
                    )
                    scen["y_no_ekman"][lpf_key] = yn_filt.chunk(
                        {k: v for k, v in chunk_dict.items() if k in yn_filt.dims}
                    )

            out[scenario] = scen

            # ---------- piControl stats -----------------------------------
            if scenario.lower() == "picontrol":
                pi_stats = {"x": {}, "y": {}, "ekman": {}, "y_no_ekman": {}}
                for grp_key in ("x", "y", "ekman", "y_no_ekman"):
                    if grp_key in scen:
                        for key, da in scen[grp_key].items():
                            mu, sig = compute_spatial_stats(da)
                            pi_stats[grp_key][key] = {"mean": mu, "std": sig}
                out["piControl_stats"] = pi_stats

        # --------------- write Zarr --------------------------------------
        zpath = output_zarr_template.format(model=model)
        root = zarr.open_group(zpath, mode="w")

        def dump(group, d):
            for k, v in d.items():
                if isinstance(v, dict):
                    dump(group.require_group(k), v)
                elif isinstance(v, xr.DataArray):
                    ch = tuple(c[0] if isinstance(c, tuple) else c for c in v.chunks)
                    ds = group.create_dataset(k, data=np.asarray(v.data), chunks=ch)
                    ds.attrs.update(v.attrs)

        dump(root, out)
        zarr.consolidate_metadata(zpath)
        print(f"[✓] {model} → {zpath}")

# ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    base_dir = "/home/am334/link_am334/praki/cmip6_data"

    models = [
         #"UKESM1-1-LL"
        "ACCESS-CM2",
        "ACCESS-ESM1-5", "ACCESS-CM2", "CESM2", "GFDL-ESM4",
        "FGOALS-g3", "MRI-ESM2-0", "MIROC6", "CanESM5",
        "GISS-E2-1-G", "NorESM2-LM", "NorESM2-MM",
        "HadGEM3-GC31-LL", "UKESM1-1-LL", "CMCC-ESM2",
        "HadGEM3-GC31-MM", "MPI-ESM1-2-HR",
        "INM-CM4-8", "CanESM5-CanOE", "GFDL-CM4",
    ]

    models  = ["FGOALS-g3"]
    scenarios = ["piControl"]#, "historical", "ssp126", "ssp245", "ssp585"] # "piControl", "ssp126", "ssp245", "ssp585"


     
    x_vars = ["tos", "zos", "sos", "tauu"] # "tos", "sos", "zos", "hfds", "pbo",


    target_grid = xe.util.grid_global(1, 1)  # 1°×1°
    monthly = True

    if monthly:
        lowpass_cutoffs = {"LPF24": 1 / 24, "LPF120": 1 / 120}#"LPF12": 1 / 12, "LPF24": 1 / 24, "LPF120": 1 / 120}
        out_tpl = "monthly_1deg_grid_with_coasts_EK_minus/output_data_{model}.zarr"
    else:
        lowpass_cutoffs = {"LPF10": 1 / 10}
        out_tpl = "yearly_mean_data_EK/output_data_{model}.zarr"

    fs = 1.0
    chunk_dict = {"time": 240}

    preprocess_and_save_zarr_by_model(
        base_dir, models, scenarios, x_vars, target_grid,
        monthly, lowpass_cutoffs, fs,
        out_tpl, chunk_dict,
    )
