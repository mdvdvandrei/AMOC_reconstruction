#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  Pre-process CMIP5 data:
#     • regrid + Atlantic mask (optional coastal trimming helper kept)
#     • low-pass filters
#     • Ekman (mass) transport  +  y_no_ekman
#     • basin-mean-removed zos / pbo / tos
#     • statistics for piControl
#
#  Andrei Medvedev · CMIP5 rewrite (Oct 2025)
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
RHO_WATER = 1025.0     # kg m-3
OMEGA     = 7.2921e-5  # s-1
R_EARTH   = 6_371_000.0
WEIGHTS_PARENT_DIR = "regrid_weights"
os.makedirs(WEIGHTS_PARENT_DIR, exist_ok=True)

# CMIP5 Y-var + basin mapping (your list)
MODEL_YVAR = {
    "ACCESS1-0":   ("msftyyz", 0),  # Atlantic = 0
    "ACCESS1-3":   ("msftyyz", 0),
    "CanESM2":     ("msftyyz", 0),
    "FGOALS-s2":   ("msftyyz", 1),
    "MPI-ESM-LR":  ("msftmyz", 0),
    "MPI-ESM-MR":  ("msftmyz", 0),
    "MRI-CGCM3":   ("msftmyz", 0),
    "MIROC5":   ("msftmyz", 0),
    "NorESM1-M":  ("msftmyz", 0),
    "NorESM1-ME":  ("msftmyz", 0),

}

DEFAULT_YVAR = ("msftmyz", 0)  # fallback if an unseen CMIP5 model appears

# ───────────────────────────────────────────────────────────────────────
# Helper functions
# ───────────────────────────────────────────────────────────────────────
def zonal_width(lat, lon):
    """Grid-cell zonal width (m)."""
    dlon = np.deg2rad(float(lon.diff("lon").mean()))
    return R_earth * np.cos(np.deg2rad(lat)) * dlon

def lowpass_filter(data, cutoff, order=5, fs=1, pad=4):
    """Butterworth low-pass along axis 0 (time)."""
    sos = butter(order, cutoff, "low", output="sos", fs=fs)
    # reflect-pad so filter start/stop aren’t distorted
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
    """Bilinear regrid with cached weight files (in-memory by default)."""
    da = da.where(np.isfinite(da))
    # normalize coord names before weight building
    rn = {}
    if "latitude" in da.coords and "lat" not in da.coords: rn["latitude"] = "lat"
    if "longitude" in da.coords and "lon" not in da.coords: rn["longitude"] = "lon"
    if rn: da = da.rename(rn)

    regridder = xe.Regridder(
        da, target_grid, "bilinear",
        periodic=True,
        extrap_method="inverse_dist",
        extrap_num_src_pnts=8,
        reuse_weights=False,
        ignore_degenerate=False,
    )
    return regridder(da)

def apply_region_mask(da):
    """Keep only Atlantic + marginal seas (Natural-Earth basin ids)."""
    lon = da["lon"] if "lon" in da.coords else da["longitude"]
    lat = da["lat"] if "lat" in da.coords else da["latitude"]
    mask = regionmask.defined_regions.natural_earth_v5_0_0.ocean_basins_50.mask(lon, lat)
    keep = [2, 6, 60, 32, 31, 17, 55]  # Atlantic + sub-basins
    return da.where(mask.isin(keep), drop=True)

def remove_coastal(da, n=2):
    """Optional – dilate NaNs then drop a couple grid cells inland."""
    m = xr.where(np.isnan(da), 0, 1)
    return da.where(binary_dilation(m == 0, iterations=n) == 0)

def compute_ekman_transport(tau_x: xr.DataArray) -> xr.DataArray:
    lat = tau_x["lat"] if "lat" in tau_x.coords else tau_x["latitude"]
    f   = 2 * OMEGA * np.sin(np.deg2rad(lat))
    return (-tau_x / (RHO_WATER * f)).rename("U_ek")  # m² s-1

def zonal_vol_transport_profile(ekman: xr.DataArray) -> xr.DataArray:
    """Zonal integral of Uₑₖ → Sverdrups vs latitude."""
    lat_dim, lon_dim = ekman.dims[-2], ekman.dims[-1]
    nlat, nlon       = ekman.sizes[lat_dim], ekman.sizes[lon_dim]

    # lon 2-D (or tiled from 1-D)
    if ekman[lon_dim].ndim == 1:
        lon_vec = ekman[lon_dim].values[:nlon]
        lon_2d  = np.tile(lon_vec, (nlat, 1))
    else:
        lon_2d  = ekman[lon_dim].values[:nlat, :nlon]
    dlam_2d = np.gradient(np.deg2rad(lon_2d), axis=-1)

    # lat 2-D (or tiled from 1-D)
    if ekman[lat_dim].ndim == 1:
        lat_vec = ekman[lat_dim].values[:nlat]
        lat_2d  = np.tile(lat_vec[:, None], (1, nlon))
    else:
        lat_2d  = ekman[lat_dim].values[:nlat, :nlon]

    dx = R_EARTH * np.cos(np.deg2rad(lat_2d)) * dlam_2d

    dx_da = xr.DataArray(
        dx,
        coords={
            lat_dim: ekman[lat_dim].values[:nlat],
            lon_dim: ekman[lon_dim].values if ekman[lon_dim].ndim == 1
                    else ekman[lon_dim].values[:nlat, :nlon],
        },
        dims=[lat_dim, lon_dim],
    )

    vol = (ekman.isel({lat_dim: slice(None, nlat), lon_dim: slice(None, nlon)}) * dx_da)\
            .sum(dim=lon_dim, skipna=True)  # m³ s⁻¹
    Sv = (vol / 1e6).rename("Sv_ek")       # Sverdrups
    if ekman[lat_dim].ndim == 2:
        lat_1d = ekman[lat_dim].values[:nlat, :nlon].mean(axis=1)
        Sv = Sv.assign_coords({lat_dim: lat_1d})
    return Sv

def build_ekman_terms(model, y_da, tauu_da, rho0=RHO_WATER):
    """Return ekman_3d (kg/s, broadcast to y_da) and y_no_ekman."""
    # collapse 2-D lat/lon coords if present
    if "lat" in tauu_da.coords and getattr(tauu_da["lat"], "ndim", 1) == 2:
        tauu_da = tauu_da.assign_coords(
            lat=tauu_da["lat"].isel(lon=0, drop=True),
            lon=tauu_da["lon"].isel(lat=0, drop=True)
        )

    U_ek  = compute_ekman_transport(tauu_da)  # m²/s
    Sv_da = zonal_vol_transport_profile(U_ek) # Sv
    vol_da  = (Sv_da * 1e6).rename("vol_ek")  # m³/s
    mass_da = (vol_da * rho0).rename("ekman_mass")

    y_lat_dim = [d for d in y_da.dims if "lat" in d][0]  # 'rlat'
    src_lat   = mass_da.dims[-1]                         # 'lat'
    mass_i = (
        mass_da.interp({src_lat: y_da[y_lat_dim]}, kwargs={"fill_value": "extrapolate"})
                .drop_vars(y_lat_dim, errors="ignore")
                .rename({src_lat: y_lat_dim})
    )

    ekman_3d   = mass_i.broadcast_like(y_da).rename("ekman")
    y_no_ekman = (y_da - ekman_3d).rename("y_no_ekman")
    return ekman_3d, y_no_ekman

def compute_spatial_stats(da):
    return da.mean("time"), da.std("time")

# ───────────────────────────────────────────────────────────────────────
# Variable-specific preprocessors
# ───────────────────────────────────────────────────────────────────────
def process_x_var(
    var, file_list, target_grid, monthly, lowpass_cutoffs, fs, weights_dir,
):
    """
    Returns dict whose KEYS are DataArray names:
      {'raw', 'LPF24', 'LPF120', 'minus_basin_mean_raw', 'minus_basin_mean_LPF24', ...}
    """
    da_list = []

    # 1) open + regrid + Atlantic mask
    for f in file_list:
        try:
            with xr.open_dataset(f, use_cftime=True) as ds:
                if var not in ds:
                    continue
                da = ds[var]
                # normalize coords
                
                da = regrid_da(da, target_grid, weights_dir)
                da = apply_region_mask(da)
                
                rn = {}
                if "latitude" in da.coords and "lat" not in da.coords: rn["latitude"] = "lat"
                if "longitude" in da.coords and "lon" not in da.coords: rn["longitude"] = "lon"
                if rn: da = da.rename(rn)


                da_list.append(da)
        except Exception as e:
            print(f"[{var}] {f}: {e}")

    if not da_list:
        return None

    combined = xr.concat(da_list, "time")
    if not monthly:
        combined = combined.resample(time="1Y").mean()

    result = {"raw": combined}

    if var in ("zos", "pbo", "tos"):
        mean_t = np.nanmean(combined.values, axis=(1, 2))
        minus_vals = combined.values - mean_t[:, None, None]
        result["minus_basin_mean_raw"] = xr.DataArray(
            minus_vals, coords=combined.coords, dims=combined.dims
        ).rename("minus_basin_mean_raw")

    # LPFs for each field in result
    for base_name, base_da in list(result.items()):
        for lpf_key, cutoff in lowpass_cutoffs.items():
            pad = int(2 / cutoff)
            filt = lowpass_filter(base_da.values, cutoff, order=5, fs=fs, pad=pad)
            new_name = f"{lpf_key}" if base_name == "raw" else f"{base_name.replace('_raw','')}_{lpf_key}"
            result[new_name] = xr.DataArray(filt, coords=base_da.coords, dims=base_da.dims).rename(new_name)

    return result

def process_y_var(model, files, monthly, lowpass_cutoffs, fs, order=5):
    """Load MOC (CMIP5), select Atlantic basin=0 when present, interp to (lev 0–2500 by 100; rlat −30..72.5 by 2.5), LPFs."""
    y_var, basin_idx = MODEL_YVAR.get(model, DEFAULT_YVAR)
    da_list = []
    for f in files:
        try:
            with xr.open_dataset(f, use_cftime=True) as ds:
                if y_var not in ds:
                    continue
                da = ds[y_var]

                # multi-basin handling (CMIP5 often has 'basin', sometimes absent)
                if "3basin" in da.dims:
                    da = da.rename({"3basin": "basin"})
                if "basin" in da.dims:
                    da = da.isel(basin=basin_idx)

                # depth coord normalize
                if "lev" not in da.dims:
                    if "olevel" in da.dims: da = da.rename({"olevel": "lev"})
                    elif "depth" in da.dims: da = da.rename({"depth": "lev"})
                if "lev" not in da.dims:
                    raise RuntimeError("No 'lev' coordinate in y-data.")

                # convert depth units if needed
                lev_units = da.lev.attrs.get("units", "").lower()
                if lev_units in ("cm", "centimeter", "centimeters"):
                    da = da.assign_coords(lev=da.lev / 100.0)

                # latitude → 'rlat'
                for old in ("rlat", "lat", "latitude", "y"):
                    if old in da.dims:
                        if old != "rlat":
                            da = da.rename({old: "rlat"})
                        break

                # slice + interp
                da = da.sel(lev=slice(500, 2500))
                new_lat = np.arange(-30, 72.5 + 1e-6, 2.5)
                target_lev = np.arange(500, 2500 + 1e-6, 100)
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
        result[lpf_key] = xr.DataArray(filt, coords=combined.coords, dims=combined.dims)
    return result

# ───────────────────────────────────────────────────────────────────────
# Main driver
# ───────────────────────────────────────────────────────────────────────
def _pick_member_dir(model_path, prefer="r1i1p1"):
    """Return a member directory path. Prefer `prefer*` if exists, else first found."""
    if not os.path.isdir(model_path):
        return None
    # exact prefer or any member starting with prefer (e.g., r1i1p1)
    candidates = [d for d in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, d))]
    if not candidates:
        return None
    # prefer exact match
    if prefer in candidates:
        return os.path.join(model_path, prefer)
    # prefer a dir that starts with prefer (e.g., r1i1p1a)
    for d in candidates:
        if d.startswith(prefer):
            return os.path.join(model_path, d)
    # otherwise pick the first one (sorted for determinism)
    return os.path.join(model_path, sorted(candidates)[0])

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

            # ------------- resolve member dir (CMIP5: r1i1p1 etc.) -------------
            model_path = os.path.join(base_dir, model, scenario)
            member_dir = _pick_member_dir(model_path, prefer="r1i1p1")
            if member_dir is None:
                print(f"[{model}/{scenario}] no member dir")
                out[scenario] = scen
                continue

            # ------------ X ------------------------------------------------
            x_group = {}
            for var in x_vars:
                var_dir = os.path.join(member_dir, var)
                files = sorted(glob.glob(f"{var_dir}/*.nc"))
                if not files:
                    print(f"[{model}/{scenario}] no {var}")
                    continue
                data_dict = process_x_var(
                    var, files, target_grid, monthly, lowpass_cutoffs, fs, model_wts
                )
                if not data_dict:
                    continue
                for key, da in data_dict.items():
                    # key is 'raw', 'LPF24', 'minus_basin_mean_LPF24', etc.
                    da = da.assign_coords(var=var, lpf=key)
                    da = da.chunk({k: v for k, v in chunk_dict.items() if k in da.dims})
                    x_group[f"{var}_{key}"] = da

            if x_group:
                # dtauu/dy from tauu_raw, then LPFs
                if "tauu_raw" in x_group:
                    da_tauu = x_group["tauu_raw"]

                    if "lat" in da_tauu.coords and getattr(da_tauu["lat"], "ndim", 1) == 2:
                        lat1d = da_tauu["lat"].isel(lon=0)
                        da_tauu = da_tauu.assign_coords(lat=lat1d)

                    dtaudlat = da_tauu.differentiate("lat")
                    dtauu_dy_raw = (dtaudlat / (R_earth * np.deg2rad(1))).rename("dtauu_dy_raw")
                    dtauu_dy_raw = dtauu_dy_raw.chunk({d: sz for d, sz in chunk_dict.items() if d in dtauu_dy_raw.dims})
                    x_group["dtauu_dy_raw"] = dtauu_dy_raw

                    for lpf_key, cutoff in lowpass_cutoffs.items():
                        pad = int(2 / cutoff)
                        filtered_vals = lowpass_filter(dtauu_dy_raw.values, cutoff, order=5, fs=fs, pad=pad)
                        da_filt = xr.DataArray(filtered_vals, coords=dtauu_dy_raw.coords, dims=dtauu_dy_raw.dims)
                        da_filt.name = f"dtauu_dy_{lpf_key}"
                        da_filt = da_filt.chunk({d: sz for d, sz in chunk_dict.items() if d in da_filt.dims})
                        da_filt = da_filt.assign_coords(lpf=lpf_key)
                        x_group[f"dtauu_dy_{lpf_key}"] = da_filt

                scen["x"] = x_group

            # ------------ Y ------------------------------------------------
            # find a directory under member_dir that contains the right y_var
            y_var, _ = MODEL_YVAR.get(model, DEFAULT_YVAR)
            y_dir = None
            for cand in os.listdir(member_dir):
                cand_path = os.path.join(member_dir, cand)
                if not os.path.isdir(cand_path):
                    continue
                f0 = sorted(glob.glob(f"{cand_path}/*.nc"))[:1]
                if not f0:
                    continue
                with xr.open_dataset(f0[0], use_cftime=True) as test_ds:
                    if y_var in test_ds:
                        y_dir = cand_path
                        break

            if y_dir:
                y_files = sorted(glob.glob(f"{y_dir}/*.nc"))
                y_dict = process_y_var(model, y_files, monthly, lowpass_cutoffs, fs)
                if y_dict:
                    y_group = {}
                    for key, da in y_dict.items():  # key: 'raw', 'LPF24',...
                        da = da.assign_coords(lpf_y=key)
                        da = da.chunk({k: v for k, v in chunk_dict.items() if k in da.dims})
                        y_group[key] = da
                    scen["y"] = y_group
            else:
                print(f"[{model}/{scenario}] no y-var ({y_var})")

            # --------- Ekman & y_no_ekman ----------------------------------
            if ("x" in scen and "y" in scen and "tauu_raw" in scen["x"] and "raw" in scen["y"]):
                ekman_raw, y_no_ek_raw = build_ekman_terms(
                    model, scen["y"]["raw"], scen["x"]["tauu_raw"]
                )
                ekman_raw = ekman_raw.chunk({k: v for k, v in chunk_dict.items() if k in ekman_raw.dims})
                y_no_ek_raw = y_no_ek_raw.chunk({k: v for k, v in chunk_dict.items() if k in y_no_ek_raw.dims})

                scen.setdefault("ekman", {})["raw"] = ekman_raw
                scen.setdefault("y_no_ekman", {})["raw"] = y_no_ek_raw

                for lpf_key, cutoff in lowpass_cutoffs.items():
                    pad = int(2 / cutoff)
                    ek_f = xr.DataArray(
                        lowpass_filter(np.nan_to_num(ekman_raw.values), cutoff, 5, fs, pad),
                        coords=ekman_raw.coords, dims=ekman_raw.dims, name=f"ekman_{lpf_key}",
                    )
                    yn_f = xr.DataArray(
                        lowpass_filter(np.nan_to_num(y_no_ek_raw.values), cutoff, 5, fs, pad),
                        coords=y_no_ek_raw.coords, dims=y_no_ek_raw.dims, name=f"y_no_ekman_{lpf_key}",
                    )
                    scen["ekman"][lpf_key] = ek_f.chunk({k: v for k, v in chunk_dict.items() if k in ek_f.dims})
                    scen["y_no_ekman"][lpf_key] = yn_f.chunk({k: v for k, v in chunk_dict.items() if k in yn_f.dims})

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

        # --------------- write Zarr (same structure as your CMIP6 script) ---------------
        zpath = output_zarr_template.format(model=model)
        root = zarr.open_group(zpath, mode="w")

        def dump(group, d):
            for k, v in d.items():
                if isinstance(v, dict):
                    dump(group.require_group(k), v)
                elif isinstance(v, xr.DataArray):
                    # respect chunking if present
                    ch = None
                    if getattr(v, "chunks", None):
                        ch = tuple(c[0] if isinstance(c, tuple) else c for c in v.chunks)
                    ds = group.create_dataset(k, data=np.asarray(v.data), chunks=ch)
                    ds.attrs.update(v.attrs)

        dump(root, out)
        zarr.consolidate_metadata(zpath)
        print(f"[✓] {model} → {zpath}")

# ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Directory layout expected:
    # <base_dir>/<model>/<scenario>/<member>/<variable>/*.nc
    # e.g., /.../cmip5_data/ACCESS1-0/historical/r1i1p1/tos/*.nc
    base_dir = "/home/am334/link_am334/praki/cmip5_data"

    # Your CMIP5 set
    models = ["ACCESS1-0", "CanESM2", "MPI-ESM-LR", "MRI-CGCM3"]
    models = ["MPI-ESM-MR"]
    #models = ["MIROC5", "NorESM1-M"]

    # CMIP5 scenarios
    scenarios = ["piControl"]

    # X-vars to process
    x_vars = ["tos","zos"]  # extend as needed (e.g., "pbo", "hfds")

    # 1°×1° global target grid
    target_grid = xe.util.grid_global(1, 1)
    monthly = True

    if monthly:
        lowpass_cutoffs = {"LPF24": 1 / 24, "LPF120": 1 / 120}
        out_tpl = "monthly_1deg_grid_with_coasts_EK_minus/output_data_{model}.zarr"
    else:
        lowpass_cutoffs = {"LPF10": 1 / 10}
        out_tpl = "yearly_mean_data_EK_cmip5/output_data_{model}.zarr"

    fs = 1.0
    chunk_dict = {"time": 240}

    preprocess_and_save_zarr_by_model(
        base_dir, models, scenarios, x_vars, target_grid,
        monthly, lowpass_cutoffs, fs,
        out_tpl, chunk_dict,
    )
