#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stream-oriented CMIP6 pre-processor.
  • Processes *one member at a time* → O(<1 GB) RAM.
  • Immediately stores results in     <model>.zarr/<scenario>/<member>/…
  • Steps (per member):
        X-vars  → regrid, basin-mean removal, LPFs, dtauu/dy
        Y-var   → slice, interp, LPFs
        Ekman   → ekman & y_no_ekman + LPFs
  • Works for monthly OR annual means.
"""
import os, glob, warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import xarray as xr
import xesmf as xe
import regionmask
from scipy.signal import butter, sosfilt
from scipy.ndimage import binary_dilation
import zarr
import traceback, logging

logging.basicConfig(
    filename="bad_members.log",
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s: %(message)s",
)



warnings.filterwarnings("ignore")

# ───────────────────────── CONSTANTS ─────────────────────────
R_EARTH   = 6_371_000.0
RHO       = 1_025.0
OMEGA     = 7.2921e-5
WEIGHTS   = Path("regrid_weights"); WEIGHTS.mkdir(exist_ok=True)
CHUNKS    = {"time": 240}            # applied per-variable

# ──────────────────── GENERIC HELPERS ───────────────────────
def member_from_path(p: str) -> str:
    for t in Path(p).parts:
        if t.startswith("r") and "i" in t and "p" in t and "f" in t:
            return t
    return "unknown"

def regrid_da(da, tgt):


    r = xe.Regridder(
        da, tgt, "bilinear",
        periodic=True,
        extrap_method="inverse_dist",
        extrap_num_src_pnts=8,
        reuse_weights=False,
        ignore_degenerate=False,
    )

    return r(da)

def basin_mask(da):
    lon = da.lon if "lon" in da.coords else da.longitude
    lat = da.lat if "lat" in da.coords else da.latitude
    m   = regionmask.defined_regions.natural_earth_v5_0_0.\
          ocean_basins_50.mask(lon, lat)
    keep=[2,6,60,32,31,17,55]
    return da.where(m.isin(keep), drop=True)

def lowpass(data, cutoff, order=5, fs=1.0, pad=4):
    sos = butter(order, cutoff, "low", fs=fs, output="sos")
    pads=((pad,pad),)+((0,0),)*(data.ndim-1)
    d   = np.pad(data, pads, mode="reflect")
    out = sosfilt(sos, d, axis=0)[pad:-pad,...]
    return np.nan_to_num(out)


# ──────────────────── DRIFT CORRECTION HELPERS ───────────────────────
def fit_linear_trend(da: xr.DataArray) -> Tuple[xr.DataArray, xr.DataArray, int]:
    """
    Fit a linear trend along the time dimension for each spatial point.
    
    Parameters
    ----------
    da : xr.DataArray
        Input data with 'time' dimension and optional spatial dimensions.
    
    Returns
    -------
    slope : xr.DataArray
        Linear trend slope (units per time step) for each spatial point.
    intercept : xr.DataArray
        Linear trend intercept (value at t=0) for each spatial point.
    nt : int
        Length of time dimension (used for midpoint calculation).
    """
    # Convert time to numeric index (0, 1, 2, ...)
    nt = da.sizes["time"]
    t_idx = np.arange(nt, dtype=np.float64)
    
    # Get data as numpy array and reshape for vectorized polyfit
    data = da.values  # shape: (time, ...) e.g. (time, lat, lon) or (time, lev, rlat)
    orig_shape = data.shape
    spatial_shape = orig_shape[1:]
    
    # Reshape to (time, n_spatial_points)
    data_2d = data.reshape(nt, -1)
    
    # Handle NaNs by fitting only valid points
    slope_flat = np.full(data_2d.shape[1], np.nan)
    intercept_flat = np.full(data_2d.shape[1], np.nan)
    
    for i in range(data_2d.shape[1]):
        col = data_2d[:, i]
        valid = ~np.isnan(col)
        if valid.sum() > 1:
            p = np.polyfit(t_idx[valid], col[valid], 1)
            slope_flat[i] = p[0]
            intercept_flat[i] = p[1]
    
    # Reshape back to spatial dimensions
    slope_arr = slope_flat.reshape(spatial_shape)
    intercept_arr = intercept_flat.reshape(spatial_shape)
    
    # Build coordinate dict excluding 'time'
    spatial_coords = {k: v for k, v in da.coords.items() if k != "time" and k in da.dims}
    spatial_dims = [d for d in da.dims if d != "time"]
    
    slope_da = xr.DataArray(slope_arr, coords=spatial_coords, dims=spatial_dims, name="trend_slope")
    intercept_da = xr.DataArray(intercept_arr, coords=spatial_coords, dims=spatial_dims, name="trend_intercept")
    
    return slope_da, intercept_da, nt


def compute_drift(slope: xr.DataArray, time_indices: np.ndarray, 
                  picontrol_midpoint: float) -> xr.DataArray:
    """
    Compute drift values at given time indices using linear trend coefficients.
    Uses the midpoint of piControl as the zero-drift reference point.
    
    Parameters
    ----------
    slope : xr.DataArray
        Slope from fit_linear_trend (spatial dims only).
    time_indices : np.ndarray
        1D array of time indices (in piControl time reference) at which to compute drift.
    picontrol_midpoint : float
        The midpoint of the piControl time series (nt_picontrol / 2).
        Drift is zero at this point.
    
    Returns
    -------
    drift : xr.DataArray
        Drift values with shape (len(time_indices), *spatial_dims).
    """
    # drift(t) = slope * (t - t_mid)
    # At t = t_mid (piControl midpoint), drift = 0
    nt = len(time_indices)
    t_centered = time_indices - picontrol_midpoint
    t_arr = t_centered.reshape(-1, *([1] * slope.ndim))  # (nt, 1, 1, ...)
    
    drift_vals = slope.values * t_arr
    
    # Build output DataArray with time as first dimension
    dims = ("time",) + tuple(slope.dims)
    coords = dict(slope.coords)
    coords["time"] = np.arange(nt)  # placeholder time coord
    
    return xr.DataArray(drift_vals, dims=dims, coords=coords, name="drift")


def read_branch_time(nc_file: str) -> float:
    """
    Read branch_time_in_parent from NetCDF global attributes.
    
    Parameters
    ----------
    nc_file : str
        Path to a NetCDF file from the scenario.
    
    Returns
    -------
    branch_time : float
        The branch time in parent units (typically days since reference).
        Returns 0.0 if attribute not found.
    """
    with xr.open_dataset(nc_file, use_cftime=True) as ds:
        # Try various attribute names used in CMIP6
        branch_time = ds.attrs.get("branch_time_in_parent", None)
        if branch_time is None:
            branch_time = ds.attrs.get("branch_time", None)
        if branch_time is None:
            branch_time = 0.0
        return float(branch_time)


def convert_branch_time_to_index(branch_time: float, parent_time_units: str,
                                  monthly: bool) -> int:
    """
    Convert branch_time (in days) to a time index for the piControl trend.
    
    Parameters
    ----------
    branch_time : float
        Branch time in days since piControl reference.
    parent_time_units : str
        Time units string from piControl (e.g., "days since 0001-01-01").
    monthly : bool
        If True, data is monthly; if False, annual.
    
    Returns
    -------
    index : int
        Time index in the piControl trend arrays.
    """
    # branch_time is typically in days
    # Convert to months or years depending on monthly flag
    if monthly:
        # Approximate: 30.4375 days per month
        index = int(branch_time / 30.4375)
    else:
        # Approximate: 365.25 days per year
        index = int(branch_time / 365.25)
    return index


def subtract_drift_from_data(da: xr.DataArray, slope: xr.DataArray, 
                              picontrol_midpoint: float,
                              branch_time_idx: int = 0) -> xr.DataArray:
    """
    Subtract piControl drift from data using consistent midpoint-centered approach.
    
    For piControl: drift(t) = slope × (t - M)
    For historical/SSP: drift(t) = slope × (t + branch_time - M)
    
    This ensures that detrended values match at the branch point!
    
    Parameters
    ----------
    da : xr.DataArray
        Data to detrend, with time dimension.
    slope : xr.DataArray
        Slope from piControl trend fit.
    picontrol_midpoint : float
        The midpoint of the piControl time series (M = nt_picontrol / 2).
    branch_time_idx : int
        Time index in piControl when this scenario branches.
        For piControl itself, use 0.
    
    Returns
    -------
    detrended : xr.DataArray
        Data with drift subtracted.
    """
    nt = da.sizes["time"]
    
    # Time indices in piControl reference frame:
    # t_picontrol = t_scenario + branch_time_idx
    # 
    # For piControl (branch_time_idx=0): t_picontrol = 0, 1, 2, ...
    # For historical: t_picontrol = branch_time, branch_time+1, branch_time+2, ...
    t_indices = np.arange(nt, dtype=np.float64) + branch_time_idx
    
    # Compute drift using the same formula for all scenarios:
    # drift(t) = slope × (t_picontrol - midpoint)
    #          = slope × (t_scenario + branch_time - midpoint)
    drift = compute_drift(slope, t_indices, picontrol_midpoint)
    
    # Align drift coordinates with data
    drift = drift.assign_coords(time=da.time.values)
    
    # Subtract drift
    detrended = da - drift
    
    return detrended


def compute_ekman_transport(tau_x: xr.DataArray) -> xr.DataArray:
    lat = tau_x["lat"] if "lat" in tau_x.coords else tau_x["latitude"]
    f   = 2 * OMEGA * np.sin(np.deg2rad(lat))
    return (-tau_x / (RHO * f)).rename("U_ek")    # m² s‑1


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

    #print(ekman[lon_dim].values)
    #print(ekman[lat_dim].values)


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


def build_ekman_terms(
    model,
    y_da,            # (time, lev, rlat)
    tauu_da,         # (time, lat, lon)
    rho0=RHO,
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



def save_da(root: zarr.hierarchy.Group, path: str, da: xr.DataArray):
    """
    Store DataArray at Zarr path <group>/<name>, re-chunked per CHUNKS.
    Works whether `da` is backed by NumPy *or* Dask.
    """
    grp_path, name = os.path.split(path)
    g = root.require_group(grp_path) if grp_path else root

    # 1) make sure we have desired chunking
    da = da.chunk({k: v for k, v in CHUNKS.items() if k in da.dims})

    # 2) final data buffer – compute() if it's Dask
    data = da.data
    if hasattr(data, "compute"):          # Dask → NumPy
        data = data.compute()

    # 3) build chunk tuple for Zarr creation
    if hasattr(data, "chunks") and data.chunks is not None:
        z_chunks = tuple(c[0] for c in data.chunks)
    else:
        z_chunks = tuple(CHUNKS.get(d, -1) if d in CHUNKS else -1
                         for d in da.dims)

    z = g.create_dataset(
        name,
        shape=da.shape,
        dtype=data.dtype,
        chunks=z_chunks,
        overwrite=True,
    )

    z[:] = data                           # now a real NumPy array
    z.attrs.update(da.attrs)


def rechunk_except(da: xr.DataArray,
                   chunk_dict: Dict[str, int]) -> xr.DataArray:
    """
    Re-chunk `da` according to `chunk_dict`, but *skip* the 'member'
    dimension (keep it as a single chunk).
    """
    desired = {dim: sz for dim, sz in chunk_dict.items()
               if dim in da.dims and dim != "member"}
    return da.chunk(desired)


# ─────────────── PER-MEMBER PROCESSORS ────────────────
def process_x_member(var:str, nc_files:List[str], tgt_grid,
                     monthly:bool, lpf:Dict[str,float]) -> Dict[str,xr.DataArray]:
    """Return dict {subname: DataArray} for a single member."""
    das=[]
    for f in nc_files:
        with xr.open_dataset(f, use_cftime=True) as ds:
            if var not in ds: continue
            da=regrid_da(ds[var], tgt_grid)
            da=basin_mask(da)
            da=da.rename({d:"lat" for d in da.dims if d in ("rlat","y","latitude")})
            da=da.rename({d:"lon" for d in da.dims if d in ("rlon","x","longitude")})
            das.append(da)
    if not das: return {}
    combined=xr.concat(das,"time")
    if not monthly: combined=combined.resample(time="1Y").mean()



    '''
    out={"raw":comb}
    if var in ("zos","pbo","tos"):
        minus=(comb-comb.mean(("lat","lon"))).rename("minus_basin_mean")
        out[minus.name]=minus

    for nm,da in list(out.items()):
        for tag,cut in lpf.items():
            lp=lowpass(da.values,cut,5,1.0,pad=int(2/cut))
            out[f"{nm}_{tag}"]=xr.DataArray(lp, coords=da.coords, dims=da.dims)
    '''

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
        for lpf_key, cutoff in lpf.items():
            pad = int(2 / cutoff)
            filt = lowpass(base_da.values, cutoff, order=5, fs=1.0, pad=pad)
            if base_name == "minus_basin_mean_raw":
                new_name = f"minus_basin_mean_{lpf_key}"
            else:
                new_name = f"{lpf_key}"
            result[new_name] = xr.DataArray(
                filt, coords=base_da.coords, dims=base_da.dims
            ).rename(new_name)

    return result








def process_y_member(model:str, yvar:str, files:List[str],
                     monthly:bool, lpf:Dict[str,float]) -> Dict[str,xr.DataArray]:
    ds_list=[]
    for f in files:
        with xr.open_dataset(f, use_cftime=True) as ds:
            if yvar not in ds: continue
            da=ds[yvar]
            if "3basin" in da.dims: da=da.rename({"3basin":"basin"})
            if "basin" in da.dims:
                idx=1 if model in [
                    "MPI-ESM1-2-HR","CNRM-CM6-1-HR","EC-Earth3",
                    "IPSL-CM6A-MR1","IPSL-CM6A-LR","E3SM-1-0", "CAS-ESM2-0"] else 0
                da=da.isel(basin=idx)
            if "depth" in da.dims and "lev" not in da.dims:
                da=da.rename({"depth":"lev"})

            if "olevel" in da.dims and "lev" not in da.dims:
                                da = da.rename({"olevel": "lev"})

                        
            lev_units = da.lev.attrs.get("units", "").lower()
            if lev_units in ("cm", "centimeter", "centimeters"):
                da = da.assign_coords(lev=da.lev / 100)

            if model == "CAS-ESM2-0":
                da = da.where(np.abs(da) < 1e30)


            if model == "IPSL-CM6A-LR":
                    da = da[:,:,:,0]
            
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


                            
            da=da.sel(lev=slice(500,2500)).interp(
                rlat=np.arange(-30,72.5,2.5)).interp(
                lev=np.arange(500,2501,100)).fillna(0)
            ds_list.append(da)
    if not ds_list: return {}
    comb=xr.concat(ds_list,"time")
    if not monthly: comb=comb.resample(time="1Y").mean()

    out={"raw":comb}
    for tag,cut in lpf.items():
        out[tag]=xr.DataArray(lowpass(comb.values,cut,5,1.0,int(2/cut)),
                              coords=comb.coords,dims=comb.dims)
    return out




# ───────────────────── STREAM PRE-PROCESS ──────────────────────
def stream_preprocess(
    base_dir: str,
    models: list[str],
    scenarios: list[str],
    x_vars: list[str],
    tgt_grid,
    monthly: bool,
    lpf: dict[str, float],
    out_tpl: str,
):
    for model in models:
        root = zarr.open_group(out_tpl.format(model=model), mode="a")
        print(f"\n### {model} ###")

        # Dictionary to store piControl trend coefficients per member
        # Structure: {mem_id: {"x": {var: (slope, intercept)}, "y": (slope, intercept)}}
        picontrol_trends: Dict[str, Dict] = {}

        # ===== STEP 1: Process piControl FIRST to get trend coefficients =====
        if "piControl" in scenarios:
            scen = "piControl"
            scen_path = Path(base_dir) / model / scen
            if scen_path.is_dir():
                print("• piControl (extracting drift trends)")

                for mem_dir in sorted([d for d in scen_path.iterdir() if d.is_dir()]):
                    mem_id = member_from_path(str(mem_dir))
                    print("  –", mem_id)

                    try:
                        trends = process_picontrol_member(
                            mem_dir, mem_id, model, scen,
                            x_vars, tgt_grid, monthly, lpf, root
                        )
                        picontrol_trends[mem_id] = trends
                    except Exception as err:
                        logging.warning(
                            f"[SKIP-MEMBER] {model}/{scen}/{mem_id}: {err}"
                        )
                        traceback.print_exc()
                        continue

        # ===== STEP 2: Process other scenarios with drift correction =====
        other_scenarios = [s for s in scenarios if s != "piControl"]
        for scen in other_scenarios:
            scen_path = Path(base_dir) / model / scen
            if not scen_path.is_dir():
                continue
            print("•", scen)

            for mem_dir in sorted([d for d in scen_path.iterdir() if d.is_dir()]):
                mem_id = member_from_path(str(mem_dir))
                print("  –", mem_id)

                # Get piControl trends for this member (if available)
                mem_trends = picontrol_trends.get(mem_id, None)
                if mem_trends is None:
                    logging.warning(
                        f"[DRIFT] No piControl trends for {model}/{scen}/{mem_id}, skipping drift correction"
                    )

                # Read branch_time_in_parent from this scenario's files
                branch_time_idx = 0
                try:
                    # Find first NC file to read branch time
                    for var in x_vars:
                        var_files = sorted((mem_dir / var).glob("*.nc"))
                        if var_files:
                            branch_time = read_branch_time(str(var_files[0]))
                            branch_time_idx = convert_branch_time_to_index(
                                branch_time, "", monthly
                            )
                            print(f"    branch_time_in_parent: {branch_time} → index {branch_time_idx}")
                            break
                except Exception as e:
                    logging.warning(f"[BRANCH-TIME] Could not read branch time for {model}/{scen}/{mem_id}: {e}")

                # ---- guard the whole member block so one crash ≠ stop run
                try:
                    process_one_member(
                        mem_dir, mem_id, model, scen,
                        x_vars, tgt_grid, monthly, lpf, root,
                        picontrol_trends=mem_trends,
                        branch_time_idx=branch_time_idx
                    )
                except Exception as err:
                    logging.warning(
                        f"[SKIP-MEMBER] {model}/{scen}/{mem_id}: {err}"
                    )
                    traceback.print_exc()
                    continue   # go on with next member

        zarr.consolidate_metadata(root.store)
        print("✓ done", root.store.path)


# ─────────────── piControl member processing ────────────────
def process_picontrol_member(
    mem_dir: Path,
    mem_id: str,
    model: str,
    scen: str,
    x_vars: list[str],
    tgt_grid,
    monthly: bool,
    lpf: dict[str, float],
    root: zarr.hierarchy.Group,
) -> Dict:
    """
    Process a piControl member: fit linear trends, subtract drift, save data.
    Uses the midpoint of piControl as the zero-drift reference point.
    
    Returns
    -------
    trends : dict
        {"x": {var: (slope_da, picontrol_midpoint), ...}, 
         "y": (slope_da, picontrol_midpoint),
         "picontrol_midpoint": float}
    """
    mem_grp = root.require_group(f"{scen}/{mem_id}")
    trends_grp = root.require_group(f"{scen}/{mem_id}/trends")
    
    trends = {"x": {}, "y": None, "picontrol_midpoint": None}
    
    tauu_raw_da = None
    y_raw_da = None
    picontrol_midpoint = None  # Will be set from first variable
    
    # ---------- X-vars -----------------------------------------------
    for var in x_vars:
        files = sorted((mem_dir / var).glob("*.nc"))
        if not files:
            continue
        
        try:
            xdict = process_x_member(var, files, tgt_grid, monthly, lpf)
        except Exception as e:
            logging.warning(f"[SKIP-VAR] {model}/{scen}/{mem_id}/{var}: {e}")
            traceback.print_exc()
            continue
        
        if "raw" not in xdict:
            continue
        
        raw_da = xdict["raw"]
        
        # Fit linear trend to raw data
        slope_da, intercept_da, nt = fit_linear_trend(raw_da)
        
        # Calculate piControl midpoint (use first variable to set it)
        if picontrol_midpoint is None:
            picontrol_midpoint = nt / 2.0
            trends["picontrol_midpoint"] = picontrol_midpoint
            print(f"    piControl midpoint: {picontrol_midpoint} (from {nt} time steps)")
        
        # Store slope and midpoint (intercept not needed with midpoint-centered approach)
        trends["x"][var] = (slope_da, picontrol_midpoint)
        
        # Save trend coefficients
        save_da(trends_grp, f"x/{var}_slope", slope_da)
        save_da(trends_grp, f"x/{var}_intercept", intercept_da)  # Still save for reference
        
        # Subtract drift from piControl data (branch_time_idx = 0 for piControl itself)
        # Drift is centered on midpoint, so detrended data will have zero mean at midpoint
        raw_detrended = subtract_drift_from_data(raw_da, slope_da, picontrol_midpoint, 0)
        xdict["raw"] = raw_detrended
        
        # Also detrend the minus_basin_mean if present
        if "minus_basin_mean_raw" in xdict:
            mbm_raw = xdict["minus_basin_mean_raw"]
            mbm_slope, mbm_intercept, _ = fit_linear_trend(mbm_raw)
            mbm_detrended = subtract_drift_from_data(mbm_raw, mbm_slope, picontrol_midpoint, 0)
            xdict["minus_basin_mean_raw"] = mbm_detrended
        
        # Re-apply LPF on detrended data
        for lpf_key, cutoff in lpf.items():
            pad = int(2 / cutoff)
            filt = lowpass(xdict["raw"].values, cutoff, order=5, fs=1.0, pad=pad)
            xdict[lpf_key] = xr.DataArray(
                filt, coords=xdict["raw"].coords, dims=xdict["raw"].dims
            ).rename(lpf_key)
            
            if "minus_basin_mean_raw" in xdict:
                mbm_filt = lowpass(xdict["minus_basin_mean_raw"].values, cutoff, order=5, fs=1.0, pad=pad)
                xdict[f"minus_basin_mean_{lpf_key}"] = xr.DataArray(
                    mbm_filt, coords=xdict["minus_basin_mean_raw"].coords, dims=xdict["minus_basin_mean_raw"].dims
                ).rename(f"minus_basin_mean_{lpf_key}")
        
        if var == "tauu":
            tauu_raw_da = xdict["raw"]
        
        for lpf_key, da in xdict.items():
            try:
                da = da.assign_coords(var=var, lpf=lpf_key)
                save_da(mem_grp, f"x/{var}_{lpf_key}", da)
            except Exception as e:
                logging.warning(f"[WRITE-FAIL] {model}/{scen}/{mem_id}/{var}_{lpf_key}: {e}")
                traceback.print_exc()
    
    # ---------- Y ----------------------------------------------------
    yvar = (
        "msftyz"
        if model in {
            "CIESM","GFDL-CM4","GFDL-ESM4","HadGEM3-GC31-LL",
            "CMCC-ESM2","CMCC-CM2-HR4","CNRM-CM6-1-HR","EC-Earth3",
            "HadGEM3-GC31-MM","IPSL-CM6A-LR","IPSL-CM6A-MR1","UKESM1-1-LL",
        }
        else "msftmz"
    )
    
    try:
        y_files = []
        for d in mem_dir.iterdir():
            if d.is_dir() and any(d.glob(f"*{yvar}*.nc")):
                y_files = sorted(d.glob("*.nc"))
                break
        if y_files:
            ydict = process_y_member(model, yvar, y_files, monthly, lpf)
            if "raw" in ydict:
                raw_y = ydict["raw"]
                
                # Fit linear trend to Y raw data
                y_slope, y_intercept, y_nt = fit_linear_trend(raw_y)
                
                # Use piControl midpoint from X vars if available, else calculate from Y
                if picontrol_midpoint is None:
                    picontrol_midpoint = y_nt / 2.0
                    trends["picontrol_midpoint"] = picontrol_midpoint
                    print(f"    piControl midpoint: {picontrol_midpoint} (from Y with {y_nt} time steps)")
                
                trends["y"] = (y_slope, picontrol_midpoint)
                
                # Save trend coefficients
                save_da(trends_grp, "y/slope", y_slope)
                save_da(trends_grp, "y/intercept", y_intercept)  # Still save for reference
                
                # Subtract drift from piControl Y data (centered on midpoint)
                raw_y_detrended = subtract_drift_from_data(raw_y, y_slope, picontrol_midpoint, 0)
                ydict["raw"] = raw_y_detrended
                y_raw_da = raw_y_detrended
                
                # Re-apply LPF on detrended Y data
                for tag, cut in lpf.items():
                    ydict[tag] = xr.DataArray(
                        lowpass(raw_y_detrended.values, cut, 5, 1.0, int(2/cut)),
                        coords=raw_y_detrended.coords, dims=raw_y_detrended.dims
                    )
            
            for sub, da in ydict.items():
                save_da(mem_grp, f"y/{sub}", da)
    except Exception as e:
        logging.warning(f"[SKIP-Y] {model}/{scen}/{mem_id}: {e}")
        traceback.print_exc()
    
    # ---------- Ekman & y_no_ekman -----------------------------------
    try:
        if tauu_raw_da is not None and y_raw_da is not None:
            tauu = tauu_raw_da
            
            rnm = {}
            if "latitude" in tauu.coords and "lat" not in tauu.coords:
                rnm["latitude"] = "lat"
            if "longitude" in tauu.coords and "lon" not in tauu.coords:
                rnm["longitude"] = "lon"
            if rnm:
                tauu = tauu.rename(rnm)
            
            if tauu["lat"].ndim == 2:
                tauu = tauu.assign_coords(
                    lat=tauu["lat"].isel(lon=0, drop=True),
                    lon=tauu["lon"].isel(lat=0, drop=True)
                )
            
            U_ek = compute_ekman_transport(tauu)
            Sv_da = zonal_vol_transport_profile(U_ek)
            vol_da = (Sv_da * 1e6).rename("vol_ek")
            mass_da = (vol_da * RHO).rename("ekman_mass")
            
            y_lat_dim = [d for d in y_raw_da.dims if "lat" in d][0]
            src_lat = mass_da.dims[-1]
            
            mass_i = mass_da.interp(
                {src_lat: y_raw_da[y_lat_dim]},
                kwargs={"fill_value": "extrapolate"}
            )
            
            ekman_3d = mass_i.broadcast_like(y_raw_da).rename("ekman")
            y_no_ekman = (y_raw_da - ekman_3d).rename("y_no_ekman")
            
            save_da(mem_grp, "ekman/raw", ekman_3d)
            save_da(mem_grp, "y_no_ekman/raw", y_no_ekman)
            
            for tag, cut in lpf.items():
                pad = int(2 / cut)
                save_da(
                    mem_grp, f"ekman/{tag}",
                    xr.DataArray(
                        lowpass(np.nan_to_num(ekman_3d.values), cut, order=5, fs=1.0, pad=pad),
                        coords=ekman_3d.coords, dims=ekman_3d.dims
                    )
                )
                save_da(
                    mem_grp, f"y_no_ekman/{tag}",
                    xr.DataArray(
                        lowpass(np.nan_to_num(y_no_ekman.values), cut, order=5, fs=1.0, pad=pad),
                        coords=y_no_ekman.coords, dims=y_no_ekman.dims
                    )
                )
        else:
            if tauu_raw_da is None:
                logging.warning(f"[EKMAN] Missing tauu_raw for {model}/{scen}/{mem_id}")
            if y_raw_da is None:
                logging.warning(f"[EKMAN] Missing y/raw for {model}/{scen}/{mem_id}")
    except Exception as e:
        logging.warning(f"[SKIP-EKMAN] {model}/{scen}/{mem_id}: {e}")
        traceback.print_exc()
    
    return trends


# ─────────────── isolate per-member logic here ────────────────
def process_one_member(
    mem_dir: Path,
    mem_id: str,
    model: str,
    scen: str,
    x_vars: list[str],
    tgt_grid,
    monthly: bool,
    lpf: dict[str, float],
    root: zarr.hierarchy.Group,
    picontrol_trends: Dict = None,
    branch_time_idx: int = 0,
):
    mem_grp = root.require_group(f"{scen}/{mem_id}")

    tauu_raw_da = None   # (time, lat, lon)
    y_raw_da    = None   # (time, lev, rlat)

    # ---------- X-vars -----------------------------------------------
    for var in x_vars:
        files = sorted((mem_dir / var).glob("*.nc"))
        if not files:
            continue

        try:
            xdict = process_x_member(var, files, tgt_grid, monthly, lpf)
        except Exception as e:
            logging.warning(f"[SKIP-VAR] {model}/{scen}/{mem_id}/{var}: {e}")
            traceback.print_exc()
            continue

        if "raw" not in xdict:
            continue
        
        # Apply drift correction if piControl trends are available
        if picontrol_trends is not None and var in picontrol_trends.get("x", {}):
            slope_da, picontrol_midpoint = picontrol_trends["x"][var]
            
            # Detrend raw data using: drift = slope × (t + branch_time - midpoint)
            # This ensures values match piControl at the branch point
            raw_detrended = subtract_drift_from_data(
                xdict["raw"], slope_da, picontrol_midpoint, branch_time_idx
            )
            xdict["raw"] = raw_detrended
            
            # Also detrend minus_basin_mean if present
            if "minus_basin_mean_raw" in xdict:
                mbm_raw = xdict["minus_basin_mean_raw"]
                # Use same trend for minus_basin_mean (approximate)
                mbm_detrended = subtract_drift_from_data(
                    mbm_raw, slope_da, picontrol_midpoint, branch_time_idx
                )
                xdict["minus_basin_mean_raw"] = mbm_detrended
            
            # Re-apply LPF on detrended data
            for lpf_key, cutoff in lpf.items():
                pad = int(2 / cutoff)
                filt = lowpass(xdict["raw"].values, cutoff, order=5, fs=1.0, pad=pad)
                xdict[lpf_key] = xr.DataArray(
                    filt, coords=xdict["raw"].coords, dims=xdict["raw"].dims
                ).rename(lpf_key)
                
                if "minus_basin_mean_raw" in xdict:
                    mbm_filt = lowpass(xdict["minus_basin_mean_raw"].values, cutoff, order=5, fs=1.0, pad=pad)
                    xdict[f"minus_basin_mean_{lpf_key}"] = xr.DataArray(
                        mbm_filt, coords=xdict["minus_basin_mean_raw"].coords, 
                        dims=xdict["minus_basin_mean_raw"].dims
                    ).rename(f"minus_basin_mean_{lpf_key}")
            
            print(f"    [DRIFT] Applied drift correction to {var}")
        
        if var == "tauu" and "raw" in xdict:
            tauu_raw_da = xdict["raw"]

        for lpf_key, da in xdict.items():
            try:
                da = da.assign_coords(var=var, lpf=lpf_key)
                save_da(mem_grp, f"x/{var}_{lpf_key}", da)
            except Exception as e:
                logging.warning(f"[WRITE-FAIL] {model}/{scen}/{mem_id}/{var}_{lpf_key}: {e}")
                traceback.print_exc()

    # ---------- Y ----------------------------------------------------
    yvar = (
        "msftyz"
        if model in {
            "CIESM","GFDL-CM4","GFDL-ESM4","HadGEM3-GC31-LL",
            "CMCC-ESM2","CMCC-CM2-HR4","CNRM-CM6-1-HR","EC-Earth3",
            "HadGEM3-GC31-MM","IPSL-CM6A-LR","IPSL-CM6A-MR1","UKESM1-1-LL",
        }
        else "msftmz"
    )

    try:
        y_files = []
        for d in mem_dir.iterdir():
            if d.is_dir() and any(d.glob(f"*{yvar}*.nc")):
                y_files = sorted(d.glob("*.nc"))
                break
        if y_files:
            ydict = process_y_member(model, yvar, y_files, monthly, lpf)
            if "raw" in ydict:
                raw_y = ydict["raw"]
                
                # Apply drift correction if piControl trends are available
                if picontrol_trends is not None and picontrol_trends.get("y") is not None:
                    y_slope, picontrol_midpoint = picontrol_trends["y"]
                    raw_y_detrended = subtract_drift_from_data(
                        raw_y, y_slope, picontrol_midpoint, branch_time_idx
                    )
                    ydict["raw"] = raw_y_detrended
                    y_raw_da = raw_y_detrended
                    
                    # Re-apply LPF on detrended Y data
                    for tag, cut in lpf.items():
                        ydict[tag] = xr.DataArray(
                            lowpass(raw_y_detrended.values, cut, 5, 1.0, int(2/cut)),
                            coords=raw_y_detrended.coords, dims=raw_y_detrended.dims
                        )
                    
                    print(f"    [DRIFT] Applied drift correction to Y ({yvar})")
                else:
                    y_raw_da = raw_y
            
            for sub, da in ydict.items():
                save_da(mem_grp, f"y/{sub}", da)
    except Exception as e:
        logging.warning(f"[SKIP-Y] {model}/{scen}/{mem_id}: {e}")
        traceback.print_exc()

    # ---------- Ekman & y_no_ekman -----------------------------------
    try:
        if tauu_raw_da is not None and y_raw_da is not None:
            tauu = tauu_raw_da

            # normalize coord names for helpers
            rnm = {}
            if "latitude" in tauu.coords and "lat" not in tauu.coords:
                rnm["latitude"] = "lat"
            if "longitude" in tauu.coords and "lon" not in tauu.coords:
                rnm["longitude"] = "lon"
            if rnm:
                tauu = tauu.rename(rnm)

            # collapse 2-D coords if any



            '''
            if "lat" in tauu.coords and getattr(tauu["lat"], "ndim", 1) == 2:
                tauu = tauu.assign_coords(lat=tauu["lat"].isel(lon=0, drop=True))


            if "lon" in tauu.coords and getattr(tauu["lon"], "ndim", 1) == 2:
                tauu = tauu.assign_coords(lon=tauu["lon"].isel(lat=0, drop=True))
            '''


            if tauu["lat"].ndim == 2:
                tauu = tauu.assign_coords(lat=tauu["lat"].isel(lon=0, drop=True), lon=tauu["lon"].isel(lat=0, drop=True))

            #print(tauu)
            #print(1/0)


            # U_ek (m^2/s) → Sv → mass (kg/s)
            U_ek  = compute_ekman_transport(tauu)

            Sv_da = zonal_vol_transport_profile(U_ek)         # (time, lat-like)

            #print(Sv_da)


            vol_da  = (Sv_da * 1e6).rename("vol_ek")
            mass_da = (vol_da * RHO).rename("ekman_mass")


            # Interp to Y latitude grid with collision-safe dim swap
            y_lat_dim = [d for d in y_raw_da.dims if "lat" in d][0]   # 'rlat'
            src_lat   = mass_da.dims[-1]                              # 'lat' or 'latitude'


            # 1) interpolate along src_lat to Y's rlat values
                        # after this call, xarray often returns dims = ('time','rlat') automatically
            mass_i = mass_da.interp({src_lat: y_raw_da[y_lat_dim]},
                                    kwargs={"fill_value": "extrapolate"})



            '''
            if y_lat_dim in mass_i.dims:
                # already good → (time, rlat)
                pass
            elif src_lat in mass_i.dims:
                # still (time, src_lat) → convert to rlat without removing index coords
                mass_i = mass_i.assign_coords({y_lat_dim: y_raw_da[y_lat_dim].values}).swap_dims({src_lat: y_lat_dim})
                if src_lat in mass_i.coords and src_lat != y_lat_dim:
                    mass_i = mass_i.drop_vars(src_lat)
            else:
                raise ValueError(
                    f"After interp, expected one of {{'{y_lat_dim}','{src_lat}'}} in dims; got {mass_i.dims}"
                )
            '''


                
            ekman_3d   = mass_i.broadcast_like(y_raw_da).rename("ekman")


            y_no_ekman = (y_raw_da - ekman_3d).rename("y_no_ekman")

            # Save raw + LPFs
            save_da(mem_grp, "ekman/raw", ekman_3d)
            save_da(mem_grp, "y_no_ekman/raw", y_no_ekman)

            for tag, cut in lpf.items():
                pad = int(2 / cut)


                #print(ekman_3d.values)

                #print(lowpass(np.nan_to_num(ekman_3d.values), cut, order=5, fs=1.0, pad=pad))

                #print(1/0)
                


                save_da(
                    mem_grp, f"ekman/{tag}",
                    xr.DataArray(lowpass(np.nan_to_num(ekman_3d.values), cut, order=5, fs=1.0, pad=pad),
                                 coords=ekman_3d.coords, dims=ekman_3d.dims)
                )

            

                #(data, cutoff, order=5, fs=1.0, pad=4):



                save_da(
                    mem_grp, f"y_no_ekman/{tag}",
                    xr.DataArray(lowpass(np.nan_to_num(y_no_ekman.values), cut, order=5, fs=1.0, pad=pad),
                                 coords=y_no_ekman.coords, dims=y_no_ekman.dims)
                )
        else:
            if tauu_raw_da is None:
                logging.warning(f"[EKMAN] Missing tauu_raw for {model}/{scen}/{mem_id}")
            if y_raw_da is None:
                logging.warning(f"[EKMAN] Missing y/raw for {model}/{scen}/{mem_id}")
    except Exception as e:
        logging.warning(f"[SKIP-EKMAN] {model}/{scen}/{mem_id}: {e}")
        traceback.print_exc()


# ───────────────────────── ENTRYPOINT ────────────────────────
if __name__=="__main__":
    base = "/home/am334/link_am334/praki/cmip6_data"
    models=["CESM2"]
    scens =["historical", "ssp126", "ssp245", "ssp585", "piControl"]
    xvars =[ "tos", "zos", "tauu"]

    models   = [  # same model list as elsewhere in the project
    "ACCESS-ESM1-5",
    "ACCESS-CM2", "CESM2",
    "GFDL-ESM4","FGOALS-g3",
    "MRI-ESM2-0","MIROC6","CanESM5","GISS-E2-1-G",
    "NorESM2-LM","NorESM2-MM", 
    "HadGEM3-GC31-LL","UKESM1-1-LL","CMCC-ESM2","HadGEM3-GC31-MM",
    "MPI-ESM1-2-HR","INM-CM4-8","CanESM5-CanOE","GFDL-CM4", "IPSL-CM6A-LR", "CAS-ESM2-0"
    ]

    #models   = ["ACCESS-CM2"]  # subset for quick tests

    #models = ["ACCESS-ESM1-5", "FGOALS-g3", "CanESM5", "MRI-ESM2-0", "NorESM2-MM", "UKESM1-1-LL"]


    grid = xe.util.grid_global(1,1)
    monthly=True
    LPF = {"LPF24":1/24,"LPF120":1/120}
    out_tpl="monthly_stream_zarr_detrended/output_{model}.zarr"

    stream_preprocess(base,models,scens,xvars,grid,monthly,LPF,out_tpl)
