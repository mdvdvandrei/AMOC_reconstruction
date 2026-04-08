#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stream-oriented CMIP5 pre-processor with a synthetic scenario:
  • "historical+rcp85_to2015" = historical (≤2005-12) + rcp85 (2006-01..2015-12).
  • Processes one member at a time → O(<1 GB) RAM.
  • Saves under <model>.zarr/<scenario>/<member>/… (scenario includes synthetic one).
"""
import os, warnings, traceback, logging, itertools
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import xarray as xr
import xesmf as xe
import regionmask
from scipy.signal import butter, sosfilt
import zarr

logging.basicConfig(
    filename="bad_members_cmip5.log",
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s: %(message)s",
)
warnings.filterwarnings("ignore")

# ───────────────────────── CONSTANTS ─────────────────────────
R_EARTH   = 6_371_000.0
RHO       = 1_025.0
OMEGA     = 7.2921e-5
WEIGHTS   = Path("regrid_weights"); WEIGHTS.mkdir(exist_ok=True)
CHUNKS    = {"time": 240}  # applied per-variable
SYN_SCEN_NAME = "historical+rcp85_to2015"
SYN_CUT_YEAR  = 2015  # inclusive

# ──────────────────── MODEL↔YVAR/ BASIN ─────────────────────
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
DEFAULT_YVAR = ("msftmyz", 0)

# ──────────────────── GENERIC HELPERS ───────────────────────
def member_from_path(p: str) -> str:
    for t in Path(p).parts:
        if t.startswith("r") and "i" in t and "p" in t:
            return t
    return "unknown"

def regrid_da(da, tgt):
    try:
        r = xe.Regridder(
            da, tgt, "bilinear",
            periodic=True,
            extrap_method="inverse_dist",
            extrap_num_src_pnts=8,
            reuse_weights=False,
            ignore_degenerate=False,
        )
        return r(da)
    except Exception:
        cand = da
        rn = {}
        if "latitude" in cand.coords and "lat" not in cand.coords: rn["latitude"] = "lat"
        if "longitude" in cand.coords and "lon" not in cand.coords: rn["longitude"] = "lon"
        if rn: cand = cand.rename(rn)
        r = xe.Regridder(
            cand, tgt, "bilinear",
            periodic=True,
            extrap_method="inverse_dist",
            extrap_num_src_pnts=8,
            reuse_weights=False,
            ignore_degenerate=False,
        )
        return r(cand)

def basin_mask(da):
    lon = da.lon if "lon" in da.coords else da.longitude
    lat = da.lat if "lat" in da.coords else da.latitude
    m   = regionmask.defined_regions.natural_earth_v5_0_0.ocean_basins_50.mask(lon, lat)
    keep=[2,6,60,32,31,17,55]  # Atlantic + relevant sub-basins
    return da.where(m.isin(keep), drop=True)

def lowpass(data, cutoff, order=5, fs=1.0, pad=4):
    sos = butter(order, cutoff, "low", fs=fs, output="sos")
    pads=((pad,pad),)+((0,0),)*(data.ndim-1)
    d   = np.pad(np.nan_to_num(data), pads, mode="reflect")
    out = sosfilt(sos, d, axis=0)[pad:-pad, ...]
    return np.nan_to_num(out)

def compute_ekman_transport(tau_x: xr.DataArray) -> xr.DataArray:
    lat = tau_x["lat"] if "lat" in tau_x.coords else tau_x["latitude"]
    f   = 2 * OMEGA * np.sin(np.deg2rad(lat))
    return (-tau_x / (RHO * f)).rename("U_ek")  # m²/s

def zonal_vol_transport_profile(ekman: xr.DataArray) -> xr.DataArray:
    lat_dim, lon_dim = ekman.dims[-2], ekman.dims[-1]
    nlat, nlon       = ekman.sizes[lat_dim], ekman.sizes[lon_dim]

    if ekman[lon_dim].ndim == 1:
        lon_vec = ekman[lon_dim].values[:nlon]
        lon_2d  = np.tile(lon_vec, (nlat, 1))
    else:
        lon_2d  = ekman[lon_dim].values[:nlat, :nlon]
    dlam_2d = np.gradient(np.deg2rad(lon_2d), axis=-1)

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
            .sum(dim=lon_dim, skipna=True)  # m³/s
    Sv = (vol / 1e6).rename("Sv_ek")
    if ekman[lat_dim].ndim == 2:
        lat_1d = ekman[lat_dim].values[:nlat, :nlon].mean(axis=1)
        Sv = Sv.assign_coords({lat_dim: lat_1d})
    return Sv

def save_da(root: zarr.hierarchy.Group, path: str, da: xr.DataArray):
    grp_path, name = os.path.split(path)
    g = root.require_group(grp_path) if grp_path else root
    da = da.chunk({k: v for k, v in CHUNKS.items() if k in da.dims})
    data = da.data
    if hasattr(data, "compute"):
        data = data.compute()
    if hasattr(data, "chunks") and data.chunks is not None:
        z_chunks = tuple(c[0] for c in data.chunks)
    else:
        z_chunks = tuple(CHUNKS.get(d, -1) if d in CHUNKS else -1 for d in da.dims)
    z = g.create_dataset(name, shape=da.shape, dtype=data.dtype, chunks=z_chunks, overwrite=True)
    z[:] = data
    z.attrs.update(da.attrs)

# ─────────────── FILE COLLECTION (SCENARIOS & SYNTHETIC) ───────────────
def _collect_var_files(mem_root: Path, var: str) -> List[Path]:
    d = mem_root / var
    return sorted(d.glob("*.nc")) if d.is_dir() else []

def _time_slice_to_year(ds_or_da: xr.Dataset | xr.DataArray, year: int):
    """
    Keep all samples with time.year <= `year` in a calendar-agnostic way.
    - Sorts by time
    - Drops duplicate timestamps (keeps first)
    - Uses boolean masking (not .sel by labels), so uniqueness not required
    """
    if "time" not in ds_or_da.coords:
        return ds_or_da

    obj = ds_or_da
    try:
        obj = obj.sortby("time")
    except Exception:
        pass

    # Drop duplicate time labels (keep first)
    t = obj["time"]
    try:
        _, keep_idx = np.unique(t.values, return_index=True)
        if len(keep_idx) != obj.sizes["time"]:
            obj = obj.isel(time=np.sort(keep_idx))
            t = obj["time"]
    except Exception:
        # Fallback: if unique() fails for any reason, continue without de-dup
        pass

    # Calendar-agnostic trim via boolean mask
    mask = t.dt.year <= year
    # If nothing matches, return an empty slice along time
    if mask.sum().item() == 0:
        return obj.isel(time=slice(0, 0))

    return obj.where(mask, drop=True)

def _concat_time_safe(dalist: List[xr.DataArray]) -> xr.DataArray:
    if not dalist: 
        raise ValueError("No DataArrays to concatenate.")
    # Repair unsorted/duplicated times if any
    dalist = [da.sortby("time") for da in dalist if "time" in da.coords]
    out = xr.concat(dalist, dim="time")
    # Drop duplicate timestamps (keep first)
    _, idx = np.unique(out["time"].values, return_index=True)
    if len(idx) != out.sizes["time"]:
        out = out.isel(time=np.sort(idx))
    return out

def _pair_members(hist_model_dir: Path, rcp_model_dir: Path) -> List[str]:
    """Intersection of member IDs present in both historical and rcp85."""
    def members_at(path: Path) -> set[str]:
        out=set()
        if not path.is_dir(): return out
        for d in path.iterdir():
            if d.is_dir():
                out.add(member_from_path(str(d)))
        return out
    return sorted(members_at(hist_model_dir) & members_at(rcp_model_dir))

# ─────────────── PER-MEMBER PROCESSORS ────────────────
def process_x_member(var: str, nc_files: List[str], tgt_grid, monthly: bool,
                     lpf: Dict[str,float]) -> Dict[str,xr.DataArray]:
    das=[]
    for f in nc_files:
        with xr.open_dataset(f, use_cftime=True) as ds:
            if var not in ds: 
                continue
            da = ds[var]
            rn = {}
            if "latitude" in da.coords and "lat" not in da.coords: rn["latitude"]="lat"
            if "longitude" in da.coords and "lon" not in da.coords: rn["longitude"]="lon"
            if rn: da = da.rename(rn)

            da_rg = regrid_da(da, tgt_grid)
            da_rg = basin_mask(da_rg)
            das.append(da_rg)

    if not das:
        return {}
    combined = _concat_time_safe(das)
    if not monthly:
        combined = combined.resample(time="1Y").mean()

    result = {"raw": combined}

    if var in ("zos", "pbo", "tos"):
        mean_t = np.nanmean(combined.values, axis=(1, 2))
        minus_vals = combined.values - mean_t[:, None, None]
        minus_da = xr.DataArray(minus_vals, coords=combined.coords, dims=combined.dims)\
                     .rename("minus_basin_mean_raw")
        result[minus_da.name] = minus_da

    for base_name, base_da in list(result.items()):
        for lpf_key, cutoff in lpf.items():
            pad = int(2 / cutoff)
            filt = lowpass(base_da.values, cutoff, order=5, fs=1.0, pad=pad)
            new_name = f"{lpf_key}" if base_name == "raw" else f"{base_name.replace('_raw','')}_{lpf_key}"
            result[new_name] = xr.DataArray(filt, coords=base_da.coords, dims=base_da.dims).rename(new_name)

    return result

def process_y_member(model: str, files: List[str], monthly: bool,
                     lpf: Dict[str,float]) -> Dict[str,xr.DataArray]:
    yvar, basin_idx = MODEL_YVAR.get(model, DEFAULT_YVAR)
    ds_list=[]
    for f in files:
        with xr.open_dataset(f, use_cftime=True) as ds:
            if yvar not in ds: 
                continue
            da = ds[yvar]

            if "basin" in da.dims:
                da = da.isel(basin=basin_idx)

            if "lev" not in da.dims:
                if "olevel" in da.dims: da = da.rename({"olevel":"lev"})
                elif "depth" in da.dims: da = da.rename({"depth":"lev"})
            lev_units = da.lev.attrs.get("units", "").lower() if "lev" in da.coords else ""
            if lev_units in ("cm", "centimeter", "centimeters"):
                da = da.assign_coords(lev=da.lev / 100.0)

            for old in ("rlat","lat","latitude","y"):
                if old in da.dims:
                    if old != "rlat":
                        da = da.rename({old:"rlat"})
                    break

            da = da.sel(lev=slice(0, 2500))\
                   .interp(rlat=np.arange(-30, 72.5+1e-6, 2.5))\
                   .interp(lev=np.arange(0, 2500+1e-6, 100))\
                   .fillna(0)
            ds_list.append(da)

    if not ds_list:
        return {}
    comb = _concat_time_safe(ds_list)
    if not monthly:
        comb = comb.resample(time="1Y").mean()

    out = {"raw": comb}
    for tag, cut in lpf.items():
        pad = int(2 / cut)
        out[tag] = xr.DataArray(
            lowpass(comb.values, cut, 5, 1.0, pad),
            coords=comb.coords, dims=comb.dims
        ).rename(tag)
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
    enable_synthetic_hist_rcp85: bool = True,
    syn_cut_year: int = SYN_CUT_YEAR,
):
    base_dir = Path(base_dir)

    for model in models:
        root = zarr.open_group(out_tpl.format(model=model), mode="a")
        print(f"\n### {model} ###")

        # 1) regular scenarios
        for scen in scenarios:
            scen_path = base_dir / model / scen
            if not scen_path.is_dir():
                continue
            print("•", scen)

            for mem_dir in sorted([d for d in scen_path.iterdir() if d.is_dir()]):
                mem_id = member_from_path(str(mem_dir))
                print("  –", mem_id)
                try:
                    process_one_member_regular(
                        mem_dir, mem_id, model, scen,
                        x_vars, tgt_grid, monthly, lpf, root
                    )
                except Exception as err:
                    logging.warning(f"[SKIP-MEMBER] {model}/{scen}/{mem_id}: {err}")
                    traceback.print_exc()
                    continue

        # 2) synthetic: historical + rcp85 (to syn_cut_year)
        if enable_synthetic_hist_rcp85:
            hist_dir = base_dir / model / "historical"
            rcp_dir  = base_dir / model / "rcp85"
            if hist_dir.is_dir() and rcp_dir.is_dir():
                print(f"• {SYN_SCEN_NAME}")
                for mem_id in _pair_members(hist_dir, rcp_dir):
                    print("  –", mem_id)
                    try:
                        process_one_member_synthetic_hist_rcp85(
                            hist_mem_dir=hist_dir / mem_id,
                            rcp_mem_dir=rcp_dir / mem_id,
                            mem_id=mem_id,
                            model=model,
                            scen=SYN_SCEN_NAME,
                            x_vars=x_vars,
                            tgt_grid=tgt_grid,
                            monthly=monthly,
                            lpf=lpf,
                            root=root,
                            cut_year=syn_cut_year,
                        )
                    except Exception as err:
                        logging.warning(f"[SKIP-MEMBER] {model}/{SYN_SCEN_NAME}/{mem_id}: {err}")
                        traceback.print_exc()
                        continue

        zarr.consolidate_metadata(root.store)
        print("✓ done", root.store.path)

def _concat_hist_rcp_files(hist_files: List[Path], rcp_files: List[Path], cut_year: int) -> List[Path]:
    """Just returns both lists; slicing by time happens after opening."""
    return hist_files + rcp_files

def process_one_member_regular(
    mem_dir: Path,
    mem_id: str,
    model: str,
    scen: str,
    x_vars: list[str],
    tgt_grid,
    monthly: bool,
    lpf: dict[str, float],
    root: zarr.hierarchy.Group,
):
    mem_grp = root.require_group(f"{scen}/{mem_id}")

    tauu_raw_da = None
    y_raw_da    = None

    # ---------- X-vars -----------------------------------------------
    for var in x_vars:
        files = sorted((mem_dir / var).glob("*.nc"))
        if not files:
            continue

        try:
            xdict = process_x_member(var, [str(f) for f in files], tgt_grid, monthly, lpf)
        except Exception as e:
            logging.warning(f"[SKIP-VAR] {model}/{scen}/{mem_id}/{var}: {e}")
            traceback.print_exc()
            continue

        if var == "tauu" and "raw" in xdict:
            tauu = xdict["raw"]
            rnm = {}
            if "latitude" in tauu.coords and "lat" not in tauu.coords: rnm["latitude"]="lat"
            if "longitude" in tauu.coords and "lon" not in tauu.coords: rnm["longitude"]="lon"
            if rnm: tauu = tauu.rename(rnm)
            if "lat" in tauu.coords and getattr(tauu["lat"], "ndim", 1) == 2:
                tauu = tauu.assign_coords(lat=tauu["lat"].isel(lon=0, drop=True),
                                          lon=tauu["lon"].isel(lat=0, drop=True))
            tauu_raw_da = tauu

        for subname, da in xdict.items():
            try:
                save_da(mem_grp, f"x/{var}_{subname}", da)
            except Exception as e:
                logging.warning(f"[WRITE-FAIL] {model}/{scen}/{mem_id}/{var}_{subname}: {e}")
                traceback.print_exc()

    # ---------- Y ----------------------------------------------------
    try:
        yvar, _ = MODEL_YVAR.get(model, DEFAULT_YVAR)
        y_files = []
        for d in mem_dir.iterdir():
            if d.is_dir() and any(yvar in p.name for p in d.glob("*.nc")):
                y_files = sorted(d.glob("*.nc"))
                break
        if y_files:
            ydict = process_y_member(model, [str(p) for p in y_files], monthly, lpf)
            if "raw" in ydict: y_raw_da = ydict["raw"]
            for sub, da in ydict.items():
                save_da(mem_grp, f"y/{sub}", da)
    except Exception as e:
        logging.warning(f"[SKIP-Y] {model}/{scen}/{mem_id}: {e}")
        traceback.print_exc()

    # ---------- Ekman & y_no_ekman -----------------------------------
    try:
        if tauu_raw_da is not None and y_raw_da is not None:
            U_ek  = compute_ekman_transport(tauu_raw_da)
            Sv_da = zonal_vol_transport_profile(U_ek)
            vol_da  = (Sv_da * 1e6).rename("vol_ek")
            mass_da = (vol_da * RHO).rename("ekman_mass")

            y_lat_dim = [d for d in y_raw_da.dims if "lat" in d or "rlat" in d][0]
            src_lat   = mass_da.dims[-1]
            mass_i = mass_da.interp({src_lat: y_raw_da[y_lat_dim]},
                                    kwargs={"fill_value": "extrapolate"})
            ekman_3d   = mass_i.broadcast_like(y_raw_da).rename("ekman")
            y_no_ekman = (y_raw_da - ekman_3d).rename("y_no_ekman")

            save_da(mem_grp, "ekman/raw", ekman_3d)
            save_da(mem_grp, "y_no_ekman/raw", y_no_ekman)

            for tag, cut in lpf.items():
                pad = int(2 / cut)
                save_da(
                    mem_grp, f"ekman/{tag}",
                    xr.DataArray(lowpass(ekman_3d.values, cut, 5, 1.0, pad),
                                 coords=ekman_3d.coords, dims=ekman_3d.dims)
                )
                save_da(
                    mem_grp, f"y_no_ekman/{tag}",
                    xr.DataArray(lowpass(y_no_ekman.values, cut, 5, 1.0, pad),
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

def process_one_member_synthetic_hist_rcp85(
    hist_mem_dir: Path,
    rcp_mem_dir: Path,
    mem_id: str,
    model: str,
    scen: str,
    x_vars: list[str],
    tgt_grid,
    monthly: bool,
    lpf: dict[str, float],
    root: zarr.hierarchy.Group,
    cut_year: int,
):
    mem_grp = root.require_group(f"{scen}/{mem_id}")

    tauu_raw_da = None
    y_raw_da    = None

    # ---------- X-vars (hist + rcp85 sliced to cut_year) ------------
    for var in x_vars:
        hist_files = _collect_var_files(hist_mem_dir, var)
        rcp_files  = _collect_var_files(rcp_mem_dir, var)
        if not hist_files and not rcp_files:
            continue

        # open & slice:
        das=[]
        for f in hist_files + rcp_files:
            with xr.open_dataset(f, use_cftime=True) as ds:
                if var not in ds: 
                    continue
                da = ds[var]
                rn = {}
                if "latitude" in da.coords and "lat" not in da.coords: rn["latitude"]="lat"
                if "longitude" in da.coords and "lon" not in da.coords: rn["longitude"]="lon"
                if rn: da = da.rename(rn)
                da = _time_slice_to_year(da, cut_year)
                if da.sizes.get("time", 0) == 0: 
                    continue
                da_rg = regrid_da(da, tgt_grid)
                da_rg = basin_mask(da_rg)
                das.append(da_rg)

        if not das:
            continue

        combined = _concat_time_safe(das)
        if not monthly:
            combined = combined.resample(time="1Y").mean()

        result = {"raw": combined}

        if var in ("zos", "pbo", "tos"):
            mean_t = np.nanmean(combined.values, axis=(1, 2))
            minus_vals = combined.values - mean_t[:, None, None]
            minus_da = xr.DataArray(minus_vals, coords=combined.coords, dims=combined.dims)\
                         .rename("minus_basin_mean_raw")
            result[minus_da.name] = minus_da

        for base_name, base_da in list(result.items()):
            for lpf_key, cutoff in lpf.items():
                pad = int(2 / cutoff)
                filt = lowpass(base_da.values, cutoff, order=5, fs=1.0, pad=pad)
                new_name = f"{lpf_key}" if base_name == "raw" else f"{base_name.replace('_raw','')}_{lpf_key}"
                result[new_name] = xr.DataArray(filt, coords=base_da.coords, dims=base_da.dims).rename(new_name)

        if var == "tauu" and "raw" in result:
            tauu = result["raw"]
            if "latitude" in tauu.coords or "longitude" in tauu.coords:
                rnm = {}
                if "latitude" in tauu.coords and "lat" not in tauu.coords: rnm["latitude"]="lat"
                if "longitude" in tauu.coords and "lon" not in tauu.coords: rnm["longitude"]="lon"
                if rnm: tauu = tauu.rename(rnm)
            if "lat" in tauu.coords and getattr(tauu["lat"], "ndim", 1) == 2:
                tauu = tauu.assign_coords(lat=tauu["lat"].isel(lon=0, drop=True),
                                          lon=tauu["lon"].isel(lat=0, drop=True))
            tauu_raw_da = tauu

        for subname, da in result.items():
            try:
                save_da(mem_grp, f"x/{var}_{subname}", da)
            except Exception as e:
                logging.warning(f"[WRITE-FAIL] {model}/{scen}/{mem_id}/{var}_{subname}: {e}")
                traceback.print_exc()

    # ---------- Y (hist + rcp85 sliced) ------------------------------
    try:
        yvar, _ = MODEL_YVAR.get(model, DEFAULT_YVAR)
        def y_files_from(mem: Path) -> List[Path]:
            for d in mem.iterdir():
                if d.is_dir() and any(yvar in p.name for p in d.glob("*.nc")):
                    return sorted(d.glob("*.nc"))
            return []

        y_files = y_files_from(hist_mem_dir) + y_files_from(rcp_mem_dir)
        ds_list=[]
        for f in y_files:
            with xr.open_dataset(f, use_cftime=True) as ds:
                if yvar not in ds: 
                    continue
                da = ds[yvar]
                if "basin" in da.dims:
                    basin_idx = MODEL_YVAR.get(model, DEFAULT_YVAR)[1]
                    da = da.isel(basin=basin_idx)
                if "lev" not in da.dims:
                    if "olevel" in da.dims: da = da.rename({"olevel":"lev"})
                    elif "depth" in da.dims: da = da.rename({"depth":"lev"})
                lev_units = da.lev.attrs.get("units", "").lower() if "lev" in da.coords else ""
                if lev_units in ("cm", "centimeter", "centimeters"):
                    da = da.assign_coords(lev=da.lev / 100.0)
                for old in ("rlat","lat","latitude","y"):
                    if old in da.dims:
                        if old != "rlat":
                            da = da.rename({old:"rlat"})
                        break
                da = _time_slice_to_year(da, cut_year)
                if da.sizes.get("time", 0) == 0:
                    continue
                da = da.sel(lev=slice(500, 3000))\
                       .interp(rlat=np.arange(-30, 72.5+1e-6, 2.5))\
                       .interp(lev=np.arange(500, 3000+1e-6, 100))\
                       .fillna(0)
                ds_list.append(da)

        if ds_list:
            comb = _concat_time_safe(ds_list)
            if not monthly:
                comb = comb.resample(time="1Y").mean()
            ydict = {"raw": comb}
            for tag, cut in lpf.items():
                pad = int(2 / cut)
                ydict[tag] = xr.DataArray(
                    lowpass(comb.values, cut, 5, 1.0, pad),
                    coords=comb.coords, dims=comb.dims
                ).rename(tag)
            y_raw_da = ydict["raw"]
            for sub, da in ydict.items():
                save_da(mem_grp, f"y/{sub}", da)
    except Exception as e:
        logging.warning(f"[SKIP-Y] {model}/{scen}/{mem_id}: {e}")
        traceback.print_exc()

    # ---------- Ekman & y_no_ekman -----------------------------------
    try:
        if tauu_raw_da is not None and y_raw_da is not None:
            U_ek  = compute_ekman_transport(tauu_raw_da)
            Sv_da = zonal_vol_transport_profile(U_ek)
            vol_da  = (Sv_da * 1e6).rename("vol_ek")
            mass_da = (vol_da * RHO).rename("ekman_mass")

            y_lat_dim = [d for d in y_raw_da.dims if "lat" in d or "rlat" in d][0]
            src_lat   = mass_da.dims[-1]
            mass_i = mass_da.interp({src_lat: y_raw_da[y_lat_dim]},
                                    kwargs={"fill_value": "extrapolate"})
            ekman_3d   = mass_i.broadcast_like(y_raw_da).rename("ekman")
            y_no_ekman = (y_raw_da - ekman_3d).rename("y_no_ekman")

            save_da(mem_grp, "ekman/raw", ekman_3d)
            save_da(mem_grp, "y_no_ekman/raw", y_no_ekman)

            for tag, cut in lpf.items():
                pad = int(2 / cut)
                save_da(
                    mem_grp, f"ekman/{tag}",
                    xr.DataArray(lowpass(ekman_3d.values, cut, 5, 1.0, pad),
                                 coords=ekman_3d.coords, dims=ekman_3d.dims)
                )
                save_da(
                    mem_grp, f"y_no_ekman/{tag}",
                    xr.DataArray(lowpass(y_no_ekman.values, cut, 5, 1.0, pad),
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
if __name__ == "__main__":
    # Adjust base to your CMIP5 layout: <base>/<model>/<scenario>/<member>/<variable>/*.nc
    base = "/home/am334/link_am334/praki/cmip5_data"

    # CMIP5 models
    models = ["ACCESS1-0", "CanESM2", "MPI-ESM-LR", "MRI-CGCM3"]


    models = ["MPI-ESM-MR"]

    # Regular scenarios you still want to process
    scens  = ["piControl", "historical"]  # include rcp85 if you also want separate processing

    # Surface/forcing fields (add tauu if you want Ekman)
    xvars  = ["tos","zos"]

    # 1°×1° target grid
    grid   = xe.util.grid_global(1, 1)

    monthly = True
    LPF     = {"LPF24": 1/24, "LPF120": 1/120}

    out_tpl = "monthly_stream_zarr_cmip5/output_{model}.zarr"

    stream_preprocess(
        base_dir=base,
        models=models,
        scenarios=scens,
        x_vars=xvars,
        tgt_grid=grid,
        monthly=monthly,
        lpf=LPF,
        out_tpl=out_tpl,
        enable_synthetic_hist_rcp85=True,
        syn_cut_year=2015,
    )
