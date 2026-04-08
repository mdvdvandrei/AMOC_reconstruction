#!/usr/bin/env python3
"""
Publication figure: SSH-based AMOC reconstruction vs RAPID vs ECCOv4r4.

Compares BNN reconstruction (with low-pass filtering), RAPID array observations,
and ECCO reanalysis on a common time axis.
"""

import os
import sys
import glob
import numpy as np
import xarray as xr
import torch
import matplotlib.pyplot as plt
from matplotlib.dates import YearLocator, DateFormatter
from pathlib import Path

try:
    from omegaconf import OmegaConf
    WEIGHTS_ROOT_DIR = str(
        OmegaConf.load(Path(__file__).resolve().parent / "conf" / "config_bnn.yaml").get("weights_root", "weights")
    )
except Exception:
    WEIGHTS_ROOT_DIR = "weights"

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# Time window for the figure (satellite era)
TIME_WINDOW = (np.datetime64('1992-01-01'), np.datetime64('2024-12-31'))

# Trim (months) at each end of overlap to avoid LPF24 edge effects.
# LPF24 uses pad=48 (4 years); trimming 24 months (2 years) each end is a conservative choice.
LPF_TRIM_MONTHS = 24

# Output directory
OUT_DIR = "bnn_real_world_rec"
os.makedirs(OUT_DIR, exist_ok=True)

# Use total (aleatoric + epistemic) uncertainty or just aleatoric?
USE_TOTAL_SIGMA = False

# Device for BNN inference
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────────────────────
#  COLOR PALETTE (Nature-quality, colorblind-friendly)
# ─────────────────────────────────────────────────────────────────────────────

# Elegant, high-contrast colors
COLORS = {
    "ssh":   "#D55E00",   # Orange - SSH reconstruction (contrasts with blue SST)
    "sst":   "#2166AC",   # Deep blue - SST reconstruction
    "rapid": "black",     # Black - RAPID observations  
    "ecco":  "#1B7837",   # Forest green - ECCO reanalysis
    "grey":  "#636363",   # Reference lines
}

# ─────────────────────────────────────────────────────────────────────────────
#  UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def get_bnn_predictions(dataset, weight_pattern, in_ch, out_dim,
                        device=None, batch_size=8, exclude_models=None):
    """Run BNN inference (same as inference_obs_new.py)."""
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=2)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpts = sorted(glob.glob(weight_pattern))
    if exclude_models:
        exclude = [s.lower() for s in exclude_models]
        ckpts = [c for c in ckpts if not any(e in c.lower() for e in exclude)]
        print(f"Excluded models: {exclude_models} → {len(ckpts)} checkpoints.")
    if not ckpts:
        raise RuntimeError(f"No checkpoints match '{weight_pattern}'" +
                          (f" (after excluding {exclude_models})" if exclude_models else ""))
    print(f"Found {len(ckpts)} checkpoints.")

    all_mus, all_sigs = [], []
    for ck in ckpts:
        model = ResidualCNNHet(in_ch, out_dim).to(device)
        model.load_state_dict(torch.load(ck, map_location=device))
        model.eval()
        mus, sigs = [], []
        with torch.no_grad():
            for x in loader:
                if not isinstance(x, torch.Tensor):
                    x = torch.as_tensor(x)
                if x.ndim == 3: x = x.unsqueeze(1)
                x = x.to(device=device, dtype=torch.float32)
                mu, logvar = model(x)
                logvar = 8.0 * torch.tanh(logvar / 8.0)  # clamp
                mu_np  = mu.squeeze(-1).detach().cpu().numpy() * 1.16
                sig_np = np.exp(0.5 * logvar.squeeze(-1).detach().cpu().numpy()) * 1.16
                mus.append(mu_np); sigs.append(sig_np)
        all_mus.append(np.concatenate(mus, axis=0))
        all_sigs.append(np.concatenate(sigs, axis=0))
        print(f" • {os.path.basename(ck)} → {all_mus[-1].shape[0]} samples")

    all_mus  = np.stack(all_mus, axis=0)     # (folds, N)
    all_sigs = np.stack(all_sigs, axis=0)    # (folds, N)
    mu_mean    = all_mus.mean(axis=0)
    alea_sigma = all_sigs.mean(axis=0)
    epi_sigma  = all_mus.std(axis=0)
    total_sigma= np.sqrt(alea_sigma**2 + epi_sigma**2)
    return all_mus, all_sigs, mu_mean, alea_sigma, epi_sigma, total_sigma


# ─────────────────────────────────────────────────────────────────────────────
#  BNN MODEL & DATASET (copied from obs_datasets.py and inference logic)
# ─────────────────────────────────────────────────────────────────────────────

# Import from local modules (same as inference_obs_new.py)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import ResidualCNNHet
from obs_datasets import (
    lowpass_filter,
    PreprocessedSSHDatasetFromZarr,
    PreprocessedDataset
)
from torch.utils.data import DataLoader

# Import trend calculation functions from inference_obs_new
import importlib.util
inference_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inference_obs_new.py")
spec = importlib.util.spec_from_file_location("inference_obs_new", inference_path)
inference_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inference_module)
compute_trend_ci = inference_module.compute_trend_ci
slope_two_sigma_from_info = inference_module.slope_two_sigma_from_info
scale_vals = inference_module.scale_vals
BOOT_B = inference_module.BOOT_B


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN SCRIPT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("="*70)
    print("  NATURE-QUALITY FIGURE: SSH vs RAPID vs ECCO (2-year LP)")
    print("="*70 + "\n")
    
    # ═══════════════════════════════════════════════════════════════════════
    #  1. LOAD SSH DATA AND RUN BNN INFERENCE
    # ═══════════════════════════════════════════════════════════════════════
    print("[1/4] Loading SSH data and running BNN inference...")
    
    ssh_zarr_path = "duacs_data_6_may.zarr"
    ssh_ds = PreprocessedSSHDatasetFromZarr(
        ssh_zarr_path, variable="__xarray_dataarray_variable__", lpf="LPF24",
        order=5, fs=1, monthly=True
    )
    
    # Determine input channels
    x0_ssh = ssh_ds[0]
    if x0_ssh.ndim == 2:
        x0_ssh = x0_ssh[np.newaxis]
    in_ch_ssh = x0_ssh.shape[0]
    
    # Weight pattern for SSH
    ssh_pattern = os.path.join(
        WEIGHTS_ROOT_DIR, "bnn_test_god_bless_it_work_all_vars_all_models_test_24",
        "zos_minus_basin_mean", "*", "best_stage2_joint.pt"
    )
    
    all_mus_ssh, all_sigs_ssh, mu_ssh, alea_ssh, epi_ssh, tot_ssh = get_bnn_predictions(
        ssh_ds, ssh_pattern, in_ch_ssh, 1, device=DEVICE, batch_size=16,
        exclude_models=["inm", "cas", "fgoals"]
    )
    print(f"    ✓ SSH BNN inference complete")
    
    # Get times from the SSH zarr file
    times_ssh = xr.open_zarr(ssh_zarr_path).coords["time"].values
    
    # ═══════════════════════════════════════════════════════════════════════
    #  2. LOAD EKMAN AND ADD TO SSH
    # ═══════════════════════════════════════════════════════════════════════
    print("[2/4] Loading Ekman transport and adding to SSH...")
    
    ek_fn = "/home/am334/link_am334/ECCO_data/ekman_transport_26p5N_monthly_Sv.nc"
    ek_ds = xr.open_dataset(ek_fn)
    ek_raw = ek_ds["ekman_transport"].sortby("time")
    
    # Interpolate Ekman to SSH timestamps
    ssh_time_da = xr.DataArray(times_ssh, dims="time")
    ek_on_ssh = ek_raw.interp(time=ssh_time_da).ffill("time").bfill("time")
    ek_on_ssh_lp = lowpass_filter(ek_on_ssh.values.astype(np.float64), 1/24, pad=48)
    
    # Create mask for time window
    m_ssh = (times_ssh >= TIME_WINDOW[0]) & (times_ssh <= TIME_WINDOW[1])
    ek_center = float(np.nanmean(ek_on_ssh_lp[m_ssh]))
    
    # SSH with Ekman
    ssh_with_ek = mu_ssh + ek_on_ssh_lp - ek_center
    ssh_sigma = tot_ssh if USE_TOTAL_SIGMA else alea_ssh
    
    print("    ✓ Ekman transport added to SSH")
    
    # ═══════════════════════════════════════════════════════════════════════
    #  3. LOAD RAPID
    # ═══════════════════════════════════════════════════════════════════════
    print("[3/4] Loading RAPID observations...")
    
    ds_moc = xr.open_dataset('moc_vertical.nc')
    rapid_monthly = ds_moc['stream_function_mar'].resample(time='1M').mean()
    rapid_max = rapid_monthly.max(dim='depth')
    rapid_times = rapid_max.time.values
    rapid_vals = rapid_max.values.astype(np.float64)
    
    # LPF24 for all RAPID-related series (same filter to preserve phase for correlation)
    # cutoff=1/24 cycles/month (2-year), order=5, fs=1, pad=48 (obs_datasets.lowpass_filter)
    rapid_lp = lowpass_filter(rapid_vals, 1/24, pad=48)  # RAPID with Ekman (stream_function_mar includes Ekman)
    rapid_time_da = xr.DataArray(rapid_times, dims="time")
    ek_on_rapid = ek_raw.interp(time=rapid_time_da).ffill("time").bfill("time")
    ek_on_rapid_lp = lowpass_filter(ek_on_rapid.values.astype(np.float64), 1/24, pad=48)
    # RAPID without Ekman: subtract LPF24 Ekman from LPF24 RAPID (same filter = no phase shift)
    rapid_wo_ek_lp = rapid_lp - ek_on_rapid_lp
    
    m_rapid = (rapid_times >= TIME_WINDOW[0]) & (rapid_times <= TIME_WINDOW[1])
    
    # Offset RAPID to match SSH mean over overlap
    ssh_mean = float(np.nanmean(ssh_with_ek[m_ssh]))
    rapid_mean = float(np.nanmean(rapid_lp[m_rapid]))
    rapid_offset = ssh_mean - rapid_mean
    
    ds_moc.close()
    print("    ✓ RAPID loaded and filtered")
    
    # ═══════════════════════════════════════════════════════════════════════
    #  4. LOAD SST DATA AND RUN BNN INFERENCE
    # ═══════════════════════════════════════════════════════════════════════
    print("[4/5] Loading SST data and running BNN inference...")
    
    sst_file = "processed_HadISST_sst.nc"
    sst_ds = PreprocessedDataset(
        sst_file, variable="__xarray_dataarray_variable__",
        lpf="LPF24", order=5, fs=1, monthly=True, minus_basin_mean=False
    )
    
    x0_sst = sst_ds[0]
    if x0_sst.ndim == 2:
        x0_sst = x0_sst[np.newaxis]
    in_ch_sst = x0_sst.shape[0]
    
    # Weight pattern for SST LPF24
    sst_pattern = os.path.join(
        WEIGHTS_ROOT_DIR, "bnn_test_god_bless_it_work_all_vars_all_models_test_24",
        "tos", "*", "best_stage2_joint.pt"
    )
    
    all_mus_sst, all_sigs_sst, mu_sst, alea_sst, epi_sst, tot_sst = get_bnn_predictions(
        sst_ds, sst_pattern, in_ch_sst, 1, device=DEVICE, batch_size=16,
        exclude_models=["inm", "cas", "fgoals"]
    )
    print("    ✓ SST BNN inference complete")
    
    # Get times from SST dataset
    times_sst = xr.open_dataset(sst_file).coords["time"].values
    
    # Interpolate Ekman to SST timestamps
    sst_time_da = xr.DataArray(times_sst, dims="time")
    ek_on_sst = ek_raw.interp(time=sst_time_da).ffill("time").bfill("time")
    ek_on_sst_lp = lowpass_filter(ek_on_sst.values.astype(np.float64), 1/24, pad=48)
    
    # Create mask for time window
    m_sst = (times_sst >= TIME_WINDOW[0]) & (times_sst <= TIME_WINDOW[1])
    ek_center_sst = float(np.nanmean(ek_on_sst_lp[m_sst]))
    
    # SST with Ekman
    sst_with_ek = mu_sst + ek_on_sst_lp - ek_center_sst
    sst_sigma = tot_sst if USE_TOTAL_SIGMA else alea_sst
    
    # Offset SST to match SSH mean over overlap
    sst_mean = float(np.nanmean(sst_with_ek[m_sst]))
    sst_offset = ssh_mean - sst_mean
    sst_with_ek = sst_with_ek + sst_offset
    
    print("    ✓ SST with Ekman added and aligned")
    
    # ═══════════════════════════════════════════════════════════════════════
    #  5. LOAD ECCO
    # ═══════════════════════════════════════════════════════════════════════
    print("[5/5] Loading ECCOv4r4...")
    
    ecco_times = None
    ecco_lp = None
    ecco_offset = 0
    
    try:
        ds_ecco = xr.open_dataset("/dat1/am334/ECCO_data/amoc_residual_26N5_Sv_monthly.nc")
        ecco = ds_ecco["amoc_residual"][24:].copy()  # skip 2 years for filter spin-up
        ecco_vals = np.asarray(ecco.values).astype(np.float64).flatten()
        ecco_times = ecco.time.values
        ecco_lp = lowpass_filter(ecco_vals, 1/24, pad=48)
        
        # Center ECCO to match SSH
        m_ecco = (ecco_times >= TIME_WINDOW[0]) & (ecco_times <= TIME_WINDOW[1])
        ecco_mean = float(np.nanmean(ecco_lp[m_ecco]))
        ecco_offset = ssh_mean - ecco_mean
        ecco_lp = ecco_lp + ecco_offset
        
        ds_ecco.close()
        print("    ✓ ECCO loaded and aligned")
    except Exception as e:
        print(f"    ⚠ ECCO not available: {e}")
    
    # ═══════════════════════════════════════════════════════════════════════
    #  6. R² (NN SSH vs RAPID) and TRENDS — trimmed for LPF edge effects
    #  All series use LPF24 (cutoff=1/24, order=5, fs=1, pad=48). We trim
    #  LPF_TRIM_MONTHS at each end of the overlap to avoid filter edge effects.
    # ═══════════════════════════════════════════════════════════════════════
    print("\n[6/6] R² (NN SSH vs RAPID) and trends (trimmed for LPF)...")
    # SSH no Ekman = raw BNN output (BNN was trained on LPF24 SSH input; output is not re-filtered to avoid extra phase lag).
    ssh_no_ek = np.asarray(mu_ssh, dtype=np.float64)
    t_ref = np.datetime64("1970-01-01")
    t_ssh_num = (np.asarray(times_ssh, dtype="datetime64[D]") - t_ref).astype(np.float64)
    t_rapid_num = (np.asarray(rapid_times, dtype="datetime64[D]") - t_ref).astype(np.float64)
    # Trim overlap: drop LPF_TRIM_MONTHS at each end (use days so datetime64[ns] + timedelta is valid)
    rapid_in_window = rapid_times[m_rapid]
    trim_days = int(round(LPF_TRIM_MONTHS * 365.25 / 12))
    if len(rapid_in_window) >= 2 * LPF_TRIM_MONTHS:
        trimmed_start = rapid_in_window[0] + np.timedelta64(trim_days, "D")
        trimmed_end   = rapid_in_window[-1] - np.timedelta64(trim_days, "D")
    else:
        trimmed_start = rapid_in_window[0]
        trimmed_end   = rapid_in_window[-1]
    m_trimmed = m_rapid & (rapid_times >= trimmed_start) & (rapid_times <= trimmed_end)
    common_ok = m_trimmed
    n_trimmed = int(m_trimmed.sum())
    print(f"    Overlap trimmed by {LPF_TRIM_MONTHS} months each end → {trimmed_start} to {trimmed_end}  (n = {n_trimmed})")
    t_common = t_rapid_num[common_ok]
    rapid_with_ek_common = rapid_lp[common_ok]
    rapid_wo_ek_common = rapid_wo_ek_lp[common_ok]
    ssh_no_ek_common = np.interp(t_common, t_ssh_num, np.nan_to_num(ssh_no_ek, nan=np.nanmean(ssh_no_ek)))
    ssh_with_ek_common = np.interp(t_common, t_ssh_num, np.nan_to_num(ssh_with_ek, nan=np.nanmean(ssh_with_ek)))
    ek_common = np.interp(t_common, t_rapid_num, ek_on_rapid_lp)
    valid = np.isfinite(rapid_with_ek_common) & np.isfinite(rapid_wo_ek_common) & np.isfinite(ssh_no_ek_common) & np.isfinite(ssh_with_ek_common)
    if valid.sum() >= 10:
        r_no = np.corrcoef(ssh_no_ek_common[valid], rapid_wo_ek_common[valid])[0, 1]
        r_with = np.corrcoef(ssh_with_ek_common[valid], rapid_with_ek_common[valid])[0, 1]
        r2_no = r_no ** 2
        r2_with = r_with ** 2
        print(f"    NN(SSH) vs RAPID — WITHOUT winds (LPF24):  R² = {r2_no:.4f}  (r = {r_no:.4f})  n = {valid.sum()}")
        print(f"    NN(SSH) vs RAPID — WITH winds (LPF24):    R² = {r2_with:.4f}  (r = {r_with:.4f})  n = {valid.sum()}")
        # Trends and R² both use TRIMMED overlap (2 years off each end)
        t_full = t_common
        ssh_no_full = ssh_no_ek_common[valid]
        ssh_with_full = ssh_with_ek_common[valid]
        rapid_full = rapid_with_ek_common[valid]
        ek_full = ek_common[valid]
        valid_full = np.isfinite(ssh_no_full) & np.isfinite(ssh_with_full) & np.isfinite(rapid_full) & np.isfinite(ek_full)
        n_full = int(valid_full.sum())
        factor_century = 12.0 * 100.0  # slope per month -> Sv/century
        trend_results_full = []
        t_idx_tr = np.arange(n_full, dtype=float)
        sigma_ones_tr = np.ones(n_full, dtype=np.float64)
        for name, y_vals in [
            ("NN_SSH", ssh_no_full[valid_full]),
            ("Ekman", ek_full[valid_full]),
            ("RAPID", rapid_full[valid_full]),
            ("NN_SSH_plus_Ekman", ssh_with_full[valid_full]),
        ]:
            s_m, _, _, _, info_m, _, k_m = compute_trend_ci(
                y_vals.copy(), sigma_ones_tr, t_idx_tr, method="mbb", hint_block_len=120
            )
            s_c = s_m * factor_century
            s_sigma = float(np.std(info_m["boot_slopes"], ddof=1)) * factor_century
            s_2sigma = s_sigma * 2
            trend_results_full.append((name, s_c, s_2sigma))
            print(f"    Trend (MBB, trimmed) {name}: {s_c:+.3f} ± {s_2sigma:.3f} Sv/century")
        r2_path = os.path.join(OUT_DIR, "r2_ssh_vs_rapid.txt")
        with open(r2_path, "w") as f:
            f.write("# LPF24: cutoff=1/24, order=5, fs=1, pad=48\n")
            f.write("# Trimmed {} months each end to avoid LPF edge effects\n".format(LPF_TRIM_MONTHS))
            f.write("# Window: {} to {}  n = {}\n".format(trimmed_start, trimmed_end, n_full))
            f.write("comparison\tR2\tr\tn\n")
            f.write(f"NN_SSH_vs_RAPID_no_winds\t{r2_no:.6f}\t{r_no:.6f}\t{valid.sum()}\n")
            f.write(f"NN_SSH_vs_RAPID_with_winds\t{r2_with:.6f}\t{r_with:.6f}\t{valid.sum()}\n")
            f.write("series\tslope_Sv_per_century\ttwo_sigma_Sv_per_century\n")
            for name, s_c, s_2s in trend_results_full:
                f.write(f"{name}\t{s_c:.6f}\t{s_2s:.6f}\n")
        print(f"    ✓ R² and trends (MBB ± 2σ) saved to {r2_path}")
        # Export time series (trimmed) to CSV
        csv_path = os.path.join(OUT_DIR, "amoc_ssh_rapid_ekman_timeseries.csv")
        times_trimmed = rapid_times[common_ok]
        with open(csv_path, "w") as f:
            f.write("time,NN_SSH,Ekman,RAPID,NN_SSH_plus_Ekman\n")
            for i in range(len(times_trimmed)):
                if valid[i]:
                    tstr = np.datetime_as_string(times_trimmed[i], unit="D")
                    f.write(f"{tstr},{ssh_no_ek_common[i]:.6f},{ek_common[i]:.6f},{rapid_with_ek_common[i]:.6f},{ssh_with_ek_common[i]:.6f}\n")
        print(f"    ✓ Time series (trimmed) saved to {csv_path}")
        # ─── TREND COMPARISON FIGURE: SSH trend ± 2σ vs RAPID trend ──────────
        print("\n    [TREND FIG] SSH trend ± 2σ (MBB) vs RAPID trend (trimmed overlap)...")
        t_idx_full = np.arange(n_full, dtype=float)
        sigma_ones_full = np.ones(n_full, dtype=np.float64)
        times_plot = times_trimmed[valid]
        # SSH+Ekman trend via MBB
        ssh_ek_vals = ssh_with_full[valid_full]
        s_ssh, i_ssh, _, _, info_ssh, _, k_ssh = compute_trend_ci(
            ssh_ek_vals.copy(), sigma_ones_full, t_idx_full, method="mbb", hint_block_len=120
        )
        ssh_slope_sigma = float(np.std(info_ssh["boot_slopes"], ddof=1))
        ssh_trend_line = s_ssh * t_idx_full + i_ssh
        # Intercept SE from WLS for the baseline width
        X_full = np.c_[np.ones(n_full), t_idx_full]
        XtX_full = X_full.T @ X_full
        XtX_inv_full = np.linalg.inv(XtX_full)
        sigma_int_ssh = float(np.sqrt(max(XtX_inv_full[0, 0], 0.0)))
        t_mid_full = np.mean(t_idx_full)
        dt_full = t_idx_full - t_mid_full
        se_ssh = np.sqrt(sigma_int_ssh**2 + (dt_full * ssh_slope_sigma)**2)
        # RAPID trend (simple linear)
        rapid_vals_ov = rapid_full[valid_full]
        s_rapid, i_rapid = np.polyfit(t_idx_full, rapid_vals_ov, 1)
        rapid_trend_line = s_rapid * t_idx_full + i_rapid
        # Offset RAPID trend to match SSH trend mean (visual alignment)
        rapid_trend_line += np.mean(ssh_trend_line) - np.mean(rapid_trend_line)
        s_ssh_c = s_ssh * factor_century
        s2_ssh_c = ssh_slope_sigma * 2 * factor_century
        s_rapid_c = s_rapid * factor_century
        # Plot
        fig_tr, ax_tr = plt.subplots(figsize=(10, 4.6), dpi=300)
        ax_tr.fill_between(times_plot, ssh_trend_line - 2*se_ssh, ssh_trend_line + 2*se_ssh,
                           facecolor=COLORS["ssh"], alpha=0.15, edgecolor="none")
        ax_tr.fill_between(times_plot, ssh_trend_line - 1*se_ssh, ssh_trend_line + 1*se_ssh,
                           facecolor=COLORS["ssh"], alpha=0.30, edgecolor="none")
        ax_tr.plot(times_plot, ssh_trend_line, lw=2.5, color=COLORS["ssh"],
                   label=f"DL SSH+Ek trend: {s_ssh_c:+.1f} ± {s2_ssh_c:.1f} Sv/century")
        ax_tr.plot(times_plot, rapid_trend_line, lw=2.5, color=COLORS["rapid"], ls="--",
                   label=f"RAPID trend: {s_rapid_c:+.1f} Sv/century")
        ax_tr.set_xlim(times_plot[0], times_plot[-1])
        y_pad_tr = max(np.nanmax(np.abs(2*se_ssh)), 0.5) * 1.3
        y_mid_tr = np.mean(ssh_trend_line)
        ax_tr.set_ylim(y_mid_tr - y_pad_tr, y_mid_tr + y_pad_tr)
        ax_tr.set_ylabel("AMOC 26.5°N (Sv)", fontsize=14)
        ax_tr.tick_params(axis="both", labelsize=14)
        from matplotlib.dates import YearLocator, DateFormatter
        ax_tr.xaxis.set_major_locator(YearLocator(2))
        ax_tr.xaxis.set_major_formatter(DateFormatter('%Y'))
        for sp in ax_tr.spines.values():
            sp.set_linewidth(0.8); sp.set_color('#333333')
        ax_tr.legend(frameon=True, framealpha=0.9, edgecolor="none", fontsize=13, loc="best")
        ax_tr.set_title("Trend comparison: DL SSH+Ekman vs RAPID (trimmed overlap, LPF24)", fontsize=13)
        fig_tr.tight_layout()
        trend_fig_path = os.path.join(OUT_DIR, "trend_ssh_vs_rapid_overlap.png")
        fig_tr.savefig(trend_fig_path, dpi=300, bbox_inches="tight")
        plt.close(fig_tr)
        print(f"    ✓ Trend comparison figure saved to {trend_fig_path}")
        # ─── SSH vs RAPID: with Ekman and without Ekman + R² ─────────────────
        # Plot both series on COMMON time grid (RAPID times) to avoid phase shift from different
        # time conventions (e.g. RAPID resample '1M' = month-end vs SSH possibly month-start).
        print("\n    [SSH vs RAPID FIG] With Ekman and without Ekman, R² annotated (common time grid)...")
        rapid_wo_ek_offset = np.nanmean(ssh_no_ek_common[valid]) - np.nanmean(rapid_wo_ek_common[valid])
        m_rapid_trim = (rapid_times >= trimmed_start) & (rapid_times <= trimmed_end)
        t_plot = t_rapid_num[m_rapid_trim]
        # Interpolate SSH onto RAPID time grid so both curves share same x (no visual phase shift)
        ssh_with_ek_plot = np.interp(t_plot, t_ssh_num, np.nan_to_num(ssh_with_ek, nan=np.nanmean(ssh_with_ek)))
        ssh_no_ek_plot = np.interp(t_plot, t_ssh_num, np.nan_to_num(ssh_no_ek, nan=np.nanmean(ssh_no_ek)))
        rapid_with_plot = (rapid_lp + rapid_offset)[m_rapid_trim]
        rapid_wo_ek_plot = (rapid_wo_ek_lp + rapid_wo_ek_offset)[m_rapid_trim]
        times_plot_common = rapid_times[m_rapid_trim]
        fig_comp, (ax_with, ax_no) = plt.subplots(1, 2, figsize=(12, 4.6), dpi=300, sharey=True)
        # Left: SSH+Ekman vs RAPID (with Ekman) — same time axis
        ax_with.plot(times_plot_common, ssh_with_ek_plot, lw=1.6, color=COLORS["ssh"], label="DL SSH + Ekman")
        ax_with.plot(times_plot_common, rapid_with_plot, lw=1.6, color=COLORS["rapid"], label="RAPID")
        ax_with.set_xlim(trimmed_start, trimmed_end)
        ax_with.set_ylabel("AMOC 26.5°N (Sv)", fontsize=14)
        bbox_props = dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.9)
        ax_with.text(0.02, 0.98, "a", transform=ax_with.transAxes, fontsize=16, fontweight="bold", va="top", ha="left", bbox=bbox_props)
        ax_with.text(0.98, 0.98, f"R$^2$ = {r2_with:.4f}", transform=ax_with.transAxes, fontsize=12, va="top", ha="right", bbox=bbox_props)
        ax_with.legend(loc="lower left", fontsize=12, framealpha=0.95)
        ax_with.xaxis.set_major_locator(YearLocator(2))
        ax_with.xaxis.set_major_formatter(DateFormatter("%Y"))
        ax_with.tick_params(axis="both", labelsize=12)
        for sp in ax_with.spines.values():
            sp.set_linewidth(0.8); sp.set_color("#333333")
        # Right: SSH (no Ekman) vs RAPID (no Ekman) — same time axis
        ax_no.plot(times_plot_common, ssh_no_ek_plot, lw=1.6, color=COLORS["ssh"], label="DL SSH (no Ekman)")
        ax_no.plot(times_plot_common, rapid_wo_ek_plot, lw=1.6, color=COLORS["rapid"], label="RAPID (no Ekman)")
        ax_no.set_xlim(trimmed_start, trimmed_end)
        ax_no.text(0.02, 0.98, "b", transform=ax_no.transAxes, fontsize=16, fontweight="bold", va="top", ha="left", bbox=bbox_props)
        ax_no.text(0.98, 0.98, f"R$^2$ = {r2_no:.4f}", transform=ax_no.transAxes, fontsize=12, va="top", ha="right", bbox=bbox_props)
        ax_no.legend(loc="lower left", fontsize=12, framealpha=0.95)
        ax_no.xaxis.set_major_locator(YearLocator(2))
        ax_no.xaxis.set_major_formatter(DateFormatter("%Y"))
        ax_no.tick_params(axis="both", labelsize=12)
        for sp in ax_no.spines.values():
            sp.set_linewidth(0.8); sp.set_color("#333333")
        fig_comp.tight_layout()
        comp_fig_path = os.path.join(OUT_DIR, "ssh_vs_rapid_with_and_without_ekman.png")
        fig_comp.savefig(comp_fig_path, dpi=300, bbox_inches="tight")
        plt.close(fig_comp)
        print(f"    ✓ SSH vs RAPID (with/without Ekman, R²) saved to {comp_fig_path}")
    else:
        print("    ⚠ Not enough valid overlap for R² (after LPF trim)")
    
    # DL SSH and DL SST: full available length; RAPID/ECCO: trimmed window
    PLOT_WINDOW = (trimmed_start, trimmed_end)
    m_ssh_plot = np.ones(len(times_ssh), dtype=bool)    # full SSH length
    m_sst_plot = (times_sst >= times_ssh[0]) & (times_sst <= times_ssh[-1])  # clipped to SSH range
    m_rapid_plot = (rapid_times >= PLOT_WINDOW[0]) & (rapid_times <= PLOT_WINDOW[1])
    
    # ═══════════════════════════════════════════════════════════════════════
    #  CREATE THE FIGURE (DL full length, RAPID/ECCO trimmed)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n[PLOTTING] Creating Nature-quality figure (DL full length, RAPID/ECCO trimmed)...")
    
    # Set up matplotlib for publication quality
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size': 16,
        'axes.linewidth': 0.8,
        'axes.labelweight': 'medium',
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.major.size': 4,
        'ytick.major.size': 4,
    })
    
    # Nature double-column width: ~183mm ≈ 7.2 inches
    fig, ax = plt.subplots(figsize=(10.5, 4.6), dpi=300)
    
    # ── SSH reconstruction with uncertainty band ────────────────────────────
    ax.fill_between(
        times_ssh[m_ssh_plot],
        (ssh_with_ek - 2*ssh_sigma)[m_ssh_plot],
        (ssh_with_ek + 2*ssh_sigma)[m_ssh_plot],
        alpha=0.22, facecolor=COLORS["ssh"], edgecolor="none",
        zorder=2, label="_nolegend_"
    )
    ax.plot(times_ssh[m_ssh_plot], ssh_with_ek[m_ssh_plot],
            lw=1.6, color=COLORS["ssh"], label="DL SSH ± 2σ", zorder=5)
    
    # ── SST reconstruction ───────────────────────────────────────────────────
    ax.plot(times_sst[m_sst_plot], sst_with_ek[m_sst_plot],
            lw=1.6, color=COLORS["sst"], label="DL SST", zorder=4.5)
    
    # ── RAPID ───────────────────────────────────────────────────────────────
    ax.plot(rapid_times[m_rapid_plot], (rapid_lp + rapid_offset)[m_rapid_plot],
            lw=1.6, color=COLORS["rapid"], label="RAPID", zorder=4)
    
    # ── ECCO ────────────────────────────────────────────────────────────────
    m_ecco_plot = None
    if ecco_lp is not None and ecco_times is not None:
        m_ecco_plot = (ecco_times >= PLOT_WINDOW[0]) & (ecco_times <= PLOT_WINDOW[1])
        ax.plot(ecco_times[m_ecco_plot], ecco_lp[m_ecco_plot],
                lw=1.6, color=COLORS["ecco"], label="ECCOv4r4", zorder=3)
    
    # ── Axis limits: span SSH reconstruction range ──────────────────────────
    ax.set_xlim(times_ssh[0], times_ssh[-1])
    
    # Dynamic y-limits (from data in plot window only)
    all_vals = np.concatenate([
        np.asarray(ssh_with_ek[m_ssh_plot]).flatten(),
        np.asarray(sst_with_ek[m_sst_plot]).flatten(),
        np.asarray((rapid_lp + rapid_offset)[m_rapid_plot]).flatten()
    ])
    if ecco_lp is not None and m_ecco_plot is not None:
        all_vals = np.concatenate([all_vals, np.asarray(ecco_lp[m_ecco_plot]).flatten()])
    y_min, y_max = np.nanmin(all_vals), np.nanmax(all_vals)
    y_pad = (y_max - y_min) * 0.12
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    
    # ── Axis labels ─────────────────────────────────────────────────────────
    #ax.set_xlabel("Year", fontsize=10, fontweight='medium')
    ax.set_ylabel("AMOC 26.5°N anomaly, Sv", fontsize=16, fontweight='medium')
    
    # ── Ticks ───────────────────────────────────────────────────────────────
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.xaxis.set_major_locator(YearLocator(5))
    ax.xaxis.set_major_formatter(DateFormatter('%Y'))
    
    # ── Spines ──────────────────────────────────────────────────────────────
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color('#333333')
    
    # ── Legend ──────────────────────────────────────────────────────────────
    legend = ax.legend(
        loc='lower left',
        frameon=True,
        framealpha=0.95,
        edgecolor='none',
        fontsize=16,
        handlelength=1.8,
        labelspacing=0.35,
        borderpad=0.4,
    )
    legend.get_frame().set_facecolor('white')
    
    # ── Tight layout ────────────────────────────────────────────────────────
    fig.tight_layout()
    
    # ── Save ────────────────────────────────────────────────────────────────
    out_png = os.path.join(OUT_DIR, "nature_ssh_rapid_ecco.png")
    out_pdf = os.path.join(OUT_DIR, "nature_ssh_rapid_ecco.pdf")
    
    fig.savefig(out_png, dpi=600, bbox_inches="tight", facecolor='white')
    fig.savefig(out_pdf, bbox_inches="tight", facecolor='white')
    
    plt.close(fig)
    
    print(f"\n✓ Saved: {out_png}")
    print(f"✓ Saved: {out_pdf}")
    
    # ═══════════════════════════════════════════════════════════════════════
    #  READ TRENDS FROM CSV FILE
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("  READING TRENDS FROM CSV FILE")
    print("="*70 + "\n")
    
    import csv
    
    csv_path = os.path.join(OUT_DIR, "trend_summary.csv")
    if not os.path.exists(csv_path):
        print(f"  ✗ CSV file not found: {csv_path}")
        return
    
    # Read CSV file
    results = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            series = row["series"]
            period = row["period"]
            method = row["method"]
            slope = float(row["slope_Sv_per_century"])
            two_sigma = float(row["two_sigma_Sv_per_century"])
            lo = float(row["lo95_Sv_per_century"])
            hi = float(row["hi95_Sv_per_century"])
            
            # Filter for desired periods: 1900-2000 and 1993-2023
            # Check original format (CSV uses en-dash U+2013)
            if period not in ["1900–2000", "1993–2023"]:
                continue
            
            # Normalize period string to LaTeX format (double hyphen)
            period_norm = period.replace("\u2013", "--").replace("–", "--").replace("-", "--")
            
            # Filter for desired methods: WLS+HAC, MBB, GLS bootstrap
            if method not in ["WLS+HAC", "MBB", "GLS bootstrap"]:
                continue
            
            results.append({
                "reconstruction": series,
                "period": period_norm,
                "method": method,
                "slope": slope,
                "two_sigma": two_sigma,
                "lo": lo,
                "hi": hi,
                "is_primary": (method == "MBB")
            })
    
    print(f"  ✓ Loaded {len(results)} trend results from CSV")
    
    # Print summary
    for recon in ["SSH→AMOC", "SST→AMOC"]:
        for period in ["1900--2000", "1993--2023"]:
            period_results = [r for r in results if r["reconstruction"] == recon and r["period"] == period]
            if period_results:
                print(f"\n[{recon}] {period}:")
                for r in sorted(period_results, key=lambda x: {"WLS+HAC": 0, "MBB": 1, "GLS bootstrap": 2}.get(x["method"], 99)):
                    marker = "★" if r["is_primary"] else " "
                    print(f"  {marker} {r['method']:20s}: {r['slope']:+.3f} ± {r['two_sigma']:.3f} Sv/century   "
                          f"| 95% CI [{r['lo']:+.3f}, {r['hi']:+.3f}]")
    
    # ═══════════════════════════════════════════════════════════════════════
    #  GENERATE LaTeX TABLE
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("  LaTeX TABLE")
    print("="*70 + "\n")
    
    # Format function for numbers
    def fmt_num(x, decimals=3):
        if abs(x) < 0.001:
            return "0.000"
        s = f"{x:+.{decimals}f}"
        if s.startswith("+"):
            return s[1:]  # Remove leading +
        return s  # Keep negative sign as is (will be in math mode)
    
    # Build LaTeX table
    latex_lines = []
    latex_lines.append("\\begin{table}[h]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{AMOC trends at 26.5°N from SSH and SST reconstructions. Trends are reported in Sv per century with $\\pm 2\\sigma$ uncertainty and 95\\% confidence intervals. MBB (Moving Block Bootstrap) is the primary method (highlighted in bold).}")
    latex_lines.append("\\label{tab:amoc_trends}")
    latex_lines.append("\\begin{tabular}{lccccc}")
    latex_lines.append("\\toprule")
    latex_lines.append("\\textbf{Reconstruction} & \\textbf{Period} & \\textbf{Method} & \\textbf{Trend} & \\textbf{95\\% CI} \\\\")
    latex_lines.append(" & & & \\textbf{(Sv/century)} & \\textbf{(Sv/century)} \\\\")
    latex_lines.append("\\midrule")
    
    # Group by reconstruction and period
    first_recon = True
    for recon in ["SSH→AMOC", "SST→AMOC"]:
        recon_has_data = False
        for period in ["1900--2000", "1993--2023"]:
            period_results = [r for r in results if r["reconstruction"] == recon and r["period"] == period]
            if not period_results:
                continue
            
            recon_has_data = True
            
            # Sort by method order
            method_order = {"WLS+HAC": 0, "MBB": 1, "GLS bootstrap": 2}
            period_results.sort(key=lambda x: method_order.get(x["method"], 99))
            
            for i, r in enumerate(period_results):
                recon_str = recon.replace("→", "$\\to$")
                method_str = r["method"]
                if r["is_primary"]:
                    method_str = f"\\textbf{{{method_str}}}"
                
                trend_str = f"${fmt_num(r['slope'])} \\pm {fmt_num(r['two_sigma'])}$"
                ci_str = f"$[{fmt_num(r['lo'])}, {fmt_num(r['hi'])}]$"
                
                if i == 0:
                    # First row: show reconstruction and period
                    latex_lines.append(f"{recon_str} & {period} & {method_str} & {trend_str} & {ci_str} \\\\")
                else:
                    # Subsequent rows: empty reconstruction and period
                    latex_lines.append(f" & & {method_str} & {trend_str} & {ci_str} \\\\")
        
        # Add spacing between reconstructions
        if recon_has_data and not first_recon:
            latex_lines.append("\\midrule")
        first_recon = False
    
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")
    
    latex_table = "\n".join(latex_lines)
    
    print(latex_table)
    
    # Save to file
    latex_file = os.path.join(OUT_DIR, "trends_table_latex.tex")
    with open(latex_file, "w") as f:
        f.write(latex_table)
    
    print(f"\n✓ LaTeX table saved to: {latex_file}")
    
    print("\n" + "="*70)
    print("  Done! Ready for Nature submission 🎉")
    print("="*70)


if __name__ == "__main__":
    main()
