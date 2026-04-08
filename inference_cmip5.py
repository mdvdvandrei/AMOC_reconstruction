#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CMIP5 TEST inference with SST-based AMOC reconstruction:
 - Per-variable reconstruction plots (+ SST recon overlay)
 - Scatter: Recon(SST) vs Pred(NN μ), per variable AND all variables
 - Full-period trends (Sv/century) for Pred, Recon, True
 - Per-fold trend CSVs + combined-across-folds aggregations (mean, 95% CI)
 - Optional violin/hist of trend errors per variable
 - Combined member plots (all vars overlaid) — FIXED

Outputs:
  examples_of_reconstructions/<DATE>/test_model(<MODEL>)/<VAR>/
      reconstruction.png
      scatter_recon_vs_pred_test_model(<MODEL>)_<VAR>.png
  examples_of_reconstructions/<DATE>/test_model(<MODEL>)/  (ALL VARS)
      scatter_recon_vs_pred_ALLVARS_test_model(<MODEL>).png
      members_combined/member_<ID>_combined.png
  results/
      infer_TEST_CMIP5_<MODEL>_<save_name>_<scenario>.csv
      infer_TEST_CMIP5_ALL_<save_name>_<scenario>.csv
      trends_full_period/
          trends_full_test_model(<MODEL>)_<VAR>.csv        (combined across folds)
          trends_full_ALLVARS_test_model(<MODEL>).csv       (combined across vars)
          trends_full_per_fold_<MODEL>_<VAR>.csv            (per-fold table)
          trends_full_ALL_MODELS_<save_name>_<scenario>.csv (grand table)
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from datetime import datetime
import math, json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from scipy.stats import norm

# Optional violin plots
try:
    import seaborn as sns
    _HAVE_SNS = True
except Exception:
    _HAVE_SNS = False

# ---------------- external (your) modules ----------------
from dataset_for_cesm2_LE import PreprocessedCMIP6Dataset_LE
from datasets_cmip_zarr import PreprocessedCMIP6Dataset
from omegaconf import OmegaConf
from models import ResidualCNNHetBaseline

# ---------------- global settings ----------------
torch.set_default_dtype(torch.float32)
torch.set_num_threads(8)
torch.set_num_interop_threads(8)

START_YEAR = 1855

# === SST index → AMOC reconstruction params (your numbers) ===
USE_SST_INDEX = True
SST_CSV = "/home/am334/link_am334/data/sst_amoc_fingerprint_spg_minus_global_DJFMAM.csv"
RECON_A_SV = 0.2648
RECON_B_SV_PER_C = 1.6057
SST_BASELINE_YEARS = 100

# ---------------- small utilities ----------------
def _as_list(x):
    if x is None: return []
    if isinstance(x, (list, tuple)): return list(x)
    if isinstance(x, str): return [s.strip() for s in x.split(",") if s.strip()]
    return [x]

class ArtifactWriter:
    """Persist series/metrics/trends to Parquet for later plotting."""
    def __init__(self, root: Path, save_name: str, scenario: str, cfg):
        date_str = datetime.now().strftime("%Y-%m-%d")
        self.dir = Path(root) / date_str / save_name / scenario
        self.dir.mkdir(parents=True, exist_ok=True)
        self._series_rows   = []
        self._trends_rows   = []
        self._metrics_rows  = []
        man = {
            "date": date_str, "save_name": save_name,
            "scenario": str(scenario),
            "target_var": str(getattr(cfg, "target_var", "")),
            "output_type": str(getattr(cfg, "output_type", "")),
            "selected_lats": [float(x) for x in _as_list(getattr(cfg, "selected_lats", []))],
            "lpf": str(getattr(cfg, "lpf", "")),
        }
        (self.dir / "manifest.json").write_text(json.dumps(man, indent=2))

    def add_member_series(self, *, model: str, var: str, member: str,
                          years: np.ndarray, mu: np.ndarray, sig: np.ndarray, tgt: np.ndarray,
                          extra: dict | None = None):
        years = np.asarray(years).ravel()
        mu    = np.asarray(mu).ravel()
        sig   = np.asarray(sig).ravel()
        tgt   = np.asarray(tgt).ravel()
        n = int(min(years.size, mu.size, sig.size, tgt.size))
        if n == 0: return
        df = pd.DataFrame({
            "model": model, "var": var, "member": str(member),
            "year": years[:n], "mu": mu[:n], "sig": sig[:n], "tgt": tgt[:n],
        })
        if extra:
            for k, v in extra.items():
                df[k] = v
        self._series_rows.append(df)

    def add_model_metrics(self, *, model: str, var: str, r2: float, corr: float,
                          std_pred: float, std_true: float, crps: float = 0.0):
        self._metrics_rows.append(pd.DataFrame([{
            "model": model, "var": var, "r2": float(r2), "corr": float(corr),
            "std_pred": float(std_pred), "std_true": float(std_true), "crps": float(crps),
        }]))

    def _save_table(self, name: str, df: pd.DataFrame):
        try:
            df.to_parquet(self.dir / f"{name}.parquet", index=False)
        except Exception:
            df.to_csv(self.dir / f"{name}.csv", index=False)

    def flush(self):
        if self._series_rows:
            self._save_table("series", pd.concat(self._series_rows, ignore_index=True))
        if self._metrics_rows:
            self._save_table("metrics", pd.concat(self._metrics_rows, ignore_index=True))


def _get_member_ids_safe(ds):
    try:
        return np.array([ds.get_sample_info(i)["member"] for i in range(len(ds))])
    except Exception:
        return None

def compute_metrics(preds: np.ndarray, targets: np.ndarray):
    preds = preds.ravel(); targets = targets.ravel()
    ss_res = ((targets - preds) ** 2).sum()
    ss_tot = ((targets - targets.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    corr = np.corrcoef(preds, targets)[0, 1] if preds.size else np.nan
    return float(r2), float(corr)

def compute_r2_only(pred: np.ndarray, true: np.ndarray) -> float:
    pred = np.asarray(pred).ravel(); true = np.asarray(true).ravel()
    if pred.size == 0 or true.size == 0: return float('nan')
    ss_res = np.sum((true - pred)**2)
    ss_tot = np.sum((true - np.mean(true))**2)
    return float('nan') if ss_tot <= 0 else float(1.0 - ss_res/ss_tot)

def _parse_lpf_months(lpf_str: str | None) -> int | None:
    if not lpf_str: return None
    d = "".join(ch for ch in lpf_str if ch.isdigit())
    return int(d) if d else None

def load_sst_index(csv_path: str, model: str, scenario: str, member: str | None = None, lpf_str: str | None = None):
    df = pd.read_csv(csv_path)
    if "time" not in df.columns: raise ValueError("SST CSV must contain 'time'.")
    val_col = "SPG_minus_GLOBAL"
    if val_col not in df.columns:
        numc = [c for c in df.select_dtypes(include="number").columns if c != "time"]
        if not numc: raise ValueError("SST CSV has no numeric data.")
        val_col = numc[0]
    if "model" in df.columns:    df = df[df["model"] == model]
    if "scenario" in df.columns: df = df[df["scenario"] == scenario]
    if df.empty: return None, None
    if "member" in df.columns:
        if (member is not None) and (df["member"] == member).any():
            df = df[df["member"] == member]
        else:
            df = df.groupby("time", as_index=False).mean(numeric_only=True)
    df = df.sort_values("time")
    years = df["time"].to_numpy(dtype=int)
    vals  = df[val_col].to_numpy(dtype=float)
    months = _parse_lpf_months(lpf_str)
    if months and months > 1:
        k = max(1, int(round(months/12)))
        if k > 1:
            vals = pd.Series(vals).rolling(window=k, center=True, min_periods=max(1,k//2)).mean().to_numpy()
    return years, vals

def load_sst_index_per_member(csv_path: str, model: str, scenario: str, lpf_str: str | None = None):
    df = pd.read_csv(csv_path)
    if "model" in df.columns:    df = df[df["model"] == model]
    if "scenario" in df.columns: df = df[df["scenario"] == scenario]
    if df.empty: raise ValueError(f"No SST rows for {model}/{scenario}")
    if "time" not in df.columns: raise ValueError("SST CSV must contain 'time'.")
    val_col = "SPG_minus_GLOBAL"
    if val_col not in df.columns:
        numc = [c for c in df.select_dtypes(include="number").columns if c != "time"]
        if not numc: raise ValueError("SST CSV has no numeric value column.")
        val_col = numc[0]
    if "member" not in df.columns:
        df["member"] = "ALL"
    df = (df[["member","time",val_col]].groupby(["member","time"], as_index=False).mean(numeric_only=True))
    k = None
    months = _parse_lpf_months(lpf_str)
    if months and months > 1: k = max(1, int(round(months/12)))
    idx_map = {}
    for member, g in df.groupby("member", sort=True):
        g = g.sort_values("time")
        years = g["time"].to_numpy(dtype=int)
        vals  = g[val_col].to_numpy(dtype=float)
        if k and k > 1:
            vals = pd.Series(vals).rolling(window=k, center=True, min_periods=max(1,k//2)).mean().to_numpy()
        idx_map[str(member)] = (years, vals)
    return idx_map

def _demean_first_n_years(years: np.ndarray, vals: np.ndarray, n_first: int = SST_BASELINE_YEARS):
    years = np.asarray(years, int); vals = np.asarray(vals, float)
    if years.size == 0: return years, vals
    order = np.argsort(years)
    y = years[order]; v = vals[order]
    uniq = np.unique(y)
    base_years = uniq[:min(n_first, len(uniq))]
    base_mask  = np.isin(y, base_years)
    base_mean  = np.nanmean(v[base_mask]) if base_mask.any() else np.nan
    return years, (vals - base_mean)

def sst_to_amoc_recon(years: np.ndarray, sst_vals: np.ndarray,
                      a_sv: float = RECON_A_SV, b_sv_per_c: float = RECON_B_SV_PER_C,
                      n_base: int = SST_BASELINE_YEARS):
    yrs, sst_prime = _demean_first_n_years(years, sst_vals, n_first=n_base)
    amoc_hat = a_sv + b_sv_per_c * sst_prime
    return yrs, amoc_hat

def r2_sst_against_true(years_true: np.ndarray, true_vals: np.ndarray,
                        sst_years: np.ndarray | None, sst_vals:  np.ndarray | None) -> float:
    if sst_years is None or sst_vals is None: return float('nan')
    yy, amoc_hat = sst_to_amoc_recon(sst_years, sst_vals)
    years_true = np.asarray(years_true, int)
    mask = np.isin(yy, years_true)
    if not np.any(mask): return float('nan')
    yy_i = yy[mask]
    idx_map = {y:i for i, y in enumerate(years_true)}
    t_idx = np.array([idx_map[y] for y in yy_i if y in idx_map], int)
    if t_idx.size == 0: return float('nan')
    return compute_r2_only(amoc_hat[mask][:t_idx.size], true_vals[:t_idx.size])

def annotate_r2(ax, r2_nn: float, r2_sst: float, fontsize=10):
    txt = f"R² NN = {np.round(r2_nn, 3)}"
    if np.isfinite(r2_sst):
        txt += f"\nR² SST index = {np.round(r2_sst, 3)}"
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, va='top', ha='left',
            fontsize=fontsize, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

def align_by_years(y1: np.ndarray, v1: np.ndarray, y2: np.ndarray, v2: np.ndarray):
    y1 = np.asarray(y1, int); v1 = np.asarray(v1, float)
    y2 = np.asarray(y2, int); v2 = np.asarray(v2, float)
    common = np.intersect1d(y1, y2)
    if common.size == 0:
        return np.array([]), np.array([]), np.array([])
    i1 = np.isin(y1, common); i2 = np.isin(y2, common)
    # preserve common ordering
    order = np.argsort(common)
    common = common[order]
    v1a = v1[i1]; v2a = v2[i2]
    idx1 = {y:i for i,y in enumerate(y1[i1])}
    idx2 = {y:i for i,y in enumerate(y2[i2])}
    v1_common = np.array([v1a[idx1[y]] for y in common], float)
    v2_common = np.array([v2a[idx2[y]] for y in common], float)
    return common, v1_common, v2_common

def linear_trend_all_period(
    years: np.ndarray,
    values: np.ndarray,
    min_points: int = 2,
    skip_first_years: int = 45,
    skip_first_samples: int = 45,
):
    """
    Linear trend [Sv/century] over all data after discarding an initial window.
    If skip_first_years>0, we drop any year earlier than (first_year + skip_first_years).
    Otherwise, we drop skip_first_samples elements from the start.
    """
    years  = np.asarray(years,  float)
    values = np.asarray(values, float)

    m = np.isfinite(years) & np.isfinite(values)
    years, values = years[m], values[m]
    if years.size < min_points:
        return float("nan")

    if skip_first_years > 0 and np.isfinite(years).any():
        first_year = float(np.nanmin(years))
        keep = years >= (first_year + skip_first_years)
        years, values = years[keep], values[keep]
    elif skip_first_samples > 0:
        years, values = years[skip_first_samples:], values[skip_first_samples:]

    if years.size < min_points:
        return float("nan")

    slope = np.polyfit(years, values, 1)[0]
    return float(slope * 100.0)  # Sv / century

# ---------------- plotting helpers ----------------
def plot_reconstruction(mu, sig, tgt, out_dir: Path, model_name: str, var: str,
                        s_year=None, s_val=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(mu))
    years_full = START_YEAR + x
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(years_full, tgt, label="Truth", lw=2, color="red")
    ax.plot(years_full, mu,  label="Prediction μ (Stage-1)", lw=2, color="blue")
    ax.fill_between(years_full, mu - 2*sig, mu + 2*sig, alpha=0.25, label="±2σ (Stage-2)", color="blue")
    r2_nn = compute_r2_only(mu, tgt)
    r2_sst = r2_sst_against_true(years_full, tgt, s_year, s_val)
    if USE_SST_INDEX and (s_year is not None) and (s_val is not None):
        _, amoc_hat = sst_to_amoc_recon(s_year, s_val)
        yy, _, a_tr = align_by_years(years_full, mu, s_year, amoc_hat)
        if yy.size:
            ax.plot(yy, a_tr, '--', lw=1.6, color='k', label="AMOĈ from SST′")
    annotate_r2(ax, r2_nn, r2_sst, fontsize=10)
    ax.set_title(f"{model_name} — {var}")
    ax.set_xlabel("Year"); ax.set_ylabel("Sv")
    ax.grid(True, alpha=0.3); ax.legend(loc="upper left")
    file = out_dir / "reconstruction.png"
    fig.tight_layout(); fig.savefig(file, dpi=150); plt.close(fig)

def plot_scatter_recon_vs_pred(years_pred: np.ndarray, mu_pred: np.ndarray,
                               years_recon: np.ndarray, recon_vals: np.ndarray,
                               out_path: Path, title: str, max_points: int | None = 40000):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ys, mu_a, r_a = align_by_years(years_pred, mu_pred, years_recon, recon_vals)
    if ys.size == 0:
        print(f"[warn] No overlapping years for scatter {out_path.name}")
        return
    if (max_points is not None) and (ys.size > max_points):
        idx = np.random.default_rng(0).choice(ys.size, size=max_points, replace=False)
        mu_a, r_a = mu_a[idx], r_a[idx]
    r2 = compute_r2_only(r_a, mu_a)
    plt.figure(figsize=(6.8, 6.4))
    plt.scatter(r_a, mu_a, s=10, alpha=0.35, edgecolor='none')
    lim = np.nanmax(np.abs(np.concatenate([r_a, mu_a]))) if r_a.size else 1.0
    lim = max(1.0, float(np.ceil(lim)))
    plt.plot([-lim, lim], [-lim, lim], 'k--', lw=1.2, label='y = x')
    plt.xlabel("Reconstructed AMOC from SST [Sv]")
    plt.ylabel("Predicted NN μ [Sv]")
    plt.xlim(-lim, lim); plt.ylim(-lim, lim)
    plt.title(title + f"\nR² = {r2:.3f}")
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def violin_trend_errors_per_var(model_label: str, var: str,
                                trend_nn_all: list[float],
                                trend_sst_all: list[float],
                                trend_sim_all: list[float],
                                out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr_sst = np.array([v for v in trend_sst_all if np.isfinite(v)])
    arr_nn  = np.array([v for v in trend_nn_all  if np.isfinite(v)])
    arr_sim = np.array([v for v in trend_sim_all if np.isfinite(v)])
    n = min(len(arr_sst), len(arr_nn), len(arr_sim))
    if n == 0:
        print(f"[warn] Not enough data for violin: {out_path.name}"); return
    e_sst = arr_sst[:n] - arr_sim[:n]
    e_nn  = arr_nn[:n]  - arr_sim[:n]
    if _HAVE_SNS:
        df = pd.DataFrame({"Source": np.repeat(["SST Index", "NN μ(Stage-1)"], repeats=n),
                           "Error":  np.concatenate([e_sst, e_nn])})
        plt.figure(figsize=(4.8,6))
        sns.violinplot(x="Source", y="Error", data=df, inner="box", cut=0)
        plt.axhline(0, ls="--", c="k", lw=1)
        plt.ylabel("Trend error vs Simulated [Sv/century]")
        plt.title(f"{model_label} — {var}")
        plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()
    else:
        plt.figure(figsize=(7,6))
        plt.boxplot([e_sst, e_nn], labels=["SST−Sim", "NN−Sim"], showmeans=True)
        plt.axhline(0, ls="--", c="k", lw=1)
        plt.ylabel("Trend error vs Simulated [Sv/century]")
        plt.title(f"{model_label} — {var} (boxplot fallback)")
        plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()

# ---------------- model builders & inference ----------------
def build_mu_model(input_shape, out_dim, device, cfg):
    in_ch, H, W = input_shape
    model = ResidualCNNHetBaseline(in_ch, out_dim, flat_hw=[144, 108], p_drop=0.5, tanh_scale=8.0).to(device)
    model.eval()
    return model

def build_het_model(input_shape, out_dim, device, cfg):
    in_ch, H, W = input_shape
    model = ResidualCNNHetBaseline(in_ch, out_dim, flat_hw=[144, 108], p_drop=0.5, tanh_scale=8.0).to(device)
    model.eval()
    return model

def safe_load_state(model, sd_raw):
    if isinstance(sd_raw, dict) and "state_dict" in sd_raw and isinstance(sd_raw["state_dict"], dict):
        sd_raw = sd_raw["state_dict"]
    msd = model.state_dict()
    filtered = {k: v for k, v in sd_raw.items() if k in msd and msd[k].shape == v.shape}
    model.load_state_dict(filtered, strict=False)

@torch.no_grad()
def infer_mu_stage1(mu_model, dataset, device, batch_size=8):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    mu_list, tgt_list = [], []
    for batch in loader:
        x, y = batch[:2]
        x = x.to(device, dtype=torch.float32)
        mu , _ = mu_model(x)
        mu_list.append(mu.cpu().numpy())
        tgt_list.append(y.numpy())
    return np.concatenate(mu_list, 0), np.concatenate(tgt_list, 0)

@torch.no_grad()
def infer_sigma_stage2(het_model, dataset, device, batch_size=8):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    sig_list = []
    for batch in loader:
        x = batch[0]
        x = x.to(device, dtype=torch.float32)
        _, logvar = het_model(x)
        sig = np.exp(0.5 * logvar.cpu().numpy())
        sig_list.append(sig)
    return np.concatenate(sig_list, 0)

# ---------------- combined member plots ----------------
def save_members_combined_plots_from_arrays(
    mu_by_var: dict[str, np.ndarray],       # var -> μ(time_total,)
    sig_by_var: dict[str, np.ndarray],      # var -> σ(time_total,)
    tgt_ref: np.ndarray,                    # (time_total,)
    member_ids: np.ndarray,                 # (time_total,)
    start_year: int,                        # e.g., 1855
    x_vars: list[str],                      # e.g., ["tos","zos"]
    out_root: Path,                         # base folder to write plots
    model_label: str,                       # e.g., "test_model(CanESM2)"
    sst_member_series: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    # optional: {"r1i1p1": (years, amoc_recon), ...}
):
    """
    Saves one combined figure per member that overlays:
      • Truth (black), per-var μ with ±2σ bands, and optional SST→AMOC reconstruction.
    Assumes all series are concatenated member-after-member in the same order.

    Requirements:
      • For each var in x_vars, mu_by_var[var], sig_by_var[var] have same length as tgt_ref.
      • member_ids gives the member label per time sample (same length).
    """
    out_dir = Path(out_root) / model_label / "members_combined"
    out_dir.mkdir(parents=True, exist_ok=True)

    # pick colors per var
    base_cycle = plt.rcParams.get("axes.prop_cycle", None)
    color_list = (base_cycle.by_key().get("color", ["C0","C1","C2","C3","C4"])
                  if base_cycle is not None else ["C0","C1","C2","C3","C4"])
    var_colors = {v: color_list[i % len(color_list)] for i, v in enumerate(x_vars)}

    uniq_m = np.unique(member_ids)
    years_full = np.arange(start_year, start_year + tgt_ref.shape[0])

    for m in uniq_m:
        idx = np.where(member_ids == m)[0]
        if idx.size == 0:
            continue

        # ✅ correct indexing (not [:idx.size])
        years_m = years_full[idx]
        truth_m = tgt_ref[idx]

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(years_m, truth_m, color="k", lw=2, label="Truth")

        r2_lines = []

        for v in x_vars:
            if v not in mu_by_var or v not in sig_by_var:  # skip missing vars
                continue
            mu_v  = mu_by_var[v][idx]
            sig_v = sig_by_var[v][idx]

            ax.plot(years_m, mu_v, lw=2, color=var_colors[v], label=f"NN μ ({v})")
            if sig_v.size == mu_v.size:
                ax.fill_between(
                    years_m, mu_v - 2.0 * sig_v, mu_v + 2.0 * sig_v,
                    alpha=0.18, color=var_colors[v], linewidth=0
                )

            # R² per var for this member (vs truth)
            r2_lines.append(f"{v}: R²={compute_r2_only(mu_v, truth_m):.3f}")

        # Optional SST recon for this member
        if sst_member_series is not None:
            key = str(m)
            if key in sst_member_series:
                yrs_sst, amoc_sst = sst_member_series[key]
                # align
                common = np.intersect1d(yrs_sst, years_m)
                if common.size > 0:
                    s_idx = np.isin(yrs_sst, common)
                    y_idx = np.isin(years_m, common)
                    ax.plot(
                        years_m[y_idx], amoc_sst[s_idx],
                        ls="--", lw=1.8, color="dimgray", label="SST index → AMOC"
                    )
                    r2_lines.append(f"SST index: R²={compute_r2_only(amoc_sst[s_idx], truth_m[y_idx]):.3f}")

        if r2_lines:
            ax.text(
                0.01, 0.90, "\n".join(r2_lines), transform=ax.transAxes,
                va="top", ha="left", fontsize=10,
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"),
            )

        ax.set_title(f"{model_label} — member {m}", fontsize=12)
        ax.set_xlabel("Year"); ax.set_ylabel("Sv")
        ax.grid(True, alpha=0.25)
        handles, labels = ax.get_legend_handles_labels()
        fig.subplots_adjust(right=0.82)
        fig.legend(handles, labels, loc="center left",
                   bbox_to_anchor=(0.83, 0.5), bbox_transform=fig.transFigure,
                   frameon=True, fontsize=10)

        fpath = out_dir / f"member_{str(m).replace('/', '_')}_combined.png"
        fig.savefig(fpath, dpi=200, bbox_inches="tight", pad_inches=0.25)
        plt.close(fig)
        print(f"Saved combined member plot: {fpath}")

# ---------------- CMIP5 evaluation ----------------
def evaluate_cmip5_test_across_folds(
        cmip5_models: list[str],
        val_models: list[str],
        x_vars: list[str],
        cfg,
        base_dir: Path,
        device: torch.device,
        selected_lats,
        scenario: str,
        target_var: str,
        output_type: str,
        add_coords: bool,
        use_sincos: bool,
        lat_values,
        lon_values,
        lpf,
        aw=None,
    ):

    date_str = datetime.now().strftime("%Y-%m-%d")
    _wr = str(getattr(cfg, "weights_root", "weights"))
    WEIGHTS_ROOT = Path.cwd() / _wr / cfg.save_name

    all_results_rows = []
    grand_trend_rows = []  # across ALL models/vars (for a single big CSV)

    for test_model in cmip5_models:
        model_label = f"test_model({test_model})"
        print(f"\n[CMIP5 TEST] evaluating: {test_model}")

        # SST indices for this CMIP5 model (global & per-member)
        s_year_test, s_val_test = (None, None)
        idx_map_test = None
        if USE_SST_INDEX:
            try:    s_year_test, s_val_test = load_sst_index(SST_CSV, test_model, scenario, member=None, lpf_str=lpf)
            except Exception as e: print(f"  [warn] CMIP5 SST (global) failed: {e}")
            try:    idx_map_test = load_sst_index_per_member(SST_CSV, test_model, scenario, lpf_str=lpf)
            except Exception as e: print(f"  [warn] CMIP5 SST (per-member) failed: {e}")

        # for ALL-VARS combined scatter later
        allvars_mu_series, allvars_years_series = [], []

        per_model_rows = []  # metrics rows for this model

        mu_by_var  = {}
        sig_by_var = {}

        # holders for members-combined saver
        member_ids_global = None
        tgt_ref_global    = None

        for var in x_vars:
            # Build test dataset for THIS CMIP5 model & var
            try:
                ds_raw = PreprocessedCMIP6Dataset(
                    zarr_dir=str(base_dir),
                    models=[test_model], x_vars=[var],
                    scenarios=[scenario], target_group=target_var, output_type=output_type,
                    selected_lats=selected_lats, lpf=lpf, member_selection="all"
                )
            except Exception:
                ds_raw = PreprocessedCMIP6Dataset_LE(
                    zarr_dir=str(base_dir),
                    models=[test_model], x_vars=[var],
                    scenarios=[scenario], target_group=target_var, output_type=output_type,
                    selected_lats=selected_lats, lpf=lpf, member_selection="all"
                )

            if len(ds_raw) == 0:
                print(f"[warn] {test_model}: empty test dataset for var={var}")
                continue

            # record member ids once
            mids = _get_member_ids_safe(ds_raw)
            if member_ids_global is None and mids is not None:
                member_ids_global = mids

            # Shapes
            x0, y0 = ds_raw[0][:2]
            out_dim = int(np.prod(y0.shape))
            bs = int(getattr(cfg.training, "batch_size", 64))

            mu_folds, sig_folds, tgt_ref_folds = [], [], []
            trend_pred_full_per_fold = []  # pred (μ) trend per fold
            trend_true_full_per_fold = []  # true trend per fold
            trend_recon_full_per_fold = [] # recon trend per fold (uses same recon per model)

            # Precompute recon (SST) once
            recon_years, recon_vals = (None, None)
            if USE_SST_INDEX and (s_year_test is not None) and (s_val_test is not None):
                recon_years, recon_vals = sst_to_amoc_recon(s_year_test, s_val_test)

            # run every fold
            for fold_name in val_models:
                wdir = WEIGHTS_ROOT / var / fold_name
                path_mu    = wdir / "best_stage1_mu.pt"
                path_joint = wdir / "best_stage2_joint.pt"

                mu_model  = build_mu_model(x0.shape, out_dim, device, cfg)
                het_model = build_het_model(x0.shape, out_dim, device, cfg)

                if path_mu.exists():
                    safe_load_state(mu_model,  torch.load(path_mu,    map_location=device))
                if path_joint.exists():
                    safe_load_state(het_model, torch.load(path_joint, map_location=device))

                # μ
                if path_mu.exists():
                    mu, tgt = infer_mu_stage1(mu_model, ds_raw, device, batch_size=bs)
                else:
                    loader = DataLoader(ds_raw, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)
                    mu_list, tgt_list = [], []
                    with torch.no_grad():
                        for batch in loader:
                            x, y = batch[:2]
                            x = x.to(device, dtype=torch.float32)
                            mu_b, _ = het_model(x)
                            mu_list.append(mu_b.cpu().numpy()); tgt_list.append(y.numpy())
                    mu = np.concatenate(mu_list, 0); tgt = np.concatenate(tgt_list, 0)

                # σ
                if path_joint.exists():
                    sig = infer_sigma_stage2(het_model, ds_raw, device, batch_size=bs)
                else:
                    res_std = np.std(tgt - mu) if mu.size else 1.0
                    sig = np.full_like(mu, max(res_std, 1e-6))

                mu_flat  = mu.ravel()
                sig_flat = sig.ravel()
                tgt_flat = tgt.ravel()
                mu_folds.append(mu_flat)
                sig_folds.append(sig_flat)
                tgt_ref_folds.append(tgt_flat)

                # per-fold full-period trend on overlap with truth
                years_pred_full = np.arange(START_YEAR, START_YEAR + mu_flat.size)
                years_true_full = np.arange(START_YEAR, START_YEAR + tgt_flat.size)
                yp, mup, trp = align_by_years(years_pred_full, mu_flat, years_true_full, tgt_flat)
                t_pred_full = linear_trend_all_period(yp, mup) if yp.size else np.nan
                t_true_full = linear_trend_all_period(yp, trp) if yp.size else np.nan

                # recon trend on overlap with pred, if available
                if (recon_years is not None) and (recon_vals is not None):
                    yr, _, rc = align_by_years(years_pred_full, mu_flat, recon_years, recon_vals)
                    t_recon_full = linear_trend_all_period(yr, rc) if yr.size else np.nan
                else:
                    t_recon_full = np.nan

                trend_pred_full_per_fold.append(t_pred_full)
                trend_true_full_per_fold.append(t_true_full)
                trend_recon_full_per_fold.append(t_recon_full)

            if len(mu_folds) == 0:
                print(f"[warn] no folds ran for {test_model}/{var}")
                continue

            mu_mean  = np.mean(np.stack(mu_folds, axis=0), axis=0)
            sig_mean = np.mean(np.stack(sig_folds, axis=0), axis=0)
            tgt_ref  = tgt_ref_folds[0]  # same ordering across folds

            if test_model == "FGOALS-s2":
                tgt_ref = tgt_ref*1000000

            # remember one tgt for members-combined
            if tgt_ref_global is None:
                tgt_ref_global = tgt_ref

            mu_by_var[var]  = mu_mean
            sig_by_var[var] = sig_mean

            # metrics
            r2, corr = compute_metrics(mu_mean, tgt_ref)
            std_pred = float(np.std(mu_mean)); std_true = float(np.std(tgt_ref))
            print(f"[CMIP5 TEST] {test_model} — {var}: R²={r2:.4f}, corr={corr:.4f}")

            if aw is not None:
                aw.add_model_metrics(model=model_label, var=var,
                                     r2=r2, corr=corr,
                                     std_pred=std_pred, std_true=std_true)

            # save per-member series to artifact
            if aw is not None and mids is not None and mids.size == mu_mean.size:
                years_full_aw = np.arange(START_YEAR, START_YEAR + mu_mean.shape[0])
                for m in np.unique(mids):
                    idx = np.where(mids == m)[0]
                    if idx.size == 0:
                        continue
                    aw.add_member_series(
                        model=model_label, var=var, member=str(m),
                        years=years_full_aw[idx],
                        mu=mu_mean[idx], sig=sig_mean[idx], tgt=tgt_ref[idx],
                        extra={"scenario": scenario, "lpf": str(lpf)},
                    )

            out_dir = Path.cwd() / "examples_of_reconstructions" / date_str / model_label / var
            # per-var reconstruction plot (+ SST recon overlay line)
            plot_reconstruction(mu_mean, sig_mean, tgt_ref, out_dir, model_label, var,
                                s_year=s_year_test, s_val=s_val_test)

            # per-var scatter Recon vs Pred
            years_pred_full = np.arange(START_YEAR, START_YEAR + mu_mean.size)
            if (recon_years is not None) and (recon_vals is not None):
                plot_scatter_recon_vs_pred(
                    years_pred=years_pred_full, mu_pred=mu_mean,
                    years_recon=recon_years, recon_vals=recon_vals,
                    out_path=out_dir / f"scatter_recon_vs_pred_{model_label}_{var}.png",
                    title=f"{model_label} — {var}"
                )

            # full-period trends (combined across folds via μ_mean) on overlap with truth & recon
            years_true_full = np.arange(START_YEAR, START_YEAR + tgt_ref.size)
            yp, mup, trp = align_by_years(years_pred_full, mu_mean, years_true_full, tgt_ref)
            trend_pred_all   = linear_trend_all_period(yp, mup) if yp.size else np.nan
            trend_true_all   = linear_trend_all_period(yp, trp) if yp.size else np.nan
            if (recon_years is not None) and (recon_vals is not None):
                yr, _, rvp = align_by_years(years_pred_full, mu_mean, recon_years, recon_vals)
                trend_recon_all  = linear_trend_all_period(yr, rvp) if yr.size else np.nan
            else:
                trend_recon_all = np.nan

            # keep for ALL-VARS combined
            allvars_mu_series.append(mu_mean)
            allvars_years_series.append(years_pred_full)

            # Save per-var per-fold trend table
            df_pf = pd.DataFrame({
                "model": test_model,
                "label": model_label,
                "var": var,
                "fold": val_models,
                "trend_pred_Sv_per_century_all": trend_pred_full_per_fold,
                "trend_recon_Sv_per_century_all": trend_recon_full_per_fold,
                "trend_true_Sv_per_century_all": trend_true_full_per_fold,
            })
            per_fold_file = Path.cwd() / "results" / "trends_full_period" / f"trends_full_per_fold_{test_model}_{var}.csv"
            per_fold_file.parent.mkdir(parents=True, exist_ok=True)
            df_pf.to_csv(per_fold_file, index=False)

            # Also save combined-across-folds stats (mean & 95% CI by bootstrap)
            def mean_ci(a):
                a = np.asarray([x for x in a if np.isfinite(x)], float)
                if a.size == 0: return np.nan, np.nan, np.nan
                m = float(np.mean(a))
                rng = np.random.default_rng(0)
                if a.size == 1:
                    return m, m, m
                boots = [np.mean(rng.choice(a, size=a.size, replace=True)) for _ in range(2000)]
                lo, hi = np.percentile(boots, [2.5, 97.5])
                return m, float(lo), float(hi)

            mp, lp, hp = mean_ci(trend_pred_full_per_fold)
            mr, lr, hr = mean_ci(trend_recon_full_per_fold)
            mt, lt, ht = mean_ci(trend_true_full_per_fold)

            # write combined (fold-aggregated) per-var CSV
            row_comb = {
                "model": model_label, "var": var,
                "trend_pred_mean": mp, "trend_pred_lo95": lp, "trend_pred_hi95": hp,
                "trend_recon_mean": mr, "trend_recon_lo95": lr, "trend_recon_hi95": hr,
                "trend_true_mean": mt, "trend_true_lo95": lt, "trend_true_hi95": ht,
                "trend_pred_on_muMean": trend_pred_all,
                "trend_recon_on_muMean": trend_recon_all,
                "trend_true_on_muMean": trend_true_all,
            }
            trend_dir = Path.cwd() / "results" / "trends_full_period"
            trend_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([row_comb]).to_csv(trend_dir / f"trends_full_{model_label}_{var}.csv", index=False)

            # Optional: violin of trend errors per var (NN-True and SST-True)
            violin_trend_errors_per_var(
                model_label=model_label, var=var,
                trend_nn_all=trend_pred_full_per_fold,
                trend_sst_all=trend_recon_full_per_fold,
                trend_sim_all=trend_true_full_per_fold,
                out_path=(Path.cwd() / "examples_of_reconstructions" / date_str / model_label / var /
                          f"trend_violin_{model_label}_{var}.png")
            )

            # metrics row
            per_model_rows.append({
                "model": model_label,
                "var": var,
                "r2": round(r2, 4), "corr": round(corr, 4),
                "std_pred": round(std_pred, 3),
                "std_true": round(std_true, 3),
            })

            # For the grand table
            grand_trend_rows.append({
                "model": test_model, "label": model_label, "var": var,
                "trend_pred_on_muMean": trend_pred_all,
                "trend_recon_on_muMean": trend_recon_all,
                "trend_true_on_muMean": trend_true_all
            })

        # Save per-model metrics CSV
        if per_model_rows:
            dfm = pd.DataFrame(per_model_rows)
            out_file = Path.cwd() / "results" / f"infer_TEST_CMIP5_{test_model}_{cfg.save_name}_{scenario}.csv"
            out_file.parent.mkdir(parents=True, exist_ok=True)
            dfm.to_csv(out_file, index=False)
            all_results_rows.extend(per_model_rows)

        # ALL-VARS combined scatter & trends
        if len(allvars_mu_series):
            mu_cat   = np.concatenate(allvars_mu_series, axis=0)
            years_cat= np.concatenate(allvars_years_series, axis=0)
            if (USE_SST_INDEX and (s_year_test is not None) and (s_val_test is not None)):
                ry, rv = sst_to_amoc_recon(s_year_test, s_val_test)
                yc, muc, rc = align_by_years(years_cat, mu_cat, ry, rv)
                if yc.size:
                    out_all = Path.cwd() / "examples_of_reconstructions" / date_str / model_label
                    plot_scatter_recon_vs_pred(
                        years_pred=yc, mu_pred=muc,
                        years_recon=yc, recon_vals=rc,
                        out_path=out_all / f"scatter_recon_vs_pred_ALLVARS_{model_label}.png",
                        title=f"{model_label} — ALL variables"
                    )
                    # Combined trend on ALL-VARS
                    t_pred = linear_trend_all_period(yc, muc)
                    t_reco = linear_trend_all_period(yc, rc)
                    df_tr = pd.DataFrame([{
                        "model": model_label,
                        "vars": "ALL",
                        "trend_pred_allvars_Sv_per_century": round(t_pred, 4) if np.isfinite(t_pred) else np.nan,
                        "trend_recon_allvars_Sv_per_century": round(t_reco, 4) if np.isfinite(t_reco) else np.nan,
                    }])
                    out_tr = Path.cwd() / "results" / "trends_full_period"
                    out_tr.mkdir(parents=True, exist_ok=True)
                    df_tr.to_csv(out_tr / f"trends_full_ALLVARS_{model_label}.csv", index=False)

        # ---------- Combined MEMBER plots (all vars + SST index) ----------
        if (member_ids_global is not None) and (tgt_ref_global is not None) and len(mu_by_var):
            save_members_combined_plots_from_arrays(
                mu_by_var=mu_by_var,
                sig_by_var=sig_by_var,
                tgt_ref=tgt_ref_global,
                member_ids=member_ids_global,
                start_year=START_YEAR,
                x_vars=x_vars,
                out_root=Path.cwd() / "examples_of_reconstructions" / date_str,
                model_label=model_label,
                sst_member_series=idx_map_test if USE_SST_INDEX else None,
            )
        else:
            print(f"[warn] Skipping members_combined for {model_label} (missing member_ids or tgt_ref)")

    # merge *all* CMIP5 rows (metrics) into one CSV too
    if all_results_rows:
        df_all = pd.DataFrame(all_results_rows)
        out_file = Path.cwd() / "results" / f"infer_TEST_CMIP5_ALL_{cfg.save_name}_{scenario}.csv"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        df_all.to_csv(out_file, index=False)

    # Grand table of trends across all models/vars
    if grand_trend_rows:
        grand = pd.DataFrame(grand_trend_rows)
        outg = Path.cwd() / "results" / "trends_full_period" / f"trends_full_ALL_MODELS_{cfg.save_name}_{scenario}.csv"
        outg.parent.mkdir(parents=True, exist_ok=True)
        grand.to_csv(outg, index=False)

# ---------------- main ----------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Config
    cfg = OmegaConf.load("conf/config_bnn.yaml")

    # Paths & core config
    base_dir = Path("/home/am334/link_am334/moc_mmodel/monthly_stream_zarr_cmip5")
    save_name   = cfg.save_name
    val_models  = _as_list(getattr(cfg, "val_model", None)) or _as_list(getattr(cfg, "val_models", None))
    x_vars      = _as_list(cfg.x_vars)
    scenario    = str(cfg.val_scenario)
    target_var  = str(cfg.target_var)
    output_type = str(cfg.output_type)
    selected_lats = [float(l) for l in _as_list(cfg.selected_lats)]
    lpf = cfg.lpf

    add_coords = bool(getattr(cfg, "data", {}).get("add_coords", True))
    use_sin_cos = bool(getattr(cfg, "data", {}).get("use_sin_cos", True))
    lat_values = getattr(cfg, "data", {}).get("lat_values", None)
    lon_values = getattr(cfg, "data", {}).get("lon_values", None)

    cmip5_list = _as_list(getattr(cfg, "cmip5_test_models", []))
    if not cmip5_list:
        raise ValueError("cfg.cmip5_test_models is empty. Provide CMIP5 models to evaluate.")

    aw = ArtifactWriter(root=Path.cwd() / "artifacts", save_name=save_name,
                        scenario=scenario, cfg=cfg)

    evaluate_cmip5_test_across_folds(
        cmip5_models=cmip5_list,
        val_models=val_models,
        x_vars=x_vars,
        cfg=cfg,
        base_dir=base_dir,
        device=device,
        selected_lats=selected_lats,
        scenario=scenario,
        target_var=target_var,
        output_type=output_type,
        add_coords=add_coords,
        use_sincos=use_sin_cos,
        lat_values=lat_values,
        lon_values=lon_values,
        lpf=lpf,
        aw=aw,
    )

    aw.flush()
    print(f"Artifacts saved to: {aw.dir}")
    print("Done.")

if __name__ == "__main__":
    main()
