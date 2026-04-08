#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ConvNeXt+LN (two-stage): inference that mixes
  μ  ← Stage-1 (best_stage1_mu.pt)
  σ  ← Stage-2 joint (best_stage2_joint.pt)
with lat/lon coordinate channels from config.

Outputs (per variable & held-out model):
  - reconstruction.png, memberwise.png, members/member_*.png
  - trend_hist_{model}.png, trend_violin_{model}.png
  - results CSVs + ALL-MODELS trend plots
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from datetime import datetime
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from scipy.stats import norm

# Optional violin plots
try:
    import seaborn as sns
    _HAVE_SNS = True
except Exception:
    _HAVE_SNS = False

# Your datasets
from dataset_for_cesm2_LE import PreprocessedCMIP6Dataset_LE
from datasets_cmip_zarr import PreprocessedCMIP6Dataset  # only for LAST_30_YEARS branch if you use it

from omegaconf import OmegaConf
from train_weights_BNN import ConvNeXtHet
from models import ConvNeXtRegressorMu, ResidualCNNHetBaseline

torch.set_default_dtype(torch.float32)
torch.set_num_threads(8)
torch.set_num_interop_threads(8)

# ------------------------------ utils ------------------------------



from pathlib import Path
import json
import numpy as np
import pandas as pd

class ArtifactWriter:
    """
    Persist artifacts for later plotting. Uses Parquet if available, else CSV.
    Creates:
      - series.parquet          # per-row: model, var, member, year, mu, sig, tgt, ...
      - trends.parquet          # per-row: model, var, member, trend_nn, trend_sim, trend_sst, start, end, trend_label
      - metrics.parquet         # per-row: model, var, r2, corr, std_pred, std_true, crps
      - coverage.parquet        # per-row: model, var, nominal, coverage, n_used
      - pit/{model}.npz         # PIT u array per model (optional)
      - manifest.json           # quick context of run
    """
    def __init__(self, root: Path, save_name: str, scenario: str, cfg):
        date_str = datetime.now().strftime("%Y-%m-%d")
        self.dir = Path(root) / date_str / save_name / scenario
        self.dir.mkdir(parents=True, exist_ok=True)
        self._series_rows   = []
        self._trends_rows   = []
        self._metrics_rows  = []
        self._coverage_rows = []
        # minimal manifest
        man = {
            "date": date_str,
            "save_name": save_name,
            "scenario": str(scenario),
            "target_var": str(getattr(cfg, "target_var", "")),
            "output_type": str(getattr(cfg, "output_type", "")),
            "selected_lats": [float(x) for x in _as_list(getattr(cfg, "selected_lats", []))],
            "lpf": str(getattr(cfg, "lpf", "")),
        }
        (self.dir / "manifest.json").write_text(json.dumps(man, indent=2))

    # ---------- adders ----------
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

    def add_member_trend(self, *, model: str, var: str, member: str,
                         trend_nn: float, trend_sim: float, trend_sst: float,
                         start: int, end: int):
        self._trends_rows.append(pd.DataFrame([{
            "model": model, "var": var, "member": str(member),
            "trend_nn": float(trend_nn), "trend_sim": float(trend_sim), "trend_sst": float(trend_sst),
            "trend_start": int(start), "trend_end": int(end),
            "trend_label": f"{start}-{end}",
        }]))

    def add_model_metrics(self, *, model: str, var: str, r2: float, corr: float,
                          std_pred: float, std_true: float, crps: float):
        self._metrics_rows.append(pd.DataFrame([{
            "model": model, "var": var, "r2": float(r2), "corr": float(corr),
            "std_pred": float(std_pred), "std_true": float(std_true), "crps": float(crps),
        }]))

    def add_coverage_curve(self, *, model: str, var: str,
                           nominals: np.ndarray, coverage: np.ndarray, n_used: int):
        nom = np.asarray(nominals).ravel()
        cov = np.asarray(coverage).ravel()
        n = int(min(nom.size, cov.size))
        if n == 0: return
        df = pd.DataFrame({
            "model": model, "var": var,
            "nominal": nom[:n], "coverage": cov[:n], "n_used": int(n_used),
        })
        self._coverage_rows.append(df)

    def save_pit(self, *, model: str, u: np.ndarray):
        pit_dir = self.dir / "pit"
        pit_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(pit_dir / f"{model}.npz", u=np.asarray(u).ravel())

    # ---------- flushers ----------
    def _save_table(self, name: str, df: pd.DataFrame):
        path_parquet = self.dir / f"{name}.parquet"
        path_csv     = self.dir / f"{name}.csv"
        try:
            df.to_parquet(path_parquet, index=False)  # requires pyarrow/fastparquet
        except Exception:
            df.to_csv(path_csv, index=False)

    def flush(self):
        if self._series_rows:
            self._save_table("series", pd.concat(self._series_rows, ignore_index=True))
        if self._trends_rows:
            self._save_table("trends", pd.concat(self._trends_rows, ignore_index=True))
        if self._metrics_rows:
            self._save_table("metrics", pd.concat(self._metrics_rows, ignore_index=True))
        if self._coverage_rows:
            self._save_table("coverage", pd.concat(self._coverage_rows, ignore_index=True))



def _as_list(x):
    if x is None: return []
    if isinstance(x, (list, tuple)): return list(x)
    if isinstance(x, str): return [s.strip() for s in x.split(",") if s.strip()]
    return [x]

# --------- coordinate channels (lat_norm + sin(lon) + cos(lon)) ----------

def make_coord_tensor_from_arrays(lat_vals, W, lon_vals=None, use_sin_cos=True):
    import torch as _torch
    lat_vals = np.asarray(lat_vals, dtype=np.float32)
    H = len(lat_vals)

    lat = _torch.from_numpy(lat_vals).view(1, H, 1).expand(1, H, W)
    lat_min = float(lat_vals.min()); lat_max = float(lat_vals.max())
    lat_norm = 2.0 * (lat - lat_min) / max(1e-6, (lat_max - lat_min)) - 1.0

    empty_like = (
        (lon_vals is None) or
        (isinstance(lon_vals, (list, tuple)) and len(lon_vals) == 0) or
        (isinstance(lon_vals, np.ndarray) and lon_vals.size == 0)
    )
    if empty_like:
        lon_vals = None
    else:
        lon_vals = np.asarray(lon_vals, dtype=np.float32)
        assert lon_vals.shape == (W,), f"lon_vals shape {lon_vals.shape} must be (W,) with W={W}"

    if lon_vals is None:
        delta = 360.0 / W
        lon_vals = (np.arange(W, dtype=np.float32) + 0.5) * delta - 180.0

    lon = _torch.from_numpy(lon_vals).view(1, 1, W).expand(1, H, W)
    if use_sin_cos:
        lon_rad = lon * (math.pi / 180.0)
        coord = _torch.cat([lat_norm, _torch.sin(lon_rad), _torch.cos(lon_rad)], dim=0)  # [3,H,W]
    else:
        lon_min = float(lon_vals.min()); lon_max = float(lon_vals.max())
        lon_norm = 2.0 * (lon - lon_min) / max(1e-6, (lon_max - lon_min)) - 1.0
        coord = _torch.cat([lat_norm, lon_norm], dim=0)  # [2,H,W]
    return coord

class WithCoordsFromArrays(torch.utils.data.Dataset):
    def __init__(self, base_ds, lat_vals, lon_vals=None, use_sin_cos=True):
        super().__init__()
        self.base = base_ds
        x0, *_ = base_ds[0]
        _, H, W = x0.shape

        lat_vals = np.asarray(lat_vals, dtype=np.float32)
        assert len(lat_vals) == H, f"len(lat_vals)={len(lat_vals)} must match H={H}"

        empty_like = (
            (lon_vals is None) or
            (isinstance(lon_vals, (list, tuple)) and len(lon_vals) == 0) or
            (isinstance(lon_vals, np.ndarray) and lon_vals.size == 0)
        )
        lon_vals_sane = None if empty_like else lon_vals

        self.coord = make_coord_tensor_from_arrays(
            lat_vals, W, lon_vals=lon_vals_sane, use_sin_cos=use_sin_cos
        )

    def __len__(self): return len(self.base)
    def __getitem__(self, i):
        item = self.base[i]
        x, y = item[:2]
        x = torch.cat([x, self.coord], dim=0)
        return (x, y, *item[2:])

# ----------------------- ConvNeXt + LayerNorm -----------------------


# ----------------------- metrics & SST overlay -----------------------

def compute_metrics(preds: np.ndarray, targets: np.ndarray):
    preds = preds.ravel(); targets = targets.ravel()
    ss_res = ((targets - preds) ** 2).sum()
    ss_tot = ((targets - targets.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    corr = np.corrcoef(preds, targets)[0, 1] if preds.size else np.nan
    return round(r2, 4), round(corr, 4)


def compute_r2_only(pred: np.ndarray, true: np.ndarray) -> float:
    pred = np.asarray(pred).ravel(); true = np.asarray(true).ravel()
    if pred.size == 0 or true.size == 0: return float('nan')
    ss_res = np.sum((true - pred)**2)
    ss_tot = np.sum((true - np.mean(true))**2)
    return float('nan') if ss_tot <= 0 else float(1.0 - ss_res/ss_tot)

def r2_sst_against_true(years_true: np.ndarray,
                        true_vals: np.ndarray,
                        sst_years: np.ndarray | None,
                        sst_vals:  np.ndarray | None) -> float:
    """R²( AMOC_from_SST , True ) on overlapping years."""
    if sst_years is None or sst_vals is None: return float('nan')
    yy, amoc_hat = sst_to_amoc_recon(sst_years, sst_vals)
    years_true = np.asarray(years_true, int)
    mask = np.isin(yy, years_true)
    if not np.any(mask): return float('nan')
    # align on years
    yy_i = yy[mask]
    # index truth by matching years:
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







# ---------------------------- PIT helpers ----------------------------
def _phi_erf(z):
    # standard normal CDF via erf; vectorized and stable
    return 0.5 * (1.0 + _erf_np(z / np.sqrt(2.0)))

def gaussian_pit(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Compute PIT values u = F(y) for a Gaussian predictive distribution N(mu, sigma^2)."""
    mu = np.asarray(mu).ravel()
    sigma = np.asarray(sigma).ravel()
    y = np.asarray(y).ravel()
    m = np.isfinite(mu) & np.isfinite(sigma) & np.isfinite(y)
    mu, sigma, y = mu[m], np.maximum(sigma[m], eps), y[m]
    z = (y - mu) / sigma
    return _phi_erf(z)

def plot_pit_hist_from_mu_sig(mu: np.ndarray, sig: np.ndarray, tgt: np.ndarray,
                              out_path: Path, title: str, bins: int = 20):
    """Plot a PIT histogram and save it."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    u = gaussian_pit(mu, sig, tgt)
    if u.size == 0:
        print(f"[warn] No PIT data for {out_path.name}"); return
    plt.figure(figsize=(6,4))
    plt.hist(u, bins=bins, range=(0,1), density=True, alpha=0.85, edgecolor='black')
    plt.axhline(1.0, linestyle='--', linewidth=1)  # uniform reference
    plt.xlabel("PIT value  u = F(y_obs)")
    plt.ylabel("Density")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()



from scipy.stats import norm

# ---------- binning ----------
def _make_bins(x, nbins=4, mode="quantile", edges=None):
    x = np.asarray(x)
    mask = np.isfinite(x)
    if edges is not None:
        edges = np.asarray(edges, float)
    elif mode == "quantile":
        qs = np.linspace(0, 1, nbins + 1)
        edges = np.quantile(x[mask], qs)
        # ensure strictly increasing
        edges = np.unique(edges)
        if edges.size < 2:  # fallback
            vmin, vmax = float(np.nanmin(x)), float(np.nanmax(x))
            edges = np.linspace(vmin, vmax, nbins + 1)
    else:  # equal width
        vmin, vmax = float(np.nanmin(x)), float(np.nanmax(x))
        edges = np.linspace(vmin, vmax, nbins + 1)
    return edges

def _bin_labels(edges, fmt="{:.2f}–{:.2f}"):
    labs = []
    for i in range(len(edges)-1):
        labs.append(fmt.format(edges[i], edges[i+1]))
    return labs

# ---------- coverage core ----------
def coverage_by_bins(mu, sig, y, x_cov, nominals=(0.5, 0.8, 0.9, 0.95),
                     nbins=4, mode="quantile", edges=None):
    """
    Returns dict with per-bin empirical coverage for central intervals at 'nominals'.
    """
    mu = np.asarray(mu).ravel()
    sig = np.maximum(np.asarray(sig).ravel(), 1e-12)
    y   = np.asarray(y).ravel()
    x   = np.asarray(x_cov).ravel()
    m = np.isfinite(mu) & np.isfinite(sig) & np.isfinite(y) & np.isfinite(x)
    mu, sig, y, x = mu[m], sig[m], y[m], x[m]
    if mu.size == 0:
        return {"edges": np.array([]), "labels": [], "nominals": np.array(nominals), "cover": np.empty((0,len(nominals)))}

    edges = _make_bins(x, nbins=nbins, mode=mode, edges=edges)
    labels = _bin_labels(edges, fmt="{:.2f}–{:.2f}" if np.issubdtype(x.dtype, np.number) else "{}–{}")

    cover = []
    for i in range(len(edges)-1):
        sel = (x >= edges[i]) & (x < edges[i+1]) if i < len(edges)-2 else (x >= edges[i]) & (x <= edges[i+1])
        if not np.any(sel):
            cover.append([np.nan] * len(nominals))
            continue
        mu_b, sig_b, y_b = mu[sel], sig[sel], y[sel]
        row = []
        for p in nominals:
            lo = norm.ppf((1 - p) / 2, loc=mu_b, scale=sig_b)
            hi = norm.ppf(1 - (1 - p) / 2, loc=mu_b, scale=sig_b)
            emp = np.mean((y_b >= lo) & (y_b <= hi))
            row.append(float(emp))
        cover.append(row)

    return {
        "edges": edges,
        "labels": labels,
        "nominals": np.array(nominals, float),
        "cover": np.array(cover, float),  # shape: (nbins, len(nominals))
        "counts": np.array([np.sum((x >= edges[i]) & (x < edges[i+1] if i < len(edges)-2 else x <= edges[i+1])) for i in range(len(edges)-1)])
    }

# ---------- plotting ----------
def plot_coverage_curves_by_bins(result, out_path: Path, title: str, annotate_counts=True):
    nom = result["nominals"]; cover = result["cover"]; labels = result["labels"]; counts = result["counts"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6.4, 5.2))
    # diagonal
    plt.plot([0,1], [0,1], 'k--', lw=1, label="Ideal")

    # one curve per bin
    for i in range(cover.shape[0]):
        lab = labels[i] if i < len(labels) else f"bin {i}"
        if annotate_counts:
            lab = f"{lab} (n={int(counts[i])})"
        plt.plot(nom, cover[i], marker='o', linewidth=1.8, label=lab)

    plt.xlim(0.3, 1.0); plt.ylim(0.3, 1.0)
    plt.xlabel("Nominal coverage")
    plt.ylabel("Empirical coverage")
    plt.title(title)
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()



from scipy.stats import norm

def coverage_nominal(mu, sig, y, nominals=(0.5, 0.8, 0.9, 0.95), return_count=False):
    """Empirical coverage for central intervals at the given nominal levels."""
    mu = np.asarray(mu).ravel()
    sig = np.maximum(np.asarray(sig).ravel(), 1e-12)
    y   = np.asarray(y).ravel()
    m = np.isfinite(mu) & np.isfinite(sig) & np.isfinite(y)
    mu, sig, y = mu[m], sig[m], y[m]
    cov = []
    for p in nominals:
        lo = norm.ppf((1-p)/2, loc=mu, scale=sig)
        hi = norm.ppf(1-(1-p)/2, loc=mu, scale=sig)
        cov.append(float(np.mean((y >= lo) & (y <= hi))))
    cov = np.array(cov, float)
    return (cov, mu.size) if return_count else cov

def plot_coverage_scatter(nominals, cover_list, labels, out_path: Path, title: str):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6.8, 5.2))
    plt.plot([0,1],[0,1],'k--',lw=1.0,label="Ideal")
    for cov, lab in zip(cover_list, labels):
        plt.plot(nominals, cov, marker='o', lw=1.8, label=lab)
    plt.xlim(0.45,0.98); plt.ylim(0.45,0.98)
    plt.xlabel("Nominal coverage"); plt.ylabel("Empirical coverage")
    plt.title(title); plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()

def plot_coverage_mean(nominals, cover_matrix, out_path: Path, title: str,
                       weights=None, ci_method="bootstrap", n_boot=2000, seed=0):
    """
    cover_matrix: array shape [n_models, n_levels]
    weights: optional per-model weights (e.g., #samples per model)
    """
    nominals = np.asarray(nominals, float)
    C = np.asarray(cover_matrix, float)  # [M, K]
    M, K = C.shape
    if weights is not None:
        w = np.asarray(weights, float)
        w = w / np.sum(w)
        mean = (w[:, None] * C).sum(axis=0)
    else:
        mean = C.mean(axis=0)

    # CI across models
    lo = hi = None
    if ci_method == "bootstrap" and M > 1:
        rng = np.random.default_rng(seed)
        boots = np.empty((n_boot, K), float)
        for b in range(n_boot):
            idx = rng.integers(0, M, size=M)
            if weights is None:
                boots[b] = C[idx].mean(axis=0)
            else:
                wb = w[idx]; wb = wb / wb.sum()
                boots[b] = (wb[:, None] * C[idx]).sum(axis=0)
        lo = np.percentile(boots, 2.5, axis=0)
        hi = np.percentile(boots, 97.5, axis=0)

    # plot
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6.8, 5.2))
    plt.plot([0,1],[0,1],'k--',lw=1.0,label="Ideal")
    if lo is not None and hi is not None:
        plt.fill_between(nominals, lo, hi, alpha=0.25, label="95% CI")
    plt.plot(nominals, mean, marker='o', lw=2.2, label="Mean across models")
    plt.xlim(0.45,0.98); plt.ylim(0.45,0.98)
    plt.xlabel("Nominal coverage"); plt.ylabel("Empirical coverage")
    plt.title(title); plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()






START_YEAR = 1855
def years_per_sample(member_ids, start_year=START_YEAR):
    member_ids = np.asarray(member_ids)
    years = np.empty(member_ids.shape[0], dtype=int)
    # assume samples for each member are contiguous (as in your dataset)
    uniq = []
    first_idx = {}
    for i, m in enumerate(member_ids):
        if m not in first_idx:
            first_idx[m] = len(uniq); uniq.append(m)
    # assign within-member year index
    for m in uniq:
        idx = np.where(member_ids == m)[0]
        years[idx] = start_year + np.arange(idx.size)
    return years




def _erf_np(x):
    s = np.sign(x); x = np.abs(x)
    t = 1.0/(1.0+0.3275911*x)
    a1,a2,a3,a4,a5 = 0.254829592,-0.284496736,1.421413741,-1.453152027,1.061405429
    y = 1.0 - (((((a5*t+a4)*t)+a3)*t+a2)*t+a1)*t*np.exp(-x*x)
    return s*y

def gaussian_crps(mu, sigma, y, eps=1e-8):
    sigma = np.maximum(sigma, eps)
    z = (y - mu) / sigma
    Phi = 0.5*(1.0 + _erf_np(z/np.sqrt(2.0)))
    phi = (1.0/np.sqrt(2.0*np.pi))*np.exp(-0.5*z*z)
    return float(np.mean(sigma*( z*(2.0*Phi-1.0) + 2.0*phi - 1.0/np.sqrt(np.pi) )))

# AMOC′ overlay from SST′
START_YEAR = 1855
USE_SST_INDEX = True
SST_CSV = "/home/am334/link_am334/data/sst_amoc_fingerprint_spg_minus_global_DJFMAM.csv"
RECON_A_SV = 0.2648
RECON_B_SV_PER_C = 1.6057
SST_BASELINE_YEARS = 100

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

# ---------------------------- plotting ----------------------------

def plot_scatter_pred_vs_sim(mu_all: np.ndarray,
                             sig_all: np.ndarray,
                             tgt_all: np.ndarray,
                             out_path: Path,
                             title: str,
                             max_points: int | None = 40000):
    """Combined over all folds: y=μ vs x=truth, vertical error bars = ±2σ."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mu = np.asarray(mu_all).ravel()
    sg = np.asarray(sig_all).ravel()
    tg = np.asarray(tgt_all).ravel()
    # drop non-finite values
    m = np.isfinite(mu) & np.isfinite(sg) & np.isfinite(tg)
    mu, sg, tg = mu[m], sg[m], tg[m]
    if mu.size == 0:
        print(f"[warn] empty scatter for {out_path.name}"); return
    # subsample for readability
    if max_points and mu.size > max_points:
        idx = np.random.default_rng(0).choice(mu.size, size=max_points, replace=False)
        mu, sg, tg = mu[idx], sg[idx], tg[idx]

    r2 = compute_r2_only(mu, tg)

    plt.figure(figsize=(7.5, 7))
    # error bars (±2σ)
    plt.errorbar(tg, mu, yerr=2.0*sg, fmt='o', alpha=0.15, markersize=2, elinewidth=0.6, capsize=0)
    # y=x reference line
    lo = float(np.nanpercentile(np.concatenate([tg, mu]), 1))
    hi = float(np.nanpercentile(np.concatenate([tg, mu]), 99))
    plt.plot([-4, 4], [-4, 4], 'k--', lw=1.2, label='y=x')
    plt.xlabel("Simulated (Truth) [Sv]")
    plt.ylabel("Predicted μ [Sv]")
    plt.xlim(-4,4)
    plt.ylim(-4,4)
    plt.title(title)
    #plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left')
    ax = plt.gca()
    #annotate_r2(ax, r2_nn=r2, r2_sst=float('nan'))
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()




TREND_WINDOWS = [
    (1855, 2014),
    (1900, 1985),
    (1993, 2014),
    (1900, 2000),
]
TREND_START = TREND_WINDOWS[0][0]
TREND_END   = TREND_WINDOWS[0][1]

def linear_trend_per_century(years: np.ndarray, values: np.ndarray,
                             start: int = TREND_START, end: int = TREND_END) -> float:
    """
    Returns linear trend (per century, same units as 'values') computed
    only on the intersection of 'years' with [start, end]. NaN if <2 points.
    """
    years = np.asarray(years, int)

    #print(years)
    #print(start, end)
    values = np.asarray(values, float)
    mask = np.isfinite(values) & (years >= start) & (years <= end)
    if mask.sum() < 2:
        return np.nan
    y = years[mask].astype(float)
    v = values[mask]
    # slope (per year) * 100 -> per century
    return float(np.polyfit(y, v, 1)[0] * 100.0)





def plot_reconstruction(mu, sig, tgt, out_dir: Path, model_name: str, var: str,
                        s_year=None, s_val=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(mu))
    years_full = START_YEAR + x
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(x, tgt, label="Truth", lw=2, color="red")
    ax.plot(x, mu,  label="Prediction μ (Stage-1)", lw=2, color="blue")
    ax.fill_between(x, mu - 2*sig, mu + 2*sig, alpha=0.25, label="±2σ (Stage-2)", color="blue")

    r2_nn = compute_r2_only(mu, tgt)
    r2_sst = r2_sst_against_true(years_full, tgt, s_year, s_val)

    if USE_SST_INDEX and (s_year is not None) and (s_val is not None):
        sx = s_year - START_YEAR
        _, amoc_hat = sst_to_amoc_recon(s_year, s_val)
        mask = (sx >= 0) & (sx < x.size)
        if np.any(mask):
            ax.plot(sx[mask], amoc_hat[mask], '--', lw=1.6, color='k', label="AMOĈ′ from SST′")

    annotate_r2(ax, r2_nn, r2_sst, fontsize=10)
    ax.set_title(f"{model_name} — {var}")
    ax.set_xlabel("Sample Index"); ax.set_ylabel("Sv")
    ax.grid(True, alpha=0.3); ax.legend(loc="upper left")
    file = out_dir / "reconstruction.png"
    fig.tight_layout(); fig.savefig(file, dpi=150); plt.close(fig)
    print(f"    → saved plot to {file}")

def plot_memberwise(mu, sig, tgt, member_ids, out_dir: Path, var: str, scenario: str, model_name: str,
                    sst_index_by_member: dict | None = None, s_year: np.ndarray | None = None, s_val: np.ndarray | None = None):
    years = np.arange(START_YEAR, START_YEAR + mu.shape[0])
    uniq = np.unique(member_ids)[:10]
    nrows, ncols = int(np.ceil(len(uniq) / 2)), 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 3.6 * nrows), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, m in enumerate(uniq):
        ax = axes[i]
        idx = np.where(member_ids == m)[0]
        yy = years[:idx.size]
        ax.fill_between(yy, mu[idx] - 2*sig[idx], mu[idx] + 2*sig[idx], alpha=0.25, label="±2σ", color="blue")
        ax.plot(yy, mu[idx],  'b-', lw=2, label='μ (Stage-1)')
        ax.plot(yy, tgt[idx], 'r-', lw=2, label='Truth')

        s_year_m = s_val_m = None
        if USE_SST_INDEX:
            key = str(m)
            if sst_index_by_member and (key in sst_index_by_member):
                s_year_m, s_val_m = sst_index_by_member[key]
            elif (s_year is not None) and (s_val is not None):
                s_year_m, s_val_m = s_year, s_val

        if (s_year_m is not None) and (s_val_m is not None):
            yys, amoc_hat = sst_to_amoc_recon(s_year_m, s_val_m)
            mask = (yys >= yy[0]) & (yys <= yy[-1])
            if np.any(mask):
                ax.plot(yys[mask], amoc_hat[mask], '--', lw=1.6, color='k', label="AMOĈ′ from SST′")

        r2_nn = compute_r2_only(mu[idx], tgt[idx])
        r2_sst = r2_sst_against_true(yy, tgt[idx], s_year_m, s_val_m)
        annotate_r2(ax, r2_nn, r2_sst, fontsize=9)

        ax.set_title(f"{model_name} — member {m}")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc='upper left')
        ax.set_xlabel('Year'); ax.set_ylabel('Sv')

    for ax in axes[len(uniq):]:
        ax.axis('off')

    fig.suptitle(f"Member-wise — {var} — {model_name}/{scenario}", fontsize=16)
    fig.text(0.5, 0.04, 'Year', ha='center')
    fig.text(0.06, 0.5, 'Sv', va='center', rotation='vertical')
    fig.tight_layout(rect=[0.06, 0.06, 1, 0.95])
    out_dir.mkdir(parents=True, exist_ok=True)
    file = out_dir / "memberwise.png"
    fig.savefig(file, dpi=150)
    plt.close(fig)
    print(f"    → saved plot to {file}")


def plot_member_individual(mu, sig, tgt, member_ids, out_dir: Path, var: str, scenario: str, model_name: str,
                           sst_index_by_member: dict | None = None, s_year: np.ndarray | None = None, s_val: np.ndarray | None = None):
    years_full = np.arange(START_YEAR, START_YEAR + mu.shape[0])
    members_dir = out_dir / "members"
    members_dir.mkdir(parents=True, exist_ok=True)

    uniq = np.unique(member_ids)
    saved = 0
    for m in uniq:
        idx = np.where(member_ids == m)[0]
        if idx.size == 0:
            continue
        years_m = years_full[:idx.size]

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.fill_between((years_m - 1855)/6 + 1855, mu[idx] - 2 * sig[idx], mu[idx] + 2 * sig[idx], alpha=0.25, label="±2σ (Stage-2)")
        ax.plot((years_m - 1855)/6 + 1855, mu[idx],  'b-', lw=2, label='NN μ (Stage-1)')
        ax.plot((years_m - 1855)/6 + 1855, tgt[idx], 'r-', lw=2, label='Truth')


        

        s_year_m = s_val_m = None
        if USE_SST_INDEX:
            key = str(m)
            if sst_index_by_member and (key in sst_index_by_member):
                s_year_m, s_val_m = sst_index_by_member[key]
            elif (s_year is not None) and (s_val is not None):
                s_year_m, s_val_m = s_year, s_val

        if (s_year_m is not None) and (s_val_m is not None):
            yy, amoc_hat = sst_to_amoc_recon(s_year_m, s_val_m)
            mask = (yy >= years_m[0]) & (yy <= years_m[-1])
            #if np.any(mask):
                #ax.plot(yy[mask], amoc_hat[mask], '--', lw=1.6, color='k', label="AMOĈ′ from SST′")

        # R² annotation
        r2_nn = compute_r2_only(mu[idx], tgt[idx])
        r2_sst = np.inf #r2_sst_against_true(years_m, tgt[idx], s_year_m, s_val_m)
        annotate_r2(ax, r2_nn, r2_sst, fontsize=10)

        ax.set_title(f"{model_name}/{scenario} — {var} — member {m}")
        ax.set_xlabel("Year"); ax.set_ylabel("Sv")
        #ax.grid(False, alpha=0.3); #ax.legend(loc="upper left")

        fpath = members_dir / f"member_{str(m).replace('/', '_')}.png"
        fig.tight_layout(); fig.savefig(fpath, dpi=150); plt.close(fig)
        saved += 1

    print(f"    → saved {saved} per-member plots to {members_dir}")


# ---------------------------- inference ----------------------------

def get_member_ids(ds):
    return np.array([ds.get_sample_info(i)["member"] for i in range(len(ds))])

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

def build_mu_model(input_shape, out_dim, device, cfg):
    in_ch, H, W = input_shape
    margs = getattr(cfg, "model_args", {})
    dims   = tuple(margs.get("dims",   [48,96,192,384]))
    depths = tuple(margs.get("depths", [3,3,9,3]))
    dropout= float(margs.get("dropout", 0.3))
    use_fs = bool(margs.get("use_flat_skip", False))
    anti   = bool(margs.get("anti_alias", True))
    bsig   = float(margs.get("blur_sigma", 1.0))
    #model = ConvNeXtRegressorMu(
   #     in_ch=in_ch, out_dim=out_dim, flat_hw=(H,W),
   #     dims=dims, depths=depths, p_drop=dropout, use_flat_skip=use_fs,
   #     anti_alias=anti, blur_sigma=bsig
   # ).to(device)

    model = ResidualCNNHetBaseline(in_ch, out_dim, flat_hw=[144, 108], p_drop=0.5, tanh_scale=8.0).to(device)


    model.eval()
    return model



def plot_trend_scatter(true_tr, recon_tr, out_path: Path, title: str):
    """
    Scatter of true (simulated) linear trends vs reconstructed (NN μ) trends.
    true_tr, recon_tr: 1D arrays (per-member trends in Sv/century).
    Saves the figure and returns the R² (float).
    """
    t = np.asarray(true_tr, float).ravel()
    r = np.asarray(recon_tr, float).ravel()
    m = np.isfinite(t) & np.isfinite(r)
    t, r = t[m], r[m]
    if t.size < 2:
        print(f"[warn] not enough points for trend scatter: {out_path.name}")
        return np.nan

    # R²
    r2 = compute_r2_only(r, t)  # yhat=recon vs ytrue=true

    # Plot
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6.4, 6.0))
    plt.scatter(t, r, s=16, alpha=0.5, edgecolor="none")
    lo = float(np.nanpercentile(np.concatenate([t, r]), 1))
    hi = float(np.nanpercentile(np.concatenate([t, r]), 99))
    pad = 0.05 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    plt.plot([lo, hi], [lo, hi], 'k--', lw=1.2)
    plt.xlim(lo, hi); plt.ylim(lo, hi)
    plt.xlabel("True trend (Simulated) [Sv/century]")
    plt.ylabel("Reconstructed trend (NN μ) [Sv/century]")
    plt.title(title)
    ax = plt.gca()
    ax.text(0.02, 0.98, f"R² = {r2:.3f}", transform=ax.transAxes,
            va='top', ha='left', fontsize=11,
            bbox=dict(facecolor='white', alpha=0.75, edgecolor='none'))
    #plt.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    return r2



def build_het_model(input_shape, out_dim, device, cfg):
    in_ch, H, W = input_shape
    margs = getattr(cfg, "model_args", {})
    dims   = tuple(margs.get("dims",   [48,96,192,384]))
    depths = tuple(margs.get("depths", [3,3,9,3]))
    dropout= float(margs.get("dropout", 0.3))
    tanh_s = float(margs.get("sigma_tanh_scale", 8.0))
    use_fs = bool(margs.get("use_flat_skip", False))
    anti   = bool(margs.get("anti_alias", True))
    bsig   = float(margs.get("blur_sigma", 1.0))
    #model = ConvNeXtHet(
    #    in_ch=in_ch, out_dim=out_dim, flat_hw=(H,W),
    #    dims=dims, depths=depths, p_drop=dropout, tanh_scale=tanh_s,
    #    use_flat_skip=use_fs, anti_alias=anti, blur_sigma=bsig
    #).to(device)
    
    model = ResidualCNNHetBaseline(in_ch, out_dim, flat_hw=[144, 108], p_drop=0.5, tanh_scale=8.0).to(device)


    model.eval()
    return model

def safe_load_state(model, sd_raw):
    if isinstance(sd_raw, dict) and "state_dict" in sd_raw and isinstance(sd_raw["state_dict"], dict):
        sd_raw = sd_raw["state_dict"]
    msd = model.state_dict()
    filtered = {k: v for k, v in sd_raw.items() if k in msd and msd[k].shape == v.shape}
    model.load_state_dict(filtered, strict=False)
    ignored = sorted(set(sd_raw.keys()) - set(filtered.keys()))
    if ignored:
        print(f"    [info] Ignoring {len(ignored)} mismatched keys (e.g., {ignored[:5]})")

# ------------------------------ main ------------------------------


def evaluate_cesm2_test_across_folds(val_models: list[str],
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
                                     aw=None):
    """
    For each CV fold, runs the CESM2 test dataset, averages μ and σ across folds,
    and builds the same plots/violins/metrics. Writes under '.../test_model(CESM2)/...'
    """
    model_label = "test_model(CESM2)"
    date_str = datetime.now().strftime("%Y-%m-%d")
    _wr = str(getattr(cfg, "weights_root", "weights"))
    WEIGHTS_ROOT = Path.cwd() / _wr / cfg.save_name
    # --- CESM2 multi-panel accumulator (one panel per member) ---
    cesm2_panel = {}   # member_id (as string) -> {"series": {var: {"years","mu","sig","tgt","r2"}}}


    # SST indices for CESM2
    s_year_test, s_val_test = (None, None)
    idx_map_test = None
    if USE_SST_INDEX:
        try:
            s_year_test, s_val_test = load_sst_index(SST_CSV, "CESM2", scenario, member=None, lpf_str=lpf)
        except Exception as e:
            print(f"  [warn] CESM2 SST (global) failed: {e}")
        try:
            idx_map_test = load_sst_index_per_member(SST_CSV, "CESM2", scenario, lpf_str=lpf)
        except Exception as e:
            print(f"  [warn] CESM2 SST (per-member) failed: {e}")

    results_rows = []
    for var in x_vars:
        # Build test dataset
        try:
            ds_raw = PreprocessedCMIP6Dataset(
                zarr_dir=str(base_dir),
                models=["CESM2"], x_vars=[var],
                scenarios=[scenario], target_group=target_var, output_type=output_type,
                selected_lats=selected_lats, lpf=lpf, member_selection="all"
            )
        except Exception:
            # Fallback to LE dataset if constructor/signature differs
            ds_raw = PreprocessedCMIP6Dataset_LE(
                zarr_dir=str(base_dir),
                models=["CESM2"], x_vars=[var],
                scenarios=[scenario], target_group=target_var, output_type=output_type,
                selected_lats=selected_lats, lpf=lpf, member_selection="all"
            )

        if len(ds_raw) == 0:
            print(f"[warn] CESM2 test dataset empty for var={var}")
            continue

        # coords
        if add_coords:
            if (lat_values is None) or (len(_as_list(lat_values)) == 0):
                if hasattr(ds_raw, "lat"):
                    lat_vals = np.asarray(ds_raw.lat, dtype=np.float32)
                elif hasattr(ds_raw, "lats"):
                    lat_vals = np.asarray(ds_raw.lats, dtype=np.float32)
                else:
                    raise ValueError("Provide data.lat_values in config.")
            else:
                lat_vals = np.asarray(_as_list(lat_values), dtype=np.float32)
            lon_vals = lon_values
            ds = WithCoordsFromArrays(ds_raw, lat_vals=lat_vals[0], lon_vals=lon_vals, use_sin_cos=use_sincos)
        else:
            ds = ds_raw

        mids = None
        try:
            mids = get_member_ids(ds_raw)
        except Exception:
            pass

        # shapes
        x0, y0 = ds[0][:2]
        out_dim = int(np.prod(y0.shape))

        # Run every fold; accumulate μ and σ
        mu_folds, sig_folds = [], []
        tgt_ref = None

        for fold_name in val_models:
            wdir = WEIGHTS_ROOT / var / fold_name
            path_mu    = wdir / "best_stage1_mu.pt"
            path_joint = wdir / "best_stage2_joint.pt"

            mu_model  = build_mu_model(x0.shape, out_dim, device, cfg)
            het_model = build_het_model(x0.shape, out_dim, device, cfg)

            if path_mu.exists():
                sd_mu = torch.load(path_mu, map_location=device)
                safe_load_state(mu_model, sd_mu)
            if path_joint.exists():
                sd_joint = torch.load(path_joint, map_location=device)
                safe_load_state(het_model, sd_joint)

            # μ
            if path_mu.exists():
                mu, tgt = infer_mu_stage1(mu_model, ds, device, batch_size=int(getattr(cfg.training, "batch_size", 64)))
            else:
                loader = DataLoader(ds, batch_size=int(getattr(cfg.training, "batch_size", 64)),
                                    shuffle=False, num_workers=4, pin_memory=True)
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
                sig = infer_sigma_stage2(het_model, ds, device, batch_size=int(getattr(cfg.training, "batch_size", 64)))
            else:
                res_std = np.std(tgt - mu) if mu.size else 1.0
                sig = np.full_like(mu, max(res_std, 1e-6))

            mu_folds.append(mu.ravel())
            sig_folds.append(sig.ravel())
            if tgt_ref is None:
                tgt_ref = tgt.ravel()

        if len(mu_folds) == 0:
            print(f"[warn] No folds ran for CESM2 {var}")
            continue

        mu_mean  = np.mean(np.stack(mu_folds, axis=0), axis=0)
        sig_mean = np.mean(np.stack(sig_folds, axis=0), axis=0)    # mean σ across folds

        # metrics
        r2, corr = compute_metrics(mu_mean, tgt_ref)
        std_pred = float(np.std(mu_mean)); std_true = float(np.std(tgt_ref))
        crps = gaussian_crps(mu_mean, sig_mean, tgt_ref)
        print(f"[CESM2 TEST] {var}: R²={r2:.4f}, corr={corr:.4f}, std_pred={std_pred:.3f}, std_true={std_true:.3f}, CRPS={crps:.3f}")

        if aw is not None:
            aw.add_model_metrics(model=model_label, var=var,
                                 r2=r2, corr=corr,
                                 std_pred=std_pred, std_true=std_true, crps=crps)

        # plots
        out_dir = Path.cwd() / "examples_of_reconstructions" / date_str / model_label / var
        years_sst, vals_sst = (s_year_test, s_val_test) if USE_SST_INDEX else (None, None)
        plot_reconstruction(mu_mean, sig_mean, tgt_ref, out_dir, model_label, var, s_year=years_sst, s_val=vals_sst)

        if mids is None:
            try:
                mids = get_member_ids(ds_raw)
            except Exception:
                pass

        if mids is not None and mids.size == mu_mean.size:


            years_full = np.arange(START_YEAR, START_YEAR + mu_mean.shape[0])
            uniq_members = np.unique(mids)
            for m in uniq_members:
                idx = np.where(mids == m)[0]
                if idx.size == 0: 
                    continue
                years_m = years_full[:idx.size]
                r2_m = compute_r2_only(mu_mean[idx], tgt_ref[idx])

                pm = cesm2_panel.setdefault(str(m), {"series": {}, "sst": None})

                pm["series"][var] = {
                    "years": years_m.copy(),
                    "mu":    mu_mean[idx].copy(),
                    "sig":   sig_mean[idx].copy(),
                    "tgt":   tgt_ref[idx].copy(),
                    "r2":    float(r2_m),
                }

                if aw is not None:
                    aw.add_member_series(
                        model=model_label, var=var, member=str(m),
                        years=years_m, mu=mu_mean[idx], sig=sig_mean[idx],
                        tgt=tgt_ref[idx],
                        extra={"scenario": scenario, "lpf": str(lpf)},
                    )


                # Add SST->AMOC once per member (aligned)
                if pm["sst"] is None and USE_SST_INDEX:
                    if (idx_map_test is not None) and (str(m) in idx_map_test):
                        s_year_m, s_val_m = idx_map_test[str(m)]
                    else:
                        s_year_m, s_val_m = (s_year_test, s_val_test)
                    if (s_year_m is not None) and (s_val_m is not None):
                        yy, amoc_hat = sst_to_amoc_recon(s_year_m, s_val_m)
                        common_years = np.intersect1d(yy, years_m)
                        if common_years.size > 0:
                            mask_yy = np.isin(yy, common_years)
                            mask_y  = np.isin(years_m, common_years)
                            yhat_sst = amoc_hat[mask_yy]
                            ytrue    = tgt_ref[idx][mask_y]
                            nmin = min(yhat_sst.size, ytrue.size)
                            if nmin > 0:
                                r2_sst = compute_r2_only(yhat_sst[:nmin], ytrue[:nmin])
                                pm["sst"] = {
                                    "years": common_years[:nmin].copy(),
                                    "amoc":  yhat_sst[:nmin].copy(),
                                    "r2":    float(r2_sst),
                                }



            # ensemble members present → memberwise/individual plots + trend violins
            plot_memberwise(mu_mean, sig_mean, tgt_ref, mids, out_dir, var, scenario, model_label,
                            sst_index_by_member=idx_map_test, s_year=years_sst, s_val=vals_sst)
            plot_member_individual(mu_mean, sig_mean, tgt_ref, mids, out_dir, var, scenario, model_label,
                                   sst_index_by_member=idx_map_test, s_year=years_sst, s_val=vals_sst)

            years_full = np.arange(START_YEAR, START_YEAR + mu_mean.shape[0])
            uniq_members = np.unique(mids)
            model_slopes_sst, model_slopes_nn, model_slopes_sim = [], [], []
            for m in uniq_members:
                idx = np.where(mids == m)[0]
                if idx.size < 2: continue
                y_m = years_full[:idx.size]
                tr_nn  = np.polyfit(y_m, mu_mean[idx], 1)[0] * 100.0
                tr_sim = np.polyfit(y_m, tgt_ref[idx], 1)[0] * 100.0
                if idx_map_test is not None and str(m) in idx_map_test:
                    yy, amoc_hat = sst_to_amoc_recon(*idx_map_test[str(m)])
                    tr_sst = np.polyfit(yy, amoc_hat, 1)[0] * 100.0
                elif (years_sst is not None) and (vals_sst is not None):
                    yy, amoc_hat = sst_to_amoc_recon(years_sst, vals_sst)
                    tr_sst = np.polyfit(yy, amoc_hat, 1)[0] * 100.0
                else:
                    tr_sst = np.nan
                model_slopes_sst.append(tr_sst)
                model_slopes_nn.append(tr_nn)
                model_slopes_sim.append(tr_sim)

            model_out_dir = Path.cwd() / "examples_of_reconstructions" / date_str / model_label
            # small local helpers (same style as in main()):
            def _plot_hist_local(sst, nn, sim, title, out_path, bins=20):
                out_path.parent.mkdir(parents=True, exist_ok=True)
                plt.figure(figsize=(10, 6))
                vs = np.array([v for v in sst if np.isfinite(v)])
                vn = np.array([v for v in nn  if np.isfinite(v)])
                vi = np.array([v for v in sim if np.isfinite(v)])
                allv = np.concatenate([vs, vn, vi]) if (vs.size+vn.size+vi.size)>0 else np.array([])
                rng = (np.nanpercentile(allv, 2), np.nanpercentile(allv, 98)) if allv.size>0 else None
                plt.hist(vs, bins=bins, alpha=0.5, density=True, label="SST→AMOC′", range=rng)
                plt.hist(vn, bins=bins, alpha=0.5, density=True, label="NN μ (Stage-1)", range=rng)
                plt.hist(vi, bins=bins, alpha=0.5, density=True, label="Simulated",  range=rng)
                plt.xlabel("Linear trend [Sv/century]"); plt.ylabel("Density")
                plt.title(title); plt.grid(True, alpha=0.3); plt.legend(loc="upper right")
                plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()

            def _plot_violin_local(sst, nn, sim, title, out_path):
                out_path.parent.mkdir(parents=True, exist_ok=True)
                arr_sst = np.array([v for v in sst if np.isfinite(v)])
                arr_nn  = np.array([v for v in nn  if np.isfinite(v)])
                arr_sim = np.array([v for v in sim if np.isfinite(v)])
                minlen = min(len(arr_sst), len(arr_nn), len(arr_sim))
                if minlen == 0:
                    print("    [warn] Not enough data for violin:", out_path.name); return
                e_sst = arr_sst[:minlen] - arr_sim[:minlen]
                e_nn  = arr_nn[:minlen]  - arr_sim[:minlen]
                if _HAVE_SNS:
                    df = pd.DataFrame({"Source": np.repeat(["SST Index", "NN μ(Stage-1)"], repeats=minlen),
                                       "Error":  np.concatenate([e_sst, e_nn])})
                    plt.figure(figsize=(4,6))
                    plt.ylim(-2,1)
                    sns.violinplot(x="Source", y="Error", data=df, inner="box", cut=0)
                    plt.axhline(0, ls="--", c="k", lw=1)
                    plt.ylabel("Trend error vs Simulated [Sv/century]")
                    plt.title(title); plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()
                else:
                    plt.figure(figsize=(7,6))
                    plt.boxplot([e_sst, e_nn], labels=["SST−Sim", "NN−Sim"], showmeans=True)
                    plt.axhline(0, ls="--", c="k", lw=1)
                    plt.ylabel("Trend error vs Simulated [Sv/century]")
                    plt.title(title + " (boxplot fallback)")
                    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()

            _plot_hist_local(model_slopes_sst, model_slopes_nn, model_slopes_sim,
                             title=f"Linear trends — {model_label}",
                             out_path=model_out_dir / f"trend_hist_{model_label}.png", bins=20)
            _plot_violin_local(model_slopes_sst, model_slopes_nn, model_slopes_sim,
                               title=f"Trend errors — {model_label}",
                               out_path=model_out_dir / f"trend_violin_{model_label}.png")
        





                # --- NEW: Per-model scatter of trends (True vs NN) for each x_var ---
    for v in x_vars:
        sim_tr = [x for x in per_var_slopes[v]["sim"] if np.isfinite(x)]
        nn_tr  = [x for x in per_var_slopes[v]["nn"]  if np.isfinite(x)]
        
        nmin = min(len(sim_tr), len(nn_tr))
        if nmin < 2:
            print(f"    [warn] not enough trend pairs for scatter ({model_name}/{v})")
            row[f"r2_trend_{v}"] = np.nan
            continue
        sim_tr = np.asarray(sim_tr[:nmin])
        nn_tr  = np.asarray(nn_tr[:nmin])
        r2_tr = plot_trend_scatter(
            sim_tr, nn_tr,
            out_path=model_out_dir / f"trend_scatter_true_vs_nn_{model_name}_{v}.png",
            title=f"True vs NN Trend — {model_name} — {v}"
        )
        row[f"r2_trend_{v}"] = float(r2_tr)



        # append CESM2 metrics to results_rows
        res_dir = Path.cwd() / "results"; res_dir.mkdir(parents=True, exist_ok=True)
        row = {
            "model": model_label, "var": var, "r2": r2, "corr": corr,
            "std_pred": round(std_pred, 3), "std_true": round(std_true, 3),
            "crps": round(crps, 3)
        }
        results_rows.append(row)

    
    
    
    

        # --- CESM2: multi-panel (one panel per member), Truth vs NN μ(+/−2σ) for each var ---
    if len(cesm2_panel) > 0:
        import math
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch

        members_list = list(cesm2_panel.keys())
        n_members = len(members_list)
        ncols = 2 if n_members <= 6 else 4
        nrows = int(math.ceil(n_members / ncols))

        # Consistent colors per variable across panels
        base_colors = plt.rcParams.get("axes.prop_cycle", None)
        color_list = (base_colors.by_key().get("color", ["C0","C1","C2","C3","C4"])
                      if base_colors is not None else ["C0","C1","C2","C3","C4"])
        var_colors = {v: color_list[i % len(color_list)] for i, v in enumerate(x_vars)}

        fig, axes = plt.subplots(nrows, ncols, figsize=(6.0*ncols, 3.2*nrows), sharex=False, sharey=False)
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])
        axes = axes.flatten()

        # Leave room on the right for legend
        fig.subplots_adjust(right=0.82)

        # Legend proxies (shared)



        vars_dict = {"tos" : "SST", "zos_minus_basin_mean" : "SSH"}



        legend_handles = [Line2D([0], [0], color="k", lw=2, label="Truth")]
        legend_handles += [Line2D([0], [0], color=var_colors[v], lw=2, label=f"NN μ ({vars_dict[v]})") for v in x_vars]
        legend_handles.append(Patch(facecolor="gray", alpha=0.18, label="±2σ"))

        for ax, mem in zip(axes, members_list):
            ser = cesm2_panel[mem]["series"]

            # Choose target series (first available var)
            tgt_years = None; tgt_vals = None
            for v in x_vars:
                if v in ser:
                    tgt_years = ser[v]["years"]; tgt_vals = ser[v]["tgt"]; break

            if tgt_years is None or tgt_vals is None or tgt_vals.size == 0:
                ax.set_title(f"CESM2 member {mem} — (no data)"); ax.axis('off'); continue

            # Plot Truth
            ax.plot(tgt_years, tgt_vals, color="k", lw=2, label="Truth")

            # Plot NN μ and ±2σ for each var present
            lines_r2 = []
            for v in x_vars:
                if v not in ser:
                    continue
                y = ser[v]["years"]; mu_v = ser[v]["mu"]; sig_v = ser[v]["sig"]
                ax.plot(y, mu_v, lw=2, color=var_colors[v], label=f"NN μ ({vars_dict[v]})")
                if sig_v.size == mu_v.size:
                    ax.fill_between(y, mu_v - 2.0*sig_v, mu_v + 2.0*sig_v, alpha=0.18, color=var_colors[v], linewidth=0)
                lines_r2.append(f"{vars_dict[v]}: R²={ser[v]['r2']:.3f}")

            # R² annotation
            if lines_r2:
                ax.text(0.02, 0.95, "\n".join(lines_r2),
                        transform=ax.transAxes, va='top', ha='left',
                        fontsize=10, bbox=dict(facecolor='white', alpha=0.75, edgecolor='none'))

            ax.set_title(f"CESM2 — member {mem}", fontsize=12)
            ax.set_xlabel("Year"); ax.set_ylabel("Sv")
            ax.grid(True, alpha=0.25)

        # Turn off unused axes
        for ax in axes[n_members:]:
            ax.axis('off')

        # Shared legend (inside figure canvas)
        fig_legend = fig.legend(
            handles=legend_handles,
            loc="center left",
            bbox_to_anchor=(0.87, 0.5),
            bbox_transform=fig.transFigure,
            frameon=True,
            fontsize=11,
        )





        fig.suptitle("CESM2 Test — NN Reconstructions by Member (Truth vs NN μ ±2σ)", fontsize=14, y=0.995)

        plt.tight_layout()
        fig.subplots_adjust(right=0.82)  # keep space for legend after tight_layout
        out_path = Path.cwd() / "examples_of_reconstructions" / date_str / "test_model(CESM2)" / "first_members_multi_panel_CESM2.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.25)
        plt.close(fig)
        print(f"Saved CESM2 multi-panel figure: {out_path}")







        # --- CESM2: one combined plot per member (all vars ±2σ + SST index) ---
    if len(cesm2_panel) > 0:
        # consistent colors
        base_colors = plt.rcParams.get("axes.prop_cycle", None)
        color_list  = (base_colors.by_key().get("color", ["C0","C1","C2","C3","C4"])
                       if base_colors is not None else ["C0","C1","C2","C3","C4"])
        var_colors = {v: color_list[i % len(color_list)] for i, v in enumerate(x_vars)}

        memb_out = Path.cwd() / "examples_of_reconstructions" / date_str / model_label / "members_combined"
        memb_out.mkdir(parents=True, exist_ok=True)

        for mem, payload in cesm2_panel.items():
            ser = payload["series"]
            if not ser:
                continue

            # pick truth from first available var
            tgt_years = tgt_vals = None
            for v in x_vars:
                if v in ser:
                    tgt_years = ser[v]["years"]; tgt_vals = ser[v]["tgt"]; break
            if tgt_years is None or tgt_vals is None or tgt_vals.size == 0:
                continue

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(tgt_years, tgt_vals, color="k", lw=2, label="Truth")

            r2_lines = []
            for v in x_vars:
                if v not in ser:
                    continue
                y = ser[v]["years"]; mu_v = ser[v]["mu"]; sig_v = ser[v]["sig"]
                ax.plot(y, mu_v, lw=2, color=var_colors[v], label=f"NN μ ({vars_dict[v]})")
                if sig_v.size == mu_v.size:
                    ax.fill_between(y, mu_v - 2.0*sig_v, mu_v + 2.0*sig_v, alpha=0.18, color=var_colors[v], linewidth=0)
                r2_lines.append(f"{vars_dict[v]}: R²={ser[v]['r2']:.3f}")

            # SST index (if available)
            if payload.get("sst"):
                ss = payload["sst"]
                ax.plot(ss["years"], ss["amoc"], ls="--", lw=1.8, color="dimgray", label="SST index → AMOC")
                r2_lines.append(f"SST index: R²={ss['r2']:.3f}")

            #if r2_lines:
            #    ax.text(0.01, 0.90, "\n".join(r2_lines),
            #            transform=ax.transAxes, va='top', ha='left',
            #            fontsize=10, bbox=dict(facecolor='white', alpha=0.75, edgecolor='none'))

            ax.set_title(f"CESM2 — member {mem}", fontsize=12)
            ax.set_xlabel("Year"); ax.set_ylabel("Sv")
            #ax.grid(True, alpha=0.25)

            # legend outside (right)
            handles, labels = ax.get_legend_handles_labels()
            fig.subplots_adjust(right=0.82)
            fig.legend(handles, labels, loc="center left",
                       bbox_to_anchor=(0.83, 0.5), bbox_transform=fig.transFigure,
                       frameon=True, fontsize=10)

            fpath = memb_out / f"member_{str(mem).replace('/', '_')}_combined.png"
            fig.savefig(fpath, dpi=200, bbox_inches="tight", pad_inches=0.25)
            plt.close(fig)
            print(f"Saved CESM2 combined member plot: {fpath}")


    
    if results_rows:
        df = pd.DataFrame(results_rows)
        out_file = Path.cwd() / "results" / f"infer_TEST_CESM2_{cfg.save_name}_{scenario}.csv"
        df.to_csv(out_file, index=False)
        print("Saved CESM2 TEST results to", out_file)





def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Config and paths
    cfg = OmegaConf.load("conf/config_bnn.yaml")
    cwd = Path.cwd()
    base_dir = cwd / cfg.base_dir

    save_name   = cfg.save_name
    val_models  = _as_list(getattr(cfg, "val_model", None)) or _as_list(getattr(cfg, "val_models", None))
    x_vars      = _as_list(cfg.x_vars)
    scenario    = str(cfg.val_scenario)  #"piControl" #str(cfg.val_scenario) 
    target_var  = str(cfg.target_var)
    output_type = str(cfg.output_type)
    selected_lats = [float(l) for l in _as_list(cfg.selected_lats)]
    lpf = cfg.lpf

    aw = ArtifactWriter(root=Path.cwd() / "artifacts", save_name=save_name, scenario=scenario, cfg=cfg)



    # coords
    add_coords = bool(getattr(cfg, "data", {}).get("add_coords", True))
    use_sin_cos = bool(getattr(cfg, "data", {}).get("use_sin_cos", True))
    lat_values = getattr(cfg, "data", {}).get("lat_values", None)
    lon_values = getattr(cfg, "data", {}).get("lon_values", None)

    # Weight checkpoint root (same layout as training)
    weights_root = str(getattr(cfg, "weights_root", "weights"))
    WEIGHTS_ROOT = cwd / weights_root / save_name

    date_str = datetime.now().strftime("%Y-%m-%d")

    # Sample one member label for SST-index alignment
    dummy = PreprocessedCMIP6Dataset_LE(
        zarr_dir=str(base_dir),
        models=[val_models[0] if val_models else "CESM2"],
        x_vars=[x_vars[0]],
        scenarios=[scenario], target_group=target_var, output_type=output_type,
        selected_lats=selected_lats, lpf=lpf, member_selection="all"
    )
    first_member_label = dummy.get_sample_info(0)["member"] if len(dummy) > 0 else None

    # --- Aggregators ---
    # Trends pooled across all models
    all_slopes_sst, all_slopes_nn, all_slopes_sim = [], [], []
    # Per-fold / per-variable result rows
    results_rows = []
    # For scatter plots aggregated over all folds
    scatter_by_var = {v: {"mu": [], "sig": [], "tgt": []} for v in x_vars}
    scatter_all = {"mu": [], "sig": [], "tgt": []}


    all_models_cov_nominals = (0.5, 0.8, 0.9, 0.95)
    all_models_cover_curves = []   # list of arrays [K]
    all_models_labels       = []
    all_models_counts       = []   # total #samples used per model


    # --- GLOBAL (ALL MODELS) accumulators by variable ---
    all_per_var_slopes = {v: {"sst": [], "nn": [], "sim": []} for v in x_vars}
    all_per_var_cov_curves = {v: [] for v in x_vars}   # list of curves (one per model)
    all_per_var_cov_counts = {v: [] for v in x_vars}   # list of sample counts (one per model)





    # --- Overall R²: collect per-member R², average per-model, then average across models ---
    overall_r2_by_model = {}  # model_name -> { "SST index": [r2_member...], "NN μ (var)": [r2_member...] }




    # --- First-member-only (combine across models) R² accumulators ---
    first_member_accum = {"SST index": {"yhat": [], "ytrue": []}}
    for v in x_vars:
        first_member_accum[f"NN μ ({v})"] = {"yhat": [], "ytrue": []}

    # --- For the "first-member per model" multi-panel figure ---
    first_member_panel = {}  # model_name -> {"series": {var: {"years": arr, "mu": arr, "tgt": arr, "r2": float}}}



    # --- Loop over validation folds (val_models) ---
    for model_name in val_models:
        print(f"\nProcessing fold: {model_name}")
        row = {"model": model_name}

        # SST indices (global + per-member)
        s_year, s_val = (None, None)
        idx_map = None
        if USE_SST_INDEX:
            try:
                s_year, s_val = load_sst_index(SST_CSV, model_name, scenario, member=first_member_label, lpf_str=lpf)
            except Exception as e:
                print(f"  [warn] SST index (global) failed: {e}")
            try:
                idx_map = load_sst_index_per_member(SST_CSV, model_name, scenario, lpf_str=lpf)
            except Exception as e:
                print(f"  [warn] SST index (per-member) failed: {e}")

        # Trends for this fold
        model_slopes_sst, model_slopes_nn, model_slopes_sim = [], [], []

        pit_mu_chunks, pit_sig_chunks, pit_tgt_chunks = [], [], []


        # --- per-variable containers (for multi-var violin & coverage) ---
        per_var_slopes = {v: {"sst": [], "nn": [], "sim": []} for v in x_vars}
        per_var_cov_curves = {v: None for v in x_vars}
        per_var_cov_counts = {v: 0 for v in x_vars}


        # Init per-model containers for SST and each NN(var)
        overall_r2_by_model[model_name] = {"SST index": []}
        for v in x_vars:
            overall_r2_by_model[model_name][f"NN μ ({v})"] = []

        per_member_panel = {} 



        # --- Loop over input variables (x_vars) ---
        for var in x_vars:
            wdir = WEIGHTS_ROOT / var / model_name
            path_mu    = wdir / "best_stage1_mu.pt"
            path_joint = wdir / "best_stage2_joint.pt"#"best.pt"

            if not path_mu.exists() and not path_joint.exists():
                print(f"  [skip] Missing both weights for {var}: {wdir}")
                for sfx in ("r2","corr","std_pred","std_true"):
                    row[f"{sfx}_{var}"] = np.nan
                continue

            # Large-ensemble dataset (all members)
            ds_raw = PreprocessedCMIP6Dataset_LE(
                zarr_dir=str(base_dir),
                models=[model_name], x_vars=[var],
                scenarios=[scenario], target_group=target_var, output_type=output_type,
                selected_lats=selected_lats, lpf=lpf, member_selection="all"
            )
            if len(ds_raw) == 0:
                print(f"  [warn] Empty dataset for {var}")
                for sfx in ("r2","corr","std_pred","std_true"):
                    row[f"{sfx}_{var}"] = np.nan
                continue

            # Coordinate channels (lat_norm, sin/cos lon)
            if add_coords:
                if (lat_values is None) or (len(_as_list(lat_values)) == 0):
                    if hasattr(ds_raw, "lat"):
                        lat_vals = np.asarray(ds_raw.lat, dtype=np.float32)
                    elif hasattr(ds_raw, "lats"):
                        lat_vals = np.asarray(ds_raw.lats, dtype=np.float32)
                    else:
                        raise ValueError("Provide data.lat_values in config.")
                else:
                    lat_vals = np.asarray(_as_list(lat_values), dtype=np.float32)
                lon_vals = lon_values  # may be None or empty
                ds = WithCoordsFromArrays(ds_raw, lat_vals=lat_vals[0], lon_vals=lon_vals, use_sin_cos=use_sin_cos)
            else:
                ds = ds_raw

            mids = get_member_ids(ds_raw)

            # Shapes and dimensions
            x0, y0 = ds[0][:2]
            in_ch, H, W = x0.shape
            out_dim = int(np.prod(y0.shape))

            # Build models
            mu_model  = build_mu_model(x0.shape, out_dim, device, cfg)
            het_model = build_het_model(x0.shape, out_dim, device, cfg)

            # Load checkpoints
            if path_mu.exists():
                sd_mu = torch.load(path_mu, map_location=device)
                safe_load_state(mu_model, sd_mu)
                print(f"    [μ] loaded {path_mu}")
            else:
                print("    [μ] missing Stage-1; will use μ from joint as fallback")

            if path_joint.exists():
                sd_joint = torch.load(path_joint, map_location=device)
                safe_load_state(het_model, sd_joint)
                print(f"    [σ] loaded {path_joint}")
            else:
                print("    [σ] missing Stage-2 joint; will fabricate σ from residuals")

            # --- Inference ---
            bs = int(getattr(cfg.training, "batch_size", 64))

            # μ: Stage-1 if available, else μ from joint model
            if path_mu.exists():
                mu, tgt = infer_mu_stage1(mu_model, ds, device, batch_size=bs)
            else:
                loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)
                mu_list, tgt_list = [], []
                with torch.no_grad():
                    for batch in loader:
                        x, y = batch[:2]
                        x = x.to(device, dtype=torch.float32)
                        mu_b, _ = het_model(x)
                        mu_list.append(mu_b.cpu().numpy()); tgt_list.append(y.numpy())
                mu = np.concatenate(mu_list, 0); tgt = np.concatenate(tgt_list, 0)

            # σ: from joint, else constant from residual std
            if path_joint.exists():
                sig = infer_sigma_stage2(het_model, ds, device, batch_size=bs)
            else:
                res_std = np.std(tgt - mu) if mu.size else 1.0
                sig = np.full_like(mu, max(res_std, 1e-6))

            mu = mu.ravel(); sig = sig.ravel(); tgt = tgt.ravel()


            uniq_members_sorted = np.unique(mids)




            if uniq_members_sorted.size > 0:
                m_first = uniq_members_sorted[0]
                idx_first = np.where(mids == m_first)[0]
                if idx_first.size > 0:
                    # NN(var): take this member’s series
                    first_member_accum[f"NN μ ({var})"]["yhat"].append(mu[idx_first])
                    first_member_accum[f"NN μ ({var})"]["ytrue"].append(tgt[idx_first])






                            # --- Store series for the multi-panel "first members" figure ---
                    years_full = np.arange(START_YEAR, START_YEAR + mu.shape[0])
                    years_first = years_full[:idx_first.size]
                    r2_first = compute_r2_only(mu[idx_first], tgt[idx_first])

                    pm = first_member_panel.setdefault(model_name, {"series": {}})
                    pm["series"][var] = {
                        "years": years_first.copy(),
                        "mu":    mu[idx_first].copy(),
                        "tgt":   tgt[idx_first].copy(),
                        "sig":   sig[idx_first].copy(),   
                        "r2":    float(r2_first),
                    }


                    # SST index: build aligned SST→AMOC vs truth for this same member
                    # prefer per-member SST if available; else fall back to global
                    if idx_map is not None and str(m_first) in idx_map:
                        s_year_m, s_val_m = idx_map[str(m_first)]
                    else:
                        s_year_m, s_val_m = (s_year, s_val)

                    if (s_year_m is not None) and (s_val_m is not None):
                        yy, amoc_hat = sst_to_amoc_recon(s_year_m, s_val_m)
                        years_first = np.arange(START_YEAR, START_YEAR + mu.shape[0])[:idx_first.size]
                        # align by common years
                        common_years = np.intersect1d(yy, years_first)
                        if common_years.size > 0:
                            mask_yy = np.isin(yy, common_years)
                            mask_y  = np.isin(years_first, common_years)
                            yhat_sst = amoc_hat[mask_yy]
                            ytrue_sst = tgt[idx_first][mask_y]
                            # guard equal lengths
                            nmin = min(yhat_sst.size, ytrue_sst.size)
                            if nmin > 0:
                                first_member_accum["SST index"]["yhat"].append(yhat_sst[:nmin])
                                first_member_accum["SST index"]["ytrue"].append(ytrue_sst[:nmin])


            pit_mu_chunks.append(mu)
            pit_sig_chunks.append(sig)
            pit_tgt_chunks.append(tgt)


            # Metrics
            r2, corr = compute_metrics(mu, tgt)
            std_pred = float(np.std(mu)); std_true = float(np.std(tgt))
            crps = gaussian_crps(mu, sig, tgt)
            print(f"  {var}: R²={r2:.4f}, corr={corr:.4f}, std_pred={std_pred:.3f}, std_true={std_true:.3f}, CRPS={crps:.3f}")

            row[f"r2_{var}"] = r2
            row[f"corr_{var}"] = corr
            row[f"std_pred_{var}"] = round(std_pred, 3)
            row[f"std_true_{var}"] = round(std_true, 3)


            # NEW: save per-model metrics
            aw.add_model_metrics(model=model_name, var=var, r2=r2, corr=corr,
                                 std_pred=std_pred, std_true=std_true, crps=crps)



            # --- Plots for this variable and fold ---
            out_dir = Path.cwd() / "examples_of_reconstructions" / date_str / model_name / var
            years_sst, vals_sst = (s_year, s_val) if USE_SST_INDEX else (None, None)
            plot_reconstruction(mu, sig, tgt, out_dir, model_name, var, s_year=years_sst, s_val=vals_sst)
            plot_memberwise(mu, sig, tgt, mids, out_dir, var, scenario, model_name,
                            sst_index_by_member=idx_map, s_year=years_sst, s_val=vals_sst)
            plot_member_individual(mu, sig, tgt, mids, out_dir, var, scenario, model_name,
                                   sst_index_by_member=idx_map, s_year=years_sst, s_val=vals_sst)

            # --- Per-member ensemble trends ---
            #years_full = np.arange(START_YEAR, START_YEAR + mu.shape[0])
            #uniq_members = np.unique(mids)


            nn_per_member, sim_per_member, sst_per_member = [], [], []


            '''
            for m in uniq_members:
                idx = np.where(mids == m)[0]
                if idx.size < 2:
                    continue
                y_m = years_full[:idx.size]
                tr_nn  = np.polyfit(y_m, mu[idx], 1)[0] * 100.0
                tr_sim = np.polyfit(y_m, tgt[idx], 1)[0] * 100.0
                # SST→AMOC′ trend
                if idx_map is not None and str(m) in idx_map:
                    yy, amoc_hat = sst_to_amoc_recon(*idx_map[str(m)])
                    tr_sst = np.polyfit(yy, amoc_hat, 1)[0] * 100.0
                elif (s_year is not None) and (s_val is not None):
                    yy, amoc_hat = sst_to_amoc_recon(s_year, s_val)
                    tr_sst = np.polyfit(yy, amoc_hat, 1)[0] * 100.0
                else:
                    tr_sst = np.nan

            '''

            #years_all = years_per_sample(mids, START_YEAR)
            #uniq_members = np.unique(mids)
            years_full = np.arange(START_YEAR, START_YEAR + mu.shape[0])   
            uniq_members = np.unique(mids)

            for m in uniq_members:
                idx = np.where(mids == m)[0]
                if idx.size < 2:
                    continue

                y_m = years_full[:idx.size]

                # windowed trends (default window for plots)
                tr_nn  = linear_trend_per_century(y_m, mu[idx])
                tr_sim = linear_trend_per_century(y_m, tgt[idx])

                # SST→AMOC′ trend (prefer per-member series)
                if idx_map is not None and str(m) in idx_map:
                    yy, amoc_hat = sst_to_amoc_recon(*idx_map[str(m)])
                    tr_sst = linear_trend_per_century(yy, amoc_hat)
                elif (s_year is not None) and (s_val is not None):
                    yy, amoc_hat = sst_to_amoc_recon(s_year, s_val)
                    tr_sst = linear_trend_per_century(yy, amoc_hat)
                else:
                    tr_sst = np.nan

                # series
                aw.add_member_series(
                    model=model_name, var=var, member=str(m),
                    years=y_m, mu=mu[idx], sig=sig[idx], tgt=tgt[idx],
                    extra={"scenario": scenario, "lpf": str(lpf)}
                )

                # trends for ALL windows → artifact
                for _tw_start, _tw_end in TREND_WINDOWS:
                    _tr_nn  = linear_trend_per_century(y_m, mu[idx],  start=_tw_start, end=_tw_end)
                    _tr_sim = linear_trend_per_century(y_m, tgt[idx], start=_tw_start, end=_tw_end)
                    if idx_map is not None and str(m) in idx_map:
                        _yy, _ah = sst_to_amoc_recon(*idx_map[str(m)])
                        _tr_sst = linear_trend_per_century(_yy, _ah, start=_tw_start, end=_tw_end)
                    elif (s_year is not None) and (s_val is not None):
                        _yy, _ah = sst_to_amoc_recon(s_year, s_val)
                        _tr_sst = linear_trend_per_century(_yy, _ah, start=_tw_start, end=_tw_end)
                    else:
                        _tr_sst = np.nan
                    aw.add_member_trend(
                        model=model_name, var=var, member=str(m),
                        trend_nn=_tr_nn, trend_sim=_tr_sim, trend_sst=_tr_sst,
                        start=_tw_start, end=_tw_end,
                    )

                model_slopes_sst.append(tr_sst)
                model_slopes_nn.append(tr_nn)
                model_slopes_sim.append(tr_sim)


                                # --- Member-level R² collection ---
                # 1) NN(var) vs truth (member)
                r2_nn_member = compute_r2_only(mu[idx], tgt[idx])
                if np.isfinite(r2_nn_member):
                    overall_r2_by_model[model_name][f"NN μ ({var})"].append(r2_nn_member)

                # 2) SST index → AMOC vs truth (member), aligned in time
                #    Prefer per-member SST series if available; fall back to global series
                if idx_map is not None and str(m) in idx_map:
                    s_year_m, s_val_m = idx_map[str(m)]
                else:
                    s_year_m, s_val_m = (s_year, s_val)

                r2_sst_member = r2_sst_against_true(y_m, tgt[idx], s_year_m, s_val_m)
                if np.isfinite(r2_sst_member):
                    overall_r2_by_model[model_name]["SST index"].append(r2_sst_member)




                per_var_slopes[var]["sst"].append(tr_sst)
                per_var_slopes[var]["nn"].append(tr_nn)
                per_var_slopes[var]["sim"].append(tr_sim)


                years_m = years_full[:idx.size]

                # Store NN(var) series

                pm = per_member_panel.setdefault(str(m), {"series": {}, "sst": None})
                r2_m = compute_r2_only(mu[idx], tgt[idx])
                pm["series"][var] = {
                    "years": years_m.copy(),
                    "mu":    mu[idx].copy(),
                    "sig":   sig[idx].copy(),
                    "tgt":   tgt[idx].copy(),
                    "r2":    float(r2_m),
                }



                # Store SST->AMOC once per member (aligned)
                if pm["sst"] is None and USE_SST_INDEX:
                    # prefer per-member SST if available; else fall back to global for this model
                    if (idx_map is not None) and (str(m) in idx_map):
                        s_year_m, s_val_m = idx_map[str(m)]
                    else:
                        s_year_m, s_val_m = (s_year, s_val)

                    if (s_year_m is not None) and (s_val_m is not None):
                        yy, amoc_hat = sst_to_amoc_recon(s_year_m, s_val_m)
                        # align to years_m
                        common_years = np.intersect1d(yy, years_m)
                        if common_years.size > 0:
                            mask_yy = np.isin(yy, common_years)
                            mask_y  = np.isin(years_m, common_years)
                            yhat_sst = amoc_hat[mask_yy]
                            ytrue    = tgt[idx][mask_y]
                            nmin = min(yhat_sst.size, ytrue.size)
                            if nmin > 0:
                                r2_sst = compute_r2_only(yhat_sst[:nmin], ytrue[:nmin])
                                pm["sst"] = {
                                    "years": common_years[:nmin].copy(),
                                    "amoc":  yhat_sst[:nmin].copy(),
                                    "r2":    float(r2_sst),
                                }




                        # per-var coverage curve
            cov_curve_v, n_used_v = coverage_nominal(mu, sig, tgt,
                                                     nominals=all_models_cov_nominals,
                                                     return_count=True)
            per_var_cov_curves[var]  = cov_curve_v
            per_var_cov_counts[var]  = n_used_v

            aw.add_coverage_curve(model=model_name, var=var,
                        nominals=np.array(all_models_cov_nominals, float),
                        coverage=cov_curve_v, n_used=int(n_used_v))





            # --- Accumulate for pooled scatter ---
            scatter_by_var[var]["mu"].append(mu)
            scatter_by_var[var]["sig"].append(sig)
            scatter_by_var[var]["tgt"].append(tgt)
            scatter_all["mu"].append(mu); scatter_all["sig"].append(sig); scatter_all["tgt"].append(tgt)







                # ---- Merge this fold's per-var data into ALL-MODELS accumulators ----
        for v in x_vars:
            all_per_var_slopes[v]["sst"].extend(per_var_slopes[v]["sst"])
            all_per_var_slopes[v]["nn"].extend(per_var_slopes[v]["nn"])
            all_per_var_slopes[v]["sim"].extend(per_var_slopes[v]["sim"])
            if per_var_cov_curves[v] is not None:
                all_per_var_cov_curves[v].append(per_var_cov_curves[v])
                all_per_var_cov_counts[v].append(per_var_cov_counts[v])






        # --- Trend plots for this fold ---
        model_out_dir = Path.cwd() / "examples_of_reconstructions" / date_str / model_name

        def _plot_hist(sst, nn, sim, title, out_path, bins=20):
            out_path.parent.mkdir(parents=True, exist_ok=True)
            plt.figure(figsize=(10, 6))
            vs = np.array([v for v in sst if np.isfinite(v)])
            vn = np.array([v for v in nn  if np.isfinite(v)])
            vi = np.array([v for v in sim if np.isfinite(v)])
            allv = np.concatenate([vs, vn, vi]) if (vs.size+vn.size+vi.size)>0 else np.array([])
            rng = (np.nanpercentile(allv, 2), np.nanpercentile(allv, 98)) if allv.size>0 else None
            plt.hist(vs, bins=bins, alpha=0.5, density=True, label="SST→AMOC′", range=rng)
            plt.hist(vn, bins=bins, alpha=0.5, density=True, label="NN μ (Stage-1)", range=rng)
            plt.hist(vi, bins=bins, alpha=0.5, density=True, label="Simulated",  range=rng)
            plt.xlabel("Linear trend [Sv/century]"); plt.ylabel("Density")
            plt.title(title); plt.grid(True, alpha=0.3); plt.legend(loc="upper right")
            plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()

        def _plot_violin(sst, nn, sim, title, out_path):
            out_path.parent.mkdir(parents=True, exist_ok=True)
            arr_sst = np.array([v for v in sst if np.isfinite(v)])
            arr_nn  = np.array([v for v in nn  if np.isfinite(v)])
            arr_sim = np.array([v for v in sim if np.isfinite(v)])
            minlen = min(len(arr_sst), len(arr_nn), len(arr_sim))
            if minlen == 0:
                print("    [warn] Not enough data for violin:", out_path.name); return
            e_sst = arr_sst[:minlen] - arr_sim[:minlen]
            e_nn  = arr_nn[:minlen]  - arr_sim[:minlen]
            if _HAVE_SNS:
                df = pd.DataFrame({"Source": np.repeat(["SST Index", "NN μ(Stage-1)"], repeats=minlen),
                                   "Error":  np.concatenate([e_sst, e_nn])})
                plt.figure(figsize=(4,6))
                plt.ylim(-2,1)
                sns.violinplot(x="Source", y="Error", data=df, inner="box", cut=0)
                plt.axhline(0, ls="--", c="k", lw=1)
                plt.ylabel("Trend error vs Simulated [Sv/century]")
                plt.title(title); plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()
            else:
                plt.figure(figsize=(7,6))
                plt.boxplot([e_sst, e_nn], labels=["SST−Sim", "NN−Sim"], showmeans=True)
                plt.axhline(0, ls="--", c="k", lw=1)
                plt.ylabel("Trend error vs Simulated [Sv/century]")
                plt.title(title + " (boxplot fallback)")
                plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()

        _plot_hist(model_slopes_sst, model_slopes_nn, model_slopes_sim,
                   title=f"Linear trends — {model_name}",
                   out_path=model_out_dir / f"trend_hist_{model_name}.png", bins=20)
        _plot_violin(model_slopes_sst, model_slopes_nn, model_slopes_sim,
                     title=f"Trend errors — {model_name}",
                     out_path=model_out_dir / f"trend_violin_{model_name}.png")

                     # --- PIT histogram per climate model (aggregate over all vars for this model) ---
        


                # ---- Multi-variable violin: categories = SST index, NN(var1), NN(var2), ... ----
        def _plot_violin_by_var(per_var_slopes, title, out_path):
            out_path.parent.mkdir(parents=True, exist_ok=True)
            # Build a long-form dataframe: one "SST Index" series + one "NN (var)" per x_var
            src_labels, errors = [], []
            # For each var: align lengths with its own sim set
            for v, dct in per_var_slopes.items():
                arr_sst = np.array([x for x in dct["sst"] if np.isfinite(x)])
                arr_nn  = np.array([x for x in dct["nn"]  if np.isfinite(x)])
                arr_sim = np.array([x for x in dct["sim"] if np.isfinite(x)])
                minlen = min(len(arr_sst), len(arr_nn), len(arr_sim))
                if minlen == 0:
                    continue
                e_sst = arr_sst[:minlen] - arr_sim[:minlen]
                e_nn  = arr_nn[:minlen]  - arr_sim[:minlen]
                # SST Index (repeat per var so violin reflects same member subset)
                src_labels.extend(["SST Index"] * minlen)
                errors.extend(e_sst.tolist())
                # NN(var)
                src_labels.extend([f"NN μ ({v})"] * minlen)
                errors.extend(e_nn.tolist())

            if len(errors) == 0:
                print("    [warn] Not enough data for multi-var violin"); return

            if _HAVE_SNS:
                import pandas as _pd, seaborn as _sns, matplotlib.pyplot as _plt
                df = _pd.DataFrame({"Source": src_labels, "Error": errors})
                _plt.figure(figsize=( max(5, 1.2 + 0.5*len(df["Source"].unique())), 6 ))
                _sns.violinplot(x="Source", y="Error", data=df, inner="box", cut=0)
                _plt.axhline(0, ls="--", c="k", lw=1)
                _plt.ylabel("Trend error vs Simulated [Sv/century]")
                _plt.title(title)
                _plt.xticks(rotation=20, ha="right")
                _plt.tight_layout(); _plt.savefig(out_path, dpi=150); _plt.close()
            else:
                import matplotlib.pyplot as _plt
                # Fallback: group by label into boxplots
                uniq = []
                series = {}
                for lab, err in zip(src_labels, errors):
                    if lab not in series:
                        series[lab] = []
                        uniq.append(lab)
                    series[lab].append(err)
                data = [series[u] for u in uniq]
                _plt.figure(figsize=( max(7, 1.2 + 0.5*len(uniq)), 6 ))
                _plt.boxplot(data, labels=uniq, showmeans=True)
                _plt.axhline(0, ls="--", c="k", lw=1)
                _plt.ylabel("Trend error vs Simulated [Sv/century]")
                _plt.title(title + " (boxplot fallback)")
                _plt.xticks(rotation=20, ha="right")
                _plt.tight_layout(); _plt.savefig(out_path, dpi=150); _plt.close()

        # ---- Coverage vs Nominal: plot one line per variable ----
        def _plot_coverage_by_var(per_var_cov_curves, nominals, out_path, title):
            labs, curves = [], []
            for v, cv in per_var_cov_curves.items():
                if cv is None: 
                    continue
                labs.append(f"NN μ ({v})")
                curves.append(cv)
            if len(curves) == 0:
                print("    [warn] No per-var coverage curves to plot"); return
            plot_coverage_scatter(nominals, curves, labs, out_path=out_path, title=title)

        # Make the two plots for this model (across variables)
        _plot_violin_by_var(
            per_var_slopes,
            title=f"Trend errors by input variable — {model_name}",
            out_path=model_out_dir / f"trend_violin_by_var_{model_name}_.png"
        )
        _plot_coverage_by_var(
            per_var_cov_curves,
            all_models_cov_nominals,
            out_path=model_out_dir / f"coverage_scatter_by_var_{model_name}_.png",
            title=f"Coverage vs Nominal (by variable) — {model_name}"
        )




                # --- NEW: Per-model scatter of trends (True vs NN) for each x_var ---
        for v in x_vars:
            sim_tr = [x for x in per_var_slopes[v]["sim"] if np.isfinite(x)]
            nn_tr  = [x for x in per_var_slopes[v]["nn"]  if np.isfinite(x)]
            nmin = min(len(sim_tr), len(nn_tr))
            if nmin < 2:
                print(f"    [warn] not enough trend pairs for scatter ({model_name}/{v})")
                row[f"r2_trend_{v}"] = np.nan
                continue
            sim_tr = np.asarray(sim_tr[:nmin])
            nn_tr  = np.asarray(nn_tr[:nmin])
            r2_tr = plot_trend_scatter(
                sim_tr, nn_tr,
                out_path=model_out_dir / f"trend_scatter_true_vs_nn_{model_name}_{v}.png",
                title=f"True vs NN Trend — {model_name} — {v}"
            )
            row[f"r2_trend_{v}"] = float(r2_tr)




        var_dict = {"tos" : "SST", "zos_minus_basin_mean" : "SSH"}




                # --- Render: one plot per member (all vars ±2σ + SST index) ---
        if len(per_member_panel) > 0:
            from matplotlib.lines import Line2D
            from matplotlib.patches import Patch

            # consistent colors per variable
            base_colors = plt.rcParams.get("axes.prop_cycle", None)
            color_list = (base_colors.by_key().get("color", ["C0","C1","C2","C3","C4"])
                          if base_colors is not None else ["C0","C1","C2","C3","C4"])
            var_colors = {v: color_list[i % len(color_list)] for i, v in enumerate(x_vars)}

            out_dir_comb = Path.cwd() / "examples_of_reconstructions" / date_str / model_name / "members_combined"
            out_dir_comb.mkdir(parents=True, exist_ok=True)

            # legend proxies (shared style, per figure)
            for mem, payload in per_member_panel.items():
                ser = payload["series"]
                if not ser:
                    continue

                # choose truth from the first available var
                tgt_years = tgt_vals = None
                for v in x_vars:
                    if v in ser:
                        tgt_years = ser[v]["years"]; tgt_vals = ser[v]["tgt"]; break
                if tgt_years is None or tgt_vals is None or tgt_vals.size == 0:
                    continue

                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(tgt_years, tgt_vals, color="k", lw=2, label="Truth")

                # NN lines + ±2σ
                r2_lines = []
                for v in x_vars:
                    if v not in ser: 
                        continue
                    y = ser[v]["years"]; mu_v = ser[v]["mu"]; sig_v = ser[v]["sig"]
                    ax.plot(y, mu_v, lw=2, color=var_colors[v], label=f"NN μ ({var_dict[v]})")
                    if sig_v.size == mu_v.size:
                        ax.fill_between(y, mu_v - 2.0*sig_v, mu_v + 2.0*sig_v,
                                        alpha=0.18, color=var_colors[v], linewidth=0)
                    r2_lines.append(f"{var_dict[v]}: R²={ser[v]['r2']:.3f}")

                # SST index line (if available)
                if payload["sst"] is not None:
                    ss = payload["sst"]
                    ax.plot(ss["years"], ss["amoc"], ls="--", lw=1.8, color="dimgray", label="SST index")
                    r2_lines.append(f"SST index: R²={ss['r2']:.3f}")

                # R² annotation block
                #if r2_lines:
                #    ax.text(0.01, 0.90, "\n".join(r2_lines),
                #            transform=ax.transAxes, va='top', ha='left',
                #            fontsize=10, bbox=dict(facecolor='white', alpha=0.75, edgecolor='none'))

                ax.set_title(f"{model_name}, {mem}", fontsize=16)
                ax.set_xlabel("Year", fontsize=16); ax.set_ylabel("Sv", fontsize=16)
                ax.tick_params(axis='both', labelsize=16)  # Make tick numbers bigger
                
                # Customize tick marks
                from matplotlib.ticker import MultipleLocator
                ax.yaxis.set_major_locator(MultipleLocator(2))  # Y-axis: every 2 units (-2, 0, 2, ...)
                ax.xaxis.set_major_locator(MultipleLocator(20))  # X-axis: every 20 years (1980, 2000, 2020, ...)
                
                #ax.grid(True, alpha=0.25)

                # outside legend (right)
                handles, labels = ax.get_legend_handles_labels()
                fig.subplots_adjust(right=0.82)
                fig.legend(handles, labels, loc="center left",
                           bbox_to_anchor=(0.83, 0.5), bbox_transform=fig.transFigure,
                           frameon=True, fontsize=10)

                fpath = out_dir_comb / f"member_{str(mem).replace('/', '_')}_combined.png"
                fig.savefig(fpath, dpi=1000, bbox_inches="tight", pad_inches=0.25)
                plt.close(fig)
                print(f"    → saved combined member plot: {fpath}")





        


        if len(pit_mu_chunks):
            mu_cat  = np.concatenate(pit_mu_chunks)
            sig_cat = np.concatenate(pit_sig_chunks)
            tgt_cat = np.concatenate(pit_tgt_chunks)

            cov_curve, n_used = coverage_nominal(mu_cat, sig_cat, tgt_cat,
                                                nominals=all_models_cov_nominals,
                                                return_count=True)            # per-model figure
            plot_coverage_scatter(
                all_models_cov_nominals, [cov_curve], [model_name],
                out_path=model_out_dir / f"coverage_scatter_{model_name}.png",
                title=f"Coverage vs Nominal — {model_name}"
            )
            # stash for ALL-MODELS
            all_models_cover_curves.append(cov_curve)
            all_models_labels.append(model_name)
            all_models_counts.append(n_used)






        if len(pit_mu_chunks):
            mu_cat  = np.concatenate(pit_mu_chunks)
            sig_cat = np.concatenate(pit_sig_chunks)
            tgt_cat = np.concatenate(pit_tgt_chunks)
            plot_pit_hist_from_mu_sig(
                mu_cat, sig_cat, tgt_cat,
                out_path=model_out_dir / f"pit_hist_{model_name}.png",
                title=f"PIT histogram — {model_name}"
            )

            # NEW: also persist PIT values for later re-binning
            u = gaussian_pit(mu_cat, sig_cat, tgt_cat)
            aw.save_pit(model=model_name, u=u)


        # --- Merge trends and append result row ---
        all_slopes_sst.extend(model_slopes_sst)
        all_slopes_nn.extend(model_slopes_nn)
        all_slopes_sim.extend(model_slopes_sim)
        results_rows.append(row)

    # --- CSV metrics per fold / variable ---
    res_dir = Path.cwd() / "results"; res_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results_rows)
    out_file = res_dir / f"infer_mix_muStage1_sigmaStage2_{save_name}_{scenario}.csv"
    df.to_csv(out_file, index=False)
    print("\nSaved detailed results to", out_file)

    # --- ALL-MODELS trend plots ---
    all_out_dir = Path.cwd() / "examples_of_reconstructions" / date_str
    def _plot_all_hist(sst, nn, sim, out_path, bins=25):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(10, 6))
        vs = np.array([v for v in sst if np.isfinite(v)])
        vn = np.array([v for v in nn  if np.isfinite(v)])
        vi = np.array([v for v in sim if np.isfinite(v)])
        allv = np.concatenate([vs, vn, vi]) if (vs.size+vn.size+vi.size)>0 else np.array([])
        rng = (np.nanpercentile(allv, 2), np.nanpercentile(allv, 98)) if allv.size>0 else None
        plt.hist(vs, bins=bins, alpha=0.5, density=True, label="SST→AMOC′", range=rng)
        plt.hist(vn, bins=bins, alpha=0.5, density=True, label="NN μ (Stage-1)", range=rng)
        plt.hist(vi, bins=bins, alpha=0.5, density=True, label="Simulated",  range=rng)
        plt.xlabel("Linear trend [Sv/century]"); plt.ylabel("Density")
        plt.title("Linear trends — ALL MODELS"); plt.grid(True, alpha=0.3)
        plt.legend(loc="upper right"); plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()

    def _plot_all_violin(sst, nn, sim, out_path):
        arr_sst = np.array([v for v in sst if np.isfinite(v)])
        arr_nn  = np.array([v for v in nn  if np.isfinite(v)])
        arr_sim = np.array([v for v in sim if np.isfinite(v)])
        minlen = min(len(arr_sst), len(arr_nn), len(arr_sim))
        if minlen == 0:
            print("Not enough data for ALL-MODELS violin"); return
        e_sst = arr_sst[:minlen] - arr_sim[:minlen]
        e_nn  = arr_nn[:minlen]  - arr_sim[:minlen]
        if _HAVE_SNS:
            df = pd.DataFrame({"Source": np.repeat(["SST Index", "NN μ(Stage-1)"], repeats=minlen),
                               "Error":  np.concatenate([e_sst, e_nn])})
            plt.figure(figsize=(4,6))
            plt.ylim(-2,1)
            sns.violinplot(x="Source", y="Error", data=df, inner="box", cut=0)
            plt.axhline(0, ls="--", c="k", lw=1)
            plt.ylabel("Trend error vs Simulated [Sv/century]")
            plt.title("Trend errors — ALL MODELS")
            plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()
        else:
            plt.figure(figsize=(7,6))
            plt.boxplot([e_sst, e_nn], labels=["SST−Sim", "NN−Sim"], showmeans=True)
            plt.axhline(0, ls="--", c="k", lw=1)
            plt.ylabel("Trend error vs Simulated [Sv/century]")
            plt.title("Trend errors — ALL MODELS (boxplot fallback)")
            plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()

    _plot_all_hist(all_slopes_sst, all_slopes_nn, all_slopes_sim,
                   out_path=all_out_dir / "trend_hist_ALL_MODELS.png", bins=25)
    _plot_all_violin(all_slopes_sst, all_slopes_nn, all_slopes_sim,
                     out_path=all_out_dir / "trend_violin_ALL_MODELS.png")

    



        # ---- ALL-MODELS: Multi-variable violin (SST Index, NN(var1), NN(var2), ...) ----
    def _plot_all_violin_by_var(all_per_var_slopes, title, out_path):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        src_labels, errors = [], []


        var_dict = {"tos" : "SST", "zos_minus_basin_mean" : "SSH"}
        for v, dct in all_per_var_slopes.items():
            arr_sst = np.array([x for x in dct["sst"] if np.isfinite(x)])
            arr_nn  = np.array([x for x in dct["nn"]  if np.isfinite(x)])
            arr_sim = np.array([x for x in dct["sim"] if np.isfinite(x)])
            minlen = min(len(arr_sst), len(arr_nn), len(arr_sim))
            if minlen == 0:
                continue
            e_sst = arr_sst[:minlen] - arr_sim[:minlen]
            e_nn  = arr_nn[:minlen]  - arr_sim[:minlen]
            src_labels.extend(["SST Index"] * minlen)
            errors.extend(e_sst.tolist())


            src_labels.extend([f"NN μ ({v})"] * minlen)
            errors.extend(e_nn.tolist())
        if len(errors) == 0:
            print("Not enough data for ALL-MODELS multi-var violin"); return
        if _HAVE_SNS:
            import pandas as _pd, seaborn as _sns, matplotlib.pyplot as _plt
            df = _pd.DataFrame({"Source": src_labels, "Error": errors})
            _plt.figure(figsize=( max(6, 1.2 + 0.5*len(df['Source'].unique())), 6 ))
            _sns.violinplot(x="Source", y="Error", data=df, inner="box", cut=0)
            _plt.axhline(0, ls="--", c="k", lw=1)
            _plt.ylabel("Trend error vs Simulated [Sv/century]")
            _plt.title(title)
            _plt.xticks(rotation=20, ha="right")
            _plt.tight_layout(); _plt.savefig(out_path, dpi=150); _plt.close()
        else:
            import matplotlib.pyplot as _plt
            uniq, series = [], {}
            for lab, err in zip(src_labels, errors):
                if lab not in series:
                    series[lab] = []; uniq.append(lab)
                series[lab].append(err)
            data = [series[u] for u in uniq]
            _plt.figure(figsize=( max(8, 1.2 + 0.5*len(uniq)), 6 ))
            _plt.boxplot(data, labels=uniq, showmeans=True)
            _plt.axhline(0, ls="--", c="k", lw=1)
            _plt.ylabel("Trend error vs Simulated [Sv/century]")
            _plt.title(title + " (boxplot fallback)")
            _plt.xticks(rotation=20, ha="right")
            _plt.tight_layout(); _plt.savefig(out_path, dpi=150); _plt.close()

    # ---- ALL-MODELS: Coverage vs Nominal (one line per variable; weighted mean across models) ----
    def _plot_all_coverage_by_var(all_per_var_cov_curves, all_per_var_cov_counts, nominals, out_path, title):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        labs, curves = [], []
        for v, curve_list in all_per_var_cov_curves.items():
            if len(curve_list) == 0:
                continue
            C = np.stack(curve_list, axis=0)  # [M_v, K]
            w = np.array(all_per_var_cov_counts[v], float)
            w = w / np.sum(w) if np.sum(w) > 0 else np.full(C.shape[0], 1.0/C.shape[0])
            mean_curve = (w[:, None] * C).sum(axis=0)
            labs.append(f"NN μ ({v})")
            curves.append(mean_curve)
        if len(curves) == 0:
            print("No ALL-MODELS per-var coverage curves to plot"); return
        plot_coverage_scatter(nominals, curves, labs, out_path=out_path, title=title)

    # --- Render the two ALL-MODELS plots by variable ---
    _plot_all_violin_by_var(
        all_per_var_slopes,
        title="Trend errors by input variable — ALL MODELS",
        out_path=all_out_dir / "trend_violin_by_var_ALL_MODELS_.png"
    )
    _plot_all_coverage_by_var(
        all_per_var_cov_curves, all_per_var_cov_counts, all_models_cov_nominals,
        out_path=all_out_dir / "coverage_scatter_by_var_ALL_MODELS_.png",
        title="Coverage vs Nominal (by variable) — ALL MODELS"
    )



    # --- Coverage scatter for ALL MODELS (no binning) ---
    all_out_dir = Path.cwd() / "examples_of_reconstructions" / date_str
    if len(all_models_cover_curves) > 0:
        plot_coverage_scatter(
            all_models_cov_nominals,
            all_models_cover_curves,
            all_models_labels,
            out_path=all_out_dir / "coverage_scatter_ALL_MODELS.png",
            title="Coverage vs Nominal — ALL MODELS"
        )

        # NEW: Mean across models (with bootstrap CI and sample-count weighting)
        C = np.stack(all_models_cover_curves, axis=0)  # [M,K]
        plot_coverage_mean(
            all_models_cov_nominals, C,
            out_path=all_out_dir / "coverage_scatter_ALL_MODELS_MEAN.png",
            title="Coverage vs Nominal — MEAN across models",
            weights=np.array(all_models_counts, float),   # or None for unweighted
            ci_method="bootstrap", n_boot=2000, seed=0
        )

        # --- Loop over validation folds (val_models) ---

    # --- SCATTER (pred vs sim) pooled over all folds ---
    # per-var
    for v in x_vars:
        if len(scatter_by_var[v]["mu"]) == 0: 
            continue
        mu_c = np.concatenate(scatter_by_var[v]["mu"])
        sg_c = np.concatenate(scatter_by_var[v]["sig"])
        tg_c = np.concatenate(scatter_by_var[v]["tgt"])
        plot_scatter_pred_vs_sim(
            mu_c, sg_c, tg_c,
            out_path=all_out_dir / f"scatter_pred_vs_sim_ALL_FOLDS_{v}.png",
            title=f"Predicted vs Simulated (±2σ) — ALL FOLDS — {v}"
        )
    # ALL_VARS
    if len(scatter_all["mu"]) > 0:
        mu_c = np.concatenate(scatter_all["mu"])
        sg_c = np.concatenate(scatter_all["sig"])
        tg_c = np.concatenate(scatter_all["tgt"])
        plot_scatter_pred_vs_sim(
            mu_c, sg_c, tg_c,
            out_path=all_out_dir / "scatter_pred_vs_sim_ALL_FOLDS_ALL_VARS.png",
            title="Predicted vs Simulated (±2σ) — ALL FOLDS — ALL VARS"
        )




    if len(scatter_all["mu"]) > 0:
        mu_c = np.concatenate(scatter_all["mu"])
        sg_c = np.concatenate(scatter_all["sig"])
        tg_c = np.concatenate(scatter_all["tgt"])
        plot_pit_hist_from_mu_sig(
            mu_c, sg_c, tg_c,
            out_path=all_out_dir / "pit_hist_ALL_MODELS.png",
            title="PIT histogram — ALL MODELS"
        )
        # --- Categories ---
        categories = ["SST index"] + [f"NN μ ({v})" for v in x_vars]

        # --- Collect all R² values (no model averaging) ---
        overall_mean_r2 = {}
        for cat in categories:
            all_vals = []
            for model_name, cat_dict in overall_r2_by_model.items():
                vals = np.array(cat_dict.get(cat, []), float)
                if vals.size > 0:
                    all_vals.extend(vals[~np.isnan(vals)])  # collect all member values directly
            overall_mean_r2[cat] = float(np.mean(all_vals)) if len(all_vals) > 0 else np.nan

        # --- Plot overall mean R² across all members ---
        labels = [cat for cat in categories if np.isfinite(overall_mean_r2.get(cat, np.nan))]
        values = [overall_mean_r2[cat] for cat in labels]

        plt.figure(figsize=(max(6.5, 1.2 * len(labels)), 4.2))
        plt.bar(labels, values)
        plt.ylim(0.0, 1.0)
        plt.ylabel("Mean R² (across all members, all models)")
        plt.title("Overall R²: SST index vs NN by variable (pooled over all members)")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(all_out_dir / "r2_overall_mean_all_members.png", dpi=150)
        plt.close()

        print("Overall mean R² across all members:",
            {k: (None if not np.isfinite(v) else float(np.round(v, 4))) for k, v in overall_mean_r2.items()})





    # --- R² when using ONLY the first member per model, then combining across models ---
    r2_first_member_combined = {}
    for cat, store in first_member_accum.items():
        # Skip empty categories
        if isinstance(store, dict):
            yh_list = store.get("yhat", [])
            yt_list = store.get("ytrue", [])
        else:
            yh_list, yt_list = [], []
        if len(yh_list) and len(yt_list):
            yhat_all = np.concatenate(yh_list).ravel()
            ytrue_all = np.concatenate(yt_list).ravel()
            r2_first_member_combined[cat] = compute_r2_only(yhat_all, ytrue_all)
        else:
            r2_first_member_combined[cat] = np.nan

    # Plot bar diagram for first-member-only combined R²
    fm_labels = [k for k, v in r2_first_member_combined.items() if np.isfinite(v)]
    fm_values = [r2_first_member_combined[k] for k in fm_labels]
    if len(fm_values):
        plt.figure(figsize=(max(6.5, 1.2*len(fm_labels)), 4.2))
        plt.bar(fm_labels, fm_values)
        plt.ylim(0.0, 1.0)
        plt.ylabel("R² (first member per model, combined)")
        plt.title("Combined R² using ONLY the first member of each model")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(all_out_dir / "r2_first_member_combined.png", dpi=150)
        plt.close()

    print("R² (first-member-only, combined across models):",
          {k: (None if not np.isfinite(v) else float(np.round(v, 4)))
           for k, v in r2_first_member_combined.items()})



    



        # --- Plot all FIRST members on one multi-panel figure (one panel per model) ---
    if len(first_member_panel) > 0:
        import math
        from matplotlib.lines import Line2D


        from matplotlib.patches import Patch  # add this import





        models_list = list(first_member_panel.keys())
        n_models = len(models_list)
        # tidy grid: up to 2 columns looks good; tweak if you prefer 3
        ncols = 2 if n_models <= 6 else 4
        nrows = int(math.ceil(n_models / ncols))

        # consistent colors per variable across all panels
        base_colors = plt.rcParams.get("axes.prop_cycle", None)
        color_list = (base_colors.by_key().get("color", ["C0","C1","C2","C3","C4"])
                      if base_colors is not None else ["C0","C1","C2","C3","C4"])
        var_colors = {v: color_list[i % len(color_list)] for i, v in enumerate(x_vars)}

        fig, axes = plt.subplots(nrows, ncols, figsize=(6.0*ncols, 3.0*nrows), sharex=False, sharey=False)

        fig.subplots_adjust(right=0.82)  # leave room for the outside legend


        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])
        axes = axes.flatten()

        # collect legend proxies once
        #legend_handles = [Line2D([0], [0], color="k", lw=2, label="Truth")]
        #legend_handles += [Line2D([0], [0], color=var_colors[v], lw=2, label=f"NN μ ({v})") for v in x_vars]

        vars_dict = {"tos" : "SST", "zos_minus_basin_mean" : "SSH"}

        legend_handles = [Line2D([0], [0], color="k", lw=2, label="Truth")]
        legend_handles += [Line2D([0], [0], color=var_colors[v], lw=2, label=f"NN μ ({vars_dict[v]})") for v in x_vars]
        legend_handles.append(Patch(facecolor="gray", alpha=0.18, label="±2σ"))  # one generic proxy




        for ax, model_name in zip(axes, models_list):
            ser = first_member_panel[model_name]["series"]

            # Determine a common target series to display; use the first available var's target
            tgt_years = None; tgt_vals = None
            for v in x_vars:
                if v in ser:
                    tgt_years = ser[v]["years"]; tgt_vals = ser[v]["tgt"]; break

            if tgt_years is None or tgt_vals is None or tgt_vals.size == 0:
                ax.set_title(f"{model_name} — (no first-member data)"); ax.axis('off'); continue

            # Plot Truth
            ax.plot(tgt_years, tgt_vals, color="k", lw=2, label="Truth")


            # Plot each var's NN μ
            lines_r2 = []
            for v in x_vars:
                if v not in ser:  # skip vars missing for this model
                    continue
                y = ser[v]["years"]; mu_v = ser[v]["mu"]
                ax.plot(y, mu_v, lw=2, label=f"NN μ ({vars_dict[v]})", color=var_colors[v])


                                # ±2σ shading (if available)
                if "sig" in ser[v] and ser[v]["sig"].size == mu_v.size:
                    ax.fill_between(y,
                                    mu_v - 2.0 * ser[v]["sig"],
                                    mu_v + 2.0 * ser[v]["sig"],
                                    alpha=0.18, color=var_colors[v], linewidth=0)




                lines_r2.append(f"{vars_dict[v]}: R²={ser[v]['r2']:.3f}")

            # annotate R² (one line per var)
            if lines_r2:
                ax.text(0.02, 0.90, "\n".join(lines_r2),
                        transform=ax.transAxes, va='top', ha='left',
                        fontsize=10, bbox=dict(facecolor='white', alpha=0.75, edgecolor='none'))

            ax.set_title(model_name, fontsize=12)
            ax.set_xlabel("Year"); ax.set_ylabel("Sv")
            ax.grid(False, alpha=0.25)

        # turn off unused axes
        for ax in axes[n_models:]:
            ax.axis('off')

        # One shared legend outside all subplots
        #fig.legend(handles=legend_handles, loc="center left",
        #           bbox_to_anchor=(0.8, 0.5), frameon=False, fontsize=11)


        fig_legend = fig.legend(
            handles=legend_handles,
            loc="center left",
            bbox_to_anchor=(0.92, 0.5),          # inside the figure, not off-canvas
            bbox_transform=fig.transFigure,
            frameon=True,
            fontsize=11,
        )



        fig.suptitle("NN Reconstructions — First Member per Model (Truth vs NN μ by variable)", fontsize=14, y=0.995)

        plt.tight_layout(rect=[0.02, 0.02, 0.86, 0.96])  # leave space on the right for legend
        out_path = all_out_dir / "first_members_multi_panel.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"Saved figure: {out_path}")


        # --- NEW: ALL-MODELS scatter of trends (True vs NN) for each x_var ---
    for v in x_vars:
        sim_tr = [x for x in all_per_var_slopes[v]["sim"] if np.isfinite(x)]
        nn_tr  = [x for x in all_per_var_slopes[v]["nn"]  if np.isfinite(x)]
        nmin = min(len(sim_tr), len(nn_tr))
        if nmin < 2:
            print(f"[ALL MODELS] not enough trend pairs for scatter ({v})")
            continue
        sim_tr = np.asarray(sim_tr[:nmin])
        nn_tr  = np.asarray(nn_tr[:nmin])
        _ = plot_trend_scatter(
            sim_tr, nn_tr,
            out_path=all_out_dir / f"trend_scatter_true_vs_nn_ALL_MODELS_{v}.png",
            title=f"True vs NN Trend — ALL MODELS — {v}"
        )

    # --- CESM2 TEST: mean over folds; same plots/metrics ---
    try:
        evaluate_cesm2_test_across_folds(
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
    except Exception as e:
        print("[warn] CESM2 TEST evaluation failed:", e)

    aw.flush()
    print("Done.")


if __name__ == "__main__":
    main()
