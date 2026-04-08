#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ===============================================================
#  BNN μ(t) + Robust Trend Uncertainty (ALL METHODS) → 2 panels
#  Nature-like aesthetics + explicit colors for lines
# ===============================================================

import os, glob, warnings, csv
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import xarray as xr
import torch
from torch.utils.data import DataLoader

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.dates import YearLocator, DateFormatter
import scipy.stats as st
import numpy.linalg as npl
import pandas as pd
import dask
dask.config.set(scheduler='synchronous')

# ─── Your model/dataset imports ────────────────────────────────
from models import ResidualCNNHet
from obs_datasets import (
    lowpass_filter,
    PreprocessedDataset,
    PreprocessedSSHDatasetFromZarr
)

try:
    from omegaconf import OmegaConf
    _OBS_CFG = OmegaConf.load(Path(__file__).resolve().parent / "conf" / "config_bnn.yaml")
    WEIGHTS_ROOT_DIR = str(_OBS_CFG.get("weights_root", "weights"))
except Exception:
    WEIGHTS_ROOT_DIR = "weights"

# ───────────────────────────────────────────────────────────────
#  Config
# ───────────────────────────────────────────────────────────────
TOP_X_RANGE       = (np.datetime64('1900-01-01'), np.datetime64('2024-12-31'))
BOTTOM_WINDOW     = (np.datetime64('1990-01-01'), np.datetime64('2000-12-31'))
SST_TREND_WIN     = (np.datetime64('1900-01-01'), np.datetime64('2000-12-31'))
USE_TOTAL_SIGMA   = False            # σ(t): total vs aleatoric
SEED              = 42

# Which method’s line to draw on the figure (printing uses ALL)
PLOT_METHOD_SST   = "gls_boot"      # "hac" | "mbb" | "gls_boot" | "profile"
PLOT_METHOD_SSH   = "gls_boot"

# Bootstrap / MBB / HAC settings
BOOT_B            = 1200            # vectorized, so ~1200 is already fast
GLS_BOOT_REEST_RHO= False
GLS_BOOT_RHO_JITTER = True
RHO_JITTER_SCALE  = 1.0

# Hints for correlation length (months) after smoothing
HINT_BLOCK_LEN_SST= 120             # LPF120
HINT_BLOCK_LEN_SSH= 48              # LPF24
HAC_HINT_L_SST    = 120
HAC_HINT_L_SSH    = 48

OUT_DIR  = "bnn_real_world_rec"
CSV_PATH = os.path.join(OUT_DIR, "trend_summary.csv")

# ───────────────────────────────────────────────────────────────
#  Nature-like plotting helpers
# ───────────────────────────────────────────────────────────────
OKABE_ITO = {
    "blue":   "#0072B2",  # Reconstruction
    "orange": "#D55E00",  # ECCO
    "verm":   "#E41A1C",
    "green":  "#009E73",
    "purple": "#CC79A7",
    "brown":  "#A6761D",
    "pink":   "#F781BF",
    "grey":   "#7A7A7A",  # CMIP6 MMM
    "black":  "#000000",  # RAPID
}

RECON_BLUE      = OKABE_ITO["blue"]
RAPID_BLACK     = OKABE_ITO["black"]
ECCO_ORANGE     = OKABE_ITO["orange"]
CMIP6_MMM_GREY  = OKABE_ITO["grey"]

mpl.rcParams.update({
    "figure.dpi": 160,
    "savefig.dpi": 300,
    "font.size": 9,
    "axes.linewidth": 0.8,
    "axes.titleweight": "bold",
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.minor.size": 2,
    "ytick.minor.size": 2,
    "legend.frameon": False,
})

def nature_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, which="major", alpha=0.12, linewidth=0.6)
    ax.xaxis.set_major_locator(YearLocator(10))
    ax.xaxis.set_minor_locator(YearLocator(1))
    ax.xaxis.set_major_formatter(DateFormatter("%Y"))
    return ax

def panel_label(ax, label):
    ax.text(0.01, 0.98, label, transform=ax.transAxes,
            ha="left", va="top", fontsize=11, fontweight="bold")

def box_axes(ax, lw=1.0, color="black"):
    """Make a full rectangular frame around an axis (all spines visible)."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(lw)
        spine.set_color(color)

# ───────────────────────────────────────────────────────────────
#  Numerics & covariance helpers
# ───────────────────────────────────────────────────────────────
def safe_inv_sym(A: np.ndarray, jitter: float = 1e-12) -> np.ndarray:
    A = np.asarray(A, float); A = 0.5 * (A + A.T)
    d = np.max(np.diag(A)); d = 1.0 if (not np.isfinite(d) or d <= 0) else d
    A = A + jitter * d * np.eye(A.shape[0])
    L = npl.cholesky(A)
    Linv = npl.solve(L, np.eye(A.shape[0]))
    return Linv.T @ Linv

def make_corr_AR1(T: int, rho: float) -> np.ndarray:
    i = np.arange(T); return rho ** np.abs(i[:, None] - i[None, :])

def estimate_rho_yw(y: np.ndarray, sigma_t: np.ndarray | None = None) -> float:
    r = np.asarray(y, float)
    if sigma_t is not None:
        s = np.asarray(sigma_t, float); s = np.where(s > 1e-12, s, 1.0)
        r = r / s
    r = r - r.mean(); T = len(r)
    if T < 3: return 0.0
    c0 = float(np.dot(r, r) / T)
    if c0 <= 0 or not np.isfinite(c0): return 0.0
    c1 = float(np.dot(r[1:], r[:-1]) / (T - 1))
    rho = c1 / c0
    return float(np.clip(rho, -0.98, 0.98))

# ───────────────────────────────────────────────────────────────
#  ACF/IACT helpers
# ───────────────────────────────────────────────────────────────
def _acf(x, max_lag=None):
    x = np.asarray(x, float); x = x - x.mean(); T = len(x)
    if max_lag is None: max_lag = min(T-2, 600)
    c0 = (x*x).sum() / T
    ac = np.empty(max_lag, float)
    for k in range(1, max_lag+1):
        ac[k-1] = np.dot(x[k:], x[:-k]) / (T-k)
    return ac / max(c0, 1e-18)

def _iact_from_resid(resid):
    ac = _acf(resid, max_lag=min(len(resid)-2, 600))
    pos = ac[ac > 0]
    iact = 1.0 + 2.0 * pos.sum()
    return float(np.clip(iact, 1.0, len(resid)/2))

# ───────────────────────────────────────────────────────────────
#  Trend-uncertainty estimators
# ───────────────────────────────────────────────────────────────
def wls_hac_trend_ci(y, sigma_t, t_idx=None, L=None, alpha=0.05, prewhiten=False, L_floor=0):
    """
    WLS slope CI with HAC (Bartlett).
    Bandwidth from IACT of standardized residuals u = (y - Xβ)/σ_t
    """
    y = np.asarray(y, float); T = len(y)
    x = np.arange(T, dtype=float) if t_idx is None else np.asarray(t_idx, float)
    X = np.c_[np.ones(T), x]
    sig = np.clip(np.asarray(sigma_t, float), 1e-12, np.inf)
    w = 1.0 / (sig**2)
    Wsqrt = np.sqrt(w)
    Xw = X * Wsqrt[:, None]
    yw = y * Wsqrt
    XtX = Xw.T @ Xw
    XtX_inv = safe_inv_sym(XtX)
    beta = XtX_inv @ (Xw.T @ yw)
    resid = y - (X @ beta)
    u = resid / sig

    if L is None:
        iact = _iact_from_resid(u)
        L = int(np.clip(np.round(3.5 * iact), 8, T-1))
        L = max(L, int(L_floor))

    resid_w = yw - Xw @ beta
    psi = Xw * resid_w[:, None]  # (T,2)
    Gamma0 = psi.T @ psi
    meat = Gamma0.copy()
    for k in range(1, L+1):
        wk = 1.0 - k/(L+1.0)
        Gk = psi[k:].T @ psi[:-k]
        meat += wk * (Gk + Gk.T)

    cov_beta = XtX_inv @ meat @ XtX_inv
    slope = float(beta[1])
    se = float(np.sqrt(max(cov_beta[1,1], 0.0)))
    dof = max(T - X.shape[1], 1)
    tcrit = st.t.ppf(1 - alpha/2.0, dof)
    lo, hi = slope - tcrit*se, slope + tcrit*se
    return slope, float(beta[0]), lo, hi, {"cov_beta": cov_beta, "L": L, "iact": float(iact) if L is None else None}




import xarray as xr
import numpy as np
import pandas as pd

def load_proxy_series_nc(nc_path: str,
                         factor: float = 3.8,
                         baseline: tuple[int,int] = (1900, 1940)) -> pd.Series:
    """
    Load a 1D time series from a NetCDF file and return anomalies in Sv.
    - Multiplies by `factor` (default 3.8 Sv/K) to convert to Sv.
    - Anomalies are relative to the mean over `baseline` inclusive.
    - Returns a pandas Series indexed by year (int).
    """
    with xr.open_dataset(nc_path) as ds:
        # pick the first data variable robustly
        data_vars = list(ds.data_vars)
        if not data_vars:
            raise ValueError(f"No data variables found in {nc_path}")
        da = ds["AMOC_LPF10"]

        # get annual series
        if "year" in da.dims:
            years = da["year"].values.astype(int)
            vals = np.asarray(da.values).squeeze()
        elif "time" in da.dims:
            ann = da.groupby("time.year").mean("time")
            years = ann["year"].values.astype(int)
            vals = np.asarray(ann.values).squeeze()
        else:
            # fallback: try to find a coordinate that looks like years
            cand = next((c for c in da.coords if "year" in c.lower()), None)
            if cand is None:
                raise ValueError("No 'time' or 'year' dim/coord found.")
            years = da[cand].values.astype(int)
            vals = np.asarray(da.values).squeeze()

    s = pd.Series(vals, index=years).sort_index()
    # baseline anomaly (inclusive)
    base = float(s.loc[baseline[0]:baseline[1]].mean())
    s_anom = (s - base) * factor
    s_anom.name = f"Caesar/HadISST ×{factor:g} (Sv; base {baseline[0]}–{baseline[1]})"
    return s_anom
    
PROXY_NC = "/dat1/am334/figs/data/AMOC_Caesar_HadISST_LPF10.nc"




def gls_parametric_boot_ci(
    y, sigma_t, t_idx=None, rho=None, B=2000, alpha=0.05,
    reestimate_rho=False, fast=True, rho_jitter=False, rho_jitter_scale=1.0, rng=None,
    *, center_on: str = "gls", estimate_with: str = "gls"
):
    """
    Parametric bootstrap under Σ = D R(ρ) D with AR(1) correlation and heteroskedastic σ_t.
    """
    if rng is None:
        rng = np.random.default_rng(SEED)

    y = np.asarray(y, float); T = len(y)
    x = np.arange(T, dtype=float) if t_idx is None else np.asarray(t_idx, float)
    X = np.c_[np.ones(T), x]
    sig = np.clip(np.asarray(sigma_t, float), 1e-6, np.inf)

    def _beta_ols(y_):
        XtX = X.T @ X
        XtX_inv = safe_inv_sym(XtX)
        return XtX_inv @ (X.T @ y_), XtX_inv

    def _beta_gls(y_, Sigma_inv_):
        XtSi = X.T @ Sigma_inv_
        cov_b = safe_inv_sym(XtSi @ X)
        return cov_b @ (XtSi @ y_), cov_b

    beta_ols, XtXinv = _beta_ols(y)
    resid_ols = y - X @ beta_ols
    if rho is None:
        rho = estimate_rho_yw(resid_ols, sigma_t=sig)

    R = make_corr_AR1(T, rho)
    Sigma = (sig[:, None] * R) * sig[None, :]
    Si = safe_inv_sym(Sigma)

    if center_on == "ols":
        beta_center, _ = beta_ols, XtXinv
    elif center_on == "gls":
        beta_center, _ = _beta_gls(y, Si)
    else:
        raise ValueError("center_on must be 'ols' or 'gls'")

    if estimate_with == "ols":
        g = (XtXinv @ X.T)[1, :]
        beta_hat_est, cov_est = beta_ols, XtXinv
    elif estimate_with == "gls":
        XtSi = X.T @ Si
        cov_b = safe_inv_sym(XtSi @ X)
        g = (cov_b @ XtSi)[1, :]
        beta_hat_est, cov_est = _beta_gls(y, Si)
    else:
        raise ValueError("estimate_with must be 'ols' or 'gls'")

    slope_hat = float(beta_hat_est[1])

    if fast and not reestimate_rho:
        L = np.linalg.cholesky(Sigma)
        Z = rng.standard_normal((T, B))
        E = L @ Z
        if rho_jitter:
            sd_rho = rho_jitter_scale / np.sqrt(T)
            rho_draws = np.clip(rho + sd_rho * rng.standard_normal(B), -0.98, 0.98)
            mix = np.abs(rho_draws)[None, :]
            E = (1.0 - 0.5*mix) * E + 0.5*mix * np.vstack([np.zeros((1,B)), E[:-1,:]])
        YB = (X @ beta_center)[:, None] + E
        slopes = g @ YB
        lo, hi = np.quantile(slopes, [alpha/2, 1-alpha/2])
        return slope_hat, float(beta_hat_est[0]), float(lo), float(hi), {
            "boot_slopes": slopes,
            "rho_init": rho,
            "center_on": center_on,
            "estimate_with": estimate_with
        }

    L = np.linalg.cholesky(Sigma)
    slopes = np.empty(B, float)
    for b in range(B):
        e = L @ rng.standard_normal(T)
        y_b = X @ beta_center + e
        rho_b = estimate_rho_yw(y_b - (X @ beta_center), sigma_t=sig) if reestimate_rho else rho
        Rb = make_corr_AR1(T, rho_b); Sig_b = (sig[:, None] * Rb) * sig[None, :]
        Si_b = safe_inv_sym(Sig_b)
        if estimate_with == "ols":
            beta_b, _ = _beta_ols(y_b)
        else:
            beta_b, _ = _beta_gls(y_b, Si_b)
        slopes[b] = float(beta_b[1])

    lo, hi = np.quantile(slopes, [alpha/2, 1-alpha/2])
    return slope_hat, float(beta_hat_est[0]), float(lo), float(hi), {
        "boot_slopes": slopes,
        "rho_init": rho,
        "center_on": center_on,
        "estimate_with": estimate_with
    }

def mbb_trend_ci(y, sigma_t, t_idx=None, B=2000, block_len=None, alpha=0.05,
                 rng=None, hint_block_len=None):
    """Moving-Block Bootstrap on standardized residuals; block_len ≈ max(2*IACT, hint)."""
    if rng is None:
        rng = np.random.default_rng(SEED)
    y = np.asarray(y, float); T = len(y)
    x = np.arange(T, dtype=float) if t_idx is None else np.asarray(t_idx, float)
    X = np.c_[np.ones(T), x]
    sig = np.clip(np.asarray(sigma_t, float), 1e-12, np.inf)
    w = 1.0 / (sig**2)
    Wsqrt = np.sqrt(w); Xw = X * Wsqrt[:, None]; yw = y * Wsqrt
    XtX = Xw.T @ Xw; XtX_inv = safe_inv_sym(XtX)
    beta_hat = XtX_inv @ (Xw.T @ yw)
    resid = y - (X @ beta_hat)
    u = resid / sig

    if block_len is None:
        iact = _iact_from_resid(u)
        auto_bl = int(np.clip(np.round(2.0 * iact), 8, T))
        if hint_block_len is not None:
            auto_bl = max(auto_bl, int(hint_block_len))
        block_len = auto_bl

    idx = np.arange(T)
    slopes = np.empty(B, float)
    n_blocks = int(np.ceil(T / block_len))
    max_start = T - block_len  # No circular padding
    if max_start < 1:
        max_start = 1
    for b in range(B):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        idx_b = np.concatenate([idx[s:s+block_len] for s in starts])[:T]
        u_b = u[idx_b]
        e_b = u_b * sig
        y_b = (X @ beta_hat) + e_b
        yw_b = y_b * Wsqrt
        beta_b = XtX_inv @ (Xw.T @ yw_b)
        slopes[b] = float(beta_b[1])

    lo, hi = np.quantile(slopes, [alpha/2, 1-alpha/2])
    return float(beta_hat[1]), float(beta_hat[0]), lo, hi, {"boot_slopes": slopes, "block_len": block_len}

def gls_profile_ci(y, sigma_t, t_idx=None, rho=None, alpha=0.05, grid_mult=8):
    y = np.asarray(y, float); T = len(y)
    x = np.arange(T, dtype=float) if t_idx is None else np.asarray(t_idx, float)
    X1 = np.ones(T)
    sig = np.clip(np.asarray(sigma_t, float), 1e-6, np.inf)
    if rho is None:
        rho = estimate_rho_yw(y, sigma_t=sig)
    R = make_corr_AR1(T, rho); Sigma = (sig[:,None]*R)*sig[None,:]
    Si = safe_inv_sym(Sigma)
    X = np.c_[X1, x]
    XtSi = X.T @ Si
    cov_beta = safe_inv_sym(XtSi @ X)
    beta_hat = cov_beta @ (XtSi @ y)
    slope_hat = float(beta_hat[1])

    def loglik_given_slope(b):
        num = X1 @ (Si @ (y - b*x))
        den = X1 @ (Si @ X1)
        a = num / den
        r = y - (a + b*x)
        quad = r @ (Si @ r)
        sign, logdet = np.linalg.slogdet(Sigma)
        return -0.5*(quad + logdet)

    ll_hat = loglik_given_slope(slope_hat)
    se_wald = np.sqrt(max(cov_beta[1,1], 1e-18))
    grid = np.linspace(slope_hat - grid_mult*se_wald, slope_hat + grid_mult*se_wald, 1201)
    thr = 0.5 * st.chi2.ppf(0.95, df=1)
    ll_vals = np.array([loglik_given_slope(b) for b in grid])
    ok = (ll_hat - ll_vals) <= thr
    if not np.any(ok):
        z = st.norm.ppf(0.975)
        return slope_hat, float(beta_hat[0]), slope_hat - z*se_wald, slope_hat + z*se_wald, {"cov_beta": cov_beta, "rho": rho}
    lo = np.min(grid[ok]); hi = np.max(grid[ok])
    return slope_hat, float(beta_hat[0]), lo, hi, {"cov_beta": cov_beta, "rho": rho}

def compute_trend_ci(y, sigma_t, t_idx, method="hac", hint_block_len=None, hac_L_floor=0):
    if method == "hac":
        s,i,lo,hi,info = wls_hac_trend_ci(y, sigma_t, t_idx=t_idx, L=None, alpha=0.05,
                                          prewhiten=False, L_floor=hac_L_floor)
        label = "WLS+HAC 95% CI"; key = "hac"
    elif method == "mbb":
        s,i,lo,hi,info = mbb_trend_ci(y, sigma_t, t_idx=t_idx, B=BOOT_B, block_len=None,
                                      alpha=0.05, hint_block_len=hint_block_len)
        label = f"MBB {BOOT_B} 95% CI"; key = "mbb"
    elif method == "gls_boot":
        s,i,lo,hi,info = gls_parametric_boot_ci(
            y, sigma_t, t_idx=t_idx, B=BOOT_B, alpha=0.05,
            reestimate_rho=GLS_BOOT_REEST_RHO, fast=True,
            rho_jitter=GLS_BOOT_RHO_JITTER, rho_jitter_scale=RHO_JITTER_SCALE,
            center_on="gls", estimate_with="gls"
        )
        label = f"GLS param. bootstrap {BOOT_B} 95% CI"; key = "gls_boot"
    elif method == "gls_boot_around_ols":
        s,i,lo,hi,info = gls_parametric_boot_ci(
            y, sigma_t, t_idx=t_idx, B=BOOT_B, alpha=0.05,
            reestimate_rho=GLS_BOOT_REEST_RHO, fast=True,
            rho_jitter=GLS_BOOT_RHO_JITTER, rho_jitter_scale=RHO_JITTER_SCALE,
            center_on="ols", estimate_with="ols"
        )
        label = f"GLS param. bootstrap (around OLS) {BOOT_B} 95% CI"; key = "gls_boot_around_ols"
    elif method == "profile":
        s,i,lo,hi,info = gls_profile_ci(y, sigma_t, t_idx=t_idx, alpha=0.05)
        label = "GLS profile 95% CI"; key = "profile"
    else:
        raise ValueError(f"Unknown method '{method}'")
    return float(s), float(i), float(lo), float(hi), info, label, key

def slope_two_sigma_from_info(method_key, info):
    if method_key in ("mbb", "gls_boot", "gls_boot_around_ols"):
        s = float(np.std(info["boot_slopes"], ddof=1))
        return 2.0 * s
    if "cov_beta" in info:
        se = float(np.sqrt(max(info["cov_beta"][1, 1], 0.0)))
        return 2.0 * se
    raise KeyError(f"No variance info found for method '{method_key}'.")

def scale_vals(*vals, factor):
    return [v * factor for v in vals]

# ───────────────────────────────────────────────────────────────
#  BNN predictions
# ───────────────────────────────────────────────────────────────
def get_bnn_predictions(dataset, weight_pattern, in_ch, out_dim,
                        device=None, batch_size=8):
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=2)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpts = sorted(glob.glob(weight_pattern))
    if not ckpts:
        raise RuntimeError(f"No checkpoints match '{weight_pattern}'")
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

# ───────────────────────────────────────────────────────────────
#  Utility: run all methods & print/save
# ───────────────────────────────────────────────────────────────
def run_all_methods_and_report(series_name, y, sigma_t, t_idx, factor, years_str, rows_accum,
                               hint_block_len=None, hac_L_floor=0):
    methods = ["hac", "mbb", "gls_boot", "gls_boot_around_ols"]
    pretty = {"hac":"WLS+HAC","mbb":"MBB","gls_boot":"GLS bootstrap","gls_boot_around_ols":"GLS boot around OLS"}

    print(f"\n[{series_name}] {years_str}:")
    for m in methods:
        slope, intercept, lo, hi, info, label, key = compute_trend_ci(
            y, sigma_t, t_idx, method=m,
            hint_block_len=(hint_block_len if m=="mbb" else None),
            hac_L_floor=(hac_L_floor if m=="hac" else 0)
        )
        two_sigma = slope_two_sigma_from_info(key, info)
        slope_c, lo_c, hi_c, two_sigma_c = scale_vals(slope, lo, hi, two_sigma, factor=factor)

        print(f"  • {pretty[key]:20s}: {slope_c:+.3f} ± {two_sigma_c:.3f} Sv/century   "
              f"| 95% CI [{lo_c:+.3f}, {hi_c:+.3f}]")

        rows_accum.append({
            "series": series_name,
            "period": years_str,
            "method": pretty[key],
            "slope_Sv_per_century": slope_c,
            "two_sigma_Sv_per_century": two_sigma_c,
            "lo95_Sv_per_century": lo_c,
            "hi95_Sv_per_century": hi_c
        })

def _auto_grid(n, max_cols=4):
    cols = min(max_cols, max(1, int(np.ceil(np.sqrt(n)))))
    rows = int(np.ceil(n / cols))
    return rows, cols

def plot_per_fold_predictions(all_mus, all_sigs, times, var_name, out_dir,
                              ensemble_mu=None, ensemble_tot_sigma=None,
                              use_total_sigma=False, top_x_range=None):
    """
    all_mus:  (F, T)
    all_sigs: (F, T)
    """
    F, T = all_mus.shape
    os.makedirs(out_dir, exist_ok=True)

    # (A) Overlay figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), dpi=150, sharex=True,
                                   gridspec_kw={'height_ratios': [2.0, 1.2], 'hspace': 0.08})

    for f in range(F):
        ax1.plot(times, all_mus[f], lw=1.0, alpha=0.6, color=RECON_BLUE)
    if ensemble_mu is not None:
        ax1.plot(times, ensemble_mu, lw=2.0, color=RECON_BLUE, label='Ensemble μ', zorder=5)
        if use_total_sigma and ensemble_tot_sigma is not None:
            ax1.fill_between(times,
                             ensemble_mu - 2*ensemble_tot_sigma,
                             ensemble_mu + 2*ensemble_tot_sigma,
                             alpha=0.15, label='±2σ (ensemble total)',
                             facecolor=RECON_BLUE, edgecolor="none")
    ax1.set_ylabel('AMOC (Sv)')
    if top_x_range is not None:
        ax1.set_xlim(*top_x_range)
    nature_axes(ax1)
    if ensemble_mu is not None:
        ax1.legend(loc='upper left', ncol=2)
    ax1.set_title(f'{var_name}: Per-fold μ overlay (F={F})')

    for f in range(F):
        ax2.plot(times, all_sigs[f], lw=1.0, alpha=0.8, color=RECON_BLUE)
    ax2.set_ylabel('σ (aleatoric)')
    ax2.set_xlabel('Time')
    if top_x_range is not None:
        ax2.set_xlim(*top_x_range)
    nature_axes(ax2)

    fname_overlay = os.path.join(out_dir, f"{var_name.lower()}_per_fold_overlay.png")
    plt.tight_layout()
    plt.savefig(fname_overlay, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved overlay → {fname_overlay}")

    # (B) Small-multiples
    rows, cols = _auto_grid(F, max_cols=4)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2*cols+1.0, 2.7*rows+0.8),
                             dpi=140, sharex=True, sharey=True)
    axes = np.atleast_2d(axes)

    for f in range(F):
        r, c = divmod(f, cols)
        ax = axes[r, c]
        mu_f  = all_mus[f]
        sig_f = all_sigs[f]
        ax.plot(times, mu_f, lw=1.4, color=RECON_BLUE, label='μ (fold)')
        ax.fill_between(times, mu_f - 2*sig_f, mu_f + 2*sig_f, alpha=0.15,
                        label='±2σ (aleatoric)', facecolor=RECON_BLUE, edgecolor="none")
        if ensemble_mu is not None:
            ax.plot(times, ensemble_mu, lw=1.0, ls='--', label='Ensemble μ', alpha=0.8, color=CMIP6_MMM_GREY)
        if top_x_range is not None:
            ax.set_xlim(*top_x_range)
        nature_axes(ax)
        ax.set_title(f"Fold {f+1}", fontsize=10)
        if r == rows - 1: ax.set_xlabel('Time')
        if c == 0: ax.set_ylabel('AMOC (Sv)')
        if f == 0:
            ax.legend(fontsize=9, loc='upper left')

    for k in range(F, rows*cols):
        r, c = divmod(k, cols)
        fig.delaxes(axes[r, c])

    fname_grid = os.path.join(out_dir, f"{var_name.lower()}_per_fold_grid.png")
    plt.tight_layout()
    plt.savefig(fname_grid, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved per-fold grid → {fname_grid}")

# ===============================================================
#  Main
# ===============================================================
if __name__ == "__main__":
    np.random.seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    factor = 12 * 100  # monthly slope → Sv/century

    # ── 1) SST ─────────────────────────────────────────────────
    sst_file = "processed_HadISST_sst.nc"
    sst_ds   = PreprocessedDataset(
        sst_file, variable="__xarray_dataarray_variable__",
        lpf="LPF120", order=5, fs=1, monthly=True, minus_basin_mean=False
    )
    x0 = sst_ds[0]
    if x0.ndim == 2: x0 = x0[np.newaxis]
    in_ch_sst = x0.shape[0]

    sst_pattern = os.path.join(WEIGHTS_ROOT_DIR, "bnn_test_god_bless_it_work_all_vars_all_models_test_120_y_new",
                               "tos", "*", "best_stage2_joint.pt")

    all_mus_sst, all_sigs_sst, mu_sst, alea_sst, epi_sst, tot_sst = get_bnn_predictions(
        sst_ds, sst_pattern, in_ch_sst, 1, device=device, batch_size=16
    )

    times_sst = xr.open_dataset(sst_file).coords["time"].values
    sigma_sst = tot_sst if USE_TOTAL_SIGMA else alea_sst

    m_tr_sst = (times_sst >= SST_TREND_WIN[0]) & (times_sst <= SST_TREND_WIN[1])
    y_sst, s_sst = mu_sst[m_tr_sst], sigma_sst[m_tr_sst]
    t_sst = np.arange(m_tr_sst.sum(), dtype=float)
    years_sst = f"{times_sst[m_tr_sst][0].astype('M8[Y]').item().year}–{times_sst[m_tr_sst][-1].astype('M8[Y]').item().year}"

    # ── 2) SSH ─────────────────────────────────────────────────
    ssh_zarr_path = "duacs_data_6_may.zarr"
    ssh_ds = PreprocessedSSHDatasetFromZarr(
        ssh_zarr_path, variable="__xarray_dataarray_variable__", lpf="LPF24",
        order=5, fs=1, monthly=True
    )
    x0_ssh = ssh_ds[0]
    if x0_ssh.ndim == 2: x0_ssh = x0_ssh[np.newaxis]
    in_ch_ssh = x0_ssh.shape[0]

    ssh_pattern = os.path.join(WEIGHTS_ROOT_DIR, "bnn_test_god_bless_it_work_all_vars_all_models_test_24",
                               "zos_minus_basin_mean", "*", "best_stage2_joint.pt")

    all_mus_ssh, all_sigs_ssh, mu_ssh, alea_ssh, epi_ssh, tot_ssh = get_bnn_predictions(
        ssh_ds, ssh_pattern, in_ch_ssh, 1, device=device, batch_size=16
    )

    times_ssh = xr.open_zarr(ssh_zarr_path).coords["time"].values
    sigma_ssh = tot_ssh if USE_TOTAL_SIGMA else alea_ssh
    t_ssh = np.arange(len(mu_ssh), dtype=float)
    years_ssh = f"{times_ssh[0].astype('M8[Y]').item().year}–{times_ssh[-1].astype('M8[Y]').item().year}"

    # ── 3) Run ALL methods & print summary ─────────────────────
    rows = []
    run_all_methods_and_report("SST→AMOC", y_sst, s_sst, t_sst, factor, years_sst, rows,
                               hint_block_len=HINT_BLOCK_LEN_SST, hac_L_floor=HAC_HINT_L_SST)
    run_all_methods_and_report("SSH→AMOC", mu_ssh, sigma_ssh, t_ssh, factor, years_ssh, rows,
                               hint_block_len=HINT_BLOCK_LEN_SSH, hac_L_floor=HAC_HINT_L_SSH)

    # Save CSV summary
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "series", "period", "method",
            "slope_Sv_per_century", "two_sigma_Sv_per_century",
            "lo95_Sv_per_century", "hi95_Sv_per_century"
        ])
        writer.writeheader()
        for r in rows: writer.writerow(r)
    print(f"\n[INFO] Saved summary → {CSV_PATH}")

    # ── 4) Build figure (anomalies & matched scale) ─────────────────────────────

    TOP_X_RANGE = (np.datetime64('1900-01-01'), np.datetime64('2024-12-31'))



    def _anomalize_to_window(t_vals, y_vals, t0, t1):
        t_vals = pd.to_datetime(t_vals).to_numpy()
        y_vals = np.asarray(y_vals, dtype=float)
        m = (t_vals >= np.datetime64(t0)) & (t_vals <= np.datetime64(t1))
        base = np.nanmean(y_vals[m]) if np.any(m) else np.nanmean(y_vals)
        return y_vals - base, float(base)

    BASE_T0, BASE_T1 = "1900-01-01", "1940-12-31"

    # Trend helpers (unchanged)
    def slope_line_for(method_key, y, s, t, hint_block_len=None, hac_L_floor=0):
        slope, intercept, *_ = compute_trend_ci(
            y, s, t, method=method_key,
            hint_block_len=hint_block_len,
            hac_L_floor=hac_L_floor
        )[:4]
        return slope * t + intercept

    trend_line1 = slope_line_for(
        PLOT_METHOD_SST, y_sst, s_sst, t_sst,
        hint_block_len=(HINT_BLOCK_LEN_SST if PLOT_METHOD_SST=="mbb" else None),
        hac_L_floor=(HAC_HINT_L_SST if PLOT_METHOD_SST=="hac" else 0)
    )
    trend_line2 = slope_line_for(
        PLOT_METHOD_SSH, mu_ssh, sigma_ssh, t_ssh,
        hint_block_len=(HINT_BLOCK_LEN_SSH if PLOT_METHOD_SSH=="mbb" else None),
        hac_L_floor=(HAC_HINT_L_SSH if PLOT_METHOD_SSH=="hac" else 0)
    )

    # Figure with matched pixels-per-Sv: ranges 6 Sv (top) vs 9 Sv (bot) → height ratio 2:3
    mpl.rcParams.update({
        "font.size": 16,
        "axes.labelsize": 16,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
    })
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(15.5, 8.4), dpi=160, sharex=True,
        gridspec_kw={"height_ratios": [3, 4], "hspace": 0.06}
    )

    # ---------- Load/prepare CMIP6 MMM (same as before) ----------
    csv_path = Path("/home/am334/link_am334/ECCO_data/forced_amoc_26N_noLPF/MULTIMODEL_AMOC26N_forced_mean_1850_2030.csv")
    df = pd.read_csv(csv_path)
    time_col = next((c for c in df.columns if c.lower() in ("time","date","datetime")), None)

    def _ensure_datetime_from_year_month(df_, year_col, month_col=None):
        y = pd.to_numeric(df_[year_col], errors="coerce").astype("Int64")
        if month_col is not None and month_col in df_.columns:
            m = pd.to_numeric(df_[month_col], errors="coerce").astype("Int64").fillna(12)
            dt = pd.PeriodIndex(year=y, month=m, freq="M").to_timestamp(how="end")
        else:
            dt = pd.to_datetime(y.astype(str) + "-12-31", errors="coerce")
        return dt

    if time_col is not None:
        t_cmip = pd.to_datetime(df[time_col], errors="coerce")
    else:
        year_like = next((c for c in df.columns if c.lower() in ("year","yr")), None)
        month_like = next((c for c in df.columns if c.lower() in ("month","mon","mm")), None)
        if year_like is not None:
            t_cmip = _ensure_datetime_from_year_month(df, year_like, month_like)
        else:
            first_col = df.columns[0]
            parsed = pd.to_datetime(df[first_col], errors="coerce")
            t_cmip = parsed if parsed.notna().any() else pd.date_range("1900-01-31", periods=len(df), freq="M")

    df["_time_dt64_"] = pd.to_datetime(t_cmip).astype("datetime64[ns]")
    amoc_candidates = [c for c in df.columns if ("amoc" in c.lower()) or ("26n" in c.lower())]
    if amoc_candidates:
        amoc_col = amoc_candidates[0]
    else:
        skip = set([time_col, "_time_dt64_"]) if time_col else {"_time_dt64_"}
        num_cols = [c for c in df.columns if c not in skip and pd.api.types.is_numeric_dtype(df[c])]
        if not num_cols:
            for c in df.columns:
                if c in skip: continue
                coerced = pd.to_numeric(df[c], errors="coerce")
                if coerced.notna().any():
                    df[c] = coerced; num_cols.append(c)
        amoc_col = num_cols[0]

    amoc = pd.to_numeric(df[amoc_col], errors="coerce").to_numpy()
    # CMIP6 → LPF10 then anomalies 1900–1940 to match instruction
    cmip6_lp10 = lowpass_filter(amoc, 1/10, 5, 1, 20)
    cmip6_anom, _ = _anomalize_to_window(df["_time_dt64_"], cmip6_lp10, BASE_T0, BASE_T1)

    # ---------- TOP PANEL (SST → AMOC anomalies; 1900–2024) ----------
    m_top = (times_sst >= TOP_X_RANGE[0]) & (times_sst <= TOP_X_RANGE[1])
    mu_sst_anom, base_sst = _anomalize_to_window(times_sst, mu_sst, BASE_T0, BASE_T1)
    sst_sigma_for_band = (tot_sst if USE_TOTAL_SIGMA else alea_sst)  # band unaffected by centering

    # shaded band ±2σ (already in Sv; mean shift doesn't change σ)
    ax_top.fill_between(
        times_sst[m_top],
        (mu_sst_anom - 2*sst_sigma_for_band)[m_top],
        (mu_sst_anom + 2*sst_sigma_for_band)[m_top],
        facecolor=RECON_BLUE, alpha=0.15, edgecolor="none", zorder=1,
        label="Reconstruction ±2σ"
    )
    ax_top.plot(times_sst[m_top], mu_sst_anom[m_top], lw=2.2, color=RECON_BLUE,
                zorder=2, label="Reconstruction (BNN μ, SST)")

    # CMIP6 MMM anomalies (grey dashed)
    m6 = (pd.to_datetime(df["_time_dt64_"]).to_numpy() >= TOP_X_RANGE[0]) & \
        (pd.to_datetime(df["_time_dt64_"]).to_numpy() <= TOP_X_RANGE[1])
    ax_top.plot(df["_time_dt64_"].values[m6], np.asarray(cmip6_anom)[m6],
                lw=1.8, color=CMIP6_MMM_GREY, ls="--", zorder=1.5, label="CMIP6 multi-model mean")

    # RAPID trend centered to SST anomalies over overlap (optional)
    try:
        ds_moc = xr.open_dataset('moc_vertical.nc')
        rapid_m = ds_moc['stream_function_mar'].resample(time='1M').mean()
        rapid_max = rapid_m.max(dim='depth')  # with Ekman
        t_r = np.arange(rapid_max.sizes["time"], dtype=float)
        y_r = rapid_max.values.astype(float)
        valid = np.isfinite(y_r)
        t_r = t_r[valid]; y_r = y_r[valid]
        if y_r.size >= 3:
            slope_r, intercept_r = np.polyfit(t_r, y_r, 1)
            trend_r = slope_r * t_r + intercept_r
            rapid_time_valid = rapid_max.time.values[valid]
            trend_da = xr.DataArray(trend_r, coords={'time': rapid_time_valid}, dims='time')

            sst_da = xr.DataArray(mu_sst_anom, coords={'time': times_sst}, dims='time')
            t0_ = max(sst_da.time.values.min(), trend_da.time.values.min())
            t1_ = min(sst_da.time.values.max(), trend_da.time.values.max())
            sst_w = sst_da.sel(time=np.s_[t0_:t1_])
            trend_w = trend_da.sel(time=np.s_[t0_:t1_]).interp(time=sst_w.time)

            mm = np.isfinite(sst_w.values) & np.isfinite(trend_w.values)
            shift = float(np.nanmean(sst_w.values[mm]) - np.nanmean(trend_w.values[mm])) if mm.sum() >= 3 else 0.0
            trend_plot = (trend_da + shift).sel(time=slice(TOP_X_RANGE[0], TOP_X_RANGE[1]))
            ax_top.plot(trend_plot.time.values[60:], trend_plot.values[60:],
                        lw=2.2, color=RAPID_BLACK, label="RAPID trend", zorder=3)
    except Exception as e:
        print(f"[WARN] RAPID overlay skipped (top): {e}")

    ax_top.set_xlim(*TOP_X_RANGE)
    ax_top.set_ylim(-3, 4)  # fixed range
    ax_top.set_ylabel("AMOC (Sv)")
    nature_axes(ax_top)
    ax_top.legend(loc="upper left", ncol=2)
    ax_top.tick_params(axis="x", labelbottom=False)

    # ---------- BOTTOM PANEL (SSH → AMOC; 1990–2024), matched pixel scale ----------
    # Ekman aligned to SSH grid
    ek_fn = "/home/am334/link_am334/ECCO_data/ekman_transport_26p5N_monthly_Sv.nc"
    ek = xr.open_dataset(ek_fn)["ekman_transport"].sortby("time")
    times_da = xr.DataArray(times_ssh, dims="time")
    ek_i = ek.interp(time=times_da).ffill("time").bfill("time")
    ek_vals = lowpass_filter(ek_i.values.astype(np.float64), 1/24, pad=48)
    ek_center = float(np.nanmean(ek_vals))

    m_bot = (times_ssh >= BOTTOM_WINDOW[0]) & (times_ssh <= BOTTOM_WINDOW[1])
    recon_ssh = mu_ssh + ek_vals - ek_center

    # DL SSH (+2σ band)
    ssh_sigma_for_band = (tot_ssh if USE_TOTAL_SIGMA else alea_ssh)
    ax_bot.fill_between(
        times_ssh[m_bot],
        (mu_ssh - 2*ssh_sigma_for_band + ek_vals - ek_center)[m_bot],
        (mu_ssh + 2*ssh_sigma_for_band + ek_vals - ek_center)[m_bot],
        facecolor=RECON_BLUE, alpha=0.15, edgecolor="none", zorder=2,
        label="Reconstruction ±2σ"
    )
    ax_bot.plot(times_ssh[m_bot], recon_ssh[m_bot], lw=2.2, color=RECON_BLUE,
                zorder=3, label="Reconstruction (BNN μ, SSH+Ekman)")

    # RAPID array with Ekman (LPF~24) and fixed offset as before
    try:
        ds_moc = xr.open_dataset('moc_vertical.nc')
        rapid = ds_moc['stream_function_mar'].resample(time='1M').mean()
        rapid_max = rapid.max(dim='depth')
        rapid_wo_lp = lowpass_filter(rapid_max, 1/24, 5, 1, 48)
        ax_bot.plot(rapid_max.time[24:], (rapid_wo_lp[24:] - 17.0),
                    lw=2.4, color=RAPID_BLACK, zorder=4, label="RAPID array")
    except Exception as e:
        print(f"[WARN] RAPID overlay skipped (bottom): {e}")

    # ECCO (optional, unchanged)
    try:
        ds_ecco = xr.open_dataset("/dat1/am334/ECCO_data/amoc_residual_26N5_Sv_monthly.nc")
        ecco = ds_ecco["amoc_residual"][24:].copy()
        ecco_lp = lowpass_filter(ecco, 1/24, 5, 1, 48)
        ECCO_OFFSET_MODE = 14.5
        ax_bot.plot(ecco.time, ecco_lp - ECCO_OFFSET_MODE,
                    lw=2.0, color=ECCO_ORANGE, zorder=3, label="ECCOv4r4")
    except Exception as e:
        print(f"[WARN] ECCO overlay skipped: {e}")

    #ax_bot.set_xlim(*BOTTOM_WINDOW)
    ax_bot.set_xlim(*TOP_X_RANGE)

    ax_bot.set_ylim(-4, 4)  # wider range; height ratio 2:3 ensures same pixels/Sv as top
    ax_bot.set_xlabel("Time"); ax_bot.set_ylabel("AMOC (Sv)")
    nature_axes(ax_bot)
    ax_bot.legend(loc="upper left", ncol=2)

    # ---- Save AMOC time series (unchanged) ----
    def _to_dt64ns(t):
        return pd.to_datetime(np.asarray(t)).astype("datetime64[ns]")
    time_sst = _to_dt64ns(times_sst)
    time_ssh = _to_dt64ns(times_ssh)
    amoc_ssh_mu = mu_ssh + ek_vals - ek_center

    ds_sst = xr.Dataset(
        data_vars=dict(
            amoc_from_sst_mu=("time", np.asarray(mu_sst, dtype=np.float32)),
            amoc_from_sst_sigma_aleatoric=("time", np.asarray(alea_sst, dtype=np.float32)),
            amoc_from_sst_sigma_epistemic=("time", np.asarray(epi_sst, dtype=np.float32)),
            amoc_from_sst_sigma_total=("time", np.asarray(tot_sst, dtype=np.float32)),
        ),
        coords=dict(time=time_sst),
        attrs=dict(title="AMOC based on SST (BNN μ) with uncertainties",
                Conventions="CF-1.10"),
    )
    ds_sst["time"].attrs.update(standard_name="time", long_name="Time", axis="T")
    ds_sst["amoc_from_sst_mu"].attrs.update(units="Sv", long_name="AMOC from SST, model mean (μ)")
    ds_sst["amoc_from_sst_sigma_aleatoric"].attrs.update(units="Sv", long_name="Aleatoric σ (SST)")
    ds_sst["amoc_from_sst_sigma_epistemic"].attrs.update(units="Sv", long_name="Epistemic σ (SST)")
    ds_sst["amoc_from_sst_sigma_total"].attrs.update(units="Sv", long_name="Total σ (SST)")
    sst_nc = os.path.join(OUT_DIR, "amoc_from_sst.nc"); ds_sst.to_netcdf(sst_nc); print(f"[NC] Saved {sst_nc}")

    ds_ssh = xr.Dataset(
        data_vars=dict(
            amoc_from_ssh_mu=("time", np.asarray(amoc_ssh_mu, dtype=np.float32)),
            amoc_from_ssh_mu_no_ekman=("time", np.asarray(mu_ssh, dtype=np.float32)),
            ekman_added=("time", np.asarray(ek_vals, dtype=np.float32)),
            amoc_from_ssh_sigma_aleatoric=("time", np.asarray(alea_ssh, dtype=np.float32)),
            amoc_from_ssh_sigma_epistemic=("time", np.asarray(epi_ssh, dtype=np.float32)),
            amoc_from_ssh_sigma_total=("time", np.asarray(tot_ssh, dtype=np.float32)),
        ),
        coords=dict(time=time_ssh),
        attrs=dict(title="AMOC based on SSH (BNN μ) + Ekman (low-pass) with uncertainties",
                Conventions="CF-1.10",
                notes="amoc_from_ssh_mu = μ_ssh + Ekman - mean(Ekman)"),
    )
    ds_ssh["time"].attrs.update(standard_name="time", long_name="Time", axis="T")
    ssh_nc = os.path.join(OUT_DIR, "amoc_from_ssh.nc"); ds_ssh.to_netcdf(ssh_nc); print(f"[NC] Saved {ssh_nc}")

    # ---- Final figure save ----
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.06)
    out_path = os.path.join(OUT_DIR, "stacked_sst_ssh_sharedx.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[DONE] Saved {out_path}")



    # ===========================
    # EXTRA FIGURES (final tweaks)
    # ===========================
    os.makedirs(OUT_DIR, exist_ok=True)

    # Global matplotlib font sizes
    plt.rcParams.update({
        "font.size": 16,
        "axes.labelsize": 16,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
    })

    # Helper: anomalies relative to mean over window [t0, t1]
    def _anomalize_to_window(t_vals, y_vals, t0, t1):
        t_vals = pd.to_datetime(t_vals).to_numpy()
        y_vals = np.asarray(y_vals, dtype=float)
        m = (t_vals >= np.datetime64(t0)) & (t_vals <= np.datetime64(t1))
        if not np.any(m):
            base = np.nanmean(y_vals)  # fallback
        else:
            base = np.nanmean(y_vals[m])
        return y_vals - base


    # ===========================
    # EXTRA FIGURE 1: SST μ ±2σ vs CMIP6 MMM split by adr_hd (HIGH vs LOW)
    # ===========================
    try:
        # Where the split MMM CSVs live (produced by your earlier script)
        CMIP_SPLIT_DIR = "/home/am334/link_am334/ECCO_data/forced_amoc_26N_noLPF"
        CSV_HIGH = os.path.join(CMIP_SPLIT_DIR, "MULTIMODEL_AMOC26N_forced_mean_1850_2030_HIGH_ADRHD.csv")
        CSV_LOW  = os.path.join(CMIP_SPLIT_DIR, "MULTIMODEL_AMOC26N_forced_mean_1850_2030_LOW_ADRHD.csv")

        # Colors for lines
        COLOR_HIGH = "#009E73"   # green
        COLOR_LOW  = "#CC79A7"   # purple
        COLOR_SST  = RECON_BLUE
        COLOR_BAND = RECON_BLUE

        BASE_T0, BASE_T1 = "1900-01-01", "1930-12-31"

        def _ensure_datetime_from_year_month(df_, year_col, month_col=None):
            y = pd.to_numeric(df_[year_col], errors="coerce").astype("Int64")
            if month_col is not None and month_col in df_.columns:
                m = pd.to_numeric(df_[month_col], errors="coerce").astype("Int64").fillna(12)
                dt = pd.PeriodIndex(year=y, month=m, freq="M").to_timestamp(how="end")
            else:
                dt = pd.to_datetime(y.astype(str) + "-12-31", errors="coerce")
            return dt

        def _load_mmm_series(csv_path):
            """Return datetime index and numeric AMOC series from a MMM CSV with flexible columns."""
            df = pd.read_csv(csv_path)
            # Time
            time_col = next((c for c in df.columns if c.lower() in ("time","date","datetime")), None)
            if time_col is not None:
                t = pd.to_datetime(df[time_col], errors="coerce")
            else:
                year_like  = next((c for c in df.columns if c.lower() in ("year","yr")), None)
                month_like = next((c for c in df.columns if c.lower() in ("month","mon","mm")), None)
                if year_like is not None:
                    t = _ensure_datetime_from_year_month(df, year_like, month_like)
                else:
                    # fallback: first column parse attempt
                    first_col = df.columns[0]
                    t_try = pd.to_datetime(df[first_col], errors="coerce")
                    t = t_try if t_try.notna().any() else pd.date_range("1850-12-31", periods=len(df), freq="Y")
            # AMOC column
            amoc_candidates = [c for c in df.columns if ("amoc" in c.lower()) or ("26n" in c.lower())]
            if amoc_candidates:
                y = pd.to_numeric(df[amoc_candidates[0]], errors="coerce").to_numpy()
            else:
                skip = set([time_col]) if time_col else set()
                num_cols = [c for c in df.columns if c not in skip and pd.api.types.is_numeric_dtype(df[c])]
                if not num_cols:
                    # try coerce everything numeric
                    for c in df.columns:
                        if c in skip: continue
                        coerced = pd.to_numeric(df[c], errors="coerce")
                        if coerced.notna().any():
                            df[c] = coerced; num_cols.append(c)
                y = pd.to_numeric(df[num_cols[0]], errors="coerce").to_numpy()
            return pd.to_datetime(t).to_numpy(), np.asarray(y, dtype=float)

        def _anom_lpf10(t_vals, y_vals, t0, t1):
            """Apply LPF10 then anomalies vs baseline window [t0, t1]."""
            y_lp10 = lowpass_filter(y_vals, 1/10, 5, 1, 20)
            t_vals = pd.to_datetime(t_vals).to_numpy()
            mask = (t_vals >= np.datetime64(t0)) & (t_vals <= np.datetime64(t1))
            base = float(np.nanmean(y_lp10[mask])) if mask.any() else float(np.nanmean(y_lp10))
            return y_lp10 - base

        # Load HIGH and LOW MMM
        t_high, y_high = _load_mmm_series(CSV_HIGH)
        t_low,  y_low  = _load_mmm_series(CSV_LOW)

        # Convert to LPF10 anomalies vs 1900–1930
        y_high_anom = _anom_lpf10(t_high, y_high, BASE_T0, BASE_T1)
        y_low_anom  = _anom_lpf10(t_low,  y_low,  BASE_T0, BASE_T1)

        # SST reconstruction anomalies and ±2σ band (already computed earlier in your script)
        # If not available (edge case), fall back to recompute quickly.
        #try:
        #    mu_sst_anom  # noqa
        #    sst_sigma_for_band = (tot_sst if USE_TOTAL_SIGMA else alea_sst)
        #except NameError:
            # Minimal fallback to ensure figure builds; assumes sst_ds / predictions already set above.
        
        print(_anomalize_to_window(times_sst, mu_sst, BASE_T0, BASE_T1))

        mu_sst_anom = _anomalize_to_window(times_sst, mu_sst, BASE_T0, BASE_T1)
        sst_sigma_for_band = (tot_sst if USE_TOTAL_SIGMA else alea_sst)

        # Window for plotting
        tmin, tmax = TOP_X_RANGE
        m_sst  = (times_sst >= tmin) & (times_sst <= tmax)
        m_high = (t_high    >= tmin) & (t_high    <= tmax)
        m_low  = (t_low     >= tmin) & (t_low     <= tmax)

        # Build figure (high resolution for poster)
        fig, ax = plt.subplots(figsize=(10.5, 4.6), dpi=300)

        # SST μ ± 2σ
        ax.fill_between(times_sst[m_sst],
                        (mu_sst_anom - 2*sst_sigma_for_band)[m_sst],
                        (mu_sst_anom + 2*sst_sigma_for_band)[m_sst],
                        alpha=0.20, facecolor=COLOR_BAND, edgecolor="none")
        ax.plot(times_sst[m_sst], mu_sst_anom[m_sst], lw=2.2, color=COLOR_SST, label="DL SST")



        try:
            ds_moc_rapid = xr.open_dataset('moc_vertical.nc')
            rapid_m = ds_moc_rapid['stream_function_mar'].resample(time='1M').mean()
            rapid_max = rapid_m.max(dim='depth')  # with Ekman
            t_r = np.arange(rapid_max.sizes["time"], dtype=float)
            y_r = rapid_max.values.astype(float)
            valid = np.isfinite(y_r)
            t_r = t_r[valid]; y_r = y_r[valid]
            if y_r.size >= 3:
                slope_r, intercept_r = np.polyfit(t_r, y_r, 1)
                trend_r = slope_r * t_r + intercept_r
                rapid_time_valid = rapid_max.time.values[valid]
                trend_da = xr.DataArray(trend_r, coords={'time': rapid_time_valid}, dims='time')

                # Center trend to SST anomalies over overlap
                sst_da = xr.DataArray(mu_sst_anom, coords={'time': times_sst}, dims='time')
                t0_ = max(sst_da.time.values.min(), trend_da.time.values.min())
                t1_ = min(sst_da.time.values.max(), trend_da.time.values.max())
                sst_w = sst_da.sel(time=np.s_[t0_:t1_])
                trend_w = trend_da.sel(time=np.s_[t0_:t1_]).interp(time=sst_w.time)

                mm = np.isfinite(sst_w.values) & np.isfinite(trend_w.values)
                shift = float(np.nanmean(sst_w.values[mm]) - np.nanmean(trend_w.values[mm])) if mm.sum() >= 3 else 0.0
                trend_plot = (trend_da + shift).sel(time=slice(TOP_X_RANGE[0], TOP_X_RANGE[1]))
                ax.plot(trend_plot.time.values[60:], trend_plot.values[60:],
                        lw=2.9, color=OKABE_ITO["verm"], label="RAPID trend", zorder=3)
        except Exception as e:
            print(f"[WARN] RAPID trend overlay skipped (extra_fig1): {e}")
            
        # HIGH / LOW adr_hd MMM (LPF10, anomalies 1900–1940)
        if np.any(m_high):
            ax.plot(t_high[m_high], y_high_anom[m_high], lw=2.2, ls="-.", color="k",
                    label="CMIP6 (high aerosol forcing)")
        if np.any(m_low):
            ax.plot(t_low[m_low],  y_low_anom[m_low],  lw=2.2, ls=":", color="k",
                    label="CMIP6 (low aerosol forcing)")



        # Axes and styling
        ax.set_xlim(*TOP_X_RANGE)
        ax.set_ylim(-3, 4)
        ax.set_ylabel("AMOC 26.5°N anomaly, Sv")
        box_axes(ax, lw=1.0, color="black")  # Full box around the graph
        ax.legend(frameon=False, ncol=2, loc="upper left")
        #ax.grid(True, which="major", alpha=0.12, linewidth=0.6)

        fig.tight_layout()
        out_extra1 = os.path.join(OUT_DIR, "extra_fig1_sst_vs_high_low_adrhd.png")
        fig.savefig(out_extra1, dpi=1000, bbox_inches="tight")  # Very high resolution for poster
        plt.close(fig)
        print(f"[PLOT] Saved → {out_extra1}")

    except Exception as e:
        print(f"[WARN] Extra Figure 1 (SST vs HIGH/LOW adr_hd MMM) failed: {e}")

    # ===========================
    # EXTRA FIGURE 1b: SST NN vs HIGH/LOW with MODEL SPREAD INTERVALS
    # ===========================
    try:
        print("\n[INFO] Building Extra Figure 1b: SST NN vs HIGH/LOW with model spread...")

        CMIP_SPLIT_DIR = "/home/am334/link_am334/ECCO_data/forced_amoc_26N_noLPF"

        # Define HIGH and LOW model groups (from adr_hd_groups.txt)
        HIGH_MODELS = [
            "ACCESS-ESM1-5", "CESM2", "GFDL-CM4", "GISS-E2-1-G",
            "HadGEM3-GC31-LL", "HadGEM3-GC31-MM", "MRI-ESM2-0",
            "NorESM2-LM", "NorESM2-MM", "UKESM1-1-LL",
        ]
        LOW_MODELS = [
            "ACCESS-CM2", "CMCC-ESM2", "CanESM5", "CanESM5-CanOE",
            "FGOALS-g3", "GFDL-ESM4", "INM-CM4-8", "MIROC6", "MPI-ESM1-2-HR",
        ]

        def _load_single_model(model_name, base_dir):
            """Load a single model CSV → (years, amoc_Sv)."""
            csv_f = os.path.join(base_dir, model_name,
                                 f"{model_name}_AMOC26N_forced_perScenario_1850_2030.csv")
            df_m = pd.read_csv(csv_f)
            years = df_m["year"].values.astype(int)
            vals  = pd.to_numeric(df_m["AMOC_forced_Sv"], errors="coerce").values
            return years, vals

        def _load_group_array(model_list, base_dir, base_t0, base_t1):
            """Load all models in a group, apply LPF10 and anomalize.
            Returns (common_datetime64, array shape (N_models, T))."""
            series_list = []
            for mname in model_list:
                try:
                    yrs, vals = _load_single_model(mname, base_dir)
                    # annual → datetime (Dec-31 of each year)
                    dt = pd.to_datetime([f"{y}-12-31" for y in yrs]).to_numpy()
                    # LPF10 on annual data (cutoff=1/10, pad=20 at annual cadence)
                    vals_lp = lowpass_filter(vals, 1/10, 5, 1, 20)
                    # anomalize
                    mask = (dt >= np.datetime64(base_t0)) & (dt <= np.datetime64(base_t1))
                    base_val = float(np.nanmean(vals_lp[mask])) if mask.any() else float(np.nanmean(vals_lp))
                    vals_anom = vals_lp - base_val
                    series_list.append(pd.Series(vals_anom, index=dt))
                except Exception as exc:
                    print(f"    [WARN] Skipping {mname}: {exc}")
            if not series_list:
                raise RuntimeError("No models loaded for group")
            # Align to common index
            df_all = pd.concat(series_list, axis=1).sort_index()
            common_t = df_all.index.values
            arr = df_all.values.T  # (N_models, T)
            return common_t, arr

        BASE_T0_1b, BASE_T1_1b = "1900-01-01", "1930-12-31"

        t_high_all, arr_high = _load_group_array(HIGH_MODELS, CMIP_SPLIT_DIR, BASE_T0_1b, BASE_T1_1b)
        t_low_all,  arr_low  = _load_group_array(LOW_MODELS,  CMIP_SPLIT_DIR, BASE_T0_1b, BASE_T1_1b)

        # Statistics per group
        high_mean = np.nanmean(arr_high, axis=0)
        high_min  = np.nanmin(arr_high, axis=0)
        high_max  = np.nanmax(arr_high, axis=0)

        low_mean = np.nanmean(arr_low, axis=0)
        low_min  = np.nanmin(arr_low, axis=0)
        low_max  = np.nanmax(arr_low, axis=0)

        # SST anomalies (reuse from earlier, baseline 1900-1930)
        mu_sst_anom_1b = _anomalize_to_window(times_sst, mu_sst, BASE_T0_1b, BASE_T1_1b)
        sst_sigma_1b = (tot_sst if USE_TOTAL_SIGMA else alea_sst)

        # Window masks
        tmin, tmax = TOP_X_RANGE
        m_sst_1b  = (times_sst >= tmin) & (times_sst <= tmax)
        m_high_1b = (t_high_all >= tmin) & (t_high_all <= tmax)
        m_low_1b  = (t_low_all  >= tmin) & (t_low_all  <= tmax)

        COLOR_HIGH_1b = "#009E73"
        COLOR_LOW_1b  = "#CC79A7"

        fig, ax = plt.subplots(figsize=(10.5, 4.6), dpi=300)

        # SST NN ±2σ
        ax.fill_between(times_sst[m_sst_1b],
                        (mu_sst_anom_1b - 2*sst_sigma_1b)[m_sst_1b],
                        (mu_sst_anom_1b + 2*sst_sigma_1b)[m_sst_1b],
                        alpha=0.20, facecolor=RECON_BLUE, edgecolor="none")
        ax.plot(times_sst[m_sst_1b], mu_sst_anom_1b[m_sst_1b],
                lw=2.2, color=RECON_BLUE, label="DL SST")

        # RAPID trend
        try:
            ds_moc_rapid_1b = xr.open_dataset('moc_vertical.nc')
            rapid_m_1b = ds_moc_rapid_1b['stream_function_mar'].resample(time='1M').mean()
            rapid_max_1b = rapid_m_1b.max(dim='depth')
            t_r = np.arange(rapid_max_1b.sizes["time"], dtype=float)
            y_r = rapid_max_1b.values.astype(float)
            valid = np.isfinite(y_r)
            t_r = t_r[valid]; y_r = y_r[valid]
            if y_r.size >= 3:
                slope_r, intercept_r = np.polyfit(t_r, y_r, 1)
                trend_r = slope_r * t_r + intercept_r
                rapid_time_valid = rapid_max_1b.time.values[valid]
                trend_da = xr.DataArray(trend_r, coords={'time': rapid_time_valid}, dims='time')
                sst_da = xr.DataArray(mu_sst_anom_1b, coords={'time': times_sst}, dims='time')
                t0_ = max(sst_da.time.values.min(), trend_da.time.values.min())
                t1_ = min(sst_da.time.values.max(), trend_da.time.values.max())
                sst_w = sst_da.sel(time=np.s_[t0_:t1_])
                trend_w = trend_da.sel(time=np.s_[t0_:t1_]).interp(time=sst_w.time)
                mm = np.isfinite(sst_w.values) & np.isfinite(trend_w.values)
                shift = float(np.nanmean(sst_w.values[mm]) - np.nanmean(trend_w.values[mm])) if mm.sum() >= 3 else 0.0
                trend_plot = (trend_da + shift).sel(time=slice(TOP_X_RANGE[0], TOP_X_RANGE[1]))
                ax.plot(trend_plot.time.values[60:], trend_plot.values[60:],
                        lw=2.9, color=OKABE_ITO["verm"], label="RAPID trend", zorder=3)
        except Exception as e:
            print(f"[WARN] RAPID trend overlay skipped (extra_fig1b): {e}")

        # HIGH group: mean + spread
        if np.any(m_high_1b):
            ax.fill_between(t_high_all[m_high_1b], high_min[m_high_1b], high_max[m_high_1b],
                            alpha=0.18, facecolor=COLOR_HIGH_1b, edgecolor="none")
            ax.plot(t_high_all[m_high_1b], high_mean[m_high_1b], lw=2.2, ls="-.",
                    color=COLOR_HIGH_1b, label="CMIP6 high aer. forcing (mean ± spread)")

        # LOW group: mean + spread
        if np.any(m_low_1b):
            ax.fill_between(t_low_all[m_low_1b], low_min[m_low_1b], low_max[m_low_1b],
                            alpha=0.18, facecolor=COLOR_LOW_1b, edgecolor="none")
            ax.plot(t_low_all[m_low_1b], low_mean[m_low_1b], lw=2.2, ls=":",
                    color=COLOR_LOW_1b, label="CMIP6 low aer. forcing (mean ± spread)")

        ax.set_xlim(*TOP_X_RANGE)
        ax.set_ylim(-3, 4)
        ax.set_ylabel("AMOC 26.5°N anomaly, Sv")
        box_axes(ax, lw=1.0, color="black")
        ax.legend(frameon=False, ncol=2, loc="upper left")
        fig.tight_layout()
        out_extra1b = os.path.join(OUT_DIR, "extra_fig1b_sst_vs_high_low_spread.png")
        fig.savefig(out_extra1b, dpi=1000, bbox_inches="tight")
        plt.close(fig)
        print(f"[PLOT] Saved → {out_extra1b}")

    except Exception as e:
        import traceback
        print(f"[WARN] Extra Figure 1b (model spread) failed: {e}")
        traceback.print_exc()

    # ===========================
    # EXTRA FIGURE 1c: SST NN vs LOW / MID / HIGH aerosol forcing
    # ===========================
    try:
        print("\n[INFO] Building Extra Figure 1c: SST NN vs LOW/MID/HIGH aerosol forcing...")

        CMIP_SPLIT_DIR_1c = "/home/am334/link_am334/ECCO_data/forced_amoc_26N_noLPF"
        CSV_HIGH_1c = os.path.join(CMIP_SPLIT_DIR_1c, "MULTIMODEL_AMOC26N_forced_mean_1850_2030_HIGH_ADRHD.csv")
        CSV_LOW_1c  = os.path.join(CMIP_SPLIT_DIR_1c, "MULTIMODEL_AMOC26N_forced_mean_1850_2030_LOW_ADRHD.csv")
        CSV_MID_1c  = os.path.join(CMIP_SPLIT_DIR_1c, "MULTIMODEL_AMOC26N_forced_mean_1850_2030.csv")

        BASE_T0_1c, BASE_T1_1c = "1900-01-01", "1940-12-31"

        # Load HIGH / LOW / MID (overall mean = mid proxy)
        t_high_1c, y_high_1c = _load_mmm_series(CSV_HIGH_1c)
        t_low_1c,  y_low_1c  = _load_mmm_series(CSV_LOW_1c)
        t_mid_1c,  y_mid_1c  = _load_mmm_series(CSV_MID_1c)

        # Zero-phase LPF10 for CMIP6 lines (annual data: cutoff=1/10, fs=1)
        from scipy.signal import butter as butter_1c, sosfiltfilt
        sos_annual = butter_1c(5, 1/10, btype='low', output='sos', fs=1)

        def _anom_lpf10_zerophase(t_vals, y_vals, t0, t1):
            """Zero-phase LPF10 then anomalies vs baseline."""
            y_lp = sosfiltfilt(sos_annual, y_vals)
            t_vals = pd.to_datetime(t_vals).to_numpy()
            mask = (t_vals >= np.datetime64(t0)) & (t_vals <= np.datetime64(t1))
            base = float(np.nanmean(y_lp[mask])) if mask.any() else float(np.nanmean(y_lp))
            return y_lp - base

        y_high_anom_1c = _anom_lpf10_zerophase(t_high_1c, y_high_1c, BASE_T0_1c, BASE_T1_1c)
        y_low_anom_1c  = _anom_lpf10_zerophase(t_low_1c,  y_low_1c,  BASE_T0_1c, BASE_T1_1c)
        y_mid_anom_1c  = _anom_lpf10_zerophase(t_mid_1c,  y_mid_1c,  BASE_T0_1c, BASE_T1_1c)

        # SST anomalies
        mu_sst_anom_1c = _anomalize_to_window(times_sst, mu_sst, BASE_T0_1c, BASE_T1_1c)
        sst_sigma_1c = (tot_sst if USE_TOTAL_SIGMA else alea_sst)

        # Window masks
        tmin, tmax = TOP_X_RANGE
        m_sst_1c  = (times_sst >= tmin) & (times_sst <= tmax)
        m_high_1c = (t_high_1c >= tmin) & (t_high_1c <= tmax)
        m_low_1c  = (t_low_1c  >= tmin) & (t_low_1c  <= tmax)
        m_mid_1c  = (t_mid_1c  >= tmin) & (t_mid_1c  <= tmax)

        COLOR_HIGH_1c = "#009E73"   # green
        COLOR_MID_1c  = CMIP6_MMM_GREY  # grey
        COLOR_LOW_1c  = "#CC79A7"   # purple

        # Zero-phase LPF for DL SST (monthly data: cutoff=1/120 = 10yr, fs=1)
        sos_monthly = butter_1c(5, 1/120, btype='low', output='sos', fs=1)
        mu_sst_anom_1c_filt = sosfiltfilt(sos_monthly, mu_sst_anom_1c)

        fig, ax = plt.subplots(figsize=(10.5, 4.6), dpi=300)

        # SST NN ±2σ (band uses unfiltered sigma, line uses filtered mean)
        ax.fill_between(times_sst[m_sst_1c],
                        (mu_sst_anom_1c_filt - 2*sst_sigma_1c)[m_sst_1c],
                        (mu_sst_anom_1c_filt + 2*sst_sigma_1c)[m_sst_1c],
                        alpha=0.20, facecolor=RECON_BLUE, edgecolor="none")
        ax.plot(times_sst[m_sst_1c], mu_sst_anom_1c_filt[m_sst_1c],
                lw=2.2, color=RECON_BLUE, label="DL SST")

        # DL SSH trend line over 1997–2023, slope ± 2σ from MBB
        mu_ssh_anom_1c = _anomalize_to_window(times_ssh, mu_ssh, BASE_T0_1c, BASE_T1_1c)
        SSH_TREND_T0 = np.datetime64('1997-01-01')
        SSH_TREND_T1 = np.datetime64('2023-12-31')
        m_ssh_tr = (times_ssh >= SSH_TREND_T0) & (times_ssh <= SSH_TREND_T1)
        t_idx_ssh = np.arange(m_ssh_tr.sum(), dtype=float)
        y_ssh_win = mu_ssh_anom_1c[m_ssh_tr]
        ssh_sigma_1c = alea_ssh[m_ssh_tr]

        # MBB trend: slope, intercept, and bootstrap slope distribution
        slope_m, intercept_m, _, _, info_mbb, _, key_mbb = compute_trend_ci(
            y_ssh_win, ssh_sigma_1c, t_idx_ssh, method="mbb", hint_block_len=120
        )
        slope_sigma = float(np.std(info_mbb["boot_slopes"], ddof=1))
        trend_central = slope_m * t_idx_ssh + intercept_m

        # Shift so trend mean matches filtered SST mean over overlap
        m_sst_overlap = (times_sst >= SSH_TREND_T0) & (times_sst <= SSH_TREND_T1)
        sst_mean_overlap = float(np.nanmean(mu_sst_anom_1c_filt[m_sst_overlap]))
        shift_val = sst_mean_overlap - float(np.mean(trend_central))
        trend_central = trend_central + shift_val

        # Uncertainty fan: intercept SE (from WLS) + slope SE (from MBB)
        sig = np.clip(ssh_sigma_1c, 1e-12, np.inf)
        w = 1.0 / (sig ** 2)
        X_tr = np.c_[np.ones_like(t_idx_ssh), t_idx_ssh]
        XwX = (X_tr * w[:, None]).T @ X_tr
        XwX_inv = np.linalg.inv(XwX)
        sigma_intercept = float(np.sqrt(max(XwX_inv[0, 0], 0.0)))
        t_mid = np.mean(t_idx_ssh)
        dt_from_mid = t_idx_ssh - t_mid
        se_total = np.sqrt(sigma_intercept**2 + (dt_from_mid * slope_sigma)**2)

        # 2σ band (outer, lighter)
        ax.fill_between(times_ssh[m_ssh_tr],
                        trend_central - 2*se_total, trend_central + 2*se_total,
                        facecolor=OKABE_ITO["orange"], alpha=0.12, edgecolor="none", zorder=2.3)
        # 1σ band (inner, darker)
        ax.fill_between(times_ssh[m_ssh_tr],
                        trend_central - 1*se_total, trend_central + 1*se_total,
                        facecolor=OKABE_ITO["orange"], alpha=0.25, edgecolor="none", zorder=2.5)

        # Central trend line on top
        slope_c = slope_m * 12 * 100
        slope_2s_c = slope_sigma * 2 * 12 * 100
        ax.plot(times_ssh[m_ssh_tr], trend_central,
                lw=2.9, color=OKABE_ITO["orange"], zorder=3,
                label="DL SSH trend")

        # CMIP6 aerosol forcing lines — all dash-dot style
        if np.any(m_high_1c):
            ax.plot(t_high_1c[m_high_1c], y_high_anom_1c[m_high_1c], lw=2.2, ls="-.",
                    color=COLOR_HIGH_1c)
        if np.any(m_mid_1c):
            ax.plot(t_mid_1c[m_mid_1c], y_mid_anom_1c[m_mid_1c], lw=2.2, ls="-.",
                    color=COLOR_MID_1c)
        if np.any(m_low_1c):
            ax.plot(t_low_1c[m_low_1c], y_low_anom_1c[m_low_1c], lw=2.2, ls="-.",
                    color=COLOR_LOW_1c)

        ax.set_xlim(*TOP_X_RANGE)
        ax.set_ylim(-3, 4)
        ax.set_ylabel("AMOC 26.5°N anomaly, Sv")
        box_axes(ax, lw=1.0, color="black")

        # --- Two legends: left for NN, right for CMIP6 ---
        from matplotlib.lines import Line2D
        # Left legend: DL SST + DL SSH trend
        handles_auto, labels_auto = ax.get_legend_handles_labels()
        leg_left = ax.legend(handles_auto, labels_auto, frameon=False, ncol=1,
                             loc="upper left", handlelength=2.0)
        ax.add_artist(leg_left)
        # Right legend: clean table — header then indented high/mid/low
        cmip6_high = Line2D([], [], color=COLOR_HIGH_1c, ls="-.", lw=2.2)
        cmip6_mid  = Line2D([], [], color=COLOR_MID_1c,  ls="-.", lw=2.2)
        cmip6_low  = Line2D([], [], color=COLOR_LOW_1c,  ls="-.", lw=2.2)
        leg_right = ax.legend(
            [cmip6_high, cmip6_mid, cmip6_low],
            ["high", "mid", "low"],
            frameon=False, ncol=1, loc="upper right",
            handlelength=2.5, handletextpad=0.6, labelspacing=0.3,
            title="CMIP6 aerosol forcing:", title_fontsize=None,
        )
        leg_right._legend_box.align = "right"
        fig.tight_layout()
        out_extra1c = os.path.join(OUT_DIR, "extra_fig1c_sst_vs_low_mid_high.png")
        fig.savefig(out_extra1c, dpi=1000, bbox_inches="tight")
        plt.close(fig)
        print(f"[PLOT] Saved → {out_extra1c}")

    except Exception as e:
        import traceback
        print(f"[WARN] Extra Figure 1c (LOW/MID/HIGH aerosol) failed: {e}")
        traceback.print_exc()

    # ===========================
    # EXTRA FIGURE 2: SST μ ±2σ vs CMIP6 MMM (mean, all models) - same style as extra_fig1
    # ===========================
    try:
        # Colors for lines
        COLOR_SST  = RECON_BLUE
        COLOR_BAND = RECON_BLUE
        COLOR_CMIP6 = CMIP6_MMM_GREY  # Grey for CMIP6 mean

        BASE_T0_FIG2, BASE_T1_FIG2 = "1900-01-01", "1930-12-31"

        # SST reconstruction anomalies (same baseline as extra_fig1)
        mu_sst_anom_fig2 = _anomalize_to_window(times_sst, mu_sst, BASE_T0_FIG2, BASE_T1_FIG2)
        sst_sigma_for_band_fig2 = (tot_sst if USE_TOTAL_SIGMA else alea_sst)

        # CMIP6 mean: apply LPF10 and anomalies vs 1900-1930
        # Reuse the loaded CMIP6 data but recompute with correct baseline
        cmip6_t_fig2 = df["_time_dt64_"].values
        # Reuse amoc if available, otherwise recompute
        try:
            amoc_raw_fig2 = amoc  # Use already computed amoc
        except NameError:
            amoc_raw_fig2 = pd.to_numeric(df[amoc_col], errors="coerce").to_numpy()
        cmip6_lp10_fig2 = lowpass_filter(amoc_raw_fig2, 1/10, 5, 1, 20)
        cmip6_anom_fig2 = _anomalize_to_window(cmip6_t_fig2, cmip6_lp10_fig2, BASE_T0_FIG2, BASE_T1_FIG2)

        # Window for plotting
        tmin, tmax = TOP_X_RANGE
        m_sst_fig2  = (times_sst >= tmin) & (times_sst <= tmax)
        m_cmip6_fig2 = (cmip6_t_fig2 >= tmin) & (cmip6_t_fig2 <= tmax)

        # Build figure (high resolution, same style as extra_fig1)
        fig, ax = plt.subplots(figsize=(10.5, 4.6), dpi=300)

        # SST μ ± 2σ
        ax.fill_between(times_sst[m_sst_fig2],
                        (mu_sst_anom_fig2 - 2*sst_sigma_for_band_fig2)[m_sst_fig2],
                        (mu_sst_anom_fig2 + 2*sst_sigma_for_band_fig2)[m_sst_fig2],
                        alpha=0.20, facecolor=COLOR_BAND, edgecolor="none")
        ax.plot(times_sst[m_sst_fig2], mu_sst_anom_fig2[m_sst_fig2], lw=2.2, color=COLOR_SST, label="DL SST")

        # RAPID trend (centered to SST anomalies over overlap)
        try:
            ds_moc_rapid = xr.open_dataset('moc_vertical.nc')
            rapid_m = ds_moc_rapid['stream_function_mar'].resample(time='1M').mean()
            rapid_max = rapid_m.max(dim='depth')  # with Ekman
            t_r = np.arange(rapid_max.sizes["time"], dtype=float)
            y_r = rapid_max.values.astype(float)
            valid = np.isfinite(y_r)
            t_r = t_r[valid]; y_r = y_r[valid]
            if y_r.size >= 3:
                slope_r, intercept_r = np.polyfit(t_r, y_r, 1)
                trend_r = slope_r * t_r + intercept_r
                rapid_time_valid = rapid_max.time.values[valid]
                trend_da = xr.DataArray(trend_r, coords={'time': rapid_time_valid}, dims='time')

                # Center trend to SST anomalies over overlap
                sst_da = xr.DataArray(mu_sst_anom_fig2, coords={'time': times_sst}, dims='time')
                t0_ = max(sst_da.time.values.min(), trend_da.time.values.min())
                t1_ = min(sst_da.time.values.max(), trend_da.time.values.max())
                sst_w = sst_da.sel(time=np.s_[t0_:t1_])
                trend_w = trend_da.sel(time=np.s_[t0_:t1_]).interp(time=sst_w.time)

                mm = np.isfinite(sst_w.values) & np.isfinite(trend_w.values)
                shift = float(np.nanmean(sst_w.values[mm]) - np.nanmean(trend_w.values[mm])) if mm.sum() >= 3 else 0.0
                trend_plot = (trend_da + shift).sel(time=slice(TOP_X_RANGE[0], TOP_X_RANGE[1]))
                ax.plot(trend_plot.time.values[60:], trend_plot.values[60:],
                        lw=2.9, color=OKABE_ITO["verm"], label="RAPID trend", zorder=3)
        except Exception as e:
            print(f"[WARN] RAPID trend overlay skipped (extra_fig2): {e}")

        # CMIP6 mean (LPF10, anomalies 1900–1930) — solid grey line
        if np.any(m_cmip6_fig2):
            ax.plot(cmip6_t_fig2[m_cmip6_fig2], cmip6_anom_fig2[m_cmip6_fig2],
                    lw=2.2, color=COLOR_CMIP6, label="CMIP6", zorder=2)

        # SST index proxy: Caesar/HadISST ×3.8 (already LPF10 in file)
        try:
            # Load proxy with same baseline as figure (1900-1930)
            proxy_series_fig2 = load_proxy_series_nc(PROXY_NC, factor=3.8, baseline=(1900, 1930))

            # convert annual Series (index=year) → datetime (Dec-31 each year)
            proxy_dt_fig2 = pd.to_datetime(proxy_series_fig2.index.astype(str) + "-12-31")
            proxy_da_fig2 = xr.DataArray(proxy_series_fig2.values, coords={'time': proxy_dt_fig2}, dims='time')

            # restrict to plot window and interpolate to the SST monthly grid (purely for smoother overlay)
            proxy_win_fig2 = proxy_da_fig2.sel(time=slice(TOP_X_RANGE[0], TOP_X_RANGE[1]))
            proxy_i_fig2 = proxy_win_fig2.interp(time=xr.DataArray(times_sst, dims='time'))

            # plot
            ax.plot(proxy_i_fig2.time.values, proxy_i_fig2.values, 
                    lw=2.2, ls='--', color="0.4", label='SST index', zorder=1.5)
        except Exception as e:
            print(f"[WARN] SST index proxy overlay skipped (extra_fig2): {e}")

        # Axes and styling (same as extra_fig1)
        ax.set_xlim(*TOP_X_RANGE)
        ax.set_ylim(-3, 4)
        ax.set_ylabel("AMOC 26.5°N anomaly, Sv")
        box_axes(ax, lw=1.0, color="black")  # Full box around the graph
        ax.legend(frameon=False, ncol=2, loc="upper left")

        fig.tight_layout()
        out_extra2 = os.path.join(OUT_DIR, "extra_fig2_sst_vs_cmip6_mean.png")
        fig.savefig(out_extra2, dpi=1000, bbox_inches="tight")  # Very high resolution for poster
        plt.close(fig)
        print(f"[PLOT] Saved → {out_extra2}")

    except Exception as e:
        print(f"[WARN] Extra Figure 2 (SST vs CMIP6 mean) failed: {e}")
        import traceback
        traceback.print_exc()


    # -----------------------------
    # Figure 1: SST-NN + CMIP6 mean (+ RAPID trend, centered)
    # -----------------------------
    try:
        # Time series
        cmip6_t = pd.to_datetime(df["_time_dt64_"]).to_numpy()
        cmip6_y = lowpass_filter(cmip6_anom, 1/10, 5, 1, 20)  # ~10-year low-pass

        # Anomalies vs baseline window for SST-NN and CMIP6 (σ band unchanged)
        BASE_T0, BASE_T1 = "2004-01-01", "2023-12-31"
        mu_sst_anom   = _anomalize_to_window(times_sst, mu_sst, BASE_T0, BASE_T1)
        cmip6_y_anom  = _anomalize_to_window(cmip6_t, cmip6_y, BASE_T0, BASE_T1)
        sst_sigma_for_band = (tot_sst if USE_TOTAL_SIGMA else alea_sst)

        # Plot window
        m_sst = (times_sst >= TOP_X_RANGE[0]) & (times_sst <= TOP_X_RANGE[1])
        m6    = (cmip6_t   >= TOP_X_RANGE[0]) & (cmip6_t   <= TOP_X_RANGE[1])

        fig, ax = plt.subplots(figsize=(10, 4.4), dpi=170)

        # SST-NN (μ ± 2σ) in anomaly space
        ax.fill_between(times_sst[m_sst],
                        (mu_sst_anom - 2*sst_sigma_for_band)[m_sst],
                        (mu_sst_anom + 2*sst_sigma_for_band)[m_sst],
                        alpha=0.20)
        ax.plot(times_sst[m_sst], mu_sst_anom[m_sst], lw=2.2, label='DL SST ±2σ')



        # CMIP6 multi-model mean (same anomaly baseline)
        if np.any(m6):
            ax.plot(cmip6_t[m6], cmip6_y_anom[m6], lw=2.2, ls='--', label='CMIP6')

        # RAPID trend (linear fit; shift to match mean SST-NN anomaly on overlap)
        try:
            ds_moc_tr = xr.open_dataset('moc_vertical.nc')
            rapid_m   = ds_moc_tr['stream_function_mar'].resample(time='1M').mean()
            rapid_max = rapid_m.max(dim='depth')  # with Ekman; trend on monthly series

            # Linear trend on valid monthly samples (no extra LPF)
            t_r = np.arange(rapid_max.sizes["time"], dtype=float)
            y_r = rapid_max.values.astype(float)
            # Drop NaNs
            valid = np.isfinite(y_r)
            t_r = t_r[valid]; y_r = y_r[valid]
            if y_r.size >= 3:
                slope_r, intercept_r = np.polyfit(t_r, y_r, 1)
                trend_r = slope_r * t_r + intercept_r

                # Wrap trend as DataArray (time coord aligned with rapid_max)
                rapid_time_valid = rapid_max.time.values[valid]
                trend_da = xr.DataArray(trend_r, coords={'time': rapid_time_valid}, dims='time')

                # Shift trend to match mean SST-NN anomaly on overlapping times
                sst_da = xr.DataArray(mu_sst_anom, coords={'time': times_sst}, dims='time')
                # Overlapping time range
                t0 = max(sst_da.time.values.min(), trend_da.time.values.min())
                t1 = min(sst_da.time.values.max(), trend_da.time.values.max())
                sst_w   = sst_da.sel(time=np.s_[t0:t1])
                trend_w = trend_da.sel(time=np.s_[t0:t1]).interp(time=sst_w.time)

                mm = np.isfinite(sst_w.values) & np.isfinite(trend_w.values)
                if mm.sum() >= 3:
                    # shift so means match on overlap
                    shift = float(np.nanmean(sst_w.values[mm]) - np.nanmean(trend_w.values[mm]))
                else:
                    shift = 0.0

                # Plot shifted trend inside display window
                trend_plot = (trend_da + shift).sel(time=slice(TOP_X_RANGE[0], TOP_X_RANGE[1]))
                ax.plot(trend_plot.time.values[60:], trend_plot.values[60:], lw=2.2, color='black', label='RAPID trend')
        except Exception as e:
            print(f"[WARN] RAPID trend overlay (SST) skipped: {e}")


        
                # ---- PROXY: Caesar/HadISST ×3.8 (already LPF10 in file) ----
        try:
            # if the caller didn't run it yet, load now
            if 'proxy_series' not in locals():
                proxy_series = load_proxy_series_nc(PROXY_NC, factor=3.8, baseline=(2004, 2023))

            # convert annual Series (index=year) → datetime (Dec-31 each year)
            proxy_dt = pd.to_datetime(proxy_series.index.astype(str) + "-12-31")
            proxy_da = xr.DataArray(proxy_series.values, coords={'time': proxy_dt}, dims='time')

            # restrict to plot window and interpolate to the SST monthly grid (purely for smoother overlay)
            proxy_win = proxy_da.sel(time=slice(TOP_X_RANGE[0], TOP_X_RANGE[1]))
            proxy_i = proxy_win.interp(time=xr.DataArray(times_sst, dims='time'))

            # plot
            ax.plot(proxy_i.time.values, proxy_i.values, lw=2.2, ls='--', color="0.4", label='SST index')
        except Exception as e:
            print(f"[WARN] Proxy overlay skipped: {e}")



        ax.set_xlim(*TOP_X_RANGE)
        ax.set_ylim(-3, 4)
        ax.set_ylabel('AMOC, Sv')
        ax.legend(frameon=False, ncol=2, loc='upper left')
        for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
        fig.tight_layout()
        out1 = os.path.join(OUT_DIR, "fig_sst_nn_plus_cmip6_plus_rapidtrend.png")
        fig.savefig(out1, dpi=170, bbox_inches="tight")
        plt.close(fig)
        print(f"[PLOT] Saved → {out1}")
    except Exception as e:
        print(f"[WARN] Could not build SST+CMIP6(+RAPID trend) figure: {e}")


    # -----------------------------------------
    # Figure 2: SSH-NN (+Ekman) + RAPID array (with Ekman), large fonts, no title
    # -----------------------------------------
    try:
        # NN SSH (μ + Ekman − mean(Ekman)) already in amoc_ssh_mu
        ssh_sigma_for_band = (tot_ssh if USE_TOTAL_SIGMA else alea_ssh)
        m_ssh = (times_ssh >= BOTTOM_WINDOW[0]) & (times_ssh <= BOTTOM_WINDOW[1])



        # Full RAPID with Ekman, ~24-month smoothing (same as main pipeline)
        ds_moc_full = xr.open_dataset('moc_vertical.nc')
        rapid_full  = ds_moc_full['stream_function_mar'].resample(time='1M').mean()
        rapid_full_max = rapid_full.max(dim='depth')                   # Sv, with Ekman
        rapid_full_lp  = lowpass_filter(rapid_full_max, 1/24, 5, 1, 48)
        rapid_plot = rapid_full_lp

        fig, ax = plt.subplots(figsize=(6, 4.4), dpi=170)

        # SSH-NN (μ ± 2σ)
        ax.fill_between(times_ssh[m_ssh],
                        (amoc_ssh_mu - 2*ssh_sigma_for_band)[m_ssh],
                        (amoc_ssh_mu + 2*ssh_sigma_for_band)[m_ssh],
                        alpha=0.20, label=' ±2σ')
        ax.plot(times_ssh[m_ssh], amoc_ssh_mu[m_ssh], lw=2.2, label='SSH DL')

        # RAPID array (with Ekman)
        ax.plot(rapid_full_max.time[24:], rapid_plot[24:] - 17.0 , lw=2.2, color='black', label='RAPID array')


        # === SSH panel with RAPID + SST-on-SSH (both with Ekman) ===
        # Assumes: mu_ssh, times_ssh, (alea_ssh|tot_ssh), USE_TOTAL_SIGMA,
        #          lowpass_filter, PreprocessedDataset, get_bnn_predictions, device,
        #          BOTTOM_WINDOW, OUT_DIR are defined.

        # 1) Ekman aligned to SSH grid (reuse if present)
        if 'ek_vals' not in locals():
            ek_fn = "/home/am334/link_am334/ECCO_data/ekman_transport_26p5N_monthly_Sv.nc"
            ek = xr.open_dataset(ek_fn)["ekman_transport"].sortby("time")
            ssh_time_da = xr.DataArray(times_ssh, dims="time")
            ek_i = ek.interp(time=ssh_time_da).ffill("time").bfill("time")
            ek_vals = lowpass_filter(ek_i.values.astype(np.float64), 1/24, pad=48)
        ek_center = float(np.nanmean(ek_vals))

        # 2) SSH reconstruction (μ + Ekman − mean(Ekman))
        recon_ssh = mu_ssh + ek_vals - ek_center
        m_ssh = (times_ssh >= BOTTOM_WINDOW[0]) & (times_ssh <= BOTTOM_WINDOW[1])

        # 3) RAPID array (with Ekman), LPF~24 (apply same offset you used elsewhere)
        ds_moc_full = xr.open_dataset("moc_vertical.nc")
        rapid_full  = ds_moc_full["stream_function_mar"].resample(time="1M").mean()
        rapid_full_max = rapid_full.max(dim="depth")                # Sv, includes Ekman
        rapid_full_lp  = lowpass_filter(rapid_full_max, 1/24, 5, 1, 48)
        RAPID_OFFSET = -17.0
        rapid_t = rapid_full_max.time.values[24:]
        rapid_y = (rapid_full_lp[24:] + RAPID_OFFSET)

        # 4) Ensure SST reconstruction exists; if not, compute it from "tos" weights
        try:
            mu_sst, times_sst  # noqa
        except NameError:
            sst_file = "processed_HadISST_sst.nc"
            sst_ds   = PreprocessedDataset(
                sst_file, variable="__xarray_dataarray_variable__",
                lpf="LPF24", order=5, fs=1, monthly=True, minus_basin_mean=False
            )
            x0 = sst_ds[0]
            if x0.ndim == 2: x0 = x0[np.newaxis]
            in_ch_sst = x0.shape[0]
            sst_pattern = os.path.join(
                WEIGHTS_ROOT_DIR, "bnn_test_god_bless_it_work_all_vars_all_models_test_24",
                "tos", "*", "best_stage2_joint.pt"
            )
            _, _, mu_sst, alea_sst, epi_sst, tot_sst = get_bnn_predictions(
                sst_ds, sst_pattern, in_ch_sst, 1, device=device, batch_size=16
            )
            times_sst = xr.open_dataset(sst_file).coords["time"].values

        # 5) Interpolate SST μ to SSH timestamps over overlap; add Ekman same way as SSH
        sst_da     = xr.DataArray(mu_sst, coords={"time": times_sst}, dims="time")
        ssh_timeDA = xr.DataArray(times_ssh, dims="time")
        overlap_start = max(sst_da.time.values.min(), ssh_timeDA.values.min())
        overlap_end   = min(sst_da.time.values.max(), ssh_timeDA.values.max())
        sst_on_ssh = sst_da.sel(time=slice(overlap_start, overlap_end)).interp(time=ssh_timeDA)

        recon_sst_on_ssh = sst_on_ssh.values + ek_vals - ek_center  # +Ekman & centered
        m_overlap = (
            (ssh_timeDA.values >= overlap_start) & (ssh_timeDA.values <= overlap_end) &
            (ssh_timeDA.values >= BOTTOM_WINDOW[0]) & (ssh_timeDA.values <= BOTTOM_WINDOW[1])
        )

        # 6) Plot
        fig, ax = plt.subplots(figsize=(10, 4.4), dpi=170)

        # Fonts 16
        ax.tick_params(axis="both", labelsize=16)
        ax.set_xlabel("Time", fontsize=16)
        ax.set_ylabel("AMOC, Sv", fontsize=16)

        # SSH band ±2σ
        ssh_sigma_for_band = (tot_ssh if USE_TOTAL_SIGMA else alea_ssh)
        ax.fill_between(
            times_ssh[m_ssh],
            (mu_ssh - 2*ssh_sigma_for_band + ek_vals - ek_center)[m_ssh],
            (mu_ssh + 2*ssh_sigma_for_band + ek_vals - ek_center)[m_ssh],
            alpha=0.20, label="SSH→AMOC ±2σ"
        )
        ax.plot(times_ssh[m_ssh], recon_ssh[m_ssh], lw=2.2, label="SSH→AMOC μ (BNN, +Ekman)")

        # RAPID array (with Ekman)
        ax.plot(rapid_t, rapid_y, lw=2.2, color="black", label="RAPID array")

        # SST→AMOC (on SSH grid, with Ekman; overlap only)
        ax.plot(    
            ssh_timeDA.values[m_overlap],
            recon_sst_on_ssh[m_overlap],
            lw=2.2, color="0.35", label="SST→AMOC μ (BNN, +Ekman)"
        )

        # Axes & save
        ax.set_xlim(*BOTTOM_WINDOW)
        ax.set_ylim(-5, 5)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        ax.legend(frameon=False, loc="upper left", ncol=2)

        os.makedirs(OUT_DIR, exist_ok=True)
        out_path = os.path.join(OUT_DIR, "fig_ssh_nn_plus_rapid_plus_sst.png")
        fig.tight_layout()
        fig.savefig(out_path, dpi=170, bbox_inches="tight")
        plt.close(fig)
        print(f"[PLOT] Saved → {out_path}")

        



        ax.set_xlim(*BOTTOM_WINDOW)
        ax.set_ylim(-5, 5)
        ax.set_ylabel('AMOC, Sv')
        #ax.set_xlabel('Time')
        ax.legend(frameon=False, ncol=2, loc='upper left')
        for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
        fig.tight_layout()
        out2 = os.path.join(OUT_DIR, "fig_ssh_nn_plus_rapid.png")
        fig.savefig(out2, dpi=170, bbox_inches="tight")
        plt.close(fig)
        print(f"[PLOT] Saved → {out2}")
    except Exception as e:
        print(f"[WARN] Could not build SSH+RAPID figure: {e}")

    # ===========================
    # NEW FIGURES: SSH vs RAPID (with and without Ekman) + SST LPF24
    # ===========================
    try:
        print("\n[INFO] Building SSH vs RAPID comparison figures (with/without Ekman)...")
        
        # Time window for these figures (1992-2024)
        SSH_RAPID_WINDOW = (np.datetime64('1992-01-01'), np.datetime64('2024-12-31'))
        
        # --- Load SST dataset with LPF24 and get BNN predictions ---
        print("[INFO] Loading SST with LPF24 for comparison figures...")
        sst_file_lpf24 = "processed_HadISST_sst.nc"
        sst_ds_lpf24 = PreprocessedDataset(
            sst_file_lpf24, variable="__xarray_dataarray_variable__",
            lpf="LPF24", order=5, fs=1, monthly=True, minus_basin_mean=False
        )
        x0_sst_lpf24 = sst_ds_lpf24[0]
        if x0_sst_lpf24.ndim == 2: x0_sst_lpf24 = x0_sst_lpf24[np.newaxis]
        in_ch_sst_lpf24 = x0_sst_lpf24.shape[0]
        
        # Weights for SST LPF24 (same dir as SSH but "tos" instead of "zos_minus_basin_mean")
        sst_lpf24_pattern = os.path.join(
            WEIGHTS_ROOT_DIR, "bnn_test_god_bless_it_work_all_vars_all_models_test_24",
            "tos", "*", "best_stage2_joint.pt"
        )
        
        _, _, mu_sst_lpf24, alea_sst_lpf24, epi_sst_lpf24, tot_sst_lpf24 = get_bnn_predictions(
            sst_ds_lpf24, sst_lpf24_pattern, in_ch_sst_lpf24, 1, device=device, batch_size=16
        )
        times_sst_lpf24 = xr.open_dataset(sst_file_lpf24).coords["time"].values
        sst_lpf24_sigma = (tot_sst_lpf24 if USE_TOTAL_SIGMA else alea_sst_lpf24)
        
        # --- Load Ekman transport and align to both SSH and RAPID grids ---
        ek_fn = "/home/am334/link_am334/ECCO_data/ekman_transport_26p5N_monthly_Sv.nc"
        ek_ds = xr.open_dataset(ek_fn)
        ek_raw = ek_ds["ekman_transport"].sortby("time")
        
        # --- RAPID data (with Ekman) ---
        ds_moc = xr.open_dataset('moc_vertical.nc')
        rapid_monthly = ds_moc['stream_function_mar'].resample(time='1M').mean()
        rapid_with_ek = rapid_monthly.max(dim='depth')  # RAPID with Ekman
        rapid_times = rapid_with_ek.time.values
        
        # --- Interpolate Ekman to RAPID timestamps ---
        rapid_time_da = xr.DataArray(rapid_times, dims="time")
        ek_on_rapid = ek_raw.interp(time=rapid_time_da).ffill("time").bfill("time")
        ek_on_rapid_lp = lowpass_filter(ek_on_rapid.values.astype(np.float64), 1/24, 5, 1, 48)
        
        # --- Interpolate Ekman to SST timestamps ---
        sst_time_da = xr.DataArray(times_sst_lpf24, dims="time")
        ek_on_sst = ek_raw.interp(time=sst_time_da).ffill("time").bfill("time")
        ek_on_sst_lp = lowpass_filter(ek_on_sst.values.astype(np.float64), 1/24, 5, 1, 48)
        ek_on_sst_center = float(np.nanmean(ek_on_sst_lp))
        
        # --- RAPID without Ekman = RAPID_with_Ekman - Ekman ---
        rapid_wo_ek_raw = rapid_with_ek.values - ek_on_rapid_lp
        
        # --- Apply LPF24 to both RAPID versions ---
        rapid_with_ek_lp = lowpass_filter(rapid_with_ek.values.astype(np.float64), 1/24, 5, 1, 48)
        rapid_wo_ek_lp = lowpass_filter(rapid_wo_ek_raw.astype(np.float64), 1/24, 5, 1, 48)
        
        # --- SSH reconstruction (mu_ssh is already without Ekman) ---
        # ek_vals and ek_center should already be defined from earlier
        ssh_no_ekman = mu_ssh  # BNN output, no Ekman
        ssh_with_ekman = mu_ssh + ek_vals - ek_center  # with Ekman added back
        
        # --- SST LPF24 reconstruction (with and without Ekman) ---
        sst_lpf24_no_ekman = mu_sst_lpf24  # BNN output, no Ekman
        sst_lpf24_with_ekman = mu_sst_lpf24 + ek_on_sst_lp - ek_on_sst_center  # with Ekman added back
        
        # --- Offset for visual alignment (center RAPID to SSH mean over overlap) ---
        m_ssh_win = (times_ssh >= SSH_RAPID_WINDOW[0]) & (times_ssh <= SSH_RAPID_WINDOW[1])
        m_rapid_win = (rapid_times >= SSH_RAPID_WINDOW[0]) & (rapid_times <= SSH_RAPID_WINDOW[1])
        m_sst_lpf24_win = (times_sst_lpf24 >= SSH_RAPID_WINDOW[0]) & (times_sst_lpf24 <= SSH_RAPID_WINDOW[1])
        
        # Compute offsets for both cases
        ssh_no_ek_mean = float(np.nanmean(ssh_no_ekman[m_ssh_win]))
        rapid_wo_ek_mean = float(np.nanmean(rapid_wo_ek_lp[m_rapid_win]))
        offset_no_ek = ssh_no_ek_mean - rapid_wo_ek_mean
        
        ssh_with_ek_mean = float(np.nanmean(ssh_with_ekman[m_ssh_win]))
        rapid_with_ek_mean = float(np.nanmean(rapid_with_ek_lp[m_rapid_win]))
        offset_with_ek = ssh_with_ek_mean - rapid_with_ek_mean
        
        # SST offset (align to SSH mean)
        sst_no_ek_mean = float(np.nanmean(sst_lpf24_no_ekman[m_sst_lpf24_win]))
        offset_sst_no_ek = ssh_no_ek_mean - sst_no_ek_mean
        
        sst_with_ek_mean = float(np.nanmean(sst_lpf24_with_ekman[m_sst_lpf24_win]))
        offset_sst_with_ek = ssh_with_ek_mean - sst_with_ek_mean
        
        # Color for SST
        SST_COLOR = OKABE_ITO["orange"]
        
        # ========================================
        # FIGURE 1: SSH (no Ekman) vs RAPID (no Ekman) + SST (no Ekman)
        # ========================================
        fig1, ax1 = plt.subplots(figsize=(10, 4.8), dpi=170)
        
        # SSH reconstruction (no Ekman) with ±2σ band
        ssh_sigma = (tot_ssh if USE_TOTAL_SIGMA else alea_ssh)
        ax1.fill_between(
            times_ssh[m_ssh_win],
            (ssh_no_ekman - 2*ssh_sigma)[m_ssh_win],
            (ssh_no_ekman + 2*ssh_sigma)[m_ssh_win],
            alpha=0.20, facecolor=RECON_BLUE, edgecolor="none",
            label="SSH→AMOC ±2σ"
        )
        ax1.plot(times_ssh[m_ssh_win], ssh_no_ekman[m_ssh_win], 
                 lw=2.2, color=RECON_BLUE, label="SSH→AMOC (BNN, no Ekman)")
        
        # SST LPF24 reconstruction (no Ekman), shifted to match SSH mean
        ax1.plot(times_sst_lpf24[m_sst_lpf24_win], (sst_lpf24_no_ekman + offset_sst_no_ek)[m_sst_lpf24_win],
                 lw=2.2, color=SST_COLOR, ls="--", label="SST→AMOC (BNN LPF24, no Ekman)")
        
        # RAPID without Ekman (LPF24), shifted to match SSH mean
        ax1.plot(rapid_times[m_rapid_win], (rapid_wo_ek_lp + offset_no_ek)[m_rapid_win],
                 lw=2.2, color=RAPID_BLACK, label="RAPID (no Ekman, LPF24)")
        
        ax1.set_xlim(*SSH_RAPID_WINDOW)
        ax1.set_ylim(-5, 5)
        ax1.set_xlabel("Time", fontsize=16)
        ax1.set_ylabel("AMOC (Sv)", fontsize=16)
        ax1.tick_params(axis="both", labelsize=16)
        for sp in ("top", "right"): ax1.spines[sp].set_visible(False)
        ax1.legend(frameon=False, loc="upper left", ncol=2, fontsize=12)
        ax1.grid(True, which="major", alpha=0.12, linewidth=0.6)
        ax1.set_title("SSH + SST vs RAPID — WITHOUT Ekman (LPF24)", fontsize=14, fontweight="bold")
        
        fig1.tight_layout()
        out_no_ek = os.path.join(OUT_DIR, "fig_ssh_vs_rapid_NO_ekman.png")
        fig1.savefig(out_no_ek, dpi=170, bbox_inches="tight")
        plt.close(fig1)
        print(f"[PLOT] Saved → {out_no_ek}")
        
        # ========================================
        # FIGURE 2: SSH (with Ekman) vs RAPID (with Ekman) + SST (with Ekman)
        # ========================================
        fig2, ax2 = plt.subplots(figsize=(10, 4.8), dpi=170)
        
        # SSH reconstruction (with Ekman) with ±2σ band
        ax2.fill_between(
            times_ssh[m_ssh_win],
            (ssh_with_ekman - 2*ssh_sigma)[m_ssh_win],
            (ssh_with_ekman + 2*ssh_sigma)[m_ssh_win],
            alpha=0.20, facecolor=RECON_BLUE, edgecolor="none",
            label="SSH→AMOC ±2σ"
        )
        ax2.plot(times_ssh[m_ssh_win], ssh_with_ekman[m_ssh_win],
                 lw=2.2, color=RECON_BLUE, label="SSH→AMOC (BNN, +Ekman)")
        
        # SST LPF24 reconstruction (with Ekman), shifted to match SSH mean
        ax2.plot(times_sst_lpf24[m_sst_lpf24_win], (sst_lpf24_with_ekman + offset_sst_with_ek)[m_sst_lpf24_win],
                 lw=2.2, color=SST_COLOR, ls="--", label="SST→AMOC (BNN LPF24, +Ekman)")
        
        # RAPID with Ekman (LPF24), shifted to match SSH mean
        ax2.plot(rapid_times[m_rapid_win], (rapid_with_ek_lp + offset_with_ek)[m_rapid_win],
                 lw=2.2, color=RAPID_BLACK, label="RAPID (with Ekman, LPF24)")
        
        ax2.set_xlim(*SSH_RAPID_WINDOW)
        ax2.set_ylim(-5, 5)
        ax2.set_xlabel("Time", fontsize=16)
        ax2.set_ylabel("AMOC (Sv)", fontsize=16)
        ax2.tick_params(axis="both", labelsize=16)
        for sp in ("top", "right"): ax2.spines[sp].set_visible(False)
        ax2.legend(frameon=False, loc="upper left", ncol=2, fontsize=12)
        ax2.grid(True, which="major", alpha=0.12, linewidth=0.6)
        ax2.set_title("SSH + SST vs RAPID — WITH Ekman (LPF24)", fontsize=14, fontweight="bold")
        
        fig2.tight_layout()
        out_with_ek = os.path.join(OUT_DIR, "fig_ssh_vs_rapid_WITH_ekman.png")
        fig2.savefig(out_with_ek, dpi=170, bbox_inches="tight")
        plt.close(fig2)
        print(f"[PLOT] Saved → {out_with_ek}")
        
    except Exception as e:
        import traceback
        print(f"[WARN] Could not build SSH vs RAPID (with/without Ekman) figures: {e}")
        traceback.print_exc()

    # ================================================================
    #  MULTI-SST COMPARISON: AMOC ANOMALIES from HadISST, COBE-SST2, ERSST V5
    #  LPF10 (10-year = 120 months), anomalies relative to 1900-1930
    # ================================================================
    print("\n" + "="*70)
    print("  AMOC ANOMALIES FROM MULTIPLE SST PRODUCTS (LPF10 = 10-year)")
    print("="*70 + "\n")
    
    SST_DATASETS = {
        "HadISST": "processed_HadISST_sst.nc",
        "COBE-SST2": "processed_COBESST2_sst.nc",
        "ERSST V5": "processed_ERSST5_sst.nc",
    }
    
    SST_COLORS = {
        "HadISST": OKABE_ITO["blue"],
        "COBE-SST2": OKABE_ITO["orange"],
        "ERSST V5": OKABE_ITO["green"],
    }
    
    SST_LINESTYLES = {
        "HadISST": "-",
        "COBE-SST2": "--",
        "ERSST V5": "-.",
    }
    
    # Reference period for anomalies (same for all datasets)
    ANOM_REF_START = np.datetime64('1900-01-01')
    ANOM_REF_END = np.datetime64('1930-12-31')
    
    # Use LPF10 = 10-year low pass (120 months)
    lpf_choice = "LPF10"  # Maps to 1/120 cutoff in obs_datasets.py
    
    # Weights for 120-month (10-year) filter
    sst_weights_pattern = os.path.join(
        WEIGHTS_ROOT_DIR, "bnn_test_god_bless_it_work_all_vars_all_models_test_120_y_new",
        "tos", "*", "best_stage2_joint.pt"
    )
    
    try:
        print(f"[INFO] Processing with {lpf_choice} (10-year low-pass)...")
        print(f"[INFO] Anomaly reference period: {ANOM_REF_START} to {ANOM_REF_END}")
        
        multi_sst_results = {}
        
        for sst_name, sst_file in SST_DATASETS.items():
            print(f"  → Loading {sst_name} ({sst_file})...")
            
            try:
                sst_ds = PreprocessedDataset(
                    sst_file, variable="__xarray_dataarray_variable__",
                    lpf=lpf_choice, order=5, fs=1, monthly=True, minus_basin_mean=False
                )
                x0_sst = sst_ds[0]
                if x0_sst.ndim == 2:
                    x0_sst = x0_sst[np.newaxis]
                in_ch_sst = x0_sst.shape[0]
                
                _, _, mu_sst, alea_sst, epi_sst, tot_sst = get_bnn_predictions(
                    sst_ds, sst_weights_pattern, in_ch_sst, 1, device=device, batch_size=16
                )
                
                # Get times from the NetCDF file directly
                times_sst = xr.open_dataset(sst_file).coords["time"].values
                
                # Compute anomaly relative to reference period
                m_ref = (times_sst >= ANOM_REF_START) & (times_sst <= ANOM_REF_END)
                if m_ref.sum() > 0:
                    ref_mean = float(np.nanmean(mu_sst[m_ref]))
                else:
                    print(f"    [WARN] No data in reference period for {sst_name}, using full mean")
                    ref_mean = float(np.nanmean(mu_sst))
                
                mu_anom = mu_sst - ref_mean
                
                multi_sst_results[sst_name] = {
                    "mu": mu_sst,
                    "mu_anom": mu_anom,
                    "ref_mean": ref_mean,
                    "alea": alea_sst,
                    "epi": epi_sst,
                    "tot": tot_sst,
                    "times": times_sst,
                }
                print(f"    ✓ {sst_name}: {len(times_sst)} months, ref_mean={ref_mean:.2f} Sv")
                
            except Exception as e:
                print(f"    ✗ {sst_name} failed: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if len(multi_sst_results) == 0:
            print(f"[WARN] No SST datasets loaded, skipping...")
        else:
            # ── Load RAPID and compute anomaly ────────────────────────
            rapid_anom = None
            rapid_times_raw = None
            try:
                ds_moc_rapid = xr.open_dataset('moc_vertical.nc')
                rapid_m = ds_moc_rapid['stream_function_mar'].resample(time='1M').mean()
                rapid_max = rapid_m.max(dim='depth')
                rapid_times_raw = rapid_max.time.values
                rapid_vals_raw = rapid_max.values.astype(float)
                
                # Apply 10-year LPF
                rapid_lp = lowpass_filter(rapid_vals_raw.astype(np.float64), 1/120, 5, 1, 120)
                
                # RAPID doesn't cover 1900-1930, so we align it to the SST anomalies over overlap
                # Just center RAPID to have zero mean over its full extent for visualization
                rapid_anom = rapid_lp - np.nanmean(rapid_lp)
                ds_moc_rapid.close()
                print(f"    ✓ RAPID loaded: {len(rapid_times_raw)} months")
            except Exception as e:
                print(f"[WARN] RAPID not loaded: {e}")
            
            # ── Plot ANOMALY comparison figure ────────────────────────
            fig, ax = plt.subplots(figsize=(12, 5), dpi=170)
            
            # Plot each SST product anomaly
            for sst_name, res in multi_sst_results.items():
                mu_anom = res["mu_anom"]
                times = res["times"]
                sigma = res["tot"] if USE_TOTAL_SIGMA else res["alea"]
                
                # Uncertainty band
                ax.fill_between(
                    times,
                    mu_anom - 2*sigma,
                    mu_anom + 2*sigma,
                    alpha=0.15, facecolor=SST_COLORS[sst_name], edgecolor="none"
                )
                
                # Mean line
                ax.plot(times, mu_anom, lw=2.0, color=SST_COLORS[sst_name],
                        ls=SST_LINESTYLES[sst_name], label=f"{sst_name}")
            
            ax.axhline(0, color='gray', ls='--', lw=0.8, alpha=0.5)
            ax.set_xlim(np.datetime64('1870-01-01'), np.datetime64('2025-01-01'))
            ax.set_ylim(-4, 4)
            ax.set_xlabel("Time", fontsize=14)
            ax.set_ylabel("AMOC anomaly (Sv)", fontsize=14)
            ax.tick_params(axis="both", labelsize=12)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            ax.legend(frameon=False, loc="upper left", ncol=2, fontsize=11)
            
            fig.tight_layout()
            out_multi = os.path.join(OUT_DIR, "amoc_multi_sst_anomaly_lpf10.png")
            fig.savefig(out_multi, dpi=200, bbox_inches="tight")
            plt.close(fig)
            print(f"[PLOT] Saved → {out_multi}")

            # ── HadISST-only plot (standard model) ─────────────────────────
            if "HadISST" in multi_sst_results:
                res_h = multi_sst_results["HadISST"]
                fig_h, ax_h = plt.subplots(figsize=(12, 5), dpi=170)
                sigma_h = res_h["tot"] if USE_TOTAL_SIGMA else res_h["alea"]
                ax_h.fill_between(res_h["times"], res_h["mu_anom"] - 2*sigma_h, res_h["mu_anom"] + 2*sigma_h,
                                 alpha=0.20, facecolor=SST_COLORS["HadISST"], edgecolor="none")
                ax_h.plot(res_h["times"], res_h["mu_anom"], lw=2.0, color=SST_COLORS["HadISST"], label="HadISST (standard)")
                ax_h.axhline(0, color='gray', ls='--', lw=0.8, alpha=0.5)
                ax_h.set_xlim(np.datetime64('1870-01-01'), np.datetime64('2025-01-01'))
                ax_h.set_ylim(-4, 4)
                ax_h.set_xlabel("Time", fontsize=14)
                ax_h.set_ylabel("AMOC anomaly (Sv)", fontsize=14)
                ax_h.tick_params(axis="both", labelsize=12)
                for sp in ("top", "right"): ax_h.spines[sp].set_visible(False)
                ax_h.legend(frameon=False, loc="upper left", fontsize=11)
                fig_h.tight_layout()
                out_hadisst = os.path.join(OUT_DIR, "amoc_hadisst_only_lpf10.png")
                fig_h.savefig(out_hadisst, dpi=200, bbox_inches="tight")
                plt.close(fig_h)
                print(f"[PLOT] Saved → {out_hadisst}")

            # ── HadISST vs COBE only (extra_fig1c style, no CMIP6) ─────────────
            if "HadISST" in multi_sst_results and "COBE-SST2" in multi_sst_results:
                from scipy.signal import butter as _butter_hc, sosfiltfilt as _sos_hc
                BASE_T0_1c, BASE_T1_1c = "1900-01-01", "1940-12-31"
                sos_monthly_hc = _butter_hc(5, 1/120, btype='low', output='sos', fs=1)
                tmin, tmax = TOP_X_RANGE

                def _anom_1900_1940(t_vals, mu_vals):
                    t_vals = pd.to_datetime(t_vals).to_numpy()
                    m = (t_vals >= np.datetime64(BASE_T0_1c)) & (t_vals <= np.datetime64(BASE_T1_1c))
                    base = float(np.nanmean(mu_vals[m])) if np.any(m) else float(np.nanmean(mu_vals))
                    return mu_vals - base

                fig_hc, ax_hc = plt.subplots(figsize=(10.5, 4.6), dpi=300)
                for sst_name in ["HadISST", "COBE-SST2"]:
                    res = multi_sst_results[sst_name]
                    times_hc = res["times"]
                    mu = res["mu"]
                    sigma = res["tot"] if USE_TOTAL_SIGMA else res["alea"]
                    anom = _anom_1900_1940(times_hc, mu)
                    anom_filt = _sos_hc(sos_monthly_hc, anom)
                    m_plot = (times_hc >= tmin) & (times_hc <= tmax)
                    ax_hc.fill_between(times_hc[m_plot],
                                       (anom_filt - 2*sigma)[m_plot],
                                       (anom_filt + 2*sigma)[m_plot],
                                       alpha=0.20, facecolor=SST_COLORS[sst_name], edgecolor="none")
                    ax_hc.plot(times_hc[m_plot], anom_filt[m_plot], lw=2.2, color=SST_COLORS[sst_name], label=sst_name)
                ax_hc.set_xlim(*TOP_X_RANGE)
                ax_hc.set_ylim(-3, 4)
                ax_hc.set_ylabel("AMOC 26.5°N anomaly, Sv")
                box_axes(ax_hc, lw=1.0, color="black")
                ax_hc.legend(frameon=False, ncol=1, loc="upper left", handlelength=2.0)
                fig_hc.tight_layout()
                out_hc = os.path.join(OUT_DIR, "extra_fig_hadisst_vs_cobe.png")
                fig_hc.savefig(out_hc, dpi=1000, bbox_inches="tight")
                plt.close(fig_hc)
                print(f"[PLOT] Saved → {out_hc}")
            
            # ── Save data to NetCDF ────────────────────────────────────
            for sst_name, res in multi_sst_results.items():
                nc_out = os.path.join(OUT_DIR, f"amoc_anom_from_{sst_name.replace(' ', '_').lower()}_lpf10.nc")
                ds_out = xr.Dataset({
                    "amoc_mean": (["time"], res["mu"]),
                    "amoc_anomaly": (["time"], res["mu_anom"]),
                    "amoc_alea_std": (["time"], res["alea"]),
                    "amoc_epi_std": (["time"], res["epi"]),
                    "amoc_tot_std": (["time"], res["tot"]),
                }, coords={"time": res["times"]})
                ds_out.attrs["source_sst"] = sst_name
                ds_out.attrs["lpf"] = "LPF10 (10-year = 120 months)"
                ds_out.attrs["reference_period"] = "1900-01-01 to 1930-12-31"
                ds_out.attrs["reference_mean_sv"] = res["ref_mean"]
                ds_out.to_netcdf(nc_out)
                print(f"[DATA] Saved → {nc_out}")
            
            # ── Compute and save trend summary for each SST product ────────
            print("\n[INFO] Computing trends for each SST product...")
            
            # Define trend windows
            TREND_WINDOWS = {
                "1900-2000": (np.datetime64('1900-01-01'), np.datetime64('2000-12-31')),
                "1900-2024": (np.datetime64('1900-01-01'), np.datetime64('2024-12-31')),
                "2004-2024": (np.datetime64('2004-01-01'), np.datetime64('2024-12-31')),
                "1950-2024": (np.datetime64('1950-01-01'), np.datetime64('2024-12-31')),
            }
            
            # Trend methods: HAC, MBB (primary), GLS bootstrap
            TREND_METHODS = ["hac", "mbb", "gls_boot"]
            PRIMARY_METHOD = "mbb"
            
            trend_rows = []
            for sst_name, res in multi_sst_results.items():
                mu = res["mu"]
                sigma = res["tot"] if USE_TOTAL_SIGMA else res["alea"]
                times = res["times"]
                
                for win_name, (t0, t1) in TREND_WINDOWS.items():
                    m_win = (times >= t0) & (times <= t1)
                    if m_win.sum() < 24:  # Need at least 2 years
                        continue
                    
                    y_win = mu[m_win]
                    sig_win = sigma[m_win]
                    t_idx = np.arange(m_win.sum(), dtype=float)
                    factor = 12 * 100  # monthly slope → Sv/century
                    
                    for method in TREND_METHODS:
                        try:
                            slope, intercept, lo, hi, info, label, key = compute_trend_ci(
                                y_win, sig_win, t_idx=t_idx, method=method,
                                hint_block_len=120, hac_L_floor=120
                            )
                            trend_sv = slope * factor
                            lo_sv = lo * factor
                            hi_sv = hi * factor
                            
                            trend_rows.append({
                                "SST_Product": sst_name,
                                "Window": win_name,
                                "Method": method.upper(),
                                "Trend_Sv_century": f"{trend_sv:.3f}",
                                "CI_95_lo": f"{lo_sv:.3f}",
                                "CI_95_hi": f"{hi_sv:.3f}",
                                "N_months": int(m_win.sum()),
                            })
                            
                            if method == PRIMARY_METHOD:
                                print(f"    {sst_name} [{win_name}]: {trend_sv:.2f} [{lo_sv:.2f}, {hi_sv:.2f}] Sv/century (MBB)")
                        except Exception as e:
                            print(f"    {sst_name} [{win_name}] ({method}): failed - {e}")
            
            # Save all trends to CSV
            if trend_rows:
                csv_out = os.path.join(OUT_DIR, "multi_sst_trend_summary.csv")
                with open(csv_out, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=trend_rows[0].keys())
                    writer.writeheader()
                    writer.writerows(trend_rows)
                print(f"[CSV] Saved → {csv_out}")
                
    except Exception as e:
        import traceback
        print(f"[WARN] Multi-SST comparison failed: {e}")
        traceback.print_exc()

    # ================================================================
    #  SAME MULTI-SST CHECK BUT WITH DETRENDED MODEL WEIGHTS
    #  <weights_root>/bnn_test_god_bless_it_work_all_vars_all_models_test_120_y_detrended
    # ================================================================
    print("\n" + "="*70)
    print("  AMOC ANOMALIES FROM MULTIPLE SST (LPF10) — DETRENDED MODEL")
    print("="*70 + "\n")

    sst_weights_pattern_detrended = os.path.join(
        WEIGHTS_ROOT_DIR, "bnn_test_god_bless_it_work_all_vars_all_models_test_120_y_detrended",
        "tos", "*", "best_stage2_joint.pt"
    )

    try:
        print(f"[INFO] Processing with {lpf_choice} (10-year low-pass) — DETRENDED weights...")
        print(f"[INFO] Anomaly reference period: {ANOM_REF_START} to {ANOM_REF_END}")

        multi_sst_results_detrended = {}

        for sst_name, sst_file in SST_DATASETS.items():
            print(f"  → Loading {sst_name} ({sst_file}) [detrended model]...")

            try:
                sst_ds = PreprocessedDataset(
                    sst_file, variable="__xarray_dataarray_variable__",
                    lpf=lpf_choice, order=5, fs=1, monthly=True, minus_basin_mean=False
                )
                x0_sst = sst_ds[0]
                if x0_sst.ndim == 2:
                    x0_sst = x0_sst[np.newaxis]
                in_ch_sst = x0_sst.shape[0]

                _, _, mu_sst, alea_sst, epi_sst, tot_sst = get_bnn_predictions(
                    sst_ds, sst_weights_pattern_detrended, in_ch_sst, 1, device=device, batch_size=16
                )

                times_sst = xr.open_dataset(sst_file).coords["time"].values

                m_ref = (times_sst >= ANOM_REF_START) & (times_sst <= ANOM_REF_END)
                if m_ref.sum() > 0:
                    ref_mean = float(np.nanmean(mu_sst[m_ref]))
                else:
                    print(f"    [WARN] No data in reference period for {sst_name}, using full mean")
                    ref_mean = float(np.nanmean(mu_sst))

                mu_anom = mu_sst - ref_mean

                multi_sst_results_detrended[sst_name] = {
                    "mu": mu_sst,
                    "mu_anom": mu_anom,
                    "ref_mean": ref_mean,
                    "alea": alea_sst,
                    "epi": epi_sst,
                    "tot": tot_sst,
                    "times": times_sst,
                }
                print(f"    ✓ {sst_name} (detrended): {len(times_sst)} months, ref_mean={ref_mean:.2f} Sv")

            except Exception as e:
                print(f"    ✗ {sst_name} failed: {e}")
                import traceback
                traceback.print_exc()
                continue

        if len(multi_sst_results_detrended) == 0:
            print(f"[WARN] No SST datasets loaded for detrended model, skipping...")
        else:
            fig, ax = plt.subplots(figsize=(12, 5), dpi=170)

            for sst_name, res in multi_sst_results_detrended.items():
                mu_anom = res["mu_anom"]
                times = res["times"]
                sigma = res["tot"] if USE_TOTAL_SIGMA else res["alea"]

                ax.fill_between(
                    times,
                    mu_anom - 2*sigma,
                    mu_anom + 2*sigma,
                    alpha=0.15, facecolor=SST_COLORS[sst_name], edgecolor="none"
                )
                ax.plot(times, mu_anom, lw=2.0, color=SST_COLORS[sst_name],
                        ls=SST_LINESTYLES[sst_name], label=f"{sst_name}")

            ax.axhline(0, color='gray', ls='--', lw=0.8, alpha=0.5)
            ax.set_xlim(np.datetime64('1870-01-01'), np.datetime64('2025-01-01'))
            ax.set_ylim(-4, 4)
            ax.set_xlabel("Time", fontsize=14)
            ax.set_ylabel("AMOC anomaly (Sv)", fontsize=14)
            ax.tick_params(axis="both", labelsize=12)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            ax.legend(frameon=False, loc="upper left", ncol=2, fontsize=11)
            ax.set_title("AMOC from SST (LPF10) — DL model trained on DETRENDED data", fontsize=12, fontweight="bold")

            fig.tight_layout()
            out_multi_det = os.path.join(OUT_DIR, "amoc_multi_sst_anomaly_lpf10_detrended.png")
            fig.savefig(out_multi_det, dpi=200, bbox_inches="tight")
            plt.close(fig)
            print(f"[PLOT] Saved → {out_multi_det}")

            # ── HadISST-only plot (detrended model) ─────────────────────────
            if "HadISST" in multi_sst_results_detrended:
                res_hd = multi_sst_results_detrended["HadISST"]
                fig_hd, ax_hd = plt.subplots(figsize=(12, 5), dpi=170)
                sigma_hd = res_hd["tot"] if USE_TOTAL_SIGMA else res_hd["alea"]
                ax_hd.fill_between(res_hd["times"], res_hd["mu_anom"] - 2*sigma_hd, res_hd["mu_anom"] + 2*sigma_hd,
                                  alpha=0.20, facecolor=SST_COLORS["HadISST"], edgecolor="none")
                ax_hd.plot(res_hd["times"], res_hd["mu_anom"], lw=2.0, color=SST_COLORS["HadISST"], label="HadISST (detrended model)")
                ax_hd.axhline(0, color='gray', ls='--', lw=0.8, alpha=0.5)
                ax_hd.set_xlim(np.datetime64('1870-01-01'), np.datetime64('2025-01-01'))
                ax_hd.set_ylim(-4, 4)
                ax_hd.set_xlabel("Time", fontsize=14)
                ax_hd.set_ylabel("AMOC anomaly (Sv)", fontsize=14)
                ax_hd.tick_params(axis="both", labelsize=12)
                for sp in ("top", "right"): ax_hd.spines[sp].set_visible(False)
                ax_hd.legend(frameon=False, loc="upper left", fontsize=11)
                fig_hd.tight_layout()
                out_hadisst_det = os.path.join(OUT_DIR, "amoc_hadisst_only_lpf10_detrended.png")
                fig_hd.savefig(out_hadisst_det, dpi=200, bbox_inches="tight")
                plt.close(fig_hd)
                print(f"[PLOT] Saved → {out_hadisst_det}")

            # ── Detrended vs standard comparison + CMIP6 (extra_fig1c style) ──
            if "HadISST" in multi_sst_results_detrended and "HadISST" in multi_sst_results:
                from scipy.signal import butter as butter_dc, sosfiltfilt as sosfiltfilt_dc
                CMIP_SPLIT_DIR_1c = "/home/am334/link_am334/ECCO_data/forced_amoc_26N_noLPF"
                CSV_HIGH_1c = os.path.join(CMIP_SPLIT_DIR_1c, "MULTIMODEL_AMOC26N_forced_mean_1850_2030_HIGH_ADRHD.csv")
                CSV_LOW_1c  = os.path.join(CMIP_SPLIT_DIR_1c, "MULTIMODEL_AMOC26N_forced_mean_1850_2030_LOW_ADRHD.csv")
                CSV_MID_1c  = os.path.join(CMIP_SPLIT_DIR_1c, "MULTIMODEL_AMOC26N_forced_mean_1850_2030.csv")
                BASE_T0_1c, BASE_T1_1c = "1900-01-01", "1940-12-31"

                def _load_mmm_csv(csv_path):
                    df = pd.read_csv(csv_path)
                    time_col = next((c for c in df.columns if c.lower() in ("time","date","datetime")), None)
                    if time_col is not None:
                        t = pd.to_datetime(df[time_col], errors="coerce").to_numpy()
                    else:
                        yr = next((c for c in df.columns if c.lower() in ("year","yr")), None)
                        if yr is not None:
                            t = pd.to_datetime(pd.to_numeric(df[yr], errors="coerce").astype("Int64").astype(str) + "-12-31", errors="coerce").to_numpy()
                        else:
                            t = pd.date_range("1850-12-31", periods=len(df), freq="Y").to_numpy()
                    amoc_c = [c for c in df.columns if ("amoc" in c.lower()) or ("26n" in c.lower())]
                    if amoc_c:
                        y = pd.to_numeric(df[amoc_c[0]], errors="coerce").to_numpy()
                    else:
                        num_c = [c for c in df.columns if c != time_col and pd.api.types.is_numeric_dtype(df[c])]
                        y = pd.to_numeric(df[num_c[0]] if num_c else df.iloc[:, 1], errors="coerce").to_numpy()
                    return t, np.asarray(y, dtype=float)

                t_high_1c, y_high_1c = _load_mmm_csv(CSV_HIGH_1c)
                t_low_1c,  y_low_1c  = _load_mmm_csv(CSV_LOW_1c)
                t_mid_1c,  y_mid_1c  = _load_mmm_csv(CSV_MID_1c)
                sos_annual = butter_dc(5, 1/10, btype='low', output='sos', fs=1)
                def _anom_lpf10_zp(t_vals, y_vals, t0, t1):
                    y_lp = sosfiltfilt_dc(sos_annual, y_vals)
                    t_vals = pd.to_datetime(t_vals).to_numpy()
                    mask = (t_vals >= np.datetime64(t0)) & (t_vals <= np.datetime64(t1))
                    base = float(np.nanmean(y_lp[mask])) if np.any(mask) else float(np.nanmean(y_lp))
                    return y_lp - base
                y_high_anom_1c = _anom_lpf10_zp(t_high_1c, y_high_1c, BASE_T0_1c, BASE_T1_1c)
                y_low_anom_1c  = _anom_lpf10_zp(t_low_1c,  y_low_1c,  BASE_T0_1c, BASE_T1_1c)
                y_mid_anom_1c  = _anom_lpf10_zp(t_mid_1c,  y_mid_1c,  BASE_T0_1c, BASE_T1_1c)

                res_std = multi_sst_results["HadISST"]
                res_det = multi_sst_results_detrended["HadISST"]
                t_vals = pd.to_datetime(res_std["times"]).to_numpy()
                m_base = (t_vals >= np.datetime64(BASE_T0_1c)) & (t_vals <= np.datetime64(BASE_T1_1c))
                base_std = float(np.nanmean(res_std["mu"][m_base])) if np.any(m_base) else float(np.nanmean(res_std["mu"]))
                base_det = float(np.nanmean(res_det["mu"][m_base])) if np.any(m_base) else float(np.nanmean(res_det["mu"]))
                anom_std = res_std["mu"] - base_std
                anom_det = res_det["mu"] - base_det
                sos_monthly = butter_dc(5, 1/120, btype='low', output='sos', fs=1)
                anom_std_filt = sosfiltfilt_dc(sos_monthly, anom_std)
                anom_det_filt = sosfiltfilt_dc(sos_monthly, anom_det)
                sigma_std = res_std["tot"] if USE_TOTAL_SIGMA else res_std["alea"]
                sigma_det = res_det["tot"] if USE_TOTAL_SIGMA else res_det["alea"]
                tmin, tmax = TOP_X_RANGE
                m_plot = (res_std["times"] >= tmin) & (res_std["times"] <= tmax)
                m_high_1c = (t_high_1c >= tmin) & (t_high_1c <= tmax)
                m_low_1c  = (t_low_1c  >= tmin) & (t_low_1c  <= tmax)
                m_mid_1c  = (t_mid_1c  >= tmin) & (t_mid_1c  <= tmax)

                COLOR_HIGH_1c = "#009E73"
                COLOR_MID_1c  = CMIP6_MMM_GREY
                COLOR_LOW_1c  = "#CC79A7"

                fig_dc, ax_dc = plt.subplots(figsize=(10.5, 4.6), dpi=300)
                ax_dc.fill_between(res_std["times"][m_plot],
                                    (anom_std_filt - 2*sigma_std)[m_plot], (anom_std_filt + 2*sigma_std)[m_plot],
                                    alpha=0.20, facecolor=RECON_BLUE, edgecolor="none")
                ax_dc.plot(res_std["times"][m_plot], anom_std_filt[m_plot], lw=2.2, color=RECON_BLUE, label="DL SST (standard)")
                ax_dc.fill_between(res_det["times"][m_plot],
                                    (anom_det_filt - 2*sigma_det)[m_plot], (anom_det_filt + 2*sigma_det)[m_plot],
                                    alpha=0.20, facecolor=ECCO_ORANGE, edgecolor="none")
                ax_dc.plot(res_det["times"][m_plot], anom_det_filt[m_plot], lw=2.2, color=ECCO_ORANGE, label="DL SST (detrended)")
                if np.any(m_high_1c):
                    ax_dc.plot(t_high_1c[m_high_1c], y_high_anom_1c[m_high_1c], lw=2.2, ls="-.", color=COLOR_HIGH_1c)
                if np.any(m_mid_1c):
                    ax_dc.plot(t_mid_1c[m_mid_1c], y_mid_anom_1c[m_mid_1c], lw=2.2, ls="-.", color=COLOR_MID_1c)
                if np.any(m_low_1c):
                    ax_dc.plot(t_low_1c[m_low_1c], y_low_anom_1c[m_low_1c], lw=2.2, ls="-.", color=COLOR_LOW_1c)
                ax_dc.set_xlim(*TOP_X_RANGE)
                ax_dc.set_ylim(-3, 4)
                ax_dc.set_ylabel("AMOC 26.5°N anomaly, Sv")
                box_axes(ax_dc, lw=1.0, color="black")
                from matplotlib.lines import Line2D
                handles_auto, labels_auto = ax_dc.get_legend_handles_labels()
                leg_left = ax_dc.legend(handles_auto, labels_auto, frameon=False, ncol=1, loc="upper left", handlelength=2.0)
                ax_dc.add_artist(leg_left)
                cmip6_high = Line2D([], [], color=COLOR_HIGH_1c, ls="-.", lw=2.2)
                cmip6_mid  = Line2D([], [], color=COLOR_MID_1c,  ls="-.", lw=2.2)
                cmip6_low  = Line2D([], [], color=COLOR_LOW_1c,  ls="-.", lw=2.2)
                leg_right = ax_dc.legend(
                    [cmip6_high, cmip6_mid, cmip6_low], ["high", "mid", "low"],
                    frameon=False, ncol=1, loc="upper right", handlelength=2.5, handletextpad=0.6, labelspacing=0.3,
                    title="CMIP6 aerosol forcing:", title_fontsize=None,
                )
                leg_right._legend_box.align = "right"
                fig_dc.tight_layout()
                out_det_vs_std = os.path.join(OUT_DIR, "amoc_hadisst_detrended_vs_standard_with_cmip6.png")
                fig_dc.savefig(out_det_vs_std, dpi=1000, bbox_inches="tight")
                plt.close(fig_dc)
                print(f"[PLOT] Saved → {out_det_vs_std}")

            for sst_name, res in multi_sst_results_detrended.items():
                nc_out = os.path.join(OUT_DIR, f"amoc_anom_from_{sst_name.replace(' ', '_').lower()}_lpf10_detrended.nc")
                ds_out = xr.Dataset({
                    "amoc_mean": (["time"], res["mu"]),
                    "amoc_anomaly": (["time"], res["mu_anom"]),
                    "amoc_alea_std": (["time"], res["alea"]),
                    "amoc_epi_std": (["time"], res["epi"]),
                    "amoc_tot_std": (["time"], res["tot"]),
                }, coords={"time": res["times"]})
                ds_out.attrs["source_sst"] = sst_name
                ds_out.attrs["lpf"] = "LPF10 (10-year = 120 months)"
                ds_out.attrs["model"] = "detrended"
                ds_out.attrs["reference_period"] = "1900-01-01 to 1930-12-31"
                ds_out.attrs["reference_mean_sv"] = res["ref_mean"]
                ds_out.to_netcdf(nc_out)
                print(f"[DATA] Saved → {nc_out}")

            print("\n[INFO] Computing trends for each SST product (detrended model)...")
            trend_rows_det = []
            for sst_name, res in multi_sst_results_detrended.items():
                mu = res["mu"]
                sigma = res["tot"] if USE_TOTAL_SIGMA else res["alea"]
                times = res["times"]

                for win_name, (t0, t1) in TREND_WINDOWS.items():
                    m_win = (times >= t0) & (times <= t1)
                    if m_win.sum() < 24:
                        continue

                    y_win = mu[m_win]
                    sig_win = sigma[m_win]
                    t_idx = np.arange(m_win.sum(), dtype=float)
                    factor = 12 * 100

                    for method in TREND_METHODS:
                        try:
                            slope, intercept, lo, hi, info, label, key = compute_trend_ci(
                                y_win, sig_win, t_idx=t_idx, method=method,
                                hint_block_len=120, hac_L_floor=120
                            )
                            trend_sv = slope * factor
                            lo_sv = lo * factor
                            hi_sv = hi * factor

                            trend_rows_det.append({
                                "SST_Product": sst_name,
                                "Window": win_name,
                                "Method": method.upper(),
                                "Trend_Sv_century": f"{trend_sv:.3f}",
                                "CI_95_lo": f"{lo_sv:.3f}",
                                "CI_95_hi": f"{hi_sv:.3f}",
                                "N_months": int(m_win.sum()),
                                "Model": "detrended",
                            })

                            if method == PRIMARY_METHOD:
                                print(f"    {sst_name} [{win_name}] (detrended): {trend_sv:.2f} [{lo_sv:.2f}, {hi_sv:.2f}] Sv/century (MBB)")
                        except Exception as e:
                            print(f"    {sst_name} [{win_name}] ({method}, detrended): failed - {e}")

            if trend_rows_det:
                csv_out_det = os.path.join(OUT_DIR, "multi_sst_trend_summary_detrended.csv")
                with open(csv_out_det, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=trend_rows_det[0].keys())
                    writer.writeheader()
                    writer.writerows(trend_rows_det)
                print(f"[CSV] Saved → {csv_out_det}")

    except Exception as e:
        import traceback
        print(f"[WARN] Multi-SST comparison (detrended model) failed: {e}")
        traceback.print_exc()
