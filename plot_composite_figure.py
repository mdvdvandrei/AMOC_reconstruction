"""
Composite 4-panel publication figure (Nature/Science style).

  a) Single-member AMOC reconstruction example — MPI-ESM1-2-HR (LPF 2 yr)
  b) Single-member AMOC reconstruction example — MRI-ESM2-0   (LPF 10 yr)
  c) R² skill grouped by low-pass filter
  d) Trend reconstruction accuracy (violin, 1855–2014)

All data are read from artifact parquet files and the SST CSV.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ── Paths & config ───────────────────────────────────────────────────────────
ARTIF_ROOT = Path("artifacts")
SST_CSV    = Path("/home/am334/link_am334/data/"
                  "sst_amoc_fingerprint_spg_minus_global_DJFMAM.csv")
SCENARIO   = "historical"

RUNS = {
    "LPF24": {
        "dir": (ARTIF_ROOT / "2025-11-14"
                / "bnn_test_god_bless_it_work_all_vars_all_models_test_24"
                / SCENARIO),
        "months": 24,
    },
    "LPF120": {
        "dir": (ARTIF_ROOT / "2026-02-17"
                / "bnn_test_god_bless_it_work_all_vars_all_models_test_120_y"
                / SCENARIO),
        "months": 120,
    },
}

OUT_DIR = RUNS["LPF120"]["dir"]

EXAMPLES = {
    "a": {"run": "LPF120", "model": "MPI-ESM1-2-HR", "member": "r4i1p1f1"},
    "b": {"run": "LPF120", "model": "MRI-ESM2-0",    "member": "r7i1p1f1"},
}

VAR_SST = "tos"
VAR_SSH = "zos_minus_basin_mean"

RECON_A_SV         = 0.2648
RECON_B_SV_PER_C   = 1.6057
SST_BASELINE_YEARS = 100

TREND_WINDOW = (1900, 2000)
EXCLUDE_MODELS = {"CAS-ESM2-0"}

C_TRUTH     = "black"
C_SST_INDEX = "#7F7F7F"
C_NN_SST    = "#0072B2"
C_NN_SSH    = "#D55E00"

METHOD_NAMES  = ["SST index", "DL (SST)", "DL (SSH)"]
METHOD_COLORS = [C_SST_INDEX, C_NN_SST, C_NN_SSH]

# ── Hardcoded R² values for panel c ──────────────────────────────────────────
R2_MEAN = {
    "2 yrs":  {"SST index": 0.018,  "DL (SST)": 0.3388, "DL (SSH)": 0.5761},
    "10 yrs": {"SST index": -0.031, "DL (SST)": 0.48,   "DL (SSH)": 0.72},
}
R2_STD = {
    "2 yrs":  {"SST index": 0.169, "DL (SST)": 0.20, "DL (SSH)": 0.25},
    "10 yrs": {"SST index": 0.351, "DL (SST)": 0.23, "DL (SSH)": 0.19},
}

# ── Matplotlib style (larger fonts for slides / readability) ────────────────
_FS = 16  # base font size
mpl.rcParams.update({
    "figure.dpi":       140,
    "savefig.dpi":      600,
    "font.size":        _FS,
    "axes.labelsize":   _FS,
    "axes.titlesize":   _FS + 1,
    "xtick.labelsize":  _FS - 1,
    "ytick.labelsize":  _FS - 1,
    "legend.fontsize":  _FS - 2,
    "axes.linewidth":   0.8,
    "legend.frameon":   False,
    "lines.linewidth":  1.2,
    "font.family":      "sans-serif",
})
_PANEL_LABEL_FS = 20  # a, b, c, d
_LEGEND_FS = _FS - 1


# ── Helper I/O ───────────────────────────────────────────────────────────────
def _load(directory: Path, name: str) -> pd.DataFrame:
    for ext in (".parquet", ".csv"):
        p = directory / (name + ext)
        if p.exists():
            return pd.read_parquet(p) if ext == ".parquet" else pd.read_csv(p)
    raise FileNotFoundError(f"{name}.(parquet|csv) not in {directory}")


# ── SST → AMOC reconstruction ───────────────────────────────────────────────
def _sst_to_amoc(years: np.ndarray, sst_vals: np.ndarray):
    yrs = np.asarray(years, int)
    vals = np.asarray(sst_vals, float)
    order = np.argsort(yrs)
    yrs, vals = yrs[order], vals[order]
    baseline = np.nanmean(vals[:SST_BASELINE_YEARS])
    sst_prime = vals - baseline
    return yrs, RECON_A_SV + RECON_B_SV_PER_C * sst_prime


def _get_sst_index(sst_df: pd.DataFrame, model: str, member: str,
                   lpf_months: int) -> tuple[np.ndarray, np.ndarray]:
    """SST index AMOC reconstruction for one model/member."""
    sub = sst_df[(sst_df["model"] == model)
                 & (sst_df["scenario"] == SCENARIO)
                 & (sst_df["member"] == member)].sort_values("time")
    if sub.empty:
        return np.array([]), np.array([])
    yrs = sub["time"].values.astype(int)
    vals = sub["SPG_minus_GLOBAL"].values.astype(float)
    k = max(1, int(round(lpf_months / 12))) if lpf_months > 1 else 1
    if k > 1:
        vals = (pd.Series(vals)
                .rolling(k, center=True, min_periods=max(1, k // 2))
                .mean().to_numpy())
    return _sst_to_amoc(yrs, vals)


def _clean(s: pd.Series) -> np.ndarray:
    return s.replace([np.inf, -np.inf], np.nan).dropna().to_numpy()


# ── Panel a / b: single-member example ───────────────────────────────────────
def _plot_example(ax, series_df: pd.DataFrame, sst_df: pd.DataFrame,
                  model: str, member: str, lpf_months: int,
                  panel_label: str):
    def _year_mean(df, col):
        return df.groupby("year")[col].mean().sort_index()

    sel_sst = series_df[(series_df["model"] == model)
                        & (series_df["member"] == member)
                        & (series_df["var"] == VAR_SST)]
    sel_ssh = series_df[(series_df["model"] == model)
                        & (series_df["member"] == member)
                        & (series_df["var"] == VAR_SSH)]

    tgt     = _year_mean(sel_sst, "tgt")
    mu_sst  = _year_mean(sel_sst, "mu")
    sig_sst = _year_mean(sel_sst, "sig")
    mu_ssh  = _year_mean(sel_ssh, "mu")
    sig_ssh = _year_mean(sel_ssh, "sig")
    years   = tgt.index.values

    # ±2σ bands
    ax.fill_between(years,
                    mu_sst.values - 2 * sig_sst.values,
                    mu_sst.values + 2 * sig_sst.values,
                    color=C_NN_SST, alpha=0.20, lw=0, edgecolor="none")
    ax.fill_between(years,
                    mu_ssh.values - 2 * sig_ssh.values,
                    mu_ssh.values + 2 * sig_ssh.values,
                    color=C_NN_SSH, alpha=0.20, lw=0, edgecolor="none")

    # center lines
    ax.plot(years, tgt.values,    color=C_TRUTH, lw=1.8, label="Truth")
    ax.plot(years, mu_sst.values, color=C_NN_SST, lw=1.3, label="DL SST")
    ax.plot(years, mu_ssh.values, color=C_NN_SSH, lw=1.3, label="DL SSH")

    # SST index
    sst_yrs, sst_amoc = _get_sst_index(sst_df, model, member, lpf_months)
    if sst_yrs.size:
        mask = (sst_yrs >= years.min()) & (sst_yrs <= years.max())
        ax.plot(sst_yrs[mask], sst_amoc[mask], ls="--", color=C_SST_INDEX,
                lw=1.3, label="SST index")

    ax.set_ylabel("AMOC anomaly at 26.5°N (Sv)")
    ax.text(-0.02, 1.02, panel_label, transform=ax.transAxes,
            fontsize=_PANEL_LABEL_FS, fontweight="bold", va="bottom", ha="right")


# ── Panel c: R² bars (hardcoded) ────────────────────────────────────────────
def _plot_r2_bars(ax):
    lpf_labels = ["2 yrs", "10 yrs"]
    bar_width = 0.22
    x = np.arange(len(lpf_labels))
    offsets = [-bar_width, 0, bar_width]

    for i, method in enumerate(METHOD_NAMES):
        means = [R2_MEAN[lbl][method] for lbl in lpf_labels]
        errs  = [R2_STD[lbl][method]  for lbl in lpf_labels]
        ax.bar(x + offsets[i], means, bar_width,
               yerr=errs, color=METHOD_COLORS[i],
               edgecolor="black", linewidth=0.4, alpha=0.85,
               capsize=4, error_kw=dict(lw=0.9),
               label=["SST index", "DL SST", "DL SSH"][i])

    ax.set_xticks(x)
    ax.set_xticklabels(lpf_labels)
    ax.set_ylabel("R²")
    ax.set_ylim(-0.4, 1.0)
    ax.axhline(0, color="0.4", ls="-", lw=0.5)
    ax.legend(loc="upper left", fontsize=_LEGEND_FS)
    ax.text(-0.02, 1.02, "c", transform=ax.transAxes,
            fontsize=_PANEL_LABEL_FS, fontweight="bold", va="bottom", ha="right")


# ── Panel d: violin ─────────────────────────────────────────────────────────
def _plot_violin(ax, trends_df: pd.DataFrame):
    tw_start, tw_end = TREND_WINDOW
    sub = trends_df[(trends_df["trend_start"] == tw_start)
                    & (trends_df["trend_end"] == tw_end)
                    & (~trends_df["model"].isin(EXCLUDE_MODELS))].copy()
    sub = sub[np.isfinite(sub["trend_sim"])].copy()
    sub["err_sst"] = sub["trend_sst"] - sub["trend_sim"]
    sub["err_nn"]  = sub["trend_nn"]  - sub["trend_sim"]

    e_sst    = _clean(sub["err_sst"])
    e_nn_sst = _clean(sub.loc[sub["var"] == VAR_SST, "err_nn"])
    e_nn_ssh = _clean(sub.loc[sub["var"] == VAR_SSH, "err_nn"])
    data = [e_sst, e_nn_sst, e_nn_ssh]
    safe = [d if d.size > 0 else np.array([0.0]) for d in data]

    rng = np.random.default_rng(42)
    vw = 0.75

    parts = ax.violinplot(safe, showmeans=False, showmedians=False,
                          showextrema=False, widths=vw)
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(METHOD_COLORS[i])
        body.set_edgecolor("black")
        body.set_alpha(0.35)
        body.set_linewidth(0.7)

    for i, arr in enumerate(data, start=1):
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue
        jx = i + 0.15 * (rng.random(arr.size) - 0.5)
        ax.scatter(jx, arr, s=16, color=METHOD_COLORS[i - 1],
                   edgecolor="black", linewidth=0.35, alpha=0.7, zorder=3)
        ax.scatter(i, arr.mean(), s=32, color="black", zorder=4)
        hw = vw * 0.4
        med = np.median(arr)
        ax.plot([i - hw, i + hw], [med, med],
                color="black", lw=1.3, zorder=4)

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["SST index", "DL SST", "DL SSH"])
    ax.set_xlim(0.2, 3.8)
    ax.axhline(0, color="0.2", ls="--", lw=0.8, zorder=1)
    ax.set_ylabel("Trend error (Sv century\u207b\u00b9)")
    ax.text(-0.02, 1.02, "d", transform=ax.transAxes,
            fontsize=_PANEL_LABEL_FS, fontweight="bold", va="bottom", ha="right")


# ── Compose figure ───────────────────────────────────────────────────────────
def main():
    sst_df = pd.read_csv(SST_CSV)

    series_cache = {"LPF120": _load(RUNS["LPF120"]["dir"], "series")}
    trends_120 = _load(RUNS["LPF120"]["dir"], "trends")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9),
                             gridspec_kw=dict(height_ratios=[0.75, 1]))
    ax_a, ax_b, ax_c, ax_d = axes.flat

    # a – MPI-ESM1-2-HR example (LPF120)
    ex = EXAMPLES["a"]
    _plot_example(ax_a, series_cache[ex["run"]], sst_df,
                  ex["model"], ex["member"],
                  RUNS[ex["run"]]["months"], "a")

    # b – MRI-ESM2-0 example (LPF120)
    ex = EXAMPLES["b"]
    _plot_example(ax_b, series_cache[ex["run"]], sst_df,
                  ex["model"], ex["member"],
                  RUNS[ex["run"]]["months"], "b")

    # shared legend for a & b — centered above top row (between panels a and b)
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color=C_TRUTH,     lw=1.8, label="Truth"),
        Line2D([0], [0], color=C_NN_SST,    lw=1.3, label="DL SST"),
        Line2D([0], [0], color=C_NN_SSH,    lw=1.3, label="DL SSH"),
        Line2D([0], [0], color=C_SST_INDEX, lw=1.3, ls="--", label="SST index"),
    ]

    # c – R² bars
    _plot_r2_bars(ax_c)

    # d – Violin
    _plot_violin(ax_d, trends_120)

    # Leave headroom for horizontal legend centered over (a | b)
    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.06, top=0.90,
                        hspace=0.35, wspace=0.32)
    # x=0.5 spans a & b; y just above row-1 axes (top=0.90) so legend sits in upper margin
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.93),
        bbox_transform=fig.transFigure,
        ncol=4,
        fontsize=_LEGEND_FS,
        handlelength=1.8,
        columnspacing=1.4,
        frameon=False,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out = OUT_DIR / f"composite_figure.{ext}"
        fig.savefig(out, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.show()


if __name__ == "__main__":
    main()
