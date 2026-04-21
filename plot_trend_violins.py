"""
Publication-quality violin plots of trend reconstruction errors
for multiple trend windows.

Reads trends.parquet (or .csv) from the artifacts directory and produces
a 2×2 figure with one panel per trend window. Each panel shows transparent
violins + jittered dots for three methods: SST index, DL SST, DL SSH.

Also provides ``plot_per_model_reconstructions(run_dir, tag="LPF120")``, which
writes ``per_model_reconstructions_LPF120.png`` (and .pdf) under ``run_dir``,
using ``series.parquet`` and ``metrics.parquet``. Example artifact layout:
``artifacts/2026-02-25/<save_name>/historical/``.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
from matplotlib.patches import Patch

# ── Config ────────────────────────────────────────────────────────────────────
DATE       = "2026-02-17"
SAVE_NAME  = "bnn_test_god_bless_it_work_all_vars_all_models_test_120_y"
SCENARIO   = "historical"
ARTIF_ROOT = Path("artifacts")

VAR_SST = "tos"
VAR_SSH = "zos_minus_basin_mean"

TREND_WINDOWS = [
    (1855, 2014),
    (1900, 1985),
    (1993, 2014),
    (1900, 2000),
]

METHODS = ["SST index", "DL SST", "DL SSH"]
COLORS  = ["#7F7F7F", "#0072B2", "#D55E00"]

# SST index → AMOC (same as plot_composite_figure / inference)
SST_CSV = Path("/home/am334/link_am334/data/sst_amoc_fingerprint_spg_minus_global_DJFMAM.csv")
RECON_A_SV = 0.2648
RECON_B_SV_PER_C = 1.6057
SST_BASELINE_YEARS = 100


def _model_for_sst_csv(model_series: str) -> str:
    s = str(model_series)
    if s.startswith("test_model(") and s.endswith(")"):
        return s[len("test_model(") : -1]
    return s


def _sst_to_amoc(years: np.ndarray, sst_vals: np.ndarray):
    yrs = np.asarray(years, int)
    vals = np.asarray(sst_vals, float)
    order = np.argsort(yrs)
    yrs, vals = yrs[order], vals[order]
    n0 = min(SST_BASELINE_YEARS, len(vals))
    baseline = np.nanmean(vals[:n0]) if n0 else np.nan
    sst_prime = vals - baseline
    return yrs, RECON_A_SV + RECON_B_SV_PER_C * sst_prime


def _get_sst_index(
    sst_df: pd.DataFrame,
    model: str,
    member: str,
    lpf_months: int,
    scenario: str,
) -> tuple[np.ndarray, np.ndarray]:
    """SST fingerprint AMOC reconstruction; aligns with composite_figure."""
    sub = sst_df[(sst_df["model"] == model) & (sst_df["scenario"] == scenario)]
    if sub.empty:
        return np.array([]), np.array([])
    if "member" in sub.columns and member is not None:
        sub_m = sub[sub["member"] == member]
        if not sub_m.empty:
            sub = sub_m
        elif "SPG_minus_GLOBAL" in sub.columns:
            sub = sub.groupby("time", as_index=False)["SPG_minus_GLOBAL"].mean()
    sub = sub.sort_values("time")
    yrs = sub["time"].values.astype(int)
    vals = sub["SPG_minus_GLOBAL"].values.astype(float)
    k = max(1, int(round(lpf_months / 12))) if lpf_months > 1 else 1
    if k > 1:
        vals = (
            pd.Series(vals)
            .rolling(k, center=True, min_periods=max(1, k // 2))
            .mean()
            .to_numpy()
        )
    return _sst_to_amoc(yrs, vals)


def _lpf_months_from_manifest(run_dir: Path) -> int:
    man = run_dir / "manifest.json"
    if not man.exists():
        return 120
    try:
        d = json.loads(man.read_text())
        s = str(d.get("lpf", "LPF120"))
        dig = "".join(c for c in s if c.isdigit())
        return int(dig) if dig else 120
    except Exception:
        return 120


def _sst_index_abs_err_per_model(
    sst_df: pd.DataFrame,
    model_label: str,
    ms_rows: pd.DataFrame,
    lpf_months: int,
    scenario: str,
) -> pd.Series:
    """|SST-index AMOC − truth| averaged over members; index = year."""
    model_sst = _model_for_sst_csv(model_label)
    per_mem: list[pd.Series] = []
    for member in ms_rows["member"].unique():
        mm = ms_rows[ms_rows["member"] == member]
        tgt_by_year = mm.groupby("year")["tgt"].mean()
        sst_yrs, sst_amoc = _get_sst_index(
            sst_df, model_sst, str(member), lpf_months, scenario
        )
        if sst_yrs.size == 0:
            continue
        pred = pd.Series(sst_amoc, index=sst_yrs.astype(int))
        common = pred.index.intersection(tgt_by_year.index)
        if len(common) == 0:
            continue
        abs_err = (pred.loc[common] - tgt_by_year.loc[common]).abs()
        per_mem.append(abs_err)
    if not per_mem:
        return pd.Series(dtype=float)
    return pd.concat(per_mem, axis=1).mean(axis=1)

# ── Style (publication clean) ─────────────────────────────────────────────────
mpl.rcParams.update({
    "figure.dpi":       140,
    "savefig.dpi":      600,
    "font.size":        10,
    "axes.labelsize":   10,
    "axes.titlesize":   11,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "axes.linewidth":   0.8,
    "legend.frameon":   False,
    "lines.linewidth":  1.2,
})


def load_trends(run_dir: Path) -> pd.DataFrame:
    parq = run_dir / "trends.parquet"
    csv  = run_dir / "trends.csv"
    if parq.exists():
        return pd.read_parquet(parq)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"trends.(parquet|csv) not found in {run_dir}")


def _clean(series: pd.Series) -> np.ndarray:
    return series.replace([np.inf, -np.inf], np.nan).dropna().to_numpy()


def plot_trend_violins(run_dir: Path, tr: pd.DataFrame) -> None:
    for col in ("var", "member", "trend_sim", "trend_nn",
                "trend_sst", "trend_start", "trend_end"):
        if col not in tr.columns:
            raise ValueError(f"Missing column '{col}' in trends file.")

    nrows, ncols = 2, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(9, 7))
    axes_flat = axes.flatten()
    rng = np.random.default_rng(42)
    labels_violin = ["SST index", "DL SST", "DL SSH"]

    violin_width = 0.55

    for idx, (tw_start, tw_end) in enumerate(TREND_WINDOWS):
        ax = axes_flat[idx]

        sub = tr[
            (tr["trend_start"] == tw_start) & (tr["trend_end"] == tw_end)
        ].copy()
        sub = sub[np.isfinite(sub["trend_sim"])].copy()
        sub["err_sst"] = sub["trend_sst"] - sub["trend_sim"]
        sub["err_nn"]  = sub["trend_nn"]  - sub["trend_sim"]

        e_sst_all = _clean(sub["err_sst"])
        e_nn_sst  = _clean(sub.loc[sub["var"] == VAR_SST, "err_nn"])
        e_nn_ssh  = _clean(sub.loc[sub["var"] == VAR_SSH, "err_nn"])

        data_violin = [e_sst_all, e_nn_sst, e_nn_ssh]

        if not any(len(d) > 0 for d in data_violin):
            ax.set_title(f"{tw_start}\u2013{tw_end}\n(no data)")
            continue

        safe = [d if len(d) > 0 else np.array([0.0]) for d in data_violin]
        parts = ax.violinplot(
            safe, showmeans=False, showmedians=False,
            showextrema=False, widths=violin_width,
        )
        for i, body in enumerate(parts["bodies"]):
            body.set_facecolor(COLORS[i])
            body.set_edgecolor("black")
            body.set_alpha(0.35)
            body.set_linewidth(0.7)

        for i, arr in enumerate(data_violin, start=1):
            arr = np.asarray(arr)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                continue
            jitter_x = i + 0.12 * (rng.random(arr.size) - 0.5)
            ax.scatter(
                jitter_x, arr, s=10, color=COLORS[i - 1],
                edgecolor="black", linewidth=0.3, alpha=0.7, zorder=3,
            )
            ax.scatter(i, arr.mean(), s=18, color="black", zorder=4)
            med = np.median(arr)
            hw = violin_width * 0.4
            ax.plot(
                [i - hw, i + hw], [med, med],
                color="black", lw=1.3, zorder=4,
            )

        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(labels_violin, rotation=0)
        ax.set_xlim(0.35, 3.65)
        ax.axhline(0, color="0.2", linestyle="--", lw=0.8, zorder=1)
        ax.set_ylabel(f"{tw_start}\u2013{tw_end}\nTrend error (Sv century\u207b\u00b9)")

        panel_label = chr(ord("a") + idx)
        ax.text(
            -0.02, 1.02, panel_label, transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="bottom", ha="right",
        )

    legend_elements = [
        Patch(facecolor=c, edgecolor="black", alpha=0.6, label=m)
        for c, m in zip(COLORS, METHODS)
    ]
    fig.legend(
        handles=legend_elements, loc="lower center",
        ncol=3, bbox_to_anchor=(0.5, -0.01), frameon=False,
    )
    fig.tight_layout(rect=[0.0, 0.04, 1, 1.0])

    out_png = run_dir / "violin_trend_errors_multi_window.png"
    out_pdf = run_dir / "violin_trend_errors_multi_window.pdf"
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.show()
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


def plot_abs_error_by_model(run_dir: Path, tr: pd.DataFrame) -> None:
    """Mean |trend error| per ESM (+ multi-model mean), one panel per window."""
    for col in ("model", "var", "trend_sim", "trend_nn",
                "trend_sst", "trend_start", "trend_end"):
        if col not in tr.columns:
            raise ValueError(f"Missing column '{col}' in trends file.")

    tr = tr[np.isfinite(tr["trend_sim"])].copy()
    tr["abs_err_sst"] = (tr["trend_sst"] - tr["trend_sim"]).abs()
    tr["abs_err_nn"]  = (tr["trend_nn"]  - tr["trend_sim"]).abs()

    nrows, ncols = 2, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 8))
    axes_flat = axes.flatten()

    bar_width = 0.25

    for idx, (tw_start, tw_end) in enumerate(TREND_WINDOWS):
        ax = axes_flat[idx]
        sub = tr[
            (tr["trend_start"] == tw_start) & (tr["trend_end"] == tw_end)
        ].copy()

        models = sorted(sub["model"].unique())

        mean_sst, mean_nn_sst, mean_nn_ssh = [], [], []
        for m in models:
            ms = sub[sub["model"] == m]
            mean_sst.append(ms["abs_err_sst"].mean())
            mean_nn_sst.append(
                ms.loc[ms["var"] == VAR_SST, "abs_err_nn"].mean()
            )
            mean_nn_ssh.append(
                ms.loc[ms["var"] == VAR_SSH, "abs_err_nn"].mean()
            )

        labels = [m.replace("-", "\n", 1) for m in models] + ["Mean"]
        mean_sst.append(np.nanmean(mean_sst))
        mean_nn_sst.append(np.nanmean(mean_nn_sst))
        mean_nn_ssh.append(np.nanmean(mean_nn_ssh))

        n = len(labels)
        x = np.arange(n)

        ax.bar(x - bar_width, mean_sst,     bar_width, color=COLORS[0],
               edgecolor="black", linewidth=0.4, alpha=0.85, label=METHODS[0])
        ax.bar(x,             mean_nn_sst,   bar_width, color=COLORS[1],
               edgecolor="black", linewidth=0.4, alpha=0.85, label=METHODS[1])
        ax.bar(x + bar_width, mean_nn_ssh,   bar_width, color=COLORS[2],
               edgecolor="black", linewidth=0.4, alpha=0.85, label=METHODS[2])

        ax.axvline(n - 1.5, color="0.4", ls="--", lw=0.7)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
        ax.set_ylabel(
            f"{tw_start}\u2013{tw_end}\n"
            "Mean |trend error| (Sv century\u207b\u00b9)"
        )

        panel_label = chr(ord("a") + idx)
        ax.text(
            -0.02, 1.02, panel_label, transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="bottom", ha="right",
        )

    legend_elements = [
        Patch(facecolor=c, edgecolor="black", alpha=0.6, label=m)
        for c, m in zip(COLORS, METHODS)
    ]
    fig.legend(
        handles=legend_elements, loc="lower center",
        ncol=3, bbox_to_anchor=(0.5, -0.01), frameon=False,
    )
    fig.tight_layout(rect=[0.0, 0.04, 1, 1.0])

    out_png = run_dir / "abs_trend_error_by_model.png"
    out_pdf = run_dir / "abs_trend_error_by_model.pdf"
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.show()
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


def load_series(run_dir: Path) -> pd.DataFrame:
    parq = run_dir / "series.parquet"
    csv  = run_dir / "series.csv"
    if parq.exists():
        return pd.read_parquet(parq)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"series.(parquet|csv) not found in {run_dir}")


EXCLUDE_MODELS = {"CAS-ESM2-0"}
RUNNING_MEAN_WINDOW = 11
# abs_error_timeseries: inter-model band transparency
ABS_ERR_FILL_ALPHA = 0.12
# abs_error_timeseries: font sizes (bigger than default mpl in this script)
ABS_ERR_AXIS_FS = 13
ABS_ERR_TICK_FS = 12
ABS_ERR_LEGEND_FS = 11


def plot_abs_error_timeseries(run_dir: Path, series: pd.DataFrame) -> None:
    """
    Single-panel |error| with inter-model 25–75 % spread for
    DL SST, DL SSH, and SST index (running-mean curves).
    """
    ex = ~series["model"].isin(EXCLUDE_MODELS)
    sub_sst = series[(series["var"] == VAR_SST) & ex].copy()
    sub_ssh = series[(series["var"] == VAR_SSH) & ex].copy()
    sub_sst["abs_err"] = (sub_sst["mu"] - sub_sst["tgt"]).abs()
    sub_ssh["abs_err"] = (sub_ssh["mu"] - sub_ssh["tgt"]).abs()

    models_sst = set(sub_sst["model"].unique())
    models_ssh = set(sub_ssh["model"].unique())
    models = sorted(models_sst & models_ssh)

    color_dl_sst = COLORS[1]
    color_dl_ssh = COLORS[2]
    color_sst_ix = COLORS[0]

    lpf_months = _lpf_months_from_manifest(run_dir)
    sst_df = pd.read_csv(SST_CSV) if SST_CSV.exists() else None
    all_means_sst: list[pd.Series] = []
    all_means_ssh: list[pd.Series] = []
    all_sst_idx: list[pd.Series] = []

    for m in models:
        ms_s = sub_sst[sub_sst["model"] == m]
        ms_h = sub_ssh[sub_ssh["model"] == m]
        all_means_sst.append(ms_s.groupby("year")["abs_err"].mean())
        all_means_ssh.append(ms_h.groupby("year")["abs_err"].mean())
        if sst_df is not None:
            sst_m = _sst_index_abs_err_per_model(sst_df, m, ms_s, lpf_months, SCENARIO)
            if not sst_m.empty:
                all_sst_idx.append(sst_m)

    comb_sst = pd.concat(all_means_sst, axis=1)
    comb_ssh = pd.concat(all_means_ssh, axis=1)
    mm_sst = comb_sst.mean(axis=1)
    mm_ssh = comb_ssh.mean(axis=1)

    w = RUNNING_MEAN_WINDOW
    smooth_sst = mm_sst.rolling(w, center=True, min_periods=1).mean()
    smooth_ssh = mm_ssh.rolling(w, center=True, min_periods=1).mean()

    def _roll_q25_q75(df: pd.DataFrame):
        lo = df.quantile(0.25, axis=1).rolling(w, center=True, min_periods=1).mean()
        hi = df.quantile(0.75, axis=1).rolling(w, center=True, min_periods=1).mean()
        return lo, hi

    p25_sst, p75_sst = _roll_q25_q75(comb_sst)
    p25_ssh, p75_ssh = _roll_q25_q75(comb_ssh)

    fig, ax = plt.subplots(1, 1, figsize=(10, 4.8))

    fa = ABS_ERR_FILL_ALPHA
    zf, zl = 1, 5
    ax.fill_between(
        p25_sst.index, p25_sst, p75_sst, color=color_dl_sst, alpha=fa, zorder=zf,
    )
    ax.fill_between(
        p25_ssh.index, p25_ssh, p75_ssh, color=color_dl_ssh, alpha=fa, zorder=zf + 1,
    )

    has_sst_index = bool(sst_df is not None and all_sst_idx)
    if has_sst_index:
        comb_ix = pd.concat(all_sst_idx, axis=1)
        mm_ix = comb_ix.mean(axis=1)
        smooth_ix = mm_ix.rolling(w, center=True, min_periods=1).mean()
        p25_ix, p75_ix = _roll_q25_q75(comb_ix)
        ax.fill_between(
            p25_ix.index, p25_ix, p75_ix, color=color_sst_ix, alpha=fa, zorder=zf + 2,
        )
        ax.plot(
            smooth_ix.index, smooth_ix.values,
            color=color_sst_ix, lw=2.4, ls="--", zorder=zl,
        )
    elif sst_df is None:
        print(f"[warn] SST CSV not found at {SST_CSV}; skip SST index curves.")

    ax.plot(smooth_sst.index, smooth_sst.values, color=color_dl_sst, lw=2.4, zorder=zl)
    ax.plot(smooth_ssh.index, smooth_ssh.values, color=color_dl_ssh, lw=2.4, zorder=zl)

    # One legend row: band + mean combined per method (3 items)
    leg_handles = [
        (
            Patch(facecolor=color_dl_sst, edgecolor="none", alpha=fa),
            Line2D([0], [0], color=color_dl_sst, lw=2.4, solid_capstyle="round"),
        ),
        (
            Patch(facecolor=color_dl_ssh, edgecolor="none", alpha=fa),
            Line2D([0], [0], color=color_dl_ssh, lw=2.4, solid_capstyle="round"),
        ),
    ]
    leg_labels = ["DL SST", "DL SSH"]
    if has_sst_index:
        leg_handles.append(
            (
                Patch(facecolor=color_sst_ix, edgecolor="none", alpha=fa),
                Line2D([0], [0], color=color_sst_ix, lw=2.4, ls="--", solid_capstyle="round"),
            )
        )
        leg_labels.append("SST index")

    ax.set_ylabel("|error| (Sv)", fontsize=ABS_ERR_AXIS_FS)
    ax.set_xlabel("Year", fontsize=ABS_ERR_AXIS_FS)
    ax.tick_params(axis="both", labelsize=ABS_ERR_TICK_FS)
    leg = ax.legend(
        leg_handles,
        leg_labels,
        handler_map={tuple: HandlerTuple(ndivide=2)},
        loc="upper left",
        ncol=len(leg_labels),
        fontsize=ABS_ERR_LEGEND_FS,
        handlelength=1.5,
        handletextpad=0.5,
        columnspacing=1.0,
        labelspacing=0.28,
        borderpad=0.4,
        framealpha=0.92,
    )
    leg.get_frame().set_linewidth(0.5)

    fig.tight_layout()

    out_png = run_dir / "abs_error_timeseries.png"
    out_pdf = run_dir / "abs_error_timeseries.pdf"
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.show()
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


DATE_LPF24  = "2025-11-14"
NAME_LPF24  = "lpf2_years"

DATE_LPF120 = "2026-02-25"
NAME_LPF120 = "lpf10_years"


def plot_per_model_reconstructions(run_dir: Path, tag: str = "") -> None:
    """Per-model first-member reconstruction: Truth vs DL μ SST & SSH."""
    from matplotlib.lines import Line2D

    series = load_series(run_dir)
    metrics = pd.read_parquet(run_dir / "metrics.parquet")

    models = sorted(series["model"].unique())
    n = len(models)
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 5, nrows * 2.5),
                             sharex=True)
    axes_flat = axes.flatten()

    for i, model in enumerate(models):
        ax = axes_flat[i]
        mseries = series[series["model"] == model]
        first_member = sorted(mseries["member"].unique())[0]

        for var, color, label in [
            (VAR_SST, COLORS[1], "DL μ SST"),
            (VAR_SSH, COLORS[2], "DL μ SSH"),
        ]:
            sel = mseries[(mseries["member"] == first_member)
                          & (mseries["var"] == var)]
            grouped = sel.groupby("year").mean(numeric_only=True).sort_index()
            years = grouped.index.values
            mu  = grouped["mu"].values
            sig = grouped["sig"].values

            ax.fill_between(years, mu - 2 * sig, mu + 2 * sig,
                            color=color, alpha=0.20, lw=0, edgecolor="none")
            ax.plot(years, mu, color=color, lw=1.0, label=label)

        tgt_s = (mseries[(mseries["member"] == first_member)
                         & (mseries["var"] == VAR_SST)]
                 .groupby("year")["tgt"].mean().sort_index())
        ax.plot(tgt_s.index, tgt_s.values, color="black", lw=1.4, label="Truth")

        r2_sst = metrics.loc[(metrics["model"] == model)
                             & (metrics["var"] == VAR_SST), "r2"]
        r2_ssh = metrics.loc[(metrics["model"] == model)
                             & (metrics["var"] == VAR_SSH), "r2"]
        r2_txt = ""
        if not r2_sst.empty:
            r2_txt += f"SST: R²={r2_sst.values[0]:.3f}\n"
        if not r2_ssh.empty:
            r2_txt += f"SSH: R²={r2_ssh.values[0]:.3f}"
        ax.text(0.02, 0.97, r2_txt.strip(), transform=ax.transAxes,
                va="top", ha="left", fontsize=7,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

        ax.set_title(model, fontsize=9)
        ax.set_ylabel("Sv")

        panel_label = chr(ord("a") + i)
        ax.text(-0.02, 1.02, panel_label, transform=ax.transAxes,
                fontsize=10, fontweight="bold", va="bottom", ha="right")

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    legend_handles = [
        Line2D([0], [0], color="black",   lw=1.4, label="Truth"),
        Line2D([0], [0], color=COLORS[1], lw=1.0, label="DL μ SST"),
        Line2D([0], [0], color=COLORS[2], lw=1.0, label="DL μ SSH"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=3,
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, 1.01))

    fig.tight_layout(rect=[0, 0, 1, 0.98])

    suffix = f"_{tag}" if tag else ""
    out_png = run_dir / f"per_model_reconstructions{suffix}.png"
    out_pdf = run_dir / f"per_model_reconstructions{suffix}.pdf"
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.show()
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    run_dir = ARTIF_ROOT / DATE / SAVE_NAME / SCENARIO
    run_dir.mkdir(parents=True, exist_ok=True)
    tr = load_trends(run_dir)
    series = load_series(run_dir)
    plot_trend_violins(run_dir, tr)
    plot_abs_error_by_model(run_dir, tr)
    plot_abs_error_timeseries(run_dir, series)

    run_dir_24 = ARTIF_ROOT / DATE_LPF24 / NAME_LPF24 / SCENARIO
    plot_per_model_reconstructions(run_dir_24, tag="LPF24")

    run_dir_120 = ARTIF_ROOT / DATE_LPF120 / NAME_LPF120 / SCENARIO
    plot_per_model_reconstructions(run_dir_120, tag="LPF120")
