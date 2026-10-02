#!/usr/bin/env python3
"""Fig. 4, both panels, with the reported GLS trend intervals.

Panel (a) is the twentieth-century case and panel (b) the satellite era, each
showing the spread of CMIP6 member trends against the reconstruction. The
observational estimate in both is the fit the paper defends and whose coverage
was measured on the CMIP6 reconstructions: generalised least squares on annual
means, weighted by the pooled within-fold sigma of the network and correlated as
the held-out reconstruction errors actually are, with the across-fold spread of
the trend added in quadrature at the trend level.

Two changes were forced by the data rather than chosen. The top panel is
1900-2000 rather than 1900-1985, because that is the window the SST case is
validated on and the one whose CMIP6 truth is available in the same artifact as
the validation; the member trends are the true AMOC of each member under the same
10-year filter. And in the bottom panel the observational intervals are also
drawn as columns on the right, because the GLS interval of the SST product over
1993-2023 is four times wider than the SSH one and would hide the violins if it
spanned the panel.

The bands and the lines are read from R_tables_unified.csv, so the figure cannot
drift from the text. The one series the tables do not carry, the 2-year SST
product over the satellite era, goes through the same estimator here; Fig. 4b
draws it and the AR(1) sensitivity beside it.

Writes R_fig4_new_trends.png/.pdf/.csv for the 1900-1985 top panel and
R_fig4_new_trends_1900-2000.* for the century one, the bottom panel being the
same in both. Each also comes in a _fullamoc version, whose bottom panel is the
whole transport rather than the non-Ekman part - the violins are trends of the
full AMOC of each member, so that is the version in which the observation and the
models are strictly the same quantity, while the non-Ekman one isolates what the
reconstruction constrains best and is what the text quotes.
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_rgb  # noqa: E402
from matplotlib.patches import Patch, Polygon  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ssh_trend_honesty as S  # noqa: E402
from gls_blocks_calibration import gls_ar1_wald  # noqa: E402
import tables_unified as U  # noqa: E402
import plot_fig4b_noekman_and_pdf as P  # noqa: E402
from plot_fig4b_noekman_and_pdf import (  # noqa: E402
    AMOC_MEAN_MAP, CSV_PATH, SCALE_DEC2CEN, add_all_violin,
    build_order_by_amoc, make_soft_cmap, normalize_model_name, panel_plot,
)

BNN = os.path.join(S.ROOT, "bnn_real_world_rec")
UNIFIED = os.path.join(HERE, "R_tables_unified.csv")
DROP_MODELS = ("CAS-ESM2-0", "INM-CM4-8")
# the LPF120 artifact calls CESM2 the test model, since it is the one ESM that
# never enters a fold. Its true AMOC is a CMIP6 member trend like any other, so
# it belongs in the violins, and the bottom panel already shows it; but its
# reconstruction is not out of sample, so it is left out of the persistence
# estimate, which is read from the held-out reconstructions
TEST_MODEL = "test_model(CESM2)"
# the top panel is drawn for both windows: 1900-1985, the span the earlier
# version of the figure used, and 1900-2000, the span the SST interval is
# quoted and validated over
TOP_WINDOWS = ((1900, 1985), (1900, 2000))
CAESAR2018 = -2.375
SST_INDEX_TREND = -2.22
ORANGE, BLUE = "#E08214", "#3B6FA8"
# one flat colour per ordering arrow, cold for the top panel and warm for the
# bottom one, both darker than anything the violins use
ARROW_A, ARROW_B = "#2166AC", "#B2182B"

# net downward shortwave sensitivity used to order and colour the top panel
ASR_MAP = {
    "ACCESS-CM2": 0.011857092227438408, "ACCESS-ESM1-5": 0.014893554203697335,
    "CAS-ESM2-0": 0.020470713524869528, "CESM2": 0.021538743949348533,
    "CMCC-ESM2": 0.004133331907451014, "CanESM5": 0.012011685045444682,
    "CanESM5-CanOE": 0.011990572698048374, "FGOALS-g3": -0.003239418791663882,
    "GFDL-CM4": 0.01800142727816291, "GFDL-ESM4": 0.010082614519302242,
    "GISS-E2-1-G": 0.01572108248807887, "HadGEM3-GC31-LL": 0.012554644085396499,
    "HadGEM3-GC31-MM": 0.012094279665669491, "INM-CM4-8": 0.005641965077352932,
    "IPSL-CM6A-LR": -0.0005154370575548654, "MIROC6": 0.004068202260233284,
    "MPI-ESM1-2-HR": 0.0011229192646631282, "MRI-ESM2-0": 0.020548003819369283,
    "NorESM2-LM": 0.022282111815345628, "NorESM2-MM": 0.012232974654473214,
    "UKESM1-1-LL": 0.022485546181942553,
}
# the stored slopes are per month; the colorbar quotes them per year
ASR_PER_YEAR = {m: 12.0 * v for m, v in ASR_MAP.items()}


def tabulated(group, label):
    """The fit the tables quote for this window: the measured-ACF row."""
    if not os.path.exists(UNIFIED):
        raise SystemExit(f"missing {UNIFIED}; run tables_unified.py first")
    d = pd.read_csv(UNIFIED)
    g = d[(d.group == group) & (d.label == label) & (d.rho_from == "ACF")]
    if not len(g):
        raise SystemExit(f"no row for {group}/{label} in {UNIFIED}")
    r = g.iloc[0]
    return dict(rho=float(r.rho), n=int(r.n), trend=float(r.trend),
                two_sigma=float(r.two_se), lo=float(r.lo), hi=float(r.hi),
                p_one_sided=float(r.p), rho_source=str(r.rho_from))


def observed(y0, y1, full=False):
    """One interval per product, as the tables state it.

    The two bands the panels shade are read from R_tables_unified.csv rather
    than recomputed, so the figure cannot drift from the text: both are
    generalised least squares on annual means, weighted by the pooled
    within-fold sigma and correlated as the reconstruction errors on held-out
    CMIP6 models actually are, with the across-fold spread of the trend added at
    the trend level. The 2-year SST line of the bottom panel is not a table row,
    so it goes through the same estimator here, which is what Fig. 4b draws for
    it as well.

    full switches the satellite era from the non-Ekman transport to the whole
    one. The violins of the bottom panel are trends of the full AMOC of each
    member, so it is that version, not the default, in which the observation and
    the models are the same quantity; the non-Ekman one isolates the part the
    reconstruction constrains best and is what the text quotes.
    """
    rows = []
    for name, period, colour, group, label in (
            ("DL SST", f"{y0}-{y1}", BLUE, "fig4", f"SST {y0}--{y1}, 10-year"),
            ("DL SSH", "1993-2023", ORANGE, "ssh",
             f"1993--2023, {'+Ekman' if full else 'non-Ekman'}")):
        rows.append(dict(product=name, period=period, colour=colour,
                         **tabulated(group, label)))

    src = U.SST_24_FULL if full else U.SST_24
    r = U.case("fig4", "SST 1993--2023, 2-year", src, 1993, 2023,
               U.BLOCK["2-year"], "ART_24", "tos", U.ACF_LAG["2-year"])[0]
    rows.append(dict(product="DL SST", period="1993-2023", colour=BLUE,
                     rho_source=r["rho_from"], rho=r["rho"], n=r["n"],
                     trend=r["trend"], two_sigma=r["two_se"], lo=r["lo"],
                     hi=r["hi"], p_one_sided=r["p"]))
    return pd.DataFrame(rows)


def cmip_window(y0, y1):
    """True member trends over the window, and the persistence the ESMs read.

    The reconstruction cannot pin down its own lag-1 over a century, so the
    interval drawn for the record uses the median the ensemble reads over a
    window of the same length. It is rounded to two digits, the precision the
    caption quotes.
    """
    d = pd.read_parquet(S.ART_120)
    d = d[(d.scenario == "historical") & (d["var"] == "tos")
          & ~d.model.isin(DROP_MODELS)].drop_duplicates(
              subset=["model", "member", "year"])
    rows, rhos = [], []
    for (model, member), g in d.groupby(["model", "member"]):
        w = g.sort_values("year")
        w = w[(w.year >= y0) & (w.year <= y1)]
        if len(w) < y1 - y0:
            continue
        name = "CESM2" if model == TEST_MODEL else model
        rows.append(dict(model=name, member=member,
                         trend=float(np.polyfit(w.year.values.astype(float),
                                                w.tgt.values, 1)[0] * 100)))
        mu = w.groupby("year").mu.mean().values
        if model != TEST_MODEL and np.isfinite(mu).all():
            rhos.append(gls_ar1_wald(mu, w.groupby("year").sig.mean().values,
                                     1)[2])
    return pd.DataFrame(rows), round(float(np.median(rhos)), 2)


def cmip_1993_2023():
    d = pd.read_csv(CSV_PATH).rename(
        columns={"run": "member", "slope_Sv_per_decade": "trend"})
    d = d.assign(model=lambda x: x["member"].str.split("_", n=1).str[0].map(
        normalize_model_name))
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=["trend", "model"])
    d = d[~d.model.isin(DROP_MODELS)]
    d["trend"] = d["trend"] * SCALE_DEC2CEN
    return d


def full_height_colorbar(ax, sm, label, ticks=None, fmt=None, **_):
    """The panel's colorbar, spanning its whole height rather than 44% of it."""
    from matplotlib.ticker import FormatStrFormatter
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    cax = inset_axes(ax, width="1.9%", height="100%", loc="center left",
                     bbox_to_anchor=(1.022, 0.0, 1, 1),
                     bbox_transform=ax.transAxes, borderpad=0.0)
    cb = plt.colorbar(sm, cax=cax)
    cb.set_label(label, fontsize=17, labelpad=8)
    cb.ax.tick_params(labelsize=14, length=2.5, width=0.8)
    if ticks is not None:
        cb.set_ticks(ticks)
    if fmt is not None:
        cb.ax.yaxis.set_major_formatter(FormatStrFormatter(fmt))
    cb.outline.set_linewidth(0.8)
    cb.outline.set_edgecolor("0.6")
    return cb


def direction_arrow(fig, ax, name, colour, y, x0=0.30, x1=0.47):
    """A thick arrow naming what the models are ordered by along the axis.

    Both panels sort the models by the quantity their colourbar shows, which the
    reader otherwise has to infer from the colours alone. The arrow deepens from
    a pale tint to the full colour along its length, which says the same thing
    as the head; one hue per panel and a shaft that tapers towards the tail keep
    that shading from reading as a second colourbar. It sits in a strip of
    headroom above the violins, at a height given in data coordinates.

    Its outline is built in inches and converted, so that the head is not
    sheared by the aspect ratio of the panel; the panel must therefore already
    be in its final position.
    """
    bb = ax.get_position()
    w_in, h_in = bb.width * fig.get_figwidth(), bb.height * fig.get_figheight()
    lo, hi = ax.get_ylim()

    def dy(inch):
        return inch / h_in * (hi - lo)

    head_len = 0.34 / w_in
    tail, neck, half_head = dy(0.045) / 2, dy(0.135) / 2, dy(0.30) / 2
    xs = x1 - head_len
    tr = blended_transform_factory(ax.transAxes, ax.transData)
    ax.set_autoscale_on(False)
    outline = Polygon(
        [(x0, y - tail), (xs, y - neck), (xs, y - half_head), (x1, y),
         (xs, y + half_head), (xs, y + neck), (x0, y + tail)],
        closed=True, transform=tr, facecolor="none", edgecolor="none")
    ax.add_patch(outline)
    pale = tuple(0.30 * c + 0.70 for c in to_rgb(colour))
    im = ax.imshow(
        np.linspace(0, 1, 512)[None, :],
        cmap=LinearSegmentedColormap.from_list("pale_to_full", [pale, colour]),
        extent=(x0, x1, y - half_head, y + half_head), transform=tr,
        aspect="auto", interpolation="bilinear", zorder=5)
    im.set_clip_path(outline)  # clip_on must stay True or the path is ignored
    # lighter, echoing the tail of the shading, but not so light as to be faint
    ax.text(x0 - 0.009, y, "weak", transform=tr, ha="right", va="center",
            fontsize=15, color=tuple(0.78 * c + 0.22 for c in to_rgb(colour)))
    ax.text(x1 + 0.011, y, f"strong {name}", transform=tr, ha="left",
            va="center", fontsize=15, color=colour)


def draw_bands(ax, est, band_products=None):
    """Central line for every product, band only for those asked for.

    A product whose interval would span the whole panel is drawn as a line
    alone; its numbers are reported in the table instead.
    """
    for _, r in est.sort_values("two_sigma", ascending=False).iterrows():
        if band_products is None or r["product"] in band_products:
            ax.axhspan(r.lo, r.hi, color=r.colour, alpha=0.16, zorder=0)
        ax.axhline(r.trend, color=r.colour, lw=2.2, zorder=3)


def figure(y0, y1, bot, stem, full=False):
    """The two-panel figure with the top panel drawn over y0-y1.

    full draws the bottom panel for the whole transport rather than the
    non-Ekman one, which is the version in which the observation and the
    violins are the same quantity.
    """
    top, rho_top = cmip_window(y0, y1)
    est = observed(y0, y1, full)
    print(f"\n=== top panel {y0}-{y1}, CMIP6 ensemble rho {rho_top:.2f}, "
          f"bottom panel {'full' if full else 'non-Ekman'} AMOC ===")
    print(est[["product", "period", "rho_source", "rho", "trend", "two_sigma",
               "lo", "hi", "p_one_sided"]].round(3).to_string(index=False))

    order_top = [m for m in sorted(top.model.unique(),
                                   key=lambda m: ASR_MAP.get(m, np.nan))
                 if m in ASR_MAP]
    order_bot = build_order_by_amoc(bot["model"], AMOC_MEAN_MAP, ascending=True)
    print(f"panel a: {len(top)} members, {top.model.nunique()} models; "
          f"panel b: {len(bot)} members, {bot.model.nunique()} models")

    P.add_colorbar_center_right = full_height_colorbar
    mpl.rcParams.update({
        "font.size": 13, "axes.labelsize": 19, "legend.fontsize": 15,
        "xtick.labelsize": 21, "ytick.labelsize": 19, "axes.linewidth": 0.9,
        "figure.dpi": 150})
    sns.set(style="ticks", context="paper")
    fig_w = 0.62 * max(18, max(len(order_top), len(order_bot)) + 4)
    fig, axes = plt.subplots(2, 1, figsize=(fig_w, 10.6))

    panel_plot(
        axes[0], add_all_violin(top), order_top, obs_mean=None, obs_ci=None,
        ylabel="Full AMOC at 26.5$\\degree$N trend\n"
               f"{y0}$-${y1} (Sv / century)",
        violin_index_map=ASR_PER_YEAR,
        violin_index_label=r"$ASR_{HD}$ (W m$^{-2}$ year$^{-1}$)",
        violin_cmap=make_soft_cmap("viridis", whiten=0.22),
        robust_colors=False, add_violin_colorbar=True, colorbar_fmt="%.2f",
        ylab_fs=19)
    draw_bands(axes[0], est[est.period == f"{y0}-{y1}"])
    axes[0].axhline(CAESAR2018, color="0.25", ls=(0, (1, 2)), lw=1.6, zorder=2)
    # headroom for the arrow, so that it is not read as part of the violins
    axes[0].set_ylim(-6, 7.6)
    axes[0].set_yticks(np.arange(-6, 6.1, 2))

    panel_plot(
        axes[1], add_all_violin(bot), order_bot, obs_mean=None, obs_ci=None,
        ylabel=("Full" if full else "Non-Ekman")
               + " AMOC at 26.5$\\degree$N trend\n1993$-$2023 (Sv / century)",
        violin_index_map=AMOC_MEAN_MAP, violin_index_label="Mean AMOC (Sv)",
        violin_cmap=make_soft_cmap("cividis", whiten=0.18),
        robust_colors=True, robust_p_lo=5, robust_p_hi=95,
        add_violin_colorbar=True, colorbar_fmt="%.0f", ylab_fs=19)
    draw_bands(axes[1], est[est.period == "1993-2023"],
               band_products={"DL SSH"})
    axes[1].axhline(SST_INDEX_TREND, color="0.25", ls=(0, (1, 2)), lw=1.6,
                    zorder=2)
    axes[1].set_ylim(-27, 9)
    axes[1].set_yticks(np.arange(-25, 5.1, 5))

    for ax, lab in zip(axes, ["a", "b"]):
        ax.text(0.008, 0.98, lab, transform=ax.transAxes, ha="left", va="top",
                fontsize=20, fontweight="bold")
        # at this label size 30 degrees is not enough to keep the model names
        # from running into one another
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

    fig.legend(
        [Line2D([0], [0], color=BLUE, lw=2.5),
         Line2D([0], [0], color=ORANGE, lw=2.5),
         Line2D([0], [0], color="0.25", ls=(0, (1, 2)), lw=1.8)],
        ["DL SST", "DL SSH", "SST-index trend"],
        loc="center", bbox_to_anchor=(0.5, 0.505), ncol=3, frameon=False,
        fontsize=15.5)
    fig.tight_layout()
    plt.subplots_adjust(top=0.965, hspace=0.62)
    # tight_layout ignores the inset colorbars, so the two panels end up with
    # different widths whenever their colorbar labels differ in length
    box = [ax.get_position() for ax in axes]
    x0, x1 = max(b.x0 for b in box), min(b.x1 for b in box)
    for ax, b in zip(axes, box):
        ax.set_position([x0, b.y0, x1 - x0, b.height])
    # after the panels are in their final boxes, so the heads stay symmetric
    direction_arrow(fig, axes[0], "aerosol forcing", ARROW_A, 6.9)
    direction_arrow(fig, axes[1], "mean AMOC", ARROW_B, 7.7)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, f"{stem}.{ext}"),
                    bbox_inches="tight", dpi=200)
    plt.close(fig)
    est.to_csv(os.path.join(HERE, f"{stem}.csv"), index=False)
    print(f"[SAVED] {HERE}/{stem}.png")


def figure_b(bot, stem, full=False):
    """The satellite-era panel on its own, for use as a single figure.

    The bottom panel does not depend on the window the top one draws, so there
    is one of these per Ekman convention rather than one per window. Everything
    it shows is the same call as in the two-panel version; only the geometry and
    the legend placement differ, the legend moving below the model names now
    that there is no gap between panels to put it in.
    """
    est = observed(*TOP_WINDOWS[-1], full)
    order_bot = build_order_by_amoc(bot["model"], AMOC_MEAN_MAP, ascending=True)
    print(f"\n=== satellite era alone, {'full' if full else 'non-Ekman'} AMOC, "
          f"{len(bot)} members, {bot.model.nunique()} models ===")
    print(est[est.period == "1993-2023"][
        ["product", "rho_source", "rho", "trend", "two_sigma", "lo", "hi",
         "p_one_sided"]].round(3).to_string(index=False))

    P.add_colorbar_center_right = full_height_colorbar
    mpl.rcParams.update({
        "font.size": 13, "axes.labelsize": 19, "legend.fontsize": 15,
        "xtick.labelsize": 21, "ytick.labelsize": 19, "axes.linewidth": 0.9,
        "figure.dpi": 150})
    sns.set(style="ticks", context="paper")
    fig, ax = plt.subplots(figsize=(0.62 * max(18, len(order_bot) + 4), 6.2))

    panel_plot(
        ax, add_all_violin(bot), order_bot, obs_mean=None, obs_ci=None,
        ylabel=("Full" if full else "Non-Ekman")
               + " AMOC\nat 26.5$\\degree$N\n1993$-$2023 (Sv / century)",
        violin_index_map=AMOC_MEAN_MAP, violin_index_label="Mean AMOC (Sv)",
        violin_cmap=make_soft_cmap("cividis", whiten=0.18),
        robust_colors=True, robust_p_lo=5, robust_p_hi=95,
        add_violin_colorbar=True, colorbar_fmt="%.0f", ylab_fs=19)
    draw_bands(ax, est[est.period == "1993-2023"], band_products={"DL SSH"})
    ax.axhline(SST_INDEX_TREND, color="0.25", ls=(0, (1, 2)), lw=1.6, zorder=2)
    ax.set_ylim(-27, 9)
    ax.set_yticks(np.arange(-25, 5.1, 5))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    fig.tight_layout()
    plt.subplots_adjust(left=0.14, bottom=0.34)
    direction_arrow(fig, ax, "mean AMOC", ARROW_B, 7.7)
    fig.legend(
        [Line2D([0], [0], color=BLUE, lw=2.5),
         Line2D([0], [0], color=ORANGE, lw=2.5),
         Line2D([0], [0], color="0.25", ls=(0, (1, 2)), lw=1.8)],
        ["DL SST", "DL SSH", "SST-index trend"],
        loc="upper center", bbox_to_anchor=(0.5, 0.10), ncol=3, frameon=False,
        fontsize=15.5)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, f"{stem}.{ext}"), bbox_inches="tight",
                    dpi=200)
    plt.close(fig)
    est[est.period == "1993-2023"].to_csv(
        os.path.join(HERE, f"{stem}.csv"), index=False)
    print(f"[SAVED] {HERE}/{stem}.png")


def main():
    bot = cmip_1993_2023()
    for y0, y1 in TOP_WINDOWS:
        stem = ("R_fig4_new_trends" if (y0, y1) == TOP_WINDOWS[0]
                else f"R_fig4_new_trends_{y0}-{y1}")
        for full in (False, True):
            figure(y0, y1, bot, stem + ("_fullamoc" if full else ""), full)
    for full in (False, True):
        figure_b(bot, "R_fig4b_only" + ("_fullamoc" if full else ""), full)


if __name__ == "__main__":
    main()
