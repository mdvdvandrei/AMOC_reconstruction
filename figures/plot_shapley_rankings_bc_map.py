#!/usr/bin/env python3
"""Rebuild R_shapley_rankings_bc with a region map panel (a).

  a  Atlantic region boxes used for exact Shapley attribution
  b  SST ranking  (1870–2025)
  c  SSH ranking  (1993–2023)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature

ROOT = Path("/home/am334/link_am334/moc_mmodel")
SRC = ROOT / "shapley_exact_temporal"
OUT = ROOT / "bnn_real_world_rec" / "reviewer_response"

# Same boxes as rise_regional_v2.py / shapley_exact_temporal
REGIONS = {
    "Nordic Seas":       (65,  80, -20,  20),
    "Labrador Sea":      (50,  65, -65, -45),
    "Irminger Basin":    (50,  65, -45, -20),
    "Eastern SPG":       (50,  65, -20,   0),
    "Gulf Stream":       (30,  50, -80, -50),
    "Sargasso Sea":      (30,  50, -50, -25),  # display: W. Subtropical Gyre
    "NE Atlantic":       (30,  50, -25,   0),
    "Caribbean":         (10,  30, -80, -55),
    "Trop. N. Atlantic": (10,  30, -55,   0),
    "Equat. Atlantic":   (-10, 10, -60,  10),
    "S. Atlantic":       (-45,-10, -60,  15),
}

DISPLAY_NAME = {
    "Sargasso Sea": "Subtropical Gyre",
}

FILL = "#9AA8B5"
EDGE = "#3D4F5F"
LABEL_FC = "#F7F7F5"

MAP_LABEL = {
    "Nordic Seas":       "Nordic\nSeas",
    "Labrador Sea":      "Lab.",
    "Irminger Basin":    "Irm.",
    "Eastern SPG":       "E. SPG",
    "Gulf Stream":       "Gulf\nStream",
    "Sargasso Sea":      "Subtrop.\nGyre",
    "NE Atlantic":       "NE Atl.",
    "Caribbean":         "Carib.",
    "Trop. N. Atlantic": "TNA",
    "Equat. Atlantic":   "Eq. Atl.",
    "S. Atlantic":       "S. Atl.",
}


def disp(name: str) -> str:
    return DISPLAY_NAME.get(name, name)


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.weight": "normal",
        "axes.titleweight": "normal",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "figure.dpi": 150,
        "font.size": 11,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })


def plot_region_map(ax):
    ax.set_extent([-85, 25, -50, 82], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="#EDEBE6", edgecolor="none", zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor="#FFFFFF", edgecolor="none", zorder=0)
    ax.coastlines(resolution="110m", linewidth=0.55, color="0.40", zorder=3)
    gl = ax.gridlines(draw_labels=True, linewidth=0.35, color="0.78",
                      linestyle=":", xlocs=range(-80, 30, 20),
                      ylocs=range(-40, 90, 20))
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 8}
    gl.ylabel_style = {"size": 8}

    for name, (la0, la1, lo0, lo1) in REGIONS.items():
        ax.add_patch(mpatches.Rectangle(
            (lo0, la0), lo1 - lo0, la1 - la0,
            transform=ccrs.PlateCarree(),
            facecolor=FILL, alpha=0.35, edgecolor=EDGE, linewidth=1.2, zorder=2,
        ))
        ax.text(
            0.5 * (lo0 + lo1), 0.5 * (la0 + la1), MAP_LABEL[name],
            transform=ccrs.PlateCarree(),
            ha="center", va="center", fontsize=10.5, color="0.15",
            fontweight="medium", zorder=4,
            bbox=dict(boxstyle="round,pad=0.18", fc=LABEL_FC, ec="none", alpha=0.85),
        )

    ax.set_title("Shapley attribution regions", fontsize=13, pad=8)
    ax.text(-0.02, 1.04, "a", transform=ax.transAxes,
            fontsize=16, fontweight="bold", va="bottom", ha="right", clip_on=False)


def density_per_area(df: pd.DataFrame) -> pd.DataFrame:
    """|phi| / area in Sv / (10^6 km^2)."""
    out = df.copy()
    out["val"] = out["mean_abs_phi_per_1e6km2"]
    out = out.sort_values("val", ascending=False).reset_index(drop=True)
    return out


def plot_ranking(ax, dens, title, letter, color):
    y = np.arange(len(dens))[::-1]
    ax.barh(y, dens["val"], height=0.72,
            color=color, edgecolor="white", lw=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels([disp(r) for r in dens["region"]], fontsize=10)
    ax.set_xlabel(r"Time-mean $|$Shapley$|$ (Sv$/10^{6}\,\mathrm{km}^{2}$)",
                  fontsize=10)
    ax.set_title(title, fontsize=12, pad=6)
    ax.text(-0.08, 1.04, letter, transform=ax.transAxes,
            fontsize=16, fontweight="bold", va="bottom", ha="right", clip_on=False)


def main():
    style()
    dens_sst = density_per_area(pd.read_csv(OUT / "R_shapley_density_SST.csv"))
    dens_ssh = density_per_area(pd.read_csv(OUT / "R_shapley_density_SSH.csv"))
    xmax = max(dens_sst["val"].max(), dens_ssh["val"].max()) * 1.08

    fig = plt.figure(figsize=(11.4, 7.6))
    gs = gridspec.GridSpec(
        2, 2, figure=fig,
        width_ratios=[1.15, 1.0],
        height_ratios=[1.0, 1.0],
        hspace=0.28, wspace=0.28,
        left=0.04, right=0.98, top=0.94, bottom=0.08,
    )

    ax_map = fig.add_subplot(gs[:, 0], projection=ccrs.PlateCarree())
    plot_region_map(ax_map)

    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1])
    plot_ranking(ax_b, dens_sst, "SST  (1870–2025)", "b", "#4C78A8")
    plot_ranking(ax_c, dens_ssh, "SSH  (1993–2023)", "c", "#E45756")
    ax_b.set_xlim(0, xmax)
    ax_c.set_xlim(0, xmax)
    # shared x label only on bottom panel
    ax_b.set_xlabel("")
    ax_c.set_xlabel(r"Time-mean $|$Shapley$|$ (Sv$/10^{6}\,\mathrm{km}^{2}$)",
                    fontsize=10)

    for name in ("R_shapley_rankings_bc", "R_shapley_combined"):
        fig.savefig(OUT / f"{name}.png", dpi=220, bbox_inches="tight")
        fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
        print(f"Saved → {OUT / name}.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
