# -*- coding: utf-8 -*-
import os
import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

BASE = r"D:\code\comfort-area\data\v5\r_nino34"

f_nh  = os.path.join(BASE, "mild_area_1981_2020_APT_annual_nh_detrend.txt")
f_sh  = os.path.join(BASE, "mild_area_1981_2020_APT_annual_sh_detrend.txt")

FILL = 32767.0
years = np.arange(1981, 2021)

very_strong_el_nino = {1983, 1998, 2016}
moderate_strong_el_nino = {1982, 1987, 1992, 1995, 1997, 2003, 2010, 2015}

weak_el_nino = {1986, 1991, 1994, 2002, 2004, 2007, 2009, 2014, 2019}
weak_la_nina = {1984, 1985, 1989, 1996, 2001, 2006, 2008, 2012, 2017, 2018}

moderate_strong_la_nina = {1988, 1999, 2000, 2005, 2007, 2008, 2010, 2011, 2020}

neutral_years = {1981, 1990, 1993, 2013}

all_classified = (
    very_strong_el_nino
    | moderate_strong_el_nino
    | moderate_strong_la_nina
    | neutral_years
)

missing_years = sorted(set(years) - all_classified)
if missing_years:
    print("Info: these years are not shaded (treated as neutral/other):", missing_years)

def read_1col_txt(path):
    arr = np.loadtxt(path)
    arr = np.asarray(arr).squeeze()
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return arr

def pick_coord(obj, cand_names):
    for n in cand_names:
        if n in obj.coords:
            return obj.coords[n]
        if n in obj.variables:
            return obj[n]
    return None

def classify_enso_year(year):

    if year in very_strong_el_nino:
        return "very_strong_el_nino"
    elif year in moderate_strong_el_nino:
        return "moderate_strong_el_nino"
    elif year in moderate_strong_la_nina:
        return "moderate_strong_la_nina"
    else:
        return None

def add_trend(ax, x, y, color, ls=(0, (4, 3)), lw=1.0):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() >= 2:
        p = np.polyfit(x[ok], y[ok], 1)
        yfit = np.polyval(p, x[ok])
        ax.plot(x[ok], yfit, color='gray', lw=lw, ls=ls, zorder=6)

nh = read_1col_txt(f_nh)
sh = read_1col_txt(f_sh)

mpl.rcParams["font.family"] = "Times New Roman"
mpl.rcParams["axes.labelsize"] = 15
mpl.rcParams["xtick.labelsize"] = 15
mpl.rcParams["ytick.labelsize"] = 15
mpl.rcParams["legend.fontsize"] = 15

fig = plt.figure(figsize=(11, 8))
gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.35], hspace=0.30)

panel_kw = dict(ha="left", va="top", fontsize=18, fontweight="bold")

ax1 = fig.add_subplot(gs[0, 0])
ax1r = ax1.twinx()

ax1.axhline(0, color="gray", lw=1.0, ls=(0, (4, 3)), zorder=1)

enso_colors = {
    "very_strong_el_nino":     "#fac4dc",
    "moderate_strong_el_nino": "#fbe4dc",
    "moderate_strong_la_nina": "#ddeef7",
}

for yr in years:
    cat = classify_enso_year(yr)
    if cat is not None:
        ax1.axvspan(
            yr - 0.48, yr + 0.48,
            color=enso_colors[cat], alpha=0.95, zorder=0
        )

c_nh = "#ee3b73"
c_sh = "#0078bf"

ax1.plot(
    years, nh, lw=2.0, color=c_nh, marker="o", ms=6.5, mew=1.8,
    label="NH", zorder=5
)
ax1r.plot(
    years, sh, lw=2.0, color=c_sh, marker="x", ms=7, mew=1.8,
    label="SH", zorder=5
)

ax1.set_xlim(1980, 2022)
ax1.set_xticks(np.arange(1980, 2021, 5))

ax1.set_ylabel("Percentage (NH) [%]", color=c_nh)
ax1r.set_ylabel("Percentage (SH) [%]", color=c_sh)

NH_YLIM  = (-5, 5)
NH_TICKS = [-5, -2.5,0, 2.5, 5]
SH_YLIM  = (-1, 1)
SH_TICKS = [-1, -0.5, 0, 0.5, 1]

ax1.set_ylim(*NH_YLIM)
ax1.set_yticks(NH_TICKS)

ax1r.set_ylim(*SH_YLIM)
ax1r.set_yticks(SH_TICKS)

ax1.tick_params(axis='y', labelsize=15, labelcolor=c_nh)
ax1r.tick_params(axis='y', labelsize=15, labelcolor=c_sh)

line_handles = [
    Line2D([0], [0], color=c_nh, lw=2.0, marker="o", ms=7, mew=1.8, label="NH"),
    Line2D([0], [0], color=c_sh, lw=2.0, marker="x", ms=7, mew=1.8, label="SH"),
]

patch_handles = [
    Patch(facecolor=enso_colors["very_strong_el_nino"], edgecolor="none", linewidth=0.0, label="Very strong El Niño"),
    Patch(facecolor=enso_colors["moderate_strong_el_nino"], edgecolor="none", linewidth=0.0, label="Moderate-strong El Niño"),
    Patch(facecolor=enso_colors["moderate_strong_la_nina"], edgecolor="none", linewidth=0.0, label="Moderate-strong La Niña"),
]

leg1 = ax1.legend(
    handles=line_handles,
    frameon=False, loc="upper left", bbox_to_anchor=(0.00, 1.055),
    ncol=3, columnspacing=1.2, handletextpad=0.5
)
ax1.add_artist(leg1)

ax1.legend(
    handles=patch_handles,
    frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1),
    ncol=3, columnspacing=1.0, handletextpad=0.5
)

out_png = os.path.join(r"D:\code\comfort-area\fig\v7", "figS11.png")
plt.savefig(out_png, dpi=800, bbox_inches="tight")