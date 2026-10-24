# -*- coding: utf-8 -*-
import os
import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

BASE = r"D:\code\comfort-area\data\v5\ERA5"
FILL = 32767.0

mpl.rcParams["font.family"] = "Times New Roman"
mpl.rcParams["font.size"] = 14

SEASONS = ["mjjaso", "ndjfma"]
SEASON_TITLES = {"mjjaso": "MJJASO", "ndjfma": "NDJFMA"}

METRICS = ["APT", "WBGT", "UTCI", "NET"]

YEARS = np.arange(1981, 2021)
assert len(YEARS) == 40


ALPHA_SIG = 0.05


Y_LIM_BOTH = (-12, 4)
Y_TICKS_BOTH = [-12, -8, -4, 0, 4]

Y_LIM_BOTH2 = (-12, 0)
Y_TICKS_BOTH2 = [-12, -8, -4, 0]


BAR_COLORS = {
    "APT":  "#00a4d2",
    "WBGT": "#a84f9e",
    "UTCI": "#00994c",
    "NET":  "#f393be",
}


MAP_GREEN = "#7bca73"
OCEAN_COLOR = "#e0f2fc"


PIE_OUT_NH = "#f393be"
PIE_OUT_SH = "#00a4d2"
PIE_IN_NH  = "#ee3b73"
PIE_IN_SH  = "#1f78b4"


PANEL_FS = 22
PANEL_DX = -0.13
PANEL_DY = -0.19

def add_panel_label(ax, label):
    ax.text(
        PANEL_DX, 1.0 - PANEL_DY, label,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=PANEL_FS, fontweight="bold"
    )


def read_nc_mild_area(path_nc, vname="mild_area", fill=FILL):
    ds = xr.open_dataset(path_nc)
    da = ds[vname].astype("float64")
    da = da.where(da != float(fill))
    lon = da["lon"].values
    lat = da["lat"].values
    data = da.values
    return lon, lat, data

def read_txt_1col(path_txt):
    arr = np.loadtxt(path_txt, dtype="float64")
    arr = np.atleast_1d(arr).squeeze()
    if arr.size != 40:
        raise ValueError(f"[ERROR] {os.path.basename(path_txt)} 行数不是40：{arr.size}")
    return arr

def get_txt_path(metric, season, hemi):
    return os.path.join(BASE, f"mild_area_1981_2020_{metric}_{season}_{hemi}.txt")

def get_nc_path_for_maps(season):
    return os.path.join(BASE, f"mild_area_1981_2020_APT_{season}.nc")

def trend_per_decade(y, x=YEARS):
    x = np.asarray(x, dtype="float64")
    y = np.asarray(y, dtype="float64")
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    n = x.size
    if n < 3:
        return np.nan, np.nan

    try:
        from scipy.stats import linregress
        res = linregress(x, y)
        return res.slope * 100.0, res.pvalue
    except Exception:
        a, b = np.polyfit(x, y, 1)
        yhat = a*x + b
        resid = y - yhat
        sse = np.sum(resid**2)
        sxx = np.sum((x - x.mean())**2)
        se_a = np.sqrt((sse / (n - 2)) / sxx) if (n > 2 and sxx > 0) else np.nan
        tval = a / se_a if (np.isfinite(se_a) and se_a > 0) else np.nan
        if np.isfinite(tval):
            from math import erf, sqrt
            def norm_cdf(z): return 0.5*(1.0 + erf(z/sqrt(2.0)))
            p = 2.0*(1.0 - norm_cdf(abs(tval)))
        else:
            p = np.nan
        return a * 100.0, p

def compute_trend_table(hemi):
    tr = np.full((len(SEASONS), len(METRICS)), np.nan)
    pv = np.full((len(SEASONS), len(METRICS)), np.nan)
    for i_s, s in enumerate(SEASONS):
        for j_m, mtr in enumerate(METRICS):
            ts = read_txt_1col(get_txt_path(mtr, s, hemi))
            slope10, p = trend_per_decade(ts, YEARS)
            tr[i_s, j_m] = slope10
            pv[i_s, j_m] = p
    return tr, pv

def move_yaxis_to_right_and_hide_left(ax, ylim, yticks, ylabel="Trend"):
    ax.tick_params(axis="y", which="both", left=False, labelleft=False)
    # ax.spines["left"].set_visible(False)

    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.tick_params(axis="y", which="both", right=True, labelright=True)

    ax.set_ylim(ylim)
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(v) for v in yticks], fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)

def plot_bars_NH_SH_for_one_season(ax,
                                  season_key,
                                  trend_nh, p_nh,
                                  trend_sh, p_sh,
                                  ylim=Y_LIM_BOTH, yticks=Y_TICKS_BOTH):

    i_s = SEASONS.index(season_key)

    tr = np.vstack([trend_nh[i_s, :], trend_sh[i_s, :]])
    pv = np.vstack([p_nh[i_s, :],    p_sh[i_s, :]])

    groups = ["NH", "SH"]
    x = np.arange(len(groups))
    width = 0.18
    offsets = np.linspace(-1.5*width, 1.5*width, len(METRICS))

    ax.set_ylim(ylim)
    y0, y1 = float(ylim[0]), float(ylim[1])
    yrange = max(1e-9, (y1 - y0))
    dy = 0.015 * yrange

    for j, mtr in enumerate(METRICS):
        y = tr[:, j]
        xpos = x + offsets[j]
        color = BAR_COLORS.get(mtr, None)

        ax.bar(
            xpos, y, width=width,
            label=mtr,
            color=color, edgecolor=color, linewidth=0.8
        )

        for i in range(len(groups)):
            if not np.isfinite(y[i]) or not np.isfinite(pv[i, j]):
                continue
            y_mark = y[i] + (dy if y[i] >= 0 else -dy)

            if pv[i, j] < ALPHA_SIG:
                ax.scatter(
                    xpos[i], y_mark, s=38, marker="o",
                    facecolors="k", edgecolors="k", linewidths=0.8,
                    zorder=6, clip_on=True
                )
            else:
                ax.scatter(
                    xpos[i], y_mark, s=38, marker="o",
                    facecolors="none", edgecolors="k", linewidths=1.2,
                    zorder=6, clip_on=True
                )

    ax.axhline(0, color="k", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=14)

    move_yaxis_to_right_and_hide_left(ax, ylim, yticks, ylabel="Trend [%/C]")

def add_pie_for_season_on_map(ax, season):
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from matplotlib.patches import Wedge

    ts_nh = read_txt_1col(get_txt_path("APT", season, "nh"))
    ts_sh = read_txt_1col(get_txt_path("APT", season, "sh"))
    pct_nh = float(np.nanmean(ts_nh))
    pct_sh = float(np.nanmean(ts_sh))

    land_nh = 66.65
    land_sh = 33.35

    x0, y0, w_box, h_box = (0.52, 0.02, 0.46, 0.60)
    ax_pie = inset_axes(
        ax, width="100%", height="100%",
        bbox_to_anchor=(x0, y0, w_box, h_box),
        bbox_transform=ax.transAxes,
        borderpad=0.0
    )
    ax_pie.set_aspect("equal")
    ax_pie.axis("off")

    outer_sizes  = [land_sh, land_nh]
    outer_colors = [PIE_OUT_SH, PIE_OUT_NH]
    wedges_outer, _ = ax_pie.pie(
        outer_sizes,
        colors=outer_colors,
        startangle=90,
        counterclock=True,
        wedgeprops=dict(edgecolor="none", linewidth=1.2)
    )

    def _mid_xy(w, r=0.55):
        th = np.deg2rad(0.5*(w.theta1 + w.theta2))
        return r*np.cos(th), r*np.sin(th)

    x_sh, y_sh = _mid_xy(wedges_outer[0], r=0.55)
    x_nh, y_nh = _mid_xy(wedges_outer[1], r=0.55)

    DX_SH, DY_SH = -0.1, -0.15
    ax_pie.text(x_sh + DX_SH, y_sh + DY_SH, "SH", ha="center", va="center",
                fontsize=12, fontweight="bold")
    ax_pie.text(x_nh, y_nh, "NH", ha="center", va="center",
                fontsize=12, fontweight="bold")

    angle_nh = max(0.0, min(360.0, 360.0 * pct_nh / 100.0))
    angle_sh = max(0.0, min(360.0, 360.0 * pct_sh / 100.0))

    theta1_sh_outer, theta2_sh_outer = wedges_outer[0].theta1, wedges_outer[0].theta2
    boundary = theta2_sh_outer
    offset = 0.10

    theta1_sh_in = theta1_sh_outer + 5
    theta2_sh_in = min(theta1_sh_in + angle_sh, boundary)

    if theta2_sh_in > theta1_sh_in:
        th_mid_sh = 0.5*(theta1_sh_in + theta2_sh_in)
        rad_sh = np.deg2rad(th_mid_sh)
        dx_sh = offset*np.cos(rad_sh)
        dy_sh = offset*np.sin(rad_sh)

        wedge_sh = Wedge(
            center=(dx_sh, dy_sh), r=1.0,
            theta1=theta1_sh_in, theta2=theta2_sh_in,
            facecolor=PIE_IN_SH, edgecolor="k", linewidth=0.7,
            alpha=0.95, zorder=20
        )
        ax_pie.add_patch(wedge_sh)
        ax_pie.text(
            dx_sh + 0.62*np.cos(rad_sh), dy_sh + 0.62*np.sin(rad_sh),
            f"{pct_sh:.1f}",
            ha="center", va="center",
            fontsize=12, fontweight="bold", color="black", zorder=30
        )

    theta1_nh_outer, theta2_nh_outer = wedges_outer[1].theta1, wedges_outer[1].theta2
    theta1_nh_in = boundary + 160
    theta2_nh_in = min(theta1_nh_in + angle_nh, theta2_nh_outer)

    if theta2_nh_in > theta1_nh_in:
        th_mid_nh = 0.5*(theta1_nh_in + theta2_nh_in)
        rad_nh = np.deg2rad(th_mid_nh)
        dx_nh = offset*np.cos(rad_nh)
        dy_nh = offset*np.sin(rad_nh)

        wedge_nh = Wedge(
            center=(dx_nh, dy_nh), r=1.0,
            theta1=theta1_nh_in, theta2=theta2_nh_in,
            facecolor=PIE_IN_NH, edgecolor="k", linewidth=0.7,
            alpha=0.95, zorder=20
        )
        ax_pie.add_patch(wedge_nh)
        ax_pie.text(
            dx_nh + 0.62*np.cos(rad_nh), dy_nh + 0.62*np.sin(rad_nh),
            f"{pct_nh:.1f}",
            ha="center", va="center",
            fontsize=12, fontweight="bold", color="black", zorder=30
        )

maps = {}
for s in SEASONS:
    lon, lat, dat = read_nc_mild_area(get_nc_path_for_maps(s))
    maps[s] = (lon, lat, dat)

trend_nh, p_nh = compute_trend_table("nh")
trend_sh, p_sh = compute_trend_table("sh")

fig = plt.figure(figsize=(14.5, 7.8))
gs = GridSpec(
    nrows=2, ncols=2,
    height_ratios=[1.55, 0.75],
    hspace=0.22, wspace=0.12
)

use_cartopy = True
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.util import add_cyclic_point
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
except Exception:
    use_cartopy = False

from matplotlib.colors import ListedColormap, BoundaryNorm
cmap_green = ListedColormap(["#7bca73"])
cmap_green.set_bad(alpha=0.0)
norm_green = BoundaryNorm([0.5, 1.5], cmap_green.N)

def add_latlon_ticks_cartopy(ax,
                             xticks=None,
                             yticks=None,
                             labelsize=14,
                             pad=2,
                             show_left_labels=True,
                             show_bottom_labels=True,
                             add_extra_30w=True,
                             extra_30w_text="30°W",
                             extra_30w_x=0.955,
                             extra_30w_y=-0.046,
                             extra_tick_y0=0.0,
                             extra_tick_y1=-0.025):
    if xticks is None:
        xticks = [-150, -90, -30, 30, 90, 150]
    if yticks is None:
        yticks = np.arange(-60, 61, 30)

    ax.set_xticks(xticks, crs=ccrs.PlateCarree())
    ax.set_yticks(yticks, crs=ccrs.PlateCarree())

    ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
    ax.yaxis.set_major_formatter(LatitudeFormatter())

    ax.tick_params(
        axis="both", which="major",
        direction="out",
        length=3.8, width=0.8,
        labelsize=labelsize,
        pad=pad
    )

    ax.tick_params(axis="x", top=False, labeltop=False,
                   bottom=True, labelbottom=show_bottom_labels)
    ax.tick_params(axis="y", right=False, labelright=False,
                   left=True, labelleft=show_left_labels)

    if add_extra_30w and show_bottom_labels:
        ax.text(
            extra_30w_x, extra_30w_y, extra_30w_text,
            transform=ax.transAxes, ha="left", va="top",
            fontsize=labelsize, zorder=200, clip_on=False
        )
        ax.plot(
            [1.0, 1.0], [extra_tick_y0, extra_tick_y1],
            transform=ax.transAxes, color="black",
            linewidth=0.7, clip_on=False, zorder=200
        )

def plot_map(ax, lon, lat, data, title, show_left_labels=True):
    data = np.asarray(data, dtype="float64")
    show_green = np.where(data == 1.0, 1.0, np.nan)

    if use_cartopy:
        ax.set_extent([-180, 180, -60, 60], crs=ccrs.PlateCarree())

        show_green_cyc, lon_cyc = add_cyclic_point(show_green, coord=lon)
        Lon, Lat = np.meshgrid(lon_cyc, lat)
        ax.pcolormesh(
            Lon, Lat, show_green_cyc,
            transform=ccrs.PlateCarree(),
            shading="auto",
            cmap=cmap_green, norm=norm_green,
            zorder=10
        )

        ax.add_feature(
            cfeature.OCEAN.with_scale("110m"),
            facecolor=OCEAN_COLOR, edgecolor="none",
            zorder=20
        )

        ax.coastlines(linewidth=0.6, zorder=30)

        add_latlon_ticks_cartopy(
            ax,
            show_left_labels=show_left_labels,
            show_bottom_labels=True,
            add_extra_30w=True
        )

        try:
            ax.spines["geo"].set_zorder(999)
            ax.spines["geo"].set_linewidth(0.8)
            ax.spines["geo"].set_edgecolor("k")
        except Exception:
            ax.outline_patch.set_zorder(999)
            ax.outline_patch.set_linewidth(0.8)
            ax.outline_patch.set_edgecolor("k")
    else:
        ax.imshow(
            show_green, origin="lower", aspect="auto",
            extent=[lon.min(), lon.max(), lat.min(), lat.max()],
            cmap=cmap_green, norm=norm_green
        )
        ax.set_xlabel("Lon")
        ax.set_ylabel("Lat")

    ax.set_title(title, fontsize=16, loc="left")

PANEL_DY = -0.19
PANEL_DX = -0.13
if use_cartopy:
    ax_a = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree(central_longitude=150))
else:
    ax_a = fig.add_subplot(gs[0, 0])

lon, lat, dat = maps["mjjaso"]
plot_map(ax_a, lon, lat, dat, f"{SEASON_TITLES['mjjaso']}", show_left_labels=True)
add_panel_label(ax_a, "a")
add_pie_for_season_on_map(ax_a, "mjjaso")

PANEL_DY = -0.19
PANEL_DX = -0.04
ax_c = fig.add_subplot(gs[0, 1])
add_panel_label(ax_c, "c")
plot_bars_NH_SH_for_one_season(
    ax_c, "mjjaso",
    trend_nh, p_nh,
    trend_sh, p_sh,
    ylim=Y_LIM_BOTH2, yticks=Y_TICKS_BOTH2
)

PANEL_DY = -0.19
PANEL_DX = -0.13
if use_cartopy:
    ax_b = fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree(central_longitude=150))
else:
    ax_b = fig.add_subplot(gs[1, 0])

lon, lat, dat = maps["ndjfma"]
plot_map(ax_b, lon, lat, dat, f"{SEASON_TITLES['ndjfma']}", show_left_labels=True)
add_panel_label(ax_b, "b")
add_pie_for_season_on_map(ax_b, "ndjfma")

PANEL_DY = -0.19
PANEL_DX = -0.042
ax_d = fig.add_subplot(gs[1, 1])
add_panel_label(ax_d, "d")
plot_bars_NH_SH_for_one_season(
    ax_d, "ndjfma",
    trend_nh, p_nh,
    trend_sh, p_sh,
    ylim=Y_LIM_BOTH, yticks=Y_TICKS_BOTH
)

handles, labels = ax_c.get_legend_handles_labels()

ax_c.legend(
    handles, labels,
    ncol=2, frameon=False, fontsize=12,
    loc="upper left", bbox_to_anchor=(0.563, 0.3),
    borderaxespad=0.0, columnspacing=1.2
)

def shift_axes_y(ax_list, dy):
    for a in ax_list:
        pos = a.get_position()
        a.set_position([pos.x0, pos.y0 + dy, pos.width, pos.height])

ROW2_DY = 0.11
shift_axes_y([ax_b, ax_d], ROW2_DY)

pos_a  = ax_a.get_position()
pos_c  = ax_c.get_position()
ax_c.set_position([pos_c.x0, pos_a.y0, pos_c.width, pos_a.height])

out_png = os.path.join(r"D:\code\comfort-area\fig\v7", "fig2.pdf")
plt.savefig(out_png, dpi=600, bbox_inches="tight")