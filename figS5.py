# -*- coding: utf-8 -*-
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import to_rgb
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


mpl.rcParams["font.family"] = "Times New Roman"
mpl.rcParams["font.size"] = 14


FILE_A = r"D:\code\comfort-area\data\v5\sensitivity_analysis\one_factor_era5_10years_5.nc"

FILL   = 32767.0
V_ORI   = "shushidays_original"
V_ORI2  = "shushidays_original2"
V_WIND  = "shushidays_wind_alone"
V_RH    = "shushidays_rh_alone"
V_TAS   = "shushidays_tas_alone"


COL_TOP_WIND = np.array(to_rgb("#00a4d2"))
COL_BL_TAS   = np.array(to_rgb("#f393be"))
COL_BR_RH    = np.array(to_rgb("#ffff00"))

OCEAN_COLOR = "#e0f2fc"
proj = ccrs.PlateCarree(central_longitude=150)

def add_panel_label(ax, label, fs=20):
    ax.text(-0.060, 1.08, label, transform=ax.transAxes,
            ha="left", va="top", fontsize=fs, fontweight="bold")

def read_da(ds, vname):
    da = ds[vname].astype("float64")
    fv = da.attrs.get("_FillValue", None)
    mv = da.attrs.get("missing_value", None)
    if fv is not None:
        da = da.where(da != float(fv))
    if mv is not None:
        da = da.where(da != float(mv))
    da = da.where(da != FILL)
    da = da.where(np.abs(da) <= 500)
    return da

def make_rgb(ori, windA, rhA, tasA):
    d_wind = (windA - ori).values
    d_rh   = (rhA   - ori).values
    d_tas  = (tasA  - ori).values

    m_tas  = np.abs(d_tas)
    m_rh   = np.abs(d_rh)
    m_wind = np.abs(d_wind)

    sum_mag = m_tas + m_rh + m_wind
    chg = np.isfinite(sum_mag) & (sum_mag > 0)

    w_tas  = np.zeros_like(sum_mag, dtype=float)
    w_rh   = np.zeros_like(sum_mag, dtype=float)
    w_wind = np.zeros_like(sum_mag, dtype=float)

    w_tas[chg]  = m_tas[chg]  / sum_mag[chg]
    w_rh[chg]   = m_rh[chg]   / sum_mag[chg]
    w_wind[chg] = m_wind[chg] / sum_mag[chg]

    rgb = (w_wind[..., None] * COL_TOP_WIND +
           w_tas[...,  None] * COL_BL_TAS   +
           w_rh[...,   None] * COL_BR_RH)

    rgb[~chg] = 1.0
    alpha = np.zeros_like(sum_mag, dtype=float)
    alpha[chg] = 1.0
    return rgb, alpha

def add_triangle_legend(ax, x_anchor=-0.20, y_anchor=0.06):
    leg_ax = inset_axes(
        ax,
        width="15%", height="26%",
        loc="lower right",
        bbox_to_anchor=(x_anchor, y_anchor, 1.0, 1.0),
        bbox_transform=ax.transAxes,
        borderpad=0.0
    )
    leg_ax.set_axis_off()
    leg_ax.set_facecolor("none")

    N = 240
    V_top = np.array([N / 2.0, 0.0])
    V_bl  = np.array([0.0, N - 1.0])
    V_br  = np.array([N - 1.0, N - 1.0])

    C_top = COL_TOP_WIND
    C_bl  = COL_BL_TAS
    C_br  = COL_BR_RH

    Y, X = np.mgrid[0:N, 0:N]
    P = np.stack([X, Y], axis=-1).astype(float)

    def barycentric(P, A, B, C):
        v0 = B - A
        v1 = C - A
        v2 = P - A
        den = v0[0] * v1[1] - v1[0] * v0[1]
        wB = (v2[..., 0] * v1[1] - v1[0] * v2[..., 1]) / den
        wC = (v0[0] * v2[..., 1] - v2[..., 0] * v0[1]) / den
        wA = 1.0 - wB - wC
        return wA, wB, wC

    wA, wB, wC = barycentric(P, V_top, V_bl, V_br)
    inside = (wA >= 0) & (wB >= 0) & (wC >= 0)

    rgba = np.zeros((N, N, 4), dtype=float)
    rgba[..., 3] = 0.0
    rgba[inside, 3] = 1.0
    rgba[inside, :3] = (wA[inside, None] * C_top +
                        wB[inside, None] * C_bl  +
                        wC[inside, None] * C_br)

    leg_ax.imshow(rgba, interpolation="nearest", origin="upper")
    leg_ax.set_xlim(5, N - 6)
    leg_ax.set_ylim(N - 1, 0)

    leg_ax.text(0.50, 1.02, "Wind", transform=leg_ax.transAxes, ha="center", va="bottom", fontsize=12)
    leg_ax.text(-0.02, -0.02, "Tas", transform=leg_ax.transAxes, ha="left",  va="top",    fontsize=12)
    leg_ax.text(1.02, -0.02, "Rh",   transform=leg_ax.transAxes, ha="right", va="top",    fontsize=12)

def plot_one(ax, file_path, panel_label, add_legend=False):
    ds = xr.open_dataset(file_path)

    ori   = read_da(ds, V_ORI)
    windA = read_da(ds, V_WIND)
    rhA   = read_da(ds, V_RH)
    tasA  = read_da(ds, V_TAS)

    rgb, alpha = make_rgb(ori, windA, rhA, tasA)

    lat = ori["lat"].values
    lon = ori["lon"].values
    extent = [float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())]

    ax.set_extent([0, 360, -60, 60], crs=ccrs.PlateCarree())
    ax.set_facecolor(OCEAN_COLOR)
    ax.add_feature(cfeature.LAND, facecolor="white", edgecolor="none", zorder=1)

    ax.imshow(
        rgb,
        origin="lower",
        extent=extent,
        transform=ccrs.PlateCarree(),
        interpolation="nearest",
        alpha=alpha,
        zorder=2
    )

    ax.coastlines(resolution="110m", linewidth=0.9, zorder=10)

    tick_pos = [-30, 30, 90, 150, -150, -90, -30]
    tick_lab = [r"30°W", r"30°E", r"90°E", r"150°E", r"150°W", r"90°W", r"30°W"]
    ax.set_xticks(tick_pos, crs=ccrs.PlateCarree())
    ax.set_xticklabels(tick_lab)

    ax.set_yticks([-60, -30, 0, 30, 60], crs=ccrs.PlateCarree())
    ax.set_yticklabels([r"60°S", r"30°S", r"0°", r"30°N", r"60°N"])

    ax.plot([1.0, 1.0], [0.0, -0.01],
            transform=ax.transAxes, color="k", lw=0.8,
            clip_on=False, zorder=20)
    ax.text(1.02, -0.022, r"30°W",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=mpl.rcParams["font.size"],
            clip_on=False, zorder=20)

    add_panel_label(ax, panel_label, fs=24)

    if add_legend:
        add_triangle_legend(ax, x_anchor=-0.20, y_anchor=0.06)

    ds.close()

fig = plt.figure(figsize=(12.5, 10.8))

ax_a = fig.add_subplot(2, 1, 1, projection=proj)

plot_one(ax_a, FILE_A, "", add_legend=True)

plt.subplots_adjust(left=0.05, right=0.98, bottom=0.08, top=0.98, hspace=-0.05)
plt.savefig(r"D:\code\comfort-area\fig\v7\figS5.png", dpi=800, bbox_inches="tight")
