# -*- coding: utf-8 -*-
import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
from matplotlib.colors import ListedColormap
import matplotlib as mpl
from scipy.stats import linregress

import cartopy.io.shapereader as shpreader
from cartopy.feature import NaturalEarthFeature

from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

from matplotlib.patches import Wedge
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 14

# ---------- ERA5 ----------
IN_NC   = r"D:\code\comfort-area\data\v5\ERA5\apt.shushiDays.mon.mean.198001-202012.nc"
VAR     = "comfortDays"
FILL    = 32767
THRESH  = 97

T_START_INDEX = 11
T_END_EXCL    = 491
CLIM_Y0 = 1981
CLIM_Y1 = 2020
N_CLIM_YEARS = CLIM_Y1 - CLIM_Y0 + 1

# ----------CMIP6 ----------
CMIP6_DIR = r"D:\code\comfort-area\data\v5\CMIP6"
CMIP6_NC  = os.path.join(CMIP6_DIR, "mild_area_1981_2020_MME_historical_r1i1p1f1_annual.nc")
CMIP6_VAR = "mild_area"
CMIP6_FILL = 32767
CMIP6_NH_TXT = os.path.join(CMIP6_DIR, "mild_area_1981_2020_MME_historical_r1i1p1f1_annual_nh.txt")
CMIP6_SH_TXT = os.path.join(CMIP6_DIR, "mild_area_1981_2020_MME_historical_r1i1p1f1_annual_sh.txt")
CMIP6_Y0, CMIP6_Y1 = 1981, 2020

# ---------- CESM2----------
CESM2_DIR = r"D:\code\comfort-area\data\v5\CESM2"
CESM2_NC  = os.path.join(CESM2_DIR, "mild_area_1981_2020_MME_BHIST_annual_2.nc")
CESM2_VAR = "mild_area"
CESM2_FILL = 32767
CESM2_NH_TXT = os.path.join(CESM2_DIR, "mild_area_1981_2020_MME_BHIST_annual_nh_2.txt")
CESM2_SH_TXT = os.path.join(CESM2_DIR, "mild_area_1981_2020_MME_BHIST_annual_sh_2.txt")
CESM2_Y0, CESM2_Y1 = 1981, 2020

LAT_MIN, LAT_MAX = -60.0, 60.0
CENTER_LON = 150.0
OCEAN_COLOR = "#e0f2fc"

BASE_ERA5 = r"D:\code\comfort-area\data\ERA5"
AREA_FILE = os.path.join(BASE_ERA5, "aera_25x25_nh_0_70.nc")
LANDSEA_FILE_NH = os.path.join(BASE_ERA5, "landsea_NH.nc")
LANDSEA_FILE_SH = os.path.join(BASE_ERA5, "landsea_SH.nc")

LAT_EDGE = 67.5
AREA_NH = 9933.0
AREA_SH = 4970.0
AREA_GL = AREA_NH + AREA_SH

TS_XPAD_YEARS = 2
TS_XTICK0 = 1980
TS_XTICK_STEP = 10

TS1_YLIM_L = (14, 26)
TS1_YLIM_R = (13.6, 17)
TS1_YTICKS_L = [14,18,22,26]
TS1_YTICKS_R = [14,15,16,17]
TS1_YTICKLABELS_L = ['14','18','22','26']
TS1_YTICKLABELS_R = ['14','15','16','17']
TS1_YLABEL_L = "Percentage (NH) [%]"
TS1_YLABEL_R = "Percentage (SH) [%]"

TS2_YLIM_L = (13, 25)
TS2_YLIM_R = (14, 17.5)
TS2_YTICKS_L = [13,17,21,25]
TS2_YTICKS_R = [14,15,16,17]
TS2_YTICKLABELS_L = ['13','17','21','25']
TS2_YTICKLABELS_R = ['14','15','16','17']
TS2_YLABEL_L = "Percentage (NH) [%]"
TS2_YLABEL_R = "Percentage (SH) [%]"

TS3_YLIM_L = (12, 21.8)
TS3_YLIM_R = (14, 17)
TS3_YTICKS_L = [12,15,18,21]
TS3_YTICKS_R = [14,15,16,17]
TS3_YTICKLABELS_L = ['12','15','18','21']
TS3_YTICKLABELS_R = ['14','15','16','17']
TS3_YLABEL_L = "Percentage (NH) [%]"
TS3_YLABEL_R = "Percentage (SH) [%]"

# =========================
# 工具函数
# =========================

ADMIN1_50M = NaturalEarthFeature(
    category="cultural",
    name="admin_1_states_provinces_lines",
    scale="50m",
    facecolor="none",
    edgecolor="0.35"
)

_CHINA_ADMIN1_GEOMS = None

def get_china_admin1_geoms(resolution="10m"):

    global _CHINA_ADMIN1_GEOMS
    if _CHINA_ADMIN1_GEOMS is not None:
        return _CHINA_ADMIN1_GEOMS

    shp = shpreader.natural_earth(
        resolution=resolution,
        category="cultural",
        name="admin_1_states_provinces_lines"
    )
    geoms = []
    for rec in shpreader.Reader(shp).records():
        attrs = rec.attributes

        if (attrs.get("adm0_a3") == "CHN") or (attrs.get("admin") == "China") or (attrs.get("geonunit") == "China"):
            geoms.append(rec.geometry)

    _CHINA_ADMIN1_GEOMS = geoms
    return geoms

def add_panel_label(ax, label, x=-0.13, y=1.17, fs=19):
    ax.text(
        x, y, label,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=fs, fontweight="bold"
    )

def read_lsm(landsea_path):
    ds = xr.open_dataset(landsea_path)
    if "lsm" in ds.data_vars:
        lsm = ds["lsm"]
    elif "LSMASK" in ds.data_vars:
        lsm = ds["LSMASK"]
    else:
        raise ValueError(
            f"{landsea_path} 里既没有 'lsm' 也没有 'LSMASK'，实际变量有：{list(ds.data_vars)}"
        )
    ds.close()
    return lsm

def mask_ocean_like(da, lsm_da):
    return da.where(lsm_da > 0.5)

def _band_sel_strict(da, lat_start, lat_end):
    lat_vals = da["lat"].values

    if lat_start == 0.0 and lat_end > 0.0:
        if lat_vals[0] > lat_vals[-1]:
            da = da.sortby("lat")
        return da.sel(lat=slice(0.0, lat_end))

    if lat_start == 0.0 and lat_end < 0.0:
        if lat_vals[0] < lat_vals[-1]:
            da = da.sortby("lat", ascending=False)
        return da.sel(lat=slice(0.0, lat_end))

    return da.sel(lat=slice(lat_start, lat_end))

def linreg_1d(years, y):
    years = years.astype(float)
    y = y.astype(float)
    m = np.isfinite(years) & np.isfinite(y)
    if m.sum() < 3:
        return {"slope": np.nan, "intercept": np.nan, "pvalue": np.nan,
                "y_fit": np.full_like(years, np.nan, dtype=float)}
    res = linregress(years[m], y[m])
    y_fit = res.slope * years + res.intercept
    return {"slope": res.slope, "intercept": res.intercept, "pvalue": res.pvalue, "y_fit": y_fit}

def load_area(area_file):
    ds_area = xr.open_dataset(area_file)
    area = ds_area["area"]
    dims = list(area.dims)
    area = area.rename({dims[0]: "lat", dims[1]: "lon"})
    ds_area.close()
    return area

def compute_hemi_area_percent_global(c_ym, hemi, area_da, thresh=THRESH, lat_edge=LAT_EDGE):

    if hemi.upper() == "NH":
        lsm = read_lsm(LANDSEA_FILE_NH)
        c_sum = c_ym.sum("month", skipna=True)
        c_sum = _band_sel_strict(c_sum, 0.0, lat_edge)
        c_sum = mask_ocean_like(c_sum, lsm)
        area_h = area_da
    elif hemi.upper() == "SH":
        lsm = read_lsm(LANDSEA_FILE_SH)
        c_sum = c_ym.sum("month", skipna=True)
        c_sum = _band_sel_strict(c_sum, 0.0, -lat_edge)
        c_sum = mask_ocean_like(c_sum, lsm)
        area_h = area_da
    else:
        raise ValueError("hemi must be 'NH' or 'SH'")

    area_3d = area_h.broadcast_like(c_sum)
    hit = (c_sum >= thresh) & np.isfinite(c_sum)
    area_year = area_3d.where(hit).sum(dim=("lat", "lon"), skipna=True)
    pct = (area_year / AREA_GL) * 100.0
    return pct

def add_double_ring_pie(ax_map, p_nh, p_sh):
    PIE_SIZES  = [66.65, 33.35]
    PIE_COLORS = ['#f393be', '#00a4d2']

    x0, y0, w_box, h_box = (0.43, 0.00, 0.6, 0.6)

    ax_pie = inset_axes(
        ax_map,
        width="100%", height="100%",
        bbox_to_anchor=(x0, y0, w_box, h_box),
        bbox_transform=ax_map.transAxes,
        borderpad=0.0
    )
    ax_pie.set_aspect('equal')

    wedges, texts, autotexts = ax_pie.pie(
        PIE_SIZES,
        colors=PIE_COLORS,
        startangle=90,
        autopct='%1.1f%%',
        counterclock=False,
        pctdistance=0.75
    )

    for j, t in enumerate(autotexts):
        t.set_fontsize(11)
        t.set_text("NH" if j == 0 else "SH")

    r_label = 0.6
    dx_nh, dy_nh = -0.1, 0.00
    dx_sh, dy_sh = 0.00, -0.3

    for j, t in enumerate(autotexts):
        w = wedges[j]
        theta = np.deg2rad(0.5 * (w.theta1 + w.theta2))
        x = r_label * np.cos(theta)
        y = r_label * np.sin(theta)

        if j == 0:
            x += dx_nh
            y += dy_nh
        else:
            x += dx_sh
            y += dy_sh

        t.set_position((x, y))
        t.set_ha("center")
        t.set_va("center")

    nh_wedge = wedges[0]
    boundary = nh_wedge.theta2

    gap_deg = 15.0
    offset = 0.10

    angle_nh_inner = 360.0 * p_nh / 100.0
    theta2_nh_inner = boundary - gap_deg / 2.0
    theta1_nh_inner = theta2_nh_inner - angle_nh_inner
    theta_mid_nh = np.deg2rad(0.5 * (theta1_nh_inner + theta2_nh_inner))
    dx_nh2 = offset * np.cos(theta_mid_nh)
    dy_nh2 = offset * np.sin(theta_mid_nh)

    inner_nh = Wedge(
        center=(dx_nh2, dy_nh2),
        r=1.0,
        theta1=theta1_nh_inner,
        theta2=theta2_nh_inner,
        facecolor='#ee3b73',
        edgecolor='k', linewidth=0.6, alpha=0.9
    )
    ax_pie.add_patch(inner_nh)
    ax_pie.text(
        dx_nh2 + 0.70*np.cos(theta_mid_nh),
        dy_nh2 + 0.70*np.sin(theta_mid_nh),
        f"{p_nh:.1f}",
        ha='center', va='center',
        fontsize=12, fontweight='bold'
    )

    angle_sh_inner = 360.0 * p_sh / 100.0
    theta1_sh_inner = boundary + gap_deg / 2.0
    theta2_sh_inner = theta1_sh_inner + angle_sh_inner
    theta_mid_sh = np.deg2rad(0.5 * (theta1_sh_inner + theta2_sh_inner))
    dx_sh2 = offset * np.cos(theta_mid_sh)
    dy_sh2 = offset * np.sin(theta_mid_sh)

    inner_sh = Wedge(
        center=(dx_sh2, dy_sh2),
        r=1.0,
        theta1=theta1_sh_inner,
        theta2=theta2_sh_inner,
        facecolor='#0078bf',
        edgecolor='k', linewidth=0.6, alpha=0.9
    )
    ax_pie.add_patch(inner_sh)
    ax_pie.text(
        dx_sh2 + 0.70*np.cos(theta_mid_sh),
        dy_sh2 + 0.70*np.sin(theta_mid_sh),
        f"{p_sh:.1f}",
        ha='center', va='center',
        fontsize=12, fontweight='bold'
    )

    ax_pie.set_xticks([])
    ax_pie.set_yticks([])
    for sp in ax_pie.spines.values():
        sp.set_visible(False)

def draw_timeseries_like_learning(
    ax, years, y_nh, y_sh, title="",
    y1lim=None, y2lim=None,
    y1ticks=None, y2ticks=None,
    y1ticklabels=None, y2ticklabels=None,
    y1label="Percentage (NH)", y2label="Percentage (SH)"
):
    COLOR_NH = '#ee3b73'
    COLOR_SH = '#0078bf'

    ax2 = ax.twinx()

    line_nh = ax.plot(years, y_nh, color=COLOR_NH, linewidth=1.5, label="NH", zorder=3)
    line_sh = ax2.plot(years, y_sh, color=COLOR_SH, linewidth=1.5, label="SH", zorder=3)

    std_nh = np.nanstd(y_nh)
    std_sh = np.nanstd(y_sh)
    ax.fill_between(years, y_nh - std_nh, y_nh + std_nh, color=COLOR_NH, alpha=0.11, linewidth=0, zorder=1)
    ax2.fill_between(years, y_sh - std_sh, y_sh + std_sh, color=COLOR_SH, alpha=0.11, linewidth=0, zorder=1)

    tr_nh = linreg_1d(years, y_nh)
    tr_sh = linreg_1d(years, y_sh)

    yfit_nh = tr_nh["y_fit"].copy()
    yfit_sh = tr_sh["y_fit"].copy()

    m_nh = np.isfinite(y_nh)
    m_sh = np.isfinite(y_sh)
    yfit_nh[~m_nh] = np.nan
    yfit_sh[~m_sh] = np.nan

    ax.plot(years, yfit_nh, color=COLOR_NH, linewidth=1.8, linestyle='--', zorder=4)
    ax2.plot(years, yfit_sh, color=COLOR_SH, linewidth=1.8, linestyle='--', zorder=4)

    if y1lim is not None:
        ax.set_ylim(y1lim)
    if y2lim is not None:
        ax2.set_ylim(y2lim)

    if y1ticks is not None:
        ax.set_yticks(y1ticks)
    if y2ticks is not None:
        ax2.set_yticks(y2ticks)

    if y1ticklabels is not None:
        ax.set_yticklabels(y1ticklabels)
    if y2ticklabels is not None:
        ax2.set_yticklabels(y2ticklabels)

    ax.set_ylabel(y1label, color=COLOR_NH, size=13)
    ax2.set_ylabel(y2label, color=COLOR_SH, size=13)
    ax.tick_params(axis='y', colors=COLOR_NH,labelsize=13)
    ax2.tick_params(axis='y', colors=COLOR_SH,labelsize=13)

    xmin = int(np.nanmin(years)) - TS_XPAD_YEARS
    xmax = int(np.nanmax(years)) + TS_XPAD_YEARS
    ax.set_xlim(xmin, xmax)

    xticks = np.arange(TS_XTICK0, xmax + 1, TS_XTICK_STEP)
    xticks = xticks[(xticks >= xmin) & (xticks <= xmax)]
    ax.set_xticks(xticks)
    ax.tick_params(axis='x', labelsize=13)

    slope_nh_100yr = tr_nh["slope"] * 100.0
    slope_sh_100yr = tr_sh["slope"] * 100.0
    txt = (
        f"NH: {slope_nh_100yr:5.2f}%"
        + ("/C*" if np.isfinite(tr_nh["pvalue"]) and tr_nh["pvalue"] <= 0.1 else "/C")
        + "     "
        f"SH: {slope_sh_100yr:5.2f}%"
        + ("/C*" if np.isfinite(tr_sh["pvalue"]) and tr_sh["pvalue"] <= 0.1 else "/C")
    )
    ax.text(0.044, 0.02, txt, transform=ax.transAxes, ha="left", va="bottom", fontsize=12)

    lines = line_nh + line_sh
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, frameon=False, fontsize=12, loc="upper right", ncol=2, bbox_to_anchor=(0.99, 1.05))

    if title:
        ax.set_title(title, loc="left", fontsize=14)

def load_txt_series(path):
    arr = np.loadtxt(path)
    arr = np.atleast_1d(arr).astype(float)
    if arr.ndim > 1:
        arr = arr[:, 0]
    return arr

def setup_map_axis(ax):
    ax.set_extent([-180, 180, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="white", edgecolor="none", zorder=0)

    xticks = [30, 90, 150, -150, -90, -30]
    yticks = np.arange(-60, 61, 30)

    ax.set_xticks(xticks, crs=ccrs.PlateCarree())
    ax.set_yticks(yticks, crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
    ax.yaxis.set_major_formatter(LatitudeFormatter())

    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        length=4,
        width=0.8,
        labelsize=13,
        pad=3
    )
    ax.tick_params(axis="x", top=False, labeltop=False, bottom=True, labelbottom=True)
    ax.tick_params(axis="y", right=False, labelright=False, left=True, labelleft=True)

    ax.text(0.955, -0.06, "30°W", transform=ax.transAxes, va="top", fontsize=13)
    ax.plot([1.0, 1.0], [0.0, -0.03], transform=ax.transAxes, color="black", linewidth=0.8, clip_on=False)

def plot_green_mask(ax, mask2d, lats, lons, green_cmap):
    mask_plot = xr.DataArray(mask2d, coords={"lat": lats, "lon": lons}, dims=("lat", "lon")).sel(
        lat=slice(LAT_MIN, LAT_MAX)
    )
    mask_cyc, lon_cyc = add_cyclic_point(mask_plot.values, coord=mask_plot["lon"].values)

    ax.pcolormesh(
        lon_cyc, mask_plot["lat"].values, mask_cyc,
        transform=ccrs.PlateCarree(),
        cmap=green_cmap, vmin=0, vmax=1,
        zorder=1
    )
    ax.add_feature(cfeature.OCEAN, facecolor=OCEAN_COLOR, edgecolor="none", zorder=2)
    ax.coastlines(linewidth=0.6, zorder=3)

def _norm_lon_0_360(da, lon_name="lon"):
    lon = (da[lon_name] % 360)
    return da.assign_coords({lon_name: lon}).sortby(lon_name)

def _weighted_corr(x, y, w):
    w = w / np.sum(w)
    mx = np.sum(w * x)
    my = np.sum(w * y)
    cov = np.sum(w * (x - mx) * (y - my))
    vx  = np.sum(w * (x - mx) ** 2)
    vy  = np.sum(w * (y - my) ** 2)
    if vx <= 0 or vy <= 0:
        return np.nan
    return cov / np.sqrt(vx * vy)

def pattern_corr(da1, da2, lat_min=LAT_MIN, lat_max=LAT_MAX, weighted=True):

    da1 = _norm_lon_0_360(da1)
    da2 = _norm_lon_0_360(da2)

    da1 = da1.sel(lat=slice(lat_min, lat_max))
    da2 = da2.sel(lat=slice(lat_min, lat_max))

    da1a, da2a = xr.align(da1, da2, join="inner")

    m = np.isfinite(da1a.values) & np.isfinite(da2a.values)
    if m.sum() < 10:
        return np.nan

    x = da1a.values[m].astype(float)
    y = da2a.values[m].astype(float)

    if not weighted:
        return np.corrcoef(x, y)[0, 1]

    lat = da1a["lat"].values
    w_lat = np.cos(np.deg2rad(lat))
    w2d = np.broadcast_to(w_lat[:, None], da1a.shape)
    w = w2d[m].astype(float)

    return _weighted_corr(x, y, w)


green_cmap = ListedColormap(["#7bca73"])
proj = ccrs.PlateCarree(central_longitude=CENTER_LON)

fig = plt.figure(figsize=(13.2, 16.2))
gs = fig.add_gridspec(nrows=3, ncols=2, wspace=0.18, hspace=0.28)

area_da = load_area(AREA_FILE)

ax_map1 = fig.add_subplot(gs[0, 0], projection=proj)
ax_ts1  = fig.add_subplot(gs[0, 1])

setup_map_axis(ax_map1)

ds = xr.open_dataset(IN_NC)
c  = ds[VAR]
if FILL is not None:
    c = c.where(c != FILL)

c2 = c.transpose("year", "month", "lat", "lon")
c_flat = c2.stack(t=("year", "month")).transpose("t", "lat", "lon")
c_sel = c_flat.isel(t=slice(T_START_INDEX, T_END_EXCL)).reset_index("t", drop=True)

nlat, nlon = c_sel.sizes["lat"], c_sel.sizes["lon"]
arr4 = c_sel.data.reshape(N_CLIM_YEARS, 12, nlat, nlon)

c_ym = xr.DataArray(
    arr4,
    dims=("year", "month", "lat", "lon"),
    coords={
        "year": np.arange(CLIM_Y0, CLIM_Y1 + 1),
        "month": np.arange(1, 13),
        "lat": c_sel["lat"],
        "lon": c_sel["lon"],
    },
    name=VAR
)

c_latlon = c_ym.mean("year", skipna=True).sum("month", skipna=True)
mask_era5 = xr.where(c_latlon >= THRESH, 1.0, np.nan)
plot_green_mask(ax_map1, mask_era5.values, mask_era5["lat"].values, mask_era5["lon"].values, green_cmap)

pct_nh = compute_hemi_area_percent_global(c_ym, "NH", area_da, thresh=THRESH, lat_edge=LAT_EDGE)
pct_sh = compute_hemi_area_percent_global(c_ym, "SH", area_da, thresh=THRESH, lat_edge=LAT_EDGE)

years1 = c_ym["year"].values.astype(int)
y1_nh = pct_nh.values.astype(float)
y1_sh = pct_sh.values.astype(float)

draw_timeseries_like_learning(
    ax_ts1, years1, y1_nh, y1_sh, title="",
    y1lim=TS1_YLIM_L, y2lim=TS1_YLIM_R,
    y1ticks=TS1_YTICKS_L, y2ticks=TS1_YTICKS_R,
    y1ticklabels=TS1_YTICKLABELS_L, y2ticklabels=TS1_YTICKLABELS_R,
    y1label=TS1_YLABEL_L, y2label=TS1_YLABEL_R
)

print(y1_nh)
print(y1_sh)

pie_p_nh = float(np.nanmean(y1_nh))
pie_p_sh = float(np.nanmean(y1_sh))
add_double_ring_pie(ax_map1, pie_p_nh, pie_p_sh)

ax_map1.set_title("ERA5", fontsize=13, loc="left")
add_panel_label(ax_map1, "a")

ds.close()

ax_map2 = fig.add_subplot(gs[1, 0], projection=proj)
ax_ts2  = fig.add_subplot(gs[1, 1])

setup_map_axis(ax_map2)

ds2 = xr.open_dataset(CMIP6_NC)
m2 = ds2[CMIP6_VAR]
m2 = m2.where(m2 != CMIP6_FILL)
mask_cmip6 = xr.where(m2 == 1, 1.0, np.nan)

plot_green_mask(ax_map2, mask_cmip6.values, mask_cmip6["lat"].values, mask_cmip6["lon"].values, green_cmap)

ax_map2.set_title("CMIP6", fontsize=13, loc="left")
add_panel_label(ax_map2, "b")

cmip6_nh = load_txt_series(CMIP6_NH_TXT)
cmip6_sh = load_txt_series(CMIP6_SH_TXT)
years2 = np.arange(CMIP6_Y0, CMIP6_Y1 + 1, dtype=int)

n2 = min(len(years2), len(cmip6_nh), len(cmip6_sh))
years2 = years2[:n2]
cmip6_nh = cmip6_nh[:n2].astype(float)
cmip6_sh = cmip6_sh[:n2].astype(float)

draw_timeseries_like_learning(
    ax_ts2, years2, cmip6_nh, cmip6_sh, title="",
    y1lim=TS2_YLIM_L, y2lim=TS2_YLIM_R,
    y1ticks=TS2_YTICKS_L, y2ticks=TS2_YTICKS_R,
    y1ticklabels=TS2_YTICKLABELS_L, y2ticklabels=TS2_YTICKLABELS_R,
    y1label=TS2_YLABEL_L, y2label=TS2_YLABEL_R
)

p2_nh = float(np.nanmean(cmip6_nh))
p2_sh = float(np.nanmean(cmip6_sh))
add_double_ring_pie(ax_map2, p2_nh, p2_sh)

ds2.close()

ax_map3 = fig.add_subplot(gs[2, 0], projection=proj)
ax_ts3  = fig.add_subplot(gs[2, 1])

setup_map_axis(ax_map3)

ds3 = xr.open_dataset(CESM2_NC)
m3 = ds3[CESM2_VAR]
m3 = m3.where(m3 != CESM2_FILL)
mask_cesm2 = xr.where(m3 == 1, 1.0, np.nan)

plot_green_mask(ax_map3, mask_cesm2.values, mask_cesm2["lat"].values, mask_cesm2["lon"].values, green_cmap)


ax_map3.set_title("CESM2-LENS2", fontsize=13, loc="left")
add_panel_label(ax_map3, "c")

cesm2_nh = load_txt_series(CESM2_NH_TXT)
cesm2_sh = load_txt_series(CESM2_SH_TXT)
years3_raw = np.arange(CESM2_Y0, CESM2_Y1 + 1, dtype=int)

n3 = min(len(years3_raw), len(cesm2_nh), len(cesm2_sh))
years3_raw = years3_raw[:n3]
cesm2_nh = cesm2_nh[:n3].astype(float)
cesm2_sh = cesm2_sh[:n3].astype(float)

years3 = np.arange(1981, 2021, dtype=int)  # 1981–2020
y3_nh = np.full(years3.shape, np.nan, dtype=float)
y3_sh = np.full(years3.shape, np.nan, dtype=float)
y3_nh[:n3] = cesm2_nh
y3_sh[:n3] = cesm2_sh

draw_timeseries_like_learning(
    ax_ts3, years3, y3_nh, y3_sh, title="",
    y1lim=TS3_YLIM_L, y2lim=TS3_YLIM_R,
    y1ticks=TS3_YTICKS_L, y2ticks=TS3_YTICKS_R,
    y1ticklabels=TS3_YTICKLABELS_L, y2ticklabels=TS3_YTICKLABELS_R,
    y1label=TS3_YLABEL_L, y2label=TS3_YLABEL_R
)

p3_nh = float(np.nanmean(cesm2_nh))
p3_sh = float(np.nanmean(cesm2_sh))
add_double_ring_pie(ax_map3, p3_nh, p3_sh)

ds3.close()

add_panel_label(ax_ts1, "d")

add_panel_label(ax_ts2, "e")

add_panel_label(ax_ts3, "f")


plt.tight_layout()

def match_height(ax_left, ax_right):
    fig = ax_left.figure
    fig.canvas.draw()
    posL = ax_left.get_position()
    posR = ax_right.get_position()
    ax_right.set_position([posR.x0, posL.y0, posR.width, posL.height])

match_height(ax_map1, ax_ts1)
match_height(ax_map2, ax_ts2)
match_height(ax_map3, ax_ts3)

GAP12 = 0.035
GAP23 = 0.045
TOP    = 0.98
BOTTOM = 0.05

fig.canvas.draw()

pos1 = ax_map1.get_position()
pos2 = ax_map2.get_position()
pos3 = ax_map3.get_position()

H = np.mean([pos1.height, pos2.height, pos3.height])

y0_row3 = BOTTOM
y0_row2 = y0_row3 + H + GAP23
y0_row1 = y0_row2 + H + GAP12

y_top = y0_row1 + H
if y_top > TOP:
    shift = y_top - TOP
    y0_row1 -= shift
    y0_row2 -= shift
    y0_row3 -= shift

def set_row(ax_left, ax_right, y0, H):
    fig.canvas.draw()
    pL = ax_left.get_position()
    pR = ax_right.get_position()
    ax_left.set_position([pL.x0, y0, pL.width, H])
    ax_right.set_position([pR.x0, y0, pR.width, H])

set_row(ax_map1, ax_ts1, y0_row1, H)
set_row(ax_map2, ax_ts2, y0_row2, H)
set_row(ax_map3, ax_ts3, y0_row3, H)

era5_bin  = xr.where(np.isfinite(mask_era5), 1.0, 0.0)
cmip6_bin = xr.where(np.isfinite(mask_cmip6), 1.0, 0.0)
cesm2_bin = xr.where(np.isfinite(mask_cesm2), 1.0, 0.0)

r_e_c = pattern_corr(era5_bin,  cmip6_bin, weighted=True)
r_e_s = pattern_corr(era5_bin,  cesm2_bin, weighted=True)
r_c_s = pattern_corr(cmip6_bin, cesm2_bin, weighted=True)


out_dir = r"D:\code\comfort-area\fig\v7"
os.makedirs(out_dir, exist_ok=True)
out_file = os.path.join(out_dir, "fig1.pdf")
plt.savefig(out_file, dpi=600, bbox_inches="tight")
print(f"图片已保存：{out_file}")