# -*- coding: utf-8 -*-
import os
import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
from matplotlib.colors import ListedColormap
from matplotlib.patches import Wedge
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import linregress
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 14

AREA_NH = 9933.0
AREA_SH = 4970.0
AREA_GL = AREA_NH + AREA_SH

NH_YLIM_WBGT     = (15, 24)
SH_YLIM_WBGT     = (13, 16)
NH_TICKS_WBGT    = [15, 18, 21, 24]
SH_TICKS_WBGT    = [13, 14, 15, 16]

NH_YLIM_UTCI     = (8, 17)
SH_YLIM_UTCI     = (11, 17)
NH_TICKS_UTCI    = [8, 11, 14, 17]
SH_TICKS_UTCI    = [11, 13, 15, 17]

NH_YLIM_NET      = (12, 24)
SH_YLIM_NET      = (14, 20)
NH_TICKS_NET     = [12, 16, 20, 24]
SH_TICKS_NET     = [14, 16, 18, 20]

LAT_MIN, LAT_MAX = -60.0, 60.0
CENTER_LON = 150.0
OCEAN_COLOR = "#e0f2fc"

BASE_ERA5   = r"D:\code\comfort-area\data\v5\ERA5"
BASE_ERA5_1 = r"D:\code\comfort-area\data\ERA5"
LANDSEA_FILE_NH = os.path.join(BASE_ERA5_1, "landsea_NH.nc")
LANDSEA_FILE_SH = os.path.join(BASE_ERA5_1, "landsea_SH.nc")

APT_MASK_NC = os.path.join(BASE_ERA5, "mild_area_1981_2020_APT_annual.nc")
APT_NH_TXT  = os.path.join(BASE_ERA5, "mild_area_1981_2020_APT_annual_nh.txt")
APT_SH_TXT  = os.path.join(BASE_ERA5, "mild_area_1981_2020_APT_annual_sh.txt")

WBGT_MASK_NC = os.path.join(BASE_ERA5, "mild_area_1981_2020_WBGT_annual.nc")
WBGT_NH_TXT  = os.path.join(BASE_ERA5, "mild_area_1981_2020_WBGT_annual_nh.txt")
WBGT_SH_TXT  = os.path.join(BASE_ERA5, "mild_area_1981_2020_WBGT_annual_sh.txt")

UTCI_MASK_NC = os.path.join(BASE_ERA5, "mild_area_1981_2020_UTCI_annual.nc")
UTCI_NH_TXT  = os.path.join(BASE_ERA5, "mild_area_1981_2020_UTCI_annual_nh.txt")
UTCI_SH_TXT  = os.path.join(BASE_ERA5, "mild_area_1981_2020_UTCI_annual_sh.txt")

NET_MASK_NC = os.path.join(BASE_ERA5, "mild_area_1981_2020_NET_annual.nc")
NET_NH_TXT  = os.path.join(BASE_ERA5, "mild_area_1981_2020_NET_annual_nh.txt")
NET_SH_TXT  = os.path.join(BASE_ERA5, "mild_area_1981_2020_NET_annual_sh.txt")

MASK_VAR_NAME = "mild_area"
MASK_FILL = 32767

def _rename_latlon(da):
    rename_map = {}
    for d in da.dims:
        dl = d.lower()
        if dl in ["latitude", "y"]:
            rename_map[d] = "lat"
        if dl in ["longitude", "x"]:
            rename_map[d] = "lon"
    if rename_map:
        da = da.rename(rename_map)
    return da

def _read_lsm(path):
    ds = xr.open_dataset(path)
    if "lsm" in ds.data_vars:
        lsm = ds["lsm"]
    elif "LSMASK" in ds.data_vars:
        lsm = ds["LSMASK"]
    else:
        raise ValueError(f"{path} 中没有 lsm/LSMASK，实际变量：{list(ds.data_vars)}")
    ds.close()
    return _rename_latlon(lsm)

def mask_ocean_by_landsea_nhsh(da, landsea_nh=LANDSEA_FILE_NH, landsea_sh=LANDSEA_FILE_SH):
    """仅用于：场相关前 mask 海洋（海洋->NaN）。"""
    da = _rename_latlon(da)
    lsm_nh = _read_lsm(landsea_nh).interp(lat=da["lat"], lon=da["lon"], method="nearest")
    lsm_sh = _read_lsm(landsea_sh).interp(lat=da["lat"], lon=da["lon"], method="nearest")
    landmask = xr.where(da["lat"] >= 0, lsm_nh, lsm_sh)
    return da.where(landmask > 0.5)

def read_comfort_mask_nc(nc_path, var_name=MASK_VAR_NAME, fill_value=MASK_FILL):
    ds = xr.open_dataset(nc_path)
    if var_name not in ds.data_vars:
        raise ValueError(f"{nc_path} 中找不到变量 {var_name}，实际变量：{list(ds.data_vars)}")
    da = ds[var_name]
    ds.close()

    da = _rename_latlon(da)

    da01 = xr.where(da == 1, 1.0, 0.0)
    return da01

def read_percent_series_txt(txt_path, years):
    arr = np.loadtxt(txt_path, dtype=float).squeeze()
    if arr.ndim > 1:
        arr = arr[:, 0]
    if arr.size != len(years):
        raise ValueError(f"{txt_path} 行数={arr.size} 与 years={len(years)} 不一致")

    vmax = np.nanmax(arr)
    if np.isfinite(vmax) and vmax <= 1.5:
        arr = arr * 100.0
    return xr.DataArray(arr, coords={"year": years}, dims=("year",))

def linreg_1d(year_da, data_da):
    years = year_da.values.astype(float)
    data  = data_da.values.astype(float)
    mask = np.isfinite(years) & np.isfinite(data)
    if mask.sum() < 3:
        return {"slope": np.nan, "intercept": np.nan, "pvalue": np.nan,
                "y_fit": xr.full_like(year_da, np.nan, dtype=float)}
    res = linregress(years[mask], data[mask])
    y_fit = res.slope * years + res.intercept
    return {"slope": res.slope, "intercept": res.intercept, "pvalue": res.pvalue,
            "y_fit": xr.DataArray(y_fit, coords={"year": year_da["year"]}, dims=("year",))}

def corrcoef_1d(da1, da2):
    x = da1.values.astype(float)
    y = da2.values.astype(float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan
    return float(np.corrcoef(x[mask], y[mask])[0, 1])

def field_corr_2d(da1, da2):
    da1, da2 = xr.align(da1, da2, join="inner")
    a = da1.values.astype(float)
    b = da2.values.astype(float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return np.nan
    return float(np.corrcoef(a[mask].ravel(), b[mask].ravel())[0, 1])

def setup_map_axis(ax, proj):
    ax.set_extent([-180, 180, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.OCEAN, facecolor=OCEAN_COLOR, edgecolor="none", zorder=0)
    ax.add_feature(cfeature.LAND,  facecolor="white",      edgecolor="none", zorder=1)

    xticks = [30, 90, 150, -150, -90, -30]
    yticks = [-60, -30, 0, 30, 60]
    ax.set_xticks(xticks, crs=ccrs.PlateCarree())
    ax.set_yticks(yticks, crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
    ax.yaxis.set_major_formatter(LatitudeFormatter())

    ax.tick_params(axis="both", which="major",
                   direction="out", length=5, width=0.8,
                   labelsize=12, pad=3)

    ax.coastlines(linewidth=0.5, zorder=3)

    ax.text(1.04, -0.07, "30°W", transform=ax.transAxes,
            ha="right", va="top", fontsize=12)
    ax.plot([1.0, 1.0], [0.00, -0.04], transform=ax.transAxes,
            color="black", linewidth=0.8, clip_on=False)

    if hasattr(ax, "spines") and "geo" in ax.spines:
        ax.spines["geo"].set_zorder(20)
        ax.spines["geo"].set_linewidth(0.8)
    else:
        ax.outline_patch.set_zorder(20)
        ax.outline_patch.set_linewidth(0.8)


def _lon_0_360_to_m180_180(lon, arr2d):
    lon = np.asarray(lon)
    if np.nanmax(lon) > 180:  # 0~360
        lon2 = ((lon + 180.0) % 360.0) - 180.0
        order = np.argsort(lon2)
        return lon2[order], arr2d[:, order]
    return lon, arr2d

def plot_green_binary_pcolormesh(ax, da_mask01):
    da = da_mask01.sel(lat=slice(LAT_MIN, LAT_MAX))

    arr = xr.where(da == 1, 1.0, np.nan).values
    lon = da["lon"].values
    lat = da["lat"].values

    lon, arr = _lon_0_360_to_m180_180(lon, arr)
    arr_cyc, lon_cyc = add_cyclic_point(arr, coord=lon)

    cmap_green = ListedColormap(["#7bca73"])

    ax.pcolormesh(
        lon_cyc, lat, arr_cyc,
        transform=ccrs.PlateCarree(),
        cmap=cmap_green, vmin=0, vmax=1,
        shading="auto",
        linewidth=0,
        antialiased=False,
        zorder=2
    )

    ax.add_feature(cfeature.OCEAN, facecolor=OCEAN_COLOR, edgecolor="none", zorder=3)
    ax.coastlines(linewidth=0.5, zorder=4)

def draw_map_with_pie_from_mask(ax, proj, da_mask01, title, season_name, pct_nh_hemi, pct_sh_hemi):
    setup_map_axis(ax, proj)
    plot_green_binary_pcolormesh(ax, da_mask01)
    ax.set_title(f"{title}", fontsize=15, loc="left")

    era_p_nh = float(pct_nh_hemi)
    era_p_sh = float(pct_sh_hemi)

    PIE_SIZES = [66.65, 33.35]
    PIE_COLORS = ['#f393be', '#00a4d2']
    PIE_FS_PERCENT = 11.0
    PIE_FS_INNER = 12.0

    x0, y0, w_box, h_box = (0.45, 0.00, 0.52, 0.62)
    ax_pie = inset_axes(
        ax, width="100%", height="100%",
        bbox_to_anchor=(x0, y0, w_box, h_box),
        bbox_transform=ax.transAxes, borderpad=0.0
    )
    ax_pie.set_aspect('equal')

    wedges, texts, autotexts = ax_pie.pie(
        PIE_SIZES, colors=PIE_COLORS, startangle=90,
        autopct='%1.1f%%', counterclock=False, pctdistance=0.75
    )

    r_label = 0.58

    dx_nh, dy_nh = 0.00, 0.00
    dx_sh, dy_sh = -0.10, -0.30

    for j, t in enumerate(autotexts):
        w = wedges[j]
        theta = np.deg2rad(0.5 * (w.theta1 + w.theta2))
        x = r_label * np.cos(theta)
        y = r_label * np.sin(theta)

        if j == 0:
            t.set_text("NH")
            x += dx_nh;
            y += dy_nh
        else:
            t.set_text("SH")
            x += dx_sh;
            y += dy_sh

        t.set_position((x, y))
        t.set_ha("center");
        t.set_va("center")
        t.set_fontsize(PIE_FS_PERCENT)

    nh_wedge = wedges[0]
    boundary = nh_wedge.theta2
    gap_deg = 15.0
    offset = 0.10

    # inner NH
    angle_nh_inner = 360.0 * era_p_nh / 100.0
    theta2_nh_inner = boundary - gap_deg / 2.0
    theta1_nh_inner = theta2_nh_inner - angle_nh_inner
    theta_mid_nh_deg = 0.5 * (theta1_nh_inner + theta2_nh_inner)
    theta_mid_nh_rad = np.deg2rad(theta_mid_nh_deg)

    dx_nh = offset * np.cos(theta_mid_nh_rad)
    dy_nh = offset * np.sin(theta_mid_nh_rad)

    inner_wedge_nh = Wedge(
        center=(dx_nh, dy_nh), r=1.0,
        theta1=theta1_nh_inner, theta2=theta2_nh_inner,
        facecolor='#ee3b73', edgecolor='k', linewidth=0.6, alpha=0.9, zorder=3
    )
    ax_pie.add_patch(inner_wedge_nh)
    ax_pie.text(
        dx_nh + 0.7 * np.cos(theta_mid_nh_rad),
        dy_nh + 0.7 * np.sin(theta_mid_nh_rad),
        f"{era_p_nh:.1f}",
        ha='center', va='center',
        fontsize=PIE_FS_INNER, color='black', fontweight='bold'
    )

    # inner SH
    angle_sh_inner = 360.0 * era_p_sh / 100.0
    theta1_sh_inner = boundary + gap_deg / 2.0
    theta2_sh_inner = theta1_sh_inner + angle_sh_inner
    theta_mid_sh_deg = 0.5 * (theta1_sh_inner + theta2_sh_inner)
    theta_mid_sh_rad = np.deg2rad(theta_mid_sh_deg)

    dx_sh = offset * np.cos(theta_mid_sh_rad)
    dy_sh = offset * np.sin(theta_mid_sh_rad)

    inner_wedge_sh = Wedge(
        center=(dx_sh, dy_sh), r=1.0,
        theta1=theta1_sh_inner, theta2=theta2_sh_inner,
        facecolor='#1f78b4', edgecolor='k', linewidth=0.6, alpha=0.9, zorder=3
    )
    ax_pie.add_patch(inner_wedge_sh)
    ax_pie.text(
        dx_sh + 0.7 * np.cos(theta_mid_sh_rad),
        dy_sh + 0.7 * np.sin(theta_mid_sh_rad),
        f"{era_p_sh:.1f}",
        ha='center', va='center',
        fontsize=PIE_FS_INNER, color='black', fontweight='bold'
    )

    ax_pie.set_xticks([]); ax_pie.set_yticks([])
    for spine in ax_pie.spines.values():
        spine.set_visible(False)

def draw_timeseries(
    ax, da_nh, da_sh, tr_nh, tr_sh,
    season_name="Annual",
    ylim_left=None, yticks_left=None,
    ylim_right=None, yticks_right=None,
    show_legend=False
):
    years = da_nh["year"].values
    y_nh  = da_nh.values
    y_sh  = da_sh.values

    COLOR_NH = '#ee3b73'
    COLOR_SH = '#1f78b4'
    ax2 = ax.twinx()

    line_nh = ax.plot(years, y_nh, color=COLOR_NH, linewidth=1.5, label="NH", zorder=3)
    line_sh = ax2.plot(years, y_sh, color=COLOR_SH, linewidth=1.5, label="SH", zorder=3)

    std_nh = np.nanstd(y_nh)
    std_sh = np.nanstd(y_sh)
    ax.fill_between(years, y_nh - std_nh, y_nh + std_nh, color=COLOR_NH, alpha=0.11, linewidth=0, zorder=1)
    ax2.fill_between(years, y_sh - std_sh, y_sh + std_sh, color=COLOR_SH, alpha=0.11, linewidth=0, zorder=1)

    ax.plot(years, tr_nh["y_fit"].values, color=COLOR_NH, linewidth=1.8, linestyle='--', zorder=4)
    ax2.plot(years, tr_sh["y_fit"].values, color=COLOR_SH, linewidth=1.8, linestyle='--', zorder=4)

    if ylim_left is not None:
        ax.set_ylim(ylim_left)
    if yticks_left is not None:
        ax.set_yticks(yticks_left)

    if ylim_right is not None:
        ax2.set_ylim(ylim_right)
    if yticks_right is not None:
        ax2.set_yticks(yticks_right)

    ax.set_ylabel("Percentage (NH) [%]", color=COLOR_NH)
    ax2.set_ylabel("Percentage (SH) [%]", color=COLOR_SH)
    ax.tick_params(axis='y', colors=COLOR_NH)
    ax2.tick_params(axis='y', colors=COLOR_SH)

    slope_nh_100a = tr_nh["slope"] * 100.0
    slope_sh_100a = tr_sh["slope"] * 100.0
    txt = (
        f"NH: {slope_nh_100a:5.2f}%/C"
        + ("*" if tr_nh["pvalue"] is not None and tr_nh["pvalue"] <= 0.1 else "")
        + "     "
        f"SH: {slope_sh_100a:5.2f}%/C"
        + ("*" if tr_sh["pvalue"] is not None and tr_sh["pvalue"] <= 0.1 else "")
    )
    ax.text(0.043, 0.02, txt, transform=ax.transAxes, ha="left", va="bottom", fontsize=14)

    if show_legend:
        lines = line_nh + line_sh
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, frameon=False, fontsize=12,
                  loc="upper left", ncol=2, bbox_to_anchor=(0.55, 1.05))

def debug_mask(name, da):
    v = da.values
    print(f"\n[{name}] finite={np.isfinite(v).sum()}  ones={(v==1).sum()}  zeros={(v==0).sum()}  nan={np.isnan(v).sum()}")
    if np.isfinite(v).sum() > 0:
        vv = v[np.isfinite(v)]
        print(f"  unique (finite) = {np.unique(vv)[:10]}")
        print(f"  std (finite)    = {np.std(vv):.6f}")

def main():
    years40 = np.arange(1981, 2021)

    apt_mask  = read_comfort_mask_nc(APT_MASK_NC,  MASK_VAR_NAME, MASK_FILL)
    wbgt_mask = read_comfort_mask_nc(WBGT_MASK_NC, MASK_VAR_NAME, MASK_FILL)
    utci_mask = read_comfort_mask_nc(UTCI_MASK_NC, MASK_VAR_NAME, MASK_FILL)
    net_mask  = read_comfort_mask_nc(NET_MASK_NC,  MASK_VAR_NAME, MASK_FILL)

    debug_mask("APT",  apt_mask)
    debug_mask("WBGT", wbgt_mask)
    debug_mask("UTCI", utci_mask)
    debug_mask("NET",  net_mask)

    apt_mask_field  = mask_ocean_by_landsea_nhsh(apt_mask)
    wbgt_mask_field = mask_ocean_by_landsea_nhsh(wbgt_mask)
    utci_mask_field = mask_ocean_by_landsea_nhsh(utci_mask)
    net_mask_field  = mask_ocean_by_landsea_nhsh(net_mask)

    apt_nh  = read_percent_series_txt(APT_NH_TXT,  years40)
    apt_sh  = read_percent_series_txt(APT_SH_TXT,  years40)
    wbgt_nh = read_percent_series_txt(WBGT_NH_TXT, years40)
    wbgt_sh = read_percent_series_txt(WBGT_SH_TXT, years40)
    utci_nh = read_percent_series_txt(UTCI_NH_TXT, years40)
    utci_sh = read_percent_series_txt(UTCI_SH_TXT, years40)
    net_nh  = read_percent_series_txt(NET_NH_TXT,  years40)
    net_sh  = read_percent_series_txt(NET_SH_TXT,  years40)

    # print(f"  APT-mask  vs WBGT-mask = {field_corr_2d(apt_mask_field,  wbgt_mask_field): .3f}")
    # print(f"  APT-mask  vs UTCI-mask = {field_corr_2d(apt_mask_field,  utci_mask_field): .3f}")
    # print(f"  APT-mask  vs NET-mask  = {field_corr_2d(apt_mask_field,  net_mask_field): .3f}")
    # print(f"  WBGT-mask vs UTCI-mask = {field_corr_2d(wbgt_mask_field, utci_mask_field): .3f}")
    # print(f"  WBGT-mask vs NET-mask  = {field_corr_2d(wbgt_mask_field, net_mask_field): .3f}")
    # print(f"  UTCI-mask vs NET-mask  = {field_corr_2d(utci_mask_field, net_mask_field): .3f}")

    # def print_pairwise(name, series_dict):
    #     keys = list(series_dict.keys())
    #     print(f"\n=== 年序列相关（{name}）===")
    #     for i in range(len(keys)):
    #         for j in range(i+1, len(keys)):
    #             k1, k2 = keys[i], keys[j]
    #             r = corrcoef_1d(series_dict[k1], series_dict[k2])
    #             print(f"  {k1:5s} vs {k2:5s} = {r: .3f}")
    #
    # print_pairwise("NH", {"APT": apt_nh, "WBGT": wbgt_nh, "UTCI": utci_nh, "NET": net_nh})
    # print_pairwise("SH", {"APT": apt_sh, "WBGT": wbgt_sh, "UTCI": utci_sh, "NET": net_sh})

    wbgt_nh_mean = float(wbgt_nh.mean("year")); wbgt_sh_mean = float(wbgt_sh.mean("year"))
    utci_nh_mean = float(utci_nh.mean("year")); utci_sh_mean = float(utci_sh.mean("year"))
    net_nh_mean  = float(net_nh.mean("year"));  net_sh_mean  = float(net_sh.mean("year"))

    tr_wbgt_nh = linreg_1d(wbgt_nh["year"], wbgt_nh); tr_wbgt_sh = linreg_1d(wbgt_sh["year"], wbgt_sh)
    tr_utci_nh = linreg_1d(utci_nh["year"], utci_nh); tr_utci_sh = linreg_1d(utci_sh["year"], utci_sh)
    tr_net_nh  = linreg_1d(net_nh["year"],  net_nh);  tr_net_sh  = linreg_1d(net_sh["year"],  net_sh)

    proj = ccrs.PlateCarree(central_longitude=CENTER_LON)

    fig = plt.figure(figsize=(11, 13.0))
    gs = fig.add_gridspec(
        nrows=3, ncols=2,
        height_ratios=[1.0, 1.0, 1.0],
        hspace=0.12,
        wspace=0.175
    )

    ax_map_wbgt = fig.add_subplot(gs[0, 0], projection=proj)
    ax_ts_wbgt  = fig.add_subplot(gs[0, 1])

    ax_map_utci = fig.add_subplot(gs[1, 0], projection=proj)
    ax_ts_utci  = fig.add_subplot(gs[1, 1])

    ax_map_net  = fig.add_subplot(gs[2, 0], projection=proj)
    ax_ts_net   = fig.add_subplot(gs[2, 1])

    # WBGT
    draw_map_with_pie_from_mask(
        ax_map_wbgt, proj, wbgt_mask,
        title="WBGT", season_name="Annual (WBGT)",
        pct_nh_hemi=wbgt_nh_mean, pct_sh_hemi=wbgt_sh_mean
    )
    draw_timeseries(
        ax_ts_wbgt,
        wbgt_nh, wbgt_sh, tr_wbgt_nh, tr_wbgt_sh,
        season_name="Annual (WBGT)",
        ylim_left=NH_YLIM_WBGT, yticks_left=NH_TICKS_WBGT,
        ylim_right=SH_YLIM_WBGT, yticks_right=SH_TICKS_WBGT,
        show_legend=True
    )

    # UTCI
    draw_map_with_pie_from_mask(
        ax_map_utci, proj, utci_mask,
        title="UTCI", season_name="Annual (UTCI)",
        pct_nh_hemi=utci_nh_mean, pct_sh_hemi=utci_sh_mean
    )
    draw_timeseries(
        ax_ts_utci,
        utci_nh, utci_sh, tr_utci_nh, tr_utci_sh,
        season_name="Annual (UTCI)",
        ylim_left=NH_YLIM_UTCI, yticks_left=NH_TICKS_UTCI,
        ylim_right=SH_YLIM_UTCI, yticks_right=SH_TICKS_UTCI,
        show_legend=False
    )

    # NET
    draw_map_with_pie_from_mask(
        ax_map_net, proj, net_mask,
        title="NET", season_name="Annual (NET)",
        pct_nh_hemi=net_nh_mean, pct_sh_hemi=net_sh_mean
    )
    draw_timeseries(
        ax_ts_net,
        net_nh, net_sh, tr_net_nh, tr_net_sh,
        season_name="Annual (NET)",
        ylim_left=NH_YLIM_NET, yticks_left=NH_TICKS_NET,
        ylim_right=SH_YLIM_NET, yticks_right=SH_TICKS_NET,
        show_legend=False
    )

    fig.subplots_adjust(left=0.05, right=0.953, bottom=0.05, top=0.98)

    pairs = [(ax_map_wbgt, ax_ts_wbgt), (ax_map_utci, ax_ts_utci), (ax_map_net, ax_ts_net)]
    for ax_map, ax_ts in pairs:
        pos_map = ax_map.get_position()
        pos_ts  = ax_ts.get_position()
        ax_ts.set_position([pos_ts.x0, pos_map.y0, pos_ts.width, pos_map.height])

    offsets = [0.0, 0.14, 0.28]
    for (ax_map, ax_ts), dy in zip(pairs, offsets):
        if dy == 0.0:
            continue
        pos_map = ax_map.get_position()
        pos_ts = ax_ts.get_position()
        ax_map.set_position([pos_map.x0, pos_map.y0 + dy, pos_map.width, pos_map.height])
        ax_ts.set_position([pos_ts.x0, pos_ts.y0 + dy, pos_ts.width, pos_ts.height])

    ax_map_wbgt.text(-0.10, 1.02, "a", transform=ax_map_wbgt.transAxes,
                     ha="right", va="bottom", fontsize=22, fontweight="bold")
    ax_map_utci.text(-0.10, 1.02, "b", transform=ax_map_utci.transAxes,
                     ha="right", va="bottom", fontsize=22, fontweight="bold")
    ax_map_net.text(-0.10, 1.02, "c", transform=ax_map_net.transAxes,
                    ha="right", va="bottom", fontsize=22, fontweight="bold")

    ax_ts_wbgt.text(-0.115, 1.02, "d", transform=ax_ts_wbgt.transAxes,
                    ha="right", va="bottom", fontsize=22, fontweight="bold")
    ax_ts_utci.text(-0.115, 1.02, "e", transform=ax_ts_utci.transAxes,
                    ha="right", va="bottom", fontsize=22, fontweight="bold")
    ax_ts_net.text(-0.115, 1.02, "f", transform=ax_ts_net.transAxes,
                   ha="right", va="bottom", fontsize=22, fontweight="bold")

    plt.savefig(r"D:\code\comfort-area\fig\v7\figS3.png", dpi=800, bbox_inches="tight")

if __name__ == "__main__":
    main()
