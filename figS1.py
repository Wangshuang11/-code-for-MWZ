# -*- coding: utf-8 -*-
import os
import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
import matplotlib.ticker as mticker
from matplotlib.colors import ListedColormap
from matplotlib.patches import Wedge
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import linregress
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from pathlib import Path

mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 14

FILL_MISS = 32767.0
YEAR0, YEAR1 = 1981, 2020
YEARS = np.arange(YEAR0, YEAR1 + 1)

NH_YLIM_1     = (15, 27)
SH_YLIM_1     = (12, 18)
NH_TICKS_1    = [15, 19, 23, 27]
SH_TICKS_1    = [12, 14, 16, 18]
NH_LABELS_1   = [15, 19, 23, 27]
SH_LABELS_1   = [12, 14, 16, 18]

NH_YLIM_2     = (14, 26)
SH_YLIM_2     = (12, 16)
NH_TICKS_2    = [14, 18, 22, 26]
SH_TICKS_2    = [12, 14, 16, 18]
NH_LABELS_2   = [14, 18, 22, 26]
SH_LABELS_2   = [12, 14, 16, 18]

NH_YLIM_3     = (12, 24)
SH_YLIM_3     = (12, 18)
NH_TICKS_3    = [12, 16, 20, 24]
SH_TICKS_3    = [12, 14, 16, 18]
NH_LABELS_3   = [12, 16, 20, 24]
SH_LABELS_3   = [12, 14, 16, 18]


def smooth9(field2d: np.ndarray) -> np.ndarray:
    tmp = np.pad(field2d, ((1, 1), (0, 0)), mode='edge')
    arr = np.pad(tmp, ((0, 0), (1, 1)), mode='wrap')
    s = (
        arr[0:-2, 0:-2] + arr[0:-2, 1:-1] + arr[0:-2, 2:] +
        arr[1:-1, 0:-2] + arr[1:-1, 1:-1] + arr[1:-1, 2:] +
        arr[2:, 0:-2] + arr[2:, 1:-1] + arr[2:, 2:]
    ) / 9.0
    return s

def linreg_1d(year_da, data_da):
    years = year_da.values.astype(float)
    data  = data_da.values.astype(float)

    mask = np.isfinite(years) & np.isfinite(data)
    if mask.sum() < 3:
        return {
            "slope": np.nan,
            "intercept": np.nan,
            "pvalue": np.nan,
            "y_fit": xr.full_like(year_da, np.nan, dtype=float),
        }

    res = linregress(years[mask], data[mask])
    y_fit = res.slope * years + res.intercept

    return {
        "slope": res.slope,
        "intercept": res.intercept,
        "pvalue": res.pvalue,
        "y_fit": xr.DataArray(y_fit, coords={"year": year_da["year"]}, dims=("year",)),
    }


def corrcoef_1d(da1, da2):
    x = da1.values.astype(float)
    y = da2.values.astype(float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def _load_txt_percent(txt_path):
    arr = np.loadtxt(txt_path).astype(float).reshape(-1)
    if arr.size != YEARS.size:
        raise ValueError(f"[TXT长度不对] {txt_path} 行数={arr.size}，应为 {YEARS.size}（{YEAR0}-{YEAR1}）")

    vmax = np.nanmax(arr)
    if np.isfinite(vmax) and vmax <= 1.5:
        arr = arr * 100.0

    da = xr.DataArray(arr, coords={"year": YEARS}, dims=("year",))
    return da


def load_dataset_group(mask_nc, nh_txt, sh_txt, var_mask="mild_area"):
    with xr.open_dataset(mask_nc) as ds:
        if var_mask in ds.data_vars:
            da = ds[var_mask].load()
        else:
            da = ds[list(ds.data_vars)[0]].load()

    if "lat" not in da.dims or "lon" not in da.dims:
        rename_map = {}
        if "latitude" in da.dims:
            rename_map["latitude"] = "lat"
        if "longitude" in da.dims:
            rename_map["longitude"] = "lon"
        if rename_map:
            da = da.rename(rename_map)

    data = da.astype(float)
    data = data.where(data != FILL_MISS)  # 32767 -> NaN

    mask01 = xr.where(np.isfinite(data) & (data >= 0.5), 1.0, 0.0)
    mask01 = mask01.where(np.isfinite(data))

    pct_nh = _load_txt_percent(nh_txt)
    pct_sh = _load_txt_percent(sh_txt)

    pct_nh_mean = float(pct_nh.mean(dim="year"))
    pct_sh_mean = float(pct_sh.mean(dim="year"))

    tr_nh = linreg_1d(pct_nh["year"], pct_nh)
    tr_sh = linreg_1d(pct_sh["year"], pct_sh)

    res_NH = {"area_percent_hemi": pct_nh, "trend_area_hemi": tr_nh}
    res_SH = {"area_percent_hemi": pct_sh, "trend_area_hemi": tr_sh}

    return mask01, res_NH, res_SH, pct_nh_mean, pct_sh_mean


def draw_map_with_pie_generic(ax, da_mask01, dataset_label, pct_nh=0.0, pct_sh=0.0):
    lat = da_mask01["lat"].values
    lon = da_mask01["lon"].values

    mask01 = da_mask01.values.astype(float)
    ocean_mask = ~np.isfinite(mask01)

    mask_val = np.where(mask01 == 1.0, 1.0, 0.0)
    mask_smooth = mask_val
    mask_bin = np.where(mask_smooth >= 0.5, 1.0, 0.0).astype(float)
    mask_bin[ocean_mask] = np.nan

    mask_cyc, lon_cyc = add_cyclic_point(mask_bin, coord=lon)
    ocean_mask_cyc, _ = add_cyclic_point(ocean_mask.astype(int), coord=lon)
    ocean_mask_cyc = ocean_mask_cyc.astype(bool)

    ax.set_extent([0, 360, -61, 61], crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.LAND, facecolor="white", edgecolor="none", zorder=0)

    green_field = np.ma.array(mask_cyc, mask=(ocean_mask_cyc | (mask_cyc < 0.5)))
    cmap_green = ListedColormap(["#7bca73"])

    ax.pcolormesh(
        lon_cyc, lat, green_field,
        cmap=cmap_green,
        vmin=0.5, vmax=1.5,
        shading="auto",
        transform=ccrs.PlateCarree(),
        zorder=2
    )

    ax.add_feature(cfeature.OCEAN, facecolor="#e0f2fc", edgecolor="none", zorder=3)

    ax.coastlines(linewidth=0.5, zorder=4)

    xticks = [30, 90, 150, 210, 270, 330]
    yticks = np.arange(-60, 61, 30)

    ax.set_xticks(xticks, crs=ccrs.PlateCarree())
    ax.set_yticks(yticks, crs=ccrs.PlateCarree())

    ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
    ax.yaxis.set_major_formatter(LatitudeFormatter())

    ax.text(0.95, -0.055, "30°W",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=14, zorder=20)

    ax.plot([1.0, 1.0], [0.0, -0.035],
            transform=ax.transAxes, color="black",
            linewidth=0.7, clip_on=False, zorder=20)

    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        length=6,
        width=0.8,
        labelsize=14,
        pad=2
    )

    ax.tick_params(axis="x", top=False, labeltop=False, bottom=True, labelbottom=True)
    ax.tick_params(axis="y", right=False, labelright=False, left=True, labelleft=True)

    ax.tick_params(
        axis="both", which="major",
        length=4, width=0.8, direction="out",
        labelsize=14, pad=2,
        top=False, right=False
    )

    ax.set_title(f"{dataset_label}", fontsize=15, loc="left")

    # print(
    #     f"[{dataset_label}] APT 舒适面积(占全球陆地的 40 年平均): "
    #     f"NH = {pct_nh:.2f}%, SH = {pct_sh:.2f}%"
    # )

    PIE_SIZES = [66.65, 33.35]
    PIE_COLORS = ['#f393be', '#00a4d2']
    PIE_FS_PERCENT = 11.0
    PIE_FS_INNER   = 12.0

    x0, y0, w_box, h_box = (0.45, 0.00, 0.52, 0.62)
    ax_pie = inset_axes(
        ax, width="100%", height="100%",
        bbox_to_anchor=(x0, y0, w_box, h_box),
        bbox_transform=ax.transAxes,
        borderpad=0.0
    )
    ax_pie.set_aspect('equal')

    wedges, texts, autotexts = ax_pie.pie(
        PIE_SIZES, colors=PIE_COLORS, startangle=90,
        autopct='%1.1f%%', counterclock=False, pctdistance=0.75
    )

    offsets = {0: (-0.15, 0.10), 1: (0.05, -0.30)}
    for j, t in enumerate(autotexts):
        x_t, y_t = t.get_position()
        dx, dy = offsets.get(j, (0.0, 0.0))
        t.set_position((x_t + dx, y_t + dy))
        t.set_fontsize(PIE_FS_PERCENT)
        t.set_text("NH" if j == 0 else "SH")

    nh_wedge = wedges[0]
    theta1_nh, theta2_nh = nh_wedge.theta1, nh_wedge.theta2
    boundary = theta2_nh
    gap_deg = 15.0
    offset  = 0.10

    # --- NH inner ---
    angle_nh_inner = 360.0 * float(pct_nh) / 100.0
    theta2_nh_inner = boundary - gap_deg / 2.0
    theta1_nh_inner = theta2_nh_inner - angle_nh_inner
    theta_mid_nh = 0.5 * (theta1_nh_inner + theta2_nh_inner)
    rad_nh = np.deg2rad(theta_mid_nh)
    dx_nh = offset * np.cos(rad_nh)
    dy_nh = offset * np.sin(rad_nh)

    inner_wedge_nh = Wedge(
        center=(dx_nh, dy_nh), r=1.0,
        theta1=theta1_nh_inner, theta2=theta2_nh_inner,
        facecolor='#ee3b73', edgecolor='k', linewidth=0.6, alpha=0.9, zorder=3
    )
    ax_pie.add_patch(inner_wedge_nh)

    ax_pie.text(
        dx_nh + 0.7*np.cos(rad_nh), dy_nh + 0.7*np.sin(rad_nh),
        f"{pct_nh:.1f}", ha='center', va='center',
        fontsize=PIE_FS_INNER, color='black', fontweight='bold'
    )

    # --- SH inner ---
    angle_sh_inner = 360.0 * float(pct_sh) / 100.0
    theta1_sh_inner = boundary + gap_deg / 2.0
    theta2_sh_inner = theta1_sh_inner + angle_sh_inner
    theta_mid_sh = 0.5 * (theta1_sh_inner + theta2_sh_inner)
    rad_sh = np.deg2rad(theta_mid_sh)
    dx_sh = offset * np.cos(rad_sh)
    dy_sh = offset * np.sin(rad_sh)

    inner_wedge_sh = Wedge(
        center=(dx_sh, dy_sh), r=1.0,
        theta1=theta1_sh_inner, theta2=theta2_sh_inner,
        facecolor='#1f78b4', edgecolor='k', linewidth=0.6, alpha=0.9, zorder=3
    )
    ax_pie.add_patch(inner_wedge_sh)

    ax_pie.text(
        dx_sh + 0.7*np.cos(rad_sh), dy_sh + 0.7*np.sin(rad_sh),
        f"{pct_sh:.1f}", ha='center', va='center',
        fontsize=PIE_FS_INNER, color='black', fontweight='bold'
    )

    ax_pie.set_xticks([])
    ax_pie.set_yticks([])
    for spine in ax_pie.spines.values():
        spine.set_visible(False)

    try:
        ax.spines["geo"].set_zorder(10)
        ax.spines["geo"].set_linewidth(0.8)
        ax.spines["geo"].set_edgecolor("k")
    except Exception:
        ax.outline_patch.set_zorder(10)
        ax.outline_patch.set_linewidth(0.8)
        ax.outline_patch.set_edgecolor("k")


def compute_green_mask_flat_from_mask01(da_mask01):
    mask01 = da_mask01.values.astype(float)
    ocean_mask = ~np.isfinite(mask01)

    mask_val = np.where(mask01 == 1.0, 1.0, 0.0)
    mask_smooth = mask_val
    mask_bin = np.where(mask_smooth >= 0.5, 1.0, 0.0).astype(float)
    mask_bin[ocean_mask] = np.nan
    return mask_bin.ravel()

def field_corr_green(mask01_a, mask01_b):
    f1 = compute_green_mask_flat_from_mask01(mask01_a)
    f2 = compute_green_mask_flat_from_mask01(mask01_b)
    ok = np.isfinite(f1) & np.isfinite(f2)
    if ok.sum() < 3:
        return np.nan
    return float(np.corrcoef(f1[ok], f2[ok])[0, 1])

def draw_timeseries(
    ax, res_NH, res_SH, dataset_label="",
    ylim_left=None, yticks_left=None, yticklabels_left=None,
    ylim_right=None, yticks_right=None, yticklabels_right=None,
    add_legend=False,
):
    da_nh = res_NH["area_percent_hemi"]
    da_sh = res_SH["area_percent_hemi"]

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

    ax.fill_between(years, y_nh-std_nh, y_nh+std_nh, color=COLOR_NH, alpha=0.15, linewidth=0, zorder=1)
    ax2.fill_between(years, y_sh-std_sh, y_sh+std_sh, color=COLOR_SH, alpha=0.15, linewidth=0, zorder=1)

    tr_nh = res_NH["trend_area_hemi"]
    tr_sh = res_SH["trend_area_hemi"]

    ax.plot(years, tr_nh["y_fit"].values, color=COLOR_NH, linewidth=1.8, linestyle='--', zorder=4)
    ax2.plot(years, tr_sh["y_fit"].values, color=COLOR_SH, linewidth=1.8, linestyle='--', zorder=4)

    if ylim_left is not None:
        ax.set_ylim(ylim_left)
    if yticks_left is not None:
        ax.set_yticks(yticks_left)
    if yticklabels_left is not None:
        ax.set_yticklabels(yticklabels_left)

    if ylim_right is not None:
        ax2.set_ylim(ylim_right)
    if yticks_right is not None:
        ax2.set_yticks(yticks_right)
    if yticklabels_right is not None:
        ax2.set_yticklabels(yticklabels_right)

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
    ax.text(0.042, 0.02, txt, transform=ax.transAxes, ha="left", va="bottom", fontsize=14)

    if add_legend:
        lines = line_nh + line_sh
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, frameon=False, fontsize=12,
                  loc="upper left", ncol=2, bbox_to_anchor=(0.55, 1.05))


def main():
    BASE_V5 = r"D:\code\comfort-area\data\v5"

    def paths_for(dsname):
        base = os.path.join(BASE_V5, dsname)
        mask_nc = os.path.join(base, "mild_area_1981_2020_APT_annual.nc")
        nh_txt  = os.path.join(base, "mild_area_1981_2020_APT_annual_nh.txt")
        sh_txt  = os.path.join(base, "mild_area_1981_2020_APT_annual_sh.txt")
        return mask_nc, nh_txt, sh_txt

    m2_mask_nc, m2_nh_txt, m2_sh_txt = paths_for("MERRA2")
    jra_mask_nc, jra_nh_txt, jra_sh_txt = paths_for("JRA55")
    n2_mask_nc, n2_nh_txt, n2_sh_txt = paths_for("NCEP2")

    mask_m2,  res_NH_m2,  res_SH_m2,  pct_nh_m2,  pct_sh_m2  = load_dataset_group(m2_mask_nc,  m2_nh_txt,  m2_sh_txt)
    mask_jra, res_NH_jra, res_SH_jra, pct_nh_jra, pct_sh_jra = load_dataset_group(jra_mask_nc, jra_nh_txt, jra_sh_txt)
    mask_n2,  res_NH_n2,  res_SH_n2,  pct_nh_n2,  pct_sh_n2  = load_dataset_group(n2_mask_nc,  n2_nh_txt,  n2_sh_txt)

    # ===== 打印 NH / SH 时间序列相关 =====
    nh_m2  = res_NH_m2["area_percent_hemi"]
    nh_jra = res_NH_jra["area_percent_hemi"]
    nh_n2  = res_NH_n2["area_percent_hemi"]

    sh_m2  = res_SH_m2["area_percent_hemi"]
    sh_jra = res_SH_jra["area_percent_hemi"]
    sh_n2  = res_SH_n2["area_percent_hemi"]

    # print("\n=== NH 相关系数（时间序列） ===")
    # print(f"  MERRA2 vs JRA55 = {corrcoef_1d(nh_m2, nh_jra):.3f}")
    # print(f"  MERRA2 vs NCEP2 = {corrcoef_1d(nh_m2, nh_n2 ):.3f}")
    # print(f"  JRA55  vs NCEP2 = {corrcoef_1d(nh_jra, nh_n2):.3f}")
    #
    # print("\n=== SH 相关系数（时间序列） ===")
    # print(f"  MERRA2 vs JRA55 = {corrcoef_1d(sh_m2, sh_jra):.3f}")
    # print(f"  MERRA2 vs NCEP2 = {corrcoef_1d(sh_m2, sh_n2 ):.3f}")
    # print(f"  JRA55  vs NCEP2 = {corrcoef_1d(sh_jra, sh_n2):.3f}")

    # print("\n=== 场相关系数===")
    # print(f"  MERRA2 vs JRA55 = {field_corr_green(mask_m2,  mask_jra):.3f}")
    # print(f"  MERRA2 vs NCEP2 = {field_corr_green(mask_m2,  mask_n2 ):.3f}")
    # print(f"  JRA55  vs NCEP2 = {field_corr_green(mask_jra, mask_n2 ):.3f}")

    proj = ccrs.PlateCarree(central_longitude=150)

    fig = plt.figure(figsize=(11, 12.0))
    gs = fig.add_gridspec(
        nrows=3, ncols=2,
        height_ratios=[1.0, 1.0, 1.0],
        hspace=0.10,
        wspace=0.175
    )

    ax_map_m2  = fig.add_subplot(gs[0, 0], projection=proj)
    ax_ts_m2   = fig.add_subplot(gs[0, 1])
    ax_map_jra = fig.add_subplot(gs[1, 0], projection=proj)
    ax_ts_jra  = fig.add_subplot(gs[1, 1])
    ax_map_n2  = fig.add_subplot(gs[2, 0], projection=proj)
    ax_ts_n2   = fig.add_subplot(gs[2, 1])

    # ---- MERRA2 ----
    draw_map_with_pie_generic(ax_map_m2, mask_m2, "MERRA2", pct_nh=pct_nh_m2, pct_sh=pct_sh_m2)
    draw_timeseries(
        ax_ts_m2, res_NH_m2, res_SH_m2, "MERRA2",
        ylim_left=NH_YLIM_1, yticks_left=NH_TICKS_1, yticklabels_left=NH_LABELS_1,
        ylim_right=SH_YLIM_1, yticks_right=SH_TICKS_1, yticklabels_right=SH_LABELS_1,
        add_legend=True
    )

    # ---- JRA55 ----
    draw_map_with_pie_generic(ax_map_jra, mask_jra, "JRA55", pct_nh=pct_nh_jra, pct_sh=pct_sh_jra)
    draw_timeseries(
        ax_ts_jra, res_NH_jra, res_SH_jra, "JRA55",
        ylim_left=NH_YLIM_2, yticks_left=NH_TICKS_2, yticklabels_left=NH_LABELS_2,
        ylim_right=SH_YLIM_2, yticks_right=SH_TICKS_2, yticklabels_right=SH_LABELS_2,
        add_legend=False
    )

    # ---- NCEP2 ----
    draw_map_with_pie_generic(ax_map_n2, mask_n2, "NCEP2", pct_nh=pct_nh_n2, pct_sh=pct_sh_n2)
    draw_timeseries(
        ax_ts_n2, res_NH_n2, res_SH_n2, "NCEP2",
        ylim_left=NH_YLIM_3, yticks_left=NH_TICKS_3, yticklabels_left=NH_LABELS_3,
        ylim_right=SH_YLIM_3, yticks_right=SH_TICKS_3, yticklabels_right=SH_LABELS_3,
        add_legend=False
    )

    fig.subplots_adjust(left=0.05, right=0.953, bottom=0.06, top=0.97)

    pairs = [(ax_map_m2, ax_ts_m2), (ax_map_jra, ax_ts_jra), (ax_map_n2, ax_ts_n2)]
    for ax_map, ax_ts in pairs:
        pos_map = ax_map.get_position()
        pos_ts  = ax_ts.get_position()
        ax_ts.set_position([pos_ts.x0, pos_map.y0, pos_ts.width, pos_map.height])

    offsets = [0.0, 0.12, 0.24]
    for (ax_map, ax_ts), dy in zip(pairs, offsets):
        if dy == 0.0:
            continue
        pos_map = ax_map.get_position()
        pos_ts  = ax_ts.get_position()
        ax_map.set_position([pos_map.x0, pos_map.y0 + dy, pos_map.width, pos_map.height])
        ax_ts.set_position([pos_ts.x0,  pos_ts.y0  + dy, pos_ts.width,  pos_ts.height])

    ax_map_m2.text(-0.10, 1.02, "a", transform=ax_map_m2.transAxes,
                   ha="right", va="bottom", fontsize=22, fontweight="bold")
    ax_map_jra.text(-0.10, 1.02, "b", transform=ax_map_jra.transAxes,
                    ha="right", va="bottom", fontsize=22, fontweight="bold")
    ax_map_n2.text(-0.10, 1.02, "c", transform=ax_map_n2.transAxes,
                   ha="right", va="bottom", fontsize=22, fontweight="bold")

    ax_ts_m2.text(-0.115, 1.02, "d", transform=ax_ts_m2.transAxes,
                  ha="right", va="bottom", fontsize=22, fontweight="bold")
    ax_ts_jra.text(-0.115, 1.02, "e", transform=ax_ts_jra.transAxes,
                   ha="right", va="bottom", fontsize=22, fontweight="bold")
    ax_ts_n2.text(-0.115, 1.02, "f", transform=ax_ts_n2.transAxes,
                  ha="right", va="bottom", fontsize=22, fontweight="bold")

    plt.savefig(r"D:\code\comfort-area\fig\v7\figS1.png",
                dpi=800, bbox_inches='tight', pad_inches=0.05)


if __name__ == "__main__":
    main()