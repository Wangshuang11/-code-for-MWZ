# -*- coding: utf-8 -*-
import os
import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import t as t_dist
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


FONT_FAMILY = "Times New Roman"
FS_BASE   = 16
FS_TITLE  = 18
FS_LABEL  = 16
FS_TICK   = 14

mpl.rcParams['font.family'] = FONT_FAMILY
mpl.rcParams['font.size'] = FS_BASE
mpl.rcParams["mathtext.fontset"] = "custom"
mpl.rcParams["mathtext.rm"] = FONT_FAMILY
mpl.rcParams["mathtext.it"] = FONT_FAMILY + ":italic"
mpl.rcParams["mathtext.bf"] = FONT_FAMILY + ":bold"

ALPHA_CI = 0.05

BAR1_YLIM_RIGHT   = (0, 100)
BAR1_YTICKS_RIGHT = [0, 25, 50, 75, 100]

BAR_YLIM_RIGHT   = (-15.2, 2)
BAR_YTICKS_RIGHT = [-15, -10, -5, 0, 2]

SCALE_YLIM = {
    "NH": (-3, 3),
    "SH": (-3, 3),
}
SCALE_YTICKS = {
    "NH": [-3, -2, -1, 0, 1, 2, 3],
    "SH": [-3, -2, -1, 0, 1, 2, 3],
}

AREA_NH = 9933.0
AREA_SH = 4970.0
AREA_GL = AREA_NH + AREA_SH

LOW, HIGH = 17.0, 26.0
LAT_EDGE  = 67.5

BASE_ERA5 = r"D:\code\comfort-area\data\ERA5"
ERA5_ANNUAL_FILE = os.path.join(BASE_ERA5, "apt.1981-2020_annual_maskocean.nc")
AREA_FILE        = os.path.join(BASE_ERA5, "aera_25x25_nh_0_70.nc")

CHANGE_FILES = {
    "Rh20"  : "apt_annual_maskocean_change_Rh_10years.nc",
    "Tas20" : "apt_annual_maskocean_change_Tas_10years.nc",
    "Wind20": "apt_annual_maskocean_change_Wind_10years.nc",
}
ERA5_VAR = "apt"

USE_MANUAL_A = True

results_refchg = {}

if USE_MANUAL_A:
    results_refchg = {
        "REF":    {"NH_pct": 20.20, "SH_pct": 15.24},
        "Tas20":  {"NH_pct": 93.34, "SH_pct": 43.85},
        "Rh20":   {"NH_pct": 2.32,  "SH_pct": 36.15},
        "Wind20": {"NH_pct": 4.34,  "SH_pct": 20.00},
    }
else:
    ds_area = xr.open_dataset(AREA_FILE)
    area_da = ds_area["area"]
    dims = list(area_da.dims)
    lat_dim, lon_dim = dims[0], dims[1]
    area_da = area_da.rename({lat_dim: "lat", lon_dim: "lon"})
    ds_area.close()

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

    def compute_pie_parts_from_area(da_field, area_da, low=LOW, high=HIGH, lat_edge=LAT_EDGE):
        field_nh = _band_sel_strict(da_field, 0.0, lat_edge)
        area_nh  = area_da
        mask_nh  = (field_nh >= low) & (field_nh <= high) & np.isfinite(field_nh)
        nh_comf_area = area_nh.where(mask_nh).sum(dim=("lat", "lon"), skipna=True)

        field_sh = _band_sel_strict(da_field, 0.0, -lat_edge)
        area_sh  = area_da
        mask_sh  = (field_sh >= low) & (field_sh <= high) & np.isfinite(field_sh)
        sh_comf_area = area_sh.where(mask_sh).sum(dim=("lat", "lon"), skipna=True)

        global_land_area = AREA_GL
        part_nh = float(100.0 * nh_comf_area / global_land_area)
        part_sh = float(100.0 * sh_comf_area / global_land_area)

        return part_nh, part_sh, float(nh_comf_area), float(sh_comf_area), float(global_land_area)

    def process_case(case_name, da_field):
        part_nh, part_sh, a_nh, a_sh, a_gl = compute_pie_parts_from_area(da_field, area_da)
        print(
            f"[{case_name}] NH(舒适/全球陆地)={part_nh:.2f}%, "
            f"SH(舒适/全球陆地)={part_sh:.2f}% | "
            f"NH_area={a_nh:.2f}, SH_area={a_sh:.2f}, Global_land={a_gl:.2f}"
        )
        return {"NH_pct": part_nh, "SH_pct": part_sh}

    ds_era = xr.open_dataset(ERA5_ANNUAL_FILE)
    da_era = ds_era[ERA5_VAR]
    mv_era = da_era.attrs.get("missing_value", da_era.attrs.get("_FillValue", 32767))
    da_era = da_era.where(da_era != mv_era)
    if "year" not in da_era.dims:
        raise ValueError("ERA5 数据中没有 year 维")
    da_era_19 = da_era.isel(year=slice(0, 10)).mean(dim="year")
    ds_era.close()

    results_refchg = {}
    print("\n===== 处理 REF(ERA5 1981–1999) =====")
    results_refchg["REF"] = process_case("REF", da_era_19)

    for key, fname in CHANGE_FILES.items():
        path = os.path.join(BASE_ERA5, fname)
        if not os.path.exists(path):
            print(f"[警告] 文件不存在：{path}，跳过。")
            continue
        print(f"\n===== 处理 {key}: {fname} =====")
        ds = xr.open_dataset(path)
        da = ds[ERA5_VAR]
        da = da.where(da != 32767)
        if "year" not in da.dims:
            raise ValueError(f"{fname} 中没有 year 维")
        da_mean = da.mean(dim="year")
        ds.close()
        results_refchg[key] = process_case(key, da_mean)

case_order_plot = ["REF", "Tas20", "Rh20", "Wind20"]
cases_ref = [c for c in case_order_plot if c in results_refchg]

nh_vals_ref = [results_refchg[c]["NH_pct"] for c in cases_ref]
sh_vals_ref = [results_refchg[c]["SH_pct"] for c in cases_ref]

nh_diff_ref = [np.nan] * len(cases_ref)
sh_diff_ref = [np.nan] * len(cases_ref)
if "REF" in cases_ref:
    idx_ref = cases_ref.index("REF")
    base_nh = nh_vals_ref[idx_ref]
    base_sh = sh_vals_ref[idx_ref]
    for i in range(len(cases_ref)):
        nh_diff_ref[i] = nh_vals_ref[i]
        sh_diff_ref[i] = sh_vals_ref[i]

label_map_ref = {
    "REF":    "Ref",
    "Tas20":  "Tas",
    "Rh20":   "RH",
    "Wind20": "Wind",
}

COLOR_NH = "#f393be"
COLOR_SH = "#00a4d2"

if "REF" in cases_ref:
    idx_ref = cases_ref.index("REF")
    base_nh = nh_vals_ref[idx_ref]
    base_sh = sh_vals_ref[idx_ref]
    print(f"REF 基准：NH={base_nh:.2f}%, SH={base_sh:.2f}%")
else:
    idx_ref = None
    print("⚠️ cases_ref 里没有 REF，差值列可能无意义。")

print("\n--- 每个case的左轴百分比 & 右轴(相对REF差值) ---")
for i, c in enumerate(cases_ref):
    lab = label_map_ref.get(c, c)
    nh = nh_vals_ref[i]
    sh = sh_vals_ref[i]
    dnh = nh_diff_ref[i]
    dsh = sh_diff_ref[i]
    print(f"{i:>2d} | {lab:<10s} ({c:<6s})  "
          f"NH={nh:6.2f}%  SH={sh:6.2f}%   "
          f"ΔNH={dnh:+6.2f}  ΔSH={dsh:+6.2f}")

BASE_TXT   = r"D:\code\comfort-area\data\v5\CMIP6"
BASE_CESM2 = r"D:\code\comfort-area\data\v5\CESM2"

SCEN_ORDER = ["ALL", "CESM2", "GHG", "AER", "NAT"]
SCEN_LABEL = ["ALL", "CESM2-LENS2", "GHG", "AA", "NAT"]

SCEN_TO_EXPTAG = {
    "ALL": "historical",
    "GHG": "hist-GHG",
    "AER": "hist-aer",
    "NAT": "hist-nat",
}

def _load_txt_any(fn: str):
    vals = np.loadtxt(fn, dtype=float)
    vals = np.atleast_1d(vals).astype(float)
    return vals

def _years_from_len(n: int):
    if n == 34:
        return np.arange(1981, 2015, dtype=int)
    if n == 40:
        return np.arange(1981, 2021, dtype=int)
    raise ValueError(f"序列长度不是 34 或 40：n={n}")

def read_series_pct_txt_mme(scen: str, hemi: str):
    if scen not in SCEN_TO_EXPTAG:
        raise FileNotFoundError(f"SCEN={scen} 没有配置对应的 txt 试验标签。")
    exptag = SCEN_TO_EXPTAG[scen]
    fn = os.path.join(
        BASE_TXT,
        f"mild_area_1981_2020_MME_{exptag}_r1i1p1f1_annual_{hemi.lower()}.txt"
    )
    if not os.path.exists(fn):
        raise FileNotFoundError(f"未找到 txt: {fn}")
    vals = _load_txt_any(fn)
    if vals.size != 40:
        raise ValueError(f"{fn} 行数不是40：读到 {vals.size} 个值（应为1981-2020共40年）")
    years = np.arange(1981, 2021, dtype=int)
    return years, vals

def read_series_pct_txt_cesm2(hemi: str):
    cand = [
        os.path.join(BASE_CESM2, f"mild_area_1981_2020_MME_BHIST_annual_{hemi.lower()}_2.txt"),
    ]
    fn = None
    for f in cand:
        if os.path.exists(f):
            fn = f
            break
    if fn is None:
        raise FileNotFoundError(f"未找到 CESM2 txt，尝试过：\n  " + "\n  ".join(cand))
    vals = _load_txt_any(fn)
    if vals.size != 40:
        raise ValueError(f"{fn} 读到 {vals.size} 个值，但 CESM2 应为 40（1981–2020）")
    years = np.arange(1981, 2021, dtype=int)
    return years, vals

series_NH_pct = {}
series_SH_pct = {}

for scen in ["ALL", "GHG", "AER", "NAT"]:
    yN, vN = read_series_pct_txt_mme(scen, "nh")
    yS, vS = read_series_pct_txt_mme(scen, "sh")
    series_NH_pct[scen] = (yN, vN)
    series_SH_pct[scen] = (yS, vS)

yN, vN = read_series_pct_txt_cesm2("nh")
yS, vS = read_series_pct_txt_cesm2("sh")
series_NH_pct["CESM2"] = (yN, vN)
series_SH_pct["CESM2"] = (yS, vS)

def calc_trend_100yr(years, series_pct):
    mask = np.isfinite(series_pct)
    x = years[mask].astype(float)
    y = series_pct[mask]
    if x.size < 3:
        return np.nan, False

    b, a = np.polyfit(x, y, 1)
    trend_100yr = b * 100.0

    n = len(x)
    y_fit = a + b * x
    resid = y - y_fit
    s2 = np.sum(resid**2) / (n - 2)
    x0 = x.mean()
    Sxx = np.sum((x - x0)**2)
    if s2 <= 0 or Sxx <= 0:
        return trend_100yr, False

    se_b = np.sqrt(s2 / Sxx)
    t_b = b / se_b
    df = n - 2
    tcrit = t_dist.ppf(1 - ALPHA_CI/2, df)
    sig = np.abs(t_b) > tcrit
    return trend_100yr, bool(sig)

trend_NH = []
trend_SH = []
sig_NH = []
sig_SH = []

for scen in SCEN_ORDER:
    years_N, series_N = series_NH_pct[scen]
    years_S, series_S = series_SH_pct[scen]
    tN, sN = calc_trend_100yr(years_N, series_N)
    tS, sS = calc_trend_100yr(years_S, series_S)
    trend_NH.append(tN); trend_SH.append(tS)
    sig_NH.append(sN);  sig_SH.append(sS)

trend_NH = np.array(trend_NH)
trend_SH = np.array(trend_SH)
sig_NH   = np.array(sig_NH, dtype=bool)
sig_SH   = np.array(sig_SH, dtype=bool)

# =================== 第2行：缩放因子 ===================
beta_inf_a = np.array([
    [0.106, -1.553, -1.091],
    [0.171, -1.237, -0.950],
])
beta_a = np.array([
    [0.613,  0.380,  0.355],
    [0.761, -0.213,  0.560],
])
beta_sup_a = np.array([
    [1.119,  2.313,  1.801],
    [1.351,  0.812,  2.070],
])

beta_inf_b = np.array([
    [0.359, -0.639, -0.718],
    [0.215, -1.551, -1.147],
])
beta_b = np.array([
    [0.703,  1.015,  0.641],
    [0.774, -0.565,  0.394],
])
beta_sup_b = np.array([
    [1.047,  2.669,  2.000],
    [1.333,  0.420,  1.935],
])

SIGNALS    = ["GHG", "AA", "NAT"]
sig_color  = {"GHG": "#a84f9e", "AA": "#f69243", "NAT": "#00994c"}
sig_marker = {"GHG": "^",       "AA": "s",       "NAT": "o"}

scen_idx = {name: i for i, name in enumerate(SCEN_ORDER)}
idx_all = scen_idx["ALL"]
idx_for_trend = {
    "GHG": scen_idx["GHG"],
    "AA":  scen_idx["AER"],
    "NAT": scen_idx["NAT"],
}

print("\n===== (single-signal scaling × trend / ALL trend) × 100 =====")
for hemi_name, row_idx, trend_arr in [("NH", 0, trend_NH), ("SH", 1, trend_SH)]:
    trend_all = trend_arr[idx_all]
    print(f"\n{hemi_name}:")
    for col_idx, sig in enumerate(SIGNALS):
        idx_sig = idx_for_trend[sig]
        trend_sig = trend_arr[idx_sig]
        beta_single = beta_b[row_idx, col_idx]
        if np.isfinite(trend_all) and trend_all != 0.0:
            value = beta_single * trend_sig / trend_all * 100.0
        else:
            value = np.nan
        print(f"  {sig}: {value:.2f}")

WSPACE = 0.08
HSPACE = 0.28

fig = plt.figure(figsize=(13, 7.5))
outer_gs = GridSpec(
    2, 1, figure=fig,
    hspace=HSPACE,
    height_ratios=[1.0, 1.0]
)

row1_gs = outer_gs[0, 0].subgridspec(1, 2, width_ratios=[0.7, 1.3], wspace=0.08)

ax_ref = fig.add_subplot(row1_gs[0, 0])

ax_bar = fig.add_subplot(row1_gs[0, 1])

cases_ref_right = [c for c in cases_ref if c != "REF"]

nh_diff_right = [results_refchg[c]["NH_pct"] for c in cases_ref_right]
sh_diff_right = [results_refchg[c]["SH_pct"] for c in cases_ref_right]

x_ref = np.arange(len(cases_ref_right))
width_ref = 0.33

ax_ref.bar(
    x_ref - width_ref/2, nh_diff_right,
    width=width_ref,
    color=COLOR_NH,
    alpha=0.9,
    label="NH"
)
ax_ref.bar(
    x_ref + width_ref/2, sh_diff_right,
    width=width_ref,
    color=COLOR_SH,
    alpha=0.9,
    label="SH"
)

ax_ref.set_xticks(x_ref)
ax_ref.set_xticklabels([label_map_ref[c] for c in cases_ref_right], fontsize=FS_TICK)
ax_ref.set_ylabel("Relative contribution [%]", fontsize=FS_LABEL)
ax_ref.yaxis.set_label_coords(-0.11, 0.49)

ax_ref.set_ylim(BAR1_YLIM_RIGHT[0], BAR1_YLIM_RIGHT[1])
ax_ref.set_yticks(BAR1_YTICKS_RIGHT)
ax_ref.set_yticklabels([f"{v:.0f}" for v in BAR1_YTICKS_RIGHT], fontsize=FS_TICK)

ax_ref.tick_params(axis='x', labelsize=FS_TICK)
ax_ref.tick_params(axis='y', labelsize=FS_TICK, right=False, labelright=False)
ax_ref.axhline(0, color="black", linewidth=1.2, linestyle="-", zorder=1)

ax_ref.spines["left"].set_visible(True)
ax_ref.spines["right"].set_visible(True)
ax_ref.spines["top"].set_visible(True)
ax_ref.spines["bottom"].set_visible(True)

rect_nh_a = Patch(facecolor=COLOR_NH, edgecolor='none', label="NH")
rect_sh_a = Patch(facecolor=COLOR_SH, edgecolor='none', label="SH")
ax_ref.legend(
    handles=[rect_nh_a, rect_sh_a],
    frameon=False,
    fontsize=FS_TICK-2,
    loc="upper left",
    ncol=1,
    bbox_to_anchor=(0.72, 1.01)
)

ax_ref.text(
    -0.223, 1.03, "a",
    transform=ax_ref.transAxes,
    fontsize=22, fontweight="bold",
    ha="left", va="bottom",
)

ax_bar.tick_params(
    axis='y',
    which='both',
    left=False,
    right=False,
    labelleft=False,
    labelright=False,
    length=0
)
ax_bar.spines["left"].set_visible(True)
ax_bar.spines["right"].set_visible(True)
ax_bar.spines["top"].set_visible(True)
ax_bar.spines["bottom"].set_visible(True)

ax_bar2 = ax_bar.twinx()

x = np.arange(len(SCEN_ORDER))
width = 0.32

bars_nh_tr = ax_bar2.bar(
    x - width/2, trend_NH, width=width,
    color=COLOR_NH, label="NH"
)
bars_sh_tr = ax_bar2.bar(
    x + width/2, trend_SH, width=width,
    color=COLOR_SH, label="SH"
)

# print(trend_NH)
# print(trend_SH)

ax_bar.set_xticks(x)
ax_bar.set_xticklabels(SCEN_LABEL, fontsize=FS_TICK)

ax_bar2.set_ylabel("Trend [%/C]", fontsize=FS_LABEL)
ax_bar2.set_ylim(BAR_YLIM_RIGHT[0], BAR_YLIM_RIGHT[1])
ax_bar2.set_yticks(BAR_YTICKS_RIGHT)
ax_bar2.set_yticklabels([f"{v:g}" for v in BAR_YTICKS_RIGHT], fontsize=FS_TICK)
ax_bar2.tick_params(axis='y', labelsize=FS_TICK, left=False, labelleft=False,
                    right=True, labelright=True)
ax_bar2.axhline(0, color="black", linewidth=1.2, linestyle="-", zorder=1)

ax_bar2.spines["right"].set_visible(True)
ax_bar2.spines["top"].set_visible(True)
ax_bar2.spines["left"].set_visible(False)

ax_bar.text(
    -0.04, 1.03, "b",
    transform=ax_bar.transAxes,
    fontsize=22, fontweight="bold",
    ha="left", va="bottom",
)

x_NH = x - width/2
x_SH = x + width/2

ax_bar2.scatter(x_NH[sig_NH],  trend_NH[sig_NH],  marker='o', s=80, color='black', zorder=5)
ax_bar2.scatter(x_NH[~sig_NH], trend_NH[~sig_NH], marker='o', s=80,
                facecolors='white', edgecolors='black', linewidths=1.5, zorder=5)

ax_bar2.scatter(x_SH[sig_SH],  trend_SH[sig_SH],  marker='o', s=80, color='black', zorder=5)
ax_bar2.scatter(x_SH[~sig_SH], trend_SH[~sig_SH], marker='o', s=80,
                facecolors='white', edgecolors='black', linewidths=1.5, zorder=5)

rect_nh = Patch(facecolor=COLOR_NH, edgecolor='none')
rect_sh = Patch(facecolor=COLOR_SH, edgecolor='none')

handle_sig = Line2D([0], [0], marker='o', linestyle='none',
                    markersize=9, markerfacecolor='black', markeredgecolor='black')
handle_nsig = Line2D([0], [0], marker='o', linestyle='none',
                     markersize=9, markerfacecolor='white', markeredgecolor='black',
                     markeredgewidth=1.2)

ax_bar2.legend(
    [rect_nh, rect_sh],
    ["NH", "SH"],
    frameon=False,
    fontsize=FS_TICK-2,
    loc="upper left",
    ncol=2,
    bbox_to_anchor=(0.65, 0.15)
)

inner_gs = outer_gs[1, 0].subgridspec(1, 2, wspace=0.08)
ax_scale_nh = fig.add_subplot(inner_gs[0, 0])
ax_scale_sh = fig.add_subplot(inner_gs[0, 1])

def draw_scaling_panel(ax, r_idx, title):
    x_single = np.arange(3)
    x_multi  = x_single + 4
    capw = 0.2

    means_single = beta_b[r_idx, :]
    lo_single    = beta_inf_b[r_idx, :]
    hi_single    = beta_sup_b[r_idx, :]

    means_multi = beta_a[r_idx, :]
    lo_multi    = beta_inf_a[r_idx, :]
    hi_multi    = beta_sup_a[r_idx, :]

    ax.axhline(0, color='0.7', lw=0.5, ls='--')
    ax.axhline(1, color='0.7', lw=0.5, ls='--')

    region = "NH" if r_idx == 0 else "SH"
    ymin, ymax = SCALE_YLIM[region]
    yticks = SCALE_YTICKS[region]

    for i in range(3):
        sig = SIGNALS[i]
        color = sig_color[sig]
        marker = sig_marker[sig]

        xs = x_single[i]
        mean = float(means_single[i])
        lo   = float(lo_single[i])
        hi   = float(hi_single[i])
        lo, hi = (min(lo, hi), max(lo, hi))

        ax.vlines(xs, lo, hi, colors=color, linewidth=11, alpha=0.5)
        ax.plot([xs - capw, xs + capw], [lo, lo], color=color, lw=1, alpha=0.5)
        ax.plot([xs - capw, xs + capw], [hi, hi], color=color, lw=1, alpha=0.5)
        ax.plot(xs, mean, marker, ms=13, color=color, mec='black', mew=1.0)

    for i in range(3):
        sig = SIGNALS[i]
        color = sig_color[sig]
        marker = sig_marker[sig]

        xm = x_multi[i]
        mean = float(means_multi[i])
        lo   = float(lo_multi[i])
        hi   = float(hi_multi[i])
        lo, hi = (min(lo, hi), max(lo, hi))

        ax.vlines(xm, lo, hi, colors=color, linewidth=11, alpha=0.5)
        ax.plot([xm - capw, xm + capw], [lo, lo], color=color, lw=1, alpha=0.5)
        ax.plot([xm - capw, xm + capw], [hi, hi], color=color, lw=1, alpha=0.5)
        ax.plot(xm, mean, marker, ms=13, color=color, mec='black', mew=1.0)

    ax.axvline(3.0, color='0.5', lw=1.2, ls='--')

    xticks   = np.concatenate([x_single, x_multi])
    xtlabels = ["GHG", "AA", "NAT", "GHG", "AA", "NAT"]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtlabels, fontsize=FS_TICK)

    ax.set_ylim(ymin, ymax)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{v:g}" for v in yticks], fontsize=FS_TICK)

    ax.set_ylabel("Scaling factor", fontsize=FS_LABEL)
    if title == "NH":
        ax.yaxis.set_label_coords(-0.08, 0.5)
    else:
        ax.yaxis.set_label_coords(1.079, 0.5)
    ax.set_title(title, fontsize=FS_TITLE-2, pad=4, loc="left")

    ax.tick_params(axis='both', labelsize=FS_TICK)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(True)

draw_scaling_panel(ax_scale_nh, 0, "NH")
draw_scaling_panel(ax_scale_sh, 1, "SH")

ax_scale_nh.text(-0.155, 1.05, "c", transform=ax_scale_nh.transAxes,
                 ha="left", va="bottom", fontsize=22, fontweight="bold")
ax_scale_nh.text(0.360, 0.02, "Single-signal", transform=ax_scale_nh.transAxes,
                 ha="center", va="bottom", fontsize=FS_TICK+3)
ax_scale_nh.text(0.865, 0.02, "Three-signal", transform=ax_scale_nh.transAxes,
                 ha="center", va="bottom", fontsize=FS_TICK+3)

ax_scale_sh.text(0.360, 0.02, "Single-signal", transform=ax_scale_sh.transAxes,
                 ha="center", va="bottom", fontsize=FS_TICK+3)
ax_scale_sh.text(0.865, 0.02, "Three-signal", transform=ax_scale_sh.transAxes,
                 ha="center", va="bottom", fontsize=FS_TICK+3)

ax_scale_sh.yaxis.set_label_position("right")
ax_scale_sh.yaxis.tick_right()
ax_scale_sh.tick_params(axis='y', labelleft=False, left=False, labelright=True, right=True)

plt.tight_layout()
plt.savefig("D:/code/comfort-area/fig/v7/fig3.pdf", dpi=600, bbox_inches="tight")