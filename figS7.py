# -*- coding: utf-8 -*-
import os
import re
import glob
import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats
from typing import Optional, Tuple, List, Dict
from pathlib import Path

mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 14

mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'Times New Roman'
mpl.rcParams['mathtext.it'] = 'Times New Roman:italic'
mpl.rcParams['mathtext.bf'] = 'Times New Roman:bold'

BASE_DIR = Path(__file__).resolve().parent.parent

FS_TITLE  = 14
FS_LABEL  = 12
FS_TICK   = 12
FS_LEGEND = 10

AREA_NH = 9933.0
AREA_SH = 4970.0
AREA_GL = AREA_NH + AREA_SH

LAT_EDGE = 67.5
START_YEAR = 2021

THRESH_DAYS_DEFAULT = 97.0

MODEL_THRESH_DAYS_BY_SCEN: Dict[str, Dict[str, float]] = {
    "ssp585": {
        "ACCESS-CM2":     91.0,
        "ACCESS-ESM1-5":  84.45,
        "CanESM5":        87.2,
        "FGOALS-g3":      95.45,
        "GFDL-ESM4":      91.4,
        "IPSL-CM6A-LR":   90.45,
        "NorESM2-LM":     93.45,
    },
    "ssp370": {
        "ACCESS-CM2":     91.9,
        "ACCESS-ESM1-5":  86.05,
        "CanESM5":        86.45,
        "FGOALS-g3":      94.4,
        "GFDL-ESM4":      90.75,
        "IPSL-CM6A-LR":   90.45,
        "NorESM2-LM":     92.05,
    }
}

MIN_MODELS_COMF  = 4
MIN_MODELS_VALID = 4

AUTO_E_YLIM = False
E_NH_LIM = (7, 22)
E_SH_LIM = (9.5, 16)
E_NH_TICKS = [7, 12, 17, 22]
E_SH_TICKS = [10, 12, 14, 16]
E_NH_TICKLABELS = ['7', '12', '17', '22']
E_SH_TICKLABELS = ['10', '12', '14', '16']

colors_region = {
    "NH": {"ssp370": "#f393be", "ssp585": "#ee3b73"},
    "SH": {"ssp370": "#00a4d2", "ssp585": "#0078bf"},
}

BASE_CMIP_585 = r"D:\code\comfort-area\data\v5\CMIP6\ssp585"
BASE_CMIP_370 = r"D:\code\comfort-area\data\v5\CMIP6\ssp370"

BASE_TAS_585 = r"D:\code\comfort-area\data\CMIP6\models\ssp585_zhang"
BASE_TAS_370 = r"D:\code\comfort-area\data\CMIP6\models\ssp370_zhang"

BASE_ERA5 = r"D:\code\comfort-area\data\ERA5"
BASE_ERA5_V5 = r"D:\code\comfort-area\data\v5\ERA5"

LANDSEA_FILE_NH = os.path.join(BASE_ERA5, "landsea_NH.nc")
LANDSEA_FILE_SH = os.path.join(BASE_ERA5, "landsea_SH.nc")
if (not os.path.exists(LANDSEA_FILE_NH)) and os.path.exists(os.path.join(BASE_ERA5_V5, "landsea_NH.nc")):
    LANDSEA_FILE_NH = os.path.join(BASE_ERA5_V5, "landsea_NH.nc")
if (not os.path.exists(LANDSEA_FILE_SH)) and os.path.exists(os.path.join(BASE_ERA5_V5, "landsea_SH.nc")):
    LANDSEA_FILE_SH = os.path.join(BASE_ERA5_V5, "landsea_SH.nc")

AREA_FILE = r"D:\code\comfort-area\data\ERA5\aera_25x25_nh_0_70.nc"
if not os.path.exists(AREA_FILE):
    alt = r"D:\code\comfort-area\data\v5\ERA5\aera_25x25_nh_0_70.nc"
    if os.path.exists(alt):
        AREA_FILE = alt

OUT_PNG = r"D:\code\comfort-area\fig\v7\figS7.png"

wanted_models = [
    "ACCESS-CM2",
    "ACCESS-ESM1-5",
    "CanESM5",
    "FGOALS-g3",
    "GFDL-ESM4",
    "IPSL-CM6A-LR",
    "NorESM2-LM"
]

def get_model_thresh(model_name: str,
                     scen: Optional[str] = None,
                     default: float = THRESH_DAYS_DEFAULT) -> float:
    if scen is None:
        return float(default)
    dd = MODEL_THRESH_DAYS_BY_SCEN.get(str(scen), {})
    return float(dd.get(str(model_name), default))

def print_model_threshold_check(models: List[str], tag: str = "", scen: Optional[str] = None):
    if tag:
        print(f"[CHECK] {tag} 模式阈值表：")
    else:
        print("[CHECK] 模式阈值表：")

    missing = []
    for m in models:
        thr = get_model_thresh(m, scen=scen, default=THRESH_DAYS_DEFAULT)
        hit = (scen in MODEL_THRESH_DAYS_BY_SCEN) and (m in MODEL_THRESH_DAYS_BY_SCEN.get(scen, {}))
        print(f"  - {m:<14s}  THRESH={thr:.3f}  ({'DICT' if hit else 'DEFAULT'})")
        if not hit:
            missing.append(m)

    if missing:
        print("[WARNING] 以下模式未在阈值表中找到，将使用 THRESH_DAYS_DEFAULT：")
        print("          " + ", ".join(missing))
    print("")

def _parse_model_from_filename(fname: str, scen: str) -> Optional[str]:
    base = os.path.basename(fname)
    m = re.match(rf"(.+?)_{scen}_.+\.nc$", base)
    if m:
        return m.group(1)
    if f"_{scen}_" in base:
        return base.split(f"_{scen}_")[0]
    return None

def _get_da_main(ds: xr.Dataset, prefer_names=("comfortDays", "shushiDays", "shushidays")) -> xr.DataArray:
    for vn in prefer_names:
        if vn in ds.data_vars:
            return ds[vn]
    return ds[list(ds.data_vars.keys())[0]]

def _apply_fill_to_nan(da: xr.DataArray, default_fill=32767.0) -> xr.DataArray:
    mv = da.attrs.get("missing_value", da.attrs.get("_FillValue", default_fill))
    try:
        return da.where(da != mv)
    except Exception:
        return da

def _to_monthly_time_index(da: xr.DataArray, base_year=2015) -> xr.DataArray:
    if ("year" in da.dims) and ("month" in da.dims):
        da2 = da.transpose("year", "month", "lat", "lon")
        years  = da2["year"].values.astype(int)
        months = da2["month"].values.astype(int)

        da_stk = da2.stack(time=("year", "month")).transpose("time", "lat", "lon")
        t = ((years[:, None] - base_year) * 12 + (months[None, :] - 1)).reshape(-1)
        da_stk = da_stk.assign_coords(time=("time", t)).sortby("time")
        return da_stk

    if "time" in da.dims:
        try:
            return da.sortby("time")
        except Exception:
            return da

    raise ValueError(f"无法识别月序列维度：da.dims={da.dims}")

def load_cmip6_comfortdays_scenario(
    scen: str,
    base_dir: str,
    wanted_models: List[str],
) -> Tuple[xr.DataArray, List[str]]:
    patterns = [
        os.path.join(base_dir, f"*_{scen}_*_shushiDays_201501-210012.nc"),
    ]

    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    files = sorted(list(set(files)))

    if len(files) == 0:
        raise FileNotFoundError(
            f"[ERROR] 在目录 {base_dir} 下找不到 {scen} 的 comfortDays/shushidays 文件。\n"
            f"        尝试过的 pattern:\n" + "\n".join([f"          - {p}" for p in patterns])
        )

    das = []
    models_in_data = []

    for f in files:
        model = _parse_model_from_filename(f, scen)
        if model is None:
            continue
        if model not in wanted_models:
            continue

        ds = xr.open_dataset(f, decode_times=False)
        da = _get_da_main(ds)
        da = _apply_fill_to_nan(da, default_fill=32767.0)
        da = _to_monthly_time_index(da, base_year=2015)
        da = da.expand_dims(model=[model])
        das.append(da)
        models_in_data.append(model)
        ds.close()

    if len(das) == 0:
        raise RuntimeError(
            f"[ERROR] 找到了文件但没有读入任何 wanted_models。wanted_models={wanted_models}"
        )

    all_days = xr.concat(das, dim="model", coords="minimal", compat="override", join="override")

    return all_days, models_in_data

def monthly_to_calendar_year_annual(all_mon: xr.DataArray, base_year=2015) -> xr.DataArray:
    t = all_mon["time"].values.astype(int)
    years = base_year + (t // 12)
    months = (t % 12) + 1

    da = all_mon.assign_coords(year=("time", years), month=("time", months))
    ann = da.groupby("year").sum(dim="time", skipna=True)
    return ann

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

def read_lsm(landsea_path):
    ds = xr.open_dataset(landsea_path)
    lsm = ds["lsm"]

    ren = {}
    for d in lsm.dims:
        dl = d.lower()
        if dl in ("latitude", "y"):
            ren[d] = "lat"
        if dl in ("longitude", "x"):
            ren[d] = "lon"
    if ren:
        lsm = lsm.rename(ren)

    ds.close()
    return lsm

def landmask_like(da_latlon: xr.DataArray, lsm_nh: xr.DataArray, lsm_sh: xr.DataArray, lat_edge=LAT_EDGE) -> xr.DataArray:
    lat = da_latlon["lat"]
    lon = da_latlon["lon"]

    try:
        lsm_nh_i = lsm_nh.interp(lat=lat, lon=lon, method="nearest")
    except Exception:
        lsm_nh_i = lsm_nh.reindex(lat=lat, lon=lon, method="nearest")

    try:
        lsm_sh_i = lsm_sh.interp(lat=lat, lon=lon, method="nearest")
    except Exception:
        lsm_sh_i = lsm_sh.reindex(lat=lat, lon=lon, method="nearest")

    mask = xr.DataArray(
        np.zeros((lat.size, lon.size), dtype=bool),
        coords={"lat": lat, "lon": lon},
        dims=("lat", "lon")
    )

    lat_nh = lat.where((lat >= 0.0) & (lat <= lat_edge), drop=True)
    if lat_nh.size > 0:
        mask.loc[dict(lat=lat_nh)] = (lsm_nh_i.sel(lat=lat_nh) > 0.5).values

    lat_sh = lat.where((lat < 0.0) & (lat >= -lat_edge), drop=True)
    if lat_sh.size > 0:
        mask.loc[dict(lat=lat_sh)] = (lsm_sh_i.sel(lat=lat_sh) > 0.5).values

    return mask

def annual_comf_area_series_from_monthly_vote(all_days_scen,
                                              area_da,
                                              lsm_nh, lsm_sh,
                                              scen_label: str,
                                              thresh_days_default=THRESH_DAYS_DEFAULT,
                                              lat_edge=LAT_EDGE,
                                              base_year=2015,
                                              min_comf=MIN_MODELS_COMF,
                                              min_valid=MIN_MODELS_VALID):
    ann = monthly_to_calendar_year_annual(all_days_scen, base_year=base_year)
    years = ann["year"].values.astype(int)

    landmask_y = landmask_like(ann.isel(model=0, year=0), lsm_nh, lsm_sh, lat_edge=lat_edge)
    ann_land = ann.where(landmask_y)

    model_names = ann_land["model"].values.astype(str)
    thr_vec = np.array([get_model_thresh(m, scen=scen_label, default=thresh_days_default) for m in model_names], dtype=float)
    thr_da = xr.DataArray(thr_vec, dims=("model",), coords={"model": ann_land["model"]})

    mask_model = xr.where(np.isfinite(ann_land) & (ann_land >= thr_da), 1.0, 0.0)
    mask_model = mask_model.where(np.isfinite(ann_land))

    count_valid = mask_model.notnull().sum(dim="model")
    count_comf  = mask_model.fillna(0.0).sum(dim="model")
    consensus_bool = (count_valid >= min_valid) & (count_comf >= min_comf)
    all_nan = (count_valid == 0)

    consensus01 = xr.where(consensus_bool, 1.0, 0.0).where(~all_nan)

    cons_nh = _band_sel_strict(consensus01, 0.0,  lat_edge)
    cons_sh = _band_sel_strict(consensus01, 0.0, -lat_edge)

    nh_area = area_da.where(cons_nh >= 0.5).sum(dim=("lat","lon"), skipna=True)
    sh_area = area_da.where(cons_sh >= 0.5).sum(dim=("lat","lon"), skipna=True)

    nh_frac = (nh_area / AREA_GL * 100.0).values
    sh_frac = (sh_area / AREA_GL * 100.0).values
    gl_frac = ((nh_area + sh_area) / AREA_GL * 100.0).values

    return years, nh_frac, sh_frac, gl_frac

def period_trend_with_p(years, data, start, end, per="century"):
    years = np.asarray(years)
    data  = np.asarray(data)
    mask = (years >= start) & (years <= end)
    x = years[mask]
    y = data[mask]
    if len(x) < 3:
        return np.nan, np.nan
    res = stats.linregress(x, y)
    slope = res.slope
    pval  = res.pvalue
    if per == "century":
        slope *= 100.0
    return slope, pval

def add_future_shading(axis, years, vals, color, alpha=0.10, std_scale=1.0):
    years = np.asarray(years)
    vals  = np.asarray(vals)
    mask_fut = years >= START_YEAR
    years_fut = years[mask_fut]
    vals_fut  = vals[mask_fut]
    if years_fut.size > 0:
        std_fut = std_scale * np.nanstd(vals_fut)
        axis.fill_between(
            years_fut,
            vals_fut - std_fut,
            vals_fut + std_fut,
            color=color,
            alpha=alpha,
            linewidth=0,
            zorder=1
        )

if not os.path.exists(AREA_FILE):
    raise FileNotFoundError(f"找不到面积文件：{AREA_FILE}")

ds_area = xr.open_dataset(AREA_FILE)
area_da = ds_area["area"]
dims = list(area_da.dims)
lat_dim, lon_dim = dims[0], dims[1]
area_da = area_da.rename({lat_dim: "lat", lon_dim: "lon"})
ds_area.close()

lsm_nh = read_lsm(LANDSEA_FILE_NH)
lsm_sh = read_lsm(LANDSEA_FILE_SH)

print_model_threshold_check(wanted_models, tag="wanted_models ssp585", scen="ssp585")
print_model_threshold_check(wanted_models, tag="wanted_models ssp370", scen="ssp370")

all_days_585, models_585 = load_cmip6_comfortdays_scenario("ssp585", BASE_CMIP_585, wanted_models)
all_days_370, models_370 = load_cmip6_comfortdays_scenario("ssp370", BASE_CMIP_370, wanted_models)

years_585_ts, nh_frac_585_ts, sh_frac_585_ts, gl_frac_585_ts = annual_comf_area_series_from_monthly_vote(
    all_days_585, area_da, lsm_nh, lsm_sh, scen_label="ssp585",
    thresh_days_default=THRESH_DAYS_DEFAULT, lat_edge=LAT_EDGE, base_year=2015
)
years_370_ts, nh_frac_370_ts, sh_frac_370_ts, gl_frac_370_ts = annual_comf_area_series_from_monthly_vote(
    all_days_370, area_da, lsm_nh, lsm_sh, scen_label="ssp370",
    thresh_days_default=THRESH_DAYS_DEFAULT, lat_edge=LAT_EDGE, base_year=2015
)

scen_list = ["ssp370", "ssp585"]
frac_series = {
    "NH": {"ssp585": (years_585_ts, nh_frac_585_ts), "ssp370": (years_370_ts, nh_frac_370_ts)},
    "SH": {"ssp585": (years_585_ts, sh_frac_585_ts), "ssp370": (years_370_ts, sh_frac_370_ts)},
}

trend_label = {"NH": {}, "SH": {}}

fig, ax_ts_l = plt.subplots(figsize=(9, 3.5))
ax_ts_r = ax_ts_l.twinx()

for scen in scen_list:
    years_nh_full, vals_nh_full = frac_series["NH"][scen]
    mask_nh = years_nh_full >= START_YEAR
    years_nh = years_nh_full[mask_nh]
    vals_nh  = vals_nh_full[mask_nh]
    color_nh = colors_region["NH"][scen]

    tr_nh, p_nh = period_trend_with_p(years_nh_full, vals_nh_full, START_YEAR, 2100, per="century")
    star_nh = "*" if (p_nh < 0.05) else ""
    trend_label["NH"][scen] = f"NH {scen.upper()}: {tr_nh:.2f}%/C{star_nh}"

    add_future_shading(ax_ts_l, years_nh_full, vals_nh_full, color=color_nh, alpha=0.08, std_scale=1.0)

    ax_ts_l.plot(
        years_nh, vals_nh,
        linestyle="-",
        color=color_nh,
        linewidth=3 if scen == "ssp585" else 2,
        zorder=2
    )

    years_sh_full, vals_sh_full = frac_series["SH"][scen]
    mask_sh = years_sh_full >= START_YEAR
    years_sh = years_sh_full[mask_sh]
    vals_sh  = vals_sh_full[mask_sh]
    color_sh = colors_region["SH"][scen]

    tr_sh, p_sh = period_trend_with_p(years_sh_full, vals_sh_full, START_YEAR, 2100, per="century")
    star_sh = "*" if (p_sh < 0.05) else ""
    trend_label["SH"][scen] = f"SH {scen.upper()}: {tr_sh:.2f}%/C{star_sh}"

    add_future_shading(ax_ts_r, years_sh_full, vals_sh_full, color=color_sh, alpha=0.08, std_scale=0.6)

    ax_ts_r.plot(
        years_sh, vals_sh,
        linestyle="-",
        color=color_sh,
        linewidth=3 if scen == "ssp585" else 2,
        zorder=2
    )

ax_ts_l.set_xlim(START_YEAR, 2100)

if not AUTO_E_YLIM:
    ax_ts_l.set_ylim(*E_NH_LIM)
    ax_ts_r.set_ylim(*E_SH_LIM)

    ax_ts_l.set_yticks(E_NH_TICKS)
    ax_ts_l.set_yticklabels(E_NH_TICKLABELS, fontsize=FS_TICK)

    ax_ts_r.set_yticks(E_SH_TICKS)
    ax_ts_r.set_yticklabels(E_SH_TICKLABELS, fontsize=FS_TICK)

custom_xticks = [2021, 2040, 2060, 2080, 2100]
ax_ts_l.set_xticks(custom_xticks)
ax_ts_l.set_xticklabels([str(y) for y in custom_xticks], fontsize=FS_TICK)

ax_ts_l.tick_params(axis='x', labelsize=FS_TICK, width=1.0, colors='k')
ax_ts_l.tick_params(axis='y', labelsize=FS_TICK, width=1.0, colors=colors_region["NH"]["ssp585"])
ax_ts_l.set_ylabel("Percentage (NH) [%]", fontsize=FS_LABEL)
ax_ts_l.yaxis.set_label_coords(-0.07, 0.5)
ax_ts_l.yaxis.label.set_color(colors_region["NH"]["ssp585"])

ax_ts_r.set_ylabel("Percentage (SH) [%]", fontsize=FS_LABEL)
ax_ts_r.tick_params(axis='y', labelsize=FS_TICK, width=1.0, colors=colors_region["SH"]["ssp585"])
ax_ts_r.yaxis.label.set_color(colors_region["SH"]["ssp585"])

handles_585, labels_585 = [], []
handles_370, labels_370 = [], []

for reg in ["NH", "SH"]:
    for scen in scen_list:
        color = colors_region[reg][scen]
        label = trend_label[reg][scen]
        if scen == "ssp585":
            handles_585.append(plt.Line2D([0], [0], color=color, lw=3))
            labels_585.append(label)
        else:
            handles_370.append(plt.Line2D([0], [0], color=color, lw=2))
            labels_370.append(label)

leg1 = ax_ts_l.legend(handles_585, labels_585, frameon=False, ncol=2, fontsize=FS_LEGEND,
                      loc="lower left", bbox_to_anchor=(0.00, 0.10))
ax_ts_l.add_artist(leg1)
ax_ts_l.legend(handles_370, labels_370, frameon=False, ncol=2, fontsize=FS_LEGEND,
               loc="lower left", bbox_to_anchor=(0.00, 0.00))

ax_ts_l.text(-0.10, 1.04, "", transform=ax_ts_l.transAxes,
             ha="right", va="center", fontsize=FS_TITLE + 6, fontweight="bold")

fig.subplots_adjust(left=0.12, right=0.88, bottom=0.15, top=0.93)

plt.savefig(OUT_PNG, dpi=800, bbox_inches='tight', pad_inches=0.05)