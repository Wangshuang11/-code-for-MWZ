# -*- coding: utf-8 -*-
import os
import re
import glob
import json
import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Wedge
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy import stats
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

from typing import Optional, Tuple, List, Dict


mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 14

mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'Times New Roman'
mpl.rcParams['mathtext.it'] = 'Times New Roman:italic'
mpl.rcParams['mathtext.bf'] = 'Times New Roman:bold'
mpl.rcParams['hatch.linewidth'] = 1

FS_BASE   = 14
FS_TITLE  = 14
FS_LABEL  = 12
FS_TICK   = 12
FS_LEGEND = 10
FS_PIE_PERCENT = 8
FS_PIE_INNER   = 9
FS_MAP_LONLAT  = 12

AREA_NH = 9933.0
AREA_SH = 4970.0
AREA_GL = AREA_NH + AREA_SH

COLOR_NH = '#f393be'
COLOR_SH = '#00a4d2'
COLOR_G  = '#7bc67b'

OCEAN_COLOR = "#e0f2fc"

LEVELS = [2.0, 3.0, 4.0]

THRESH_DAYS_DEFAULT = 97.0

THRESH_DAYS_ERA5 = 97.0

MODEL_THRESH_DAYS: Dict[str, float] = {
    "ACCESS-CM2":     99.6,
    "ACCESS-ESM1-5":  91.05,
    "CanESM5":        91.4,
    "FGOALS-g3":      99.2,
    "GFDL-ESM4":      96.6,
    "IPSL-CM6A-LR":   97.075,
    "NorESM2-LM":     101.65,
}

THRESH_DAYS = THRESH_DAYS_DEFAULT

MIN_MODELS_COMF  = 4
MIN_MODELS_VALID = 4

SMOOTH_MODEL_CONSENSUS = False

AUTO_D_YLIM = False
D_NH_LIM = (6, 14)
D_SH_LIM = (11.8, 15)
D_NH_TICKS = [6, 8, 10, 12, 14]
D_SH_TICKS = [12, 13, 14, 15]
D_NH_TICKLABELS = ['6', '8', '10', '12', '14']
D_SH_TICKLABELS = ['12', '13', '14', '15']

AUTO_E_YLIM = False
E_NH_LIM = (3, 19)
E_SH_LIM = (8, 16)
E_NH_TICKS = [3, 7, 11, 15,19]
E_SH_TICKS = [8, 10, 12, 14, 16]
E_NH_TICKLABELS = ['3', '7', '11', '15', '19']
E_SH_TICKLABELS = ['8', '10', '12', '14', '16']

AUTO_F_YLIM = False
F_LEFT_LIM  = (-20, 20)
F_RIGHT_LIM = (-13, 13)
F_LEFT_TICKS  = [0, 5, 10, 15, 20]
F_RIGHT_TICKS = [-12,-9,-6, -3, 0]
F_LEFT_TICKLABELS  = ['0', '5', '10', '15', '20']
F_RIGHT_TICKLABELS = ['-12', '-9', '-6', '-3', '0']

EXPORT_MME_SEG_ANN = True
EXPORT_DIR_MME_SEG_ANN = r"D:\code\comfort-area\data\v5\cmip6_warming_levels"

EXPORT_FIG_E_TXT = True

def get_model_thresh(model_name: str, default: float = THRESH_DAYS_DEFAULT) -> float:
    return float(MODEL_THRESH_DAYS.get(str(model_name), default))


def print_model_threshold_check(models: List[str], tag: str = ""):
    missing = []
    for m in models:
        thr = get_model_thresh(m, default=THRESH_DAYS_DEFAULT)
        hit = (m in MODEL_THRESH_DAYS)
        # print(f"  - {m:<14s}  THRESH={thr:.3f}  ({'DICT' if hit else 'DEFAULT'})")
        if not hit:
            missing.append(m)


def smooth9(field2d: np.ndarray) -> np.ndarray:
    tmp = np.pad(field2d, ((1, 1), (0, 0)), mode='edge')
    arr = np.pad(tmp,      ((0, 0), (1, 1)), mode='wrap')
    s = (
        arr[0:-2, 0:-2] + arr[0:-2, 1:-1] + arr[0:-2, 2:] +
        arr[1:-1, 0:-2] + arr[1:-1, 1:-1] + arr[1:-1, 2:] +
        arr[2:,   0:-2] + arr[2:,   1:-1] + arr[2:,   2:]
    ) / 9.0
    return s


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

    raise ValueError(f"无法识别月序列维度：da.dims={da.dims}（需要 time 或 year+month）")


def load_cmip6_comfortdays_scenario(
    scen: str,
    base_dir: str,
    wanted_models: List[str],
) -> Tuple[xr.DataArray, List[str]]:

    patterns = [
        os.path.join(base_dir, f"*_{scen}_*_shushidays_201501-210012.nc"),
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
        da = _to_monthly_time_index(da, base_year=2015)      # (time,lat,lon)
        da = da.expand_dims(model=[model])
        das.append(da)
        models_in_data.append(model)
        ds.close()

    if len(das) == 0:
        raise RuntimeError(
            f"[ERROR] 找到了文件但没有读入任何 wanted_models。wanted_models={wanted_models}"
        )

    all_days = xr.concat(das, dim="model", coords="minimal", compat="override", join="override")

    print_model_threshold_check(list(all_days["model"].values.astype(str)), tag=scen)

    return all_days, models_in_data


def monthly_segment_to_climyear_annual(seg_mon: xr.DataArray) -> xr.DataArray:
    seg_ann = seg_mon.coarsen(time=12, boundary="trim").sum()
    return seg_ann


def monthly_to_calendar_year_annual(all_mon: xr.DataArray, base_year=2015) -> xr.DataArray:
    t = all_mon["time"].values.astype(int)
    years = base_year + (t // 12)
    months = (t % 12) + 1

    da = all_mon.assign_coords(year=("time", years), month=("time", months))
    ann = da.groupby("year").sum(dim="time", skipna=True)
    return ann


def _year_dim(da: xr.DataArray) -> str:
    if "climYear" in da.dims:
        return "climYear"
    if "time" in da.dims:
        return "time"
    raise ValueError(f"[ERROR] 找不到逐年维(climYear/time)：da.dims={da.dims}")


def _isel_first_year(da: xr.DataArray) -> xr.DataArray:
    yd = _year_dim(da)
    return da.isel({yd: 0})

LAT_EDGE  = 67.5

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

BASE_ERA5 = r"D:\code\comfort-area\data\ERA5"
BASE_ERA5_V5 = r"D:\code\comfort-area\data\v5\ERA5"

LANDSEA_FILE_NH = os.path.join(BASE_ERA5, "landsea_NH.nc")
LANDSEA_FILE_SH = os.path.join(BASE_ERA5, "landsea_SH.nc")
if (not os.path.exists(LANDSEA_FILE_NH)) and os.path.exists(os.path.join(BASE_ERA5_V5, "landsea_NH.nc")):
    LANDSEA_FILE_NH = os.path.join(BASE_ERA5_V5, "landsea_NH.nc")
if (not os.path.exists(LANDSEA_FILE_SH)) and os.path.exists(os.path.join(BASE_ERA5_V5, "landsea_SH.nc")):
    LANDSEA_FILE_SH = os.path.join(BASE_ERA5_V5, "landsea_SH.nc")

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

def clim_mask_from_seg_ann(seg_ann: xr.DataArray,
                           model_name: Optional[str] = None,
                           thresh_days: float = THRESH_DAYS_DEFAULT) -> xr.DataArray:
    yd = _year_dim(seg_ann)
    clim_ann = seg_ann.mean(dim=yd, skipna=True)  # (lat,lon)

    if model_name is not None:
        thr = get_model_thresh(model_name)
    else:
        thr = float(thresh_days)

    m = xr.where(np.isfinite(clim_ann) & (clim_ann >= thr), 1.0, 0.0)
    return m


def vote_consensus_mask(mask_model_stack: xr.DataArray,
                        min_comf=MIN_MODELS_COMF,
                        min_valid=MIN_MODELS_VALID) -> xr.DataArray:

    count_valid = mask_model_stack.notnull().sum(dim="model")
    count_comf  = mask_model_stack.fillna(0.0).sum(dim="model")

    consensus_bool = (count_valid >= min_valid) & (count_comf >= min_comf)

    all_nan = (count_valid == 0)
    consensus = xr.where(consensus_bool, 1.0, 0.0)
    consensus = consensus.where(~all_nan)  # all_nan -> NaN
    return consensus


def area_from_binary_mask(mask_latlon_01: xr.DataArray,
                          area_da: xr.DataArray,
                          lat_edge=LAT_EDGE) -> Tuple[float, float]:

    m = mask_latlon_01.where(np.isfinite(mask_latlon_01))
    m_nh = _band_sel_strict(m, 0.0, lat_edge)
    m_sh = _band_sel_strict(m, 0.0, -lat_edge)

    nh_area = float(area_da.where(m_nh >= 0.5).sum(dim=("lat", "lon"), skipna=True))
    sh_area = float(area_da.where(m_sh >= 0.5).sum(dim=("lat", "lon"), skipna=True))
    return nh_area, sh_area


def export_mme_seg_ann_for_gwl(gwl_all_dict, out_dir,
                              scen_label="ssp585",
                              lsm_nh=None, lsm_sh=None,
                              lat_edge=LAT_EDGE,
                              thresh_days=THRESH_DAYS_DEFAULT,
                              compress=True):

    os.makedirs(out_dir, exist_ok=True)

    for wl, lst in gwl_all_dict.items():
        if lst is None or len(lst) == 0:
            # print(f"[EXPORT] {scen_label} GWL{wl}: no data, skip.")
            continue

        seg_list = []
        model_names = []

        for da in lst:
            model = str(da["model"].values[0]) if ("model" in da.dims) else "unknown"
            seg = da.isel(model=0) if ("model" in da.dims) else da

            if (lsm_nh is not None) and (lsm_sh is not None):
                landmask_seg = landmask_like(_isel_first_year(seg), lsm_nh, lsm_sh, lat_edge=lat_edge)
                seg = seg.where(landmask_seg)

            seg_list.append(seg.expand_dims(model=[model]))
            model_names.append(model)

        seg_all = xr.concat(seg_list, dim="model", coords="minimal", compat="override", join="override")
        seg_mme = seg_all.mean(dim="model", skipna=True)
        seg_mme.name = "seg_ann_mme"

        ds_out = xr.Dataset(
            {
                "seg_ann_mme": seg_mme,
                "nmodel": xr.DataArray(len(model_names)),
            }
        )
        ds_out["seg_ann_mme"].attrs["description"] = (
            "MME of annual total comfortDays (Dec–Nov climYear) over models; "
            "computed by: per-model 20yr window -> monthly->annual sum -> MME."
        )
        ds_out["seg_ann_mme"].attrs["scenario"] = scen_label
        ds_out["seg_ann_mme"].attrs["gwl_level_degC"] = float(wl)
        ds_out["seg_ann_mme"].attrs["thresh_days_default"] = float(thresh_days)
        ds_out["seg_ann_mme"].attrs["lat_edge"] = float(lat_edge)

        used_thresh = {m: get_model_thresh(m) for m in model_names}
        ds_out.attrs["models_used"] = ", ".join(model_names)
        ds_out.attrs["model_thresh_days_json"] = json.dumps(used_thresh, ensure_ascii=False)

        out_fn = os.path.join(out_dir, f"seg_ann_mme_{scen_label}_GWL{wl:.0f}C.nc")
        if compress:
            enc = {"seg_ann_mme": {"zlib": True, "complevel": 4}}
            ds_out.to_netcdf(out_fn, encoding=enc)
        else:
            ds_out.to_netcdf(out_fn)

        # print(f"[EXPORT] saved: {out_fn}  (models={len(model_names)})")


def export_e_txt_onecol_80(out_dir: str, scen: str,
                           years, nh_frac, sh_frac,
                           start_year: int = 2021, end_year: int = 2100):

    os.makedirs(out_dir, exist_ok=True)

    years = np.asarray(years).astype(int)
    nh = np.asarray(nh_frac, dtype=float)
    sh = np.asarray(sh_frac, dtype=float)

    mask = (years >= start_year) & (years <= end_year)
    y = years[mask]
    nh_out = nh[mask]
    sh_out = sh[mask]

    need_n = end_year - start_year + 1
    if y.size != need_n:
        raise ValueError(
            f"[EXPORT e] {scen}: 期望 {start_year}-{end_year} 共 {need_n} 年(80)，"
            f"但实际得到 {y.size} 年。年份范围：{y.min() if y.size else 'NA'}-{y.max() if y.size else 'NA'}"
        )

    fn_nh = os.path.join(out_dir, f"mild_area_2021_2100_MME_{scen}_annual_nh.txt")
    fn_sh = os.path.join(out_dir, f"mild_area_2021_2100_MME_{scen}_annual_sh.txt")

    np.savetxt(fn_nh, nh_out.reshape(-1, 1), fmt="%.6f")
    np.savetxt(fn_sh, sh_out.reshape(-1, 1), fmt="%.6f")

    # print(f"[EXPORT e] saved: {fn_nh}  shape={nh_out.shape}")
    # print(f"[EXPORT e] saved: {fn_sh}  shape={sh_out.shape}")

wanted_models = [
    "ACCESS-CM2",
    "ACCESS-ESM1-5",
    "CanESM5",
    "FGOALS-g3",
    "GFDL-ESM4",
    "IPSL-CM6A-LR",
    "NorESM2-LM"
]
NMODELS_TOTAL = len(wanted_models)

print_model_threshold_check(wanted_models, tag="wanted_models")

BASE_CMIP_585 = r"D:\code\comfort-area\data\v5\CMIP6\ssp585"
all_days_585, models_585 = load_cmip6_comfortdays_scenario("ssp585", BASE_CMIP_585, wanted_models)

BASE_TAS_585 = r"D:\code\comfort-area\data\CMIP6\models\ssp585_zhang"
pattern_tas_585 = os.path.join(
    BASE_TAS_585,
    "tas_ano_base_1861-1900_*_ssp585_2015_2100_21years_slide_cos.mon.mean_2.nc"
)
tas_files_585 = sorted(glob.glob(pattern_tas_585))

tas_dict_585  = {}
year_dict_585 = {}

for f in tas_files_585:
    name = os.path.basename(f)
    try:
        model = name.split("1900_")[1].split("_ssp585")[0]
    except Exception:
        continue

    if model not in models_585:
        continue

    ds_t = xr.open_dataset(f)
    da_t = ds_t["tas_ano"].where(ds_t["tas_ano"] != 32767.)
    years = ds_t["year"].values
    tas_dict_585[model]  = da_t
    year_dict_585[model] = years
    ds_t.close()

def collect_days_for_gwl_model_first_reach(all_days, model_name, da_tas, years, level):
    da_model = all_days.sel(model=model_name)
    vals_tas = da_tas.values

    mask = vals_tas >= level
    yrs_over = years[mask]
    if yrs_over.size == 0:
        # print(f"    未达到 {level:.1f}°C，跳过")
        return None

    Y0 = int(yrs_over[0])
    start_year = Y0 - 9
    end_year   = Y0 + 10


    ntime_model = da_model.sizes["time"]

    start_idx = (start_year - 2015) * 12 + 11
    end_idx   = (end_year   - 2015) * 12 + 10

    if end_idx < 0 or start_idx >= ntime_model:
        # print("    20 年片段完全落在数据时间范围之外，跳过")
        return None

    start_idx_clip = max(start_idx, 0)
    end_idx_clip   = min(end_idx, ntime_model - 1)
    if start_idx_clip > end_idx_clip:
        # print("    有效下标为空，跳过")
        return None

    n_months = end_idx_clip - start_idx_clip + 1
    # print(
    #     f"      → 对应月度时段: {start_year}年12月 - {end_year}年11月；"
    #     f"time idx = {start_idx_clip} ~ {end_idx_clip} (共 {n_months} 个月)"
    # )

    seg_mon = da_model.isel(time=slice(start_idx_clip, end_idx_clip + 1))
    seg_ann = monthly_segment_to_climyear_annual(seg_mon)

    nyr = seg_ann.sizes["time"]
    clim_year = np.arange(start_year + 1, start_year + 1 + nyr)

    seg2 = xr.DataArray(
        seg_ann.data,
        dims=("climYear", "lat", "lon"),
        coords={"climYear": clim_year, "lat": seg_ann["lat"], "lon": seg_ann["lon"]},
        name=seg_mon.name,
        attrs=seg_mon.attrs,
    )
    seg2.attrs["Y0_first_reach"] = int(Y0)
    seg2.attrs["window_start_year"] = int(start_year)
    seg2.attrs["window_end_year"]   = int(end_year)
    return seg2

gwl_all_585 = {2.0: [], 3.0: [], 4.0: []}

for model in models_585:
    if model not in tas_dict_585:
        # print(f"[警告] 模式 {model} 没有 ΔGMT 文件，跳过 GWL 计算。")
        continue


    da_tas = tas_dict_585[model]
    years  = year_dict_585[model]

    for lvl in LEVELS:
        # print(f"\n  ---- {lvl}°C 阶段 ----")
        da_gwl = collect_days_for_gwl_model_first_reach(all_days_585, model, da_tas, years, lvl)
        if da_gwl is None:
            continue
        da_gwl = da_gwl.expand_dims(model=[model])
        gwl_all_585[lvl].append(da_gwl)

ERA5_CANDIDATES = [
    r"D:\code\comfort-area\data\v5\ERA5\apt.shushiDays.mon.mean.198001-202012.nc",
    r"D:\code\comfort-area\data\ERA5\apt.shushiDays.mon.mean.198001-202012.nc",
]
ERA5_MON_FILE = None
for fp in ERA5_CANDIDATES:
    if os.path.exists(fp):
        ERA5_MON_FILE = fp
        break
if ERA5_MON_FILE is None:
    raise FileNotFoundError(
        "[ERROR] 找不到 ERA5 comfortDays 月数据文件（apt.shushiDays.mon.mean.198001-202012.nc）。\n"
        "        请确认以下候选路径是否存在：\n" + "\n".join([f"          - {p}" for p in ERA5_CANDIDATES])
    )

# print(f"\n[INFO] ERA5 comfortDays 月数据：{ERA5_MON_FILE}")
ds_era = xr.open_dataset(ERA5_MON_FILE, decode_times=False)

da_era = _get_da_main(ds_era, prefer_names=("comfortDays", "shushiDays"))
da_era = _apply_fill_to_nan(da_era, default_fill=32767.0)

if ("year" not in da_era.dims) or ("month" not in da_era.dims):
    raise ValueError(f"[ERROR] ERA5 comfortDays 维度应含 month/year：da_era.dims={da_era.dims}")

da_era_ym = da_era.transpose("year", "month", "lat", "lon")
era_stack = da_era_ym.stack(t=("year", "month")).transpose("t", "lat", "lon")

def _find_t_index(stack_da, y, m):
    yy = stack_da["year"].values.astype(int)
    mm = stack_da["month"].values.astype(int)
    idx = np.where((yy == int(y)) & (mm == int(m)))[0]
    if idx.size == 0:
        raise ValueError(f"[ERROR] ERA5 月序列里找不到 {y:04d}-{m:02d}，请检查文件 year/month 坐标范围。")
    return int(idx[0])

t0 = _find_t_index(era_stack, 1980, 12)
t1 = _find_t_index(era_stack, 1999, 11)

era_seg = era_stack.isel(t=slice(t0, t1 + 1)).reset_index("t", drop=True)

n_months = era_seg.sizes["t"]
if n_months != 19 * 12:
    raise ValueError(
        f"[ERROR] 期望 1980-12..1999-11 共 228 个月，但实际取到 {n_months} 个月；"
        f"t0={t0}, t1={t1}。"
    )

nlat, nlon = era_seg.sizes["lat"], era_seg.sizes["lon"]
arr4 = np.asarray(era_seg.data).reshape(19, 12, nlat, nlon)

era_cym = xr.DataArray(
    arr4,
    dims=("climYear", "climMonth", "lat", "lon"),
    coords={
        "climYear": np.arange(1981, 2000),
        "climMonth": np.arange(1, 13),
        "lat": era_seg["lat"],
        "lon": era_seg["lon"],
    },
    name=da_era.name,
)

era_ann = era_cym.sum(dim="climMonth", skipna=True)
da_era_ref = era_ann.mean(dim="climYear", skipna=True)

lat_era = da_era_ref["lat"].values
lon_era = da_era_ref["lon"].values
data_era = da_era_ref.values

era_mask_val = np.where(np.isfinite(data_era) & (data_era >= THRESH_DAYS_ERA5), 1.0, 0.0)
era_mask_smooth = era_mask_val
era_mask_bin    = np.where(era_mask_smooth >= 0.5, 1.0, 0.0)
era_mask_cyc, lon_era_cyc = add_cyclic_point(era_mask_bin, coord=lon_era)

# print(f"[OK] ERA5 hatch period: 1980-12..1999-11 (ClimYear 1981..1999), t={t0}..{t1}, months={n_months}")
ds_era.close()

AREA_FILE = r"D:\code\comfort-area\data\ERA5\aera_25x25_nh_0_70.nc"
if not os.path.exists(AREA_FILE):
    alt = r"D:\code\comfort-area\data\v5\ERA5\aera_25x25_nh_0_70.nc"
    if os.path.exists(alt):
        AREA_FILE = alt

ds_area = xr.open_dataset(AREA_FILE)
area_da = ds_area["area"]
dims = list(area_da.dims)
lat_dim, lon_dim = dims[0], dims[1]
area_da = area_da.rename({lat_dim: "lat", lon_dim: "lon"})
ds_area.close()

lsm_nh = read_lsm(LANDSEA_FILE_NH)
lsm_sh = read_lsm(LANDSEA_FILE_SH)

landmask_global = landmask_like(area_da, lsm_nh, lsm_sh, lat_edge=LAT_EDGE)
landmask_era    = landmask_like(da_era_ref, lsm_nh, lsm_sh, lat_edge=LAT_EDGE)


da_mme_prob = {}
PIE_PARTS = {}
ABS_AREAS = {}

for wl in LEVELS:

    if len(gwl_all_585[wl]) == 0:

        da_mme_prob[wl] = None
        PIE_PARTS[wl] = (np.nan, np.nan)
        ABS_AREAS[wl] = {"NH_comf": np.nan, "SH_comf": np.nan, "Global_land": AREA_GL}
        continue

    mask_list = []

    for da in gwl_all_585[wl]:
        model = str(da["model"].values[0])
        seg_ann = da.isel(model=0)

        landmask_seg = landmask_like(_isel_first_year(seg_ann), lsm_nh, lsm_sh, lat_edge=LAT_EDGE)
        seg_ann = seg_ann.where(landmask_seg)

        mmod = clim_mask_from_seg_ann(seg_ann, model_name=model)
        mmod = mmod.where(landmask_seg)  # 海洋=NaN
        mmod = mmod.expand_dims(model=[model])
        mask_list.append(mmod)

    da_mask_all = xr.concat(mask_list, dim="model", coords="minimal", compat="override", join="override")

    consensus01 = vote_consensus_mask(da_mask_all, min_comf=MIN_MODELS_COMF, min_valid=MIN_MODELS_VALID)  # (lat,lon) 0/1/NaN
    da_mme_prob[wl] = consensus01

    nh_area, sh_area = area_from_binary_mask(consensus01, area_da, lat_edge=LAT_EDGE)
    part_nh = 100.0 * nh_area / AREA_GL
    part_sh = 100.0 * sh_area / AREA_GL

    PIE_PARTS[wl] = (part_nh, part_sh)
    ABS_AREAS[wl] = {"NH_comf": nh_area, "SH_comf": sh_area, "Global_land": AREA_GL}

    n_valid_models = int(da_mask_all.notnull().sum("model").max().values)

def compute_era5_ref_parts(da_era_ref, area_da, landmask_era,
                           thresh_days=THRESH_DAYS_ERA5, lat_edge=LAT_EDGE):
    da_field = da_era_ref.where(landmask_era)

    field_nh = _band_sel_strict(da_field, 0.0,  lat_edge)
    mask_nh  = np.isfinite(field_nh) & (field_nh >= thresh_days)
    nh_comf_area = float(area_da.where(mask_nh).sum(dim=("lat", "lon"), skipna=True))

    field_sh = _band_sel_strict(da_field, 0.0, -lat_edge)
    mask_sh  = np.isfinite(field_sh) & (field_sh >= thresh_days)
    sh_comf_area = float(area_da.where(mask_sh).sum(dim=("lat", "lon"), skipna=True))

    part_nh = 100.0 * nh_comf_area / AREA_GL
    part_sh = 100.0 * sh_comf_area / AREA_GL
    return part_nh, part_sh, nh_comf_area, sh_comf_area

era_p_nh, era_p_sh, era_a_nh, era_a_sh = compute_era5_ref_parts(da_era_ref, area_da, landmask_era)

era_global_frac = 100.0 * (era_a_nh + era_a_sh) / AREA_GL
era_nh_frac     = 100.0 * era_a_nh / AREA_NH
era_sh_frac     = 100.0 * era_a_sh / AREA_SH


def collect_days_for_level_model_20yr(all_days, model_name, da_tas, years, level):
    da_model = all_days.sel(model=model_name)
    vals_tas = da_tas.values

    mask_over = vals_tas >= level
    yrs_over  = years[mask_over]
    if yrs_over.size == 0:
        return None

    Y0 = int(yrs_over[0])
    start_year = Y0 - 9
    end_year   = Y0 + 10

    ntime_model = da_model.sizes["time"]
    start_idx = (start_year - 2015) * 12 + 11
    end_idx   = (end_year   - 2015) * 12 + 10

    if end_idx < 0 or start_idx >= ntime_model:
        return None

    start_idx_clip = max(start_idx, 0)
    end_idx_clip   = min(end_idx, ntime_model - 1)
    if start_idx_clip > end_idx_clip:
        return None

    seg_mon = da_model.isel(time=slice(start_idx_clip, end_idx_clip + 1))
    seg_ann = monthly_segment_to_climyear_annual(seg_mon)

    nyr = seg_ann.sizes["time"]
    clim_year = np.arange(start_year + 1, start_year + 1 + nyr)

    seg2 = xr.DataArray(
        seg_ann.data,
        dims=("climYear", "lat", "lon"),
        coords={"climYear": clim_year, "lat": seg_ann["lat"], "lon": seg_ann["lon"]},
        name=seg_mon.name,
        attrs=seg_mon.attrs,
    )
    seg2.attrs["Y0_first_reach"] = int(Y0)
    seg2.attrs["window_start_year"] = int(start_year)
    seg2.attrs["window_end_year"]   = int(end_year)
    return seg2


CENTERS = np.arange(1.2, 4.21, 0.2)

def compute_gwl_series_for_scenario_vote(scen_label,
                                         all_days_scen,
                                         models_in_data_scen,
                                         tas_dict_scen,
                                         year_dict_scen,
                                         lsm_nh, lsm_sh,
                                         area_da,
                                         min_comf=MIN_MODELS_COMF,
                                         min_valid=MIN_MODELS_VALID):

    centers_valid   = []
    frac_nh_series  = []
    frac_sh_series  = []
    frac_gl_series  = []

    for c in CENTERS:
        mask_list = []

        for model in models_in_data_scen:
            if model not in tas_dict_scen:
                continue
            da_tas = tas_dict_scen[model]
            years  = year_dict_scen[model]

            seg_ann = collect_days_for_level_model_20yr(all_days_scen, model, da_tas, years, c)
            if seg_ann is None:
                continue

            landmask_seg = landmask_like(_isel_first_year(seg_ann), lsm_nh, lsm_sh, lat_edge=LAT_EDGE)
            seg_ann = seg_ann.where(landmask_seg)

            mmod = clim_mask_from_seg_ann(seg_ann, model_name=model)
            mmod = mmod.where(landmask_seg)
            mmod = mmod.expand_dims(model=[model])
            mask_list.append(mmod)

        if len(mask_list) == 0:
            continue

        da_mask_all = xr.concat(mask_list, dim="model", coords="minimal", compat="override", join="override")
        consensus01 = vote_consensus_mask(da_mask_all, min_comf=min_comf, min_valid=min_valid)

        nh_area, sh_area = area_from_binary_mask(consensus01, area_da, lat_edge=LAT_EDGE)
        frac_nh = 100.0 * nh_area / AREA_GL
        frac_sh = 100.0 * sh_area / AREA_GL
        frac_gl = 100.0 * (nh_area + sh_area) / AREA_GL

        centers_valid.append(c)
        frac_nh_series.append(frac_nh)
        frac_sh_series.append(frac_sh)
        frac_gl_series.append(frac_gl)

    return (np.array(centers_valid),
            np.array(frac_nh_series),
            np.array(frac_sh_series),
            np.array(frac_gl_series))


gwl_vals_585, frac_nh_585, frac_sh_585, frac_global_585 = compute_gwl_series_for_scenario_vote(
    "ssp585", all_days_585, models_585, tas_dict_585, year_dict_585, lsm_nh, lsm_sh, area_da
)

BASE_CMIP_370 = r"D:\code\comfort-area\data\v5\CMIP6\ssp370"
all_days_370, models_370 = load_cmip6_comfortdays_scenario("ssp370", BASE_CMIP_370, wanted_models)

BASE_TAS_370 = r"D:\code\comfort-area\data\CMIP6\models\ssp370_zhang"
pattern_tas_370 = os.path.join(
    BASE_TAS_370,
    "tas_ano_base_1861-1900_*_ssp370_2015_2100_21years_slide_cos.mon.mean_2.nc"
)
tas_files_370 = sorted(glob.glob(pattern_tas_370))
tas_dict_370  = {}
year_dict_370 = {}

# print("\n[INFO] ΔGMT (tas_ano) 文件（ssp370，21 年滑动）：")
for f in tas_files_370:
    name = os.path.basename(f)
    try:
        model = name.split("1900_")[1].split("_ssp370")[0]
    except Exception:

        continue

    if model not in models_370:
        continue

    ds_t = xr.open_dataset(f)
    da_t = ds_t["tas_ano"].where(ds_t["tas_ano"] != 32767.)
    years = ds_t["year"].values
    tas_dict_370[model]  = da_t
    year_dict_370[model] = years
    ds_t.close()


gwl_vals_370, frac_nh_370, frac_sh_370, frac_global_370 = compute_gwl_series_for_scenario_vote(
    "ssp370", all_days_370, models_370, tas_dict_370, year_dict_370, lsm_nh, lsm_sh, area_da
)

gwl_all_370 = {2.0: [], 3.0: [], 4.0: []}

for model in models_370:
    if model not in tas_dict_370:
        continue

    da_tas = tas_dict_370[model]
    years  = year_dict_370[model]

    for lvl in LEVELS:
        da_gwl = collect_days_for_gwl_model_first_reach(all_days_370, model, da_tas, years, lvl)
        if da_gwl is None:
            continue
        da_gwl = da_gwl.expand_dims(model=[model])
        gwl_all_370[lvl].append(da_gwl)

if EXPORT_MME_SEG_ANN:
    export_mme_seg_ann_for_gwl(
        gwl_all_585,
        out_dir=EXPORT_DIR_MME_SEG_ANN,
        scen_label="ssp585",
        lsm_nh=lsm_nh, lsm_sh=lsm_sh,
        lat_edge=LAT_EDGE,
        thresh_days=THRESH_DAYS_DEFAULT,
        compress=True
    )
    export_mme_seg_ann_for_gwl(
        gwl_all_370,
        out_dir=EXPORT_DIR_MME_SEG_ANN,
        scen_label="ssp370",
        lsm_nh=lsm_nh, lsm_sh=lsm_sh,
        lat_edge=LAT_EDGE,
        thresh_days=THRESH_DAYS_DEFAULT,
        compress=True
    )

use_anomaly = False

if use_anomaly:
    frac_nh_585_plot     = frac_nh_585     - era_p_nh
    frac_sh_585_plot     = frac_sh_585     - era_p_sh
    frac_global_585_plot = frac_global_585 - era_global_frac

    frac_nh_370_plot     = frac_nh_370     - era_p_nh
    frac_sh_370_plot     = frac_sh_370     - era_p_sh
    frac_global_370_plot = frac_global_370 - era_global_frac

    y_label = "Change in comfortable area fraction (percentage points)"
else:
    frac_nh_585_plot     = frac_nh_585
    frac_sh_585_plot     = frac_sh_585
    frac_global_585_plot = frac_global_585

    frac_nh_370_plot     = frac_nh_370
    frac_sh_370_plot     = frac_sh_370
    frac_global_370_plot = frac_global_370

    y_label = "Percentage"

START_YEAR = 2021

def annual_comf_area_series_from_monthly_vote(all_days_scen,
                                              area_da,
                                              lsm_nh, lsm_sh,
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
    thr_vec = np.array([get_model_thresh(m, default=thresh_days_default) for m in model_names], dtype=float)
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


years_585_ts, nh_frac_585_ts, sh_frac_585_ts, gl_frac_585_ts = annual_comf_area_series_from_monthly_vote(
    all_days_585, area_da, lsm_nh, lsm_sh, thresh_days_default=THRESH_DAYS_DEFAULT, lat_edge=LAT_EDGE, base_year=2015
)
years_370_ts, nh_frac_370_ts, sh_frac_370_ts, gl_frac_370_ts = annual_comf_area_series_from_monthly_vote(
    all_days_370, area_da, lsm_nh, lsm_sh, thresh_days_default=THRESH_DAYS_DEFAULT, lat_edge=LAT_EDGE, base_year=2015
)

if EXPORT_FIG_E_TXT:
    export_e_txt_onecol_80(BASE_CMIP_585, "ssp585", years_585_ts, nh_frac_585_ts, sh_frac_585_ts,
                           start_year=START_YEAR, end_year=2100)
    export_e_txt_onecol_80(BASE_CMIP_370, "ssp370", years_370_ts, nh_frac_370_ts, sh_frac_370_ts,
                           start_year=START_YEAR, end_year=2100)


scen_list = ["ssp370", "ssp585"]
frac_series = {
    "NH": {"ssp585": (years_585_ts, nh_frac_585_ts), "ssp370": (years_370_ts, nh_frac_370_ts)},
    "SH": {"ssp585": (years_585_ts, sh_frac_585_ts), "ssp370": (years_370_ts, sh_frac_370_ts)},
}
glob_frac_series = {
    "ssp585": (years_585_ts, gl_frac_585_ts),
    "ssp370": (years_370_ts, gl_frac_370_ts),
}

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

trend_label = {"NH": {}, "SH": {}, "GL": {}}

colors_region = {
    "NH": {"ssp370": "#f393be", "ssp585": "#ee3b73"},
    "SH": {"ssp370": "#00a4d2", "ssp585": "#0078bf"},
    "GL": {"ssp370": "#64B3DA", "ssp585": "#E63A30"},
}
colors_region2 = {
    "NH": {"ssp370": "#f6aece", "ssp585": "#f05988"},
    "SH": {"ssp370": "#46bdde", "ssp585": "#0179bf"},
}

cmap_green = ListedColormap(["white", "#7bca73"])
bounds = [0, 0.5, 1.5]
norm = BoundaryNorm(bounds, cmap_green.N)

PIE_SIZES  = [AREA_NH / AREA_GL * 100.0, AREA_SH / AREA_GL * 100.0]
PIE_COLORS = ['#f393be', '#00a4d2']

PIE_FS_PERCENT = FS_PIE_PERCENT
PIE_FS_INNER   = FS_PIE_INNER

PIE_BOXES = [
    (0.45, 0.001, 0.545, 0.645),
    (0.45, 0.001, 0.545, 0.645),
    (0.45, 0.001, 0.545, 0.645),
]

proj = ccrs.PlateCarree(central_longitude=150)

fig = plt.figure(figsize=(12, 12))
gs  = fig.add_gridspec(
    nrows=3, ncols=3,
    height_ratios=[2.2, 1.0, 1.0],
    hspace=0.2, wspace=0.15
)

axes_maps = [fig.add_subplot(gs[0, i], projection=proj) for i in range(3)]

ax3 = fig.add_subplot(gs[2, :])

fig.subplots_adjust(top=0.96, bottom=0.08, left=0.06, right=0.99)

panel_order  = [2.0, 3.0, 4.0]
panel_titles = ["GWL: 2°C", "GWL: 3°C", "GWL: 4°C"]

GREEN_CMAP = ListedColormap(["#7bca73"])

for i, (ax, wl) in enumerate(zip(axes_maps, panel_order)):
    if (wl not in da_mme_prob) or (da_mme_prob[wl] is None):
        ax.set_visible(False)
        continue

    da_panel = da_mme_prob[wl].sel(lat=slice(-60, 60))  # (lat,lon) 0/1, ocean=NaN
    lon = da_panel["lon"].values
    lat = da_panel["lat"].values
    prob = da_panel.values

    ocean_mask = np.isnan(prob)

    mask_val = np.where(np.isfinite(prob) & (prob >= 0.5), 1.0, 0.0)

    if SMOOTH_MODEL_CONSENSUS:
        mask_smooth = mask_val
        mask_bin    = np.where(mask_smooth >= 0.5, 1.0, 0.0)
    else:
        mask_bin = mask_val

    mask_green = np.where(mask_bin == 1, 1.0, np.nan)

    mask_cyc, lon_cyc = add_cyclic_point(mask_green, coord=lon)
    ocean_cyc, _      = add_cyclic_point(ocean_mask.astype(int), coord=lon)
    ocean_cyc = ocean_cyc.astype(bool)

    mask_cyc_plot = np.ma.array(mask_cyc, mask=ocean_cyc)

    ax.set_extent([-180, 180, -60, 60], crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.LAND, facecolor="white", edgecolor="none", zorder=0)

    era_fill = np.where(era_mask_cyc >= 0.5, 1.0, np.nan)
    ax.pcolormesh(
        lon_era_cyc, lat_era, era_fill,
        transform=ccrs.PlateCarree(),
        cmap=GREEN_CMAP, vmin=0, vmax=1,
        zorder=1
    )

    with mpl.rc_context({'hatch.color': '#333333',
                         'hatch.linewidth': 0.8}):
        cs = ax.contourf(
            lon_cyc, lat, mask_cyc_plot,
            levels=bounds,
            colors='none',
            hatches=['', '....'],
            transform=ccrs.PlateCarree(),
            zorder=1.5
        )

    ax.add_feature(
        cfeature.OCEAN,
        facecolor=OCEAN_COLOR,
        edgecolor="none",
        zorder=2
    )

    ax.coastlines(linewidth=0.5, zorder=3)

    ax.set_title(panel_titles[i], fontsize=FS_TITLE, loc="left")

    xticks = [30, 90, 150, -150, -90, -30]
    yticks = np.arange(-60, 61, 30)

    ax.set_xticks(xticks, crs=ccrs.PlateCarree())
    ax.set_yticks(yticks, crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
    ax.yaxis.set_major_formatter(LatitudeFormatter())

    ax.tick_params(
        axis="both", which="major",
        direction="out", length=3.8, width=0.8,
        labelsize=FS_MAP_LONLAT, pad=3
    )
    ax.tick_params(axis="x", top=False, labeltop=False, bottom=True, labelbottom=True)
    ax.tick_params(axis="y", right=False, labelright=False, left=True, labelleft=True)

    ax.text(0.948, -0.0845, "30°W",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=12, zorder=20)

    ax.plot([1.0, 1.0], [0.0, -0.04],
            transform=ax.transAxes, color="black",
            linewidth=0.7, clip_on=False, zorder=20)

    part_NH, part_SH = PIE_PARTS[wl]
    x0, y0, w_box, h_box = PIE_BOXES[i]
    ax_pie = inset_axes(
        ax,
        width="100%", height="100%",
        bbox_to_anchor=(x0, y0, w_box, h_box),
        bbox_transform=ax.transAxes,
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

    offsets = {0: (-0.2, 0.10), 1: (0.05, -0.28)}
    for j, t in enumerate(autotexts):
        x_t, y_t = t.get_position()
        dx, dy = offsets.get(j, (0.0, 0.0))
        t.set_position((x_t + dx, y_t + dy))
        t.set_fontsize(PIE_FS_PERCENT)
        t.set_text("NH" if j == 0 else "SH")

    nh_wedge = wedges[0]
    boundary = nh_wedge.theta2
    gap_deg = 15.0
    offset  = 0.10

    angle_nh_inner = 360.0 * part_NH / 100.0
    theta2_nh_inner = boundary - gap_deg / 2.0
    theta1_nh_inner = theta2_nh_inner - angle_nh_inner
    theta_mid_nh_rad = np.deg2rad(0.5 * (theta1_nh_inner + theta2_nh_inner))
    dx_nh = offset * np.cos(theta_mid_nh_rad)
    dy_nh = offset * np.sin(theta_mid_nh_rad)

    ax_pie.add_patch(Wedge(
        center=(dx_nh, dy_nh),
        r=1.0,
        theta1=theta1_nh_inner,
        theta2=theta2_nh_inner,
        facecolor='#ee3b73',
        edgecolor='k', linewidth=0.6, alpha=0.9,
        zorder=3
    ))
    ax_pie.text(
        dx_nh + 0.70*np.cos(theta_mid_nh_rad),
        dy_nh + 0.70*np.sin(theta_mid_nh_rad),
        f"{part_NH:.1f}",
        ha='center', va='center',
        fontsize=PIE_FS_INNER, fontweight='bold'
    )

    angle_sh_inner = 360.0 * part_SH / 100.0
    theta1_sh_inner = boundary + gap_deg / 2.0
    theta2_sh_inner = theta1_sh_inner + angle_sh_inner
    theta_mid_sh_rad = np.deg2rad(0.5 * (theta1_sh_inner + theta2_sh_inner))
    dx_sh = offset * np.cos(theta_mid_sh_rad)
    dy_sh = offset * np.sin(theta_mid_sh_rad)

    ax_pie.add_patch(Wedge(
        center=(dx_sh, dy_sh),
        r=1.0,
        theta1=theta1_sh_inner,
        theta2=theta2_sh_inner,
        facecolor='#1f78b4',
        edgecolor='k', linewidth=0.6, alpha=0.9,
        zorder=3
    ))
    ax_pie.text(
        dx_sh + 0.70*np.cos(theta_mid_sh_rad),
        dy_sh + 0.70*np.sin(theta_mid_sh_rad),
        f"{part_SH:.1f}",
        ha='center', va='center',
        fontsize=PIE_FS_INNER, fontweight='bold'
    )

    ax_pie.set_xticks([])
    ax_pie.set_yticks([])
    for spine in ax_pie.spines.values():
        spine.set_visible(False)

gs_row2 = gs[1, :].subgridspec(1, 5, wspace=0.65)

ax2     = fig.add_subplot(gs_row2[0, 0:2])
ax2_r   = ax2.twinx()

ax_ts_l = fig.add_subplot(gs_row2[0, 2:5])
ax_ts_r = ax_ts_l.twinx()

if use_anomaly:
    ax2.axhline(0.0, color='k', linewidth=1.0, linestyle='--', zorder=0)
    ax2_r.axhline(0.0, color='k', linewidth=1.0, linestyle='--', zorder=0)

style_NH_585 = dict(color=colors_region["NH"]["ssp585"], linestyle="-", linewidth=3)
style_SH_585 = dict(color=colors_region["SH"]["ssp585"], linestyle="-", linewidth=3)
style_NH_370 = dict(color=colors_region["NH"]["ssp370"], linestyle="-", linewidth=2)
style_SH_370 = dict(color=colors_region["SH"]["ssp370"], linestyle="-", linewidth=2)

ax2.plot(gwl_vals_585, frac_nh_585_plot, label="NH SSP585", **style_NH_585)
ax2.plot(gwl_vals_370, frac_nh_370_plot, label="NH SSP370", **style_NH_370)

ax2_r.plot(gwl_vals_585, frac_sh_585_plot, label="SH SSP585", **style_SH_585)
ax2_r.plot(gwl_vals_370, frac_sh_370_plot, label="SH SSP370", **style_SH_370)

ax2.set_xlabel(r'GWL [°C]', fontsize=FS_LABEL)
ax2.set_xlim(1.2, 4.2)
ax2_r.set_ylabel(y_label + " (SH) [%]", fontsize=FS_LABEL,)
ax2_r.yaxis.label.set_color(colors_region["SH"]["ssp585"])

ticks  = [1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2]
labels = ['1.2','','','', '2','','','','', '3', '','','','','4','']
ax2.set_xticks(ticks)
ax2.set_xticklabels(labels, fontsize=FS_TICK)
ax2.set_ylabel(y_label + " (NH) [%]", fontsize=FS_LABEL)

if AUTO_D_YLIM:
    nh_all = np.concatenate([frac_nh_585_plot, frac_nh_370_plot]) if (len(frac_nh_585_plot) and len(frac_nh_370_plot)) else frac_nh_585_plot
    sh_all = np.concatenate([frac_sh_585_plot, frac_sh_370_plot]) if (len(frac_sh_585_plot) and len(frac_sh_370_plot)) else frac_sh_585_plot

    if nh_all.size > 0:
        ymin, ymax = np.nanmin(nh_all), np.nanmax(nh_all)
        pad = max(0.3, 0.08 * (ymax - ymin if ymax > ymin else 1.0))
        ax2.set_ylim(ymin - pad, ymax + pad)
    if sh_all.size > 0:
        ymin, ymax = np.nanmin(sh_all), np.nanmax(sh_all)
        pad = max(0.5, 0.08 * (ymax - ymin if ymax > ymin else 1.0))
        ax2_r.set_ylim(ymin - pad, ymax + pad)
else:
    ax2.set_ylim(*D_NH_LIM)
    ax2_r.set_ylim(*D_SH_LIM)

    if D_NH_TICKS is not None:
        ax2.set_yticks(D_NH_TICKS)
    if D_NH_TICKLABELS is not None:
        ax2.set_yticklabels(D_NH_TICKLABELS, fontsize=FS_TICK)

    if D_SH_TICKS is not None:
        ax2_r.set_yticks(D_SH_TICKS)
    if D_SH_TICKLABELS is not None:
        ax2_r.set_yticklabels(D_SH_TICKLABELS, fontsize=FS_TICK)

ax2.tick_params(axis='x', labelsize=FS_TICK, width=1.0, colors='k')
ax2.tick_params(axis='y', labelsize=FS_TICK, width=1.0, colors=colors_region["NH"]["ssp585"])
ax2.yaxis.label.set_color(colors_region["NH"]["ssp585"])

ax2_r.tick_params(axis='y', labelsize=FS_TICK, width=1.0, colors=colors_region["SH"]["ssp585"])

handles_left,  labels_left  = ax2.get_legend_handles_labels()
handles_right, labels_right = ax2_r.get_legend_handles_labels()
ax2.legend(
    handles_left + handles_right,
    labels_left  + labels_right,
    frameon=False, fontsize=FS_LEGEND, loc='lower left', ncol=2
)

region_styles = {"NH": "-", "SH": "-"}

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
        linestyle=region_styles["NH"],
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
        linestyle=region_styles["SH"],
        color=color_sh,
        linewidth=3 if scen == "ssp585" else 2,
        zorder=2
    )

ax_ts_l.set_xlim(START_YEAR, 2100)

if AUTO_E_YLIM:
    pass
else:
    ax_ts_l.set_ylim(*E_NH_LIM)
    ax_ts_r.set_ylim(*E_SH_LIM)

    if E_NH_TICKS is not None:
        ax_ts_l.set_yticks(E_NH_TICKS)
    if E_NH_TICKLABELS is not None:
        ax_ts_l.set_yticklabels(E_NH_TICKLABELS, fontsize=FS_TICK)

    if E_SH_TICKS is not None:
        ax_ts_r.set_yticks(E_SH_TICKS)
    if E_SH_TICKLABELS is not None:
        ax_ts_r.set_yticklabels(E_SH_TICKLABELS, fontsize=FS_TICK)

custom_xticks = [2021, 2040, 2060, 2080, 2100]
ax_ts_l.set_xticks(custom_xticks)
ax_ts_l.set_xticklabels([str(y) for y in custom_xticks], rotation=0, fontsize=FS_TICK)

ax_ts_l.tick_params(axis='x', labelsize=FS_TICK, width=1.0, colors='k')
ax_ts_l.tick_params(axis='y', labelsize=FS_TICK, width=1.0, colors=colors_region["NH"]["ssp585"])
ax_ts_l.set_ylabel("Percentage (NH) [%]", fontsize=FS_LABEL)
ax_ts_l.yaxis.set_label_coords(-0.044, 0.5)  # x越负越往左，y=0.5居中
ax_ts_l.yaxis.label.set_color(colors_region["NH"]["ssp585"])

ax_ts_r.set_ylabel("Percentage (SH) [%]", fontsize=FS_LABEL)
ax_ts_r.tick_params(axis='y', labelsize=FS_TICK, width=1.0, colors=colors_region["SH"]["ssp585"])
ax_ts_r.yaxis.label.set_color(colors_region["SH"]["ssp585"])

handles_585, labels_585 = [], []
handles_370, labels_370 = [], []

for reg in ["NH", "SH"]:
    for scen in scen_list:
        color  = colors_region[reg][scen]
        label  = trend_label[reg][scen]
        if scen == "ssp585":
            handles_585.append(Line2D([0], [0], color=color, lw=3))
            labels_585.append(label)
        else:
            handles_370.append(Line2D([0], [0], color=color, lw=2))
            labels_370.append(label)

leg1 = ax_ts_l.legend(handles_585, labels_585, frameon=False, ncol=2, fontsize=FS_LEGEND,
                      loc="lower left", bbox_to_anchor=(0.00, 0.099))
ax_ts_l.add_artist(leg1)
ax_ts_l.legend(handles_370, labels_370, frameon=False, ncol=2, fontsize=FS_LEGEND,
               loc="lower left", bbox_to_anchor=(0.00, 0.0))


def pick_values_at_levels(gwl_vals, series, levels):
    out = []
    for lv in levels:
        idx = np.where(np.isclose(gwl_vals, lv, atol=1e-6))[0]
        if idx.size == 0:
            out.append(np.nan)
            # print(f"[警告] 在 gwl_vals 中找不到 ΔGMT={lv:.1f}°C，对应柱状图值为 NaN")
        else:
            out.append(float(series[idx[0]]))
    return np.array(out)

ref_G  = era_global_frac
ref_NH = era_p_nh
ref_SH = era_p_sh

levels_gwl = [2.0, 3.0, 4.0]

NH_585_levels = pick_values_at_levels(gwl_vals_585, frac_nh_585, levels_gwl)
SH_585_levels = pick_values_at_levels(gwl_vals_585, frac_sh_585, levels_gwl)
G_585_levels  = pick_values_at_levels(gwl_vals_585, frac_global_585, levels_gwl)

NH_370_levels = pick_values_at_levels(gwl_vals_370, frac_nh_370, levels_gwl)
SH_370_levels = pick_values_at_levels(gwl_vals_370, frac_sh_370, levels_gwl)
G_370_levels  = pick_values_at_levels(gwl_vals_370, frac_global_370, levels_gwl)

labels_bar = ['ref', '2°C - ref', '3°C - ref', '4°C - ref']
x_bar = np.arange(len(labels_bar))

NH_585 = np.concatenate(([ref_NH], NH_585_levels))
SH_585 = np.concatenate(([ref_SH], SH_585_levels))
G_585  = np.concatenate(([ref_G],  G_585_levels))

NH_370 = np.concatenate(([ref_NH], NH_370_levels))
SH_370 = np.concatenate(([ref_SH], SH_370_levels))
G_370  = np.concatenate(([ref_G],  G_370_levels))

NH_diff_585 = NH_585 - ref_NH
SH_diff_585 = SH_585 - ref_SH
G_diff_585  = G_585  - ref_G

NH_diff_370 = NH_370 - ref_NH
SH_diff_370 = SH_370 - ref_SH
G_diff_370  = G_370  - ref_G

width_hemi = 0.22
width_scen = 0.22 * 0.45

ax3_right = ax3.twinx()

REF_COLOR_NH = colors_region2["NH"]["ssp370"]
REF_COLOR_SH = colors_region2["SH"]["ssp370"]

COLOR_BAR_NH_585 = colors_region2["NH"]["ssp585"]
COLOR_BAR_NH_370 = colors_region2["NH"]["ssp370"]
COLOR_BAR_SH_585 = colors_region2["SH"]["ssp585"]
COLOR_BAR_SH_370 = colors_region2["SH"]["ssp370"]

ax3.bar(x_bar[0] - width_hemi / 2, NH_585[0], width_hemi, color=REF_COLOR_NH, edgecolor='none')
ax3.bar(x_bar[0] + width_hemi / 2, SH_585[0], width_hemi, color=REF_COLOR_SH, edgecolor='none')

for i in range(1, len(labels_bar)):
    center_nh = x_bar[i] - width_hemi / 2
    center_sh = x_bar[i] + width_hemi / 2

    ax3_right.bar(center_nh - width_scen / 2, NH_diff_585[i], width_scen, color=COLOR_BAR_NH_585, edgecolor='none')
    ax3_right.bar(center_nh + width_scen / 2, NH_diff_370[i], width_scen, color=COLOR_BAR_NH_370, edgecolor='none')

    ax3_right.bar(center_sh - width_scen / 2, SH_diff_585[i], width_scen, color=COLOR_BAR_SH_585, edgecolor='none')
    ax3_right.bar(center_sh + width_scen / 2, SH_diff_370[i], width_scen, color=COLOR_BAR_SH_370, edgecolor='none')

ax3_right.axhline(0, color='k', linewidth=1.0)
ax3.axvline(0.5, color='k', linestyle='--', linewidth=1.0)

ax3.set_ylabel('Percentage [%]', fontsize=FS_LABEL)
ax3_right.set_ylabel(r'$\Delta$Percentage [%]', fontsize=FS_LABEL)

ax3.set_xticks(x_bar)
ax3.set_xticklabels(labels_bar, fontsize=FS_TICK)

ax3.tick_params(direction='out', length=3, width=0.8, labelsize=FS_TICK)
ax3_right.tick_params(direction='out', length=3, width=0.8, labelsize=FS_TICK)


if AUTO_F_YLIM:
    left_vals = np.array([NH_585[0], SH_585[0]])
    if np.all(np.isfinite(left_vals)):
        ymin, ymax = left_vals.min(), left_vals.max()
        pad = max(1.0, 0.15 * (ymax - ymin if ymax > ymin else 1.0))
        ax3.set_ylim(max(0.0, ymin - pad), ymax + pad)

    right_vals = np.concatenate([NH_diff_585[1:], SH_diff_585[1:], NH_diff_370[1:], SH_diff_370[1:]])
    right_vals = right_vals[np.isfinite(right_vals)]
    if right_vals.size > 0:
        ymin, ymax = right_vals.min(), right_vals.max()
        pad = max(0.5, 0.15 * (ymax - ymin if ymax > ymin else 1.0))
        ax3_right.set_ylim(ymin - pad, ymax + pad)
else:
    ax3.set_ylim(*F_LEFT_LIM)
    ax3_right.set_ylim(*F_RIGHT_LIM)

    if F_LEFT_TICKS is not None:
        ax3.set_yticks(F_LEFT_TICKS)
    if F_LEFT_TICKLABELS is not None:
        ax3.set_yticklabels(F_LEFT_TICKLABELS, fontsize=FS_TICK)

    if F_RIGHT_TICKS is not None:
        ax3_right.set_yticks(F_RIGHT_TICKS)
    if F_RIGHT_TICKLABELS is not None:
        ax3_right.set_yticklabels(F_RIGHT_TICKLABELS, fontsize=FS_TICK)

handles_bar = [
    mpatches.Patch(facecolor=COLOR_BAR_NH_585, label='NH SSP585'),
    mpatches.Patch(facecolor=COLOR_BAR_NH_370, label='NH SSP370'),
    mpatches.Patch(facecolor=COLOR_BAR_SH_585, label='SH SSP585'),
    mpatches.Patch(facecolor=COLOR_BAR_SH_370, label='SH SSP370'),
]
ax3.legend(handles=handles_bar, frameon=False, loc='upper right',
           fontsize=FS_LEGEND, handlelength=1.4, handleheight=0.7, ncol=2)


fig.canvas.draw()

delta_row2 = 0.08
delta_row3 = 0.073

for axm in [ax2, ax2_r, ax_ts_l, ax_ts_r]:
    pos = axm.get_position()
    axm.set_position([pos.x0, pos.y0 + delta_row2, pos.width, pos.height])

ax2_r.set_position(ax2.get_position())
ax_ts_r.set_position(ax_ts_l.get_position())

for axm in [ax3, ax3_right]:
    pos = axm.get_position()
    axm.set_position([pos.x0, pos.y0 + delta_row3, pos.width, pos.height])


fig.canvas.draw()

def add_panel_label(fig, ax, label, dx=0.04):
    pos = ax.get_position()
    x = pos.x0 - dx
    y = pos.y1 + 0.014
    fig.text(x, y, label, ha='right', va='center',
             fontsize=FS_TITLE + 6, fontweight='bold')

add_panel_label(fig, axes_maps[0], 'a')
add_panel_label(fig, axes_maps[1], 'b', 0.025)
add_panel_label(fig, axes_maps[2], 'c', 0.0275)
add_panel_label(fig, ax2, 'd')
add_panel_label(fig, ax_ts_l, 'e', 0.027)
add_panel_label(fig, ax3, 'f')


fig.canvas.draw()
fig_w, fig_h = fig.get_size_inches()

axes_dict = {
    "a  (map 2°C)": axes_maps[0],
    "b  (map 3°C)": axes_maps[1],
    "c  (map 4°C)": axes_maps[2],
    "d  (ΔGMT–NH/SH 折线左轴)": ax2,
    "dR (ΔGMT–NH/SH 折线右轴)": ax2_r,
    "e  (NH/SH 未来时间序列左轴)": ax_ts_l,
    "eR (NH/SH 未来时间序列右轴)": ax_ts_r,
    "f  (柱状图左轴)": ax3,
    "fR (柱状图右轴)": ax3_right,
}

plt.savefig(r"D:\code\comfort-area\fig\v7\fig4.pdf",
            dpi=600, bbox_inches='tight', pad_inches=0.05)
