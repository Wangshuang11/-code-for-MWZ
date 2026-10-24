# -*- coding: utf-8 -*-
import os
import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import linregress

mpl.rcParams["font.family"] = "Times New Roman"
mpl.rcParams["font.size"] = 14

BASE_CMIP6 = r"D:\code\comfort-area\data\v5\CMIP6"
BASE_CESM2 = r"D:\code\comfort-area\data\v5\CESM2"

SAVE_FIG = True
OUT_PNG  = os.path.join(r"D:\code\comfort-area\fig\v7", "figS6.png")

FILL_MISS = 32767.0

X_PAD = 2
X_TICKS = np.arange(1980, 2021, 10)

TS_CFG = {
    "ALL":   dict(nh_ylim=(14,23), nh_ticks=[14, 17, 20, 23], nh_ticklabels=["14", "17", "20","23"],
                  sh_ylim=(14,17), sh_ticks=[14, 15, 16, 17], sh_ticklabels=["14", "15", "16", "17"]),
    "CESM2": dict(nh_ylim=(13,22), nh_ticks=[13, 16, 19, 22], nh_ticklabels=["13", "16", "19","22"],
                  sh_ylim=(14.5,16.3), sh_ticks=[14.5, 15, 15.5, 16], sh_ticklabels=["14.5", "15", "15.5", "16"]),
    "GHG":   dict(nh_ylim=(14,23), nh_ticks=[14, 17, 20, 23], nh_ticklabels=["14", "17", "20","23"],
                  sh_ylim=(14,17), sh_ticks=[14, 15, 16, 17], sh_ticklabels=["14", "15", "16", "17"]),
    "AA":    dict(nh_ylim=(16,22), nh_ticks=[16, 18, 20, 22], nh_ticklabels=["16", "18", "20","22"],
                  sh_ylim=(15.5,17.5), sh_ticks=[15.5, 16.0, 16.5, 17.0, 17.5], sh_ticklabels=["15.5", "16.0", "16.5", "17.0", "17.5"]),
    "NAT":   dict(nh_ylim=(16,22), nh_ticks=[16, 18, 20, 22], nh_ticklabels=["16", "18", "20","22"],
                  sh_ylim=(15.5,17.5), sh_ticks=[15.5, 16.0, 16.5, 17.0, 17.5], sh_ticklabels=["15.5", "16.0", "16.5", "17.0", "17.5"]),
}

COLOR_NH = "#ee3b73"
COLOR_SH = "#1f78b4"

PANEL_LABELS = ["a", "b", "c"]
PANEL_FONTSIZE = 23
PANEL_WEIGHT = "bold"
PANEL_XY = (-0.14, 1.15)
PANEL_VA = "top"
PANEL_HA = "left"

def add_panel_label(ax, label: str):
    ax.text(
        PANEL_XY[0], PANEL_XY[1], label,
        transform=ax.transAxes,
        ha=PANEL_HA, va=PANEL_VA,
        fontsize=PANEL_FONTSIZE,
        fontweight=PANEL_WEIGHT,
        zorder=100
    )

def pick_varname(ds: xr.Dataset) -> str:
    for v in ["midl_area", "mild_area", "midlarea", "mildarea"]:
        if v in ds.data_vars:
            return v
    dvs = list(ds.data_vars)
    if not dvs:
        raise ValueError("nc 文件里没有 data_vars")
    return dvs[0]

def ensure_latlon_names(da: xr.DataArray) -> xr.DataArray:
    if "lat" in da.dims and "lon" in da.dims:
        return da
    rename_map = {}
    if "latitude" in da.dims:
        rename_map["latitude"] = "lat"
    if "longitude" in da.dims:
        rename_map["longitude"] = "lon"
    if rename_map:
        da = da.rename(rename_map)
    return da

def read_mask01_from_nc(nc_path: str) -> xr.DataArray:
    if not os.path.exists(nc_path):
        raise FileNotFoundError(f"找不到 nc：{nc_path}")

    with xr.open_dataset(nc_path) as ds:
        vname = pick_varname(ds)
        da = ds[vname].load()

    while da.ndim > 2:
        da = da.isel({da.dims[0]: 0})

    da = ensure_latlon_names(da).astype(float)
    mask01 = xr.where((da >= 0.5) & (da <= 1.5), 1.0, 0.0)
    mask01 = mask01.fillna(0.0)
    return mask01

def read_1col_txt(path: str, expected_len: int) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到 txt：{path}")
    y = np.loadtxt(path).astype(float).reshape(-1)
    if y.size != expected_len:
        raise ValueError(f"期望 {expected_len} 行，但读到 {y.size} 行：{path}")
    return y

def linreg_1d(years: np.ndarray, y: np.ndarray):
    m = np.isfinite(years) & np.isfinite(y)
    if m.sum() < 3:
        return dict(slope=np.nan, intercept=np.nan, pvalue=np.nan, y_fit=np.full_like(y, np.nan, dtype=float))
    res = linregress(years[m], y[m])
    y_fit = res.slope * years + res.intercept
    return dict(slope=res.slope, intercept=res.intercept, pvalue=res.pvalue, y_fit=y_fit)

def draw_timeseries_like_ref(ax, years, nh, sh, cfg, add_legend=False, xlim_pad=None, show_xlabel=False, title_left=None):
    ax2 = ax.twinx()

    ln1 = ax.plot(years, nh, color=COLOR_NH, lw=1.5, label="NH", zorder=3)
    ln2 = ax2.plot(years, sh, color=COLOR_SH, lw=1.5, label="SH", zorder=3)

    std_nh = np.nanstd(nh)
    std_sh = np.nanstd(sh)
    ax.fill_between(years, nh - std_nh, nh + std_nh, color=COLOR_NH, alpha=0.15, linewidth=0, zorder=1)
    ax2.fill_between(years, sh - std_sh, sh + std_sh, color=COLOR_SH, alpha=0.15, linewidth=0, zorder=1)

    tr_nh = linreg_1d(years, nh)
    tr_sh = linreg_1d(years, sh)
    ax.plot(years, tr_nh["y_fit"], color=COLOR_NH, lw=1.8, ls="--", zorder=4)
    ax2.plot(years, tr_sh["y_fit"], color=COLOR_SH, lw=1.8, ls="--", zorder=4)

    if xlim_pad is None:
        xlim_pad = (years.min() - X_PAD, years.max() + X_PAD)
    ax.set_xlim(*xlim_pad)

    ax.set_xticks(X_TICKS)
    ax.tick_params(axis="x", labelbottom=True)

    if cfg.get("nh_ylim") is not None:
        ax.set_ylim(cfg["nh_ylim"])
    if cfg.get("nh_ticks") is not None:
        ax.set_yticks(cfg["nh_ticks"])
    if cfg.get("nh_ticklabels") is not None:
        ax.set_yticklabels(cfg["nh_ticklabels"])

    if cfg.get("sh_ylim") is not None:
        ax2.set_ylim(cfg["sh_ylim"])
    if cfg.get("sh_ticks") is not None:
        ax2.set_yticks(cfg["sh_ticks"])
    if cfg.get("sh_ticklabels") is not None:
        ax2.set_yticklabels(cfg["sh_ticklabels"])

    ax.set_ylabel("Percentage (NH) [%]", color=COLOR_NH)
    ax2.set_ylabel("Percentage (SH) [%]", color=COLOR_SH)
    ax.tick_params(axis="y", colors=COLOR_NH)
    ax2.tick_params(axis="y", colors=COLOR_SH)

    slope_nh_100a = tr_nh["slope"] * 100.0
    slope_sh_100a = tr_sh["slope"] * 100.0
    star_nh = "*" if (np.isfinite(tr_nh["pvalue"]) and tr_nh["pvalue"] <= 0.1) else ""
    star_sh = "*" if (np.isfinite(tr_sh["pvalue"]) and tr_sh["pvalue"] <= 0.1) else ""

    txt = (
        f"NH: {slope_nh_100a:5.2f}%/C{star_nh}     "
        f"SH: {slope_sh_100a:5.2f}%/C{star_sh}"
    )
    ax.text(0.042, 0.02, txt, transform=ax.transAxes, ha="left", va="bottom", fontsize=14)

    if add_legend:
        lines = ln1 + ln2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, frameon=False, fontsize=12,
                  loc="upper left", ncol=2, bbox_to_anchor=(0.55, 1.05))

    if title_left is not None:
        ax.set_title(title_left, loc="left", fontsize=15)

BASE_ERA5_1 = r"D:\code\comfort-area\data\ERA5"
LANDSEA_FILE_NH = os.path.join(BASE_ERA5_1, "landsea_NH.nc")
LANDSEA_FILE_SH = os.path.join(BASE_ERA5_1, "landsea_SH.nc")

def _rename_latlon_generic(da: xr.DataArray) -> xr.DataArray:
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

def _fix_lon_like(src: xr.DataArray, target_lon: xr.DataArray) -> xr.DataArray:
    if "lon" not in src.coords:
        return src
    lon_s = src["lon"].values
    lon_t = target_lon.values
    if np.nanmax(lon_t) > 180 and np.nanmax(lon_s) <= 180:
        lon_new = (lon_s % 360.0)
        src = src.assign_coords(lon=lon_new).sortby("lon")
    elif np.nanmax(lon_t) <= 180 and np.nanmax(lon_s) > 180:
        lon_new = ((lon_s + 180.0) % 360.0) - 180.0
        src = src.assign_coords(lon=lon_new).sortby("lon")
    return src

def _read_lsm(path: str) -> xr.DataArray:
    with xr.open_dataset(path) as ds:
        if "lsm" in ds.data_vars:
            lsm = ds["lsm"]
        elif "LSMASK" in ds.data_vars:
            lsm = ds["LSMASK"]
        else:
            raise ValueError(f"{path} 中没有 lsm/LSMASK，实际变量：{list(ds.data_vars)}")
        lsm = lsm.load()
    lsm = _rename_latlon_generic(lsm)
    while lsm.ndim > 2:
        lsm = lsm.isel({lsm.dims[0]: 0})
    return lsm.astype(float)

def mask_ocean_by_landsea_nhsh(da: xr.DataArray,
                              landsea_nh: str = LANDSEA_FILE_NH,
                              landsea_sh: str = LANDSEA_FILE_SH) -> xr.DataArray:
    da = _rename_latlon_generic(da)

    lsm_nh = _read_lsm(landsea_nh)
    lsm_sh = _read_lsm(landsea_sh)

    lsm_nh = _fix_lon_like(lsm_nh, da["lon"])
    lsm_sh = _fix_lon_like(lsm_sh, da["lon"])

    lsm_nh = lsm_nh.interp(lat=da["lat"], lon=da["lon"], method="nearest")
    lsm_sh = lsm_sh.interp(lat=da["lat"], lon=da["lon"], method="nearest")

    landmask = xr.where(da["lat"] >= 0, lsm_nh, lsm_sh)
    return da.where(landmask > 0.5)

def field_corr_2d(da1: xr.DataArray, da2: xr.DataArray) -> float:
    da1, da2 = xr.align(da1, da2, join="inner")
    a = da1.values.astype(float)
    b = da2.values.astype(float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return np.nan
    x = a[m].ravel()
    y = b[m].ravel()
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])

def _debug_field(name: str, da: xr.DataArray):
    v = da.values
    fin = np.isfinite(v)
    nfin = int(fin.sum())
    ones = int((v == 1).sum()) if nfin > 0 else 0
    zeros = int((v == 0).sum()) if nfin > 0 else 0
    std = float(np.std(v[fin])) if nfin > 0 else np.nan
    print(f"[{name}] finite={nfin}, ones={ones}, zeros={zeros}, std={std:.6f}")

def _print_series_mean(tag: str, years: np.ndarray, nh: np.ndarray, sh: np.ndarray):
    mnh = float(np.nanmean(nh))
    msh = float(np.nanmean(sh))
    print(f"[Series mean] {tag:>5s}  ({int(years.min())}-{int(years.max())})  NH={mnh:.4f}   SH={msh:.4f}")

def main():
    years40 = np.arange(1981, 2021)
    years34 = np.arange(1981, 2015)

    ROWS = [
        dict(
            key="ALL", title="ALL",
            nc=os.path.join(BASE_CMIP6, "mild_area_1981_2020_MME_historical_r1i1p1f1_annual.nc"),
            nh=os.path.join(BASE_CMIP6, "mild_area_1981_2020_MME_historical_r1i1p1f1_annual_nh.txt"),
            sh=os.path.join(BASE_CMIP6, "mild_area_1981_2020_MME_historical_r1i1p1f1_annual_sh.txt"),
            years=years40, n=40
        ),
        dict(
            key="CESM2", title="CESM2",
            nc=os.path.join(BASE_CESM2, "mild_area_1981_2014_MME_BHIST_annual.nc"),
            nh=os.path.join(BASE_CESM2, "mild_area_1981_2014_MME_BHIST_annual_nh.txt"),
            sh=os.path.join(BASE_CESM2, "mild_area_1981_2014_MME_BHIST_annual_sh.txt"),
            years=years34, n=34
        ),
        dict(
            key="GHG", title="GHG",
            nc=os.path.join(BASE_CMIP6, "mild_area_1981_2020_MME_hist-GHG_r1i1p1f1_annual.nc"),
            nh=os.path.join(BASE_CMIP6, "mild_area_1981_2020_MME_hist-GHG_r1i1p1f1_annual_nh.txt"),
            sh=os.path.join(BASE_CMIP6, "mild_area_1981_2020_MME_hist-GHG_r1i1p1f1_annual_sh.txt"),
            years=years40, n=40
        ),
        dict(
            key="AA", title="AA",
            nc=os.path.join(BASE_CMIP6, "mild_area_1981_2020_MME_hist-aer_r1i1p1f1_annual.nc"),
            nh=os.path.join(BASE_CMIP6, "mild_area_1981_2020_MME_hist-aer_r1i1p1f1_annual_nh.txt"),
            sh=os.path.join(BASE_CMIP6, "mild_area_1981_2020_MME_hist-aer_r1i1p1f1_annual_sh.txt"),
            years=years40, n=40
        ),
        dict(
            key="NAT", title="NAT",
            nc=os.path.join(BASE_CMIP6, "mild_area_1981_2020_MME_hist-nat_r1i1p1f1_annual.nc"),
            nh=os.path.join(BASE_CMIP6, "mild_area_1981_2020_MME_hist-nat_r1i1p1f1_annual_nh.txt"),
            sh=os.path.join(BASE_CMIP6, "mild_area_1981_2020_MME_hist-nat_r1i1p1f1_annual_sh.txt"),
            years=years40, n=40
        ),
    ]

    # print("\n=== Time series mean (NH/SH) ===")
    # for cfg in ROWS:
    #     nh = read_1col_txt(cfg["nh"], cfg["n"])
    #     sh = read_1col_txt(cfg["sh"], cfg["n"])
    #     _print_series_mean(cfg["key"], cfg["years"], nh, sh)

    # keys_need = ["ALL", "GHG", "AA", "NAT"]
    # masks = {}
    # for cfg in ROWS:
    #     if cfg["key"] in keys_need:
    #         m01 = read_mask01_from_nc(cfg["nc"])
    #         m01 = mask_ocean_by_landsea_nhsh(m01)
    #         masks[cfg["key"]] = m01
    #
    # print("\n=== Field correlation (land-only, 0/1 masks) ===")
    # for k in keys_need:
    #     _debug_field(k, masks[k])
    #
    # pairs = [("ALL","GHG"), ("ALL","AA"), ("ALL","NAT"), ("GHG","AA"), ("GHG","NAT"), ("AA","NAT")]
    # for k1, k2 in pairs:
    #     r = field_corr_2d(masks[k1], masks[k2])
    #     print(f"{k1:>4s} vs {k2:<4s} : r = {r: .3f}")

    PLOT_ROWS = [cfg for cfg in ROWS if cfg["key"] in ["GHG", "AA", "NAT"]]

    fig = plt.figure(figsize=(7.2, 9))
    gs = fig.add_gridspec(
        nrows=len(PLOT_ROWS), ncols=1,
        hspace=0.3
    )

    for i, cfg in enumerate(PLOT_ROWS):
        ax_ts = fig.add_subplot(gs[i, 0])

        add_panel_label(ax_ts, PANEL_LABELS[i])

        nh = read_1col_txt(cfg["nh"], cfg["n"])
        sh = read_1col_txt(cfg["sh"], cfg["n"])

        xlim_pad = (cfg["years"].min() - X_PAD, cfg["years"].max() + X_PAD)

        draw_timeseries_like_ref(
            ax_ts,
            cfg["years"], nh, sh,
            TS_CFG.get(cfg["key"], {}),
            add_legend=(i == 0),
            xlim_pad=xlim_pad,
            show_xlabel=(i == len(PLOT_ROWS) - 1),
            title_left=cfg["title"]
        )

    fig.subplots_adjust(left=0.14, right=0.88, bottom=0.06, top=0.95)

    plt.savefig(OUT_PNG, dpi=800, bbox_inches="tight", pad_inches=0.05)

if __name__ == "__main__":
    main()
