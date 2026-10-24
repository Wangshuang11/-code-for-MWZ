# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import genextreme
from matplotlib.ticker import ScalarFormatter
import matplotlib as mpl
from matplotlib.lines import Line2D

mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 15
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'Times New Roman'
mpl.rcParams['mathtext.it'] = 'Times New Roman:italic'
mpl.rcParams['mathtext.bf'] = 'Times New Roman:bold'

PATH_NH_HIST = r"D:\code\comfort-area\data\v5\ERA5\mild_area_1981_2020_APT_annual_nh.txt"
PATH_SH_HIST = r"D:\code\comfort-area\data\v5\ERA5\mild_area_1981_2020_APT_annual_sh.txt"

PATH_NH_FUT  = r"D:\code\comfort-area\data\v5\CMIP6\ssp585\mild_area_2021_2100_MME_ssp585_annual_nh.txt"
PATH_SH_FUT  = r"D:\code\comfort-area\data\v5\CMIP6\ssp585\mild_area_2021_2100_MME_ssp585_annual_sh.txt"

HIST_BASE_START, HIST_BASE_END = 1981, 2000
HIST_NEW_START,  HIST_NEW_END  = 2001, 2020

FUT_BASE_START, FUT_BASE_END = 2021, 2040
FUT_NEW_START,  FUT_NEW_END  = 2041, 2060

EVENT_YEAR_NH = 2016
EVENT_YEAR_SH = 2016

USE_MINIMA = True

STYLE_CFG = {
    "NH": {
        "hb": dict(color="#0081c8", ls="-",  lw=3.0,  shade_alpha=0.20),
        "hn": dict(color="#00994c", ls="-",  lw=3.0,  shade_alpha=0.20),
        "fb": dict(color="#f69243", ls="-",  lw=2.5,  shade_alpha=0.12),
        "fn": dict(color="#f393be", ls="-",  lw=2.5,  shade_alpha=0.12),
        "event_line": dict(color="0.7", ls="--", lw=2.0),
        "event_marker": {
            "hb": dict(marker="^", s=42, show=True),
            "hn": dict(marker="^", s=42, show=True),
            "fb": dict(marker="^", s=42, show=True),
            "fn": dict(marker="^", s=42, show=False),
        },
        "show_event_marker": True,
        "show_event_legend": True,
    },
    "SH": {
        "hb": dict(color="#0081c8", ls="-",  lw=3.0,  shade_alpha=0.20),
        "hn": dict(color="#00994c", ls="-",  lw=3.0,  shade_alpha=0.20),
        "fb": dict(color="#f69243", ls="-",  lw=2.5,  shade_alpha=0.12),
        "fn": dict(color="#f393be", ls="-",  lw=2.5,  shade_alpha=0.12),
        "event_line": dict(color="0.7", ls="--", lw=2.0),
        "event_marker": {
            "hb": dict(marker="^", s=42, show=True),
            "hn": dict(marker="^", s=42, show=True),
            "fb": dict(marker="^", s=42, show=True),
            "fn": dict(marker="^", s=42, show=True),
        },
        "show_event_marker": True,
        "show_event_legend": True,
    }
}

Y_LIM_NH   = (6, 24)
Y_TICKS_NH = [6, 12, 18, 24]

Y_LIM_SH   = (11, 17)
Y_TICKS_SH = [11, 13, 15, 17]

def load_series_txt(path, year_start, year_end):
    years = np.arange(year_start, year_end + 1)
    arr = np.loadtxt(path)
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = arr[:, 0]
    arr = arr.astype(float).ravel()

    if len(arr) != len(years):
        raise RuntimeError(
            f"{path}\n"
            f"期望长度 {len(years)}（{year_start}-{year_end}），但读到 {len(arr)} 行。"
        )

    # print(f"{path}")
    # print(f"  year: {years[0]}–{years[-1]}, N={len(years)}")
    # print(f"  value(min,max) = {arr.min():.6f}, {arr.max():.6f} (already normalized)")
    return years, arr


def subset_range(years, area, start, end, use_minima=True):
    data = -area if use_minima else area
    mask = (years >= start) & (years <= end)
    return data[mask]

def fit_gev(sample):
    c, loc, scale = genextreme.fit(sample)
    return c, loc, scale


def return_level(T, c, loc, scale):
    T = np.asarray(T)
    p = 1.0 - 1.0 / T
    return genextreme.ppf(p, c, loc=loc, scale=scale)


def event_return_period(event_value, c, loc, scale, use_minima=True):
    x = -event_value if use_minima else event_value
    F = genextreme.cdf(x, c, loc=loc, scale=scale)
    p_exceed = 1.0 - F
    if p_exceed <= 0:
        return np.inf
    return 1.0 / p_exceed


def bootstrap_ci(sample, T, n_boot=300, seed=0):
    rng = np.random.default_rng(seed)
    n = len(sample)
    T = np.asarray(T)
    boot_rl = np.full((n_boot, T.size), np.nan, dtype=float)

    for i in range(n_boot):
        boot = rng.choice(sample, size=n, replace=True)
        try:
            c_b, loc_b, scale_b = fit_gev(boot)
            boot_rl[i, :] = return_level(T, c_b, loc_b, scale=scale_b)
        except Exception:
            pass

    low = np.nanpercentile(boot_rl, 5, axis=0)
    high = np.nanpercentile(boot_rl, 95, axis=0)
    return low, high


def debug_sample(name, sample, c, loc, scale):
    print(f"\n[{name}]")
    print("  sample size =", len(sample))
    print("  min/mean/max =", sample.min(), sample.mean(), sample.max())
    print("  GEV c, loc, scale =", c, loc, scale)


def _get_style(hemi_label: str, key: str):
    if hemi_label in STYLE_CFG and key in STYLE_CFG[hemi_label]:
        return STYLE_CFG[hemi_label][key]
    raise KeyError(f"STYLE_CFG 缺少 {hemi_label}.{key}")


def _scatter_if(ax, Tval, yval, color, mkcfg):
    if mkcfg is None:
        return
    if isinstance(mkcfg, dict) and (mkcfg.get("show", True) is False):
        return
    marker = mkcfg.get("marker", "o") if isinstance(mkcfg, dict) else "o"
    size   = mkcfg.get("s", 35) if isinstance(mkcfg, dict) else 35
    if np.isfinite(Tval):
        ax.scatter(Tval, yval, color=color, marker=marker, s=size, zorder=6)


def plot_empirical_points(ax, sample_extreme, use_minima, color,
                          marker='o', size=24, alpha=0.9):

    if use_minima:
        vals = -sample_extreme.copy()
    else:
        vals = sample_extreme.copy()

    n = len(vals)
    order = np.argsort(vals)
    vals_sorted = vals[order]
    ranks = np.arange(1, n + 1)
    p = ranks / (n + 1.0)
    T_emp = 1.0 / p

    ax.scatter(T_emp, vals_sorted, color=color,
               marker=marker, s=size, alpha=alpha, zorder=5)

def process_one_hemisphere(years_hist, area_hist,
                           years_fut, area_fut,
                           hemi_label, ax, event_year, show_legend=True):

    ev_mask = (years_hist == event_year)
    if not ev_mask.any():
        raise RuntimeError(f"{hemi_label} ERA5 历史序列里没有年份 {event_year}")
    event_val = area_hist[ev_mask].item()

    hist_base = subset_range(years_hist, area_hist, HIST_BASE_START, HIST_BASE_END, USE_MINIMA)
    hist_new  = subset_range(years_hist, area_hist, HIST_NEW_START,  HIST_NEW_END,  USE_MINIMA)

    fut_base  = subset_range(years_fut,  area_fut,  FUT_BASE_START, FUT_BASE_END, USE_MINIMA)
    fut_new   = subset_range(years_fut,  area_fut,  FUT_NEW_START,  FUT_NEW_END,  USE_MINIMA)

    c_hb, loc_hb, sc_hb = fit_gev(hist_base)

    c_hn, loc_hn, sc_hn = fit_gev(hist_new)

    c_fb, loc_fb, sc_fb = fit_gev(fut_base)

    c_fn, loc_fn, sc_fn = fit_gev(fut_new)

    T_hb = event_return_period(event_val, c_hb, loc_hb, sc_hb, USE_MINIMA)
    T_hn = event_return_period(event_val, c_hn, loc_hn, sc_hn, USE_MINIMA)
    T_fb = event_return_period(event_val, c_fb, loc_fb, sc_fb, USE_MINIMA)
    T_fn = event_return_period(event_val, c_fn, loc_fn, sc_fn, USE_MINIMA)

    # print(f"\n[{hemi_label}] Event={event_year}, value={event_val:.6f}")
    # print(f"  Return period (yr): HIST_BASE={('inf' if np.isinf(T_hb) else f'{T_hb:.1f}')}, "
    #       f"HIST_NEW={('inf' if np.isinf(T_hn) else f'{T_hn:.1f}')}, "
    #       f"FUT_BASE={('inf' if np.isinf(T_fb) else f'{T_fb:.1f}')}, "
    #       f"FUT_NEW={('inf' if np.isinf(T_fn) else f'{T_fn:.1f}')}")

    finite_T = [t for t in [T_hb, T_hn, T_fb, T_fn] if np.isfinite(t)]
    T_target = max(finite_T) if len(finite_T) else 200.0
    T_max = max(200.0, min(5000.0, 1.3 * T_target))

    T_min = 1.01
    T = np.logspace(np.log10(T_min), np.log10(T_max), 260)

    rl_hb = return_level(T, c_hb, loc_hb, sc_hb)
    rl_hn = return_level(T, c_hn, loc_hn, sc_hn)
    rl_fb = return_level(T, c_fb, loc_fb, sc_fb)
    rl_fn = return_level(T, c_fn, loc_fn, sc_fn)

    ci_hb_lo, ci_hb_hi = bootstrap_ci(hist_base, T, n_boot=300, seed=1)
    ci_hn_lo, ci_hn_hi = bootstrap_ci(hist_new,  T, n_boot=300, seed=2)
    ci_fb_lo, ci_fb_hi = bootstrap_ci(fut_base,  T, n_boot=300, seed=3)
    ci_fn_lo, ci_fn_hi = bootstrap_ci(fut_new,   T, n_boot=300, seed=4)

    if USE_MINIMA:
        rl_hb, rl_hn, rl_fb, rl_fn = -rl_hb, -rl_hn, -rl_fb, -rl_fn

        ci_hb_lo, ci_hb_hi = -ci_hb_lo, -ci_hb_hi
        ci_hn_lo, ci_hn_hi = -ci_hn_lo, -ci_hn_hi
        ci_fb_lo, ci_fb_hi = -ci_fb_lo, -ci_fb_hi
        ci_fn_lo, ci_fn_hi = -ci_fn_lo, -ci_fn_hi

        ci_hb_lo, ci_hb_hi = np.minimum(ci_hb_lo, ci_hb_hi), np.maximum(ci_hb_lo, ci_hb_hi)
        ci_hn_lo, ci_hn_hi = np.minimum(ci_hn_lo, ci_hn_hi), np.maximum(ci_hn_lo, ci_hn_hi)
        ci_fb_lo, ci_fb_hi = np.minimum(ci_fb_lo, ci_fb_hi), np.maximum(ci_fb_lo, ci_fb_hi)
        ci_fn_lo, ci_fn_hi = np.minimum(ci_fn_lo, ci_fn_hi), np.maximum(ci_fn_lo, ci_fn_hi)

    ax.set_xscale("log")
    ax.set_xlim(1, T_max)

    st_hb = _get_style(hemi_label, "hb")
    st_hn = _get_style(hemi_label, "hn")
    st_fb = _get_style(hemi_label, "fb")
    st_fn = _get_style(hemi_label, "fn")
    st_ev = _get_style(hemi_label, "event_line")
    st_mk = _get_style(hemi_label, "event_marker")
    show_mk = bool(STYLE_CFG.get(hemi_label, {}).get("show_event_marker", True))
    show_evleg = bool(STYLE_CFG.get(hemi_label, {}).get("show_event_legend", True))

    h1, = ax.plot(T, rl_hb, color=st_hb["color"], linestyle=st_hb["ls"], linewidth=st_hb["lw"],
                  label=f"{HIST_BASE_START}-{HIST_BASE_END}")
    h2, = ax.plot(T, rl_hn, color=st_hn["color"], linestyle=st_hn["ls"], linewidth=st_hn["lw"],
                  label=f"{HIST_NEW_START}-{HIST_NEW_END}")

    f1, = ax.plot(T, rl_fb, color=st_fb["color"], linestyle=st_fb["ls"], linewidth=st_fb["lw"],
                  label=f"{FUT_BASE_START}-{FUT_BASE_END}")
    f2, = ax.plot(T, rl_fn, color=st_fn["color"], linestyle=st_fn["ls"], linewidth=st_fn["lw"],
                  label=f"{FUT_NEW_START}-{FUT_NEW_END}")

    ax.fill_between(T, ci_hb_lo, ci_hb_hi, color=st_hb["color"], alpha=st_hb.get("shade_alpha", 0.20), linewidth=0)
    ax.fill_between(T, ci_hn_lo, ci_hn_hi, color=st_hn["color"], alpha=st_hn.get("shade_alpha", 0.20), linewidth=0)
    ax.fill_between(T, ci_fb_lo, ci_fb_hi, color=st_fb["color"], alpha=st_fb.get("shade_alpha", 0.12), linewidth=0)
    ax.fill_between(T, ci_fn_lo, ci_fn_hi, color=st_fn["color"], alpha=st_fn.get("shade_alpha", 0.12), linewidth=0)

    plot_empirical_points(ax, hist_base, USE_MINIMA, st_hb["color"], marker='o', size=26, alpha=0.9)
    plot_empirical_points(ax, hist_new,  USE_MINIMA, st_hn["color"], marker='o', size=26, alpha=0.9)
    plot_empirical_points(ax, fut_base,  USE_MINIMA, st_fb["color"], marker='o', size=24, alpha=0.8)
    plot_empirical_points(ax, fut_new,   USE_MINIMA, st_fn["color"], marker='o', size=24, alpha=0.8)

    if show_mk:
        _scatter_if(ax, T_hb, event_val, "#000000", st_mk.get("hb"))
        _scatter_if(ax, T_hn, event_val, "#000000", st_mk.get("hn"))
        _scatter_if(ax, T_fb, event_val, "#000000", st_mk.get("fb"))
        _scatter_if(ax, T_fn, event_val, "#000000", st_mk.get("fn"))

    ax.axhline(y=event_val, color=st_ev.get("color", "0.7"),
               linestyle=st_ev.get("ls", "--"), linewidth=st_ev.get("lw", 2.0))

    event_handle = Line2D([0], [0],
                          color=st_ev.get("color", "0.7"),
                          linestyle=st_ev.get("ls", "--"),
                          linewidth=st_ev.get("lw", 2.0),
                          label=f"{event_year}")

    ax.set_title(f"{hemi_label}",loc="left")
    ax.set_xlabel("Return period [yr]")
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)

    if show_legend:
        hist_handles = [h1, h2] + ([event_handle] if show_evleg else [])
        leg1 = ax.legend(handles=hist_handles, title="",
                         loc="upper left", frameon=False, bbox_to_anchor=(0.15, 1.03))
        ax.add_artist(leg1)

        ax.legend(handles=[f1, f2], title="",
                  loc="upper left", frameon=False, bbox_to_anchor=(0.56, 1.03))


def main():
    years_nh_hist, area_nh_hist = load_series_txt(PATH_NH_HIST, 1981, 2020)
    years_sh_hist, area_sh_hist = load_series_txt(PATH_SH_HIST, 1981, 2020)

    years_nh_fut,  area_nh_fut  = load_series_txt(PATH_NH_FUT,  2021, 2100)
    years_sh_fut,  area_sh_fut  = load_series_txt(PATH_SH_FUT,  2021, 2100)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), gridspec_kw={"wspace": 0.15})

    process_one_hemisphere(years_nh_hist, area_nh_hist,
                           years_nh_fut,  area_nh_fut,
                           "NH", axes[0], EVENT_YEAR_NH, show_legend=True)
    axes[0].set_ylabel("Percentage [%]")
    axes[0].set_ylim(Y_LIM_NH)
    axes[0].set_yticks(Y_TICKS_NH)

    process_one_hemisphere(years_sh_hist, area_sh_hist,
                           years_sh_fut,  area_sh_fut,
                           "SH", axes[1], EVENT_YEAR_SH, show_legend=False)
    axes[1].yaxis.set_label_position("right")
    axes[1].yaxis.tick_right()
    axes[1].set_ylabel("Percentage [%]")
    axes[1].set_ylim(Y_LIM_SH)
    axes[1].set_yticks(Y_TICKS_SH)

    axes[0].text(-0.16, 1.09, 'a', transform=axes[0].transAxes,
                 ha='left', va='top', fontsize=24, fontweight='bold')
    axes[1].text(-0.07, 1.09, 'b', transform=axes[1].transAxes,
                 ha='left', va='top', fontsize=24, fontweight='bold')

    plt.tight_layout()
    plt.savefig(r"D:\code\comfort-area\fig\v7\figS8.png",
                dpi=800, bbox_inches='tight', pad_inches=0.05)

if __name__ == "__main__":
    main()
