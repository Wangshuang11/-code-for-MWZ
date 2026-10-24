# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap

mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 17
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'Times New Roman'
mpl.rcParams['mathtext.it'] = 'Times New Roman:italic'
mpl.rcParams['mathtext.bf'] = 'Times New Roman:bold'


def truncate_cmap(cmap_name="Reds", minval=0.0, maxval=0.85, n=256):
    base = cm.get_cmap(cmap_name, n)
    new_colors = base(np.linspace(minval, maxval, n))
    return LinearSegmentedColormap.from_list(f"{cmap_name}_trunc", new_colors)

cmap_red_soft = truncate_cmap("Reds", 0.2, 0.85)


def plot_rect_heatmap(ax, data, xlabels, ylabels, title,
                      annot=None, vmin=0.5, vmax=1.0, cmap=cmap_red_soft):
    im = ax.imshow(
        data, vmin=vmin, vmax=vmax, interpolation="nearest",
        aspect="equal", cmap=cmap
    )

    ax.set_xticks(range(len(xlabels)), labels=xlabels)
    ax.set_yticks(range(len(ylabels)), labels=ylabels)
    # ax.set_title(title, pad=10)
    plt.setp(ax.get_xticklabels(), ha="center")
    ax.tick_params(length=0)

    if annot is None:
        annot = np.vectorize(lambda x: "" if np.isnan(x) else f"{x:.2f}")(data)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            s = annot[i, j]
            if s:
                ax.text(
                    j, i, s, ha="center", va="center",
                    fontsize=20, color="white", fontweight="bold"
                )

    ax.set_xlim(-0.5, len(xlabels) - 0.5)
    ax.set_ylim(len(ylabels) - 0.5, -0.5)

    for spine in ax.spines.values():
        spine.set_visible(False)

    return im

rean_pairs_clim = {
    ("ERA5",   "MERRA2"): 0.89,
    ("ERA5",   "JRA55"):  0.90,
    ("ERA5",   "NCEP2"):  0.78,
    ("MERRA2", "JRA55"):  0.85,
    ("MERRA2", "NCEP2"):  0.83,
    ("JRA55",  "NCEP2"):  0.79,
}
rean_pairs_ts = {
    ("ERA5",   "MERRA2"): (0.91, 0.84),
    ("ERA5",   "JRA55"):  (0.97, 0.88),
    ("ERA5",   "NCEP2"):  (0.86, 0.80),
    ("MERRA2", "JRA55"):  (0.92, 0.81),
    ("MERRA2", "NCEP2"):  (0.88, 0.72),
    ("JRA55",  "NCEP2"):  (0.85, 0.81),
}

rean_y = ["MERRA2", "JRA55", "NCEP2"]
rean_x = ["ERA5", "MERRA2", "JRA55"]

rean_clim_3x3 = np.full((3, 3), np.nan, dtype=float)
rean_ts_mean_3x3  = np.full((3, 3), np.nan, dtype=float)
rean_ts_annot_3x3 = np.full((3, 3), "", dtype=object)

for i, ry in enumerate(rean_y):
    for j, cx in enumerate(rean_x):
        if ry == cx:
            continue
        key = (cx, ry)
        if key in rean_pairs_clim:
            rean_clim_3x3[i, j] = rean_pairs_clim[key]
        if key in rean_pairs_ts:
            nh, sh = rean_pairs_ts[key]
            rean_ts_mean_3x3[i, j]  = (nh + sh) / 2.0
            rean_ts_annot_3x3[i, j] = f"{nh:.2f}/{sh:.2f}"


idx_pairs_clim = {
    ("APT",  "WBGT"): 0.74,
    ("APT",  "UTCI"): 0.70,
    ("APT",  "NET"):  0.70,
    ("WBGT", "UTCI"): 0.64,
    ("WBGT", "NET"):  0.59,
    ("UTCI", "NET"):  0.62,
}
idx_pairs_ts = {
    ("APT",  "WBGT"): (0.85, 0.80),
    ("APT",  "UTCI"): (0.85, 0.77),
    ("APT",  "NET"):  (0.82, 0.81),
    ("WBGT", "UTCI"): (0.84, 0.65),
    ("WBGT", "NET"):  (0.83, 0.73),
    ("UTCI", "NET"):  (0.92, 0.82),
}

idx_y = ["WBGT", "UTCI", "NET"]
idx_x = ["APT", "WBGT", "UTCI"]

idx_clim_3x3 = np.full((3, 3), np.nan, dtype=float)
idx_ts_mean_3x3  = np.full((3, 3), np.nan, dtype=float)
idx_ts_annot_3x3 = np.full((3, 3), "", dtype=object)

for i, ry in enumerate(idx_y):
    for j, cx in enumerate(idx_x):
        if ry == cx:
            continue
        key = (cx, ry)
        if key in idx_pairs_clim:
            idx_clim_3x3[i, j] = idx_pairs_clim[key]
        if key in idx_pairs_ts:
            nh, sh = idx_pairs_ts[key]
            idx_ts_mean_3x3[i, j]  = (nh + sh) / 2.0
            idx_ts_annot_3x3[i, j] = f"{nh:.2f}/{sh:.2f}"


fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=300)

im00 = plot_rect_heatmap(axes[0, 0], rean_clim_3x3, rean_x, rean_y, "Reanalyses: Climatology")
im01 = plot_rect_heatmap(axes[0, 1], rean_ts_mean_3x3, rean_x, rean_y, "Reanalysis: Time series (NH/SH)",
                         annot=rean_ts_annot_3x3)
im10 = plot_rect_heatmap(axes[1, 0], idx_clim_3x3, idx_x, idx_y, "Indices: Climatology")
im11 = plot_rect_heatmap(axes[1, 1], idx_ts_mean_3x3, idx_x, idx_y, "Indices: Time series (NH/SH)",
                         annot=idx_ts_annot_3x3)

panel_map = {(0, 0): "a", (1, 0): "b", (0, 1): "c", (1, 1): "d"}
for (r, c), lab in panel_map.items():
    axes[r, c].text(
        -0.28, 1.08, lab,
        transform=axes[r, c].transAxes,
        ha="left", va="top",
        fontsize=28, fontweight="bold", color="black"
    )

cbar = fig.colorbar(
    im11, ax=axes.ravel().tolist(),
    orientation="horizontal",
    fraction=0.05, pad=0.08
)

pos = cbar.ax.get_position()
cbar.ax.set_position([pos.x0, pos.y0 - 0.17, pos.width, pos.height])

fig.tight_layout()
plt.savefig(r"D:\code\comfort-area\fig\v7\figS2.png",
            dpi=800, bbox_inches='tight', pad_inches=0.05)
