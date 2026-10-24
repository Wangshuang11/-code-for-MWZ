# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


mpl.rcParams["font.family"] = "Times New Roman"
mpl.rcParams["font.size"] = 18
mpl.rcParams["axes.unicode_minus"] = False

obs_names = ["ERA5", "MERRA2", "JRA55", "NCEP2"]
cmip6_models = [
    "ACCESS-CM2",
    "ACCESS-ESM1-5",
    "CanESM5",
    "FGOALS-g3",
    "GFDL-ESM4",
    "IPSL-CM6A-LR",
    "NorESM2-LM"
]

nh_obs = np.array([-7.04, -8.76, -6.57, -12.73])
nh_cmip6 = np.array([-8.68, -10.46, -12.12, -7.68, -9.77, -9.70, -14.46])

nh_cesm2 = np.array([
    -10.60, -10.05, -10.76, -11.42, -10.80, -12.96, -10.46, -12.26, -8.63, -11.62,
    -10.91, -12.83, -13.74, -9.89, -16.01, -13.09, -10.13, -7.16, -14.19, -10.88,
    -7.76, -11.89, -13.24, -10.54, -12.44, -9.30, -13.10, -11.82, -10.99, -12.62,
    -12.78, -10.66, -10.66, -12.67, -14.18, -10.97, -9.20, -12.82, -8.39, -10.49,
    -12.04, -10.81, -7.89, -12.57, -8.75, -11.62, -13.62, -9.57, -10.81, -12.57,
    -11.88, -11.44, -10.01, -11.64, -8.86, -12.25, -7.29, -11.69, -7.89, -9.96,
    -8.02, -10.03, -12.73, -13.36, -9.89, -10.23, -15.02, -10.21, -9.19, -6.68,
    -9.53, -7.32, -8.80, -11.34, -9.34, -8.75, -8.64, -7.92, -5.58, -11.12,
    -8.99, -7.68, -12.80, -7.67, -7.46, -9.98, -10.99, -6.53, -13.43, -10.61,
    -11.53, -12.51, -8.84, -9.03, -9.99, -8.89, -10.89, -7.21, -9.35, -11.40
])

sh_obs = np.array([-2.37, -2.98, -1.99, -2.08])
sh_cmip6 = np.array([-3.46, -3.16, -4.22, -3.40, -4.02, -4.04, -2.73])

sh_cesm2 = np.array([
    -2.44, -4.07, -3.84, -3.06, -4.96, -4.03, -3.49, -3.85, -2.09, -3.34,
    -2.63, -3.68, -3.15, -2.57, -3.15, -4.86, -2.30, -2.35, -3.08, -4.65,
    -3.00, -2.59, -3.31, -3.74, -2.89, -3.28, -3.38, -2.48, -5.20, -4.41,
    -2.58, -2.52, -1.89, -3.71, -2.67, -2.84, -2.41, -3.22, -2.72, -4.49,
    -3.90, -2.09, -2.37, -3.58, -0.87, -2.70, -3.17, -2.28, -2.50, -4.16,
    -3.42, -4.71, -2.85, -3.68, -3.12, -3.60, -3.81, -3.63, -3.85, -2.13,
    -3.14, -2.40, -3.53, -2.09, -3.27, -4.33, -3.20, -3.34, -3.38, -2.86,
    -2.28, -2.94, -3.23, -3.68, -3.85, -2.67, -2.76, -3.30, -3.22, -2.98,
    -3.29, -2.50, -3.17, -2.46, -3.09, -3.36, -2.64, -2.67, -2.31, -4.03,
    -3.70, -2.82, -2.98, -2.02, -3.19, -2.98, -3.00, -2.55, -3.76, -2.77
])


color_obs = "#0078bf"
color_cmip6 = "#ee3b73"
color_cesm2 = "#54A24B"

def draw_group_marker(ax, xpos, data, marker_size=180):
    data_median = np.median(data)

    ax.scatter(
        xpos, data_median,
        marker="^",
        s=marker_size,
        color="black",
        edgecolors="black",
        zorder=5
    )


def plot_panel(ax, obs, cmip6, cesm2, panel_label, panel_title,
               ylim=None, yticks=None, show_ylabel=False):
    positions = [1, 2, 3]
    labels = ["Observation", "CMIP6", "CESM2-LENS2"]

    rng = np.random.default_rng(2025)
    jit1 = rng.normal(0, 0.045, len(obs))
    jit2 = rng.normal(0, 0.045, len(cmip6))
    jit3 = rng.normal(0, 0.06, len(cesm2))

    x_obs = np.full(len(obs), 1) + jit1
    x_cmip6 = np.full(len(cmip6), 2) + jit2
    x_cesm2 = np.full(len(cesm2), 3) + jit3

    scatter_size = 200
    scatter_alpha = 0.58

    ax.scatter(x_obs, obs, s=scatter_size, alpha=scatter_alpha,
               color=color_obs, zorder=3)
    ax.scatter(x_cmip6, cmip6, s=scatter_size, alpha=scatter_alpha,
               color=color_cmip6, zorder=3)
    ax.scatter(x_cesm2, cesm2, s=scatter_size, alpha=scatter_alpha,
               color=color_cesm2, zorder=3)


    obs_x_offsets = [0.08, 0.08, 0.08, 0.08]
    obs_y_offsets = [-0.10, 0.00, 0.05, 0.05]
    for x, y, name, dx, dy in zip(x_obs, obs, obs_names, obs_x_offsets, obs_y_offsets):
        ax.text(x + dx, y + dy, name, va="center", ha="left", fontsize=12)


    if panel_label == "a":
        cmip6_side = ['right', 'right', 'right', 'right', 'right', 'left', 'right']
        cmip6_x_offsets = [0.08, 0.06, 0.08, 0.06, 0.08, -0.08, 0.08]
        cmip6_y_offsets = [0.02, -0.05, -0.05, -0.02, -0.08, 0.02, 0.02]
    elif panel_label == "b":
        cmip6_side = ['left', 'right', 'right', 'right', 'right', 'left', 'right']
        cmip6_x_offsets = [-0.08, 0.06, 0.08, 0.06, 0.08, -0.08, 0.08]
        cmip6_y_offsets = [0.05, -0.01, -0.05, -0.02, -0.08, 0.02, 0.02]

    for x, y, name, side, dx, dy in zip(
        x_cmip6, cmip6, cmip6_models, cmip6_side, cmip6_x_offsets, cmip6_y_offsets
    ):
        ha = 'left' if side == 'right' else 'right'
        ax.text(x + dx, y + dy, name, va="center", ha=ha, fontsize=12)

    draw_group_marker(ax, 1, obs,   marker_size=180)
    draw_group_marker(ax, 2, cmip6, marker_size=180)
    draw_group_marker(ax, 3, cesm2, marker_size=180)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_xlim(0.5, 3.5)

    if ylim is not None:
        ax.set_ylim(ylim)
    if yticks is not None:
        ax.set_yticks(yticks)
        ax.set_yticklabels([str(v) for v in yticks])

    ax.text(
        0.024, 1.01, panel_title,
        transform=ax.transAxes,
        ha="center", va="bottom",
        fontsize=22
    )

    if panel_label == "a":
        ax.text(
            -0.13, 1.07, panel_label,
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=34, fontweight="bold"
        )
    elif panel_label == "b":
        ax.text(
            -0.10, 1.07, panel_label,
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=34, fontweight="bold"
        )

    if show_ylabel:
        ax.set_ylabel("Trend [%/C]")


fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=False)

plot_panel(
    axes[0], nh_obs, nh_cmip6, nh_cesm2,
    panel_label="a", panel_title="NH",
    ylim=(-18, -2),
    yticks=[-18, -16, -14, -12, -10, -8, -6, -4, -2],
    show_ylabel=True
)

plot_panel(
    axes[1], sh_obs, sh_cmip6, sh_cesm2,
    panel_label="b", panel_title="SH",
    ylim=(-7, 0),
    yticks=[-7, -6, -5, -4, -3, -2, -1, 0],
    show_ylabel=False
)

plt.tight_layout()

plt.savefig("D:/code/comfort-area/fig/v7/figS4.png", dpi=800, bbox_inches="tight")