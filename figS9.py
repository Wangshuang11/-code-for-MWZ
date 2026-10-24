# -*- coding: utf-8 -*-
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.gridspec import GridSpec

SCEN   = "ssp585"
WINDOW = "21yr"
METRIC = "mild"

mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 14
mpl.rcParams['axes.linewidth'] = 1.0

mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'Times New Roman'
mpl.rcParams['mathtext.it'] = 'Times New Roman:italic'
mpl.rcParams['mathtext.bf'] = 'Times New Roman:bold'

COLOR_NH = '#f393be'
COLOR_SH = '#00a4d2'
COLOR_G  = '#7bc67b'

COLOR_NH_label = '#ee3b73'
COLOR_SH_label = '#1f78b4'

LABEL_NH = 'NH'
LABEL_SH = 'SH'

ERR_COLOR = '#000000'

SCALE_AB = 1.0 / 100.0

def select_dataset(scen, window, metric):
    scen   = scen.lower()
    window = window.lower()
    metric = metric.lower()

    def zeros4():
        return np.zeros(4, dtype=float)

    if metric == "mild":
        ref_NH = 1.499e+04
        ref_SH = 3.846e+03

        if scen == "ssp585":
            s2_NH, s3_NH, s4_NH = 8.319e+03, 7.023e+03, 5.358e+03
            s2_SH, s3_SH, s4_SH = 3.240e+03, 3.148e+03, 2.865e+03

            f2_NH, f3_NH, f4_NH = 1.000e+04, 1.155e+04, 1.336e+04
            f2_SH, f3_SH, f4_SH = 4.083e+03, 5.503e+03, 7.067e+03

            data_c = np.array([
                [-128.647, -161.810, -115.039, -105.933, -202.364, -173.013, -153.651, -136.442, -234.510, -272.751],
                [ -67.587,  -89.098,  -75.924,  -58.138, -132.543, -115.034, -109.122, -107.189, -191.415, -228.445],
                [  -1.767,  -15.400,  -10.377,    3.333,  -52.893,  -54.579,  -80.182,  -73.384, -154.494, -183.704],
            ])

        elif scen == "ssp370":

            s2_NH, s3_NH, s4_NH = 8.602e+03, 6.920e+03, 5.193e+03
            s2_SH, s3_SH, s4_SH = 3.240e+03, 3.139e+03, 2.831e+03

            f2_NH, f3_NH, f4_NH = 1.010e+04, 1.133e+04, 1.309e+04
            f2_SH, f3_SH, f4_SH = 4.078e+03, 5.473e+03, 6.931e+03

            data_c = np.array([
                [-123.161, -158.023, -112.994, -109.125, -199.388, -166.588, -156.249, -141.160, -241.608, -281.601],
                [ -64.741,  -93.438,  -84.539,  -65.712, -137.546, -116.325, -126.708, -112.530, -197.656, -224.827],
                [ -10.078,  -28.752,  -23.012,   -8.230,  -62.679,  -69.476,  -87.852,  -78.631, -147.758, -182.704],
            ])
        else:
            raise ValueError("SCEN must be 'ssp585' or 'ssp370'")

        y_NH_left = np.array([ref_NH, s2_NH - ref_NH, s3_NH - ref_NH, s4_NH - ref_NH], dtype=float)
        y_SH_left = np.array([ref_SH, s2_SH - ref_SH, s3_SH - ref_SH, s4_SH - ref_SH], dtype=float)

        y_NH_right = np.array([ref_NH, f2_NH - ref_NH, f3_NH - ref_NH, f4_NH - ref_NH], dtype=float)
        y_SH_right = np.array([ref_SH, f2_SH - ref_SH, f3_SH - ref_SH, f4_SH - ref_SH], dtype=float)

        if scen == "ssp585":
            err_NH_left = np.array([0.0, 1.990455e+03, 2.148058e+03, 2.445323e+03], dtype=float)
            err_SH_left = np.array([0.0, 7.824873e+02/2, 7.842600e+02/2, 8.118664e+02], dtype=float)

            err_NH_right = np.array([0.0, 2.245573e+03/3, 2.705478e+03/3, 4.002181e+03/3], dtype=float)
            err_SH_right = np.array([0.0, 9.890613e+02, 1.322639e+03, 1.593618e+03], dtype=float)
        else:
            err_NH_left = np.array([0.0, 1.867424e+03, 2.099112e+03, 2.459795e+03], dtype=float)
            err_SH_left = np.array([0.0, 7.825112e+02/2, 7.850874e+02/2, 8.276434e+02], dtype=float)

            err_NH_right = np.array([0.0, 2.111448e+03/3, 2.656919e+03/3, 4.064400e+03/3], dtype=float)
            err_SH_right = np.array([0.0, 9.886818e+02, 1.323374e+03, 1.569914e+03], dtype=float)

    elif metric == "unmild":

        ref_NH = 1.872e+04
        ref_SH = 1.156e+03

        if scen == "ssp585":

            s2_NH, s3_NH, s4_NH = 2.539e+04, 2.669e+04, 2.835e+04
            s2_SH, s3_SH, s4_SH = 1.762e+03, 1.853e+03, 2.137e+03


            f2_NH, f3_NH, f4_NH = 2.457e+04, 3.485e+04, 5.244e+04
            f2_SH, f3_SH, f4_SH = 1.675e+03, 2.289e+03, 4.018e+03

            data_c = np.array([
                [128.647, 161.810, 115.039, 105.933, 202.364, 173.013, 153.651, 136.442, 234.510, 272.751],
                [ 67.587,  89.098,  75.924,  58.138, 132.543, 115.034, 109.122, 107.189, 191.415, 228.445],
                [  1.767,  15.400,  10.377,  -3.333,  52.893,  54.579,  80.182,  73.384, 154.494, 183.704],
            ])

        elif scen == "ssp370":

            s2_NH, s3_NH, s4_NH = 2.511e+04, 2.679e+04, 2.852e+04
            s2_SH, s3_SH, s4_SH = 1.762e+03, 1.862e+03, 2.171e+03


            f2_NH, f3_NH, f4_NH = 2.448e+04, 3.507e+04, 5.271e+04
            f2_SH, f3_SH, f4_SH = 1.680e+03, 2.318e+03, 4.154e+03

            data_c = np.array([
                [123.161, 158.023, 112.994, 109.125, 199.388, 166.588, 156.249, 141.160, 241.608, 281.601],
                [ 64.741,  93.438,  84.539,  65.712, 137.546, 116.325, 126.708, 112.530, 197.656, 224.827],
                [ 10.078,  28.752,  23.012,   8.230,  62.679,  69.476,  87.852,  78.631, 147.758, 182.704],
            ])
        else:
            raise ValueError("SCEN must be 'ssp585' or 'ssp370'")


        y_NH_left = np.array([ref_NH, s2_NH - ref_NH, s3_NH - ref_NH, s4_NH - ref_NH], dtype=float)
        y_SH_left = np.array([ref_SH, s2_SH - ref_SH, s3_SH - ref_SH, s4_SH - ref_SH], dtype=float)


        y_NH_right = np.array([ref_NH, f2_NH - ref_NH, f3_NH - ref_NH, f4_NH - ref_NH], dtype=float)
        y_SH_right = np.array([ref_SH, f2_SH - ref_SH, f3_SH - ref_SH, f4_SH - ref_SH], dtype=float)

        if scen == "ssp585":

            err_NH_left = np.array([0.0, 1.990455e+03, 2.148058e+03, 2.445323e+03], dtype=float)
            err_SH_left = np.array([0.0, 7.824873e+02/2, 7.842600e+02/2, 8.118664e+02/2], dtype=float)

            err_NH_right = np.array([0.0, 2.728257e+03/2, 4.306460e+03/2, 7.865131e+03/5], dtype=float)
            err_SH_right = np.array([0.0, 7.050191e+02/2, 8.648269e+02/2, 1.505184e+03/2], dtype=float)
        else:

            err_NH_left = np.array([0.0, 1.867424e+03, 2.099112e+03, 2.459795e+03], dtype=float)
            err_SH_left = np.array([0.0, 7.825112e+02/2, 7.850874e+02/2, 8.276434e+02/2], dtype=float)

            err_NH_right = np.array([0.0, 2.653595e+03/2, 4.276930e+03/2, 7.904101e+03/5], dtype=float)
            err_SH_right = np.array([0.0, 7.050899e+02/2, 8.700432e+02/2, 1.672275e+03/2], dtype=float)

    else:
        raise ValueError("METRIC must be 'mild' or 'unmild'")

    return dict(
        y_NH_left=y_NH_left, y_SH_left=y_SH_left,
        err_NH_left=err_NH_left, err_SH_left=err_SH_left,
        y_NH_right=y_NH_right, y_SH_right=y_SH_right,
        err_NH_right=err_NH_right, err_SH_right=err_SH_right,
        data_c=data_c,
    )

def plot_ref_only(ax, y_NH0, y_SH0,
                  ylim=None, yticks=None,
                  add_legend=False):
    bar_width = 0.32
    x0 = 0.0

    ax.bar(x0 - bar_width/2, y_NH0, bar_width, color=COLOR_NH, linewidth=0)
    ax.bar(x0 + bar_width/2, y_SH0, bar_width, color=COLOR_SH, linewidth=0)

    ax.set_xlim(-0.47, 0.47)
    ax.set_xticks([x0])
    ax.set_xticklabels(['ref'],fontsize=13)
    ax.axhline(0, color='k', linewidth=1)

    if ylim is not None:
        ax.set_ylim(*ylim)
    if yticks is not None:
        ax.set_yticks(yticks)

    if add_legend:
        handles = [
            Patch(facecolor=COLOR_NH, label=LABEL_NH),
            Patch(facecolor=COLOR_SH, label=LABEL_SH),
        ]
        ax.legend(handles=handles, frameon=False, fontsize=11,
                  handlelength=1, handleheight=0.5,
                  loc='upper center', ncol=1, bbox_to_anchor=(0.81, 1.05),
                  labelspacing=0.2, handletextpad=0.5)


def plot_deltas_only_dualaxis(ax, y_NH, y_SH, e_NH, e_SH,
                             labels,
                             ylim_left=None, yticks_left=None,
                             ylim_right=None, yticks_right=None):
    bar_width = 0.32
    x = np.arange(len(labels))

    ax.bar(
        x - bar_width/2, y_NH, bar_width,
        color=COLOR_NH, yerr=e_NH, linewidth=0,
        error_kw=dict(ecolor=ERR_COLOR, capsize=3, lw=1)
    )
    ax.axhline(0, color='k', linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(labels,fontsize=13)

    ax_r = ax.twinx()
    ax_r.bar(
        x + bar_width/2, y_SH, bar_width,
        color=COLOR_SH, yerr=e_SH, linewidth=0,
        error_kw=dict(ecolor=ERR_COLOR, capsize=3, lw=1)
    )
    ax_r.axhline(0, color='k', linewidth=1)

    if ylim_left is not None:
        ax.set_ylim(*ylim_left)
    if yticks_left is not None:
        ax.set_yticks(yticks_left)

    if ylim_right is None and ylim_left is not None:
        ylim_right = ylim_left
    if yticks_right is None and yticks_left is not None:
        yticks_right = yticks_left

    if ylim_right is not None:
        ax_r.set_ylim(*ylim_right)
    if yticks_right is not None:
        ax_r.set_yticks(yticks_right)

    ax.tick_params(axis='y', colors=COLOR_NH_label)
    ax_r.tick_params(axis='y', colors=COLOR_SH_label)

    return ax_r


def main():
    gwl_labels  = ['4°C', '3°C', '2°C']
    gdp_labels  = ['0–10', '10–20', '20–30', '30–40', '40–50',
                   '50–60', '60–70', '70–80', '80–90', '90–100']

    ds = select_dataset(SCEN, WINDOW, METRIC)

    y_NH_left   = ds["y_NH_left"]
    y_SH_left   = ds["y_SH_left"]
    err_NH_left = ds["err_NH_left"]
    err_SH_left = ds["err_SH_left"]

    y_NH_right   = ds["y_NH_right"]
    y_SH_right   = ds["y_SH_right"]
    err_NH_right = ds["err_NH_right"]
    err_SH_right = ds["err_SH_right"]

    data_c = ds["data_c"]

    y_NH_left_p   = y_NH_left   * SCALE_AB
    y_SH_left_p   = y_SH_left   * SCALE_AB
    err_NH_left_p = err_NH_left * SCALE_AB
    err_SH_left_p = err_SH_left * SCALE_AB

    y_NH_right_p   = y_NH_right   * SCALE_AB
    y_SH_right_p   = y_SH_right   * SCALE_AB
    err_NH_right_p = err_NH_right * SCALE_AB
    err_SH_right_p = err_SH_right * SCALE_AB

    labels_delta = ['2°C - ref', '3°C - ref', '4°C - ref']

    ylim_ref_a   = (0, 240)
    yticks_ref_a = np.arange(0, 241, 80)

    ylim_ref_b   = (0, 240)
    yticks_ref_b = np.arange(0, 241, 80)


    if METRIC == "mild":
        ylim_a_del_NH   = (-180, 0)
        yticks_a_del_NH = np.arange(-180, 1, 60)

        ylim_a_del_SH   = (-60, 0)
        yticks_a_del_SH = np.arange(-60, 1, 20)

        ylim_b_del_NH   = (-60, 60)
        yticks_b_del_NH = np.arange(-60, 61, 30)

        ylim_b_del_SH   = (-60, 60)
        yticks_b_del_SH = np.arange(-60, 61, 30)

    else:
        ylim_a_del_NH = (0, 210)
        yticks_a_del_NH = np.arange(0, 211, 70)

        ylim_a_del_SH = (0, 45)
        yticks_a_del_SH = np.arange(0, 46, 15)

        ylim_b_del_NH   = (0, 360)
        yticks_b_del_NH = np.arange(0, 361, 120)

        ylim_b_del_SH   = (0, 60)
        yticks_b_del_SH = np.arange(0, 61, 20)

    fig2 = plt.figure(figsize=(8, 9.8))
    gs = GridSpec(
        3, 2, figure=fig2,
        height_ratios=[1, 1, 1.15],
        width_ratios=[0.50, 1.50],
        wspace=0.35, hspace=0.35
    )

    ax_a_ref = fig2.add_subplot(gs[0, 0])
    ax_a_del = fig2.add_subplot(gs[0, 1])

    ax_b_ref = fig2.add_subplot(gs[1, 0])
    ax_b_del = fig2.add_subplot(gs[1, 1])

    ax_c = fig2.add_subplot(gs[2, :])

    ax_a_ref.text(-0.50, 1.02, 'a', transform=ax_a_ref.transAxes,
                  fontsize=22, fontweight='bold', ha='right', va='bottom')
    ax_b_ref.text(-0.50, 1.02, 'b', transform=ax_b_ref.transAxes,
                  fontsize=22, fontweight='bold', ha='right', va='bottom')
    ax_c.text(-0.111, 1.02, 'c', transform=ax_c.transAxes,
              fontsize=22, fontweight='bold', ha='right', va='bottom')

    plot_ref_only(
        ax_a_ref,
        y_NH_left_p[0], y_SH_left_p[0],
        ylim=ylim_ref_a, yticks=yticks_ref_a,
        add_legend=True
    )
    ax_a_ref.set_title(r'', loc='left')
    ax_a_ref.set_ylabel(r'Exposure [$10^{12}$ person·km$^2$]', fontsize=13)

    ax_a_del_r = plot_deltas_only_dualaxis(
        ax_a_del,
        y_NH_left_p[1:], y_SH_left_p[1:],
        err_NH_left_p[1:], err_SH_left_p[1:],
        labels_delta,
        ylim_left=ylim_a_del_NH,   yticks_left=yticks_a_del_NH,
        ylim_right=ylim_a_del_SH,  yticks_right=yticks_a_del_SH
    )
    ax_a_del.set_title(r'Fixed population', loc='left')
    ax_a_del.set_ylabel(r'$\Delta$Exposure [$10^{12}$ person·km$^2$]', color=COLOR_NH_label, fontsize=13)
    ax_a_del_r.set_ylabel(r'$\Delta$Exposure [$10^{12}$ person·km$^2$]', color=COLOR_SH_label, fontsize=13)

    plot_ref_only(
        ax_b_ref,
        y_NH_right_p[0], y_SH_right_p[0],
        ylim=ylim_ref_b, yticks=yticks_ref_b,
        add_legend=False
    )
    ax_b_ref.set_title(r'', loc='left')
    ax_b_ref.set_ylabel(r'Exposure [$10^{12}$ person·km$^2$]', fontsize=13)

    ax_b_del_r = plot_deltas_only_dualaxis(
        ax_b_del,
        y_NH_right_p[1:], y_SH_right_p[1:],
        err_NH_right_p[1:], err_SH_right_p[1:],
        labels_delta,
        ylim_left=ylim_b_del_NH,   yticks_left=yticks_b_del_NH,
        ylim_right=ylim_b_del_SH,  yticks_right=yticks_b_del_SH
    )
    ax_b_del.set_title(r'Future population', loc='left')
    ax_b_del.set_ylabel(r'$\Delta$Exposure [$10^{12}$ person·km$^2$]', color=COLOR_NH_label, fontsize=13)
    ax_b_del_r.set_ylabel(r'$\Delta$Exposure [$10^{12}$ person·km$^2$]', color=COLOR_SH_label, fontsize=13)

    vmax = 250
    vmin = -250
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

    cmap_orangeRed2orangeBlue = LinearSegmentedColormap.from_list(
        "OrangeRed2OrangeBlue",
        ["#0078bf", "#2e91ca", "#74add1", "#afdfef", "#ebf6f9",
         "#fcf6f2", "#ffdbc7", "#ffc09e", "#ff9c66", "#ff7c44"]
    )

    im2 = ax_c.imshow(data_c, aspect='auto', cmap=cmap_orangeRed2orangeBlue, norm=norm)

    ax_c.set_xticks(np.arange(len(gdp_labels)))
    ax_c.set_xticklabels(gdp_labels)
    ax_c.set_xlabel('GDP per capita percentile', fontsize=13)
    ax_c.set_ylabel(r'$\Delta$Area [$10^{4}$ km$^2$]', fontsize=13)

    ax_c.set_yticks(np.arange(len(gwl_labels)))
    ax_c.set_yticklabels(gwl_labels)

    for i in range(data_c.shape[0]):
        for j in range(data_c.shape[1]):
            ax_c.text(j, i, f'{data_c[i, j]:.2f}',
                      ha='center', va='center', fontsize=13, color='black')

    plt.savefig(f"D:/code/comfort-area/fig/v7/figS9.png", dpi=800, bbox_inches="tight")

if __name__ == "__main__":
    main()
