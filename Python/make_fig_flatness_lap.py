#!/usr/bin/env python
import pylab as pl
import seaborn as sns
import numpy as np
import fluidsim as fls

from base import _k_f, set_figsize, _rxs_str_func, matplotlib_rc, set_share_axes
from paths import paths_sim, paths_lap, exit_if_figure_exists, load_df
from make_fig_flatness import *


if __name__ == '__main__':
    matplotlib_rc(fontsize=fontsize)
    path_fig = exit_if_figure_exists(__file__)
    set_figsize(5, 4)
    fig, ax = pl.subplots(2, 2, sharey=False, sharex=False)

    df_w = load_df("df_lap")
    df_3840 = df_w[df_w["$n$"] == 3840]
    df_7680 = df_w[df_w["$n$"] == 7680]
    
    ax_inset3 = None  # _ax_inset(fig, '$r/L_f$', 0.325, 0.362)
    ax_inset7 = None  # _ax_inset(fig, '$r:/L_f$', 0.775, 0.362)

    sns.set_palette("cubehelix", 5)
    plot_df(df_3840, fig, ax[:,0], ax_inset3)
    sns.set_palette("cubehelix", 3)
    plot_df(df_7680, fig, ax[:,1], ax_inset7)

    for ax1 in ax[1,:]:
        ax1.set_yscale("linear")
        ax1.hlines([1.5], 1e-3, 10, linestyles="dashed", linewidths=(0.5,))

    for ax1 in ax.flat:
        ax1.set_xlim([1e-3, 10])

    for ax1 in ax[1,:].flat:
        ax1.set_ylim([0.7, 1.6])

    for ax1 in ax[0,:]:
        ax1.set_ylim([2, 500])
        ax1.set_xlabel(None)
        ax1.xaxis.set_tick_params(which='both', labelleft=False, labelright=False)
        ax1.xaxis.offsetText.set_visible(False)

    for ax1 in ax[:,1]:
        ax1.set_ylabel(None)
        ax1.yaxis.set_tick_params(which='both', labelleft=False, labelright=False)
        ax1.yaxis.offsetText.set_visible(False)

    ax[0,0].set_ylabel('$F_T$')
    ax[1,0].set_ylabel('$F_T/F_L$')
    ax[0,0].annotate(
        "increasing $c$",
        (3e-2, 3e0),
        (2e-1, 1e1), 
        "data",
        arrowprops={"arrowstyle": "simple"})
    
    for row in range(2):
        set_share_axes(ax[row,:], sharey=True)

    for col in range(2):
        set_share_axes(ax[:, col], sharex=True)

    fig.tight_layout()
    pl.savefig(path_fig)
    pl.savefig(path_fig.replace(".png", ".pdf"))