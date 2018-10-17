#!/usr/bin/env python
from fractions import Fraction
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import seaborn as sns
import fluidsim as fls
import h5py

from base import (
    _k_max, _k_diss, _eps, set_figsize, matplotlib_rc, _index_where, linestyles, rev_legend)
from paths import paths_sim, exit_if_figure_exists, load_df
from make_fig_spectra import _mean_spectra

#color_list = \
#    iter(['r', 'b', 'g', 'c', 'm', 'y', 'k'])
    # iter(plt.cm.jet(pl.linspace(0,1,3)))
# style = linestyles()


def _label():
    numerator = r'2 \nu_8 k^{8} k_d E(k) / \epsilon'
    return f"${numerator}$"


def fig7_spectra(path, fig, ax, Fr, c, t_start, run_nb):
    sim = fls.load_sim_for_plot(path, merge_missing_params=True)
    kh, E_tot, EK, EA = _mean_spectra(sim, t_start)
    eps = _eps(sim, t_start)
    k_d = _k_diss(sim.params)


    o = 2
    norm = (k_d * sim.params.nu_8 * kh**8)
    color_list = sns.color_palette()

    kh_f = kh / k_d
    integral = np.trapz(2 * E_tot*norm, kh_f, dx=kh_f[1] - kh_f[0])

    kdiss_max = kh_f[np.where(
        (E_tot * norm) == (E_tot * norm).max())
    ]
    ax.plot(
        kh_f, 2 * E_tot * norm / eps, # (2 * integral),
        c=color_list[run_nb],
        linewidth=1, 
        # label=f'$c = {c}, n= {sim.params.oper.nx}$'
        label=f'{kdiss_max}'
        # label=f'int={integral}'
    )
    
    
    ax.vlines(kdiss_max, 0, 2.2, colors=color_list[run_nb], linewidth=0.5)

    # more minor ticks
    minor_locator = AutoMinorLocator(5)
    ax.xaxis.set_minor_locator(minor_locator)

    ax.set_ylim(0, 2.2)
    ax.set_xlim(0.6, 2.5) #2

    ax.set_xlabel('$k/k_d$')
    ax.set_ylabel(_label())
    # ax.legend()
    # rev_legend(ax, loc=1, fontsize=8)


def plot_df(df, fig, ax):     
    for run_nb, (idx, row) in enumerate(df.iterrows()):
        short_name = row["short name"]
        tmin = row["$t_{stat}$"]
        Fr = row["$F_f$"]
        c = row["$c$"]
        fig7_spectra(paths_sim[short_name], fig, ax, Fr, c, 
                     t_start=tmin,
                     run_nb=run_nb)


if __name__ == '__main__':
    sns.set_palette("cubehelix", 3)
    matplotlib_rc(11)
    path_fig = exit_if_figure_exists(__file__)
    set_figsize(7, 3)
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
    
    df_w = load_df("df_w")
    df_c20 = df_w[df_w["$c$"] == 20]
    df_n1920 = df_w[df_w["$n$"] == 1920]

    sns.set_palette("cubehelix", 5)
    plot_df(df_c20, fig, ax[0])
    sns.set_palette("cubehelix", 6)
    plot_df(df_n1920, fig, ax[1])
    
    ax[1].set_ylabel(None)
    arrowkwargs = dict(
        xycoords="data", arrowprops={"arrowstyle": "simple"})
    ax[0].annotate("increasing $n$", (1.5, 2.05), (0.65, 2), **arrowkwargs)
    ax[1].annotate("increasing $c$", (1, 2.05), (1.32, 2), **arrowkwargs)

    fig.tight_layout()

    fig.savefig(path_fig)
    fig.savefig(path_fig.replace(".png", ".pdf"))