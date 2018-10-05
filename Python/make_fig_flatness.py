#!/usr/bin/env python
import pylab as pl
import seaborn as sns
import numpy as np
import fluidsim as fls

from base import _k_f, set_figsize, _rxs_str_func, matplotlib_rc, set_share_axes
from paths import paths_sim, exit_if_figure_exists, load_df


fontsize = 9
fontsize_inset = fontsize - 2


def _ax_inset(fig, xlabel, left, bottom=0.6, width=0.35 / 2, height=0.45 / 4):
    ax_inset = fig.add_axes([left, bottom, width, height])
    ax_inset.set_xlabel(xlabel, size=fontsize_inset)
    ax_inset.set_ylabel('$F_T / F_L$', size=fontsize_inset)
    ax_inset.set_ylim([0, 1.6])
    ax_inset.set_yticks(pl.arange(0, 2., 0.5))
    ax_inset.axhline(1.5, linewidth=0.5, color='k', linestyle=':')

    # Set the tick labels font
    for ticklabel in (ax_inset.get_xticklabels() + ax_inset.get_yticklabels()):
        ticklabel.set_fontsize(fontsize_inset)

    return ax_inset


def fig12_flatness(path, fig, ax, tmin=0, tmax=1000, delta_t=0.5, 
                   key_var=('uy', 'ux'), run_nb=0, ax_inset=None, cache=False):
    sim = fls.load_sim_for_plot(path, merge_missing_params=True)
    order = [2, 4]

    rxs, So_var_dict, deltax = _rxs_str_func(
        sim, order, tmin, tmax, delta_t, key_var, cache=cache)

    ax.set_xlabel('$r/L_f$')
    # ax.set_ylabel('$F_T, F_L$')

    # ax.set_title('Flatness of longitundinal and transverse increments')
    # ax.hold(True)
    ax.set_xscale('log')
    ax.set_yscale('log')

    _label = {'ux': 'F_L', 'uy': 'F_T'}
    L_f = pl.pi / _k_f(sim.params)
    # color_list = ['r', 'b', 'g', 'c', 'm', 'r', 'b']
    color_list = sns.color_palette()
    def get_F(key):
        So_4 = So_var_dict['{0}_{1:.0f}'.format(key, 4)]
        So_2 = So_var_dict['{0}_{1:.0f}'.format(key, 2)]
        return So_4 / So_2 ** 2
    

    if len(key_var) == 1:
        key = key_var[0]
        F = get_F(key)
        label = _label[key]
        
        # r^-1 line
        cond = pl.logical_and(rxs > 0.015 * L_f, rxs < 0.17 * L_f)
        fac = 12
        ax.plot(rxs[cond] / L_f, fac / rxs[cond], 'k', linewidth=0.5)
        x_text = rxs[cond].mean() / 2 / L_f + 0.02
        y_text = fac / rxs[cond].mean() * 2
        ax.text(x_text, y_text, '$r^{-1}$')
    else:
        F = np.divide(*map(get_F, key_var))
        print("F=",F.shape)
        label = f"{_label[key_var[0]]}/{_label[key_var[1]]}"

    label = f"${label}$"
    color1 = color_list[run_nb] 
    ax.plot(rxs / L_f, F, c=color1,
            linewidth=1, label=label)

    if ax_inset is not None:
        ax_inset.semilogx(
            rxs / L_f, get_F('uy') / get_F('ux'), color_list[run_nb],
            linewidth=1)

    # ax.legend(fontsize=fontsize, loc='lower left')

    
def plot_df(df, fig, ax, ax_inset):     
    ax_a, ax_b = ax
    test_mode = False
    for run_nb, (idx, row) in enumerate(df.iterrows()):
        short_name = row["short name"]
        tmin = row["$t_{stat}$"]
        fig12_flatness(
            paths_sim[short_name], fig, ax_a, tmin=tmin, run_nb=run_nb,
            key_var=('uy',), cache=test_mode
        )
        fig12_flatness(
            paths_sim[short_name], fig, ax_b, tmin=tmin, run_nb=run_nb,
            key_var=('uy', 'ux',), cache=test_mode, ax_inset=ax_inset
        )
        if test_mode:
            break
        # fig12_atm(fig, ax_b)


if __name__ == '__main__':
    matplotlib_rc(fontsize=fontsize)
    path_fig = exit_if_figure_exists(__file__)
    set_figsize(5, 4)
    fig, ax = pl.subplots(2, 2, sharey=False, sharex=False)

    df_w = load_df("df_w")
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

    for ax1 in ax.flat:
        ax1.set_xlim([None, 10])

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