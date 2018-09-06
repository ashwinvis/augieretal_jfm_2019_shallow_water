#!/usr/bin/env python
import pylab as pl
import fluidsim as fls

from base import _k_f, set_figsize, _rxs_str_func, matplotlib_rc
from paths import paths_sim, exit_if_figure_exists


fontsize = 7
fontsize_inset = fontsize - 2


def _ax_inset(fig, xlabel, left, bottom=0.6, width=0.35 / 2, height=0.45 / 2):
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


def fig12_flatness(path, fig, ax, tmin=0, tmax=1000, delta_t=0.5):
    sim = fls.load_sim_for_plot(path, merge_missing_params=True)
    order = [2, 4]
    key_var = ['uy', 'ux']
    rxs, So_var_dict, deltax = _rxs_str_func(
        sim, order, tmin, tmax, delta_t, key_var)

    ax.set_xlabel('$r_x/L_f$')
    ax.set_ylabel('$F_T, F_L$')

    # ax.set_title('Flatness of longitundinal and transverse increments')
    # ax.hold(True)
    ax.set_xscale('log')
    ax.set_yscale('log')

    _label = {'ux': '$F_L$', 'uy': '$F_T$'}
    L_f = pl.pi / _k_f(sim.params)
    color_list = ['r', 'b', 'g', 'c', 'm', 'r', 'b']
    cond = pl.logical_and(rxs > 0.012 * L_f, rxs < 0.06 * L_f)
    F = {}
    for i, key in enumerate(key_var):
        color1 = color_list[i]
        So_4 = So_var_dict['{0}_{1:.0f}'.format(key, 4)]
        So_2 = So_var_dict['{0}_{1:.0f}'.format(key, 2)]
        F[key] = So_4 / So_2 ** 2
        ax.plot(rxs / L_f, F[key], color1,
                linewidth=1, label=_label[key])

    ax.set_xlim([2e-3, 10])
    ax.set_ylim([2, 500])
    ax.plot(rxs[cond] / L_f, 1 / rxs[cond], 'k', linewidth=0.5)
    x_text = rxs[cond].mean() / L_f
    y_text = 1. / rxs[cond].mean() * 1.05
    ax.text(x_text, y_text, '$r^{-1}$')
    ax_inset = _ax_inset(fig, '$r_x/L_f$', 0.25 + 0.025)
    ax_inset.semilogx(rxs / L_f, F['uy'] / F['ux'], 'k',
                      linewidth=1)

    ax.legend(fontsize=fontsize, loc='lower left')


def fig12_atm(fig, ax):
    ax.set_xlabel('$r$ (km)')
    ax.set_ylabel('$F_T, F_L$')
    ax.hold(True)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim([1, 2e3])
    ax.set_ylim([2, 500])
    # ax_inset = _ax_inset(fig, '$r$', 0.75 + 0.01)


if __name__ == '__main__':
    matplotlib_rc(fontsize=fontsize)
    path_fig = exit_if_figure_exists(__file__)
    set_figsize(5.12, 2.)
    fig, ax = pl.subplots(ncols=2)
    ax_a, ax_b = ax.ravel()
    fig.tight_layout(pad=2)
    fig12_flatness(paths_sim['noise_c20nh7680Buinf'], fig, ax_a, tmin=20)
    fig12_atm(fig, ax_b)
    pl.savefig(path_fig)
