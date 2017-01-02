#!/usr/bin/env python
import pylab as pl
import fluidsim as fls

from base import _k_f, set_figsize, _rxs_str_func
from paths import paths_sim, path_pyfig, exit_if_figure_exists


path_fig = path_pyfig + 'fig_12.png'


def fig12_flatness(path, fig, ax, tmin=0, tmax=1000, delta_t=0.5):
    sim = fls.load_sim_for_plot(path)
    order = [2, 4]
    key_var = ['ux', 'uy']
    rxs, So_var_dict, deltax = _rxs_str_func(
        sim, order, tmin, tmax, delta_t, key_var)

    ax.set_xlabel('$r_x/L_f$')
    ax.set_ylabel('$F_T, F_L$')

    # ax.set_title('Flatness of longitundinal and transverse increments')
    ax.hold(True)
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
                linewidth=2, label=_label[key])

    ax.set_xlim([2e-3, 10])
    ax.set_ylim([2, 500])
    ax.plot(rxs[cond] / L_f, 1 / rxs[cond], 'k', label='$r^{-1}$')
    ax_inset = fig.add_axes([0.5, 0.5, 0.3, 0.3])
    ax_inset.set_xlabel('$r_x/L_f$')
    ax_inset.set_ylabel('$F_T / F_L$')
    ax_inset.semilogx(rxs / L_f, F['uy'] / F['ux'], 'k',
                      linewidth=2)
    ax_inset.set_ylim([0, 1.6])
    ax_inset.plot(rxs / L_f, pl.ones_like(rxs) * 1.5, 'k:')
    ax.legend()


if __name__ == '__main__':
    exit_if_figure_exists(__file__)
    set_figsize(10, 6)
    fig, ax = pl.subplots()
    fig12_flatness(paths_sim['noise_c20nh7680Buinf'], fig, ax)
    pl.savefig(path_fig)
