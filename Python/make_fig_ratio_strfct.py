#!/usr/bin/env python
import pylab as pl
import fluidsim as fls

from base import _k_f, set_figsize, _rxs_str_func, matplotlib_rc
from paths import paths_sim, exit_if_figure_exists


def fig11_ratio_struct(path, fig, ax1, order=[2, 3, 4, 5], tmin=0, tmax=1000, delta_t=0.5):
    sim = fls.load_sim_for_plot(path)

    key_var = ['ux', 'uy']
    rxs, So_var_dict, deltax = _rxs_str_func(
        sim, order, tmin, tmax, delta_t, key_var)

    ax1.set_xlabel('$r_x/L_f$')
    ax1.set_ylabel('$R_p(r)$')

    # ax1.set_title('Ratio of longitundinal and transverse struct. functions')
    ax1.hold(True)
    ax1.set_xscale('log')
    ones = pl.ones(rxs.shape)
    shock_model = {0: 1.,
                   1: pl.pi / 4,
                   2: 2,
                   3: 6 * pl.pi / 8,
                   4: 8. / 3,
                   5: 15. * pl.pi / 16,
                   6: 16. / 5}
    L_f = pl.pi / _k_f(sim.params)
    color_list = ['r', 'b', 'g', 'c', 'm', 'y', 'k']
    for i, o in enumerate(order):
        color1 = color_list[i]
        color2 = ':' + color1
        So_ux = So_var_dict['{0}_{1:.0f}'.format('ux', o)]
        So_uy = So_var_dict['{0}_{1:.0f}'.format('uy', o)]
        ax1.plot(rxs / L_f, abs(So_ux) / abs(So_uy), color1,
                 linewidth=2, label='$R_{:.0f}$'.format(o))
        ax1.plot(rxs / L_f, ones * shock_model[int(o)], color2)

    ax1.set_ylim([0., 10.])
    ax1.legend()


if __name__ == '__main__':
    matplotlib_rc()
    path_fig = exit_if_figure_exists(__file__)
    set_figsize(10, 6)
    fig, ax = pl.subplots()
    fig11_ratio_struct(
        paths_sim['noise_c20nh7680Buinf'], fig, ax, pl.arange(2, 7), tmin=10)
    pl.savefig(path_fig)
