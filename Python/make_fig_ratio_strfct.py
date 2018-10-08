#!/usr/bin/env python
import pylab as pl
import seaborn as sns
import fluidsim as fls

from base import _k_f, set_figsize, _rxs_str_func, matplotlib_rc, rev_legend, palette
from paths import paths_sim, exit_if_figure_exists


def fig11_ratio_struct(path, fig, ax1, order=[2, 3, 4, 5], tmin=0, tmax=1000,
                       delta_t=0.5):
    sim = fls.load_sim_for_plot(path, merge_missing_params=True)

    key_var = ['ux', 'uy']
    rxs, So_var_dict, deltax = _rxs_str_func(
        sim, order, tmin, tmax, delta_t, key_var, force_absolute=True)

    ax1.set_xlabel('$r/L_f$')
    ax1.set_ylabel('$R_p(r)$')

    # ax1.set_title('Ratio of longitundinal and transverse struct. functions')
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
    # color_list = ['r', 'b', 'g', 'c', 'm', 'y', 'k']
    color_list = iter(sns.color_palette())
    for o in order:
        color1 = next(color_list)
        # color2 = ':' + color1
        So_ux = So_var_dict['{0}_{1:.0f}'.format('ux', o)]
        So_uy = So_var_dict['{0}_{1:.0f}'.format('uy', o)]
        ax1.plot(rxs / L_f, abs(So_ux) / abs(So_uy), c=color1,
                 linewidth=1, label='$R_{:.0f}$'.format(o))
        ax1.plot(rxs / L_f, ones * shock_model[int(o)], linestyle=":",
                 c=color1)

    ax1.set_xlim([0.001, 3])
    ax1.set_ylim([0., 11])

    rev_legend(ax, loc=1, fontsize=9)
    
    # ax1.legend()


if __name__ == '__main__':
    sns.set_palette("cubehelix", 5)
    matplotlib_rc(11)
    path_fig = exit_if_figure_exists(__file__, '.png')
    set_figsize(5, 3)
    fig, ax = pl.subplots()
    fig11_ratio_struct(
        paths_sim['noise_c20nh7680Buinf'], fig, ax, pl.arange(2, 7), tmin=10)
    fig.tight_layout()
    pl.savefig(path_fig)
    pl.savefig(path_fig.replace(".png", ".pdf"))
