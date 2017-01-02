#!/usr/bin/env python
import pylab as pl
import fluidsim as fls
import os

from base import _k_f, _eps, _font, set_figsize
from paths import paths_sim, path_pyfig, exit_if_figure_exists




def fig1_energy(paths, fig=None, ax=None, t_start=10., legend=None):
    if fig is None or ax is None:
        ax = pl

    if legend is None:
        legend = [os.path.basename(p) for p in paths]

    for i, path in enumerate(paths):
        sim = fls.load_sim_for_plot(path)

        P0 = _eps(sim, t_start)
        k_f = _k_f(sim.params)  # path + '/params_simul.xml')

        dico = sim.output.spatial_means.load()
        E = dico['E']
        t = dico['t']
        E_f = (P0 / k_f) ** (2. / 3)
        T_f = (P0 * k_f ** 2) ** (-1. / 3)
        E = E / E_f
        t = t / T_f
        label = legend[i]
        ax.plot(t, E, linewidth=2, label=label)
        print('{}\teps={}\tk_f={}\tE_f={}\tT_f={}'.format(
            label, P0, k_f, E_f, T_f))

    try:
        ax.xlabel('$t/T_f$', fontdict=_font)
        ax.ylabel('$E/E_f$', fontdict=_font)
    except:
        ax.set_xlabel('$t/T_f$', fontdict=_font)
        ax.set_ylabel('$E/E_f$', fontdict=_font)

    ax.legend()


def get_legend_and_paths(c_list, nh_list):
    keys = ['c{}nh{}'.format(_c, _nh) for _c in c_list for _nh in nh_list]
    return keys, [paths_sim['noise_' + k + 'Buinf'] for k in keys]


if __name__ == '__main__':
    path_fig = exit_if_figure_exists(__file__)
    set_figsize(16, 16)
    fig = pl.figure()
    ax = []
    c = [20, 100, 400]
    nh = [960, 1920, 3840, 7680]

    for i in range(4):
        ax.append(fig.add_subplot(4, 1, i + 1))
        legend, paths_subplot = get_legend_and_paths(c, nh[i:i + 1])
        fig1_energy(paths_subplot, fig, ax[i], legend=legend, t_start=20.)

    pl.savefig(path_fig)
