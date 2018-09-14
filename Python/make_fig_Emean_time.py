#!/usr/bin/env python
import pylab as pl
import os
import itertools
from collections import OrderedDict
import fluidsim as fls
from fluiddyn.io import stdout_redirected

from base import _k_f, _eps, set_figsize, matplotlib_rc
from paths import keyparams_from_path, paths_sim, exit_if_figure_exists


fontsize = 8


def fig1_plot_all(paths):
    c_nh = OrderedDict.fromkeys([10, 20, 40, 100, 400, 700])
    c_nh = {c: [] for c in c_nh.keys()}
    for path in paths.values():
        init_field, c, nh, Bu, efr = keyparams_from_path(path)
        if init_field == "noise" and Bu == "inf":
            c_nh[int(c)].append(int(nh))
            c_nh[int(c)].sort(reverse=True)

    c_nh_skip = [(400, 7680), (40, 7680)]

    fig, ax = pl.subplots()

    styles = itertools.product("cmkrgb", ("-", "--", "-.", ":"))
    for c, nh_list in sorted(c_nh.items(), key=lambda t: t[0], reverse=True):
        count_nh = list(range(4))

        def count_matches_nh(it):
            print(len(count_nh))
            
            count_nh.pop()
            return len(count_nh) > len(nh_list)

        for nh, style in zip(
            nh_list, itertools.dropwhile(count_matches_nh, styles)
        ):
            # style = "".join(style)
            style = "k-"
            if (c, nh) in c_nh_skip:
                style = "".join(next(styles))
                continue

            legend, paths = get_legend_and_paths([c], [nh])
            fig1_energy(paths, fig, ax, legend=legend, t_start=30., linestyle=style)

    ax_settings(ax)


def fig1_energy(paths, fig=None, ax=None, t_start=0., legend=None, linestyle=None):
    fig.tight_layout(pad=2)
    # fig.subplots_adjust(right=0.78)

    if legend is None:
        legend = [os.path.basename(p) for p in paths]

    for i, path in enumerate(paths):
        # with stdout_redirected():
        sim = fls.load_sim_for_plot(path, merge_missing_params=True)

        P0 = _eps(sim, t_start)
        k_f = _k_f(sim.params)  # path + '/params_simul.xml')

        dico = sim.output.spatial_means.load()
        E = dico["E"]
        t = dico["t"]
        E_f = (P0 / k_f) ** (2. / 3)
        T_f = (P0 * k_f ** 2) ** (-1. / 3)
        # E = E / E_f
        t = t / T_f

        label = legend[i]

        ax.plot(t, E, linestyle, linewidth=1., label=label)

        print("{}\teps={}\tk_f={}\tE_f={}\tT_f={}".format(label, P0, k_f, E_f, T_f))

    # ax.text(t.max(), E.max(), label)


def ax_settings(ax):
    ax.set_xlim([0., None])
    ax.set_ylim([0., 23.])
    ax.set_xlabel("$t/T_f$")
    # ax.set_ylabel("$E/E_f$")
    ax.set_ylabel("$E$")
    # ax.grid(True, axis='y', linestyle=':')

    # ax.legend(fontsize=fontsize - 1)
    # ax.legend(fontsize=fontsize - 1, bbox_to_anchor=(1.01, 1.))


def get_legend_and_paths(c_list, nh_list):
    keys = []
    legend = []
    for c in c_list:
        for nh in nh_list:
            keys.append("c{}nh{}".format(c, nh))
            legend.append("c={}, n={}".format(c, nh))

    return legend, [paths_sim["noise_" + k + "Buinf"] for k in keys]


if __name__ == "__main__":
    matplotlib_rc(fontsize)
    path_fig = exit_if_figure_exists(__file__, ".pdf", override_exit=False)
    set_figsize(5.12, 3.0)
    fig1_plot_all(paths_sim)
    pl.savefig(path_fig)
