#!/usr/bin/env python
import pylab as pl
import fluidsim as fls
import os
import h5py
from fluidsim.base.output.spect_energy_budget import cumsum_inv

from base import _index_where, _k_f, _eps, set_figsize, matplotlib_rc, epsetstmax
from paths import paths_lap, exit_if_figure_exists
from make_fig_spect_energy_budg import fig2_seb


def fig2_seb(path, fig=None, ax=None, t_start=None):
    sim = fls.load_sim_for_plot(path, merge_missing_params=True)

    path_file = os.path.join(path, 'spect_energy_budg.h5')
    f = h5py.File(path_file, 'r')

    k_f = _k_f(sim.params)
    # eps = _eps(sim, t_start)
    eps, E, ts, tmax = epsetstmax(path)
    if t_start is None:
        t_start = ts
    imin_plot = _index_where(f['times'][...], t_start)
    khE = (f['khE'][...] + 0.1) / k_f
    transfers = 0
    for key in ('Tq_AAA', 'Tq_GAAs', 'Tq_GAAd', 'Tq_AGG', 'Tq_GGG'):
        transfers += f[key][imin_plot:].mean(0) / eps

    Pi_tot = cumsum_inv(transfers) * sim.oper.deltak

    print(eps)
    ax.axhline(1., color='k', ls=':')

    ax.set_xlabel('$k/k_f$')
    ax.set_ylabel(r'$\Pi(k)/\epsilon$')
    ax.set_xscale('log')
    ax.set_yscale('linear')
    ax.plot(khE, Pi_tot, 'k', linewidth=2, label=r'$\Pi$')

    ax.set_ylim([-0.1, 1.1])
#     ax.legend()

    
if __name__ == '__main__':
    matplotlib_rc()
    path_fig = exit_if_figure_exists(__file__)
    set_figsize(5, 3)
    fig, ax = pl.subplots()
    fig2_seb(paths_lap['noise_c10nh7680Buinf'], fig, ax, t_start=12)
    fig.tight_layout()
#     fig.savefig(path_fig)
    fig.savefig(path_fig.replace(".png", ".pdf"))
