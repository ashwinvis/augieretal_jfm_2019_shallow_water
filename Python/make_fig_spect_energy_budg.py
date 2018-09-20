#!/usr/bin/env python
import pylab as pl
import fluidsim as fls
import os
import h5py
from fluidsim.base.output.spect_energy_budget import cumsum_inv

from base import _index_where, _k_f, _eps, set_figsize, matplotlib_rc, epsetstmax
from paths import paths_sim, exit_if_figure_exists


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
    transferEKr = f['transfer2D_EKr'][imin_plot:].mean(0) / eps
    transferEKd = f['transfer2D_EKd'][imin_plot:].mean(0) / eps
    transferEAr = f['transfer2D_EAr'][imin_plot:].mean(0) / eps
    transferEAd = f['transfer2D_EAd'][imin_plot:].mean(0) / eps
    # transferEPd = f['transfer2D_EPd'][imin_plot:].mean(0) / eps

    PiEKr = cumsum_inv(transferEKr) * sim.oper.deltak
    PiEKd = cumsum_inv(transferEKd) * sim.oper.deltak
    PiEAr = cumsum_inv(transferEAr) * sim.oper.deltak
    PiEAd = cumsum_inv(transferEAd) * sim.oper.deltak
    # PiEPd = cumsum_inv(transferEPd) * sim.oper.deltak

    print(eps)
    ax.axhline(1., color='k', ls=':')
    PiEK = (PiEKr + PiEKd)
    PiEA = (PiEAr + PiEAd)
    PiE = (PiEK + PiEA)

    ax.set_xlabel('$k/k_f$')
    ax.set_ylabel(r'$\Pi(k)/\epsilon$')
    ax.set_xscale('log')
    ax.set_yscale('linear')
    ax.plot(khE, PiE, 'k', linewidth=2, label=r'$\Pi$')
    ax.plot(khE, PiEK, 'r', linewidth=2, label=r'$\Pi_K$')
    ax.plot(khE, PiEA, 'b', linewidth=2, label=r'$\Pi_A$')

    ax.set_ylim([-0.1, 1.1])
    ax.legend()


if __name__ == '__main__':
    matplotlib_rc()
    path_fig = exit_if_figure_exists(__file__)
    set_figsize(5, 3)
    fig, ax = pl.subplots()
    fig2_seb(paths_sim['noise_c100nh3840Buinf'], fig, ax)  # , t_start=20)
    pl.savefig(path_fig)
