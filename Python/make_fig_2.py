#!/usr/bin/env python
import pylab as pl
import fluidsim as fls
import os
import h5py
from fluidsim.base.output.spect_energy_budget import cumsum_inv

from base import _index_where, _k_f, _eps, set_figsize
from paths import paths_sim, path_pyfig, exit_if_figure_exists


path_fig = path_pyfig + 'fig_2.png'


def fig2_seb(path, fig=None, ax=None, t_start=10):
    sim = fls.load_sim_for_plot(path)

    path_file = os.path.join(path, 'spect_energy_budg.h5')
    f = h5py.File(path_file, 'r')
    imin_plot = _index_where(f['times'][...], t_start)

    k_f = _k_f(sim.params)
    eps = _eps(sim, t_start)
    khE = (f['khE'][...] + 0.1) / k_f
    transferEKr = f['transfer2D_EKr'][imin_plot:].mean(0) / eps
    transferEKd = f['transfer2D_EKd'][imin_plot:].mean(0) / eps
    transferEAr = f['transfer2D_EAr'][imin_plot:].mean(0) / eps
    transferEAd = f['transfer2D_EAd'][imin_plot:].mean(0) / eps
    # transferEPd = f['transfer2D_EPd'][imin_plot:].mean(0) / eps

    PiEKr = cumsum_inv(transferEKr) * sim.oper.deltakh
    PiEKd = cumsum_inv(transferEKd) * sim.oper.deltakh
    PiEAr = cumsum_inv(transferEAr) * sim.oper.deltakh
    PiEAd = cumsum_inv(transferEAd) * sim.oper.deltakh
    # PiEPd = cumsum_inv(transferEPd) * sim.oper.deltakh

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
    ax.legend()


if __name__ == '__main__':
    exit_if_figure_exists(__file__)
    set_figsize(10, 6)
    fig, ax = pl.subplots()
    fig2_seb(paths_sim['noise_c100nh3840Buinf'], fig, ax, t_start=20)
    pl.savefig(path_fig)
