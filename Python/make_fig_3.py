#!/usr/bin/env python
import pylab as pl
import fluidsim as fls
import os
import h5py

from base import _k_f, _eps, set_figsize
from paths import paths_sim, path_pyfig, exit_if_figure_exists


path_fig = path_pyfig + 'fig_3.png'


def fig3_struct(path, fig, ax1, tmin=0, tmax=1000):
    sim = fls.load_sim_for_plot(path)
    path_file = os.path.join(path, 'increments.h5')
    f = h5py.File(path_file, 'r')
    dset_times = f['times']
    times = dset_times[...]

    if tmax is None:
        tmax = times.max()

    rxs = f['rxs'][...]

    oper = f['/info_simul/params/oper']
    nx = oper.attrs['nx']
    Lx = oper.attrs['Lx']
    deltax = Lx / nx

    rxs = pl.array(rxs, dtype=pl.float64) * deltax

    imin_plot = pl.argmin(abs(times - tmin))
    imax_plot = pl.argmin(abs(times - tmax))

    S_uL2JL = f['struc_func_uL2JL'][imin_plot:imax_plot + 1].mean(0)
    S_uT2JL = f['struc_func_uT2JL'][imin_plot:imax_plot + 1].mean(0)
    S_c2h2uL = f['struc_func_c2h2uL'][imin_plot:imax_plot + 1].mean(0)
    S_Kolmo = f['struc_func_Kolmo'][imin_plot:imax_plot + 1].mean(0)
    # S_uT2uL = f['struc_func_uT2uL'][imin_plot:imax_plot + 1].mean(0)

    eps = _eps(sim, tmin)
    S_Kolmo_theo = -4 * eps * rxs
    Lf = pl.pi / _k_f(sim.params)

    ax1.set_xscale('log')
    ax1.set_yscale('linear')

    ax1.set_xlabel('$r/L_f$')
    ax1.set_ylabel('$S(r) / (4\epsilon r)$')

    ax1.axhline(1., color='k', ls=':')

    # ax1.plot(rxs / Lf,
    #          (S_uL2JL+S_uT2JL+S_c2h2uL)/S_Kolmo_theo,
    #          'y', linewidth=1, label='sum check')
    ax1.plot(rxs / Lf, S_Kolmo / S_Kolmo_theo, 'k', linewidth=2, label='S')
    ax1.plot(rxs / Lf, (S_uL2JL + S_uT2JL) / S_Kolmo_theo, 'r', linewidth=2,
             label='$Su^2_LJ+Su^2_TJ$')
    ax1.plot(rxs / Lf, S_c2h2uL / S_Kolmo_theo,
             'b', linewidth=2, label='$S(ch)^2u_L$')
    ax1.plot(rxs / Lf, S_uL2JL / S_Kolmo_theo,
             'r--', linewidth=1, label='$Su^2_LJ$')
    ax1.plot(rxs / Lf, S_uT2JL / S_Kolmo_theo,
             'r-.', linewidth=1, label='$Su^2_TJ$')

    cond = rxs < 6 * deltax
    ax1.plot(rxs[cond] / Lf, 1.e0 * rxs[cond] ** 3 / S_Kolmo_theo[cond],
             'y', linewidth=2, label='$r^3/S; r<6dx$')

    ax1.plot(rxs, pl.ones(rxs.shape), 'k:', linewidth=1)
    ax1.legend()


if __name__ == '__main__':
    exit_if_figure_exists(__file__)
    set_figsize(10, 6)
    fig, ax = pl.subplots()
    fig3_struct(paths_sim['noise_c20nh3840Buinf'], fig, ax, tmin=15, tmax=75)
    pl.savefig(path_fig)
