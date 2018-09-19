import pylab as pl
import fluidsim as fls
import os
import h5py

from base import _k_f, _eps, set_figsize, matplotlib_rc
from paths import paths_sim, exit_if_figure_exists

from make_fig_spect_energy_budg import fig2_seb
from make_fig_Kolmo import fig3_struct


if __name__ == '__main__':
    matplotlib_rc(fontsize=12)
    path_fig = exit_if_figure_exists(__file__, '.pdf')
    set_figsize(16, 6)
    fig, ax = pl.subplots(1, 2)
    run = 'noise_c20nh3840Buinf'
    fig2_seb(paths_sim[run], fig, ax[0], t_start=20)
    fig3_struct(paths_sim[run], fig, ax[1], tmin=20, tmax=75)
    fig.tight_layout()
    pl.savefig(path_fig)