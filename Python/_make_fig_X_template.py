#!/usr/bin/env python
import pylab as pl
import fluidsim as fls
import os
import h5py

from base import _index_where, _k_f, _eps, set_figsize
from paths import paths_sim, path_pyfig, exit_if_figure_exists


path_fig = path_pyfig + 'fig_X.png'


def figX_(path, fig=None, ax=None, t_start=10):
    pass
    
if __name__ == '__main__':
    exit_if_figure_exists(__file__)
    set_figsize(10, 6)
    fig, ax = pl.subplots()
    figX_seb(paths_sim['noise_c100nh3840Buinf'], fig, ax, t_start=20)
    pl.savefig(path_fig)
