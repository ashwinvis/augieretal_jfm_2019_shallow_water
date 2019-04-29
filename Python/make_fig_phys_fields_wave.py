#!/usr/bin/env python
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import fluidsim as fls
from base import _k_f, set_figsize, matplotlib_rc, set_figsize, _index_where
from paths import paths_sim, exit_if_figure_exists


def fig_phys_subplot(sim, fig, ax, key_field, x_slice=None, y_slice=None, cmap='inferno',
                     vmin=None, vmax=None):
    kf = _k_f(sim.params)
    Lf = np.pi / kf
    X = sim.oper.x_seq / Lf
    Y = sim.oper.y_seq / Lf

    def slice_where(R, r1, r2):
        i1 = _index_where(R, r1)
        i2 = _index_where(R, r2)
        return slice(i1, i2)

    if x_slice is not None:
        x_slice = slice_where(X, *x_slice)
        X = X[x_slice]

    if y_slice is not None:
        y_slice = slice_where(Y, *y_slice)
        Y = Y[y_slice]

    try:
        field = sim.state.get_var(key_field)
    except ValueError:
        field = sim.oper.ifft(sim.state.get_var(key_field + '_fft'))
    
    field = field[y_slice, x_slice]

    cmap = plt.get_cmap(cmap)
    ax.set_xlabel('$x/L_f$')
    ax.set_ylabel('$y/L_f$')
    ax.set_rasterization_zorder(1)
    norm = None
    contours = ax.pcolormesh(
        X, Y, field, norm=norm, vmin=vmin, vmax=vmax, cmap=cmap, zorder=0)

    cbar = fig.colorbar(contours, ax=ax)
    cbar.solids.set_rasterized(True)
    
if __name__ == '__main__':
    matplotlib_rc(fontsize=10)
    path_fig = exit_if_figure_exists(__file__)
    set_figsize(6.65, 5.8)
    fig, axes = plt.subplots(2, 2)
    short_names = [
        'noise_c20nh1920Buinf',
        'noise_c400nh1920Buinf'
    ]
    sim0 = fls.load_state_phys_file(paths_sim[short_names[0]], merge_missing_params=True)
    sim1 = sim0
    sim1 = fls.load_state_phys_file(paths_sim[short_names[1]], merge_missing_params=True)
    keys = ['div', 'uy', 'div', 'uy']
    for ax, sim, key_field in zip(axes.ravel(), [sim0, sim0, sim1, sim1], keys):
        fig_phys_subplot(sim, fig, ax, key_field, x_slice=[0,3], y_slice=[0,3])

    label ={
        'h': 'h',
        'uy': 'u_y',
        'div': r'\nabla.\mathbf{u}'
    }
    fig.text(0.46, 0.05, r'${}$'.format(label[keys[0]]))
    fig.text(0.95, 0.05, r'${}$'.format(label[keys[1]]))
    fig.text(0.46, 0.55, r'${}$'.format(label[keys[2]]))
    fig.text(0.95, 0.55, r'${}$'.format(label[keys[3]]))
    fig.tight_layout()

    fig.savefig(path_fig, rasterized=True)
