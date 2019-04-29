#!/usr/bin/env python
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import fluidsim as fls
from base import (
    _k_f, set_figsize, matplotlib_rc, set_figsize, _index_where,
    load_sim as load_sim_from_path
)
from paths import paths_lap as paths_sim, exit_if_figure_exists
from make_fig_phys_fields_wave import fig_phys_subplot


load_sim = lambda short_name: load_sim_from_path(paths_sim.get(short_name), coarse=False)


if __name__ == '__main__':
    matplotlib_rc(fontsize=10)
    path_fig = exit_if_figure_exists(__file__)
    set_figsize(6.65, 5.8)
    fig, axes = plt.subplots(2, 2)

    short_names = [
          'noise_c10nh960Buinf',   # WL1
          'noise_c10nh2880Buinf',  # WL3
          'noise_c200nh960Buinf',   # WL17
          'noise_c200nh2880Buinf',  # WL18
    ]
#         'noise_c20nh2880Buinf',  # WL7
#         'noise_c200nh2880Buinf'  # WL13
#     ]
#     sim0 = load_sim(short_names[0])
#     sim1 = sim0
#     sim1 = load_sim(short_names[1])
#     keys = ['div', 'uy', 'div', 'uy']
#     for ax, sim, key_field in zip(axes.ravel(), [sim0, sim0, sim1, sim1], keys):
    keys = ['div'] * 4
    for ax, sim, key_field in zip(axes.ravel(), map(load_sim, short_names), keys):
        vmax = 10 if key_field == "div" else 3.5
        vmin = -50 if key_field == "div" else -3.5
        fig_phys_subplot(
            sim, fig, ax, key_field, x_slice=[0,3.01], y_slice=[0,3.01], vmax=vmax, vmin=vmin
        )
        sim.output.close_files()
        ax.set_xticks(np.arange(0, 4.))
        ax.set_yticks(np.arange(0, 4.))

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
