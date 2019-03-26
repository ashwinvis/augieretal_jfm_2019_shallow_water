#!/usr/bin/env python
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import fluidsim as fls
from base import _k_f, set_figsize, matplotlib_rc, set_figsize, _index_where
from paths import paths_lap as paths_sim, exit_if_figure_exists
from make_fig_phys_fields_wave import fig_phys_subplot


def load_sim(short_name):
    params, Simul = fls.load_for_restart(
        paths_sim[short_name], merge_missing_params=True)
    params.oper.type_fft = "default"  # Enforce
    params.output.HAS_TO_SAVE = False
    params.output.ONLINE_PLOT_OK = False
    params.ONLY_COARSE_OPER = False
    return Simul(params)

# load_sim = lambda short_name: fls.load_state_phys_file(
#     paths_sim.get(short_name), merge_missing_params=True)
 
if __name__ == '__main__':
    matplotlib_rc(fontsize=10)
    path_fig = exit_if_figure_exists(__file__, extension='.pdf')
    set_figsize(6.65, 5.8)
    fig, axes = plt.subplots(2, 2)
    
    short_names = [
          'noise_c10nh960Buinf',   # WL1
          'noise_c10nh3840Buinf',  # WL5
          'noise_c200nh960Buinf',   # WL12
          'noise_c200nh2880Buinf',  # WL13
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
            sim, fig, ax, key_field, x_slice=[0,3], y_slice=[0,3], vmax=vmax, vmin=vmin
        )
        sim.output.close_files()

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