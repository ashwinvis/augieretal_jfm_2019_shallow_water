#!/usr/bin/env python
#coding=utf8

from createfigs import CreateFigs

import numpy as np
import matplotlib.pyplot as plt
import glob

import scipy.optimize as optimize

import baseSW1lw
from solveq2d import solveq2d

from baseSW1lw import Tf, Ef


SAVE_FIG = 1
name_file = 'fig_Emean_time'

fontsize = 15


create_figs = CreateFigs(
    path_relative='../Figs',
    SAVE_FIG=SAVE_FIG, 
    FOR_BEAMER=False, 
    fontsize=fontsize
    )





def load_from_path(path):
    sim = solveq2d.create_sim_plot_from_dir(path)
    dico = sim.output.spatial_means.load()
    c = np.sqrt(sim.param.c2)
    return dico, c



colors = {10 : 'k', 
          20:'r', 
          40:'b',
          70:'y',
          100:'g', 
          200:'m', 
          400:'c',
          700:'r',
          1000:'k'
    }

size_axe = [0.08, 0.13, 0.21, 0.84]
fig, ax1 = create_figs.figure_axe(name_file=name_file,
                                 fig_width_mm=220, fig_height_mm=120,
                                 size_axe=size_axe, 
                                 )
ax1.set_xlabel(r'$t/T_f$')
ax1.set_ylabel(r'$E/E_f$')

size_axe = [0.35, 0.13, 0.62, 0.84]


ax2 = fig.add_axes(size_axe)
ax2.set_xlabel(r'$t/T_f$')


dir_base  = baseSW1lw.path_base_dir_results

tn = Tf
En = Ef

def plot_one_resol(resol, style_lines='-', linewidth=1):
    str_resol = repr(resol)
    str_to_find_path = (
        dir_base+'/Pure_standing_waves_'+
        str_resol+'*'
        )

    # print str_to_find_path
    paths_dir = glob.glob(str_to_find_path)
    # print paths_dir
    path_base = paths_dir[0]


    set_of_dir = solveq2d.SetOfDirResults(path_base)

    set_of_dir = set_of_dir.filter(solver='SW1lwaves',
                                   FORCING=True, 
                                   f=0)

    paths = set_of_dir.paths

    print paths
    nb_dirs = len(paths)

    # dicos = []
    for ii in xrange(nb_dirs):
        dico, c = load_from_path(paths[ii])
        # dicos.append(dico)
        t = dico['t']
        E = dico['E']

        color = colors[c]

        ax1.plot(t/tn, E/En, color+style_lines, linewidth=linewidth)
        ax2.plot(t/tn, E/En, color+style_lines, linewidth=linewidth)



thin_line = 0.8
thick_line = 2.5


resol = 240*2**0
plot_one_resol(resol, style_lines='-', linewidth=thin_line)

resol = 240*2**1
plot_one_resol(resol, style_lines='--', linewidth=thick_line)

resol = 240*2**2
plot_one_resol(resol, style_lines=':', linewidth=thin_line)

resol = 240*2**3
plot_one_resol(resol, style_lines='-', linewidth=thick_line)

resol = 240*2**4
plot_one_resol(resol, style_lines='--', linewidth=thin_line)

resol = 5760
plot_one_resol(resol, style_lines=':', linewidth=thick_line)

resol = 240*2**5
plot_one_resol(resol, style_lines='-.', linewidth=thin_line)





ax1.set_xlim([0,80])
ax1.set_ylim([0,17])

ax1.xaxis.set_ticks(np.linspace(0, 80, 5))


ax2.set_xlim([80,232/tn])
ax2.set_ylim([0,17])




ax1.text(30/tn, 5.6/En, '$n = 240$', fontsize=fontsize)
ax2.text(128/tn, 6.2/En, r'$n = 480$', fontsize=fontsize)
ax2.text(152/tn, 7.15/En, r'$n = 960$', fontsize=fontsize)



ax2.text(191/tn, 18/En, r'$c=1000$', fontsize=fontsize)

ax2.text(190/tn, 15/En, r'$c=700$', fontsize=fontsize)

ax2.text(190/tn, 11.8/En, r'$c=400$', fontsize=fontsize)

ax2.text(191/tn, 10./En, r'$c=200$', fontsize=fontsize)

ax2.text(205/tn, 7.5/En, r'$c=100$', fontsize=fontsize)

ax2.text(200/tn, 5.7/En, r'$c=40$', fontsize=fontsize)

ax2.text(205/tn, 3.8/En, r'$c=20$', fontsize=fontsize)

ax2.text(192/tn, 1.4/En, r'$c=10$', fontsize=fontsize)


create_figs.save_fig(fig=fig)

plt.show()











