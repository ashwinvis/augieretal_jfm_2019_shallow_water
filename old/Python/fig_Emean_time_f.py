#!/usr/bin/env python
#coding=utf8

from create_figs_articles import CreateFigArticles

import numpy as np
import matplotlib.pyplot as plt
import glob

import scipy.optimize as optimize

import baseSW1lw
from solveq2d import solveq2d




SAVE_FIG = 0
name_file = 'fig_Emean_time_f'

create_fig = CreateFigArticles(
    short_name_article='SW1l', 
    SAVE_FIG=SAVE_FIG, 
    FOR_BEAMER=False, 
    fontsize=19
    )




def load_from_namedir(set_of_dir, name_dir_results):
    path_dir_results = set_of_dir.path_dirs[name_dir_results]
    sim = solveq2d.create_sim_plot_from_dir(path_dir_results)
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

size_axe = [0.13, 0.12, 0.845, 0.85]
fig, ax1 = create_fig.figure_axe(name_file=name_file,
                                 fig_width_mm=200, fig_height_mm=130,
                                 size_axe=size_axe, 
                                 )
ax1.set_xlabel(r'$t$')
ax1.set_ylabel(r'$E$')



dir_base  = baseSW1lw.path_base_dir_results

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


    set_of_dir_results = solveq2d.SetOfDirResults(dir_base=path_base)
    paths = set_of_dir_results.dirs_from_values(solver='SW1lwaves',
                                                FORCING=True, 
                                                c2=20**2)

    print paths
    nb_dirs = len(paths)

    # dicos = []
    for ii in xrange(nb_dirs):
        dico, c = load_from_namedir(set_of_dir_results, paths[ii])
        # dicos.append(dico)
        t = dico['t']
        E = dico['E']

        color = colors[c]

        ax1.plot(t, E, color+style_lines, linewidth=linewidth)




resol = 1920
plot_one_resol(resol, style_lines='-', linewidth=1)



# ax1.set_xlim([0,1e2])
# ax1.set_ylim([0,19])

# ax1.xaxis.set_ticks(np.linspace(0, 100, 5))



fontsize = 10


# ax2.text(181, 17.5, r'$c=1000$', fontsize=fontsize)



create_fig.save_fig(fig=fig)

create_fig.show()











