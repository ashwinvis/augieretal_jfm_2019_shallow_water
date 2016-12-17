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
name_file = 'fig_Emean_f'
fontsize=21

create_fig = CreateFigArticles(
    short_name_article='SW1l', 
    SAVE_FIG=SAVE_FIG, 
    FOR_BEAMER=False, 
    fontsize=fontsize
    )


Lf = baseSW1lw.Lf
kf = baseSW1lw.kf



def load_from_f(set_of_dir, f):

    set_of_dir_f = set_of_dir.filter(f=f)
    path = set_of_dir_f.path_larger_t_start()

    sim = solveq2d.create_sim_plot_from_dir(path)

    tmin, tmax, tstatio = baseSW1lw.tminmaxstatio_from_sim(sim, VERBOSE=True)

    (dico_time_means, dico_results
     ) = sim.output.spatial_means.compute_time_means(tstatio)
    c2 = sim.param['c2']
    c = np.sqrt(c2)
    f = sim.param['f']
    kd = f/c
    EK = dico_time_means['EK']
    Fr = np.sqrt(2*EK/c2)
    EA = dico_time_means['EA']
    EKr = dico_time_means['EKr']
    EKd = EK - EKr
    E = EK + EA
    epsK = dico_time_means['epsK']
    epsA = dico_time_means['epsA']
    eps = epsK + epsA
    epsK_tot = dico_time_means['epsK_tot']
    epsA_tot = dico_time_means['epsA_tot']
    print 'c = {0:8.1f}, Fr = {1:8.3f}, eps = {2:8.2f}'.format(
        c, Fr, eps)
    return locals()





fig, ax1 = create_fig.figure_axe(name_file=name_file,
                                 fig_width_mm=200, fig_height_mm=155,
                                 size_axe=[0.132, 0.125, 0.85, 0.845]
                                 )

# fig, ax1 = create_fig.figure_axe(name_file=name_file,
#                                  fig_width_mm=200, fig_height_mm=155,
#                                  size_axe=[0.1, 0.11, 0.88, 0.85]
#                                  )


# ax1.set_xscale('log')
# ax1.set_yscale('log')

ax1.set_xlabel(r'$1/Bu$')
ax1.set_ylabel(r'$E$')
fig.text(0.01, 0.945, r'(\textit{a})', fontsize=fontsize+2)

dir_base  = baseSW1lw.path_base_dir_results

def plot_one_resol(resol, style_lines='-'):
    """plot on resolution."""
    str_resol = repr(resol)
    str_to_find_path = (
        dir_base+'/Pure_standing_waves_'+
        str_resol+'*'
        )

    paths = glob.glob(str_to_find_path)
    path = paths[0]

    set_of_dir = solveq2d.SetOfDirResults(path)
    set_of_dir = set_of_dir.filter(solver='SW1lwaves',
                                   FORCING=True,
                                   c=20)


    print('poum Emean_f', set_of_dir.dico_values['f'])


    dicos = []
    for f in set_of_dir.dico_values['f']:
        dico = load_from_f(set_of_dir, f)
        dicos.append(dico)

    nb_f = len(dicos)

    f = np.empty([nb_f])
    kd = np.empty([nb_f])
    EKd = np.empty([nb_f])
    EA = np.empty([nb_f])
    eps = np.empty([nb_f])
    E = np.empty([nb_f])
    for ii in xrange(nb_f):
        f[ii] = dicos[ii]['f']
        kd[ii] = dicos[ii]['kd']
        EKd[ii] = dicos[ii]['EKd']
        EA[ii] = dicos[ii]['EA']
        eps[ii] = dicos[ii]['eps']
        E[ii] = dicos[ii]['E']


    kadim = (kd/kf)**2

    ax1.plot(kadim, E, 'k'+style_lines, linewidth=2)


    return kadim, E



resol = 240*2**3
kadim, E = plot_one_resol(resol, style_lines='-')

colors = ['k', 'b', 'c', 'r', 'g', 'y']

for ii in xrange(len(E)):
    ax1.plot(kadim[ii], E[ii], 'o'+colors[ii], markersize=12)





# plt.rc('legend', numpoints=1)
# leg1 = plt.figlegend(
#         [l_E[0], l_fit[0]], 
#         [r'$E$', r'$C_N\sqrt{\varepsilon L_f c}$'], 
#         loc=(0.65, 0.2), 
#         labelspacing = 0.2
# )



ax1.set_xlim([0, 2.2])
ax1.set_ylim([0, 10])




create_fig.save_fig(fig=fig)

create_fig.show()











