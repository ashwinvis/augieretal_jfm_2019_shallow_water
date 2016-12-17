#!/usr/bin/env python
#coding=utf8


from createfigs import CreateFigs

import numpy as np
import matplotlib.pyplot as plt
import glob

import scipy.optimize as optimize

import baseSW1lw
from solveq2d import solveq2d




SAVE_FIG = 1
name_file = 'fig_Emean_resol'
fontsize=21

create_fig = CreateFigs(
    path_relative='../Figs',
    SAVE_FIG=SAVE_FIG, 
    FOR_BEAMER=False, 
    fontsize=fontsize
    )


Lf = baseSW1lw.Lf
kf = baseSW1lw.kf




def load_from_nh(set_of_dir, nh):

    set_of_dir_nh = set_of_dir.filter(nh=nh, f=0)
    path = set_of_dir_nh.path_larger_t_start()

    sim = solveq2d.create_sim_plot_from_dir(path)

    tmin, tmax, tstatio = baseSW1lw.tminmaxstatio_from_sim(sim, VERBOSE=True)

    (dico_time_means, dico_results
     ) = sim.output.spatial_means.compute_time_means(tstatio)
    c2 = sim.param['c2']
    c = np.sqrt(c2)
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

    Enorm = np.sqrt(eps*Lf*c)

    print 'c = {0:8.1f}, Fr = {1:8.3f}, eps = {2:8.2f}'.format(
        c, Fr, eps)
    return locals()





fig, ax1 = create_fig.figure_axe(name_file=name_file,
                                 fig_width_mm=200, fig_height_mm=155,
                                 size_axe=[0.134, 0.126, 0.834, 0.822]
                                 )

# ax1.set_xscale('log')
# ax1.set_yscale('log')

ax1.set_xlabel(r'$n$')
ax1.set_ylabel(r'$E/\sqrt{\varepsilon L_f c}$')

dir_base  = baseSW1lw.path_base_dir_results

def plot_one_c(c, style_lines='-', linewidth=2.2):

    paths_c = baseSW1lw.paths_from_c(c)
    # print paths_c


    set_of_dir = solveq2d.SetOfDirResults(paths_c)
    set_of_dir = set_of_dir.filter(solver='SW1lwaves',
                                   FORCING=True,
                                   f=0., 
                                   c2=c**2)


    dicos = []
    for nh in set_of_dir.dico_values['nh']:
        if nh<7000:
            dico = load_from_nh(set_of_dir, nh)
            dicos.append(dico)

    nb_nh = len(dicos)

    nh = np.empty([nb_nh])
    c = np.empty([nb_nh])
    EKd = np.empty([nb_nh])
    EA = np.empty([nb_nh])
    eps = np.empty([nb_nh])
    E = np.empty([nb_nh])

    Enorm = np.empty([nb_nh])
    for ii in xrange(nb_nh):
        nh[ii] = dicos[ii]['nh']
        c[ii] = dicos[ii]['c']
        EKd[ii] = dicos[ii]['EKd']
        EA[ii] = dicos[ii]['EA']
        eps[ii] = dicos[ii]['eps']
        E[ii] = dicos[ii]['E']
        Enorm[ii] = dicos[ii]['Enorm']

    Ueps = eps**(1./3)

    line = ax1.plot(nh, E/Enorm, 'kx'+style_lines, linewidth=linewidth)

    return line[0]


thin_line = 1.
thick_line = 2.5


c = 10
l10 = plot_one_c(c, style_lines=':', linewidth=thin_line)

c = 20
l20 = plot_one_c(c, style_lines='--', linewidth=thin_line)

c = 40
l40 = plot_one_c(c, style_lines='-', linewidth=thin_line)

c = 100
l100 = plot_one_c(c, style_lines=':', linewidth=thick_line)

c = 200
l200 = plot_one_c(c, style_lines='--', linewidth=thick_line)

c = 700
l700 = plot_one_c(c, style_lines='-', linewidth=thick_line)







plt.rc('legend', numpoints=1)
leg1 = plt.figlegend(
        [l10, l20, l40, l100, l200, l700], 
        [r'$c = 10$', r'$c = 20$', r'$c = 40$', 
         r'$c = 100$',r'$c = 200$', r'$c = 700$'], 
        loc=(0.65, 0.2), 
        labelspacing = 0.2
)




# ax1.set_xlim([0,1e3])
# ax1.set_ylim([0,2e1])

ax1.set_xticks([240, 960, 1920, 3840, 5760])


plt.figtext(0.012, 0.945, r'(\textit{b})', fontsize=fontsize+2)


create_fig.save_fig(fig=fig)

create_fig.show()











