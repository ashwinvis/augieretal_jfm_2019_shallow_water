#!/usr/bin/env python
#coding=utf8

import numpy as np
import matplotlib.pyplot as plt
import glob

import baseSW1lw
from solveq2d import solveq2d

from create_figs_articles import CreateFigArticles


SAVE_FIG = 0
c = 10
# c = 40


name_file = 'fig_spectra_resol_c={0}'.format(c)

fontsize = 21
create_fig = CreateFigArticles(
    short_name_article='SW1l', 
    SAVE_FIG=SAVE_FIG, 
    FOR_BEAMER=False, 
    fontsize=fontsize
    )

kf = baseSW1lw.kf


paths_c = baseSW1lw.paths_from_c(c)

set_of_dir = solveq2d.SetOfDirResults(paths_c)

set_of_dir = set_of_dir.filter(solver='SW1lwaves',
                               FORCING=True,
                               f=0)





def spectra_from_namedir(nh):

    set_of_dir_nh = set_of_dir.filter(nh=nh)

    path = set_of_dir_nh.path_larger_t_start()
    sim = solveq2d.create_sim_plot_from_dir(path)
    tmin, tmax, tstatio = baseSW1lw.tminmaxstatio_from_sim(sim, VERBOSE=True)

    dico_results = sim.output.spectra.load1D_mean(tmin=tstatio)
    kh = dico_results['kh']

    spectEK = (dico_results['spectrum1Dkx_EK'] + 
          dico_results['spectrum1Dkx_EK'])/2
    spectEA = (dico_results['spectrum1Dkx_EA'] + 
          dico_results['spectrum1Dkx_EA'])/2
    spectEKr = (dico_results['spectrum1Dkx_EKr'] + 
           dico_results['spectrum1Dkx_EKr'])/2

    spectEKd = spectEK - spectEKr
    spectE = spectEK + spectEA
    dico1, dico2 = sim.output.spatial_means.compute_time_means(tstatio=tstatio)
    epsK = dico1['epsK']
    epsA = dico1['epsA']
    eps = epsK + epsA
    EK =  dico1['EK']
    U = np.sqrt(2*EK)
    c = np.sqrt(sim.param['c2'])
    return locals()

dicos = []
strings = []
for nh in set_of_dir.dico_values['nh']:
    dico = spectra_from_namedir(nh)
    dicos.append(dico)
    strings.append(r'$n = {0}$'.format(nh))




fig, ax1 = create_fig.figure_axe(name_file=name_file,
                                 fig_width_mm=200, fig_height_mm=155,
                                 size_axe=[0.132, 0.12, 0.855, 0.84]
                                 )

ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel(r'$k/k_f$')
ax1.set_ylabel(r'$E(k) /( k^{-3/2} \sqrt{\varepsilon c} )$')


coef_compensate = 3./2

def myplot(dico, codecolor=None):
    kh = dico['kh']
    spectE  = dico['spectE']
    spectEKr = dico['spectEKr']
    spectEKd = dico['spectEKd']
    eps = dico['eps']
    c = dico['c']
    U = dico['U']

    Fr = U/c
    print(
'{0}, c = {1:8.2f}, U = {2:8.2f}, U/c = {3:8.4f}, eps = {4:8.2f}'.format(
            codecolor, c, U, Fr, eps)
)
    coef_norm = kh**coef_compensate/np.sqrt(eps*c)

    line = ax1.plot(kh/kf, spectE*coef_norm, codecolor+'-', linewidth=2)
    return line


colors = ['k', 'b', 'c', 'r', 'g', 'y']
ncolors = len(colors)
lines = []
for ii in xrange(len(dicos)):
    line = myplot(dicos[ii], codecolor=colors[ii%ncolors])
    lines.append(line[0])




kh = dicos[-1]['kh']

kadim = kh/kf


eqline = 1e-1*kh**(-3./2+coef_compensate)
cond = np.logical_and(kadim>40, kadim<300)
ax1.plot(kadim[cond], eqline[cond], 'k')
plt.figtext(0.8, 0.7, '$k^{-3/2}$', fontsize=fontsize)

eqline = 1e-1*kh**(-5./3+coef_compensate)
cond = np.logical_and(kadim>40, kadim<300)
ax1.plot(kadim[cond], eqline[cond], 'k')
plt.figtext(0.8, 0.53, '$k^{-5/3}$', fontsize=fontsize)

eqline = 5e-1*kh**(-6./3+coef_compensate)
cond = np.logical_and(kadim>1.5, kadim<30)
ax1.plot(kadim[cond], eqline[cond], 'k')
plt.figtext(0.5, 0.83, '$k^{-2}$', fontsize=fontsize)




ax1.set_xlim([1e-1, 5e2])
ax1.set_ylim([1e-3, 1e0])






plt.figtext(0.15, 0.9, '$c ={0}$'.format(c), fontsize=fontsize+2)

if c == 10:
    plt.figtext(0.007, 0.945, r'(\textit{a})', fontsize=fontsize+2)
    plt.rc('legend', numpoints=1)
    leg1 = plt.figlegend(
        lines, 
        strings, 
        loc=(0.24, 0.15), 
        labelspacing = 0.18
        )

elif c == 40:
    plt.figtext(0.007, 0.945, r'(\textit{b})', fontsize=fontsize+2)




create_fig.save_fig()

solveq2d.show()











