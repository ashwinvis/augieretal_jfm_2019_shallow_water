#!/usr/bin/env python
#coding=utf8

import numpy as np
import matplotlib.pyplot as plt
import glob

import baseSW1lw
from solveq2d import solveq2d

from create_figs_articles import CreateFigArticles

SAVE_FIG = 1

c = 40
nh = 240*2**5

name_file = (
'fig_spectra_c={0}_nh={1}'.format(c, nh)
)

create_fig = CreateFigArticles(
    short_name_article='SW1l', 
    SAVE_FIG=SAVE_FIG, 
    FOR_BEAMER=False, 
    fontsize=19
    )


paths = baseSW1lw.paths_from_nh_c_f(nh, c, f=0)



set_of_dir = solveq2d.SetOfDirResults(paths)
path = set_of_dir.path_larger_t_start()


sim = solveq2d.create_sim_plot_from_dir(path)

tmin, tmax, tstatio = baseSW1lw.tminmaxstatio_from_sim(sim, VERBOSE=True)





dico_results = sim.output.spectra.load1D_mean(tmin=tstatio)

kh = dico_results['kh']

EK = (dico_results['spectrum1Dkx_EK'] + 
      dico_results['spectrum1Dkx_EK'])/2
EA = (dico_results['spectrum1Dkx_EA'] + 
      dico_results['spectrum1Dkx_EA'])/2
EKr = (dico_results['spectrum1Dkx_EKr'] + 
       dico_results['spectrum1Dkx_EKr'])/2

E_tot = EK + EA
EKd = EK - EKr



c = np.sqrt(sim.param['c2'])

dico1, dico2 = sim.output.spatial_means.compute_time_means(tstatio=tstatio)
epsK = dico1['epsK']
epsA = dico1['epsA']
epsKsuppl = dico1['epsKsuppl']
eps = epsK + epsA
eps_small_scales = eps-epsKsuppl

# epsn = eps_small_scales
epsn = eps



fig, ax1 = create_fig.figure_axe(name_file=name_file,
                                 fig_width_mm=200, fig_height_mm=130,
                                 size_axe=[0.14, 0.135, 0.845, 0.84])

ax1.set_xscale('log')
ax1.set_yscale('log')

coef_compensate = 2.

kf = baseSW1lw.kf
kn = kf

alpha = 5./3
coef_norm = (
            kh**coef_compensate/(
                c**(2-alpha)*(eps/kf)**(alpha/3)*kf)
            )


l_Etot = ax1.plot(kh/kn, E_tot*coef_norm, 'k', linewidth=3)
l_EK = ax1.plot(kh/kn, EK*coef_norm, 'r', linewidth=1.5)
l_EA = ax1.plot(kh/kn, EA*coef_norm, 'c', linewidth=1.5)




# cond = np.logical_and(kh > 1 , kh < 20)
# ax1.plot(kh[cond], 1e1*kh[cond]**(-3.)*coef_norm[cond], 'k--', linewidth=1)
# plt.figtext(0.6, 0.78, '$k^{-3}$', fontsize=20)

cond = np.logical_and(kh > 2.5e0 , kh < 8e1)
ax1.plot(kh[cond], 2e-1*kh[cond]**(-2.)*coef_norm[cond], 'k', linewidth=1)
plt.figtext(0.6, 0.37, '$k^{-2}$', fontsize=20)

cond = np.logical_and(kh > 3e1 , kh < 2e2)
ax1.plot(kh[cond], 1.5e-1*kh[cond]**(-3./2)*coef_norm[cond], 
         'k', linewidth=1)
plt.figtext(0.75, 0.75, '$k^{-3/2}$', fontsize=20)







ax1.set_xlabel(r'$k/k_f$')
ax1.set_ylabel(
    r'$E(k)/\left(k^{-2} c^{1/3} \varepsilon^{5/9}{k_f}^{4/9}\right)$')




plt.rc('legend', numpoints=1)
leg1 = plt.figlegend(
        [l_Etot[0], l_EK[0], l_EA[0]], 
        ['$E$', '$E_K$', '$E_A$'], 
        loc=(0.3, 0.17), 
        labelspacing = 0.2
)



ax1.set_xlim([1e-1,5e2])
ax1.set_ylim([1e-2,2e0])


create_fig.save_fig()


create_fig.show()











