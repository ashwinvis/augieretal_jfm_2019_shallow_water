#!/usr/bin/env python
#coding=utf8

import numpy as np
import matplotlib.pyplot as plt
import glob

from solveq2d import solveq2d

from create_figs_articles import CreateFigArticles


num_fig = 1000
SAVE_FIG = 0

c = 40
resol = 240*2**5

name_file = (
'fig_nothin_c={0}_nh={1}'.format(c, resol)

)

create_fig = CreateFigArticles(
    short_name_article='SW1l', 
    SAVE_FIG=SAVE_FIG, 
    FOR_BEAMER=False, 
    fontsize=19
    )

dir_base  = create_fig.path_base_dir+'/Results_SW1lw'



str_resol = repr(resol)
str_to_find_path = (
    dir_base+'/Pure_standing_waves_'+
    str_resol+'*/SE2D*c='+repr(c))+'_*'
print str_to_find_path

paths_dir = glob.glob(str_to_find_path)

print paths_dir


sim = solveq2d.create_sim_plot_from_dir(paths_dir[0])

tmin = sim.output.spatial_means.first_saved_time()
tstatio = tmin + 4.




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



dico1, dico2 = sim.output.spatial_means.compute_time_means(tstatio=tstatio)
epsK = dico1['epsK']
epsA = dico1['epsA']
eps = epsK + epsA
# EK =  dico1['EK']
U = np.sqrt(2*EK)
c = np.sqrt(sim.param['c2'])






fig, ax1 = create_fig.figure_axe(name_file=name_file,
                                 fig_width_mm=200, fig_height_mm=150,
                                 size_axe=[0.124, 0.1, 0.85, 0.88])
ax1.set_xscale('log')
ax1.set_yscale('log')

coef_compensate = 6./3
coef_norm = (kh**coef_compensate)*c/eps

l_Etot = ax1.plot(kh, E_tot*coef_norm, 'k', linewidth=4)
l_EK = ax1.plot(kh, EK*coef_norm, 'r', linewidth=2)
l_EA = ax1.plot(kh, EA*coef_norm, 'b', linewidth=2)

cond = np.logical_and(kh > 2e0 , kh < 8e1)
ax1.plot(kh[cond], 2e-1*kh[cond]**(-2.)*coef_norm[cond], 'k:', linewidth=1)
plt.figtext(0.5, 0.55, '$k^{-2}$', fontsize=20)

cond = np.logical_and(kh > 2e0 , kh < 8e1)
ax1.plot(kh[cond], 1e-2*kh[cond]**(-5./3)*coef_norm[cond], 
         'k-.', linewidth=1)
plt.figtext(0.5, 0.35, '$k^{-5/3}$', fontsize=20)










ax1.set_xlabel(r'$k$')
ax1.set_ylabel(r'$E(k)k^2 c/\varepsilon $')




plt.rc('legend', numpoints=1)
leg1 = plt.figlegend(
        [l_Etot[0], l_EK[0], l_EA[0]], 
        ['$E$', '$E_K$', '$E_A$'], 
        loc=(0.2, 0.2), 
        labelspacing = 0.2
)



ax1.set_xlim([1e-1,4e2])
ax1.set_ylim([1e-1,2e2])


create_fig.save_fig()


create_fig.show()











