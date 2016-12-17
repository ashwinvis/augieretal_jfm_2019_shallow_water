#!/usr/bin/env python
#coding=utf8

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import glob

import baseSW1lw
from solveq2d import solveq2d

from createfigs import CreateFigs

num_fig = 1000
SAVE_FIG = 1

c = 20
resol = 240*2**4

name_file = (
'fig_spatiotempspectra_c={0}_Nh={1}'.format(c, resol)
)

fontsize=21
create_fig = CreateFigs(
    path_relative='../Figs',
    SAVE_FIG=SAVE_FIG, 
    FOR_BEAMER=False, 
    fontsize=fontsize
    )


dir_base  = baseSW1lw.path_base_dir_results



str_resol = repr(resol)
str_to_find_path = (
    dir_base+'/Pure_standing_waves_'+
    str_resol+'*/SE2D*c='+repr(c))+'_*'
print(str_to_find_path)

paths = glob.glob(str_to_find_path)

print(paths)

set_of_dir = solveq2d.SetOfDirResults(paths)
path = set_of_dir.path_larger_t_start()


sim = solveq2d.create_sim_plot_from_dir(path)

tmin, tmax, tstatio = baseSW1lw.tminmaxstatio_from_sim(sim, VERBOSE=True)










dico_spectra, dico_results = sim.output.time_sigK.compute_spectra()

omega = dico_spectra['omega']
time_spectra_q = dico_spectra['time_spectra_q']
time_spectra_a = dico_spectra['time_spectra_a']
time_spectra_d = dico_spectra['time_spectra_d']
omega_shell = dico_results['omega_shell']
kh_shell = dico_results['kh_shell']



deltak = baseSW1lw.deltak

print('deltak/k = ', deltak/kh_shell)


print('k/deltak = ', kh_shell/deltak)






c = np.sqrt(sim.param['c2'])

# dico1, dico2 = sim.output.spatial_means.compute_time_means(tstatio=tstatio)
# epsK = dico1['epsK']
# epsA = dico1['epsA']
# epsKsuppl = dico1['epsKsuppl']
# eps = epsK + epsA
# eps_small_scales = eps-epsKsuppl

# # epsn = eps_small_scales
# epsn = eps



fig, ax1 = create_fig.figure_axe(name_file=name_file,
                                 fig_width_mm=220, fig_height_mm=145,
                                 size_axe=[0.13, 0.134, 0.845, 0.822])

ax1.set_xscale('log')
ax1.set_yscale('log')


ax1.set_xlabel(r'$\omega/\omega_l$')
ax1.set_ylabel(r'$E(\omega, k)$')

dark_red = (0.9,0,0)

nb_shells = dico_results['nb_shells']
for ish in xrange(nb_shells):
    # ax1.loglog(omega/omega_shell[ish], 
    #            time_spectra_q[ish], 'k', linewidth=1)
    l_EA = ax1.loglog(omega/omega_shell[ish], 
                      time_spectra_a[ish], 'c', linewidth=1.5)
    l_EK = ax1.loglog(omega/omega_shell[ish], 
                      time_spectra_d[ish], color=dark_red, linewidth=1.5)


leg1 = plt.figlegend(
    [l_EK[0], l_EA[0]], 
    ['$E_K$', '$E_A$'], 
    loc=(0.7, 0.7), 
    labelspacing = 0.2
    )


ax1.set_xlim([4e-2,3e1])
ax1.set_ylim([1e-14,1e-4])


create_fig.save_fig()


create_fig.show()











