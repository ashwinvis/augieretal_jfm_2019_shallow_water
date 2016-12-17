#!/usr/bin/env python
#coding=utf8

from __future__ import print_function

import h5py

import numpy as np
import matplotlib.pyplot as plt
import glob

import baseSW1lw
from solveq2d import solveq2d

from create_figs_articles import CreateFigArticles


num_fig = 1000
SAVE_FIG = 1

c = 20
nh = 240*2**3
nh = 240*2**4

name_file = (
'fig_Kolmo_c={0}_N={1}'.format(c, nh)
)

def cumsum_inv(a):
    return a[::-1].cumsum()[::-1]


fontsize = 19
create_fig = CreateFigArticles(
    short_name_article='SW1l', 
    SAVE_FIG=SAVE_FIG, 
    FOR_BEAMER=False, 
    fontsize=fontsize
    )


paths = baseSW1lw.paths_from_nh_c_f(nh, c, f=0)

set_of_dir = solveq2d.SetOfDirResults(paths)
path = set_of_dir.path_larger_t_start()

sim = solveq2d.create_sim_plot_from_dir(path)

tmin, tmax, tstatio = baseSW1lw.tminmaxstatio_from_sim(sim, VERBOSE=True)
# tstatio = tmin + 3.5

# if resol == 1920:
#     tstatio = tmin
# elif resol == 3840:
#     pass


c = np.sqrt(sim.param['c2'])

dico1, dico2 = sim.output.spatial_means.compute_time_means(tstatio=tstatio)
epsK = dico1['epsK']
epsA = dico1['epsA']
epsKsuppl = dico1['epsKsuppl']
eps = epsK + epsA
eps_small_scales = eps-epsKsuppl

"""
Interestingly, there is a "large-scale" dissipation ~1%
due to the fact that the kinetic energy is not quadratic.
"""

print('eps = {0:.2f}, eps_small_scales = {1:.2f}'.format(eps, eps_small_scales))

# epsn = eps_small_scales
epsn = eps



path_file = sim.output.increments.path_file
f = h5py.File(path_file,'r')
dset_times = f['times']
times = dset_times[...]
nt = len(times)

tmin = tstatio
tmax = times.max()

rxs = f['rxs'][...]

nx = f['Obj_param']['nx'][...]
Lx = f['Obj_param']['Lx'][...]
deltax = Lx/nx



rxs = np.array(rxs, dtype=np.float64)*deltax

imin_plot = np.argmin(abs(times-tmin))
imax_plot = np.argmin(abs(times-tmax))

tmin_plot = times[imin_plot]
tmax_plot = times[imax_plot]

to_print = '''plot structure functions
tmin = {0:8.6g} ; tmax = {1:8.6g}
imin = {2:8d} ; imax = {3:8d}'''.format(
    tmin_plot, tmax_plot,
    imin_plot, imax_plot)
print(to_print)

S_uL2JL = f['struc_func_uL2JL'][imin_plot:imax_plot+1].mean(0)
S_uT2JL = f['struc_func_uT2JL'][imin_plot:imax_plot+1].mean(0)
S_c2h2uL = f['struc_func_c2h2uL'][imin_plot:imax_plot+1].mean(0)
S_Kolmo = f['struc_func_Kolmo'][imin_plot:imax_plot+1].mean(0)
S_uT2uL = f['struc_func_uT2uL'][imin_plot:imax_plot+1].mean(0)

S_Kolmo_theo = -4*rxs*epsn










fig, ax1 = create_fig.figure_axe(name_file=name_file,
                                 fig_width_mm=200, fig_height_mm=135,
                                 size_axe=[0.13, 0.13, 0.845, 0.84])
ax1.set_xscale('log')
# ax1.set_yscale('log')

Lf = baseSW1lw.Lf
rn = Lf

ax1.plot(rxs/rn, S_Kolmo/S_Kolmo_theo, 'k', linewidth=2.2)
ax1.plot(rxs/rn, (S_uL2JL+S_uT2JL)/S_Kolmo_theo, 'r', linewidth=1.5)

ax1.plot(rxs/rn, S_c2h2uL/S_Kolmo_theo, 'c', linewidth=1.5)

# ax1.plot(rxs/rn, S_uL2JL/S_Kolmo_theo, 'r--', linewidth=1)
# ax1.plot(rxs/rn, S_uT2JL/S_Kolmo_theo, 'r-.', linewidth=1)



ax1.plot([5e-3, 5e-1], eps_small_scales/epsn*np.ones([2,1]), 
         'k:', linewidth=2)






ax1.set_xlabel('$r/L_f$')

ax1.set_ylabel(
r'$ \langle |\delta \textbf{u}|^2 \delta J_L'
r' + c^2(\delta h)^2 \delta u_L \rangle /(4\varepsilon r)$')




# plt.rc('legend', numpoints=1)
# leg1 = plt.figlegend(
#         [l_PiE[0], l_PiEK[0], l_PiEA[0]], 
#         ['$\Pi$', '$\Pi_K$', '$\Pi_A$'], 
#         loc=(0.15, 0.7), 
#         labelspacing = 0.2
# )


if nh == 1920:
    ax1.set_xlim([6e-3,2e0])
elif nh == 3840:
    ax1.set_xlim([2.5e-3,2e0])
# ax1.set_ylim([-0.1,1.1])


create_fig.save_fig()


create_fig.show()











