#!/usr/bin/env python
#coding=utf8

import h5py

import numpy as np
import matplotlib.pyplot as plt
import glob

import baseSW1lw
from solveq2d import solveq2d

from create_figs_articles import CreateFigArticles


num_fig = 1000
SAVE_FIG = 1

c = 100
nh = 240*2**3
nh = 240*2**4

name_file = (
'fig_spect_energy_budg_c={0}_N={1}'.format(c, nh)
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

c = np.sqrt(sim.param['c2'])

dico1, dico2 = sim.output.spatial_means.compute_time_means(tstatio=tstatio)
epsK = dico1['epsK']
epsA = dico1['epsA']
epsKsuppl = dico1['epsKsuppl']
eps = epsK + epsA
eps_small_scales = eps-epsKsuppl

# epsn = eps_small_scales
epsn = eps

"""
Interestingly, there is a "large-scale" dissipation ~1%
due to the fact that the kinetic energy is not quadratic.
Therefore, the flux decreases slightly at large scale (~1%).
"""


path_file = sim.output.spect_energy_budg.path_file

f = h5py.File(path_file,'r')

dset_times = f['times']
times = dset_times[...]
nt = len(times)

dset_khE = f['khE']
kh = dset_khE[...]

dset_transfer2D_EKr = f['transfer2D_EKr']
dset_transfer2D_EKd = f['transfer2D_EKd']
dset_transfer2D_EAr = f['transfer2D_EAr']
dset_transfer2D_EAd = f['transfer2D_EAd']
dset_transfer2D_EPd = f['transfer2D_EPd']
dset_convP2D = f['convP2D']
dset_convK2D = f['convK2D']
dset_transfer2D_CPE = f['transfer2D_CPE']



dset_transfer2D_Errr = f['transfer2D_Errr']
dset_transfer2D_Edrd = f['transfer2D_Edrd']
dset_transfer2D_Edrr_rrd = f['transfer2D_Edrr_rrd']
dset_transfer2D_Eureu = f['transfer2D_Eureu']

dset_transfer2D_Eddd = f['transfer2D_Eddd']
dset_transfer2D_Erdr = f['transfer2D_Erdr']
dset_transfer2D_Eddr_rdd = f['transfer2D_Eddr_rdd']
dset_transfer2D_Eudeu = f['transfer2D_Eudeu']








delta_t_save = np.mean(times[1:]-times[0:-1])

delta_t = 0.
delta_i_plot = int(np.round(delta_t/delta_t_save))
if delta_i_plot == 0 and delta_t != 0.:
    delta_i_plot=1
delta_t = delta_i_plot*delta_t_save

tmin = tstatio
tmax = times.max()

imin_plot = np.argmin(abs(times-tmin))
imax_plot = np.argmin(abs(times-tmax))

tmin_plot = times[imin_plot]
tmax_plot = times[imax_plot]




to_print = '''plot fluxes 2D
tmin = {0:8.6g} ; tmax = {1:8.6g} ; delta_t = {2:8.6g}
imin = {3:8d} ; imax = {4:8d} ; delta_i = {5:8d}'''.format(
tmin_plot, tmax_plot, delta_t,
imin_plot, imax_plot, delta_i_plot)
print(to_print)







transferEKr = dset_transfer2D_EKr[imin_plot:imax_plot].mean(0)
transferEKd = dset_transfer2D_EKd[imin_plot:imax_plot].mean(0)
transferEAr = dset_transfer2D_EAr[imin_plot:imax_plot].mean(0)
transferEAd = dset_transfer2D_EAd[imin_plot:imax_plot].mean(0)
transferEPd = dset_transfer2D_EPd[imin_plot:imax_plot].mean(0)

deltakh = 2*np.pi/sim.param['Lx']

PiEKr = cumsum_inv(transferEKr)*deltakh
PiEKd = cumsum_inv(transferEKd)*deltakh
PiEAr = cumsum_inv(transferEAr)*deltakh
PiEAd = cumsum_inv(transferEAd)*deltakh
PiEPd = cumsum_inv(transferEPd)*deltakh

PiEK = PiEKr + PiEKd
PiEA = PiEAr + PiEAd
PiE = PiEK + PiEA






if nh == 1920:
    kmin = 1e-1
    kmax = 2e2
elif nh == 3840:
    kmin = 1e-1
    kmax = 4e2



fig, ax1 = create_fig.figure_axe(name_file=name_file,
                                 fig_width_mm=200, fig_height_mm=140,
                                 size_axe=[0.11, 0.12, 0.87, 0.85])
ax1.set_xscale('log')
# ax1.set_yscale('log')

kf = baseSW1lw.kf
kn = kf

l_PiE = ax1.plot(kh/kn, PiE/epsn, 'k', linewidth=2)
l_PiEK = ax1.plot(kh/kn, PiEK/epsn, 'r', linewidth=1)
l_PiEA = ax1.plot(kh/kn, PiEA/epsn, 'c', linewidth=1)


ax1.plot([1, 200], eps_small_scales/epsn*np.ones([2,1]), 
         'k:', linewidth=2)


ax1.set_xlabel(r'$k/k_f$')
ax1.set_ylabel(r'$\Pi(k)/\varepsilon$')


plt.rc('legend', numpoints=1)
leg1 = plt.figlegend(
        [l_PiE[0], l_PiEK[0], l_PiEA[0]], 
        ['$\Pi$', '$\Pi_K$', '$\Pi_A$'], 
        loc=(0.15, 0.7), 
        labelspacing = 0.2
)



ax1.set_xlim([kmin,kmax])
ax1.set_ylim([-0.1,1.1])


create_fig.save_fig()


create_fig.show()











