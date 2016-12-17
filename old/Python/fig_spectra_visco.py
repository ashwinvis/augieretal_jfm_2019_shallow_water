#!/usr/bin/env python
#coding=utf8

import numpy as np
import matplotlib.pyplot as plt
import glob

import baseSW1lw
from solveq2d import solveq2d

from createfigs import CreateFigs


num_fig = 1000
SAVE_FIG = 1

nh = 240*2**3*3
name_file = 'fig_spectra_visco_Nx={0}'.format(nh)

fontsize = 21
create_figs = CreateFigs(
    path_relative='../Figs',
    SAVE_FIG=SAVE_FIG, 
    FOR_BEAMER=False, 
    fontsize=fontsize
    )

kf = baseSW1lw.kf


paths = baseSW1lw.paths_from_nh(nh)

set_of_dir = solveq2d.SetOfDirResults(paths)
set_of_dir = set_of_dir.filter(solver='SW1lwaves',
                               FORCING=True, 
                               f=0, c=20)

# nh = 240*2**2
# paths = baseSW1lw.paths_from_nh(nh)

# set_of_dir2 = solveq2d.SetOfDirResults(paths)
# set_of_dir2 = set_of_dir2.filter(solver='SW1lwaves',
#                                  FORCING=True, 
#                                  f=0, c=20)





def sprectra_from_path(path):

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



def compute_coef_norm(coef_compensate, k, eps, c):
    if coef_compensate == 3./2:
        coef_norm = k**coef_compensate /np.sqrt(eps*c)
    elif coef_compensate == 2.:
        # coef_norm = k**coef_compensate/c**(1./4)
        # delta = 0.0#3
        # alpha = 7./4/(1+delta)
        alpha = 5./3
        coef_norm = (
            k**coef_compensate/(
                c**(2-alpha)*(eps/kf)**(alpha/3)*kf)
            )
    elif coef_compensate == 5./3:
        coef_norm = eps**(-2./3)*kh**coef_compensate
    else:
        raise ValueError('bad value for coef_compensate')
    return coef_norm


def set_ylabel(ax1, coef_compensate):
    if coef_compensate == 3./2:
        ax1.set_ylabel(r'$E(k) / (k^{-3/2} \sqrt{\varepsilon c})$')
    elif coef_compensate == 2.:
        ax1.set_ylabel(r'$E(k)/(k^{-2} c^{1/3} \varepsilon^{5/9}{k_f}^{4/9})$')
    elif coef_compensate == 5./3:
        ax1.set_ylabel(r'$E(k) \varepsilon^{-2/3} k^{5/3}$')
    else:
        raise ValueError('bad value for coef_compensate')


def myplot(ax1, dico, codecolor, coef_compensate):
    k = dico['kh']
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
    coef_norm = compute_coef_norm(coef_compensate, k, eps, c)

    # ax1.plot(k/kf, spectEKr*coef_norm, codecolor+'--', linewidth=2)
    # ax1.plot(k/kf, spectEKd*coef_norm, codecolor+':', linewidth=2)
    return ax1.plot(k/kf, spectE*coef_norm, codecolor, linewidth=2)





dicos = []
for path in set_of_dir.paths:
        dico = sprectra_from_path(path)
        dicos.append(dico)



# dico = sprectra_from_path(set_of_dir2.paths[-1])
# dicos.append(dico)




colors = ['k', 'c', 'r--']
ncolors = len(colors)

k = dicos[0]['kh']
eps = dicos[0]['eps']
c = dicos[0]['c']

kadim = k/kf







size_axe=[0.135, 0.13, 0.84, 0.822]

fig, ax1 = create_figs.figure_axe(name_file=name_file+'_k32',
                                 fig_width_mm=200, fig_height_mm=155,
                                 size_axe=size_axe
                                 )

ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel(r'$k/k_f$')


ax1.text(1.6e-1, 6e-1, '$n = {}$'.format(nh), fontsize=fontsize)




coef_compensate = 3./2
lines = []
for ii in xrange(len(dicos)):
    l = myplot(ax1, dicos[ii], colors[ii%ncolors], coef_compensate)
    lines.append(l)

coef_norm = compute_coef_norm(coef_compensate, k, eps, c)


cond = np.logical_and(kadim>1.5, kadim<20)
eqline = 8e-1*k**(-2.)*coef_norm
ax1.plot(kadim[cond], eqline[cond], 'k')
ax1.text(5, 1.1e-1, '$k^{-2}$', fontsize=fontsize)

cond = np.logical_and(kadim>6e1, kadim<1.6e2)
eqline = 1e-1*k**(-3./2)*coef_norm
ax1.plot(kadim[cond], eqline[cond], 'k')
ax1.text(1.7e2, 2.e-2, '$k^{-3/2}$', fontsize=fontsize)

cond = np.logical_and(kadim>3e1, kadim<1e2)
eqline = 3e-1*k**(-5./3)*coef_norm
ax1.plot(kadim[cond], eqline[cond], 'k')
ax1.text(5e1, 4e-2, '$k^{-5/3}$', fontsize=fontsize)


set_ylabel(ax1, coef_compensate)

plt.rc('legend', numpoints=1)
leg1 = plt.figlegend(
    [lines[0][0], lines[1][0]],
    ['hyper-viscosity', 'Newtonian viscosity'],
    loc=(0.2, 0.2),
    labelspacing = 0.28
)

ax1.set_xlim([1.4e-1,4e2])
ax1.set_ylim([4e-4,1e0])





# ax2 = fig.add_axes([-0.01, -0.01, 1.02, 1.02])

# ax2.set_xlim([0,1])
# ax2.set_ylim([0,1])

# ax2.patch.set_visible(False)
# ax2.xaxis.set_visible(False)
# ax2.yaxis.set_visible(False)

# ax2.arrow(0.62, 0.62, -0.06, -0.16, fc="k", ec="k",
#           head_width=0.015, head_length=0.03)

# ax1.text(6.5e0, 1.3e-2, '$c$', fontsize=fontsize)



create_figs.save_fig()














# fig, ax1 = create_figs.figure_axe(name_file=name_file+'_k2',
#                                  fig_width_mm=200, fig_height_mm=155,
#                                  size_axe=size_axe
#                                  )
# ax1.set_xscale('log')
# ax1.set_yscale('log')
# ax1.set_xlabel(r'$k/k_f$')

# ax1.text(1.6e-1, 1.35, '$n = {}$'.format(nh), fontsize=fontsize)

# coef_compensate = 2.

# for ii in xrange(len(dicos)):
#     myplot(ax1, dicos[ii], colors[ii%ncolors], coef_compensate)

# coef_norm = compute_coef_norm(coef_compensate, k, eps, c)

# cond = np.logical_and(kadim>1.5, kadim<20)
# eqline = 8e-1*k**(-2.)*coef_norm
# ax1.plot(kadim[cond], eqline[cond], 'k')
# ax1.text(5, 4.5e-1, '$k^{-2}$', fontsize=fontsize)

# cond = np.logical_and(kadim>3e1, kadim<1e2)
# eqline = 1.2e-1*k**(-3./2)*coef_norm
# ax1.plot(kadim[cond], eqline[cond], 'k')
# ax1.text(8e1, 3.5e-1, '$k^{-3/2}$', fontsize=fontsize)

# cond = np.logical_and(kadim>3e1, kadim<1e2)
# eqline = 4e-1*k**(-5./3)*coef_norm
# ax1.plot(kadim[cond], eqline[cond], 'k')
# ax1.text(4e1, 8e-1, '$k^{-5/3}$', fontsize=fontsize)

# set_ylabel(ax1, coef_compensate)


# # plt.rc('legend', numpoints=1)
# # leg1 = plt.figlegend(
# #         [l_Etot[0], l_EK[0], l_EA[0]], 
# #         ['$E$', '$E_K$', '$E_A$'], 
# #         loc=(0.78, 0.7), 
# #         labelspacing = 0.2
# # )

# ax1.set_xlim([1.4e-1, 2e2])
# ax1.set_ylim([1e-2, 2e0])



# create_figs.save_fig()







create_figs.show()











