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

nh = 240*2**3
c = 20

name_file = 'fig_spectra_f_Nx={0}'.format(nh)

fontsize = 21
create_figs = CreateFigs(
    path_relative='../Figs',
    SAVE_FIG=SAVE_FIG, 
    FOR_BEAMER=False, 
    fontsize=fontsize
    )


kf = baseSW1lw.kf

paths = baseSW1lw.paths_from_nh_c_f(nh, c)

set_of_dir = solveq2d.SetOfDirResults(paths)
set_of_dir = set_of_dir.filter(solver='SW1lwaves',
                               FORCING=True,
                               c2=c**2)


def sprectra_from_f(f):

    set_of_dir_c = set_of_dir.filter(f=f)
    path = set_of_dir_c.path_larger_t_start()

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

    spectEK[spectEK<1e-16] = 0.
    spectEA[spectEA<1e-16] = 0.
    spectEKr[spectEKr<1e-16] = 0.


    spectEKd = spectEK - spectEKr
    spectE = spectEK + spectEA
    dico1, dico2 = sim.output.spatial_means.compute_time_means(tstatio=tstatio)
    epsK = dico1['epsK']
    epsA = dico1['epsA']
    eps = epsK + epsA
    EK =  dico1['EK']
    U = np.sqrt(2*EK)
    c = np.sqrt(sim.param['c2'])
    f = sim.param['f']
    kd = f/c
    Bu = (kf/kd)**2


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
                c**(2-alpha)*(eps/Kf)**(alpha/3)*Kf)
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
        ax1.set_ylabel(r'$E(k)/(k^{-2} c^{1/3} (\varepsilon/k_f)^{5/9}k_f)$')
    elif coef_compensate == 5./3:
        ax1.set_ylabel(r'$E(k) \varepsilon^{-2/3} k^{5/3}$')
    else:
        raise ValueError('bad value for coef_compensate')


def myplot(ax1, dico, codecolor, coef_compensate):
    k = dico['kh']
    spectE  = dico['spectE']
    spectEK = dico['spectEK']
    spectEA = dico['spectEA']
    eps = dico['eps']

    f = dico['f']
    kd = dico['kd']

    c = dico['c']
    U = dico['U']



    Fr = U/c
    print(
'{0}, c = {1:8.2f}, U = {2:8.2f}, U/c = {3:8.4f}, eps = {4:8.2f}'.format(
            codecolor, c, U, Fr, eps)
)
    coef_norm = compute_coef_norm(coef_compensate, k, eps, c)

    kn = kf
    # kn = kd

    line = ax1.plot(k/kn, spectE*coef_norm, codecolor, linewidth=2)

    # ax1.plot(k/kn, spectEK*coef_norm, codecolor+'--', linewidth=2)
    # ax1.plot(k/kn, spectEA*coef_norm, codecolor+':', linewidth=2)

    return line[0]


dicos = []
strings = []
for f in set_of_dir.dico_values['f']:
    dico = sprectra_from_f(f)
    dicos.append(dico)

    if f == 0:
        # pass
        strings.append(r'$Bu = \infty$')
    else:
        strings.append(r'$Bu = {0}$'.format(dico['Bu']))




colors = ['k', 'b--', 'c', 'r-.', 'g:', 'y']
ncolors = len(colors)

k = dicos[0]['kh']
eps = dicos[0]['eps']
c = dicos[0]['c']

kadim = k/kf




fig, ax1 = create_figs.figure_axe(name_file=name_file,
                                 fig_width_mm=200, fig_height_mm=155,
                                 size_axe=[0.135, 0.125, 0.845, 0.845]
                                 )

ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel(r'$k/k_d$')

fig.text(0.01, 0.945, r'(\textit{b})', fontsize=fontsize+2)


coef_compensate = 3./2

lines = []
for ii in xrange(len(dicos)):
    dico = dicos[ii]
    kd = dico['kd']
    if kd > -1:
        line = myplot(ax1, dicos[ii], colors[ii%ncolors], coef_compensate)
        lines.append(line)



coef_norm = compute_coef_norm(coef_compensate, k, eps, c)


cond = np.logical_and(kadim>1.5, kadim<20)
eqline = 3e-1*k**(-2.)*coef_norm
ax1.plot(kadim[cond], eqline[cond], 'k')
ax1.text(4, 2.1e-2, '$k^{-2}$', fontsize=fontsize)

cond = np.logical_and(kadim>2, kadim<7)
eqline = 7e0*k**(-3.)*coef_norm
ax1.plot(kadim[cond], eqline[cond], 'k')
ax1.text(4, 3.5e-1, '$k^{-3}$', fontsize=fontsize)

cond = np.logical_and(kadim>2e1, kadim<7e1)
eqline = 2e-1*k**(-3./2)*coef_norm
ax1.plot(kadim[cond], eqline[cond], 'k')
ax1.text(3e1, 5e-2, '$k^{-3/2}$', fontsize=fontsize)

# cond = np.logical_and(kadim>3e1, kadim<1e2)
# eqline = 3e-1*k**(-5./3)*coef_norm
# ax1.plot(kadim[cond], eqline[cond], 'k--')
# plt.figtext(0.8, 0.6, '$k^{-5/3}$', fontsize=fontsize)

set_ylabel(ax1, coef_compensate)


plt.rc('legend', numpoints=1)
leg1 = plt.figlegend(
        lines, 
        strings, 
        loc=(0.22, 0.15), 
        labelspacing = 0.2
)

ax1.set_xlim([0.1,1.5e2])
ax1.set_ylim([2e-3,2e0])


create_figs.save_fig()

















# fig, ax1 = create_figs.figure_axe(name_file=name_file+'_k2')
# ax1.set_xscale('log')
# ax1.set_yscale('log')
# ax1.set_xlabel(r'$k/k_f$')

# if resol == 1920:
#     fig.text(0.01, 0.95, r'(\textit{b})', fontsize=fontsize)
# elif resol == 3840:
#     fig.text(0.01, 0.95, r'(\textit{d})', fontsize=fontsize)

# coef_compensate = 2.

# for ii in xrange(len(dicos)):
#     myplot(ax1, dicos[ii], colors[ii%ncolors], coef_compensate)

# coef_norm = compute_coef_norm(coef_compensate, k, eps, c)

# cond = np.logical_and(kadim>1.5, kadim<20)
# eqline = 8e-1*k**(-2.)*coef_norm
# ax1.plot(kadim[cond], eqline[cond], 'k-.')
# plt.figtext(0.5, 0.72, '$k^{-2}$', fontsize=fontsize)

# cond = np.logical_and(kadim>3e1, kadim<1e2)
# eqline = 1e-1*k**(-3./2)*coef_norm
# ax1.plot(kadim[cond], eqline[cond], 'k')
# plt.figtext(0.8, 0.7, '$k^{-3/2}$', fontsize=fontsize)

# cond = np.logical_and(kadim>3e1, kadim<1e2)
# eqline = 3e-1*k**(-5./3)*coef_norm
# ax1.plot(kadim[cond], eqline[cond], 'k--')
# plt.figtext(0.8, 0.8, '$k^{-5/3}$', fontsize=fontsize)

# set_ylabel(ax1, coef_compensate)


# # plt.rc('legend', numpoints=1)
# # leg1 = plt.figlegend(
# #         [l_Etot[0], l_EK[0], l_EA[0]], 
# #         ['$E$', '$E_K$', '$E_A$'], 
# #         loc=(0.78, 0.7), 
# #         labelspacing = 0.2
# # )

# # ax1.set_xlim([0.1,150])
# ax1.set_ylim([1e-2,3e0])


# create_figs.save_fig()







create_figs.show()











