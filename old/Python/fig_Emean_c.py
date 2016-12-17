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
name_file = 'fig_Emean_c'
fontsize=21

create_fig = CreateFigs(
    path_relative='../Figs',
    SAVE_FIG=SAVE_FIG, 
    FOR_BEAMER=False, 
    fontsize=fontsize
    )



Lf = baseSW1lw.Lf
kf = baseSW1lw.kf



def load_from_c(set_of_dir, c):

    set_of_dir_c = set_of_dir.filter(c=c, f=0)
    path = set_of_dir_c.path_larger_t_start()

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
    print 'c = {0:8.1f}, Fr = {1:8.3f}, eps = {2:8.2f}'.format(
        c, Fr, eps)
    return locals()







# fig2, ax2 = create_fig.figure_axe(name_file=name_file+'_ratio')
fig, ax1 = create_fig.figure_axe(name_file=name_file,
                                 fig_width_mm=200, fig_height_mm=155,
                                 # size_axe=[0.12, 0.126, 0.83, 0.822]
                                 size_axe=[0.125, 0.126, 0.834, 0.822]
                                 )

# ax1.set_xscale('log')
# ax1.set_yscale('log')

ax1.set_xlabel(r'$c$')
ax1.set_ylabel(r'$E/\sqrt{\varepsilon}$')

def plot_one_resol(nh, style_lines='-'):

    paths = baseSW1lw.paths_from_nh(nh)

    set_of_dir = solveq2d.SetOfDirResults(paths)
    set_of_dir = set_of_dir.filter(solver='SW1lwaves',
                                   FORCING=True,
                                   f=0)

    path = set_of_dir.path_larger_t_start()

    dicos = []
    for c in set_of_dir.dico_values['c']:
        dico = load_from_c(set_of_dir, c)
        dicos.append(dico)

    nb_c = len(dicos)

    c = np.empty([nb_c])
    EKd = np.empty([nb_c])
    EA = np.empty([nb_c])
    eps = np.empty([nb_c])
    E = np.empty([nb_c])
    for ii in xrange(nb_c):
        c[ii] = dicos[ii]['c']
        EKd[ii] = dicos[ii]['EKd']
        EA[ii] = dicos[ii]['EA']
        eps[ii] = dicos[ii]['eps']
        E[ii] = dicos[ii]['E']

    K0 = 0.
    K1 = 0.
    K2 = 0.
    K4 = 0.

    K0 = 0.
    K1 = 0.
    K2 = 0.3
    K4 = 0
    Ueps = eps**(1./3)



    def Efit_from_params(params):
        K0 = params[0]
        K1 = params[1]
        K2 = params[2]
        Efit_classic = 0 #K0*Ueps**2  #/(1 + 0*K1/c)**(2./3)
        Efit_weak = K2*np.sqrt(eps*Lf*c)
        # Efit = Efit_classic + Efit_weak

        Efit = Efit_weak
        return Efit, Efit_classic, Efit_weak

    def to_minimize(params):
        Efit, Efit_classic, Efit_weak = Efit_from_params(params)
        for_sum = (Efit - E)**2
        return np.sum(for_sum[c>0])

    params = np.array([K0, K1, K2])
    params = optimize.fmin(to_minimize, params, xtol=1e-4, disp=True)
    Efit, Efit_classic, Efit_weak = Efit_from_params(params)

    K2 = params[2]
    print('K2 = {0}'.format(K2))

    norm = np.sqrt(eps)

    ax1.plot(c, E/norm, 'k'+style_lines, linewidth=2.2)
    ax1.plot(c, Efit/norm, 'c'+style_lines, linewidth=2.2)


    # ax2.plot(c, E/Efit, 'kx'+style_lines, linewidth=2)


    # ax1.plot(c, Efit_classic, 'mx'+style_lines, linewidth=2)
    # ax1.plot(c, Efit_weak, 'yx'+style_lines, linewidth=2)

    # pol = np.polyfit(c, E, 1)
    # print pol
    # ax1.plot(c, pol[1] + c*pol[0], 'y', linewidth=2)


resol = 240*2**2
plot_one_resol(resol, style_lines=':^')

resol = 240*2**3
plot_one_resol(resol, style_lines='-s')

resol = 240*2**4
plot_one_resol(resol, style_lines='--o')

# resol = 240*2**5
#### plot_one_resol(resol, style_lines='-.')




plt.figtext(0.55, 0.5, r'$n = 960$, $C_n = 0.23$', fontsize=fontsize)
plt.figtext(0.55, 0.86, r'$n = 1920$, $C_n = 0.29$', fontsize=fontsize)
plt.figtext(0.15, 0.56, '$n = 3840$\n $C_n = 0.37$', fontsize=fontsize)


l_E = ax1.plot(-1,-1, 'k', linewidth=2.4)
l_fit = ax1.plot(-1,-1, 'c', linewidth=2.4)

# plt.rc('legend', numpoints=1)
leg1 = plt.figlegend(
        [l_E[0], l_fit[0]], 
        [r'$E$', r'$C_n\sqrt{\varepsilon L_f c}$'], 
        loc=(0.62, 0.2), 
        labelspacing = 0.2
)




ax1.set_xlim([0,1e3])
ax1.set_ylim([0,2e1])


fig.text(0.012, 0.945, r'(\textit{a})', fontsize=fontsize+2)


create_fig.save_fig(fig=fig)

create_fig.show()











