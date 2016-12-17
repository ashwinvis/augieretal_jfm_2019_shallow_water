#!/usr/bin/env python
#coding=utf8

from create_figs_articles import CreateFigArticles

import numpy as np
import matplotlib.pyplot as plt

import scipy.optimize as optimize

from solveq2d import solveq2d




num_fig = 1000
SAVE_FIG = 1
name_file = 'fig_time_mean_forcingw_f_c.eps'

create_fig = CreateFigArticles(
    short_name_article='SW1l', 
    SAVE_FIG=SAVE_FIG, 
    FOR_BEAMER=False, 
    fontsize=19
    )




def load_from_namedir(set_of_dir, name_dir_results, tstatio):
    path_dir_results = set_of_dir.path_dirs[name_dir_results]
    sim = solveq2d.create_sim_plot_from_dir(path_dir_results)
    (dico_time_means, dico_results
     ) = sim.output.spatial_means.compute_time_means(tstatio)
    c2 = sim.param['c2']
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
    c = np.sqrt(set_of_dir.dico_c2[name_dir_results])
    print 'c = {0:8.1f}, Fr = {1:8.3f}, eps = {2:8.2f}'.format(
        c, Fr, eps)
    return locals()





fig, ax1 = create_fig.figure_axe(name_file=name_file)
# ax1.set_xscale('log')
# ax1.set_yscale('log')

ax1.set_xlabel(r'$c$')
ax1.set_ylabel(r'$E$')





def plot_one_dir_base(dir_base, tstatio=40):

    set_of_dir_results = solveq2d.SetOfDirResults(dir_base=dir_base)
    dirs = set_of_dir_results.dirs_from_values(solver='SW1lwaves',
                                               FORCING=True)

    print dirs

    print set_of_dir_results.dico_c2.keys()

    nb_dirs = len(dirs)

    dicos = []
    temp = 0
    for ii in xrange(nb_dirs):
        if set_of_dir_results.dico_c2[dirs[ii]] > 50**2:
            temp += 1
            dico = load_from_namedir(set_of_dir_results, dirs[ii], tstatio)
            dicos.append(dico)

    nb_dirs = temp
    c = np.empty([nb_dirs])
    EKd = np.empty([nb_dirs])
    EA = np.empty([nb_dirs])
    eps = np.empty([nb_dirs])
    E = np.empty([nb_dirs])
    for ii in xrange(nb_dirs):
        c[ii] = dicos[ii]['c']
        EKd[ii] = dicos[ii]['EKd']
        EA[ii] = dicos[ii]['EA']
        eps[ii] = dicos[ii]['eps']
        E[ii] = dicos[ii]['E']

    K0 = 0.
    K1 = 0.
    K2 = 0.
    K4 = 0.

    K0 = 0.1
    K1 = -10.
# K2 = 0.23
    K4 = 0
    Ueps = eps**(1./3)



    def Efit_from_params(params):
        K0 = params[0]
        K1 = params[1]
        K2 = params[2]
        Efit_classic = K0*Ueps**2/(1 + 0*K1/c)**(2./3)
        Efit_weak = K2*np.sqrt(Ueps**3*c)
        Efit = Efit_classic + Efit_weak
        return Efit, Efit_classic, Efit_weak

    def to_minimize(params):
        Efit, Efit_classic, Efit_weak = Efit_from_params(params)
        return np.sum((Efit - E)**2)

    params = np.array([K0, K1, K2])
    params = optimize.fmin(to_minimize, params, xtol=1e-4, disp=True)
    Efit, Efit_classic, Efit_weak = Efit_from_params(params)

    K0 = params[0]
    K1 = params[1]
    K2 = params[2]
    print('K0 ={0} ; K1 = {1} ; K2 = {2}'.format(K0, K1, K2))
    
    # ax1.plot(c, EA/norm, 'b', linewidth=2)
    # ax1.plot(c, EKd/norm, 'r', linewidth=2)

    ax1.plot(c, E, 'k-x', linewidth=2)
    ax1.plot(c, Efit, 'c-x', linewidth=2)


    ax1.plot(c, Efit_classic, 'r-x', linewidth=2)
    ax1.plot(c, Efit_weak, 'y-x', linewidth=2)

    # ax1.plot(c, E/norm, 'k-x', linewidth=2)

    # pol = np.polyfit(c, E, 1)
    # print pol
    # ax1.plot(c, pol[1] + c*pol[0], 'y', linewidth=2)



# dir_super_base = create_fig.path_base_dir+'/Results_SW1lw'

# dir_base = dir_super_base+'/Pure_standing_waves_256x256'

dir_base = '/scratch/augier/Results_solveq2d'

plot_one_dir_base(dir_base, tstatio=30)

# dir_base = dir_super_base+'/Pure_standing_waves_512x512'
# plot_one_dir_base(dir_base, tstatio=55)

# dir_base = dir_super_base+'/Pure_standing_waves_1024x1024'
# plot_one_dir_base(dir_base, tstatio=80)

# dir_base = dir_super_base+'/Pure_standing_waves_2048x2048'
# plot_one_dir_base(dir_base, tstatio=105)



# def ffit_from_params(params):
#     Ks = params[0]
#     Kw = params[1]
#     E0 = params[2]
#     epss = Ks*(E)**(3./2)
#     epsw = Kw*(E)**2/c
#     F =  eps - (epss + epsw)
#     return F, epss, epsw

# def to_minimize(params):
#     F, epss, epsw = ffit_from_params(params)
#     return np.sum(F**2)

# Ks = 1.
# Kw = 1.
# E0 = 1.
# params = np.array([Ks, Kw, E0])
# params = optimize.fmin(to_minimize, params, xtol=1e-4, disp=True)
# F, epss, epsw = ffit_from_params(params)


# fig, ax1 = create_fig.figure_axe(name_file=name_file)
# # ax1.set_xscale('log')
# # ax1.set_yscale('log')

# ax1.set_xlabel('$c$')
# ax1.set_ylabel('$\epsilon$')

# ax1.plot(c, F/eps, 'kx-')
# ax1.plot(c, epss/eps, 'rx-')
# ax1.plot(c, epsw/eps, 'bx-')

# Ks = params[0]
# Kw = params[1]
# print('Ks ={0} ; Kw = {1}'.format(Ks, Kw))

# print params[2]






# plt.rc('legend', numpoints=1)
# leg1 = plt.figlegend(
#         [l_Etot[0], l_EK[0], l_EA[0]], 
#         ['$E$', '$E_K$', '$E_A$'], 
#         loc=(0.78, 0.7), 
#         labelspacing = 0.2
# )


# ax1.set_xlim([0, 110])
# ax1.set_ylim([0, 4.8])


create_fig.save_fig()

solveq2d.show()











