
from __future__ import print_function

import h5py

import matplotlib.pylab as plt
import glob
import numpy as np

import baseSW1lw
from solveq2d import solveq2d

from create_figs_articles import CreateFigArticles


SAVE_FIG = 0





fontsize = 19
create_fig = CreateFigArticles(
    short_name_article='SW1l', 
    SAVE_FIG=SAVE_FIG, 
    FOR_BEAMER=False, 
    fontsize=fontsize
    )

dir_base  = create_fig.path_base_dir+'/Results_SW1lw'

c = 40
resol = 240*2**5

key_var = 'uy' # for transverse since r = delta x
# key_var = 'ux' # for longitudinal since r = delta x


str_resol = repr(resol)
str_to_find_path = (
    dir_base+'/Pure_standing_waves_'+
    str_resol+'*/SE2D*c='+repr(c))+'_*'
# print str_to_find_path

paths = glob.glob(str_to_find_path)
print(paths)

set_of_dir = solveq2d.SetOfDirResults(path_dirs_results=paths)
path = set_of_dir.path_larger_t_start()

sim = solveq2d.create_sim_plot_from_dir(path)

tmin, tmax, tstatio = baseSW1lw.tminmaxstatio_from_sim(sim, VERBOSE=True)


increments = sim.output.increments


pdf_ux, values_inc_ux, nb_rx_to_plot = increments.load_pdf_from_file(
    tmin=tmin, tmax=tmax, key_var='ux')

pdf_uy, values_inc_uy, nb_rx_to_plot = increments.load_pdf_from_file(
    tmin=tmin, tmax=tmax, key_var='uy')

pdf_eta, values_inc_eta, nb_rx_to_plot = increments.load_pdf_from_file(
    tmin=tmin, tmax=tmax, key_var='eta')


S2_uL = increments.strfunc_from_pdf(pdf_ux, values_inc_ux, 2)
S3_uL = increments.strfunc_from_pdf(pdf_ux, values_inc_ux, 3)
S4_uL = increments.strfunc_from_pdf(pdf_ux, values_inc_ux, 4)

S2_uT = increments.strfunc_from_pdf(pdf_uy, values_inc_uy, 2)
S3_uT = increments.strfunc_from_pdf(pdf_uy, values_inc_uy, 3)
S4_uT = increments.strfunc_from_pdf(pdf_uy, values_inc_uy, 4)

S2_ceta = c**2*increments.strfunc_from_pdf(pdf_eta, values_inc_eta, 2)
S3_ceta = c**3*increments.strfunc_from_pdf(pdf_eta, values_inc_eta, 3)
S4_ceta = c**4*increments.strfunc_from_pdf(pdf_eta, values_inc_eta, 4)




path_file = sim.output.increments.path_file
f = h5py.File(path_file,'r')
dset_times = f['times']
times = dset_times[...]
nt = len(times)

tmin = tstatio
tmax = times.max()

imin_plot = np.argmin(abs(times-tmin))
imax_plot = np.argmin(abs(times-tmax))

S_uT2uL = f['struc_func_uT2uL'][imin_plot:imax_plot+1].mean(0)







flatness = S4_uL / S2_uL**2
ratioS2 = S2_uL / S2_uT
ratioS3 = S3_uL / S_uT2uL










Lf = baseSW1lw.Lf

deltax = sim.param.Lx/sim.param.nx
rxs = np.array(sim.output.increments.rxs, dtype=np.float64)*deltax

# if 7680
rmin = 8*deltax
rmax = 40*deltax

condr = np.logical_and(rxs>rmin, rxs<rmax)






fig, ax1 = create_fig.figure_axe(name_file='fig_strfct_S2')

ax1.set_xlabel('$r/L_f$')
ax1.set_ylabel('$\langle |\delta u_L|^2\\rangle/r$')

ax1.hold(True)
ax1.set_xscale('log')
ax1.set_yscale('log')

norm = rxs**(1)
order = 2.

ax1.plot(rxs/Lf, S2_uL/norm, 'b', linewidth=2)
ax1.plot(rxs/Lf, S2_uT/norm, 'r', linewidth=2)
ax1.plot(rxs/Lf, S2_ceta/norm, 'y', linewidth=2)


cond = np.logical_and(rxs > 7e-3*Lf , rxs < 2e-1*Lf)
l_1 = ax1.plot(rxs[cond]/Lf, 2*rxs[cond]/norm[cond], 
               'k', linewidth=1)
ax1.text(2.2e-2, 1.2,'shocks, $r^1$',fontsize=fontsize)


cond = np.logical_and(rxs > 2e-2*Lf , rxs < 4e-1*Lf)
l_K41 = ax1.plot(rxs[cond]/Lf, 2e0*rxs[cond]**(order/3)/norm[cond], 
                 'k--', linewidth=1)
ax1.text(2.2e-2, 3e0,'K41, $r^{q/3}$',fontsize=fontsize)

cond = rxs < 4*deltax
temp = rxs**(order)/norm
l_smooth= ax1.plot(
    rxs[cond]/Lf,temp[cond]/temp[0]*S2_ceta[0]/norm[0],
    'k:', linewidth=1)
ax1.text(2e-3, 3e0,'smooth, $r^q$',fontsize=fontsize)



# plt.rc('legend', numpoints=1)
# leg1 = ax1.legend(
#     [l_smooth[0], l_K41[0], l_1[0]], 
#     ['smooth $r^q$', 'K41 $r^{q/3}$', 
#      'shocks $r^1$'], 
#     loc=(0.57, 0.13),
#     labelspacing = 0.2
#     )


# ax1.set_xlim([1e-3, 1.5e1])
# ax1.set_ylim([7e-1, 1e3])


create_fig.save_fig(fig=fig)

















fig, ax1 = create_fig.figure_axe(name_file='fig_strfct_S3')

ax1.set_xlabel('$r/L_f$')
ax1.set_ylabel('$\langle |\delta u_L|^3\\rangle/r$')

ax1.hold(True)
ax1.set_xscale('log')
ax1.set_yscale('log')

norm = rxs**(1)
order = 3.

ax1.plot(rxs/Lf, S3_uL/norm, 'b', linewidth=2)
ax1.plot(rxs/Lf, S3_uT/norm, 'r', linewidth=2)
ax1.plot(rxs/Lf, S3_ceta/norm, 'y', linewidth=2)

ax1.plot(rxs/Lf, S_uT2uL/norm, 'g', linewidth=2)



ax1.plot(rxs/Lf, -S3_uL/norm, '--b', linewidth=2)
ax1.plot(rxs/Lf, -S3_uT/norm, '--r', linewidth=2)
ax1.plot(rxs/Lf, -S3_ceta/norm, '--y', linewidth=2)
ax1.plot(rxs/Lf, -S_uT2uL/norm, '--g', linewidth=2)





cond = np.logical_and(rxs > 7e-3*Lf , rxs < 2e-1*Lf)
l_1 = ax1.plot(rxs[cond]/Lf, 2*rxs[cond]/norm[cond], 
               'k', linewidth=1)
ax1.text(2.2e-2, 1.2,'shocks, $r^1$',fontsize=fontsize)


cond = np.logical_and(rxs > 2e-2*Lf , rxs < 4e-1*Lf)
l_K41 = ax1.plot(rxs[cond]/Lf, 2e0*rxs[cond]**(order/3)/norm[cond], 
                 'k--', linewidth=1)
ax1.text(2.2e-2, 3e0,'K41, $r^{q/3}$',fontsize=fontsize)

cond = rxs < 4*deltax
temp = rxs**(order)/norm
l_smooth= ax1.plot(
    rxs[cond]/Lf,temp[cond]/temp[0]*S3_ceta[0]/norm[0],
    'k:', linewidth=1)
ax1.text(2e-3, 3e0,'smooth, $r^q$',fontsize=fontsize)



# plt.rc('legend', numpoints=1)
# leg1 = ax1.legend(
#     [l_smooth[0], l_K41[0], l_1[0]], 
#     ['smooth $r^q$', 'K41 $r^{q/3}$', 
#      'shocks $r^1$'], 
#     loc=(0.57, 0.13),
#     labelspacing = 0.2
#     )


# ax1.set_xlim([1e-3, 1.5e1])
# ax1.set_ylim([7e-1, 1e3])


create_fig.save_fig(fig=fig)
















fig, ax1 = create_fig.figure_axe(name_file='fig_flatness')

ax1.set_xlabel('$r/L_f$')
ax1.set_ylabel('flatness')

ax1.hold(True)
ax1.set_xscale('log')
ax1.set_yscale('log')

ax1.plot(rxs/Lf, flatness, 'k', linewidth=2)

# ax1.set_xlim([1e-3, 1.5e1])
# ax1.set_ylim([7e-1, 1e3])


create_fig.save_fig(fig=fig)









fig, ax1 = create_fig.figure_axe(name_file='fig_ratio')

ax1.set_xlabel('$r/L_f$')
ax1.set_ylabel('ratio')

ax1.hold(True)
ax1.set_xscale('log')
ax1.set_yscale('log')


ax1.plot(rxs/Lf, ratioS3, 'b', linewidth=2)
ax1.plot(rxs/Lf, ratioS2, 'r', linewidth=2)


# ax1.set_xlim([1e-3, 1.5e1])
# ax1.set_ylim([7e-1, 1e3])


create_fig.save_fig(fig=fig)














solveq2d.show()
