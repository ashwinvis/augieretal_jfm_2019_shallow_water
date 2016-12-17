
from __future__ import print_function

import h5py

import matplotlib.pylab as plt
import glob
import numpy as np

import baseSW1lw
from solveq2d import solveq2d

from createfigs import CreateFigs


SAVE_FIG = 1


fontsize = 21

create_figs = CreateFigs(
    path_relative='../Figs',
    SAVE_FIG=SAVE_FIG, 
    FOR_BEAMER=False, 
    fontsize=fontsize
    )


dir_base  = create_figs.path_base_dir+'/Results_SW1lw'

c = 10
nh = 240*2**5

paths = baseSW1lw.paths_from_nh_c_f(nh, c, f=0)
print(paths)

set_of_dir = solveq2d.SetOfDirResults(paths)
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


S2_uL = increments.strfunc_from_pdf(pdf_ux, abs(values_inc_ux), 2)
S3_uL = increments.strfunc_from_pdf(pdf_ux, abs(values_inc_ux), 3)
S4_uL = increments.strfunc_from_pdf(pdf_ux, abs(values_inc_ux), 4)
S5_uL = increments.strfunc_from_pdf(pdf_ux, abs(values_inc_ux), 5)

S2_uT = increments.strfunc_from_pdf(pdf_uy, abs(values_inc_uy), 2)
S3_uT = increments.strfunc_from_pdf(pdf_uy, abs(values_inc_uy), 3)
S4_uT = increments.strfunc_from_pdf(pdf_uy, abs(values_inc_uy), 4)
S5_uT = increments.strfunc_from_pdf(pdf_uy, abs(values_inc_uy), 5)


# S2_ceta = c**2*increments.strfunc_from_pdf(pdf_eta, values_inc_eta, 2)
# S3_ceta = c**3*increments.strfunc_from_pdf(pdf_eta, values_inc_eta, 3)
# S4_ceta = c**4*increments.strfunc_from_pdf(pdf_eta, values_inc_eta, 4)




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

S_uL3 = increments.strfunc_from_pdf(pdf_ux, values_inc_ux, 3)






flatnessL = S4_uL / S2_uL**2
flatnessT = S4_uT / S2_uT**2

ratioS2 = S2_uL / S2_uT
ratioS3 = S3_uL / S3_uT
ratioS4 = S4_uL / S4_uT
ratioS5 = S5_uL / S5_uT

ratioS3bis = S_uL3 / S_uT2uL










Lf = baseSW1lw.Lf

deltax = sim.param.Lx/sim.param.nx
rxs = np.array(sim.output.increments.rxs, dtype=np.float64)*deltax

radim = rxs/Lf

# if 7680
rmin = 8*deltax
rmax = 40*deltax

condr = np.logical_and(rxs>rmin, rxs<rmax)






# fig, ax1 = create_figs.figure_axe(name_file='fig_strfct_S2')

# ax1.set_xlabel('$r/L_f$')
# ax1.set_ylabel('$\langle |\delta u_L|^2\\rangle/r$')

# ax1.hold(True)
# ax1.set_xscale('log')
# ax1.set_yscale('log')

# norm = rxs**(1)
# order = 2.

# ax1.plot(rxs/Lf, S2_uL/norm, 'b', linewidth=2)
# ax1.plot(rxs/Lf, S2_uT/norm, 'r', linewidth=2)
# ax1.plot(rxs/Lf, S2_ceta/norm, 'y', linewidth=2)


# cond = np.logical_and(rxs > 7e-3*Lf , rxs < 2e-1*Lf)
# l_1 = ax1.plot(rxs[cond]/Lf, 2*rxs[cond]/norm[cond], 
#                'k', linewidth=1)
# ax1.text(2.2e-2, 1.2,'shocks, $r^1$',fontsize=fontsize)


# cond = np.logical_and(rxs > 2e-2*Lf , rxs < 4e-1*Lf)
# l_K41 = ax1.plot(rxs[cond]/Lf, 2e0*rxs[cond]**(order/3)/norm[cond], 
#                  'k--', linewidth=1)
# ax1.text(2.2e-2, 3e0,'K41, $r^{q/3}$',fontsize=fontsize)

# cond = rxs < 4*deltax
# temp = rxs**(order)/norm
# l_smooth= ax1.plot(
#     rxs[cond]/Lf,temp[cond]/temp[0]*S2_ceta[0]/norm[0],
#     'k:', linewidth=1)
# ax1.text(2e-3, 3e0,'smooth, $r^q$',fontsize=fontsize)



# # plt.rc('legend', numpoints=1)
# # leg1 = ax1.legend(
# #     [l_smooth[0], l_K41[0], l_1[0]], 
# #     ['smooth $r^q$', 'K41 $r^{q/3}$', 
# #      'shocks $r^1$'], 
# #     loc=(0.57, 0.13),
# #     labelspacing = 0.2
# #     )


# # ax1.set_xlim([1e-3, 1.5e1])
# # ax1.set_ylim([7e-1, 1e3])


# create_fig.save_fig(fig=fig)

















# fig, ax1 = create_fig.figure_axe(name_file='fig_strfct_S3')

# ax1.set_xlabel('$r/L_f$')
# ax1.set_ylabel('$\langle |\delta u_L|^3\\rangle/r$')

# ax1.hold(True)
# ax1.set_xscale('log')
# ax1.set_yscale('log')

# norm = rxs**(1)
# order = 3.

# ax1.plot(rxs/Lf, S3_uL/norm, 'b', linewidth=2)
# ax1.plot(rxs/Lf, S3_uT/norm, 'r', linewidth=2)
# ax1.plot(rxs/Lf, S3_ceta/norm, 'y', linewidth=2)

# ax1.plot(rxs/Lf, S_uT2uL/norm, 'g', linewidth=2)



# ax1.plot(rxs/Lf, -S3_uL/norm, '--b', linewidth=2)
# ax1.plot(rxs/Lf, -S3_uT/norm, '--r', linewidth=2)
# ax1.plot(rxs/Lf, -S3_ceta/norm, '--y', linewidth=2)
# ax1.plot(rxs/Lf, -S_uT2uL/norm, '--g', linewidth=2)





# cond = np.logical_and(rxs > 7e-3*Lf , rxs < 2e-1*Lf)
# l_1 = ax1.plot(rxs[cond]/Lf, 2*rxs[cond]/norm[cond], 
#                'k', linewidth=1)
# ax1.text(2.2e-2, 1.2,'shocks, $r^1$',fontsize=fontsize)


# cond = np.logical_and(rxs > 2e-2*Lf , rxs < 4e-1*Lf)
# l_K41 = ax1.plot(rxs[cond]/Lf, 2e0*rxs[cond]**(order/3)/norm[cond], 
#                  'k--', linewidth=1)
# ax1.text(2.2e-2, 3e0,'K41, $r^{q/3}$',fontsize=fontsize)

# cond = rxs < 4*deltax
# temp = rxs**(order)/norm
# l_smooth= ax1.plot(
#     rxs[cond]/Lf,temp[cond]/temp[0]*S3_ceta[0]/norm[0],
#     'k:', linewidth=1)
# ax1.text(2e-3, 3e0,'smooth, $r^q$',fontsize=fontsize)



# # plt.rc('legend', numpoints=1)
# # leg1 = ax1.legend(
# #     [l_smooth[0], l_K41[0], l_1[0]], 
# #     ['smooth $r^q$', 'K41 $r^{q/3}$', 
# #      'shocks $r^1$'], 
# #     loc=(0.57, 0.13),
# #     labelspacing = 0.2
# #     )


# # ax1.set_xlim([1e-3, 1.5e1])
# # ax1.set_ylim([7e-1, 1e3])


# create_fig.save_fig(fig=fig)









r1 = 3e-3
r2 = 1e-0


size_axe = [0.12, 0.133, 0.845, 0.823]

fig, ax1 = create_figs.figure_axe(
    name_file='fig_flatness',
    fig_width_mm=200, fig_height_mm=155,
    size_axe=size_axe
    )

ax1.set_xlabel('$r/L_f$')
ax1.set_ylabel('$F_L$, $F_T$')

ax1.hold(True)
ax1.set_xscale('log')
ax1.set_yscale('log')

linewidth = 2.2
l_FT = ax1.plot(radim, flatnessT, 'k', linewidth=linewidth)
l_FL = ax1.plot(radim, flatnessL, 'y', linewidth=linewidth)

cond = np.logical_and(radim > 1.4e-2 , radim < 2e-1)
ax1.plot(radim[cond], 4e0*rxs[cond]**(-1), 
                 'k', linewidth=1)
ax1.text(4e-2, 1e1,'$r^{-1}$',fontsize=fontsize)

ax1.set_xlim([r1, r2])
# ax1.set_ylim([1, 10])

plt.rc('legend', numpoints=1)
leg1 = plt.figlegend(
    [l_FT[0], l_FL[0]], 
    ['$F_T$', '$F_L$'], 
    loc=(0.2, 0.2), 
    labelspacing = 0.18
    )





ax2 = fig.add_axes([0.62, 0.6, 0.32, 0.32])
ax2.set_xscale('log')
ax2.plot(radim, flatnessT/flatnessL, 'k', linewidth=linewidth)
value_shocks = 1.5
ax2.plot([r1, r2], [value_shocks, value_shocks], 'k:', linewidth=linewidth)

ax2.set_xlim([r1, r2])
ax2.set_ylim([0, 1.7])
ax2.set_yticks([0, 0.5, 1, 1.5])

ax2.set_xlabel('$r/L_f$')
ax2.set_ylabel('$F_T/F_L$')


for item in ([ax2.xaxis.label, ax2.yaxis.label] +
             ax2.get_xticklabels() + ax2.get_yticklabels()):
    item.set_fontsize(fontsize-4)

# fig.text(0.01, 0.945, r'(\textit{a})', fontsize=fontsize+2)




create_figs.save_fig(fig=fig)







fig, ax1 = create_figs.figure_axe(
    name_file='fig_ratio_strfct',
    fig_width_mm=200, fig_height_mm=155,
    size_axe=size_axe
    )

ax1.set_xlabel('$r/L_f$')
ax1.set_ylabel(
    r'$R_p(r) \equiv'
    r'  \langle |\delta u_L|^p \rangle '
    r'/ \langle |\delta u_T|^p \rangle$')

ax1.hold(True)
ax1.set_xscale('log')
# ax1.set_yscale('log')

dark_red = (0.9,0,0)

value_shocks = 2
linewidth=2.2
lq2 = ax1.plot(radim, ratioS2, color=dark_red, linewidth=linewidth)
ax1.plot([r1, r2], [value_shocks, value_shocks], ':',
         color=dark_red, linewidth=linewidth)

value_shocks = 6*np.pi/8
lq3 = ax1.plot(radim, ratioS3, 'b--', linewidth=linewidth)
ax1.plot([r1, r2], [value_shocks, value_shocks], 'b:', linewidth=linewidth)

value_shocks = 8./3
lq4 = ax1.plot(radim, ratioS4, 'c', linewidth=linewidth)
ax1.plot([r1, r2], [value_shocks, value_shocks], 'c:', linewidth=linewidth)

# value_shocks = 15*np.pi/16
# ax1.plot(radim, ratioS5, 'k', linewidth=2)
# ax1.plot([r1, r2], [value_shocks, value_shocks], 'k:', linewidth=2)

# value_shocks = 3
# ax1.plot(radim, ratioS3bis, 'y--', linewidth=2)
# ax1.plot([r1, r2], [value_shocks, value_shocks], 'y:', linewidth=2)

ax1.set_xlim([r1, r2])
ax1.set_ylim([1, 4])

# fig.text(0.01, 0.945, r'(\textit{a})', fontsize=fontsize+2)




plt.rc('legend', numpoints=1)
leg1 = plt.figlegend(
    [lq2[0], lq3[0], lq4[0]], 
    ['$R_2$', '$R_3$', '$R_4$'], 
    loc=(0.6, 0.17), 
    labelspacing = 0.18
    )





create_figs.save_fig(fig=fig)














solveq2d.show()
