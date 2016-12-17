

import matplotlib.pylab as plt
import glob
import numpy as np

import baseSW1lw
from baseSW1lw import Lf
from solveq2d import solveq2d

from create_figs_articles import CreateFigArticles
name_file = 'fig_interm'

SAVE_FIG = 1




fontsize = 21
create_fig = CreateFigArticles(
    short_name_article='SW1l', 
    SAVE_FIG=SAVE_FIG, 
    FOR_BEAMER=False, 
    fontsize=fontsize
    )



c = 10 
c = 40
nh = 240*2**5

# c = 200
# nh = 240*2**4

key_var = 'uy' # for transverse since r = delta x
# key_var = 'ux' # for longitudinal since r = delta x









paths = baseSW1lw.paths_from_nh_c_f(nh, c, f=0)

set_of_dir = solveq2d.SetOfDirResults(paths)
path = set_of_dir.path_larger_t_start()

sim = solveq2d.create_sim_plot_from_dir(path)

tmin, tmax, tstatio = baseSW1lw.tminmaxstatio_from_sim(sim, VERBOSE=True)





(pdf_timemean, values_inc_timemean, nb_rx_to_plot
 ) = sim.output.increments.load_pdf_from_file(
    tmin=tmin, tmax=tmax, key_var=key_var)


deltax = sim.param.Lx/sim.param.nx
rxs = np.array(sim.output.increments.rxs, dtype=np.float64)*deltax

if nh == 7680:
    rmin = 8*deltax
    rmax = 40*deltax

elif nh == 3840:
    rmin = 6*deltax
    rmax = 20*deltax


print('rmin = {0} ; rmax = {1}'.format(rmin, rmax))

print('rmin/Lf = {0} ; rmax/Lf = {1}'.format(rmin/Lf, rmax/Lf))

# if 4096
# rmin = 8e-2
# rmin = 8*deltax
# rmax = 5e-1
# rmax = 50*deltax

# if 2048
# rmin = 1.5e-1
# rmin = 5*deltax
# rmax = 20*deltax
# rmax = 3e-1

condr = np.logical_and(rxs>rmin, rxs<rmax)

size_axe=[0.13, 0.12, 0.86, 0.835]


def expo_from_order(order, PLOT_STRFCT=False, PLOT_PDF=False):
    order = float(order)
    M_order = np.empty(rxs.shape)
    for irx in xrange(rxs.size):
        deltainc = values_inc_timemean[irx, 1] - values_inc_timemean[irx, 0]
        M_order[irx] = deltainc*np.sum( 
            pdf_timemean[irx]
            *abs(values_inc_timemean[irx])**order)

        # M_order[irx] = np.abs(deltainc*np.sum( 
        #     pdf_timemean[irx]
        #     *values_inc_timemean[irx]**order
        #     ))

    pol = np.polyfit(np.log(rxs[condr]), np.log(M_order[condr]), 1)
    expo = pol[0]

    print 'order = {0:.2f} ; expo = {1:.2f}'.format(order, expo)

    if PLOT_STRFCT:

        fig, ax1 = create_fig.figure_axe(
            name_file=name_file+'_strfct_'+key_var,
            fig_width_mm=200, fig_height_mm=155,
            size_axe=size_axe
            )

        ax1.set_xlabel('$r/L_f$')

        if key_var == 'ux':
            for_ylabel = ('$\langle |\delta u_L|^{'
                          +'{0:.0f}'.format(order)+'}\\rangle/r$')
            for_letter = r'(\textit{a})'
            for_title = '$p = 5$'
            y_for_zeta = 3.5
        elif key_var == 'uy':
            for_ylabel = ('$\langle |\delta u_T|^{'
                          +'{0:.0f}'.format(order)+'}\\rangle/r$')
            for_letter = r'(\textit{b})'
            for_title = '$p = 5$'
            y_for_zeta = 3.6

        ax1.set_ylabel(for_ylabel)

        ax1.hold(True)
        ax1.set_xscale('log')
        ax1.set_yscale('log')

        norm = rxs**1

        ax1.plot(rxs/Lf, M_order/norm, 
                 '-', linewidth=1)

        ax1.plot(rxs[condr]/Lf, M_order[condr]/norm[condr], 'x-r', linewidth=1)


        radimmin = 1.2e-3
        radimmax = 2

        radim = np.logspace(np.log10(radimmin), np.log10(radimmax), 200)
        r = radim*Lf

        norm = r**1

        cond = np.logical_and(radim > 7e-3 , radim < 2e-1)

        M_lin = np.exp((pol[1] + np.log(r)*pol[0]))
        ax1.plot(radim[cond], M_lin[cond]/norm[cond], 'y', linewidth=2)

        ax1.text(2e-2, y_for_zeta,
                 '$\zeta_p=$ {0:.3f}'.format(expo),fontsize=fontsize)

        cond = np.logical_and(radim > 7e-3 , radim < 2e-1)
        l_1 = ax1.plot(radim[cond], r[cond]/norm[cond], 
                       'k', linewidth=1)
        ax1.text(2.2e-2, 5e-1,'shocks, $r^1$',fontsize=fontsize)


        cond = np.logical_and(radim > 2e-2 , radim < 4e-1)
        l_K41 = ax1.plot(radim[cond], 2e2*r[cond]**(order/3)/norm[cond], 
                         'k--', linewidth=1)
        ax1.text(5e-2, 1.1e2,'$r^{p/3}$',fontsize=fontsize)


        cond = np.logical_and(r < 4*deltax , r > deltax)
        temp = r**order
        temp = temp[cond]/temp[cond][0]*M_order[0]/norm[0]
        rtemp = radim[cond]
        cond = temp < 1.9e1
        l_smooth= ax1.plot(
            rtemp[cond],temp[cond],
            'k', linewidth=1)
        ax1.text(2e-3, 2.e1,'smooth, $r^p$',fontsize=fontsize)


        ax1.set_xlim([radimmin, radimmax])
        ax1.set_ylim([5e-2, 6e2])

        ax1.text(radimmin*1.2, 3e2, for_title,fontsize=fontsize+2)
        fig.text(0.005, 0.945, for_letter, fontsize=fontsize+2)

        create_fig.save_fig(fig=fig)




    if PLOT_PDF:
        fig, ax1 = create_fig.figure_axe(name_file=name_file+'_pdf_'+key_var,
                                         fig_width_mm=200, fig_height_mm=155,
                                         size_axe=size_axe
                                         )
        ax1.hold(True)
        ax1.set_xscale('linear')
        ax1.set_yscale('linear')

        ax1.set_xlabel(key_var)
        ax1.set_ylabel('PDF x $\delta v^'+repr(order)+'$')
        
        colors = ['k', 'y', 'r', 'b', 'g', 'm', 'c']

        irx_to_plot = [10, 50, 100]
        for irxp, irx in enumerate(irx_to_plot):
            
            val_inc = values_inc_timemean[irx]

            ax1.plot(val_inc, pdf_timemean[irx]*abs(val_inc)**order, 
                     colors[irxp]+'x-', linewidth=1)

    return expo






PLOT_STRFCT = False
if PLOT_STRFCT:
    delta_order = 1
else:
    delta_order = 0.05

orders = np.arange(0., 6, delta_order)
expos = np.empty(orders.shape)

for iorder, order in enumerate(orders):
    expos[iorder] = expo_from_order(order, 
                                    PLOT_STRFCT=PLOT_STRFCT, 
                                    PLOT_PDF=False)

expos_K41 = orders/3

fig, ax1 = create_fig.figure_axe(name_file=name_file+'_'+key_var,
                                 fig_width_mm=200, fig_height_mm=155,
                                 size_axe=size_axe
                                 )


if key_var == 'ux':
    for_letter = r'(\textit{a})'
    for_title = '$\delta u_L$'

elif key_var == 'uy':
    for_letter = r'(\textit{b})'
    for_title = '$\delta u_T$'

fig.text(0.005, 0.945, for_letter, fontsize=fontsize+2)
ax1.text(0.12, 1.85, for_title,fontsize=fontsize+2)


ax1.hold(True)
ax1.set_xscale('linear')
ax1.set_yscale('linear')

ax1.set_xlabel('order p')
ax1.set_ylabel('$\zeta_p$')

ax1.plot(orders, expos)

ax1.plot(orders, expos_K41, 'k--')

ax1.text(4, 1.65,'$\zeta_p = p/3$',fontsize=fontsize)

ax1.set_ylim([0, 2])


create_fig.save_fig(fig=fig)









expo_from_order(order=5, 
                PLOT_STRFCT=True, 
                PLOT_PDF=False)












solveq2d.show()
