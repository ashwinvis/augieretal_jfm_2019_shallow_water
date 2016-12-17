#!/usr/bin/env python
#coding=utf8

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import glob

import baseSW1lw
from solveq2d import solveq2d

from create_figs_articles import CreateFigArticles



SAVE_FIG = 0
nh = 960*2
c = 20
# c = 200

if SAVE_FIG:
    nh = 960*2



fontsize = 21
create_fig = CreateFigArticles(
    short_name_article='SW1l',
    SAVE_FIG=SAVE_FIG, 
    FOR_BEAMER=False, 
    fontsize=fontsize
    )

paths = baseSW1lw.paths_from_nh_c_f(nh, c, f=0)

set_of_dir = solveq2d.SetOfDirResults(paths)
dirs = set_of_dir.dirs_from_values(solver='SW1lwaves',
                                   FORCING=True,
                                   c2=c**2)

path = set_of_dir.paths[0]

sim = solveq2d.load_state_phys_file(t_approx=1000, name_dir=path)

nx = sim.param.nx
c2 = sim.param.c2
c = np.sqrt(c2)
f = sim.param.f
name_solver = sim.output.name_solver

Lf = baseSW1lw.Lf

eta = sim.state.state_phys['eta']

ux = sim.state.state_phys['ux']
uy = sim.state.state_phys['uy']


x_seq = sim.oper.x_seq
y_seq = sim.oper.y_seq
[XX_seq, YY_seq] = np.meshgrid(x_seq, y_seq)


Lmax = 3.001*Lf
nxLmax = np.sum(x_seq<=Lmax)
nyLmax = np.sum(y_seq<=Lmax)

x_seq = x_seq[x_seq<=Lmax]
y_seq = y_seq[x_seq<=Lmax]



cond = np.logical_and(XX_seq<Lmax, YY_seq<Lmax)
XX_seq = XX_seq[cond].reshape([nyLmax, nxLmax])
YY_seq = YY_seq[cond].reshape([nyLmax, nxLmax])
eta = eta[cond].reshape([nyLmax, nxLmax])
ux = ux[cond].reshape([nyLmax, nxLmax])
uy = uy[cond].reshape([nyLmax, nxLmax])


field = eta
nb_contours = 25


maxfield = field.max()
minfield = field.min()

bornefield = np.max([maxfield, -minfield])
bornefield = round(bornefield*0.95, 2)
# print(bornefield)
# if c == 200:
#     bornefield = 0.03


name_file = 'fig_2Dh_c{0:.0f}'.format(c)


fig_width_mm=200
fig_height_mm=170
size_axe = [0.105, 0.11, 0.89, 0.82]
fig, ax = create_fig.figure_axe(name_file=name_file,
                                fig_width_mm=fig_width_mm, 
                                fig_height_mm=fig_height_mm,
                                size_axe=size_axe)

ax.set_xlabel(r'$x/L_f$')
ax.set_ylabel(r'$y/L_f$')


type_plot = 'contourf'

if type_plot == 'contourf':
    V = np.linspace(-bornefield, bornefield, num=nb_contours)
    field[field<-bornefield] = -bornefield*0.99
    field[field>bornefield] = bornefield*0.99
    contours = ax.contourf(x_seq/Lf, y_seq/Lf, field+1, 
                           V+1, cmap=plt.cm.jet)
    ticks = np.linspace(-bornefield, bornefield, num=5)+1
    # ticks = ticks.round(2)
    fig.colorbar(contours, ticks=ticks, ax=ax, shrink=1., pad=.02, aspect=20)
    fig.contours = contours
elif type_plot == 'pcolor':
    pc = ax.pcolor(x_seq, y_seq, field, 
                   cmap=plt.cm.jet)
    fig.colorbar(pc)






ax.set_xticks([0, 1, 2, 3])
ax.set_yticks([0, 1, 2, 3])
ax.set_xlim([0, 3])
ax.set_ylim([0, 3])


pas_vector = np.round(sim.oper.nx_seq/64)
if pas_vector < 1:
    pas_vector = 1

vecx = ux
vecy = uy

ax.quiver(XX_seq[::pas_vector, ::pas_vector]/Lf, 
          YY_seq[::pas_vector, ::pas_vector]/Lf, 
          vecx[::pas_vector, ::pas_vector], 
          vecy[::pas_vector, ::pas_vector])


if c == 20:
    for_letter = r'(\textit{a})'+r' $c = '+'{0:.0f}'.format(c)+'$'
elif c == 200:
    for_letter = r'(\textit{b})'+r' $c = '+'{0:.0f}'.format(c)+'$'
fig.text(0.01, 0.955, for_letter, fontsize=fontsize+2)

fig.text(0.9, 0.04, r'$h$', fontsize=fontsize+4)


create_fig.save_fig(format='eps')

























name_file = 'fig_2Duy_c{0:.0f}'.format(c)

fig, ax = create_fig.figure_axe(name_file=name_file,
                                fig_width_mm=fig_width_mm, 
                                fig_height_mm=fig_height_mm,
                                size_axe=size_axe)

ax.set_xlabel(r'$x/L_f$')
ax.set_ylabel(r'$y/L_f$')


field = uy

maxfield = field.max()
minfield = field.min()

bornefield = np.max([maxfield, -minfield])
bornefield = round(bornefield*0.75, 2)
# print(maxfield, minfield, bornefield)

type_plot = 'contourf'

if type_plot == 'contourf':
    V = np.linspace(-bornefield, bornefield, num=nb_contours)
    field[field<-bornefield] = -bornefield*0.99
    field[field>bornefield] = bornefield*0.99
    contours = ax.contourf(x_seq/Lf, y_seq/Lf, field, 
                           V, cmap=plt.cm.jet)
    ticks = np.linspace(-bornefield, bornefield, num=5)
    ticks = ticks.round(2)
    fig.colorbar(contours, ticks=ticks, ax=ax, shrink=1., pad=.02, aspect=20)
    fig.contours = contours
elif type_plot == 'pcolor':
    pc = ax.pcolor(x_seq, y_seq, field, 
                   cmap=plt.cm.jet)
    fig.colorbar(pc)


ax.set_xticks([0, 1, 2, 3])
ax.set_yticks([0, 1, 2, 3])
ax.set_xlim([0, 3])
ax.set_ylim([0, 3])

ax.quiver(XX_seq[::pas_vector, ::pas_vector]/Lf, 
          YY_seq[::pas_vector, ::pas_vector]/Lf, 
          vecx[::pas_vector, ::pas_vector], 
          vecy[::pas_vector, ::pas_vector])


if c == 20:
    for_letter = r'(\textit{c})'+r' $c = '+'{0:.0f}'.format(c)+'$'
elif c == 200:
    for_letter = r'(\textit{d})'+r' $c = '+'{0:.0f}'.format(c)+'$'
fig.text(0.01, 0.955, for_letter, fontsize=fontsize+2)

fig.text(0.9, 0.04, r'$u_y$', fontsize=fontsize+4)




create_fig.save_fig(format='eps')










create_fig.show()
