


from __future__ import print_function

import os
import sys
import glob
import numpy as np

import matplotlib.pylab as plt


import baseSW1lw

# small trick in order to import the module lindborg1999.py
path_here = os.getcwd()
path_mod_lindborg1999 = os.path.split(path_here)[0]+'/Flatness_atm'
sys.path.append(path_mod_lindborg1999)
from lindborg1999 import r, FL, FT

flatnessT = FT
flatnessL = FL

from create_figs_articles import CreateFigArticles
SAVE_FIG = 1






fontsize = 21

create_fig = CreateFigArticles(
    short_name_article='SW1l', 
    SAVE_FIG=SAVE_FIG, 
    FOR_BEAMER=False, 
    fontsize=fontsize
    )




fig, ax1 = create_fig.figure_axe(
    name_file='fig_flatness_atm',
    fig_width_mm=200, fig_height_mm=155,
    size_axe=[0.13, 0.127, 0.84, 0.835]
    )

ax1.set_xscale('log')
ax1.set_yscale('log')

ax1.set_xlabel('$r$ (km)')
ax1.set_ylabel('$F_T$, $F_L$')

l_FT = ax1.plot(r, flatnessT, 'k', linewidth=2)
l_FL = ax1.plot(r, flatnessL, 'y', linewidth=2)


plt.rc('legend', numpoints=1)
leg1 = plt.figlegend(
    [l_FT[0], l_FL[0]], 
    ['$F_T$', '$F_L$'], 
    loc=(0.2, 0.2), 
    labelspacing = 0.18
    )

cond = np.logical_and(r > 3 , r < 2e2)
ax1.plot(r[cond], 6e2*r[cond]**(-1), 
                 'k', linewidth=1)
ax1.text(1e1, 1.8e1,'$r^{-1}$',fontsize=fontsize)

cond = np.logical_and(r > 3 , r < 1e2)
ax1.plot(r[cond], 4e4*r[cond]**(-3./2), 
                 'k--', linewidth=1)
ax1.text(1e1, 1.8e3,'$r^{-3/2}$',fontsize=fontsize)


r1 = 1
r2 = 3e3

ax1.set_xlim([r1, r2])
ax1.set_ylim([1, 8e3])



ax2 = fig.add_axes([0.62, 0.6, 0.32, 0.32])

ax2.set_xscale('log')
ax2.plot(r, flatnessT/flatnessL, 'k', linewidth=2)
value_shocks = 1.5
ax2.plot([r1, r2], [value_shocks, value_shocks], 'k:', linewidth=2)

ax2.set_xlim([r1, r2])
ax2.set_ylim([0, 1.7])
ax2.set_yticks([0, 0.5, 1, 1.5])

ax2.set_xlabel('$r$')
ax2.set_ylabel('$F_T/F_L$')

for item in ([ax2.xaxis.label, ax2.yaxis.label] +
             ax2.get_xticklabels() + ax2.get_yticklabels()):
    item.set_fontsize(fontsize-4)



fig.text(0.01, 0.945, r'(\textit{b})', fontsize=fontsize+2)

create_fig.save_fig(fig=fig)



create_fig.show()

