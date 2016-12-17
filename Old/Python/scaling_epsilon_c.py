
from __future__ import print_function

import os
import numpy as np
import matplotlib.pyplot as plt


from create_figs_articles import CreateFigArticles
SAVE_FIG = 0



c = np.array([10, 20, 40, 70, 100, 200])
eps = np.array([1, 0.94, 0.93, 0.89, 0.76, 0.8])
# eps = np.array([1, 0.94, 0.93, 0.89, 0.85, 0.8])

c = np.array([10, 20, 20, 40, 70, 100, 200, 400, 700, 1000])
eps = np.array([1, 0.99, 0.97, 0.93, 0.9, 0.89, 0.88, 0.85, 0.79, 0.79])






fontsize=20
create_fig = CreateFigArticles(
    short_name_article='SW1l', 
    SAVE_FIG=SAVE_FIG, 
    FOR_BEAMER=False, 
    fontsize=fontsize
    )


fig, ax1 = create_fig.figure_axe(name_file='fig_eps_c')
ax1.set_xlabel(r'$c$')
ax1.set_ylabel(r'$\varepsilon$')

coef = 0.0

ax1.plot(c, eps*c**coef, 'k', linewidth=2)


# ax1.set_xscale('log')
# ax1.set_yscale('log')

# ax1.set_xlim([kmin, kmax])
# ax1.set_ylim([8e-3, 3e0])


# create_fig.save_fig()


create_fig.show()

