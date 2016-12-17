

from __future__ import print_function

import os
import glob
import numpy as np

import baseSW1lw
from solveq2d import solveq2d

nb_mm_between_c = 1

dir_base  = baseSW1lw.path_base_dir_results
paths_from_nh_c_f = baseSW1lw.paths_from_nh_c_f

kf = baseSW1lw.kf
Lf = baseSW1lw.Lf
print('kf = {0:.2f};    Lf =  {1:.2f}'.format(kf, Lf))

path_here = os.getcwd()
path_file = 'table_try'
ii=0
while os.path.exists(
    path_here+'/Table_files/'+path_file+'{0}.txt'.format(ii)):
    ii+=1
path_file = path_here+'/Table_files/'+path_file+'{0}.txt'.format(ii)
ftable = open(path_file, 'w')


ftable.write(
r'$n$ & $c$ & '
r'$\nu_8$ & '
r'$f$ & $Ro$ & $Bu$ & '
r'$\eps$ & '
r'$\displaystyle\frac{\kmax}{k_d}$ & '
r'$\displaystyle\frac{k_d}{k_f}$ & '
r'$F_f$ & '
r'$\min h$ & $\displaystyle\frac{\max |\uu|}{c}$ '
r'\\[3mm]'
)



# ftable.write(
# r'$n$ & $c$ & '
# r'$f$ & $Ro$ & $Bu$ & '
# r'$\eps$ & $F_f$ & '
# r'$\min h$ & $\displaystyle\frac{\max |\uu|}{c}$ & '
# r'$\displaystyle\frac{\kmax}{k_d}$ \\[3mm]'
# )


def minhmaxu_from_sim(sim):
    """Return minh and maxu."""
    dico = sim.output.prob_dens_func.load()
    bin_edges_eta = dico['bin_edges_eta']
    bin_edges_u = dico['bin_edges_u']
    return 1. + bin_edges_eta.min(), bin_edges_u.max()




def print1sim(set_of_dir, f):
    """Print informations on one simulation."""

    print('\n\n')

    set_of_dir = set_of_dir.filter(f=f)
    path = set_of_dir.path_larger_t_start()

    sim = solveq2d.create_sim_plot_from_dir(path)
    param = sim.param
    nx = param['nx']
    c = np.sqrt(sim.param['c2'])
    nu8 = param['nu_8']
    f = sim.param['f']

    tmin, tmax, tstatio = baseSW1lw.tminmaxstatio_from_sim(sim, VERBOSE=True)

    dico1, dico2 = sim.output.spatial_means.compute_time_means(tstatio=tstatio)
    epsK = dico1['epsK']
    epsA = dico1['epsA']
    eps = epsK + epsA
    Ff = eps**(1./3)  *kf**(-1./3) /c

    n = 8
    kd8 = (eps/nu8**3)**(1./(3*n-2))
    kmax = param['coef_dealiasing']*np.pi*nx/param['Lx']


    minh, maxu = minhmaxu_from_sim(sim)

    ftable.write('\n')
    ftable.write('{0:4d} & {1:4.0f} & '.format(nx, c))

    ftable.write('{0:7.1e} & '.format(nu8))

    if f > 0:
        Ro = eps**(1./3) *kf**(2./3) /f
        Bu = (kf*c/f)**2
        print('Ro = {0:.2f};    Bu =  {1:.2f}'.format(Ro, Bu))

        ftable.write('{0:4.1f} & {1:6.2g} & {2:6.2g} & '.format(f, Ro, Bu))
    else:
        ftable.write(' 0   &  $\infty$ & $\infty$ & ')



    ftable.write('{0:4.2f} & '.format(eps))

    ftable.write('{0:.2f} & '.format(kmax/kd8))

    ftable.write('{0:3.0f} & '.format(kd8/kf))

    ftable.write('{0:6.3f} & '.format(Ff))

    ftable.write('{0:4.2f} & {1:4.2f} '.format(minh, maxu/c))

    ftable.write('\\\\')












c = 20
nh = 1920
paths = paths_from_nh_c_f(nh, c)
set_of_dir = solveq2d.SetOfDirResults(paths)

# print(set_of_dir.paths)



for f in set_of_dir.dico_values['f']:
    print1sim(set_of_dir, f)



