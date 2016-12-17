

import numpy as np
import commands
import glob
from solveq2d import solveq2d


Lh = 50.
deltak = 2*np.pi/Lh
kf = 6*deltak
Lf = 2*np.pi/kf/2


P0 = 1.

Tf = (P0*kf**2)**(-1./3)
Ef = (P0/kf)**(2./3)


out = commands.getstatusoutput('hostname')
computer = out[1]


if computer == 'pierre-KTH':
    path_base_dir_results = '/storage1/Results_SW1lw'
elif computer == 'pierre-voyage':
    path_base_dir_results = '/home/pierre/Dropbox/Results_SW1lw'
elif computer == 'pelvoux':
    path_base_dir_results = '/scratch/augier/Results_SW1lw'
else:
    raise ValueError('Unknown computer...')



def tminmaxstatio_from_sim(sim, VERBOSE=True):
    """Return tmin, tmax and tstatio."""
    tmin = sim.output.spatial_means.time_first_saved()
    tmax = sim.output.spatial_means.time_last_saved()
    Deltat = tmax - tmin
    if Deltat > 29:
        tstatio = tmin+18
    elif Deltat > 10:
        tstatio = tmin+5
    else:
        tstatio = tmax-2.5

    # special cases:
    if sim.param.dir_save_run.endswith('3840_c=20_f=0_2013-10-04_12-17-26'):
        tstatio = tmin+18
        # tstatio = tmin
    elif sim.param.dir_save_run.endswith(
        '1920_c=20_f=0_2013-10-03_14-13-34'):
        tstatio = tmin

    elif sim.param.dir_save_run.endswith(
        '1920_c=1000_f=0_2013-10-15_15-40-29'):
        tstatio = tmin

    elif sim.param.dir_save_run.endswith(
        '1920_c=700_f=0_2013-10-15_15-33-51'):
        tstatio = tmin

    elif sim.param.dir_save_run.endswith(
        '1920_c=400_f=0_2013-10-17_08-58-25'):
        tstatio = tmin




    if VERBOSE:
        print('tmin = {0:.2f}, tmax = {1:.2f}, tstatio = {2:.2f}'.format(
                tmin,tmax,tstatio)
              )

    return tmin, tmax, tstatio



def paths_from_c(c):
    """Return paths corresponding to one c."""
    resols = np.array([240, 480, 960, 1920, 3840, 5760, 7680])

    paths = []
    for resol in resols:
        str_resol = repr(resol)
        str_to_find_path = (
            path_base_dir_results+'/Pure_standing_waves_'+
            str_resol+'*/SE2D*c='+repr(c)+'_*'
            )
        paths_dir = glob.glob(str_to_find_path)
        for path in paths_dir:
            paths.append(path)
    return paths



def paths_from_nh_c_f(nh, c, f=None):
    """Return paths corresponding to nh and c."""
    str_nh = repr(nh)
    str_to_find_path = (
        path_base_dir_results+'/Pure_standing_waves_'+
        str_nh+'*/SE2D*c='+repr(c)+'_*'
        )
    # print str_to_find_path
    paths = glob.glob(str_to_find_path)

    if len(paths) == 0:
        print('No path for resol = {0} and c = {1}'.format(nh, c))

    if f is not None:
        set_of_dir = solveq2d.SetOfDirResults(paths)
        set_of_dir = set_of_dir.filter(f=f)
        paths = set_of_dir.paths

    return paths


def paths_from_nh(nh):
    """Return paths corresponding to nh and c."""
    str_nh = repr(nh)
    str_to_find_path = (
        path_base_dir_results+'/Pure_standing_waves_'+
        str_nh+'*/SE2D*'
        )
    print str_to_find_path
    paths = glob.glob(str_to_find_path)

    if len(paths) == 0:
        print('No path for resol = {0}'.format(nh))

    return paths

