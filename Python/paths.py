import os
import sys
import re
import gc
from glob import glob
from socket import gethostname
from collections import OrderedDict
import pandas as pd
import numpy as np

from fluiddyn.util.paramcontainer import ParamContainer
from fluiddyn.io.redirect_stdout import stdout_redirected

import fluidsim as fls

from base import _eps, _t_stationary, _k_d, _k_f, _k_diss, epsetstmax


def get_pathbase():
    hostname = gethostname()
    hostname = hostname.lower()
    if hostname.startswith('pelvoux'):
        pathbase = '/media/avmo/lacie/13KTH'
    elif any(map(hostname.startswith, ['legilnx', 'nrj1sv', 'meige'])):
        pathbase = '$HOME/useful/project/13KTH/DataSW1L_Ashwin/'
    elif hostname.startswith('kthxps'):
        pathbase = '/run/media/avmo/lacie/13KTH/'
        if not os.path.exists(pathbase):
            pathbase = '/scratch/avmo/13KTH/'
    else:
        raise ValueError('Unknown hostname')

    pathbase = os.path.abspath(os.path.expandvars(pathbase))
    if not os.path.exists(pathbase):
        raise ValueError('Path not found ' + pathbase)

    return pathbase


def keyparams_from_path(p):
    c = re.search('(?<=c=)[0-9]*', p, re.X).group(0)
    nh = re.search('(?<=_)[0-9]*(?=x)', p).group(0)
    try:
        Bu = re.search('(?<=Bu=)[0-9]*(\.[0-9])', p, re.X).group(0)
    except AttributeError:
        Bu = 'inf'

    params_xml_path = os.path.join(p, 'params_simul.xml')
    params = ParamContainer(path_file=params_xml_path)
    init_field = params.init_fields.type
    if init_field == 'noise':
        return init_field, c, nh, Bu, None
    else:
        return init_field, c, nh, Bu, params.preprocess.init_field_const

EFR = r'$\frac{<\bf \Omega_0 >}{{(P k_f^2)}^{2/3}}$'
pd_columns = [
    r'$n$', r'$c$', r'$\nu_8$', r'$f$', r'$\epsilon$', r'$\frac{k_{diss}}{k_f}$', 
    r'$F_f$', r'$Ro_f$', r'$Bu$',
    # '$\min h$', r'$\frac{\max |\bf u|}{c}$',
    EFR, r'$E$',
    '$t_{stat}$', r'$t_{\max}$', 'short name'
]


def pandas_from_path(p, key, as_df=False):
    init_field, c, nh, Bu, init_field_const = keyparams_from_path(p)
    params_xml_path = os.path.join(p, 'params_simul.xml')
    params = ParamContainer(path_file=params_xml_path)
    # sim = fls.load_sim_for_plot(p, merge_missing_params=True)
    
    c = int(c)
    kf = _k_f(params)
    kd_kf = _k_diss(params) / kf
    # ts = _t_stationary(path=p)
    # eps = _eps(t_start=ts, path=p)
    eps, E, ts, tmax = epsetstmax(p)
    efr = params.preprocess.init_field_const
    Fr = (eps / kf) ** (1./3) / c
    try:
        Ro = (eps * kf**2) ** (1./3) / params.f
    except ZeroDivisionError:
        Ro = np.inf
    minh = 0
    maxuc = 0
    # del sim
    gc.collect()
    data = [nh, c, params.nu_8, params.f, eps, kd_kf,
         Fr, Ro, Bu,
         # minh, maxuc,
         efr, E,
         ts, tmax, key
        ]
    if as_df:
        return pd.DataFrame(data, columns=pd_columns)
    else:
        return pd.Series(data, index=pd_columns)


def make_paths_dict(glob_pattern='SW1L*'):
    '''Returns an ordered dictionary of paths, containing keys as the
    independent simulation parameters

    '''
    paths = glob(glob_pattern)
    paths_dict = OrderedDict()
    for p in paths:
        init_field, c, nh, Bu, efr = keyparams_from_path(p)
        if efr is None:
            key = '{}_c{}nh{}Bu{}'.format(init_field, c, nh, Bu)
        else:
            key = '{}_c{}nh{}Bu{}efr{:.2e}'.format(init_field, c, nh, Bu, efr)

        paths_dict[key] = p

    return paths_dict


def specific_paths_dict():
    paths_dict = {}
    pathbase = get_pathbase()

    for pattern in ['/noise/SW1L*NOISE2*', '/vortex_grid/SW1L*VG*']:
        paths_dict.update(make_paths_dict(pathbase + pattern))

    return paths_dict


paths_sim = specific_paths_dict()
path_pyfig = os.path.join(os.path.dirname(__file__), '../Pyfig/')
if not os.path.exists(path_pyfig):
    os.mkdir(path_pyfig)


def exit_if_figure_exists(scriptname, extension='.png'):
    scriptname = os.path.basename(scriptname)
    figname = os.path.splitext(scriptname)[0].lstrip('make_') + extension
    figpath = os.path.join(path_pyfig, figname)

    if len(sys.argv) > 1 and 'remake'.startswith(sys.argv[1]) and os.path.exists(figpath):
        os.remove(figpath)

    if os.path.exists(figpath):
        print('Figure {} already made. {} exiting...'.format(figname, scriptname))
        sys.exit(0)
    else:
        print('Making Figure {}.. '.format(figname))
        return figpath
