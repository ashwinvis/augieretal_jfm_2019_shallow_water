import os
import subprocess
from glob import glob
import re
from collections import OrderedDict
from fluiddyn.util.paramcontainer import ParamContainer


def get_pathbase():
    hostname = subprocess.check_output('hostname')
    hostname = hostname.lower()
    if hostname.startswith('pelvoux'):
        pathbase = '/scratch/avmo/data/'
    elif any(map(hostname.startswith, ['legilnx', 'nrj1sv', 'meige'])):
        pathbase = '$HOME/useful/project/13KTH/DataSW1L_Ashwin/'
    else:
        raise ValueError('Unknown hostname')

    pathbase = os.path.abspath(os.path.expandvars(pathbase))
    if not os.path.exists(pathbase):
        raise ValueError('Path not found ' + pathbase)

    return pathbase


def make_paths_dict(glob_pattern='SW1L*'):
    '''Returns an ordered dictionary of paths, containing keys as the
    independent simulation parameters

    '''
    paths = glob(glob_pattern)
    paths_dict = OrderedDict()
    for p in paths:
        params_xml_path = os.path.join(p, 'params_simul.xml')
        params = ParamContainer(path_file=params_xml_path)
        c = re.search('(?<=c=)[0-9]*', p, re.X).group(0)
        nh = re.search('(?<=_)[0-9]*(?=x)', p).group(0)
        try:
            Bu = re.search('(?<=Bu=)[0-9]*(\.[0-9])', p, re.X).group(0)
        except AttributeError:
            Bu = 'inf'

        init_field = params.init_fields.type
        if init_field == 'noise':
            key = '{}_c{}nh{}Bu{}'.format(init_field, c, nh, Bu)
        else:
            efr = params.preprocess.init_field_const
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
