from __future__ import print_function, division
import pylab as pl
from fluiddyn.util.paramcontainer import ParamContainer


def _font(size=14):
    _font = {'family': 'serif',
             'weight': 'normal',
             'size': size,
             }
    pl.rc('font', **_font)


def _index_where(arr, value):
    return pl.argmin(abs(arr - value))


def _delta_x(params):
    Lh = min(params.oper.Lx, params.oper.Ly)
    nh = min(params.oper.nx, params.oper.ny)
    return Lh / nh


def _k_f(params=None, params_xml_path=None):
    if params is None:
        params = ParamContainer(path_file=params_xml_path)

    Lh = min(params.oper.Lx, params.oper.Ly)
    return 2 * pl.pi / Lh * ((params.forcing.nkmax_forcing +
                              params.forcing.nkmin_forcing) // 2)


def _k_max(params):
    delta_x = _delta_x(params)
    k_max = pl.pi / delta_x * params.oper.coef_dealiasing
    return k_max


def _eps(sim, t_start):
    dico = sim.output.spatial_means.load()
    t = dico['t']
    ind = _index_where(t, t_start)
    eps = dico['epsK_tot'] + dico['epsA_tot']
    return eps[ind:].mean()
