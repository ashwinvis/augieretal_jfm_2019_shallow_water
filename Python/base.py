from __future__ import print_function, division
import pylab as pl
import h5py
from fluiddyn.util.paramcontainer import ParamContainer


def get_font(size=16):
    _font = {'family': 'serif',
             'weight': 'normal',
             'size': size,
             }
    return _font


def matplotlib_rc(mathjax=True, fontsize=16, interactive=False):
    pl.rc('text', usetex=mathjax)
    _font = get_font(fontsize)
    pl.rc('font', **_font)
    if interactive:
        pl.ion()
    else:
        pl.ioff()


def set_figsize(*size):
    pl.rcParams['figure.figsize'] = tuple(size)


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


So_var_dict = {}


def _rxs_str_func(sim, order, tmin, tmax, delta_t, key_var):
    np = pl
    self = sim.output.increments

    f = h5py.File(self.path_file, 'r')
    dset_times = f['times']
    times = dset_times[...]
    # nt = len(times)

    if tmax is None:
        tmax = times.max()

    rxs = f['rxs'][...]

    oper = f['/info_simul/params/oper']
    nx = oper.attrs['nx']
    Lx = oper.attrs['Lx']
    deltax = Lx/nx

    rxs = np.array(rxs, dtype=np.float64)*deltax

    delta_t_save = np.mean(times[1:]-times[0:-1])
    delta_i_plot = int(round(delta_t/delta_t_save))
    if delta_i_plot == 0 and delta_t != 0.:
        delta_i_plot=1
    delta_t = delta_i_plot*delta_t_save

    imin_plot = np.argmin(abs(times-tmin))
    imax_plot = np.argmin(abs(times-tmax))

    tmin_plot = times[imin_plot]
    tmax_plot = times[imax_plot]

    to_print = 'plot(tmin={0}, tmax={1}, delta_t={2:.2f})'.format(
        tmin, tmax, delta_t)
    print(to_print)

    to_print = '''plot structure functions
tmin = {0:8.6g} ; tmax = {1:8.6g} ; delta_t = {2:8.6g}
imin = {3:8d} ; imax = {4:8d} ; delta_i = {5:8d}'''.format(
tmin_plot, tmax_plot, delta_t,
imin_plot, imax_plot, delta_i_plot)
    print(to_print)
    
    for o in order:
        o = float(o)
        for key in key_var:
            key_order = '{0}_{1:.0f}'.format(key, o)
            if key_order in So_var_dict.keys():
                continue

            pdf_var, values_inc_var, nb_rx_to_plot = self.load_pdf_from_file(
                tmin=tmin, tmax=tmax, key_var=key)
            
            So_var_dict[key_order] = self.strfunc_from_pdf(
                pdf_var, values_inc_var, o, absolute=True)

    return rxs, So_var_dict, deltax

