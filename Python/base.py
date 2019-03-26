from __future__ import print_function, division
from warnings import warn
import itertools
import os

import pylab as pl
import numpy as np
import pandas as pd
import xarray as xr

import seaborn as sns
from scipy.optimize import curve_fit
from scipy.signal import medfilt
from scipy import stats

import h5py
from fluiddyn.util.paramcontainer import ParamContainer
from fluiddyn.output import rcparams
from fluidsim.solvers.sw1l.output.spatial_means import SpatialMeansSW1L

#from _try_dask_strfunc import strfunc_from_pdf3 as strfunc_from_pdf
from fluidsim.base.output.increments import strfunc_from_pdf
import fluidsim as fls

DPI = 300


def load_spatial_means(path, sim=None):
    if sim is None and path:
        if os.path.isdir(path):
            path_txt = os.path.join(path, "spatial_means.txt")
            path_json = os.path.join(path, "spatial_means.json")
        try:
            try:
                return SpatialMeansSW1L._load(path_txt)
            except AttributeError:
                from fluidsim.solvers.sw1l.output.spatial_means import _old_load_txt
                return _old_load_txt(path_txt)
        except FileNotFoundError:
            df = pd.read_json(path_json, orient="records", lines=True)
            return xr.Dataset.from_dataframe(df)
    else:
        dico = sim.output.spatial_means.load()

def get_font(size=10):
    _font = {'family': 'serif',
             'serif': ['DejaVu Sans', 'Liberations Sans'],
             'weight': 'normal',
             'size': size,
             }
    return _font


def matplotlib_rc(
        fontsize=7, dpi=DPI, tex=True, interactive=False, pad=2):

    # pl.rc('text', usetex=tex)
    _font = get_font(fontsize)
    pl.rc('font', **_font)

    rcparams.set_rcparams(fontsize)
    pl.rc('figure', dpi=dpi)
    pl.rc('xtick', direction='in')
    pl.rc('ytick', direction='in')

    if interactive:
        pl.ion()
    else:
        pl.ioff()

    # pl.tight_layout(pad=pad)


def set_figsize(*size):
    pl.rcParams['figure.figsize'] = tuple(size)

def load_params(path):
    if os.path.isdir(path):
        path = os.path.join(path, 'params_simul.xml')
    return ParamContainer(path_file=path)

def _index_where(arr, value, reverse=False):
    if not reverse:
        idx = pl.argmin(abs(arr - value))
    else:
        revidx = pl.argmin(abs(arr[::-1] - value))
        idx = len(arr) - (revidx + 1)
    return idx

def _index_flat(y, x=None):
    dx = x[1]-x[0]
    dy = np.gradient(y, dx)
    ddy = np.gradient(dy, dx)
    curv = np.abs(ddy / (dy + 1) ** 1.5)
    # value = max(1e-5, 0.01 * curv.max())
    value = curv.std() * 0.01
    return _index_where(curv, value, True)
    # return np.argmin(abs(curv))


def _delta_x(params):
    Lh = min(params.oper.Lx, params.oper.Ly)
    nh = min(params.oper.nx, params.oper.ny)
    return Lh / nh

def _k_d(params):
    return params.f / params.c2 ** 0.5

def _k_f(params):
    Lh = min(params.oper.Lx, params.oper.Ly)
    return 2 * pl.pi / Lh * ((params.forcing.nkmax_forcing +
                              params.forcing.nkmin_forcing) // 2)

def _k_max(params):
    delta_x = _delta_x(params)
    k_max = pl.pi / delta_x * params.oper.coef_dealiasing
    return k_max

def _k_diss(params):
    k_max = _k_max(params)
    C = params.preprocess.viscosity_const
    return k_max / C / pl.pi  # FIXME: Not sure why the pi is needed

def _eps(sim=None, t_start=0, path=None):
    dico = load_spatial_means(path, sim)
    t = dico['t']
    ind = _index_where(t, t_start)
    eps = dico['epsK_tot'] + dico['epsA_tot']
    return float(eps[ind:].mean())


def _t_stationary(sim=None, eps_percent=15, path=None):
    # Old function! Use epststmax instead
    dico = load_spatial_means(path, sim)
    time = dico['t']
    epsilon = dico['epsK_tot'] + dico['epsA_tot']
    eps_end = epsilon[-1]

    for t, eps in zip(time[::-1], epsilon[::-1]):
        percent = abs(eps - eps_end) / eps_end * 100
        if percent > eps_percent:
            break

    return t


def epsetstmax(path):
    dico = load_spatial_means(path)
    time = dico['t']
    eps = dico['epsK_tot'] + dico['epsA_tot']
    E = dico['EK'] + dico['EA']
    
    # if 'noise' in path:
    if eps.max() > 2 * eps[-1]:
        def f(x, amptan, ttan):
            return amptan * pl.tanh(2 * (x / ttan)**4)

        guesses = np.array([pl.median(eps), time[eps==eps.max()][0]])
    else:
        # def f(x, amptan, ttan, amplog, sigma):
        def f(x, amptan, ttan, amplog, tlog, sigma):
            return (amptan * pl.tanh(2 * (x/ttan)**4) +
                    amplog * stats.lognorm.pdf(x, scale=pl.exp(tlog), s=sigma))

        guesses = np.array((
            # amptan'
            pl.median(eps),
            # ttan'
            time[eps==eps.max()],
            # amplog
            eps.max(),
            # tlog
            time[eps==eps.max()],
            # sigma
            eps.std()
        ), dtype=float)
        # guesses = pl.array(list(guessesd.values()), dtype=float)

    try:
        popt, pcov = curve_fit(f, time, eps, guesses, maxfev=3000)
    except RuntimeError:
        print("Error while curve fitting data from path =", path)
        raise

    eps_fit = f(time, *popt)
    eps_stat = float(eps_fit[-1])
    try:
        # idx = _index_flat(eps_fit, time)
        idx = locate_knee(time, eps_fit, eps_stat)
        time_stat = time[idx]
    except ValueError:
        raise ValueError("While calculating curvature in {}".format(path))
        # warn("While calculating curvature in {}".format(path))
        # time_stat = popt[1] + 6 * popt[3]

    E_stat = E[idx:].mean()
    return eps_stat, E_stat, time_stat, time[-1]

def locate_knee(time, eps_fit, eps_stat):
    from kneed import KneeLocator
    
    # cast as numpy arrays due to some bug with xarrays/pandas indexing
    time = np.array(time)
    eps_fit = np.array(eps_fit)
    eps_stat = np.array(eps_stat)

    while not np.array_equal(time, np.sort(time)):
        idx_del = np.where(np.diff(time) < 0)[0] + 1
        time = np.delete(time, idx_del)
        eps_fit = np.delete(eps_fit, idx_del)
    
    if eps_fit.max() > 2 * eps_stat:
        # log-norm + tanh
        knee = KneeLocator(time, eps_fit, direction='decreasing')
        idx = knee.knee_x
    else:
        knee = KneeLocator(time, eps_fit)
        idx = knee.knee_x

    if idx is None:
        # non-stationary case
        idx = -1

    return idx

So_var_dict = {}

def step_info(t, yout, thresh_rise=10, thresh_settling=30):
    thresh_rise = 1 - thresh_rise / 100
    thresh_settling = 1 + thresh_settling / 100
    
    yout = medfilt(yout, 69)  # median filter

    overshoot_percent=(yout.max() / yout[-1] - 1) * 100
    # rise_time_orig=(
    #     t[next(i for i in range(0,len(yout)-1)
    #            if yout[i]>yout[-1]*thresh_rise)]
    #     - t[0]
    # )
    
    # idx = np.where(yout > yout[-1] * thresh_rise)[0][0]
    # rise_time = t[idx] - t[0]
    # assert rise_time_orig == rise_time

    # settling_time_orig=(
    #     t[next(len(yout)-i for i in range(2,len(yout)-1)
    #            if abs(yout[-i]/yout[-1])>thresh_settling)]
    #     - t[0]
    # )
    idx = np.where(np.abs(yout / yout[-1]) > thresh_settling)[0][-1]
    settling_time = t[idx] - t[0]
    # assert settling_time_orig == settling_time

    result = dict(
        overshoot_percent=(yout.max() / yout[-1] - 1) * 100,
        # rise_time=rise_time,
        settling_time=settling_time,
    )
    return result

def epststmax_stepinfo(path):
    dico = load_spatial_means(path)
    time = dico['t']
    eps = dico['epsK_tot'] + dico['epsA_tot']
    
    try:
        step = step_info(time, eps)
    except IndexError:
        print("index error in", path)
        return 0,0,0
    time_stat = step["settling_time"]
    idx = _index_where(time, time_stat)
    eps_stat = np.median(eps[idx:])
    return eps_stat, time_stat, time[-1]

def _rxs_str_func(
    sim, order, tmin, tmax, delta_t, key_var, cache=True, force_absolute=False
):
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
    deltax = Lx / nx

    rxs = np.array(rxs, dtype=np.float64) * deltax

    delta_t_save = np.mean(times[1:] - times[0:-1])
    delta_i_plot = int(round(delta_t / delta_t_save))
    if delta_i_plot == 0 and delta_t != 0.:
        delta_i_plot = 1
    delta_t = delta_i_plot * delta_t_save

    imin_plot = np.argmin(abs(times - tmin))
    imax_plot = np.argmin(abs(times - tmax))

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
            if key_order in So_var_dict.keys() and cache:
                print(
                    "Skipping calculation because it exists in cache.")
                continue

            pdf_var, values_inc_var, nb_rx_to_plot = self.load_pdf_from_file(
                tmin=tmin, tmax=tmax, key_var=key)

            absolute = True if force_absolute else (o % 2 == 0) # Use absolute values if order is even

            So_var_dict[key_order] = (
                self.strfunc_from_pdf(
                #strfunc_from_pdf(rxs,
                pdf_var, values_inc_var, o, absolute)
            )

    return rxs, So_var_dict, deltax


def markers():
    return iter(['o', '^', 'x', 's', 'D', '*'])


def linestyles():
    return itertools.cycle(['-', '--', ':', '-.'])


def rev_legend(ax, *args, **kwargs):
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], *args, **kwargs)


def palette(nb_colors=5, preview=False, **kwargs):
    defaults = dict(rot=0.85, hue=1)
    defaults.update(kwargs)
    pal = sns.cubehelix_palette(nb_colors, **defaults)
    if preview:
        sns.palplot(pal2)
    return pal


def set_share_axes(axs, target=None, sharex=False, sharey=False):
    if target is None:
        target = axs.flat[0]
    # Manage share using grouper objects
    for ax in axs.flat:
        if sharex:
            target._shared_x_axes.join(target, ax)
        if sharey:
            target._shared_y_axes.join(target, ax)
    # Turn off x tick labels and offset text for all but the bottom row
    if sharex and axs.ndim > 1:
        for ax in axs[:-1,:].flat:
            ax.xaxis.set_tick_params(
                which='both', labelbottom=False, labeltop=False)
            ax.xaxis.offsetText.set_visible(False)
    # Turn off y tick labels and offset text for all but the left most column
    if sharey and axs.ndim > 1:
        for ax in axs[:,1:].flat:
            ax.yaxis.set_tick_params(
                which='both', labelleft=False, labelright=False)
            ax.yaxis.offsetText.set_visible(False)
