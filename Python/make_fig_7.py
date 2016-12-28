import pylab as pl
import fluidsim as fls
import h5py

from base import _k_f, _eps, set_figsize
from paths import paths_sim, path_pyfig


path_fig = path_pyfig + 'fig_7.png'


def _mean_spectra(sim, tmin=0, tmax=1000):

    f = h5py.File(sim.output.spectra.path_file2D, 'r')
    dset_times = f['times']
    # nb_spectra = dset_times.shape[0]
    times = dset_times[...]
    # nt = len(times)

    dset_khE = f['khE']
    kh = dset_khE[...]

    dset_spectrumEK = f['spectrum2D_EK']
    dset_spectrumEA = f['spectrum2D_EA']
    imin_plot = pl.argmin(abs(times - tmin))
    imax_plot = pl.argmin(abs(times - tmax))

    tmin_plot = times[imin_plot]
    tmax_plot = times[imax_plot]
    machine_zero = 1e-15
    EK = dset_spectrumEK[imin_plot:imax_plot + 1].mean(0)
    EA = dset_spectrumEA[imin_plot:imax_plot + 1].mean(0)
    EK[abs(EK) < 1e-15] = machine_zero
    EA[abs(EA) < 1e-15] = machine_zero
    E_tot = EK + EA

    return kh, E_tot, EK, EA


def fig7_spectra(path, fig, ax, t_start):
    sim = fls.load_sim_for_plot(path)
    kh, E_tot, EK, EA = _mean_spectra(sim, t_start)
    eps = _eps(sim, t_start)
    k_f = _k_f(sim.params)

    norm = (kh ** (-2) *
            sim.params.c2 ** (1. / 6) *
            eps ** (5. / 9) *
            k_f ** (4. / 9))

    ax.plot(kh / k_f, E_tot / norm, 'k', linewidth=4, label='E')
    ax.plot(kh / k_f, EK / norm, 'r', linewidth=2, label='$E_K$')
    ax.plot(kh / k_f, EA / norm, 'b', linewidth=2, label='$E_A$')
    ax.set_xscale('log')
    ax.set_yscale('log')

    lin_inf, lin_sup = ax.get_ylim()
    if lin_inf < 1e-2:
        lin_inf = 1e-2

    ax.set_ylim([lin_inf, lin_sup])

    ax.set_xlabel('$k/k_f$')
    ax.set_ylabel(r'$E(k)/(k^{-2}c^{1/3}\epsilon^{5/9}k^{4/9}_f$')
    ax.legend()


if __name__ == '__main__':
    set_figsize(10, 6)
    fig, ax = pl.subplots()
    fig7_spectra(paths_sim['noise_c100nh7680Buinf'], fig, ax, t_start=25)
    pl.savefig(path_fig)
