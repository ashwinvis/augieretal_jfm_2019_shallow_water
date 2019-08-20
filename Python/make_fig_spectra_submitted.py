#!/usr/bin/env python
from fractions import Fraction
import pylab as pl
import matplotlib.pyplot as plt
import seaborn as sns
import fluidsim as fls
import h5py

from base import (
    _k_f, _eps, set_figsize, matplotlib_rc, _index_where, linestyles, rev_legend)
from paths import paths_sim, exit_if_figure_exists, load_df


#color_list = \
#    iter(['r', 'b', 'g', 'c', 'm', 'y', 'k'])
    # iter(plt.cm.jet(pl.linspace(0,1,3)))
style = linestyles()


def _label(odr=2):
    numerator = 'E(k)'
    exp2 = Fraction(odr, 3)
    exp1 = exp2 - 1
    denom_first_term = (
        (r'(L_f F_f^{1/2})^{' + f"{exp1}" '}')
        if exp1 != 1 else r'L_f F_f^{1/2}'
    )
    denominator = (
        denom_first_term +
        r'\epsilon^{' + f"{exp2}" + '}' + 'k^{-2}'
    )
    return f"${numerator} / {denominator}$"


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

    # tmin_plot = times[imin_plot]
    # tmax_plot = times[imax_plot]
    machine_zero = 1e-15
    EK = dset_spectrumEK[imin_plot:imax_plot + 1].mean(0)
    EA = dset_spectrumEA[imin_plot:imax_plot + 1].mean(0)
    EK[abs(EK) < 1e-15] = machine_zero
    EA[abs(EA) < 1e-15] = machine_zero
    E_tot = EK + EA
    f.close()

    return kh, E_tot, EK, EA


def fig7_spectra(path, fig, ax, Fr, c, t_start, run_nb):
    sim = fls.load_sim_for_plot(path, merge_missing_params=True)
    kh, E_tot, EK, EA = _mean_spectra(sim, t_start)
    eps = _eps(sim, t_start)
    k_f = _k_f(sim.params)

#     norm = (kh ** (-2) *
#             sim.params.c2 ** (1. / 6) *
#             eps ** (5. / 9) *
#             k_f ** (4. / 9))
    o = 2
    L_f = pl.pi / k_f
    norm = (L_f * Fr**0.5)**(o / 3 - 1) * eps**(o/3) * kh**-2

    kh_f = kh / k_f
    ax.plot(
        kh_f, E_tot / norm,
        color="k",
        # color_list[run_nb],
        linestyle=next(style),
        linewidth=1.5,
        label=f'$c = {c}$')

    # ax.plot(kh_f, EK / norm, 'r', linewidth=2, label='$E_K$')
    # ax.plot(kh_f, EA / norm, 'b', linewidth=2, label='$E_A$')
    
    if run_nb == 0:
        s1 = slice(_index_where(kh_f, 3), _index_where(kh_f, 150))
        s2 = slice(_index_where(kh_f, 30), _index_where(kh_f, 200))
        ax.plot((kh_f)[s1], 0.7 * (kh_f ** -2 / norm)[s1], 'k-', linewidth=1)
        ax.text(10, 0.2, '$k^{-2}$')
        # ax.plot((kh_f)[s2], (kh_f ** -1.5 / norm)[s2], 'k-', linewidth=1)
        # ax.text(70, 1.5, '$k^{-3/2}$')
    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlabel('$k/k_f$')
    ax.set_ylabel(_label())
    # ax.legend()
    rev_legend(ax)
    fig.tight_layout()


def plot_df(df, fig, ax):     
    for run_nb, (idx, row) in enumerate(df.iterrows()):
        short_name = row["short name"]
        tmin = row["$t_{stat}$"]
        Fr = row["$F_f$"]
        c = row["$c$"]
        fig7_spectra(paths_sim[short_name], fig, ax, Fr, c, 
                     t_start=tmin,
                     run_nb=run_nb)


if __name__ == '__main__':
    #  sns.set_palette("GnBu_d", 3)
    matplotlib_rc(11)
    path_fig = exit_if_figure_exists(__file__)
    set_figsize(5, 3)
    fig, ax = pl.subplots()
    
    df_w = load_df("df_w")
    df_7680 = df_w[df_w["$n$"] == 7680]
    plot_df(df_7680, fig, ax)
    
    ax.set_ylim(1e-2, 3e1)
    ax.set_xlim(2e-1, 5e2)
    fig.savefig(path_fig)
    fig.savefig(path_fig.replace(".png", ".pdf"))