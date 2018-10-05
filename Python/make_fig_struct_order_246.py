import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import fluidsim as fls
from fractions import Fraction
from base import _rxs_str_func, _k_f, matplotlib_rc, So_var_dict, set_share_axes
from paths import paths_sim, exit_if_figure_exists, load_df




def _label(key='ux', odr=5):
    suffix = {'': '',
        'ux': '_L',
        'uy': '_T'}
    numerator = (
        r'\langle |\delta u' + suffix[key] +'|^{' +
        f'{int(odr)}' + '} \\rangle'
    )
    exp2 = Fraction(odr, 3)
    exp1 = exp2 - 1
    denom_first_term = (
        (r'(L_f F_f^{1/2})^{' + f"{exp1}" '}')
        if exp1 != 1 else r'L_f F_f^{1/2}'
    )
    denominator = (
        denom_first_term +
        r'\epsilon^{' + f"{exp2}" + '}' + 'r'
    )
    return f"${numerator} / {denominator}$"


def fig_struct_order(
    path, fig, ax, eps, Fr, order=[2, 4, 6], tmin=10, tmax=1000,
    delta_t=0.5, key="ux",
    run_nb = 0, label_func=None, coeff=1, ylabel=True, test=False
):
    sim = fls.load_sim_for_plot(path, merge_missing_params=True)
    rxs, So_var_dict, deltax = _rxs_str_func(
        sim, order, tmin, tmax, delta_t, [key],
        cache=test
    )
    
    for ax1 in ax:
        ax1.set_xlabel('$r/L_f$')
        ax1.set_xscale('log')
        ax1.set_yscale('log')
    if label_func is None:
        label_func = _label
    
    L_f = np.pi / _k_f(sim.params)

    #color_list = ['r', 'b', 'g', 'c', 'm', 'y', 'k']
    color_list = sns.color_palette()
    if coeff == "1/c":
       coeff = 1 / sim.params.c2 ** 0.5
    elif coeff == "c":
        coeff = sim.params.c2 ** 0.5

    for i, (o, ax1) in enumerate(zip(order, ax)):
        ax1.set_ylabel(label_func(key, o))
        key_order = '{0}_{1:.0f}'.format(key, o)
        norm = (L_f * Fr**0.5)**(o / 3 - 1) * eps**(o/3) * rxs * coeff ** o
        So_var = So_var_dict[key_order] / norm
        ax1.plot(rxs / L_f, So_var, 
                 c=color_list[run_nb],
                 linewidth=1,
                 label=label_func(key, o))

        if o == 2:
            ax1.set_ylim([1e-1, 8])
        ax1.set_xlim([None, 2])

def plot_df(df, fig, ax, **kwargs):
    for run_nb, (idx, values) in enumerate(df.iterrows()):
        run = values["short name"]
        eps = values["$\epsilon$"]
        Fr = values["$F_f$"]
        tmin = values["$t_{stat}$"]
        fig_struct_order(
            paths_sim[run], fig, ax, eps, Fr, tmin=tmin, run_nb=run_nb, **kwargs)
        if "test" in kwargs and kwargs["test"] and run_nb == 1:
            break


if __name__ == '__main__':
    matplotlib_rc(fontsize=9)

    path_fig = exit_if_figure_exists(__file__, '.png')
    fig, ax = plt.subplots(3, 2, figsize=(5, 6),
                           sharex=True, sharey=False)
    # ax[0,0].set_title('$n=3840$')
    # ax[0,1].set_title('$n=7680$')
    for row in range(3):
        set_share_axes(ax[row,:], sharey=True)
    set_share_axes(ax, sharex=True)
    df_w = load_df("df_w")
    df_3840 = df_w[df_w["$n$"] == 3840]
    df_7680 = df_w[df_w["$n$"] == 7680]

    sns.set_palette("cubehelix", 5)
    plot_df(df_3840, fig, ax[:,0])
    sns.set_palette("cubehelix", 3)
    plot_df(df_7680, fig, ax[:,1])
    for ax1 in ax[:,1].flat:
        ax1.set_ylabel(None)
        ax1.yaxis.set_tick_params(which='both', labelleft=False, labelright=False)
        ax1.yaxis.offsetText.set_visible(False)

    for ax1 in ax[:-1,:].flat:
        ax1.set_xlabel(None)
        ax1.xaxis.offsetText.set_visible(False)

    # Add arrow and text
    xy = (0.03, 2)
    xytext = (0.03, 0.5)
    ax[0,0].annotate(
        "increasing $c$", xy, xytext, "data",
        arrowprops={"arrowstyle": "simple"})


    fig.tight_layout()
    fig.savefig(path_fig)
    fig.savefig(path_fig.replace('.png', ".pdf"))