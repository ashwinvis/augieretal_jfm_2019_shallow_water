import numpy as np
import matplotlib.pyplot as plt
import fluidsim as fls
from fractions import Fraction
from base import _rxs_str_func, _k_f, matplotlib_rc, So_var_dict
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
    path, fig, ax, eps, Fr, order=[2, 4, 6], tmin=10, tmax=1000, delta_t=0.5,
    run_nb = 0, label_func=None, coeff=1
):
    sim = fls.load_sim_for_plot(path, merge_missing_params=True)
    key = "ux"
    rxs, So_var_dict, deltax = _rxs_str_func(
        sim, order, tmin, tmax, delta_t, [key], cache=False)
    
    for ax1 in ax:
        ax1.set_xlabel('$r_x/L_f$')
        ax1.set_xscale('log')
        ax1.set_yscale('log')
    if label_func is None:
        label_func = _label
    
    L_f = np.pi / _k_f(sim.params)

    color_list = ['r', 'b', 'g', 'c', 'm', 'y', 'k']

    for i, (o, ax1) in enumerate(zip(order, ax)):
        ax1.set_ylabel(label_func(key, o))
        key_order = '{0}_{1:.0f}'.format(key, o)
        norm = (L_f * Fr**0.5)**(o / 3 - 1) * eps**(o/3) * rxs * coeff
        So_var = So_var_dict[key_order] / norm
        ax1.plot(rxs / L_f, So_var, color_list[run_nb], linewidth=1,
                 label=label_func(key, o))

        # ax1.set_ylim([0.1, None])
        ax1.set_xlim([None, 2])

def plot_df(df, fig, ax, **kwargs):
    for run_nb, (idx, values) in enumerate(df.iterrows()):
        run = values["short name"]
        eps = values["$\epsilon$"]
        Fr = values["$F_f$"]
        tmin = values["$t_{stat}$"]
        fig_struct_order(
            paths_sim[run], fig, ax, eps, Fr, tmin=tmin, run_nb=run_nb, **kwargs)
        # if run_nb == 1:
        #     break


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
            ax.xaxis.set_tick_params(which='both', labelbottom=False, labeltop=False)
            ax.xaxis.offsetText.set_visible(False)
    # Turn off y tick labels and offset text for all but the left most column
    if sharey and axs.ndim > 1:
        for ax in axs[:,1:].flat:
            ax.yaxis.set_tick_params(which='both', labelleft=False, labelright=False)
            ax.yaxis.offsetText.set_visible(False)

if __name__ == '__main__':
    matplotlib_rc(fontsize=9)
    path_fig = exit_if_figure_exists(__file__, '.png')
    fig, ax = plt.subplots(3, 2, figsize=(5, 6), sharey=False)

    df_w = load_df("df_w")
    df_3840 = df_w[df_w["$n$"] == 3840]
    df_7680 = df_w[df_w["$n$"] == 7680]
    
    plot_df(df_3840, fig, ax[:,0])
    plot_df(df_7680, fig, ax[:,1])
    
    # Add arrow and text
    xy = (0.03, 3)
    xytext = (0.03, 0.3)
    ax[0,0].annotate(
        "increasing $c$", xy, xytext, "data",
        arrowprops={"arrowstyle": "simple"})

    # ax[0,0].set_title('$n=3840$')
    # ax[0,1].set_title('$n=7680$')
    for row in range(3):
        set_share_axes(ax[row,:], sharey=True)
    fig.tight_layout()
    fig.savefig(path_fig)
    fig.savefig(path_fig.replace('.png', ".pdf"))