import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import fluidsim as fls
from fractions import Fraction
from base import _rxs_str_func, _k_f, matplotlib_rc, So_var_dict
from paths import paths_sim, exit_if_figure_exists, load_df
from make_fig_struct_order_246 import plot_df, fig_struct_order


def _label(key='ux', odr=5):
    suffix = {'': '',
        'ux': '_L',
        'uy': '_T'}
    numerator = (
        r'-\langle \delta u' + suffix[key] +'^{' +
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


if __name__ == '__main__':
    matplotlib_rc(fontsize=9)
    sns.set_palette("GnBu_d", 5)
    path_fig = exit_if_figure_exists(__file__, '.png')
    fig, ax = plt.subplots(1, 2, figsize=(5, 2.2), sharey=True)

    df_w = load_df("df_w")
    df_3840 = df_w[df_w["$n$"] == 3840]
    df_7680 = df_w[df_w["$n$"] == 7680]
    
    plot_df(df_3840, fig, [ax[0]], order=[5], label_func=_label, coeff=-1)
    plot_df(df_7680, fig, [ax[1]], order=[5], label_func=_label, coeff=-1)
    
    # Add arrow and text
    xy = (0.03, 40)
    xytext = (0.03, 4)
    ax[0].annotate(
        "increasing $c$", xy, xytext, "data",
        arrowprops={"arrowstyle": "simple"})
    for ax1 in ax:
        ax1.set_xlim([None, 1.5])
        ax1.set_ylim([1, 500])
    ax[1].set_ylabel(None)
    fig.tight_layout()
    fig.savefig(path_fig)
    fig.savefig(path_fig.replace('.png', ".pdf"))