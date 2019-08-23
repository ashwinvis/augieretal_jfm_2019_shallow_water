import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import fluidsim as fls
from fractions import Fraction
from base import _rxs_str_func, _k_f, matplotlib_rc, So_var_dict
from paths import paths_sim, exit_if_figure_exists, load_df
from make_fig_struct_order_246 import fig_struct_order, plot_df, set_share_axes


def _label(key="eta", odr=5):
    numerator = r"\langle (c|\delta h|)^{" + f"{int(odr)}" + "} \\rangle"
    exp2 = Fraction(odr, 3)
    exp1 = exp2 - 1
    denom_first_term = (
        (r"(L_f F_f^{1/2})^{" + f"{exp1}" "}") if exp1 != 1 else r"L_f F_f^{1/2}"
    )
    denominator = denom_first_term + r"\epsilon^{" + f"{exp2}" + "}" + "r"
    return f"${numerator} / {denominator}$"


if __name__ == "__main__":
    matplotlib_rc(fontsize=9)

    path_fig = exit_if_figure_exists(__file__)
    fig, ax = plt.subplots(2, 2, figsize=(5, 4.2), sharex=True, sharey=False)

    for row in range(2):
        set_share_axes(ax[row, :], sharey=True)
    set_share_axes(ax, sharex=True)
    df_w = load_df("df_w")
    df_3840 = df_w[df_w["$n$"] == 3840]
    df_7680 = df_w[df_w["$n$"] == 7680]

    kwargs = dict(order=(4, 6), coeff=1, label_func=_label, test=False)
    sns.set_palette("cubehelix", 5)
    plot_df(df_3840, fig, ax[:, 0], **kwargs)
    sns.set_palette("cubehelix", 3)
    plot_df(df_7680, fig, ax[:, 1], **kwargs)
    for ax1 in ax[:, 1].flat:
        ax1.set_ylabel(None)
        ax1.yaxis.set_tick_params(which="both", labelleft=False, labelright=False)
        ax1.yaxis.offsetText.set_visible(False)

    for ax1 in ax[:-1, :].flat:
        ax1.set_xlabel(None)
        ax1.xaxis.offsetText.set_visible(False)

    # Add arrow and text
    xy = (0.03, 20)
    xytext = (0.03, 1.8)
    ax[0, 0].annotate(
        "increasing $c$", xy, xytext, "data", arrowprops={"arrowstyle": "simple"}
    )

    fig.tight_layout()
    fig.savefig(path_fig)
    fig.savefig(path_fig.replace(".png", ".pdf"))
