import numpy as np
# from matplotlib import ticker
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from base import set_figsize, markers


L_f = 50. / 6  # hardcoded Lh / nk_f


def filter_df_by(df, keys, val):
    groups = df.groupby(keys)
    return groups.filter(
        lambda x:all(x[keys[0]] == val),
        dropna=True
    )


def model_func(c_eps, Cn):
    return Cn * (L_f * c_eps) ** 0.5


def get_cn(c_eps, E):
    c_eps = c_eps.values
    E = E.values
    popt, pcov = curve_fit(model_func, c_eps, E)
    return popt[0]


def model_func2(n, C, alpha):
    return C * (n ** alpha)


def get_alpha(n, E):
    n = n.values
    E = E.values
    popt, pcov = curve_fit(model_func2, n, E)
    return popt


def plot_energy(
    df_main, fig=None, ax=None,
    N=[960, 1920, 3840, 7680],
    C=[10, 20, 40, 100, 400, 700],
):
    if fig is None and ax is None:
        set_figsize(6.5, 3)
        fig, ax = plt.subplots(1, 2)

    df_n = {}
    for n in N:
        df_n[n] = filter_df_by(df_main, ['$n$'], n)

    E_eps = '$E/\sqrt{\epsilon}$'
    mark = markers()
    for n, df in df_n.items():
        df = df.assign(
            **{E_eps: df['$E$'] / df['$\epsilon$'] ** 0.5}
        )
        c_eps = df['$c$'] * df['$\epsilon$']
        Cn = get_cn(c_eps, df['$E$'])
        E_fit = model_func(c_eps.values, Cn) /  df['$\epsilon$'].values ** 0.5
        print("n =", n, "Cn =", Cn)
        # ax[0].plot(
        ax[0].loglog(
                df['$c$'].values, E_fit, ':', linewidth=1)

        ax[0].scatter(
            '$c$', E_eps, marker=next(mark),
            data=df,
            label='$n={}$'.format(n),                       
            # logx=True, logy=True,
        )
    ax[0].set_ylabel(E_eps)
    ax[0].set_xlabel('$c$')

    # ax[0].set_xticks(C)


    # Subplot on right : Energy vs n
    df_c = {}
    for c in C:
        df_c[c] = filter_df_by(df_main, ['$c$'], c)

    E_eLc = '$E/\sqrt{\epsilon L_f c}$'
    mark = markers()
    for c, df in df_c.items():
        df = df.assign(
            **{E_eLc: df['$E$'] / (df['$\epsilon$'] * L_f * df['$c$']) ** 0.5}
        )
        n = df['$n$']
        # alpha = get_alpha(n, df['$E$'])
        # E_fit = model_func2(n, alpha) / (df['$\epsilon$'] * L_f * df['$c$']) ** 0.5
        alpha = get_alpha(n, df[E_eLc])
        print("c =", c, "alpha =", alpha)
        E_fit = model_func2(n, *alpha)


        ax[1].scatter(
            '$n$', E_eLc, data=df,  marker=next(mark),
            label=r"$c={}; \alpha={:.2f}$".format(c, alpha[1]),
            # logx=True, logy=True
        )
        ax[1].loglog(
            n, E_fit, ':',
            label="",
            linewidth=1
        )
        # print('c =', c)
        # print(df[E_eLc])
    ax[1].set_ylabel(E_eLc)
    ax[1].set_xlabel('$n$')
    ax[1].set_xlim([None, 1e4])
    # ax[1].set_xticks(N)

    for a in ax.ravel():
         a.legend(fontsize=7)

    fig.tight_layout()
    return fig, ax