import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from base import set_figsize, matplotlib_rc


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


def plot_energy(
    df_main, fig=None, ax=None,
    N=[960, 1920, 3840, 7680],
    C=[10, 20, 40, 100, 400, 700],
):
    if fig is None and ax is None:
        matplotlib_rc()
        set_figsize(6.5, 3)
        fig, ax = plt.subplots(1, 2)

    df_n = {}
    for n in N:
        df_n[n] = filter_df_by(df_main, ['$n$'], n)

    E_eps = r'$E/\sqrt{\epsilon}$'
    for n, df in df_n.items():
        df = df.assign(**{
            E_eps: df['$E$'] / df['$\epsilon$'] ** 0.5})

        c_eps = df['$c$'] * df['$\epsilon$']
        Cn = get_cn(c_eps, df['$E$'])

        E_eps_fit = model_func(c_eps.values, Cn) /  df['$\epsilon$'].values ** 0.5
        # ax[0].plot(
        ax[0].loglog(
                df['$c$'].values, E_eps_fit, 'k:', linewidth=1)
        df.plot(
            '$c$', E_eps, ax=ax[0], style='x-',
            label='$n={}, C_n={:.2f}$'.format(n, Cn),                       
            logx=True, logy=True,
        )
    ax[0].set_ylabel(E_eps)
    # ax[0].set_xticks(C)


    df_c = {}
    for c in C:
        df_c[c] = filter_df_by(df_main, ['$c$'], c)

    E_eps = r'$E/\sqrt{\epsilon L_f c}$'
    for c, df in df_c.items():
        df = df.assign(**{
            E_eps: df['$E$'] / (df['$\epsilon$'] * L_f * df['$c$']) ** 0.5})
        df.plot('$n$', E_eps, ax=ax[1], style='x-', label='$c={}$'.format(c))
        print('c =', c)
        print(df[E_eps])
    ax[1].set_ylabel(E_eps)
    ax[1].set_xticks(N)


    for a in ax.ravel():
        a.legend()

    fig.tight_layout()
    return fig, ax