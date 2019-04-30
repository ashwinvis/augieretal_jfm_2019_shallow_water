# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.1
#   kernelspec:
#     display_name: py-augier_vishnu_lindborg_sw1l
#     language: python
#     name: py-augier_vishnu_lindborg_sw1l
# ---

# %load_ext autoreload
# %aimport fluidsim

# ## Actual simulation

# +
# %matplotlib inline
import os
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from fluidsim.base.output.base import FLUIDDYN_PATH_SCRATCH, FLUIDSIM_PATH
from paths import exit_if_figure_exists
from paths import load_df

path_fig = exit_if_figure_exists(__file__)
tmp = Path(FLUIDDYN_PATH_SCRATCH)
data = Path(FLUIDSIM_PATH)
lacie = Path("/run/media/avmo/lacie/13KTH")

# paths = (tmp / "laplacian").glob("*")
# paths = sorted(paths)
paths = []
# paths.extend(sorted((data/ "laplacian").glob("*")))
# paths.extend(sorted((lacie/ "noise_laplacian").glob("*")))
paths.extend(sorted((lacie/ "laplacian_nupt1").glob("*")))
# paths

# +
# import numpy as np
from fluiddyn.io import stdout_redirected
import fluidsim as fls
from base import epsetstmax, _k_f, load_sim


def init_df(path):
    with stdout_redirected():
        sim = load_sim(str(path))
        #  dico_sp = sim.output.spatial_means.load()
        dico = sim.output.spectra.load2d_mean()
    kmax = dico["kh"].max()

    dealias = sim.params.oper.coef_dealiasing
    ratio = sim.params.preprocess.viscosity_const*np.pi
    kdiss = kmax / ratio
    #  last_file = sorted(path.glob("state_phys*"))[-1].name

    kf = _k_f(sim.params)
    Lf = np.pi / kf
    eps, E, ts, tmax = epsetstmax(path)
    c = sim.params.c2 ** 0.5
    # Fr = (eps / kf) ** (1./3) / c
    Fr = (eps * Lf) ** (1./3) / c
    return {
        "short name": os.path.basename(path),
        "kmax": kmax,
        "kmax_resolved": kmax * dealias,
        "k_d": kdiss,
        "kmax_by_kdiss": ratio,
        "c": c,
        "$n$": sim.params.oper.nx,
        "nu": sim.params.nu_2,
        "visc_const": sim.params.preprocess.viscosity_const,
        "dealias": dealias,
        "tmax": float(tmax),
        "$F_f$": float(Fr),
        "E": float(E),
        "$\epsilon$": float(eps),
    }


# +
import warnings
from scipy import optimize
from paths import shortname_from_path

df = pd.DataFrame({"path": paths})
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=optimize.OptimizeWarning)
    params = df.path.apply(init_df)
df_params = pd.DataFrame(list(params))
df = df.join(df_params)
df = df.sort_values(["c", "$n$"])
short = df["path"].apply(str).apply(shortname_from_path)
df.set_index(short, inplace=True)
df.head()

# ## Load shock separation estimates from cache

df_shocks = load_df("shock_sep_laplacian_nupt1_by_counting_mean")
df_shocks

#  _df = df[(df["$n$"] == 960) | (df["$n$"] == 2880)]
_df = df
try:
    _df["d"] = df_shocks["mean"].values
except ValueError:
    breakpoint()
    raise
df = _df

fig, ax = plt.subplots(1, 2, figsize=(10, 3), dpi=150)

colors = [mpl.colors.to_hex(c) for c in sns.color_palette("cubehelix", len(df))]


kurt_div = []
energy = []
diss = []
for path, color in zip(df.path, colors):
    sim = load_sim(path)
    dset = sim.output.spatial_means.load_dataset()
    idx = int(abs(dset.t  - 10).argmin())

    dset["eps"] = (dset.epsK_tot + dset.epsA_tot)
    eps = dset.eps
    eps.plot(ax=ax[0], c=color, label=f"c={int(sim.params.c2**0.5)}, n={int(sim.params.oper.nx)}")
    ax[0].hlines(eps[idx:].mean(), xmin=dset.t.min(), xmax=dset.t.max(),color=color, label="", linestyle='--')


    dset.kurt_div.plot(ax=ax[1], c=color, label="")
    kurt_mean = dset.kurt_div[idx:].mean()
    E_mean = dset.E[idx:].mean()
    eps_mean = dset.eps[idx:].mean()

    ax[1].hlines(kurt_mean, xmin=dset.t.min(), xmax=dset.t.max(), color=color, linestyle="--",
                 label=f'mean={float(kurt_mean):.2f}')

    kurt_div.append(float(kurt_mean))
    energy.append(float(E_mean))
    diss.append(float(eps_mean))

df["kurt_div"] = kurt_div
df["E"] = energy
df["$\epsilon$"] = diss

    
ax[0].legend()
ax[1].legend()
ax[1].set_yscale("log")
# ax.set_ylim(0.96,1.05)
# ax.set_xlim(0, 50)

# +
# %%capture --no-display

fig, ax = plt.subplots()
for path, color in zip(df.path, colors):
    sim = load_sim(path)
    dset = sim.output.spatial_means.load_dataset()

    dset.E.plot(color="k")

# +
# %matplotlib inline
from base import markers, matplotlib_rc

mark = markers()
# yscale =df["$n$"]
matplotlib_rc(11)
fig, ax = plt.subplots(figsize=(5,3))
# plt.loglog("c", "kurt_div", 'x', data=df)
# sns.scatterplot(df.c, df.kurt_div / yscale, hue=df["$n$"], markers=markers(), legend="full")

# Filter by resolution
# _df = df[(df["$n$"] == 960) | (df["$n$"] == 2880)]
_df = df

for n, grp in _df.groupby("$n$"):
    # yscale = grp["$F_f$"] ** (2/3) * grp["$\epsilon$"]**(1/3) / ( grp["k_d"] ** (4/3) * grp["nu"]) 
    # yscale = grp["d"] ** (4/3) * grp["$\epsilon$"]**(1/3) / (grp["nu"]) 
    yscale = 1 / grp["nu"]
    # yscale = 1
    ax.scatter(
        grp['$F_f$'], grp['kurt_div'] / yscale,
        marker=next(mark),
        # kind="scatter", loglog=True, ax=ax
        # data=grp,
        label=f"$n={int(n)}$"
    )
ax.set_xscale("log")
ax.set_yscale("log")
# ax.set_ylim(1e-1, 20)

ax.legend(loc=4, fontsize=9)

ax.loglog(df["$F_f$"],  35 * df["$F_f$"] ** (2/3), "k--", linewidth=0.5)
ax.text(0.01, 2, "$F_f^{2/3}$")

ax.set_xlabel("$F_f$")
# ax.set_ylabel(r"$F_{\delta_x u} \nu k_d ^{4/3} / (F_f^{2/3} \epsilon^{1/3}) $")
ax.set_ylabel(r"$F_{\delta_x u} \nu $")
# ax.set_ylabel(r"$F_{\delta_x u} \nu / (d^{4/3} \epsilon^{1/3}) $")

# ax.set_ylim(1e-3, 0.3)
# low = (df.kurt_div / yscale).min()
#ax.hlines(50, df["$F_f$"].min(), df["$F_f$"].max(), linestyles="dashed")
#ax.text(30, low, f"{low:.2e}")
# low
fig.tight_layout()
# -

fig.savefig(path_fig)

