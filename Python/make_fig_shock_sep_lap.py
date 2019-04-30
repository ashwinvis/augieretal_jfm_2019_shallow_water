
# Load cached data

import pandas as pd
import numpy as np
import fluidsim as fs
import matplotlib.pyplot as plt
from base import matplotlib_rc, markers

from paths import paths_lap as paths_sim
from paths import load_df, exit_if_figure_exists

from base import _k_f

path_fig = exit_if_figure_exists(__file__)
df = load_df("df_lap")
df.set_index("short name", inplace=True, drop=False)

_dfs = []

for t in range(10, 26):
    _df = pd.read_csv(
        # "dataframes/shock_sep_laplacian_nupt1.csv",
        f"dataframes/shock_sep_laplacian_nupt1_by_counting_t{t}.csv",
        comment="#"
    )
    _dfs.append(_df.set_index("short_name"))

_df_concat = pd.concat(_dfs, keys=range(10, 26))
df_shock_sep = _df_concat.groupby(level=1).mean()
df_shock_sep
df_shock_sep.to_csv("dataframes/shock_sep_laplacian_nupt1_by_counting_mean.csv")



sim = fs.load_sim_for_plot(paths_sim[df_shock_sep.iloc[0].name])
Lf = np.pi / _k_f(sim.params)

df["shock separation"] = df_shock_sep["mean"] / Lf
df.head()

df["shock separation"] = df_shock_sep["mean"] / Lf
df.head()



matplotlib_rc(11)
fig, ax = plt.subplots(figsize=(5,3))
mark = markers()
for n, grp in df.groupby("$n$"):
    ax.scatter(
        r'$F_f$', 'shock separation',
        marker=next(mark),
        #kind="scatter", loglog=True, ax=ax
        data=grp, label=f"$n={int(n)}$")


uniq_F_f = np.array(sorted(set(df[r'$F_f$'])))
ax.loglog(uniq_F_f, 2.5 * np.pi * uniq_F_f ** (1./2), 'k:', label="")

# import seaborn as sns
# sns.regplot(
#     '$F_f$', 'shock separation',
#     data=df,
#     scatter=False, ax=ax, truncate=True,
#     color="k", line_kws=dict(linestyle="dashed"),
#     # order=2,
#     logx=True,
#     # robust=True,  
# )
ax.set_xlim(df["$F_f$"].min() * 0.9)
ax.text(2e-2, 1.3, r"$F_f ^ {1/2}$")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel('$F_f$')
ax.set_ylabel("$d / L_f$")
ax.legend(fontsize=9)
fig.tight_layout()
#  plt.show()

fig.savefig(path_fig)
