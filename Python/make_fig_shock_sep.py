
# Load cached data

import pandas as pd
import numpy as np
from paths import load_df
import matplotlib.pyplot as plt
import fluidsim as fs
from base import matplotlib_rc, markers, _k_f
from paths import paths_sim, exit_if_figure_exists


path_fig = exit_if_figure_exists(__file__)
df = load_df()
df.set_index("short name", inplace=True, drop=False)
df_shock_sep = pd.read_csv("dataframes/shock_sep.csv", comment="#")
df_shock_sep.set_index("short_name", inplace=True)
df.head()


sim = fs.load_sim_for_plot(paths_sim[df_shock_sep.iloc[0].name], merge_missing_params=True)
Lf = np.pi / _k_f(sim.params)

df["shock separation"] = df_shock_sep["mean"] / Lf
df.head()

matplotlib_rc(11)
fig, ax = plt.subplots(figsize=(5, 3))
mark = markers()
for n, grp in df.groupby("$n$"):
    ax.scatter(
        r'$F_f$', 'shock separation',
        marker=next(mark),
        # kind="scatter", loglog=True, ax=ax
        data=grp, label=f"$n={n}$")


uniq_F_f = np.array(sorted(set(df[r'$F_f$'])))
ax.loglog(uniq_F_f, 1.5 * np.pi * uniq_F_f ** 0.5, 'k:', label="")

# import seaborn as sns
# sns.regplot(
#     '$F_f$', 'shock separation',
#     data=df,
#     scatter=False, ax=ax, order=2
# )
ax.text(1e-2, 0.6, r"$F_f ^ {1/2}$")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel('$F_f$')
ax.set_ylabel("$d / L_f$")
ax.legend(fontsize=9)
fig.tight_layout()
#  plt.show()

fig.savefig(path_fig)
