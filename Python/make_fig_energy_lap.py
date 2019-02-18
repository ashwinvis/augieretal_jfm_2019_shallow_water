import matplotlib.pyplot as plt
from base import matplotlib_rc
from paths import load_df, exit_if_figure_exists
from base_fig_energy import plot_energy

matplotlib_rc(fontsize=10)
path_fig = exit_if_figure_exists(__file__, '.png')
df_w = load_df("df_lap")
fig, ax = plt.subplots(1, 2, figsize=(6.5,3))
plot_energy(df_w, fig, ax, N=[960, 2880], C=[10])
fig.tight_layout()
fig.savefig(path_fig)
fig.savefig(path_fig.replace(".png", ".pdf"))