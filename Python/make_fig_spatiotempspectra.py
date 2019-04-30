import shutil
from paths import exit_if_figure_exists


if __name__ == "__main__":
    path_fig = exit_if_figure_exists(__file__)
    extension = ".eps" if path_fig.endswith(".eps") else ".pdf"
    path_orig = '../Old/Figs/fig_spatiotempspectra_c20_Nh3840' + extension
    shutil.copyfile(path_orig, path_fig)
