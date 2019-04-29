#!/usr/bin/env python
import runpy
import os
import sys
import shutil
from pathlib import Path


def run(path):
    """run

    Parameters
    ----------
    path : Python script path

    """
    #  runpy.run_path(path)
    os.system(f"{sys.executable} {path}")


fig_num = 1
for script in (
    "make_fig_Emean_time.py",
    "make_fig_energy_w.py",
    "make_fig_flux_struct_combined.py",
    #  "make_fig_spatiotempspectra.py",
    "make_fig_phys_fields_wave.py",
        #  "make_fig_shock_sep.py", TODO:
    "make_fig_struct_order_246.py",
    "make_fig_ratio_strfct.py",
    "make_fig_flatness.py",
    "make_fig_spectra.py",
    "make_fig_diss_spectra.py",
    "make_fig_energy_lap.py",
        #  "make_fig_shock_sep_lap.py", TODO:
        #  "make_fig_flatness_div_norm_nu.py", TODO:
    "make_fig_phys_fields_wave_lap.py",
):
    print(f"Running {script}")
    pyfig = Path.cwd() / ".." / "Pyfig_final"
    figname, _ = os.path.splitext(script.lstrip("make_"))
    output = os.path.abspath(
         pyfig / (figname + ".eps")
    )
    run(f"./{script}")
    dest = pyfig / f"fig{fig_num}.eps"
    if not dest.exists():
        shutil.copyfile(output, dest)
    fig_num += 1
