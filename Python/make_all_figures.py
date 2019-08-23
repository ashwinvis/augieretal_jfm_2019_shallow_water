#!/usr/bin/env python
import runpy
import os

# import subprocess
import asyncio
from asyncio import subprocess
import sys
import shutil
from pathlib import Path
from fluiddyn.util import modification_date


BLOCK = False


async def run(path):
    """run

    Parameters
    ----------
    path : Python script path

    """
    #  runpy.run_path(path)
    proc = await asyncio.create_subprocess_shell(
        f"{sys.executable} {path}", stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if BLOCK:
        stdout, stderr = await proc.communicate()
        if stdout or stderr:
            print(f"[{path!r} exited with {proc.returncode}]")
        else:
            await asyncio.sleep(1)


async def copy(pyfig, figname, fig_num):
    output = (pyfig / (figname + ".eps")).absolute()
    dest = pyfig / f"fig{fig_num}.eps"
    if output.exists():
        if not dest.exists() or (
            dest.exists()
            and (modification_date(output) < modification_date(dest))
        ):
            return shutil.copyfile(output, dest)

    await asyncio.sleep(1)


async def main():
    fig_num = 0
    pyfig = Path.cwd() / ".." / "Pyfig_final"

    for script in (
        "make_fig_Emean_time.py",  # 1
        "make_fig_energy_w.py",
        "make_fig_flux_struct_combined.py",
        "make_fig_spatiotempspectra.py",
        "make_fig_phys_fields_wave.py",  # 5
        "make_fig_shock_sep.py",
        "make_fig_struct_order_246.py",
        "make_fig_ratio_strfct.py",
        "make_fig_flatness.py",
        "make_fig_spectra_submitted.py",  # 10
        "make_fig_diss_spectra.py",
        "make_fig_energy_lap.py",
        "make_fig_shock_sep_lap.py",
        "make_fig_flatness_div_norm_nu.py",
        "make_fig_phys_fields_wave_lap.py",  # 15
    ):
        figname, _ = os.path.splitext(script.lstrip("make_"))
        print(f"Running {script}")
        fig_num += 1
        await asyncio.gather(run(f"./{script}"), copy(pyfig, figname, fig_num))


if __name__ == "__main__":
    # Requires python>=3.7
    asyncio.run(main())
