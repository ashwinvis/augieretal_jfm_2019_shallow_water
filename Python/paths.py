import os
import sys
import re
import gc
from glob import glob
from pathlib import Path
from socket import gethostname
from collections import OrderedDict

import pandas as pd
import numpy as np

from fluiddyn.util.paramcontainer import ParamContainer
from fluiddyn.util import modification_date
from fluiddyn.io.redirect_stdout import stdout_redirected

import fluidsim as fls

from base import _eps, _t_stationary, _k_d, _k_f, _k_diss, epsetstmax


def get_pathbase():
    hostname = gethostname()
    hostname = hostname.lower()
    if hostname.startswith("pelvoux"):
        pathbase = "/media/avmo/lacie/13KTH"
    elif any(map(hostname.startswith, ["legilnx", "nrj1sv", "meige"])):
        pathbase = "$HOME/useful/project/13KTH/DataSW1L_Ashwin/"
    elif hostname.startswith("kthxps"):
        pathbase = "/run/media/avmo/lacie/13KTH/"
        if not os.path.exists(pathbase):
            pathbase = "/scratch/avmo/13KTH/"
    elif os.getenv("SNIC_RESOURCE") == "beskow":
        pathbase = "$SNIC_NOBACKUP/data/"
    elif os.getenv("SNIC_RESOURCE") == "tetralith":
        pathbase = "/proj/kthmech/users/$USER/data/"
    else:
        pathbase = Path(__file__).parent / "dataset"

    pathbase = os.path.abspath(os.path.expandvars(pathbase)).rstrip(os.path.sep)
    if not os.path.exists(pathbase):
        raise ValueError("Path not found " + pathbase)
    elif not len(os.listdir(pathbase)):
        raise IOError("Path seems to be empty" + pathbase)

    return pathbase


def params_from_path(p):
    params_xml_path = os.path.join(p, "params_simul.xml")
    params = ParamContainer(path_file=params_xml_path)
    return params


def keyparams_from_path(p):
    c = re.search("(?<=c=)[0-9]*", p, re.X).group(0)
    nh = re.search("(?<=_)[0-9]*(?=x)", p).group(0)
    try:
        Bu = re.search("(?<=Bu=)[0-9]*(\.[0-9])", p, re.X).group(0)
    except AttributeError:
        Bu = "inf"

    params = params_from_path(p)
    init_field = params.init_fields.type
    if init_field == "noise":
        return init_field, c, nh, Bu, None
    else:
        return init_field, c, nh, Bu, params.preprocess.init_field_const


def shortname_from_path(path):
    init_field, c, nh, Bu, efr = keyparams_from_path(path)
    if efr is None:
        key = "{}_c{}nh{}Bu{}".format(init_field, c, nh, Bu)
    else:
        key = "{}_c{}nh{}Bu{}efr{:.2e}".format(init_field, c, nh, Bu, efr)
    return key


EFR = r"$\frac{<\bf \Omega_0 >}{{(P k_f^2)}^{2/3}}$"
pd_columns = [
    r"$n$",
    r"$c$",
    r"$\nu_8$",
    r"$\nu_2$",
    r"$f$",
    r"$\epsilon$",
    r"$\frac{k_{diss}}{k_f}$",
    r"$F_f$",
    r"$Ro_f$",
    "$Re$",
    "$ReF_f^{2/3}$",
    r"$Bu$",
    # '$\min h$', r'$\frac{\max |\bf u|}{c}$',
    EFR,
    r"$E$",
    "$t_{stat}$",
    r"$t_{\max}$",
    "short name",
]


def pandas_from_path(p, key, as_df=False):
    init_field, c, nh, Bu, init_field_const = keyparams_from_path(p)
    params_xml_path = os.path.join(p, "params_simul.xml")
    params = ParamContainer(path_file=params_xml_path)
    # sim = fls.load_sim_for_plot(p, merge_missing_params=True)

    c = int(c)
    kf = _k_f(params)
    Lf = np.pi / kf
    kd_kf = _k_diss(params) / kf
    # ts = _t_stationary(path=p)
    # eps = _eps(t_start=ts, path=p)
    eps, E, ts, tmax = epsetstmax(p)
    efr = params.preprocess.init_field_const
    if params.nu_2 > 0:
        Re = eps ** (1 / 3) * Lf ** (4 / 3) / params.nu_2
    else:
        Re = np.nan  # defined differently

    Fr = (eps * Lf) ** (1.0 / 3) / c
    try:
        Ro = (eps / Lf ** 2) ** (1.0 / 3) / params.f
    except ZeroDivisionError:
        Ro = np.inf
    minh = 0
    maxuc = 0
    # del sim
    gc.collect()
    data = [
        nh,
        c,
        params.nu_8,
        params.nu_2,
        params.f,
        eps,
        kd_kf,
        Fr,
        Ro,
        Re,
        Re * Fr ** (2 / 3),
        Bu,
        # minh, maxuc,
        efr,
        E,
        ts,
        tmax,
        key,
    ]
    if as_df:
        return pd.DataFrame(data, columns=pd_columns)
    else:
        return pd.Series(data, index=pd_columns)


def make_paths_dict(glob_pattern="SW1L*"):
    """Returns an ordered dictionary of paths, containing keys as the
    independent simulation parameters

    """
    paths = glob(glob_pattern)
    paths_dict = OrderedDict()
    for p in sorted(paths):
        key = shortname_from_path(p)
        paths_dict[key] = p

    return paths_dict


def specific_paths_dict(patterns=("noise/SW1L*NOISE2*", "vortex_grid/SW1L*VG*")):
    paths_dict = {}
    pathbase = get_pathbase()

    for pattern in patterns:
        glob_pattern = os.path.join(pathbase, pattern)
        paths_dict.update(make_paths_dict(glob_pattern))

    return paths_dict


paths_sim = specific_paths_dict()
paths_sim_old = specific_paths_dict(["noise/SW1L*NOISE_*"])
path_pyfig = os.path.join(os.path.dirname(__file__), "../Pyfig_final/")
paths_lap = specific_paths_dict(["laplacian_nupt1/*"])
if "noise_c40nh7680Buinf" in paths_sim:
    del paths_sim["noise_c40nh7680Buinf"]  # dissipation != 1

if not os.path.exists(path_pyfig):
    os.mkdir(path_pyfig)


def exit_if_figure_exists(scriptname, extension=".eps", override_exit=False):
    scriptname = os.path.basename(scriptname)
    figname = os.path.splitext(scriptname)[0].lstrip("make_") + extension
    figpath = os.path.join(path_pyfig, figname)

    if (
        len(sys.argv) > 1
        and "remake".startswith(sys.argv[1])
        and os.path.exists(figpath)
    ):
        os.remove(figpath)

    if os.path.exists(figpath) and not override_exit:
        if (
            # figure is outdated
            modification_date(scriptname)
            < modification_date(figpath)
        ):
            print(
                "Figure {} already made. {} exiting...".format(
                    figname, scriptname
                )
            )
            sys.exit(0)

    print("Making Figure {}.. ".format(figname))
    return figpath


def load_df(name="df_w"):
    return pd.read_csv(f"dataframes/{name}.csv", index_col=0, comment="#")
