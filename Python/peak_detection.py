import os
import gc
from peakutils import indexes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fluiddyn.io import stdout_redirected
import fluidsim as fls
from paths import paths_sim, paths_sim_old, load_df


def detect_shocks(sim, i0=None, i1=None, debug=False, thres=0.5, **kwargs):
    div = sim.state.get_var("div")
    if i0 is not None:
        div1d = div[i0, :]
    elif i1 is not None:
        div1d = div[:, i1]
    else:
        raise ValueError

    div1d_orig = div1d

    # filter
    # height = (div1d.min(), np.median(div1d))
    # peaks, _ = find_peaks(div1d, height=height, **kwargs)
    peaks = indexes(-div1d, thres=thres, min_dist=3)
    if debug:
        # print("Height =", height)
        print(f"Detected {len(peaks)} peaks")
        plt.figure(dpi=150)
        x = sim.oper.x_seq
        ax = plt if "ax" not in kwargs else kwargs["ax"]
        ax.plot(
            x, div1d_orig, "k",
            label=f"c={int(sim.params.c2**0.5)}, n={int(sim.params.oper.nx)}"
        )
        # plt.plot(div1d, 'g--', label="preprocessed div")
        ax.plot(x[peaks], div1d[peaks], "x")
        ax.legend()
    else:
        return peaks


def avg_shock_seperation_1d(sim, i0=None, i1=None):
    peaks = detect_shocks(sim, i0, i1)
    # Compute shock seperation by counting no. of shocks
    return sim.oper.Lx / len(peaks)
    ## Old method which could be unreliable when there are too few shocks
    ## Also, it does not consider periodicity
    # x_peaks = sim.oper.x_seq[peaks]
    # dx_peaks = np.diff(x_peaks)
    # return dx_peaks.mean()

def avg_shock_seperation(sim, num_samples=200, averaged=True, ci=0.95):
    ds = []
    for i in np.linspace(0, sim.oper.nx_seq - 1, num_samples, dtype=int):
        ds0 = avg_shock_seperation_1d(sim, i0=i)
        ds1 = avg_shock_seperation_1d(sim, i1=i)
        ds.extend([ds0, ds1])
    if averaged:
        return np.nanmean(ds), ci * np.nanstd(ds) / np.sqrt(2 * num_samples)
    else:
        return ds


def avg_shock_seperation_from_shortname(
    short_name, save_as="dataframes/shock_sep.csv", dict_paths=paths_sim,
    t_approx=None,
):
    path = dict_paths[short_name]
    with stdout_redirected():
        # Need to load so because type_fft maybe set to use an MPI class
        params, Simul = fls.load_for_restart(path, t_approx, merge_missing_params=True)
        params.output.HAS_TO_SAVE = False
        params.output.ONLINE_PLOT_OK = False
        params.ONLY_COARSE_OPER = False
        params.oper.type_fft = "default"
        sim = Simul(params)
        mean, std = avg_shock_seperation(sim)
        sim.output.close_files()
        del sim
        gc.collect()

    if not os.path.exists(save_as):
        with open(save_as, "a") as f:
            heading = "# short_name,path,mean,std\n"
            f.write(heading)

    with open(save_as, "a") as f:
        result = f"{short_name},{path},{mean},{std}\n"
        f.write(result)
    return mean, std


def run(nh_min, nh_max=10_000, df=None, save_as="dataframes/shock_sep.csv",
        dict_paths=paths_sim, t_approx=None, overwrite=False):
    if df is None:
        df = load_df()
    df = df[(df["$n$"] > nh_min) & (df["$n$"] < nh_max)]
    for i in range(len(df)):
        short = df.iloc[i]["short name"]
        print(dict_paths[short])

    if os.path.exists(save_as) and overwrite:
        os.remove(save_as)

    return (
        df["short name"]
        .apply(
            avg_shock_seperation_from_shortname,
            dict_paths=dict_paths,
            save_as=save_as,
            t_approx=t_approx,
        )
        .apply(pd.Series)
    )


if __name__ == "__main__":
    df = load_df()
    for short in df["short name"]:
        print(short)
        if "nh7680" in short:
            print(f"Skipping {short}")
            continue

        # if short in paths_sim_old:
        #     dict_paths = paths_sim_old
        if short in paths_sim:
            dict_paths = paths_sim
        else:
            print(f"Cannot find {short}")
            continue

        mean, std = avg_shock_seperation_from_shortname(
            short, dict_paths=dict_paths,
            save_as="dataframes/shock_sep_new.csv"
        )
        print(f"mean = {mean}, std = {std}")
