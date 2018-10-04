import os
from peakutils import indexes
import matplotlib.pyplot as plt
import numpy as np
import fluidsim as fls
from paths import paths_sim, paths_sim_old


def detect_shocks(sim, i0=None, i1=None, debug=False, **kwargs):
    div = sim.state.get_var("div")
    if i0 is not None:
        div1d = div[i0, :]
    elif i1 is not None:
        div1d = div[:, i1]
    else:
        raise ValueError

    div1d_orig = div1d
        
    # filter
    height = (div1d.min(), np.median(div1d))
    # peaks, _ = find_peaks(div1d, height=height, **kwargs)
    peaks = indexes(-div1d, thres=0.5, min_dist=3)
    if debug:
        print("Height =",height)
        print(f"Detected {len(peaks)} peaks")
        plt.figure(dpi=150)
        x = sim.oper.x_seq
        plt.plot(x, div1d_orig, 'k', label="original div")
        # plt.plot(div1d, 'g--', label="preprocessed div")    
        plt.plot(x[peaks], div1d[peaks], "x")
        plt.xlabel("$x$")
        plt.ylabel(r"$\nabla.\bf u$")
        plt.legend()
    else:
        return peaks


def avg_shock_seperation_1d(sim, i0=None, i1=None):
    peaks = detect_shocks(sim, i0, i1)
    x_peaks = sim.oper.x_seq[peaks]
    dx_peaks = x_peaks[1:] - x_peaks[:-1]
    return dx_peaks.mean()


def avg_shock_seperation(sim, num_samples=200, averaged=True, ci=0.95):
    ds = []
    for i in np.linspace(0, sim.oper.nx_seq-1, num_samples, dtype=int):
        ds0 = avg_shock_seperation_1d(sim, i0=i)
        ds1 = avg_shock_seperation_1d(sim, i1=i)
        ds.extend([ds0, ds1])
    if averaged:
        return np.nanmean(ds), ci * np.nanstd(ds) / np.sqrt(2 * num_samples)
    else:
        return ds


def avg_shock_seperation_from_shortname(short_name, save_as="shock_sep.csv",
                                        dict_paths=paths_sim):
    path = dict_paths[short_name]
    sim = fls.load_state_phys_file(path, merge_missing_params=True)
    mean, std = avg_shock_seperation(sim)
    if not os.path.exists(save_as):
        with open(save_as, "a") as f:
            result = "# short_name,\t path,\t mean,\t std\n"
            f.write(result)
        
    with open(save_as, "a") as f:
        result = f"{short_name},\t {path},\t {mean}, \t {std}\n"
        f.write(result)
    return mean, std

def run(nh_min, nh_max=10_000):
    df_all = load_df()
    df = df[df["$n$" > nh_min] and df["$n$" < nh_max]]
    for i in range(len(df)):
        short = df.iloc[i]["short name"]
        print(paths_sim[short])
    
    return (
        df["short name"].apply(
            avg_shock_seperation_from_shortname).apply(
                pd.Series)
    )


if __name__ == "__main__":
    for dict_paths in (paths_sim, paths_sim_old):
        for short in dict_paths:
            print(short)
            avg_shock_seperation_from_shortname(short, dict_paths=dict_paths)
            print(f"mean = {mean}, std = {std}")
