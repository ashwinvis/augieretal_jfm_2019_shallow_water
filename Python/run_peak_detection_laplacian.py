import numpy as np
from peak_detection import run
from paths import paths_lap, load_df


df = load_df("df_lap")
df = df[np.logical_not((df["$n$"] == 960) | (df["$n$"] == 2880))]
print(df)
for t in range(24,26):
    print("Time:", t)
    run(
        nh_min=0,
        df=df,
        save_as=f"dataframes/shock_sep_laplacian_nupt1_by_counting_t{t}.csv",
        dict_paths=paths_lap,
        t_approx=t
    )
