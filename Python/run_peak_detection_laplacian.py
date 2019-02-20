from peak_detection import run
from paths import paths_lap, load_df


df = load_df("df_lap")
df = df[(df["$n$"] == 960) | (df["$n$"] == 2880)]
print(df)
run(
    nh_min=0,
    df=df,
    save_as="dataframes/shock_sep_laplacian_nupt1_by_counting.csv",
    dict_paths=paths_lap,
)
