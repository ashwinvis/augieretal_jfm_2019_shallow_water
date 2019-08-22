from make_archive import *
from paths import paths_sim as paths_dict


# Make zip files
output_dir = (Path.cwd() / "zenodo_embargo").absolute()
os.makedirs(output_dir, exist_ok=True)

for df in map(load_df, ["df_wv", "df_wr", "df_wvr"]):
    df.apply(
        lambda row: zip_from_df(
            row.name, row["short name"], paths_dict,
                                output_dir),
        axis=1
    )

