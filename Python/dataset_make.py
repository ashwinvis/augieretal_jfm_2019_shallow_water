import os
import sys
import zipfile
from pprint import pprint
from pathlib import Path
from stat import S_IREAD, S_IRGRP, S_IROTH

from paths import paths_sim, paths_lap
from table_utils import load_df, sort_reindex


# Conditions on pathlib.Path objects
def only_state_file(f): return f.name.startswith("state")
def not_state_file(f): return not only_state_file(f)
def only_increments(f): return f.name.startswith("increments")
def all_files(f): return True
def small_files(f):
    return (
        not_state_file(f)
        and
        not only_increments(f)
    )


# Functions
def listdir(short_name, paths_dict, condition):
    path = Path(paths_dict.get(short_name))
    return [f for f in path.glob('*') if condition(f)]

def get_size_dir(short_name, paths_dict, condition):
    sizes = [os.path.getsize(f) for f in listdir(short_name, paths_dict, condition)]
    return sum(sizes) / 1024**3

def get_num_state_files(short_name, paths_dict):
    path = Path(paths_dict.get(short_name))
    return len(listdir(short_name, paths_dict, only_state_file))

def get_num_files(short_name, paths_dict):
    return len(listdir(short_name, paths_dict, condition=lambda f:True))


def load(csv, prefix):
    return sort_reindex(load_df(csv), prefix, zerofill=True)

df_w = load("df_w", "W")
df_lap = load("df_lap", "WL")

for df, paths_dict in ((df_w, paths_sim), (df_lap, paths_lap)):
    print(df)
    cum_size = 0
    for condition in (
            # not_state_file,
            only_increments,
            small_files,
            # all_files,
    ):
        print("\n", condition.__name__)
        sizes = df["short name"].apply(get_size_dir, args=[paths_dict, condition])
        cum_size += sizes
        # print(sizes)
        print(sizes.sum() , "GiB")

    print("\nonly state files: size for one file each")
    sizes = df["short name"].apply(get_size_dir, args=[paths_dict, only_state_file])
    num_state_files = df["short name"].apply(get_num_state_files, args=[paths_dict])
    sizes /= num_state_files
    cum_size += sizes
    # print(sizes)
    print(sizes.sum() , "GiB")

    print("cumulative size")
    print(cum_size)

    print("number of files")
    num_files = df["short name"].apply(get_num_files, args=[paths_dict]) - num_state_files
    print(num_files.sum())



def zip_dir(in_dir, out_file, files):
    if os.path.exists(out_file):
        print("File exists", out_file)
        return
    else:
        print("Writing", in_dir, "to", out_file, end="...")
    with zipfile.ZipFile(out_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for f in files:
            # arc = f.name
            arc = f.relative_to(f.parent.parent)
            zipf.write(f, arc)
    print("Done!")


def read_only(filename):
    os.chmod(filename, S_IREAD|S_IRGRP|S_IROTH)


def zip_from_df(run, short_name, paths_dict, output_dir):
    # print("Making zip file for", short_name)
    state_files = listdir(short_name, paths_dict, only_state_file)
    light_files = listdir(short_name, paths_dict, not_state_file)

    files = light_files + sorted(state_files)[-1:]
    pprint(files)
    out_file = output_dir / f"{run}.zip"
    in_dir = paths_dict[short_name]
    zip_dir(in_dir, out_file, files)
    read_only(out_file)
    # with zipfile.ZipFile(out_file, 'r') as zipf:
    #     zipf.printdir()
    # sys.exit(0)



if __name__ == "__main__":
    # Make zip files
    output_dir = (Path.cwd() / "dataset").absolute()
    os.makedirs(output_dir, exist_ok=True)


    for df, paths_dict in (
            (df_w, paths_sim),
            (df_lap, paths_lap),
    ):
        df.apply(
            lambda row: zip_from_df(
                row.name, row["short name"], paths_dict, output_dir
            ),
            axis=1
        )
