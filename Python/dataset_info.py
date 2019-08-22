# coding: utf-8
"""Preview dataset content without extracting."""
import os
import itertools
from pathlib import Path
from zipfile import ZipFile


cwd = Path(__file__).parent / "dataset"
ls = lambda pattern: sorted(cwd.glob(pattern))


def all_files(prefix="W"):
    return itertools.chain(ls(f"{prefix}[0-9].zip"), ls(f"{prefix}[0-9][0-9].zip"))


for prefix in ("W", "WL"):
    for zipf in all_files(prefix):
        with ZipFile(zipf) as zipf:
            print(
                os.path.basename(zipf.filename),  # Zip file
                os.path.split(zipf.namelist()[0])[0]  # First and only directory
            )
            # Uncomment to see all contents
            # zipf.printdir()
    print()
