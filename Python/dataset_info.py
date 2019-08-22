# coding: utf-8
"""Preview dataset content without extracting."""
import os
import itertools
from pathlib import Path
from zipfile import ZipFile
import hashlib


cwd = Path(__file__).parent / "dataset"
ls = lambda pattern: sorted(cwd.glob(pattern))


def all_files(prefix="W"):
    return itertools.chain(ls(f"{prefix}[0-9].zip"), ls(f"{prefix}[0-9][0-9].zip"))


def md5(filename):
    md5 = hashlib.md5()
    def update(chunk):
        md5.update(chunk)

    with open(filename,'rb') as f:
        chunks = iter(lambda: f.read(8192), b'')
        for chunk in chunks:
             update(chunk)
    return md5.hexdigest()


for prefix in ("W", "WL"):
    for filename in all_files(prefix):
        with ZipFile(filename) as zipf:
            print(
                os.path.basename(zipf.filename),  # Zip file
                os.path.split(zipf.namelist()[0])[0],  # First and only directory
                md5(filename)  # Checksum slow
            )
            # Uncomment to see all contents
            # zipf.printdir()
    print()
