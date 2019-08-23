import json
import os
from pprint import pprint
from pathlib import Path
from urllib.request import Request, urlopen, urlretrieve
from zipfile import ZipFile

from fluiddyn.io.query import query_yes_no
from dataset_info import md5


def get_dataset_json(zenodo_id, verbose=False):
    req = Request(
        f"https://zenodo.org/api/records/{zenodo_id}",
        headers={"Accept": "application/json"},
    )
    txt = urlopen(req).read().decode("utf-8")
    data = json.loads(txt)
    if verbose:
        pprint(data)
    return data


def download(f, dest, prompt=True):
    filename, filesize, link = (
        f["filename"],
        f["filesize"],
        f["links"]["download"],
    )
    ans = (
        query_yes_no(
            f"Download {filename} ({filesize / 1024**3:.2f} GiB) to {dest}?"
        )
        if prompt
        else True
    )
    if ans:
        os.makedirs(dest, exist_ok=True)
        zipf = dest / filename
        if zipf.exists():
            print(f"File exists: {zipf}")
        else:
            urlretrieve(link, zipf)


def verify(filename, dest, csum, prompt=True):
    zipf = dest / filename
    ans = query_yes_no("Verify file integrity?", "no") if prompt else False
    if ans:
        if md5(zipf) == csum:
            print("File OK")
        else:
            print("File seems damaged. Try downloading again.")


def unzip(filename, dest, prompt=True):
    zipf = dest / filename
    with ZipFile(zipf) as zipf:
        zipf.printdir()
        ans = query_yes_no(f"Extract {filename} to {dest}") if prompt else True
        if ans:
            zipf.extractall(dest)


if __name__ == "__main__":
    zenodo_id = 3372756
    files = get_dataset_json(zenodo_id)["files"]
    dest = Path(__file__).parent / "dataset"
    for f in files:
        download(f, dest)
        filename = f["filename"]
        verify(filename, dest, f["checksum"])
        unzip(filename, dest)
