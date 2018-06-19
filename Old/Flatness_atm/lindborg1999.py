

from __future__ import print_function

import os

import numpy as np

path_flatness = os.path.split(os.path.realpath(__file__))[0]


rmin = 2.03   # (km)

rmax = 2200.  # (km)

Nr = np.round(rmax/rmin)

r = rmin*np.arange(1, Nr)

def flatness_from_file(name_file):

    path_file = path_flatness+'/'+name_file

    if not os.path.exists(path_file):
        raise ValueError('file does not exist? path_file :\n'+path_file)

    file = open(path_file, 'r')
    lines = file.readlines()
    file.close()

    nr = len(lines)

    r = np.zeros(nr)
    F = np.zeros(nr)

    for ir in xrange(nr):
        line = lines[ir]
        words = line.split()
        r[ir] = float(words[0])
        F[ir] = float(words[1])

    return r, F

def flatness_from_files(r):
    """Load the data..."""
    name_file = 'r_FT'
    rT, FT = flatness_from_file(name_file)

    name_file = 'r_FL'
    rL, FL = flatness_from_file(name_file)

    FT = np.interp(r, rT, FT)
    FL = np.interp(r, rL, FL)

    return FT, FL


FT, FL = flatness_from_files(r)


def logspace(min, max, nb=100):
    return np.logspace(np.log10(min), np.log10(max), nb)



