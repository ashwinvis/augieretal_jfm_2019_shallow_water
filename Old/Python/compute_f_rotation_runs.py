#!/usr/bin/env python
#coding=utf8

from __future__ import print_function

import numpy as np

import baseSW1lw


Bu = np.array([0.5, 1., 2., 4.])
print('Bu =', Bu)

c = 20.

kf = baseSW1lw.Kf


f = c*kf/np.sqrt(Bu)

print('f =', f)


kd = f/c
print('kd/kf =', kd/kf)
