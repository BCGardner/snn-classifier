#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created Oct 2018

@author: BG
"""

import numpy as np


def norm_points(p1=(None, None), p2=(None, None), **kwargs):
    """
    Generate two random, associated points with missing values filled in
    according to a normal distribution.
    """
    ps = np.full((2, 2), np.nan)
    for idx, v in enumerate(p1):
        if v is not None:
            ps[0, idx] = v
        else:
            ps[0, idx] = norm(**kwargs)
    for idx, v in enumerate(p2):
        if v is not None:
            ps[1, idx] = v
        else:
            ps[1, idx] = norm(**kwargs)
    return tuple(ps[0]), tuple(ps[1])


def norm(loc=0.5, scale=0.25, bounds=(0., 1.)):
    """
    Generate normally-distributed value within given bounds.
    """
    while True:
        r = np.random.normal(loc, scale)
        if bounds[0] <= r < bounds[1]:
            return r
