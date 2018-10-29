#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created Oct 2018

@author: BG
"""

from itertools import cycle, islice

import numpy as np

import lib.helpers as hp

# Number of unique pair interceptions on a grid (side A <-> side B)
num_unique_pairs = 6
# Line boundary conditions on a grid
eqs_grid = [((0., None), (None, 1.)),  # Left <-> Bottom
            ((0., None), (1., None)),  # Left <-> Right
            ((0., None), (None, 0.)),  # Left <-> Top
            ((None, 1.), (1., None)),  # Bottom <-> Right
            ((None, 1.), (None, 0.)),  # Bottom <-> Top
            ((1., None), (None, 0.))]  # Right <-> Top


def norm(loc=0.5, scale=0.25, bounds=(0., 1.), rng=np.random.RandomState()):
    """
    Generate normally-distributed value within given bounds.
    """
    while True:
        r = rng.normal(loc, scale)
        if bounds[0] <= r < bounds[1]:
            return r


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


def generate_eqs_ctr(num_eqs, bounds=(28., 28.), rng=np.random.RandomState(),
                     **kwargs):
    """
    Generate a set of line equations, according to intercepts randomly
    positioned about the grid centre, and random line orientations.
    """
    line_eqs = []
    for idx in xrange(num_eqs):
        # Random point intercepts, following normal distribution located at
        # grid centre
        x = norm(rng=rng, **kwargs)
        y = norm(rng=rng, **kwargs)
        # Random line orientations following uniform distr.
        theta = rng.uniform(0., np.pi)
        m = np.tan(theta)
        c = y * bounds[1] - m * x * bounds[0]
        # Line eq.
        line_eqs.append((m, c))
    return line_eqs


def generate_eqs(num_eqs, seq=[4, 1, 2, 3, 0, 5], random_sides=False,
                 distr=norm_points, bounds=(28., 28.), **kwargs):
    """
    Generate a set of line equations intercepting two sides of a grid
    (following a given, repeating sequence if random is False), with missing
    values filled in according to a given distribution.
    """
    if random_sides:
        idxs = np.random.randint(0, num_unique_pairs, num_eqs)
    else:
        idxs = np.array(list(islice(cycle(seq), num_eqs)))
    line_eqs = []
    for idx in idxs:
        points = distr(*eqs_grid[idx], **kwargs)
        eq = hp.find_eq(*points, bounds=bounds)
        line_eqs.append(eq)
    return line_eqs
