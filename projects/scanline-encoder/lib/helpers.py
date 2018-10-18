#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 14 00:02:03 2018

@author: BG
@licence: GNU v3

Loihi relative p1, p2 measurements for five line equations:
    - (0, .224), (.919, 1.)
    - (.165, 0.), (1., .835)
    - (.815, 1,), (.819, 0.)
    - (0, .827), (0.5, 0)
    - (0.152, 1.), (1., .5)
"""

from __future__ import division

import numpy as np


def find_eq(p1, p2, bounds=(28., 28.)):
    """
    Find line equation given two intercepts (as relative positions).
    y = m * x + c.

    Inputs
    ------
    p1 : 2-tuple
        First intercept (x1_rel, y2_rel).
    p2 : 2-tuple
        Second intercept (x2_rel, y2_rel).
    bounds : 2-tuple
        Image bounds (X_max, Y_max)

    Outputs
    -------
    line_eq : 2-tuple
        Line equation: (m, c).
    """
    # Gradient, intercept
    m = (p2[1] - p1[1]) / (p2[0] - p1[0])
    c = p1[1] * bounds[1] - m * p1[0] * bounds[0]
    # Check
#    assert(np.abs(p2[1] * bounds[1] - (m * p2[0] * bounds[0] + c)) < 1e-15)
    return (m, c)
