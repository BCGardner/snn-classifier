#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 14 00:02:03 2018

@author: BG
@licence: GNU v3
"""

from __future__ import division

import numpy as np


def find_eq(p1, p2):
    """
    Find line equation given two intercepts.
    y = m * x + c.

    Inputs
    ------
    p1 : 2-tuple
        First intercept (x1, y2).
    p2 : 2-tuple
        Second intercept (x2, y2).

    Outputs
    -------
    line_eq : 2-tuple
        Line equation: (m, c).
    """
    # Gradient, intercept
    m = (p2[1] - p1[1]) / (p2[0] - p1[0])
    c = p1[1] - m * p1[0]
    # Check
    assert(np.abs(p2[1] - (m * p2[0] + c)) < 1e-15)
    return (m, c)
