#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Routine used to find line equation given two points.

Loihi relative p1, p2 measurements for five line equations:
    - (0, .224), (.919, 1.)
    - (.165, 0.), (1., .835)
    - (.815, 1,), (.819, 0.)
    - (0, .827), (0.5, 0)
    - (0.152, 1.), (1., .5)

This file is part of snn-classifier.

Copyright (C) 2018  Brian Gardner <brgardner@hotmail.co.uk>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
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
