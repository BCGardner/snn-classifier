#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Load image data and cast to array.

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

import numpy as np


def load_data(fname, width=16, height=16):
    """
    Read binary-valued digits with shape (width, height) and convert to
    list of 2D-arrays.
    """
    num_pixels = width * height

    # Read data as X, Y (tr_data, tr_labels)
    data = np.genfromtxt(fname)
    tr_data = data[:, :num_pixels].astype(int)
    Y = data[:, num_pixels:].astype(int)
    # Convert to list of 2D image arrays
    X = []
    for tr_case in tr_data:
        x = tr_case.reshape((width, height))
        X.append(x)
    return X, Y
