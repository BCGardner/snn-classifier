#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 10 19:00:27 2018

@author: BG
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
