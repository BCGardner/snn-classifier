#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on June 2018

@author: BG

Helper functions.
"""

import gzip
import cPickle as pkl
import itertools

import numpy as np


def save_data(data, filename):
    """
    Save data as gzipped pickle file.
    """
    with gzip.open(filename, 'wb') as f:
        pkl.dump(data, f)


def load_data(path):
    """
    Load data saved as gzipped pickle file.
    """
    with gzip.open(path, 'rb') as f:
        data = pkl.load(f)
    return data


def mean_errs(data):
    """
    Return 2D array of average training errors, given 2D array of dictionaries
    containing 'tr_err' key.
    """
    num_rows, num_cols = data.shape
    tr_errs_av = np.zeros((num_rows, num_cols))
    for i, j in itertools.product(xrange(num_rows), xrange(num_cols)):
        tr_errs_av[i, j] = np.mean(data[i, j]['tr_err'])
    return tr_errs_av
