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


def mean_accs(data, k='tr_err'):
    """
    Return 2D array of average accuracies and associated SEMs,
    given 2D array of dictionaries containing a given errors key.
    """
    num_rows, num_cols = data.shape
    num_runs = len(data[0, 0][k])
    accuracies = np.zeros((num_rows, num_cols, 2))
    for i, j in itertools.product(xrange(num_rows), xrange(num_cols)):
        accuracies[i, j, 0] = np.mean(100. - data[i, j][k])
        accuracies[i, j, 1] = np.std(100. - data[i, j][k])
    # Return SEMs
    accuracies[:, :, 1] /= np.sqrt(num_runs)
    return accuracies
