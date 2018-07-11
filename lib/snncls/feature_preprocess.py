#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mar 2017

@author: BG
"""

from __future__ import division

import numpy as np


def rescale_unitrange(x):
    """
    Rescale data to unit range [0, 1].

    Inputs
    ------
    x : array
        Input data: <examples> by <features>.

    Output
    ------
    x' : array
        Rescaled data: <examples> by <features>.
    """
    x_min, x_max = np.min(x, 0), np.max(x, 0)
    x_range = x_max - x_min
    return (x - x_min) / x_range


def standardize(x):
    """
    Adjusts each feature variable to have zero-mean and unit-variance.

    Inputs
    ------
    x : array
        Input data: <examples> by <features>.

    Output
    ------
    x' : array
        Standardized data: <examples> by <features>.
    """
    return (x - np.mean(x, 0)) / np.std(x, 0)


def onehot_encode(y):
    """
    Transform array of class labels into sparse one-hot encoded matrix.
    """
    m = len(y)  # No. examples
    labels = np.unique(y)  # Fill in matrix in this order
    num_labels = len(labels)

    # Sparse matrix
    y_ = np.zeros((m, num_labels))
    for i in xrange(m):
        index = np.where(labels == y[i])[0][0]
        y_[i, index] = 1.0
    return y_


def receptive_fields(x, neurons_f=12, beta=1.5):
    """
    Convert input vectors into spike latencies based on Gaussian receptive
    fields.

    Inputs
    ------
    x : array
        Input data: <num_samples> by <num_features>.
    neurons_f : int
        Neurons encoding each feature.
    beta : float
        Width of Gaussian receptive field.

    Output
    ------
    spike_trains : array
        List of arrays: <num_samples> by <num_encoding_neurons>.
    """
    m = x.shape[0]
    n = x.shape[1]
    # Center receptive fields
    x_min, x_max = np.min(x, 0, keepdims=True).T, np.max(x, 0, keepdims=True).T
    indices = np.arange(1, neurons_f+1)
    centers = x_min + (2*indices - 3) / 2. * (x_max - x_min) / (neurons_f - 2)
    sigmas = 1. / beta * (x_max - x_min) / (neurons_f - 2)
    # Intensity of responses
    spikes = np.empty((m, neurons_f * n))
    for f in xrange(n):
        spikes[:, neurons_f*f:neurons_f*(f + 1)] = gaussian(x[:, [f]],
                                                            centers[f, :],
                                                            sigmas[f])
    # Convert to spike latencies
    spikes = 10. * (1 - spikes)
    spikes[spikes > 9.] = np.inf
    return spikes


def predet_psp(data_inputs, dt, duration, cell_params):
    """
    Predetermine psp's due to feature vectors for duration of each sim.
    """
    t = np.arange(0.0, duration, dt)
    return [psp(t, spikes, cell_params) for spikes in data_inputs]


def gaussian(x, mu, sigma):
    """
    Gaussian function for firing intensity.
    """
    return np.exp(-0.5 * ((x - mu) / sigma)**2)


def psp(t, spikes, cell_params):
    """PSP at time(s) t due to  array of single spikes, given parameter set"""
    s = t - spikes
    u = s > 0.
    values = np.zeros(s.shape)
    values[u] = cell_params['psp_coeff'] * \
        (np.exp(-s[u] / cell_params['tau_m']) -
         np.exp(-s[u] / cell_params['tau_s']))
    return values


def fold_data(data, n_folds):
    """
    Partition data of form [(X1, Y1), ...], y in {0, 1, ...} into n_folds,
    maintaining relative proportion of classes in each fold. 5 standard.
    """
    # Separate data by class
    if np.isscalar(data[0][1]):
        class_ids = [d_case[1] for d_case in data]
    else:
        class_ids = [np.argmax(d_case[1]) for d_case in data]
    n_classes = len(np.unique(class_ids))
    data_c = [[] for i in xrange(n_classes)]
    for i in xrange(len(data)):
        data_c[class_ids[i]].append(data[i])

    # Partition data into n_folds
    data_f = [[] for k in xrange(n_folds)]
    for d_cases in data_c:
        n_samples = len(d_cases) / n_folds  # No. class samples per fold
        assert n_samples % 1 == 0  # Ensure round number
        n_samples = int(n_samples)
        d_cases_f = [d_cases[i:i+n_samples] for i in xrange(0, len(d_cases),
                     n_samples)]  # Disjoint sets of class samples
        for k in xrange(n_folds):
            data_f[k] += d_cases_f[k]

    return data_f
