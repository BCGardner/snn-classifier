#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Aug 2018

@author: BG

Preprocessing features for spike-based learning.
"""

from __future__ import division

import numpy as np

from snncls.parameters import ParamSet
from snncls.transform import ReceptiveFields


def transform_data(X, y, param, receptor=None, num_classes=None):
    """
    Transform a data set into predetermined PSPs, spike latencies, one-hot
    encoded class labels.

    Inputs
    ------
    X : array, shape (num_samples, features)
        Unprocessed input data.
    y : array, shape (num_samples,)
        Unprocessed labels.
    param : object
        Top level parameter container, using pattern and cell.
    receptor : object, optional
        Model for transforming real-valued features into spike latencies.
    num_classes : int, optional
        Set one-hot encoded labels to this number of classes.

    Output
    ------
    data : list
        Processed data as list of 2-tuples, [(X1, y1), (X2, y2), ...].
        Each X is itself a 2-tuple, containing (psps, spike_trains). y's are
        one-hot encoded.
    """
    if receptor is None:
        # Prepare feature preprocessor fitted to this specific data
        receptor = ReceptiveFields(param.pattern['neurons_f'],
                                   param.pattern['beta'])
        receptor.fit(X)
    # Transform data
    latencies = receptor.transform(X)
    y_enc = onehot_encode(y, num_classes=num_classes)
    # Predetermine PSPs, return as list of 2-tuples containing (psps, spikes)
    inputs_tr = latencies_psps(latencies, param.dt,
                               param.pattern['duration'],
                               return_trains=True, **param.cell)
    return zip(inputs_tr, y_enc)


def transform_spikes(X, y, param, receptor=None, num_classes=None):
    """
    Transform a data set into predetermined PSPs, spike trains, one-hot
    encoded class labels.

    Inputs
    ------
    X : list, len (num_samples)
        Input data, represented as set of spike patterns.
    y : array, shape (num_samples,)
        Unprocessed labels.
    param : object
        Top level parameter container, using pattern and cell.
    receptor : object, optional
        Treat each set of spike trains as real-valued features, and encode
        as spike latencies.
    num_classes : int, optional
        Set one-hot encoded labels to this number of classes.

    Output
    ------
    data : list
        Processed data as list of 2-tuples, [(X1, y1), (X2, y2), ...].
        Each X is itself a 2-tuple, containing (psps, spike_trains). y's are
        one-hot encoded.
    """
    y_enc = onehot_encode(y, num_classes=num_classes)
    if receptor is None:
        # Predetermine PSPs, return as list of 2-tuples
        # containing [(psps_0, pattern_0), ...]
        inputs = [None] * len(X)
        for idx, pattern in enumerate(X):
            psps = pattern2psps(pattern, param.dt, param.pattern['duration'],
                                **param.cell)
            inputs[idx] = (psps, pattern)
        return zip(inputs, y_enc)
    else:
        pass


def onehot_encode(y, num_classes=None):
    """
    Transform array of class labels into one-hot encoded matrix.

    Inputs
    ------
    y : array, shape (num_samples,)
        Class labels.
    num_classes : int, optional
        Number of unique classes.

    Output
    ------
    return : array, shape (num_samples, num_classes)
        One-hot encoded class labels.
    """
    m = len(y)  # Num examples
    labels = np.unique(y)  # Fill in matrix in this order
    if num_classes is None:
        num_classes = len(labels)
    y_ = np.zeros((m, num_classes))
    for i in xrange(m):
        index = np.where(labels == y[i])[0][0]
        y_[i, index] = 1.0
    return y_


def pattern2psps(pattern, dt, duration, **kwargs):
    """
    Transform a spike pattern containing a set of spike trains into their
    evoked PSPs, for given duration and time step.
    """
    times = np.arange(0., duration, dt)
    # Evoked PSPs due to the spike pattern
    psps = np.stack([psp_reduce(times, spike_train, **kwargs)
                     for spike_train in pattern])
    return psps


def latencies_psps(spikes, dt, duration, return_trains=False, **kwargs):
    """
    Transform a set of spike latencies into their evoked PSPs, for a given
    duration and time step. Optionally also return associated spike trains.

    Inputs
    ------
    spikes : array, shape (num_samples, num_enc_nrns)
        Set of spike latencies, one spike contributed per encoding neuron.
    return_trains: bool, optional
        Return list of 2-tuples, containing (psps, spike_trains) per sample.

    Output
    ------
    psps : list, len (num_samples)
        Set of predetermined PSPs evoked by encoding neurons, each with shape
        (num_enc_nrns, num_iter)
    psps, spike_trains : list, len (num_samples)
        List of 2-tuples [(psps_0, spike_trains_0), ...].
    """
    times = np.arange(0., duration, dt)
    # Determine evoked PSPs, each sample has shape (num_nrns, num_iter)
    psps = [psp(times, s, **kwargs).T for s in spikes]
    if return_trains:
        patterns = latencies2patterns(spikes)
        return zip(psps, patterns)
    else:
        return psps


def latencies2patterns(latencies):
    """
    Converts an array of spike latencies into their corresponding spike
    patterns.

    Input
    -----
    latencies : array, shape (num_samples, num_nrns)
        Array of spike latencies.

    Output
    ------
    patterns : list, len (num_samples)
        List of spike patterns, each containing a list of spike trains.
    """
    patterns = [None] * len(latencies)
    for idx, latencies in enumerate(latencies):
        patterns[idx] = [np.array([s]) if not np.isinf(s)
                         else np.array([]) for s in latencies]
    return patterns


def psp(times, spikes, **kwargs):
    """
    PSP evoked at each t in times due to each spike.

    Inputs
    ------
    times : array, shape (num_times)
        Evoked times.
    spikes : array, shape (num_spikes)
        Presynaptic spike times.

    Output
    ------
    return : array, shape (num_times, num_spikes)
        Evoked PSPs.
    """
    params = ParamSet({'psp_coeff': 4.,
                       'tau_m': 10.,
                       'tau_s': 5.})
    params.overwrite(**kwargs)
    s = times[:, np.newaxis] - spikes[np.newaxis, :]
    u = s > 0.
    values = np.zeros(s.shape)
    values[u] = params['psp_coeff'] * \
        (np.exp(-s[u] / params['tau_m']) - np.exp(-s[u] / params['tau_s']))
    return values


def psp_reduce(times, spikes, **kwargs):
    """
    Evoked PSP at each t in times, summed w.r.t. array of spikes.
    Returns evoked PSPs with shape (times,).
    """
    return np.sum(psp(times, spikes, **kwargs), 1)


def gaussian(x, mu, sigma):
    """
    Gaussian function for firing intensity.
    """
    return np.exp(-0.5 * ((x - mu) / sigma)**2)
