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


def transform_data(X, y, param, receptor=None, num_classes=None,
                   return_psps=True):
    """
    Transform a data set into predetermined PSPs, spike latencies, one-hot
    encoded class labels.

    Inputs
    ------
    X : array, shape (num_samples, features)
        Input data.
    y : array, shape (num_samples,)
        Class labels.
    param : object
        Top level parameter container, using pattern and cell.
    receptor : object, optional
        Model for transforming real-valued features into spike latencies.
    num_classes : int, optional
        Set one-hot encoded labels to this number of classes.
    predet_psps : bool
        Predetermine PSP's evoked by input neurons for optimisation.

    Output
    ------
    data : list
        Processed data as list of 2-tuples, [(X_0, y_0), ...].
        Each X consists of either predetermined PSP's evoked by input LIF
        neurons, or an input spike pattern, and y's are one-hot encoded labels.
    """
    if receptor is None:
        # Prepare feature preprocessor fitted to this specific data
        receptor = ReceptiveFields(param.pattern['neurons_f'],
                                   param.pattern['beta'])
        receptor.fit(X)
    # Transform data
    latencies = receptor.transform(X)
    y_enc = onehot_encode(y, num_classes=num_classes)
    # Input representation
    if return_psps:
        input_data = latencies2psps(latencies, param.dt,
                                    param.pattern['duration'],
                                    return_trains=True, **param.cell)
    else:
        input_data = latencies2patterns(latencies)
    return zip(input_data, y_enc)


def transform_spikes(X, y, param, receptor=None,
                     num_classes=None, return_psps=True):
    """
    Transform a data set into predetermined PSPs, spike trains, one-hot
    encoded class labels.

    TODO: Transform input patterns into latencies using integrating receptors.

    Inputs
    ------
    X : list, len (num_samples)
        Input data, represented as set of spike patterns.
    y : array, shape (num_samples,)
        Class labels.
    param : object
        Top level parameter container, using pattern and cell.
    receptor : object, optional
        Treat each set of spike trains as real-valued features, and encode
        as spike latencies.
    num_classes : int, optional
        Set one-hot encoded labels to this number of classes.
    predet_psps : bool
        Predetermine PSP's evoked by input neurons for optimisation.

    Output
    ------
    data : list
        Processed data as list of 2-tuples, [(X_0, y_0), ...].
        Each X consists of either predetermined PSP's evoked by input LIF
        neurons, or an input spike pattern, and y's are one-hot encoded labels.
    """
    y_enc = onehot_encode(y, num_classes=num_classes)
    if receptor is None:
        if return_psps:
            inputs = [None] * len(X)
            for idx, pattern in enumerate(X):
                psps = pattern2psps(pattern, param.dt,
                                    param.pattern['duration'], **param.cell)
                inputs[idx] = psps
            return zip(inputs, y_enc)
        else:
            return zip(X, y_enc)
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

    Parameters
    ----------
    pattern : list, len (num_pre)
        Presynaptic spike trains.
    dt : float
        Simulation time step.
    duration : float
        Response duration.

    Returns
    -------
    array, shape (num_pre, num_iter)
        PSPs evoked due to each presynaptic spike train.
    """
    times = np.arange(0., duration, dt)
    num_iter = len(times)
    # Optimisation
    lut_psp = psp_reduce(times, np.array([0.]), **kwargs)
    # Evoked PSPs due to the spike pattern
    psps = np.stack([np.zeros(num_iter) for i in xrange(len(pattern))])
    for idx, spike_train in enumerate(pattern):
        t_iters = np.round(np.asarray(spike_train) / dt).astype(int)
        for t_iter in t_iters:
            psps[idx][t_iter:] += lut_psp[:num_iter - t_iter]
    # psps = np.stack([psp_reduce(times, spike_train, **kwargs)
    #                      for spike_train in pattern])
    return psps


def latencies2psps(spikes, dt, duration, **kwargs):
    """
    Transform a set of spike latencies into their evoked PSPs, for a given
    duration and time step.

    Inputs
    ------
    spikes : array, shape (num_samples, num_enc_nrns)
        Set of spike latencies, one spike contributed per encoding neuron.

    Output
    ------
    psps : list, len (num_samples)
        Set of predetermined PSPs evoked by encoding neurons, each with shape
        (num_enc_nrns, num_iter).
    """
    times = np.arange(0., duration, dt)
    # Determine evoked PSPs, each sample has shape (num_nrns, num_iter)
    psps = [psp(times, s, **kwargs).T for s in spikes]
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


def refr(times, spikes, **kwargs):
    """
    Refractory kernel at each t in times in response to array of spikes.
    Returns response array, shape (num_times, num_spikes).
    """
    params = ParamSet({'kappa_0': -15.,
                       'tau_m': 10.})
    params.overwrite(**kwargs)
    s = times[:, np.newaxis] - spikes[np.newaxis, :]
    u = s > 0.
    values = np.zeros(s.shape)
    values[u] = params['kappa_0'] * np.exp(-s[u] / params['tau_m'])
    return values


def refr_reduce(times, spikes, **kwargs):
    """
    Refractory effects at each t in times, summed w.r.t. array of spikes.
    Returns total refractoriness as array, shape (times,).
    """
    return np.sum(refr(times, spikes, **kwargs), 1)


def gaussian(x, mu, sigma):
    """
    Gaussian function for firing intensity.
    """
    return np.exp(-0.5 * ((x - mu) / sigma)**2)
