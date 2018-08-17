#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Jun 2017

@author: BG
"""

from __future__ import division

import numpy as np


def van_rossum(spikes, spikes_ref, tau_c=10.0):
    """
    Exactly computes the van Rossum distance between two spike trains.
    """
    # Contribution from interaction between output spikes
    dist_out = 0.0
    for i in spikes:
        for j in spikes:
            dist_out += np.exp(-(2*max(i, j) - i - j) / tau_c)
    # Contribution from interaction between reference spikes
    dist_ref = 0.0
    for i in spikes_ref:
        for j in spikes_ref:
            dist_ref += np.exp(-(2*max(i, j) - i - j) / tau_c)
    # Contribution from interaction between output and reference spikes
    dist_out_ref = 0.0
    for i in spikes:
        for j in spikes_ref:
            dist_out_ref += np.exp(-(2*max(i, j) - i - j) / tau_c)
    return (dist_out + dist_ref - 2*dist_out_ref) / 2


def distance_spatio(spike_trains, spike_trains_ref, dist_metric=van_rossum):
    """
    Computes the distance (using a given metric) between two spatiotemporal
    spike patterns.
    """
    n_trains = len(spike_trains)
    if len(spike_trains_ref) != n_trains:
        raise ValueError('Spike patterns differ spatially in size')
    # Average distance between spike trains
    spatio_dist = 0.0
    for spikes, spikes_ref in zip(spike_trains, spike_trains_ref):
        spatio_dist += dist_metric(spikes, spikes_ref)
    return spatio_dist / n_trains


def confusion_matrix(net, data, raw=True):
    """
    Measure confusion matrix of a network on provided data.
    Data is a list of 2-tuples [(X1, y1), (X2, y2), ...], where each X is also
    a 2-tuple containing (psps_in, spikes_in). Raw returns counts per matrix
    element. Final column is the null class, containing counts for no spike.
    """
    num_classes = len(data[0][1])
    # Confusion matrix: <true classes> by <predicted classes>
    conf_mat = np.zeros((num_classes, num_classes + 1))
    for (X, _), y in data:
        label = np.argmax(y)  # True class label
        predict = net.predict(X)  # Predicted class label
        # Given a prediction by the network
        if not np.isnan(predict):
            conf_mat[label, predict] += 1.
        else:
            conf_mat[label, -1] += 1.
    if raw:
        return conf_mat
    else:
        # Percentage of occurances per row
        return conf_mat * 100. / np.sum(conf_mat, 1, keepdims=True)
