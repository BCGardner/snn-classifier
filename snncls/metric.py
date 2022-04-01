"""
Network and classifier metrics.

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


def confusion_matrix(clf, data, raw=False, return_acc=False):
    """
    Determine confusion matrix of a classifier on provided data.

    Inputs
    ------
    clf : object
        Classifier with predict method.
    data : list, len (num_samples)
        Dataset samples, [(X0, y0), (X1, y1), ...].
    raw : bool, optional
        Confusion matrix as counts per element, otherwise as (%).
    return_acc : bool, optional
        Additionally return overall classifier accuracy.

    Output
    ------
    return : array or 2-tuple
        Confusion matrix[, overall prediction accuracy on input data]. Final
        column of the confusion matrix indicates a null prediction (no-spike).
    """
    num_classes = len(data[0][1])
    # Confusion matrix: <true classes> by <predicted classes>
    conf_mat = np.zeros((num_classes, num_classes + 1))
    # [Optional overall accuracy]
    if return_acc:
        correct = 0.
    for X, y in data:
        label = np.argmax(y)  # True class label
        predict = clf.predict(X)  # Predicted class label
        # Given a prediction by the network
        if not np.isnan(predict):
            conf_mat[label, predict] += 1.
        else:
            conf_mat[label, -1] += 1.
        # [Optional overall accuracy]
        if return_acc:
            if predict == np.argmax(y):
                correct += 1.0
    if not raw:
        # Percentage of occurances per row
        conf_mat *= 100. / np.sum(conf_mat, 1, keepdims=True)
    if return_acc:
        return conf_mat, correct / len(data) * 100.
    else:
        return conf_mat


def rates_expt(net, data, weights):
    """
    Find average firing rates per layer, on each epoch of an experiment. Expt
    may consist of several independent runs. Num_layers excludes input layer.

    Inputs
    ------
    net : object
        Pre-initialised network object.
    data : list, len (num_samples)
        Dataset samples, [(X0, y0), (X1, y1), ...], to sample rates from.
    weights : list, len (num_layers)
        Recorded network weights. Each list element contains an array of shape
        ([num_runs], num_epochs, num_post, num_pre).

    Output
    ------
    return : list, len (num_layers)
        Average firing rates of neurons in each layer. Each list element
        contains array of shape (num_epochs[, num_runs], num_nrns).
    """
    # Cast each weights elem to array with shape (num_runs, ...)
    if np.ndim(weights[-1]) == 3:
        weights = [ws[np.newaxis, :] for ws in weights]
    elif np.ndim(weights[-1]) != 4:
        raise ValueError
    # Dims
    num_runs, num_epochs = weights[-1].shape[:2]
    # For each epoch: iterate through each run, sample average rate of each
    # neuron from data, per layer
    rates = [np.full((num_epochs, num_runs, i), np.nan)
             for i in net.sizes[1:]]
    for epoch in range(num_epochs):
        for run in range(num_runs):
            ws = [w[run, epoch] for w in weights]
            net.reset(weights=ws)
            rs_l = [rs.mean(0) for rs in rates_data(net, data)]
            for idx, rs in enumerate(rs_l):
                rates[idx][epoch, run, :] = rs
    if num_runs == 1:
        return [rs[:, 0, :] for rs in rates]
    else:
        return rates


def rates_data(net, data):
    """
    Find the firing rate of each neuron per layer, in response to each data
    sample. Returns a list, len (num_layers), where each elem contains an array
    of shape (num_samples, num_nrns).
    """
    num_samples = len(data)
    rates = [np.full((num_samples, i), np.nan) for i in net.sizes[1:]]
    for idx, (X, _) in enumerate(data):
        spike_trains_l = net.simulate(X)
        for i, spike_trains in enumerate(spike_trains_l):
            for j, spikes in enumerate(spike_trains):
                rates[i][idx, j] = len(spikes)
    return rates
