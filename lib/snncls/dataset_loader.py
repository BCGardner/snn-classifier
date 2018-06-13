#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on June 2018

@author: BG

Dataset loading functions.
"""

import os

import numpy as np

import feature_preprocess as f_p
import helpers as hp


def load_data_wrapper(data_id, f_transform=f_p.standardize):
    """
    Loads a dataset in form of 2-tuple: (X, y), preprocesses features, changes
    format of data, and returns a list of 2-tuples [(X1, y1), (X2, y2), ...].

    Inputs
    ------
    data_id : str
        Name of dataset.
    f_transform : function
        Pre-processing transformation on data.
    """
    dataset = load_data_file(data_id)
    # Preprocess
    data_inputs = f_transform(dataset[0])
    data_targets = f_p.onehot_encode(dataset[1])
    # Transform to list of vectors
    data_inputs = list(data_inputs)
    data_targets = list(data_targets)
    training_data = zip(data_inputs, data_targets)
    return training_data


def load_data_spiking(data_id, param):
    """
    Loads a dataset in form of 2-tuple: (X, y), preprocesses features, changes
    format of data, and returns a list of 2-tuples [(X1, y1), (X2, y2), ...].
    This is for a spiking classifier.

    Inputs
    ------
    data_id : str
        Name of dataset.
    param : container
        Using dt and paramsets: pattern, cell.
    """
    # Load dataset
    dataset = load_data_file(data_id)
    # Init
    dt = param.dt
    neurons_f = param.pattern['neurons_f']
    duration = param.pattern['duration']
    cell_params = param.cell
    # Preprocess
    data_inputs = f_p.receptive_fields(dataset[0], neurons_f)
    data_targets = f_p.onehot_encode(dataset[1])
    # Transform each input vector to size: <n_activations> by <n_cases>
    data_inputs = [np.reshape(x, (-1, 1)) for x in data_inputs]
    # Transform each target vector to size: <n_activations> by <n_cases>
    data_targets = list(data_targets)
#    data_targets = [np.reshape(y, (-1, 1)) for y in data_targets]
    # Predetermine input PSPs
    psp_inputs = f_p.predet_psp(data_inputs, dt, duration, cell_params)
    # List of 2-tuples: [(input1, target1), ...]
    training_data = zip(psp_inputs, data_targets)
    return training_data, data_inputs


def load_data_file(data_id):
    """
    Loads a data file from ../data/.
    """
    # Load dataset
    data_path = os.path.join(os.path.dirname(__file__), '../data')
    fname = os.path.join(data_path, '{}.pkl.gz'.format(data_id))
    return hp.load_data(fname)
