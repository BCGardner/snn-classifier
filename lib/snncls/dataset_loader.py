#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on June 2018

@author: BG

Dataset loading functions.
"""

import os

import snncls.preprocess as pp
import snncls.helpers as hp


def load_data_spiking(data_id, param):
    """
    Loads a dataset in form of 2-tuple: (X, y), preprocesses features, changes
    format of data, and returns predetermined PSPs evoked due to input layer,
    assuming current-based LIF-type neurons in the network. Also returns list
    of input spike times.

    Inputs
    ------
    data_id : str
        Name of dataset.
    param : container
        Using dt and paramsets: pattern, cell.

    Outputs
    -------
    return : list
        List of 2-tuples [(X1, Y1), ...]. X's are 2-tuples containing
        evoked PSPs due to input layer, and list of associated spike times.
        Y's are one-hot encoded class labels. X[0] has shape
        (num_inputs, num_iter), X[1] (num_inputs,), and Y (num_classes,).
    """
    # Load dataset
    X, y = load_data_file(data_id)
    # Preprocess data
    data_tr = pp.transform_data(X, y, param)
    return data_tr


def load_data_file(data_id):
    """
    Loads a data file from ../data/.
    """
    # Load dataset
    data_path = os.path.join(os.path.dirname(__file__), '../data')
    fname = os.path.join(data_path, '{}.pkl.gz'.format(data_id))
    return hp.load_data(fname)
