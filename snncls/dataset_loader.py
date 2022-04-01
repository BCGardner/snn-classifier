"""
Dataset loading functions.

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

import os
import numpy as np
from mnist import MNIST

import snncls.preprocess as pp
import snncls.helpers as hp


def load_data_transform(data_id, param, transform=pp.transform_data,
                        return_psps=True):
    """
    Loads a dataset in form of 2-tuple: (X, y), and transforms features into
    a form suitable for SNN training. Returns predetermined PSPs evoked due to
    input layer, assuming current-based LIF-type neurons in the network.

    Inputs
    ------
    data_id : str
        Name of dataset.
    param : container
        Using dt and paramsets: pattern, cell.
    transform : function
        Transformation method to convert features into spike-based
        representation.
    return_psps : bool
        Predetermine PSP's evoked by input neurons for optimisation.

    Outputs
    -------
    return : list
        List of 2-tuples [(X_0, y_0), ...]. X's contain evoked PSPs due to
        input layer. y's are one-hot encoded class labels. X has shape
        (num_inputs, num_iter), and y (num_classes,).
    """
    # Load dataset
    X, y = load_data_file(data_id)
    # Preprocess data
    data_set = transform(X, y, param, return_psps=return_psps)
    return data_set


def load_data_file(data_id):
    """
    Loads a data file from ./data for model training / validation / test.
    """
    # Load dataset
    data_path = os.path.join(os.path.dirname(__file__), 'data')
    fname = os.path.join(data_path, '{}.pkl.gz'.format(data_id))
    if data_id == 'mnist':
        data_path = os.path.join(data_path, 'mnist')
        mndata = MNIST(data_path)
        data, labels = mndata.load_training()
        return np.array(data, dtype=float), np.array(labels, dtype=int)
    else:
        return hp.load_data(fname)


def load_mnist_testing():
    """
    Loads mnist test data from ./data for final model evaluation.
    """
    # Load dataset
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'mnist')
    mndata = MNIST(data_path)
    data, labels = mndata.load_testing()
    return np.array(data, dtype=float), np.array(labels, dtype=int)


def load_subset(data_id, num_cases, randomise=False, rng=None):
    """
    Subsample a loaded dataset. Either the first num_cases are returned, or
    a randomised set.
    """
    # Setup
    if not isinstance(rng, int):
        rng = np.random.RandomState(rng)
    # Load data and subsample[, randomised order of first num_cases]
    X, y = load_data_file(data_id)
    idxs = np.arange(len(X))
    if randomise:
        rng.shuffle(idxs)
    return X[idxs][:num_cases], y[idxs][:num_cases]
