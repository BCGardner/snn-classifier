"""
Helper functions.

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

import gzip
import pickle as pkl
import itertools
import os

import numpy as np

from snncls import metric
from snncls.decorators import deprecated


def get_basename(filename, return_ext=False):
    """
    Get basename of file, with or without extension.
    """
    basename = os.path.basename(filename)
    if return_ext:
        return basename
    else:
        return basename.split('.')[0]


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
        data = pkl.load(f, encoding='latin1')
    return data


@deprecated
def evaluate_clf(clf, data_tr, data_te=None, return_mats=True, report=False):
    """
    Evaluate classifier training[, test] accuracies[, and confusion matrices].

    Inputs
    ------
    clf : object
        Network classifier with prediction and evaluate methods.
    data_tr : list
        Training data as list of 2-tuples: [(X1, y1), ...].
    data_te : list, optional
        Test data as list of 2-tuples: [(X1, y1), ...].
    return_mats : bool
        Evaluate network confusion matrices.
    report : bool
        Report mean training[,test] accuracies.

    Output
    ------
    return : dict
        Recorded training[, test] accuracies.
    """
    rec = {}
    # Evaluate on train / test data w.r.t. all classes (error rate %)
    rec['tr_err'] = clf.evaluate(data_tr)
    if data_te is not None:
        rec['te_err'] = clf.evaluate(data_te)
    # Evaluate on train and test data w.r.t. each class (accuracy %)
    rec['mat_tr'] = metric.confusion_matrix(clf, data_tr, False)
    if data_te is not None:
        rec['mat_te'] = metric.confusion_matrix(clf, data_te, False)
    # Report values
    if report:
        print('Train accuracy:\t{0:.1f} %'.format(100. - rec['tr_err']))
        if data_te is not None:
            print('Test accuracy:\t{0:.1f} %'.format(100. - rec['te_err']))
    return rec


def mean_accs(data, k='tr_err'):
    """
    Return 2D array of average accuracies and associated SEMs,
    given 2D array of dictionaries containing a given errors key.
    """
    num_rows, num_cols = data.shape
    num_runs = len(data[0, 0][k])
    accuracies = np.zeros((num_rows, num_cols, 2))
    for i, j in itertools.product(range(num_rows), range(num_cols)):
        accuracies[i, j, 0] = np.mean(100. - data[i, j][k])
        accuracies[i, j, 1] = np.std(100. - data[i, j][k])
    # Return SEMs
    accuracies[:, :, 1] /= np.sqrt(num_runs)
    return accuracies
