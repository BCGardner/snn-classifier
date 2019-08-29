#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
MNIST: one-one encoding of each input pixel as a spike latency. Spike latencies
are encoded into the first layer of a feedforward multilayer network with
all-to-all connectivity between layers. This example demonstrates a network
with one hidden layer, containing escape noise SRM neurons. The final layer
contains deterministic SRM neurons, which predict input classes based on the
first output neuron to respond with a spike.

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
import time
from sklearn.model_selection import train_test_split

from snncls import values, transform, preprocess, helpers, metric, plotter
from snncls.supervised import SoftmaxClf
from lib import common


# %% === Pattern stats ====================================================== #

# Dataset-specific parameters
data_name = 'data/mnist3750.pkl.gz'  # Load-in first 3750 training samples
shuffle = True
test_size = 0.2

# Load dataset
X, y = helpers.load_data(data_name)
num_inputs = len(X[0])  # Num. input features / neurons (one-one mapping)
num_classes = len(np.unique(y))

# %% === Network stats ====================================================== #

# Simulation parameters
num_nrns = 80
opt = {'seed': None,  # Used to initialise rand. num. gen.
       'w_lim': 2.,  # w_{hi, lo} ~ 240 / num_input_spikes
       'neurons_f': 1,
       'w_h_init': (0., 0.4),
       'w_o_init': (0., 32. / num_nrns),
       'cpd_scale': 4.,
       'l2_pen': 1E-4,
       'rate_pow': 2}
# Batch learning
mini_batch_size = 150
num_epochs = 10    # Num. training epochs = 10: < 20 mins on i5 6600K
# Early-stopping
early_stopping = False
tol = 1E-2
num_iter_stopping = 5
# Learning schedule
solver = 'rmsprop'
alpha = 0.01  # alpha ~ 0.0067 * w_{hi}
# Optional
predet_psps = False  # Memory intensive (requires > 16 GB)
debug = False
plot = True
save_data = False

# Top-level parameters container used to initialise classifier / pattern stats
prms = values.LatencyParam(**opt)

# %% === Preprocessing ====================================================== #

# Split dataset into training and test sets (4:1 split if test_size=0.2)
if shuffle:
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size,
                                              shuffle=True, stratify=y,
                                              random_state=prms.rng)
else:
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size,
                                              shuffle=False)

# Transform data into spike latencies in [0, 9) ms: one-one association
# Prepare feature preprocessor - only fitted to training data
receptor = transform.Integrator()
receptor.fit(X_tr)
# Transformed data is a list of 2-tuples: [(X_1, y1), (X2, y2), ...], where
# each X is a list of input spike trains (or predetermined, evoked PSPs), and
# each y is the corresponding one-hot encoded class label
data_tr = preprocess.transform_data(X_tr, y_tr, prms, receptor, num_classes,
                                    return_psps=predet_psps)
data_te = preprocess.transform_data(X_te, y_te, prms, receptor, num_classes,
                                    return_psps=predet_psps)

# %% === Classifier ========================================================= #

clf = SoftmaxClf(sizes=[num_inputs, num_nrns, num_classes], param=prms)

# %% === Training =========================================================== #

# Collect classifier training arguments
kwargs_tr = dict(mini_batch_size=mini_batch_size, data_te=data_te,
                 report=True, debug=debug, solver=solver, alpha=alpha,
                 early_stopping=early_stopping, tol=tol,
                 num_iter_stopping=num_iter_stopping)

# Initialise recorder
if debug:
    rec = {'w_init': [w for w in clf.net.get_weights()],
           'prms': vars(prms).copy()}
else:
    rec = {}

t1 = time.time()
rec_tr = clf.SGD(data_tr, num_epochs, **kwargs_tr)
t2 = time.time()
print('Time:\t{:.3f} s'.format(t2-t1))

rec.update(rec_tr)
if not debug:
    rec['w'] = [w for w in clf.net.get_weights()]

# %% === Evaluate =========================================================== #

rec['mat_tr'], rec['tr_acc'] = metric.confusion_matrix(clf, data_tr,
                                                       return_acc=True)
rec['mat_te'], rec['te_acc'] = metric.confusion_matrix(clf, data_te,
                                                       return_acc=True)
print('Test accuracy:\t{:.1f} %'.format(rec['te_acc']))

# %% === Probe the network with a pattern =================================== #

#idx = 0
#spikes, rec_s = clf.net.simulate(data_te[idx][0], debug=True)

# %% === Plot selected results ============================================== #

if plot:
    pltr = plotter.Plotter()
    pltr.confusion_matrix(rec['mat_te'])

# %% === Save results ======================================================= #

# Results
if save_data:
    # Config prms
    cfg = vars(prms).copy()
    cfg.update({'net_size': clf.net.sizes,
                'mini_batch_size': mini_batch_size,
                'num_epochs': num_epochs,
                'early_stopping': early_stopping,
                'tol': tol,
                'num_iter_stopping': num_iter_stopping,
                'solver': solver,
                'alpha': alpha,
                'fname': 'latencies'})
    del cfg['rng_st0'], cfg['rng']
    basename = helpers.get_basename(data_name)
    common.save_data(rec, cfg, basename=basename)
