#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Stratified k-fold cross-validation on one of the following datasets:
    - iris
    - wisconsin (wisc)

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
from sklearn.model_selection import StratifiedKFold

from snncls import values, dataset_loader, preprocess, metric, plotter
from snncls.supervised import SoftmaxClf
from lib import common


# %% === Pattern stats ====================================================== #

# Datasets: {'iris', 'wisc'}
data_id = 'iris'

# Load data set in format (X, y) with raw values
X, y = dataset_loader.load_data_file(data_id)
num_classes = len(np.unique(y))

# %% === Network stats ====================================================== #

# Specify parameters
num_nrns = 40
opt, _ = common.default_opt(data_id)
opt.update({'seed': None,
            'w_lim': 15.,
            'cpd_scale': 4.,
            'l2_pen': 1E-4,
            'rate_pow': 2})
# Batch learning
num_splits = 3  # 3-fold cross-validation
mini_batch_size = 150
num_epochs = 100    # Iris: num_epochs=100; num_splits=3: ~ 3 mins on i5 6600K
# Learning schedule: {'sgd', 'rmsprop', 'adam'}
solver = 'rmsprop'
alpha = 0.1
# Optional
predet_psps = True
plot = True
save_data = False

# Top-level parameters container
prms = values.LatencyParam(**opt)

# %% === Preprocessing ====================================================== #

# Statified k-fold cross-validation model
skf = StratifiedKFold(n_splits=num_splits, shuffle=True,
                      random_state=prms.rng)
idxs = list(skf.split(X, y))
# Prepare feature preprocessor - only fitted to training data
receptor = preprocess.ReceptiveFields(prms.pattern['neurons_f'],
                                      prms.pattern['beta'])

# %% === Classifier ========================================================= #

num_inputs = prms.pattern['neurons_f'] * X.shape[1]
clf = SoftmaxClf(sizes=[num_inputs, num_nrns, num_classes], param=prms)

# %% === Training and testing =============================================== #

# Cross-validate
recs = {'tr_loss': np.full((num_splits, num_epochs), np.nan),
        'te_loss': np.full((num_splits, num_epochs), np.nan),
        'te_errs': np.full(num_splits, np.nan),
        'w': [None] * num_splits}
recs['idxs'] = idxs
t1 = time.time()
for i, (tr_idxs, te_idxs) in enumerate(idxs):
    # Select training and val samples, one-hot encoded labels
    X_tr, X_te = X[tr_idxs], X[te_idxs]
    y_tr, y_te = y[tr_idxs], y[te_idxs]
    # Fit receptor to training data
    receptor.fit(X_tr)
    # Prepare data sets for network
    data_tr = preprocess.transform_data(X_tr, y_tr, prms, receptor,
                                        num_classes, return_psps=predet_psps)
    data_te = preprocess.transform_data(X_te, y_te, prms, receptor,
                                        num_classes, return_psps=predet_psps)
    # Train the network
    clf.net.reset()
    rec = clf.SGD(data_tr, num_epochs, mini_batch_size, te_data=data_te,
                  report=True, epochs_r=10, debug=True, solver=solver,
                  alpha=alpha)
    recs['tr_loss'][i] = rec['tr_loss']
    if 'te_loss' in rec:
        recs['te_loss'][i] = rec['te_loss']
    if 'w' in rec:
        recs['w'][i] = [w.copy() for w in rec['w']]
    recs['te_errs'][i] = clf.evaluate(data_te)
t2 = time.time()

# Profile
print('Time:\t{:.3f} s'.format(t2-t1))

# %% === Analyse results ==================================================== #

ws = [[w[-1] for w in w_fold] for w_fold in recs['w']]
mats = []
for i, (tr_idxs, te_idxs) in enumerate(idxs):
    # Select training and val samples, one-hot encoded labels
    X_tr, X_te = X[tr_idxs], X[te_idxs]
    y_tr, y_te = y[tr_idxs], y[te_idxs]
    # Fit receptor to training data
    receptor.fit(X_tr)
    # Prepare data sets for network
    data_tr = preprocess.transform_data(X_tr, y_tr, prms, receptor,
                                        num_classes, return_psps=predet_psps)
    data_te = preprocess.transform_data(X_te, y_te, prms, receptor,
                                        num_classes, return_psps=predet_psps)
    clf.net.reset(weights=ws[i])
    mats.append(metric.confusion_matrix(clf, data_te, False))
mats = np.stack(mats, 2)
recs['mats'] = mats
# Average test accuracy
print('Test accuracy:\t{:.1f} %'.format(100. - recs['te_errs'].mean()))

# %% === Plot selected results ============================================== #

if plot:
    pltr = plotter.Plotter()
    pltr.confusion_matrix(recs['mats'].mean(-1))
    # Plot spikeraster on a test pattern
    idx = 0
    sample = preprocess.transform_data(X_te[[idx]], y_te[[idx]], prms,
                                       receptor, num_classes, False)
    stimulus, label = sample[0]
    spikes, rec_s = clf.net.simulate(stimulus, debug=True)
    spikers = [stimulus] + spikes
    class_id = np.argmax(label)
    pltr.spike_rasters(spikers,
                       text='{}: input class is {}'.format(data_id, class_id))


# %% === Save results ======================================================= #

if save_data:
    # Config prms
    cfg = vars(prms).copy()
    cfg.update({'net_size': clf.net.sizes,
                'num_splits': num_splits,
                'mini_batch_size': mini_batch_size,
                'num_epochs': num_epochs,
                'solver': solver,
                'alpha': alpha,
                'fname': 'skf'})
    del cfg['rng_st0'], cfg['rng']
    common.save_data(recs, cfg, basename=data_id)
