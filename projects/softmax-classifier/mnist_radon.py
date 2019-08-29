#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Performs Radon transform on provided MNIST data, where the transformation is
then encoded using a one-one mapping per input nrn.
See: <https://scikit-image.org/docs/dev/auto_examples/transform/
      plot_radon_transform.html>.

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
from skimage.transform import radon
from sklearn.model_selection import train_test_split

from snncls import values, transform, preprocess, helpers, metric, plotter
from snncls.supervised import SoftmaxClf
from lib import common


# %% === Pattern stats ====================================================== #

# Dataset-specific parameters
data_name = 'data/mnist3750.pkl.gz'
num_proj = 9
theta = np.linspace(0., 180., num_proj, endpoint=False)
circle = True
shuffle = True
test_size = 0.2

# Load dataset
X, y = helpers.load_data(data_name)
num_classes = len(np.unique(y))

# %% === Network stats ====================================================== #

# Simulation parameters
num_nrns = 80
opt = {'seed': None,
       'w_lim': 2.,
       'neurons_f': 1,
       'w_h_init': (0., 4. / num_proj),
       'w_o_init': (0., 32. / num_nrns),
       'cpd_scale': 4.,
       'l2_pen': 1E-4,
       'rate_pow': 2}
# Batch learning
mini_batch_size = 150
num_epochs = 15     # Num. training epochs = 15: ~ 20 mins on i5 6600K
# Early-stopping
early_stopping = False
tol = 1E-2
num_iter_stopping = 5
# Learning schedule
solver = 'rmsprop'
alpha = 0.01  # alpha ~ 0.0067 * w_{hi}
# Optional
predet_psps = True
debug = False
plot = True
save_data = False

# Top-level parameters container
prms = values.LatencyParam(**opt)

# %% === Preprocessing ====================================================== #

# Radon transform of MNIST images
X = X.reshape((-1, 28, 28))
Q = []
for x in X:
    Q.append(radon(x, theta, circle=circle))
Q = np.stack(Q)
# Rearrange data into array with shape (num_samples, num_features)
X = X.reshape((-1, np.prod(X.shape[1:])))
Q = Q.reshape((-1, np.prod(Q.shape[1:])))
idxs = np.arange(len(Q))

# Split dataset into training and test sets (4:1 split)
if shuffle:
    Q_tr, Q_te, y_tr, y_te, idxs_tr, idxs_te = \
        train_test_split(Q, y, idxs, test_size=0.2, shuffle=True, stratify=y,
                         random_state=prms.rng)
else:
    Q_tr, Q_te, y_tr, y_te, idxs_tr, idxs_te = \
        train_test_split(Q, y, idxs, test_size=0.2, shuffle=False)

# Transform data into spike latencies in [0, 9) ms: one-one association
# Prepare feature preprocessor - only fitted to training data
receptor = transform.Integrator(theta=8.)
receptor.fit(Q_tr)
data_tr = preprocess.transform_data(Q_tr, y_tr, prms, receptor, num_classes,
                                    return_psps=predet_psps)
data_te = preprocess.transform_data(Q_te, y_te, prms, receptor, num_classes,
                                    return_psps=predet_psps)
# DEBUG: Proportion of active encoding nrns
if debug:
    counts = np.empty(len(data_tr))
    for idx, case in enumerate(data_tr):
        counts[idx] = np.mean([len(s) for s in case[0]])
    print('{:.3f} ({:.3f})'.format(np.mean(counts), np.std(counts)))

# %% === Classifier ========================================================= #

num_inputs = len(Q[0])  # Num. input features / neurons (one-one mapping)
clf = SoftmaxClf(sizes=[num_inputs, num_nrns, num_classes], param=prms)

# %% === Training =========================================================== #

# Initialise recorder
if debug:
    rec = {'w_init': [w for w in clf.net.get_weights()],
           'prms': vars(prms).copy()}
else:
    rec = {}

# Simulate
t1 = time.time()
rec.update(clf.SGD(data_tr, num_epochs, mini_batch_size=mini_batch_size,
                   data_te=data_te, report=True, debug=debug, solver=solver,
                   alpha=alpha, early_stopping=early_stopping, tol=tol,
                   num_iter_stopping=num_iter_stopping))
t2 = time.time()
print('Time:\t{:.3f} s'.format(t2-t1))

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

if save_data:
    # Config prms
    cfg = vars(prms).copy()
    cfg.update({'num_proj': num_proj,
                'circle': circle,
                'net_size': clf.net.sizes,
                'mini_batch_size': mini_batch_size,
                'num_epochs': num_epochs,
                'early_stopping': early_stopping,
                'tol': tol,
                'num_iter_stopping': num_iter_stopping,
                'solver': solver,
                'alpha': alpha,
                'fname': 'radon'})
    del cfg['rng_st0'], cfg['rng']
    basename = helpers.get_basename(data_name)
    common.save_data(rec, cfg, basename=basename)
