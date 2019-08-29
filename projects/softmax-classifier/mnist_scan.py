#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
MNIST: encoded using scanlines as sequences of precisely-timed input spikes, to
provide dimensionality reduction for constrained SNN sizes.
This code implements a multilayer SNN, with the option for subconnections (i.e.
multiple connections between connected pre- and postsynaptic neurons).
Subconnections differ in their delays, and work to integrate presynaptic spikes
occuring on different timescales.

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

from snncls import values, preprocess, helpers, metric, plotter
from snncls.network import MultilayerSRM, MultilayerSRMSub
from snncls.supervised import SoftmaxClf
from lib import common


# %% === Pattern stats ====================================================== #

# Select network type: 0 network, 1 network with delayed subconnections
net_id = 1

# Specify dataset
data_name = 'data/mnist3750_32s.pkl.gz'
shuffle = True
test_size = 0.2

# Load dataset
X, y = helpers.load_data(data_name)
num_inputs = len(X[0])  # Num. input features
num_classes = len(np.unique(y))

# %% === Network stats ====================================================== #

# Simulation parameters
num_nrns = 40
opt = {'seed': None,
       'w_bounds': (-8., 8.),  # w_{hi, lo} ~ 240 / num_input_spikes
       'neurons_f': 1,
       'w_h_init': (0., 40. / num_inputs),  # 1 spike: ~ 40 / num_inputs
       'w_o_init': (0., 32. / num_nrns),     # 1 spike: ~ 32 / num_nrns
       'cpd_scale': 4.,
       'l2_pen': 1E-4,
       'rate_pow': 2}
# Batch learning
mini_batch_size = 150
num_epochs = 40    # Num. training epochs = 40: ~ 25 mins on i5 6600K
# Early-stopping
early_stopping = False
tol = 1E-2
num_iter_stopping = 5
# Learning schedule
solver = 'rmsprop'
alpha = 0.1
# Optional
predet_psps = True
debug = False
plot = True
save_data = False

# Network class
opt_net = {}
if net_id == 0:
    NetType = MultilayerSRM
elif net_id == 1:
    # Network with multiple feedforward connections (or subconns) between nrns
    opt.update(duration=60., w_h_init=(0., 40. / num_inputs))
    # num_subs : [num_hidden_subconns, num_output_subconns]
    # conns_fr : fraction of eligible hidden subconns, randomly selected.
    # For 8 hidden subconns, conns_fr=0.125, each hidden nrn has one connection
    # with each input node, with a random delay between 1 and 10 ms. All other
    # hidden subconns have their weight values clamped to zero.
    opt_net.update(num_subs=[8, 1], max_delay=10., conns_fr=0.125)
    NetType = MultilayerSRMSub

# Top-level parameters container
prms = values.LatencyParam(**opt)

# %% === Preprocessing  ===================================================== #

# Split dataset into training and test sets (4:1 split if test_size=0.2)
if shuffle:
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size,
                                              shuffle=True, stratify=y,
                                              random_state=prms.rng)
else:
    # First fraction are for training, remaining for test.
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size,
                                              shuffle=False)

# Prepare data[, transform data into predetermined psps]
data_tr = preprocess.transform_spikes(X_tr, y_tr, prms, receptor=None,
                                      num_classes=num_classes,
                                      return_psps=predet_psps)
data_te = preprocess.transform_spikes(X_te, y_te, prms, receptor=None,
                                      num_classes=num_classes,
                                      return_psps=predet_psps)

# %% === Classifier ========================================================= #

clf = SoftmaxClf(sizes=[num_inputs, num_nrns, num_classes], param=prms,
                 Network=NetType, **opt_net)

# %% === Training =========================================================== #

# Initialise recorder
if debug:
    rec = {'w_init': [w for w in clf.net.get_weights()],
           'prms': vars(prms).copy()}
else:
    rec = {}

# SGD
t1 = time.time()
rec.update(clf.SGD(data_tr, num_epochs, mini_batch_size, data_te=data_te,
                   epochs_r=1, debug=debug, solver=solver, alpha=alpha,
                   early_stopping=early_stopping, tol=tol,
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
    cfg.update({'net_id': net_id,
                'net_size': clf.net.sizes,
                'mini_batch_size': mini_batch_size,
                'num_epochs': num_epochs,
                'early_stopping': early_stopping,
                'tol': tol,
                'num_iter_stopping': num_iter_stopping,
                'solver': solver,
                'alpha': alpha,
                'fname': 'scan'})
    del cfg['rng_st0'], cfg['rng']
    basename = helpers.get_basename(data_name)
    common.save_data(rec, cfg, basename=basename)
