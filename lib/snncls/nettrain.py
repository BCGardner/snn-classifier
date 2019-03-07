#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Jun 2017

@author: BG

Base class for network training.
"""

from __future__ import division

from argparse import Namespace
import numpy as np

from snncls import network, learnwindow
from snncls.parameters import ParamSet


class NetworkTraining(object):
    """
    Abstract network training class. Expected to be compatible with any output
    cost function.
    """
    def __init__(self, sizes, param, LearnWindow=learnwindow.PSPWindow,
                 Network=network.Network, **kwargs):
        """
        Set network learning parameters and learning window.
        Contains spiking neural network with no conduction delays as default.
        Log average firing rate per epoch by default.

        Inputs
        ------
        sizes : list
            Num. neurons in [input, hidden, output layers].
        param : container
            Using:
                eta : learning rate.
                cell_params : dict of neuron parameters.
                w_init : 2-tuple containing initial weight range.
        LearnWindow: class
            Define pre / post spiking correlation window for weight updates.
        """
        # Contain network
        self.net = Network(sizes, param, **kwargs)
        # Learning rule specific / common parameters
        self.rng = param.rng
        self.eta = param.net['eta0'] / self.sizes[0]
        self.corr = LearnWindow(param)

    def SGD(self, data_tr, epochs, mini_batch_size, data_te=None,
            report=True, epochs_r=1, debug=False, early_stopping=False,
            tol=1e-5, solver='sgd', **kwargs):
        """
        Stochastic gradient descent - present training data in mini batches,
        and updates network weights.

        Inputs
        ------
        data_tr : list
            Training data: list of 2-tuples (X, y), where X is input data,
            and y a one-hot encoded class label. X is either a list of input
            spike trains (spike pattern) or an array of predetermined PSPs
            evoked by the network's input layer.
        epochs : int
            Num. training epochs (maximum with early_stopping).
        mini_batch_size : int
            Patterns per mini batch (must be factor of epochs).
        data_te : list, optional
            Test data: list of 2-tuples (X, y), same format as data_tr.
        report : bool
            Print loss every epochs_r.
        epochs_r : int
            Num. epochs per report.
        debug : bool
            Additional readings: weights.
        early_stopping : bool, requires data_te
            Early stopping, if data_te used.
        tol : float, requires early_stopping and data_te
            Tolerance for optimisation. If loss on data_te is not improving by
            at least tol (relative amount, proportional to initial te_loss)
            for two consecutive epochs, then convergence is considered and
            learning stops.
        solver : str
            Choices: 'sgd' (default), 'rmsprop', 'adam'.

        Outputs
        -------
        tr_loss : array
            Loss on training data, per epoch.
        te_loss : array, optional
            Loss on test data.
        w : list, optional
            Weight arrays in each layer, per epoch.
        gas : list, optional
            Moving average of squared gradients (gas).
        """
        # === Inititalise =================================================== #

        # Prepare data
        data_tr = list(data_tr)  # Copy list of data samples
        tr_cases = len(data_tr)
        # Learning rate schedule
        assert solver in ['sgd', 'rmsprop', 'adam']
        svr_prms = ParamSet({})
        weights = self.net.get_weights()
        if solver == 'rmsprop':
            # grad_w_av_sq : EMA of squared gradients
            # decay : decay rate; alpha : lrate; epsilon : numerical stability
            svr_prms.update({'grad_w_av_sq': [np.zeros(w.shape) for
                                              w in weights],
                             'decay': 0.9,
                             'alpha': 0.1,
                             'epsilon': 1e-8,
                             'warmstart': False})
        elif solver == 'adam':
            # m : 1st moments (mean / EMA of gradients)
            # v : 2nd moments (uncentered variance / EMA of squared gradients)
            # betas : decay terms; alpha : lrate; epsilon : numerical stability
            svr_prms.update({'m': [np.zeros(w.shape) for
                                   w in weights],
                             'v': [np.zeros(w.shape) for
                                   w in weights],
                             'betas': (0.9, 0.999),
                             'alpha': 0.1,
                             'epsilon': 1e-8})
        svr_prms.overwrite(**kwargs)
        # Recordings
        rec = {'tr_loss': np.full(epochs, np.nan)}
        if debug:
            rec['w'] = [np.full((epochs,) + w.shape, np.nan) for w in weights]
#            if solver == 'rmsprop':
#                rec['gas'] = [np.full((epochs,) + w.shape, np.nan)
#                              for w in self.w]
        if data_te is not None:
            rec['te_loss'] = np.full(epochs, np.nan)
            # Initial test loss for early stopping
            if early_stopping:
                te_loss0 = self.loss(data_te)
#                print "Test loss\t\t{0:.3f}".format(te_loss0)

        # === Training ====================================================== #

        # Warm start for rmsprop
        if solver == 'rmsprop' and svr_prms['warmstart']:
            # Ensure stratified samples
            self.rng.shuffle(data_tr)
            mini_batch = data_tr[:mini_batch_size]
            # Estimate initial squared gradients
            grad_w_acc = self.grad_accum(mini_batch)
            svr_prms['grad_w_av_sq'] = [dC**2 for dC in grad_w_acc]
        for j in xrange(epochs):
            # Partition data into mini batches
            self.rng.shuffle(data_tr)
            mini_batches = [data_tr[k:k+mini_batch_size]
                            for k in xrange(0, tr_cases, mini_batch_size)]
            for idx, mini_batch in enumerate(mini_batches):
                iters = idx + j * len(mini_batches)
                self.update_mini_batch(mini_batch, solver=solver,
                                       svr_prms=svr_prms, iters=iters)
            # Debugging recordings
            if debug:
                weights = self.net.get_weights()
                for l in xrange(self.num_layers-1):
                    rec['w'][l][j] = weights[l]
#                if solver == 'rmsprop':
#                    for l in xrange(self.num_layers-1):
#                        rec['gas'][l][j] = svr_prms['grad_w_av_sq'][l].copy()
            # Determine loss on training / test data
            rec['tr_loss'][j] = self.loss(data_tr)
            if data_te is not None:
                rec['te_loss'][j] = self.loss(data_te)
                # Early stopping
                if early_stopping and j > 1:
                    te_losses = rec['te_loss'][j-2:j+1]
                    delta_losses = np.diff(te_losses) / te_loss0
                    cond = (delta_losses < 0.) & (np.abs(delta_losses) > tol)
                    if not cond.any():
                        print "Stop Epoch {0}\t\t{1:.3f}".format(j, rec['tr_loss'][j])
                        break
            # Report training / test error rates per epoch
            if report and not j % epochs_r:
                print "Epoch: {0}\t\t{1:.3f}".format(j, rec['tr_loss'][j])
#                print "Epoch {0}:\ttrain:\t{1:.3f}".format(
#                        j + 1, self.evaluate(data_tr))
#                if data_te is not None:
#                    print "\t\ttest:\t{0} / {1}".format(
#                        self.evaluate(data_te), te_cases)
        return rec

    def update_mini_batch(self, mini_batch, solver='sgd', svr_prms={},
                          iters=None):
        """
        Update network weights based on a mini batch of training data.

        Inputs
        ------
        mini_batch : list
            Training data: list of 2-tuples (X, y), where X is input data,
            and y a one-hot encoded class label.
        solver : str, optional
            Choices: 'sgd' (default), 'rmsprop', 'adam'.
        svr_prms : dict, optional
            Contains solver-specific parameters.
        iters : int, optional
            Iterations completed.
        """
        # Collect solver parameters
        prms = Namespace(**svr_prms)
        # Accumulated gradients of cost function w.r.t. weights
        grad_w_acc = self.grad_accum(mini_batch)
        # Apply accumulated weight changes
        weights = self.net.get_weights()
        if solver == 'sgd':
            weights = [w - self.eta / len(mini_batch) * dC
                       for w, dC in zip(weights, grad_w_acc)]
        elif solver == 'rmsprop':
            for idx, dC in enumerate(grad_w_acc):
                prms.grad_w_av_sq[idx] = prms.decay * \
                    prms.grad_w_av_sq[idx] + (1. - prms.decay) * dC**2
                weights[idx] -= \
                    prms.alpha / np.sqrt(prms.grad_w_av_sq[idx] +
                                         prms.epsilon) * dC
        elif solver == 'adam':
            iters += 1  # Update for next iteration
            for idx, dC in enumerate(grad_w_acc):
                prms.m[idx] = prms.betas[0] * prms.m[idx] + \
                    (1. - prms.betas[0]) * dC
                prms.v[idx] = prms.betas[1] * prms.v[idx] + \
                    (1. - prms.betas[1]) * dC**2
                m_unbias = prms.m[idx] / (1. - prms.betas[0]**iters)
                v_unbias = prms.v[idx] / (1. - prms.betas[1]**iters)
                weights[idx] -= (prms.alpha * m_unbias) / \
                    (np.sqrt(v_unbias) + prms.epsilon)
        self.net.set_weights(weights)

    def grad_accum(self, mini_batch):
        """
        Apply backprop to each sample in a mini_batch of data and return list
        of accumulated weight gradients for each layer.
        """
        weights = self.net.get_weights()
        grad_w_acc = [np.zeros(w.shape) for w in weights]
        for X, y in mini_batch:
            grad_w = self.backprop(X, y)
            grad_w_acc = [dC + dC_x for dC, dC_x in
                          zip(grad_w_acc, grad_w)]
        return grad_w_acc

    def backprop(self, X, y):
        raise NotImplementedError

    def evaluate(self, data):
        """
        Evaluate performace of model on data [(X1, y1), ...] as percentage of
        errors.
        """
        correct = 0.0
        num_cases = len(data)
        for d_case in data:
            X, y = d_case
            p = self.predict(X)
            if p == np.argmax(y):
                correct += 1.0
        return (1.0 - correct / num_cases) * 100.0

    def predict(self, X):
        raise NotImplementedError

    def loss(self, data):
        raise NotImplementedError
