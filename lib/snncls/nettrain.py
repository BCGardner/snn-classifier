#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Jun 2017

@author: BG

Base class for network training and data classification.
"""

from __future__ import division

import numpy as np

from snncls import network, learnwindow
import snncls.solver


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
        self.eta = param.net['eta0'] / sizes[0]
        self.corr = LearnWindow(param)

    def SGD(self, data_tr, epochs, mini_batch_size, data_te=None,
            report=True, epochs_r=1, debug=False, early_stopping=False,
            tol=1e-2, num_iter_stopping=5, solver='sgd', warmstart=False,
            **kwargs):
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
            Print loss every epochs_r[, early stopping].
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
        num_iter_stopping : int
            Number of epochs checked for early stopping. If the test loss
            doesn't improve by at least tol on one of these consecutive epochs,
            then network training is stopped.
        solver : str
            Choices: 'sgd' (default), 'rmsprop', 'adam'.
        warmstart : bool
            Optionally initialise solver state on subset of training data.

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

        # Learning schedules
        svr_dict = {'sgd': snncls.solver.ConstLR,
                    'rmsprop': snncls.solver.RMSProp,
                    'adam': snncls.solver.Adam}
        weights = self.net.get_weights()
        # TODO make eta an svr prm
        svr = svr_dict[solver](weights, eta=self.eta, **kwargs)
        if warmstart:
            self.rng.shuffle(data_tr)
            mini_batch = data_tr[:mini_batch_size]
            grad_w_acc = self.grad_accum(mini_batch)
            svr.warmup(grad_w_acc)

        # Recordings
        rec = {'tr_loss': np.full(epochs, np.nan)}
        if debug:
            rec['w'] = [np.full((epochs,) + w.shape, np.nan) for w in weights]
#            if solver == 'rmsprop':
#                rec['gas'] = [np.full((epochs,) + w.shape, np.nan)
#                              for w in self.w]
        if data_te is not None:
            rec['te_loss'] = np.full(epochs, np.nan)

        # === Training ====================================================== #

        for j in xrange(epochs):
            # Partition data into mini batches
            self.rng.shuffle(data_tr)
            mini_batches = [data_tr[k:k+mini_batch_size]
                            for k in xrange(0, tr_cases, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, solver=svr)
            # Debugging recordings
            if debug:
                weights = self.net.get_weights()
                for l, w in enumerate(weights):
                    rec['w'][l][j] = w
#                if solver == 'rmsprop':
#                    for l in xrange(self.num_layers-1):
#                        rec['gas'][l][j] = svr_prms['grad_w_av_sq'][l].copy()
            # Determine loss on training / test data
            rec['tr_loss'][j] = self.loss(data_tr)
            if data_te is not None:
                rec['te_loss'][j] = self.loss(data_te)
                # Early stopping
                if early_stopping:
                    if stoptraining(rec['te_loss'][:j+1], num_iter_stopping,
                                    tol, report=report):
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

    def SGD_iter(self, data_tr, num_iter, mini_batch_size=150, data_te=None,
                 report=True, num_iter_r=1, debug=False, early_stopping=False,
                 tol=1e-2, num_iter_stopping=5, solver='sgd', warmstart=False,
                 **kwargs):
        """
        Stochastic gradient descent - present training data in mini batches and
        update network weights. This method records values after each iteration
        / mini_batch presented to the network. Early stopping is determined
        based on recorded values.

        Inputs
        ------
        num_iter : int
            Maximum num. iterations network trained on data.
        num_iter_r : int
            Num. iters to each recording.
        num_iter_stopping : int
            Num. recorded iters for stop condition.
        """
        # === Rand. select subset of data  ================================== #

        def rand_subset(data, num_cases, size=mini_batch_size):
            idxs = self.rng.choice(num_cases, size=size,
                                   replace=False)
            return data[idxs]

        # === Inititalise =================================================== #

        # Prepare data
        data_tr = np.array(data_tr)
        tr_cases = len(data_tr)
        # Estimates based on subsampled training data
        mini_batch = rand_subset(data_tr, tr_cases, size=mini_batch_size)
        # Initial weights
        weights = self.net.get_weights()

        # Setup recordings containers
        num_r = num_iter // num_iter_r + 1  # Num. recorded iters. (incl. init)
        rec = {'tr_loss': np.full(num_r, np.nan)}
        if debug:
            rec['w'] = [np.full((num_r,) + w.shape, np.nan)
                        for w in weights]
        if data_te is not None:
            rec['te_loss'] = np.full(num_r, np.nan)
        # Estimate initial losses
        idx_r = 0
        rec['tr_loss'][idx_r] = self.loss(mini_batch)
        if data_te is not None:
            rec['te_loss'][idx_r] = self.loss(data_te)
        # Initial weights
        if debug:
            for l, w in enumerate(weights):
                rec['w'][l][idx_r] = w

        # Learning schedules
        svr_dict = {'sgd': snncls.solver.ConstLR,
                    'rmsprop': snncls.solver.RMSProp,
                    'adam': snncls.solver.Adam}
        svr = svr_dict[solver](weights, eta=self.eta, **kwargs)
        if warmstart:
            grad_w_acc = self.grad_accum(mini_batch)
            svr.warmup(grad_w_acc)

        # === Training ====================================================== #

        for j in xrange(num_iter):
            # Randomly select subset of training data without replacement
            mini_batch = rand_subset(data_tr, tr_cases, size=mini_batch_size)
            self.update_mini_batch(mini_batch, solver=svr)
            # Report / record values
            if not (j + 1) % num_iter_r:
                idx_r += 1
                # Record iterative weight updates
                if debug:
                    weights = self.net.get_weights()
                    for l, w in enumerate(weights):
                        rec['w'][l][idx_r] = w
                # Determine loss on training / test data
                rec['tr_loss'][idx_r] = self.loss(mini_batch)
                if data_te is not None:
                    rec['te_loss'][idx_r] = self.loss(data_te)
                    # Early stopping
                    if early_stopping:
                        if stoptraining(rec['te_loss'][:idx_r+1],
                                        num_iter_stopping, tol, report=report):
                            break
                # Report training / test error rates per epoch
                if report:
                    print "Iters: {0}\t\t{1:.3f}".format(j+1,
                                                         rec['tr_loss'][idx_r])
        return rec

    def update_mini_batch(self, mini_batch, solver):
        """
        Update network weights based on a mini batch of training data.

        Inputs
        ------
        mini_batch : list or array
            Training data: list of 2-tuples (X, y), where X is input data,
            and y a one-hot encoded class label.
        solver : object
            Learning schedule used to compute weight changes from grad_w.
        """
        # Accumulated gradients of cost function w.r.t. weights
        grad_w_acc = self.grad_accum(mini_batch)
        # Weight changes per layer
        delta_weights = solver.weight_changes(grad_w_acc,
                                              num_samples=len(mini_batch))
        # Apply accumulated weight changes
        weights = [w + dw for w, dw in zip(self.net.get_weights(),
                                           delta_weights)]
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


def stoptraining(te_losses, num_iter_stopping=5, tol=1E-2, report=False):
    """
    Determine if training should be stopped based on validation losses.
    Training continues if any loss change is negative and sufficiently
    large, over the num. values checked.

    Inputs
    ------
    te_losses : array, shape (iters_tr,)
        Historic loss values on validation set, up to latest recording.
    num_iter_stopping : int
        Num. loss changes checked for stopping criterion.
    tol : float
        Stopping tolerance.
    report : bool
        Report if stop condition met.

    Output
    ------
    return : bool
        Stop signal.
    """
    num_iters = len(te_losses)  # Number of iters. net. has been trained for
    if num_iters > num_iter_stopping:
        losses_stopping = te_losses[-(num_iter_stopping+1):]
        delta_losses = np.diff(losses_stopping)
        # Check if any conditions for continuing are satisfied
        cond = (delta_losses < 0.) & (np.abs(delta_losses) > tol)
        if not cond.any():
            if report:
                print("Stop after {0} iterations\t\t{1:.3f}".format(num_iters,
                      te_losses[-1]))
            return True
    return False
