#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Jun 2017

@author: BG

Base class for network training.
"""

from __future__ import division

import numpy as np

from snncls import network, learnwindow


class NetworkTraining(network.Network):
    """
    Abstract network training class.
    """
    def __init__(self, sizes, param, LearnWindow=learnwindow.PSPWindow,
                 **kwargs):
        """
        Set network learning parameters and learning window.
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
        super(NetworkTraining, self).__init__(sizes, param, **kwargs)
        self.eta = param.net['eta0'] / self.sizes[0]
        self.corr = LearnWindow(param)

    def SGD(self, tr_data, epochs, mini_batch_size, te_data=None,
            report=True, debug=False, early_stopping=False, tol=1e-5):
        """
        Stochastic gradient descent - present training data in mini batches,
        and updates network weights.

        Inputs
        ------
        tr_data : list
            Training data: list of 2-tuples (X, Y), where X is input data,
            and Y a one-hot encoded class label. X is a 2-tuple, containing
            predetermined PSP's evoked by input layer, and the list of
            associated input spike times of size <num_inputs>.
        epochs : int
            Num. training epochs (maximum with early_stopping).
        mini_batch_size : int
            Patterns per mini batch (must be factor of epochs).
        te_data : list, optional
            Test data: list of 2-tuples (X, Y).
        report : bool
            Print num. epochs.
        debug : bool
            Additional readings: weights.
        early_stopping : bool, requires te_data
            Early stopping, if te_data used.
        tol : float, requires early_stopping and te_data
            Tolerance for optimisation. If loss on te_data is not improving by
            at least tol (relative amount, proportional to initial te_loss)
            for two consecutive epochs, then convergence is considered and
            learning stops.

        Outputs
        -------
        tr_loss : array
            Loss on training data, per epoch.
        te_loss : array, optional
            Loss on test data.
        w : list, optional
            Weight arrays in each layer, per epoch.
        """
        tr_data = list(tr_data)  # Copy list of data samples
        tr_cases = len(tr_data)
        # assert tr_cases % mini_batch_size == 0
        # Preallocate
        rec = {'tr_loss': np.full(epochs, np.nan)}
        if debug:
            rec['w'] = [np.full((epochs,) + w.shape, np.nan) for w in self.w]
        if te_data is not None:
            rec['te_loss'] = np.full(epochs, np.nan)
            # Initial test loss for early stopping
            if early_stopping:
                te_loss0 = self.loss(te_data)
#                print "Test loss\t\t{0:.3f}".format(te_loss0)
        # Training
        for j in xrange(epochs):
            # Partition data into mini batches
            self.rng.shuffle(tr_data)
            mini_batches = [tr_data[k:k+mini_batch_size]
                            for k in xrange(0, tr_cases, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch)
            # Debugging recordings
            if debug:
                for l in xrange(self.num_layers-1):
                    rec['w'][l][j] = self.w[l]
            # Determine loss on training / test data
            rec['tr_loss'][j] = self.loss(tr_data)
            if te_data is not None:
                rec['te_loss'][j] = self.loss(te_data)
                # Early stopping
                if early_stopping and j > 1:
                    te_losses = rec['te_loss'][j-2:j+1]
                    delta_losses = np.diff(te_losses) / te_loss0
                    cond = (delta_losses < 0.) & (np.abs(delta_losses) > tol)
                    if not cond.any():
                        print "Stop Epoch {0}\t\t{1:.3f}".format(j, rec['tr_loss'][j])
                        break
            # Report training / test error rates per epoch
            if report and not j % 5:
                print "Epoch {0}\t\t{1:.3f}".format(j, rec['tr_loss'][j])
#                print "Epoch {0}:\ttrain:\t{1:.3f}".format(
#                        j + 1, self.evaluate(tr_data))
#                if te_data is not None:
#                    print "\t\ttest:\t{0} / {1}".format(
#                        self.evaluate(te_data), te_cases)
        return rec

    def update_mini_batch(self, mini_batch):
        """
        Update network weights based on a mini batch of training data.

        Inputs
        ------
        mini_batch : list
            Training data - list of 2-tuples (X, y). Each X is a 2-tuple
            containing predetermined PSPs and their associated input spike
            times. Each y is a one-hot encoded class label.
        """
        # Gradients of cost function w.r.t. weights
        grad_w_acc = [np.zeros(w.shape) for w in self.w]
        num_cases = len(mini_batch)

        # Accumulate weight gradients from each training case in mini_batch
        for X, y in mini_batch:
            grad_w = self.backprop(X, y)
            grad_w_acc = [dw + dw_x for dw, dw_x in zip(grad_w_acc, grad_w)]

        # Apply accumulated weight changes
        self.w = [w - self.eta / num_cases * dw
                  for w, dw in zip(self.w, grad_w_acc)]
        self.clip_weights()

    def backprop(self, tr_input, tr_target):
        raise NotImplementedError

    def evaluate(self, data):
        """
        Evaluate performace of model on data [(X1, y1), ...] as percentage of
        errors.
        """
        correct = 0.0
        num_cases = len(data)
        for d_case in data:
            (x, _), y = d_case
            p = self.predict(x)
            if p == np.argmax(y):
                correct += 1.0
        return (1.0 - correct / num_cases) * 100.0

    def predict(self, psp_inputs):
        raise NotImplementedError

    def loss(self, data):
        raise NotImplementedError
