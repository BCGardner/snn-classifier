#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Jun 2017

@author: BG

Base class for network training.
"""

from __future__ import division

import numpy as np

from snncls import network, learning_window


class NetworkTraining(network.Network):
    """
    Abstract network training class.
    """
    def __init__(self, sizes, param, LearnWindow=learning_window.PSPWindow,
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

    def cross_validation(self, tr_data_f, epochs, va_losses=False,
                         report=True):
        """
        Stochastic gradient descent - trained using cross-validation on
        K-1 folds. Network evaluated on average of left out folds. Full batch
        learning.

        Inputs
        ------
        tr_data_f : list
            Training data, k folds.

        Outputs
        -------
        tr_losses : array
            Training loss of size: <num_folds> by <num_epochs>.
        va_errs : array
            Validation error after training of size: <num_folds>.
        va_loss : array, optional
            Validation loss of size: <num_folds> by <num_epochs>.
        """
        num_folds = len(tr_data_f)
        rec = {'tr_losses': np.full((num_folds, epochs), np.nan),
               'va_errs': np.full(num_folds, np.nan)}
        if va_losses:
            rec['va_losses'] = np.full((num_folds, epochs), np.nan)
        # Training-validation
        for k in xrange(num_folds):
            # Partition data into training and validation sets
            tr_data = []
            for l in xrange(num_folds):
                if l != k:
                    tr_data += tr_data_f[l]
            va_data = tr_data_f[k]
            # Train model on tr_data
            self.reset()
            if va_losses:
                rec_f = self.SGD(tr_data, epochs, len(tr_data),
                                 te_data=va_data, report=report)
                rec['va_losses'][k] = rec_f['te_loss']
            else:
                rec_f = self.SGD(tr_data, epochs, len(tr_data),
                                 report=report)
            rec['tr_losses'][k] = rec_f['tr_loss']
            # Validation error
            rec['va_errs'][k] = self.evaluate(va_data)
            if report:
                print "Folds: {0}".format(k + 1)
        return rec

    def SGD(self, tr_data, epochs, mini_batch_size, te_data=None,
            report=True, debug=False):
        """
        Stochastic gradient descent - present training data in mini batches,
        and updates network weights.

        Inputs
        ------
        tr_data : list
            Training data: list of 2-tuples (X, Y), where X is an array of
            predetermined PSPs at input layer and Y a one-hot target output
            vector.
        epochs : int
            Num. training epochs.
        mini_batch_size : int
            Patterns per mini batch (must be factor of epochs).
        te_data : list, optional
            Test data: list of 2-tuples (X, Y).
        report : bool
            Print num. epochs.
        debug : bool
            Additional readings: weights.

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
        # Training
        for j in xrange(epochs):
            # Partition data into mini batches
            self.rng.shuffle(tr_data)
            mini_batches = [tr_data[k:k+mini_batch_size]
                            for k in xrange(0, tr_cases, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch)
            # Determine loss on training / test data
            rec['tr_loss'][j] = self.loss(tr_data)
            if te_data is not None:
                rec['te_loss'][j] = self.loss(te_data)
            # Report training / test error rates per epoch
            if report and not (j + 1) % 1:
                print "Epoch {0}\t\t{1:.3f}".format(j + 1, rec['tr_loss'][j])
            if debug:
                for l in xrange(self.num_layers-1):
                    rec['w'][l][j] = self.w[l]
#                print "Epoch {0}:\ttrain:\t{1:.3f}".format(
#                        j + 1, self.evaluate(tr_data))
#                if te_data is not None:
#                    print "\t\ttest:\t{0} / {1}".format(
#                        self.evaluate(te_data), te_cases)
        return rec

    def update_mini_batch(self, mini_batch):
        """
        Update model parameters based on a mini batch of training data.

        Inputs
        ------
        mini_batch : list
            Training data: list of 2-tuples (X, y). X is of size:
                <n_inputs> by <n_iterations>, Y is <n_classes>.
        """
        # Gradients of cost function w.r.t. weights
        grad_w_acc = [np.zeros(w.shape) for w in self.w]
        num_cases = len(mini_batch)

        # Accumulate weight gradients from each training case in mini_batch
        for X, y in mini_batch:
            grad_w = self.backprop(X, y)
            grad_w_acc = [dw + dw_x for dw, dw_x in zip(grad_w_acc, grad_w)]

        # Apply accumulated weight changes (incl. scaling)
        self.w = [w - self.eta / num_cases * dw
                  for w, dw in zip(self.w, grad_w_acc)]
        # self.w = [w - self.eta / num_cases * (dw + self.l2_pen * w)
        #           for w, dw in zip(self.w, grad_w_acc)]

    def backprop(self, tr_input, tr_target):
        raise NotImplementedError

    def evaluate(self, data):
        """
        Evaluate performace of model on data [(x1, y1), ...] as percentage of
        errors.
        """
        correct = 0.0
        num_cases = len(data)
        for d_case in data:
            x, y = d_case
            p = self.predict(x)
            if p == np.argmax(y):
                correct += 1.0
        return (1.0 - correct / num_cases) * 100.0

    def predict(self, psp_inputs):
        raise NotImplementedError

    def loss(self, data):
        raise NotImplementedError
