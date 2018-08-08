#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Aug 2018

@author: BG

Preprocessing features for spike-based learning.
"""

from __future__ import division

import numpy as np


class ReceptiveFields(object):
    """
    Converts real-valued feature vectors into spike latencies based on
    Gaussian receptive fields.
    """
    def __init__(self, neurons_f=12, beta=1.5):
        """
        Sets transformation parameters.

        Inputs
        ------
        neurons_f : int
            Number of neurons encoding each feature.
        beta : float
            Width of each Gaussian receptive field.
        """
        self.neurons_f = neurons_f
        self.beta = beta
        self._num_fitted_samples = 0
        self._num_fitted_features = 0
        self._X_min = np.nan
        self._X_max = np.nan

    def transform(self, X):
        """
        Transforms data samples into array of spike latencies, using this
        previously fitted model. Encoding neurons fire with a delay of up to
        9 ms, and those insufficiently activated fire with infinite delay.

        Inputs
        ------
        X : array, shape (num_samples, num_features)
            Data set.

        Output
        ------
        latencies : array, shape (num_samples, num_encoding_nrns)
            Set of spike latencies.
        """
        if self._num_fitted_samples == 0:
            raise Exception('This model has not been fitted yet.')
        assert X.shape[1] == self._num_fitted_features
        # Intensity of responses
        m, n = X.shape
        latencies = np.empty((m, self.neurons_f * n))
        for f in xrange(n):
            latencies[:, self.neurons_f*f:self.neurons_f*(f + 1)] = \
                gaussian(X[:, [f]], self.centers[f, :], self.sigmas[f])
        # Convert to spike latencies
        latencies = 10. * (1 - latencies)
        latencies[latencies > 9.] = np.inf
        return latencies

    def fit_transform(self, X):
        """
        Fit the model to training data, and return its transformation.

        Inputs
        ------
        X : array, shape (num_samples, num_features)
            Training set.

        Output
        ------
        latencies : array, shape (num_samples, num_encoding_nrns)
            Set of spike latencies.
        """
        self.fit(X)
        return self.transform(X)

    def fit(self, X):
        """
        Fits this model to example training data, by  estimating
        each feature range and centering receptive fields.

        Inputs
        ------
        X : array, shape (num_samples, num_features)
            Training set.
        """
        # num_samples, num_features
        m, n = X.shape
        # Center receptive fields
        X_min = np.min(X, 0, keepdims=True).T
        X_max = np.max(X, 0, keepdims=True).T
        indices = np.arange(1, self.neurons_f+1)
        self.centers = X_min + (2*indices - 3) / 2. * \
            (X_max - X_min) / (self.neurons_f - 2)
        self.sigmas = 1. / self.beta * (X_max - X_min) / (self.neurons_f - 2)
        # Update fitted parameters
        self._num_fitted_samples = m
        self._num_fitted_features = n
        self._X_min, self._X_max = X_min, X_max

    def get_params(self):
        return vars(self)


def gaussian(x, mu, sigma):
    """
    Gaussian function for firing intensity.
    """
    return np.exp(-0.5 * ((x - mu) / sigma)**2)
