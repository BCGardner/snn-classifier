#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Models for transforming various feature types into spike latencies for network
input.

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

from __future__ import division

import numpy as np

from snncls import scanline


class BaseModel(object):
    """
    Model template for transforming features into sets of spike latencies.
    """
    def __init__(self):
        """
        Declares and initialises common transformation parameters.
        """
        self._num_fitted_samples = 0
        self._num_fitted_features = 0
        self._X_min = np.nan
        self._X_max = np.nan

    def fit_transform(self, X):
        """
        Fit the model to some data, and return its transformation.

        Inputs
        ------
        X : array, shape (num_samples, num_features)
            Input data.

        Output
        ------
        spike trains : array, shape (num_samples, num_encoding_nrns)
            Set of spike latencies.
        """
        self.fit(X)
        return self.transform(X)

    def transform(self, X):
        """
        Transforms data samples into array of spike latencies, given a previous
        fit on some data.
        X (num_samples, num_features) -> latencies (num_samples, num_enc_nrns),
        where num_enc_nrns are the number of neurons encoding each feature.
        """
        raise NotImplementedError

    def fit(self, X):
        """
        Fits this model to example data.

        Inputs
        ------
        X : array, shape (num_samples, num_features)
            Data set.
        """
        raise NotImplementedError

    def update_fit(self, X, X_bounds=None):
        """
        Updates model fit to presented data.

        Inputs
        ------
        X : array, shape (num_samples, num_features)
            Data set.
        X_bounds : 2-tuple, optional
            Specify the data range: (X_min, X_max), in advance.
        """
        # num_samples, num_features
        m, n = X.shape
        # Update fitted parameters
        self._num_fitted_samples = m
        self._num_fitted_features = n
        if X_bounds is None:
            self._X_min, self._X_max = np.min(X), np.max(X)
        else:
            self._X_min, self._X_max = X_bounds

    def check_fit(self, X):
        """
        Check if model has been fitted to this data form.
        """
        if self._num_fitted_samples == 0:
            raise Exception('This model has not been fitted yet.')
        assert X.shape[1] == self._num_fitted_features

    def get_params(self):
        return vars(self)


class ConvFilter(BaseModel):
    """
    Considers each data sample as an array of feature-sequences, where each
    sequences consists of real values that are typically in the range [0, 9).
    This model transforms arrays of feature-sequences into spike latencies via
    convolutional filtering.
    """
    def __init__(self, neurons_f=12, beta=1.5, curr_max=20., tau_m=10., cm=2.5,
                 theta=15., ltcy_max=9.):
        """
        Sets transformation parameters.

        Inputs
        ------
        neurons_f : int
            Number of neurons encoding arrays of feature sequences.
        beta : float
            Width of each Gaussian receptive field.
        curr_max : float
            Maximum injected current per input.
        tau_m : float
            LIF membrane time constant.
        cm : float
            LIF membrane capacitance.
        ltcy_max : float
            Maximum latency.
        """
        super(ConvFilter, self).__init__()
        # Receptive fields
        self.neurons_f = neurons_f
        self.beta = beta
        # Encoding neurons
        self.curr_max = curr_max
        self.tau_m = tau_m
        self.R = tau_m / cm  # LIF membrane resistance
        self.theta = theta  # LIF firing threshold
        self.ltcy_max = ltcy_max
        # Minimum driving voltage for target output latency range
        self.volts_min = theta / (1. - np.exp(-ltcy_max / tau_m))

    def transform(self, X):
        """
        Transform input data X (num_samples, num_features) into input
        activations and then spike latencies.
        """
        self.check_fit(X)
        # Intensity of responses
        m, n = X.shape
        activations = np.zeros((m, self.neurons_f * n))
        for idx, x in enumerate(X):
            activations[idx, :] = \
                np.hstack([np.sum(gaussian(seq[:, np.newaxis],
                                           self.centers, self.sigma), 0)
                           for seq in x])
        activations *= self.curr_max
        # Convert to spike latencies
        volts = self.R * activations
        ltcys = np.full(activations.shape, np.inf)
        ltcy_mask = volts > self.volts_min
        ltcys[ltcy_mask] = self.tau_m * np.log(volts[ltcy_mask] /
                                               (volts[ltcy_mask] - self.theta))
        return ltcys

    def fit(self, X, X_min=0., X_max=9.):
        """
        Sets up this model for example data. Each feature is a sequence of
        float values.

        Inputs
        ------
        X : array, shape (num_samples, num_features)
            Data set.
        X_min : float
            Minimum feature value.
        X_max : float
            Maximum feature value.
        """
        # num_samples, num_features
        m, n = X.shape
        # Center receptive fields
        indices = np.arange(1, self.neurons_f+1)
        self.centers = X_min + (2*indices - 3) / 2. * \
            (X_max - X_min) / (self.neurons_f - 2)
        self.sigma = 1. / self.beta * (X_max - X_min) / (self.neurons_f - 2)
        # Update fitted parameters
        self.update_fit(X, (X_min, X_max))


class Integrator(BaseModel):
    """
    Converts real-valued feature vectors into spike latencies based on constant
    current stimulus driving each LIF neuron. Assume features shares the same
    data range. One-one association: input features -> encoders.
    """
    def __init__(self, curr_max=20., tau_m=10., cm=2.5, theta=15.,
                 ltcy_max=9.):
        """
        Sets transformation parameters.

        Inputs
        ------
        curr_max : float
            Maximum injected current per input.
        tau_m : float
            LIF membrane time constant.
        cm : float
            LIF membrane capacitance.
        ltcy_max : float
            Maximum latency.
        """
        super(Integrator, self).__init__()
        self.curr_max = curr_max
        self.tau_m = tau_m
        self.R = tau_m / cm  # LIF membrane resistance
        self.theta = theta  # LIF firing threshold
        self.ltcy_max = ltcy_max
        # Minimum driving voltage for target output latency range
        self.volts_min = theta / (1. - np.exp(-ltcy_max / tau_m))

    def transform(self, X):
        """
        Transforms data samples into array of spike latencies, up to 9 ms,
        given a previous fit on training data. Values are clipped to fitted
        data range.
        X (num_samples, num_features) -> latencies (num_samples, num_enc_nrns)
        """
        self.check_fit(X)
        # Intensity of responses
        m, n = X.shape
        ltcys = np.full((m, n), np.inf)
        # Clip and normalise data
        X_ = np.clip(X, self._X_min, self._X_max)
        X_ /= self._X_max
        # Convert to constant current stimuli
        X_ *= self.curr_max
        volts = self.R * X_  # Driving voltage per input
        ltcy_mask = volts > self.volts_min
        ltcys[ltcy_mask] = self.tau_m * np.log(volts[ltcy_mask] /
                                               (volts[ltcy_mask] - self.theta))
        return ltcys

    def fit(self, X):
        """
        Fits this model to example training data, assuming all features share
        the same data range.

        Inputs
        ------
        X : array, shape (num_samples, num_features)
            Training set.
        """
        self.update_fit(X)


class ReceptiveFields(BaseModel):
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
        super(ReceptiveFields, self).__init__()
        self.neurons_f = neurons_f
        self.beta = beta

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
        self.check_fit(X)
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

    def fit(self, X):
        """
        Fits this model to example data, by finding
        each individual feature range (by column) and centering receptive
        fields.

        Inputs
        ------
        X : array, shape (num_samples, num_features)
            Data set.
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
        self.update_fit(X, (X_min, X_max))


class ScanLines(BaseModel):
    """
    Considers each data sample as a flat, 1D array of image pixels with square
    boundaries. Images consist of grayscale real values that are e.g. in the
    range [0, 256). This model transforms images into spike trains via
    scanline encoding.
    """
    def __init__(self, num_scans=5, duration=9., dt=0.1, distr='loihi',
                 scale=0.25, scantype='lif', rng=None, **kwargs):
        """
        Sets transformation parameters.

        Inputs
        ------
        num_scans : int
            Number of scanline-encoders.
        duration : float
            Scan duration (ms).
        dt : float
            Scan step size (ms).
        distr : str
            Scanline distribution: {'loihi', 'edges', 'ctr'}.
        scale : float
            Scale parameter used for line distribution.
        scantype : str
            Scanner type: {'lif', 'izh'}.
        rng : int or RandomState object
            Seed or random number generator.
        """
        super(ScanLines, self).__init__()
        if isinstance(rng, np.random.RandomState):
            self.rng = rng
        else:
            self.rng = np.random.RandomState(rng)
        # Scanlines
        self.num_scans = num_scans
        self.duration = duration
        self.dt = dt
        self._times = np.arange(0., duration, dt)
        # Distr. type
        if distr in ['loihi', 'edges', 'ctr']:
            self.distr = distr
        else:
            raise TypeError('Invalid scanline distribution.')
        if distr == 'loihi':
            assert num_scans == 5
        self.scale = scale
        # Scanline-encoder / neuron setup
        if scantype == 'lif':
            NrnType = scanline.neuron.LIF
        elif scantype == 'izh':
            NrnType = scanline.neuron.Izhikevich
        else:
            raise TypeError('Invalid scanner type.')
        self._nrns = []
        for i in xrange(self.num_scans):
            self._nrns.append(NrnType(dt, **kwargs))

    def transform(self, X):
        """
        Scan input data X (num_samples, num_features): transforming images into
        spike trains.
        """
        self.check_fit(X)
        # Normalise data
        X_ = (X - self._X_min) / (self._X_max - self._X_min)
        # List of spike patterns
        spike_trains = [[np.array([]) for i in xrange(self.num_scans)]
                        for j in xrange(len(X_))]
        # Scan each sample
        for x, spike_train in zip(X_, spike_trains):
            img = x.reshape(self.bounds)
            # Reset scanners and associated neurons
            for scnr, nrn in zip(self._scanners, self._nrns):
                scnr.reset()
                nrn.reset()
            for t in self._times:
                for idx, scnr in enumerate(self._scanners):
                    # Read
                    stimulus = scnr.read(img)
                    # Update
                    fired = self._nrns[idx].update(stimulus)
                    scnr.translate(self.dt)
                    # Record nrn
                    if fired:
                        spike_time = np.around(t, decimals=1)
                        spike_train[idx] = \
                            np.append(spike_train[idx], spike_time)
        return spike_trains

    def fit(self, X, xlim=None):
        """
        Sets up this model for example data. Each feature is a sequence of
        float values.

        Inputs
        ------
        X : array, shape (num_samples, num_features)
            Data set.
        xlim : 2-tuple, optional
            Feature value range: if None then inferred from X.
        """
        # Data bounds
        self.bounds = (int(np.sqrt(len(X[0]))),) * 2
        if xlim is None:
            xlim = np.min(X), np.max(X)
        # Scanline Eqs.
        if self.distr == 'loihi':
            line_eqs = scanline.linegen.loihi_eqs
        elif self.distr == 'edges':
            line_eqs = scanline.linegen.generate_eqs(self.num_scans,
                                                     bounds=self.bounds,
                                                     scale=self.scale,
                                                     rng=self.rng)
        elif self.distr == 'ctr':
            line_eqs = scanline.linegen.generate_eqs_ctr(self.num_scans,
                                                         bounds=self.bounds,
                                                         scale=self.scale,
                                                         rng=self.rng)
        # Setup scanners
        self._scanners = []
        for eq in line_eqs:
            self._scanners.append(scanline.scanner.Scanner(eq, self.bounds,
                                                           self.duration))
        # Update fitted parameters
        self.update_fit(X, xlim)


def gaussian(x, mu, sigma):
    """
    Gaussian function for firing intensity.
    """
    return np.exp(-0.5 * ((x - mu) / sigma)**2)
