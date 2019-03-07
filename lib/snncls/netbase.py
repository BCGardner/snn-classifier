#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mar 2019

@author: BG

Base network class.
"""

from __future__ import division

import numpy as np

from snncls import escape_noise, preprocess


class NetBase(object):
    """
    Abstract class of spiking neural network with hidden neurons.
    Connectivity is feedforward with all-all connectivity.
    Subthreshold neuronal dynamics are governed by LIF model in all layers.
    """
    def __init__(self, sizes, param, weights=None,
                 EscapeRate=escape_noise.ExpRate):
        """
        Initialise common network parameters, constraints, cell parameters,
        optimisations.

        Inputs
        ------
        sizes : list
            Num. neurons in [input, hidden, output layers].
        param : container
            Using cell, pattern, net and common parameters.
        EscapeRate : class
            Hidden layer neuron firing density.
        """
        self.sizes = sizes  # No. neurons in each layer
        self.num_layers = len(sizes)  # Total num. layers (incl. input)
        self.dt = param.dt  # Sim. time elapsed per iter
        self.duration = param.pattern['duration']  # Default sim. duration
        self.rng = param.rng
        # Parameters
        self.cell_params = param.cell
        self.w_h_init = param.net['w_h_init']
        self.w_o_init = param.net['w_o_init']
        self.w_bounds = param.net['w_bounds']
        # Hidden neuron model
        self.neuron_h = EscapeRate(param.cell['theta'])
        # Optimisations
        self.decay_m = param.decay_m
        self.decay_s = param.decay_s
        # Look up tables for default sim. durations
        times = np.arange(0., self.duration, self.dt)
        self.lut = {'psp': preprocess.psp_reduce(times, np.array([0.]),
                                                 **self.cell_params),
                    'refr': preprocess.refr_reduce(times, np.array([0.]),
                                                   **self.cell_params)}

    def reset(self, rng_st=None, weights=None):
        raise NotImplementedError

    def simulate(self, stimulus, latency=False, return_psps=False,
                 debug=False):
        """
        Simulates network response to stimulus, returns list of spike trains
        for layers l > 0. Optionally returns PSPs evoked due to layers l < L,
        or debug recordings.
        """
        raise NotImplementedError

    def get_weights(self):
        """
        Returns a copy of the network weights as a list.
        """
        return [w.copy() for w in self.w]

    def set_weights(self, weights, assert_bounds=False):
        """
        Set network weights to new values, subject to constraints.
        """
        # Check input weight dims match network sizes
        assert len(weights) == len(self.w)
        for w1, w2 in zip(weights, self.w):
            assert w1.shape == w2.shape
        # Check bounds satisfied
        if assert_bounds:
            for w in weights:
                assert np.all(w >= self.w_bounds[0])
                assert np.all(w <= self.w_bounds[1])
        # Set values
        self.w = [w.copy().astype(float) for w in weights]
        # Apply constraints
        self.clip_weights()

    def stimulus_as_psps(self, stimulus):
        """
        Sets a stimulus as an array of evoked psps.
        Stimulus: list / array -> psps: array, shape (num_nrns, num_iter).
        """
        if type(stimulus) is list:
            # Presents as a list of spike trains, len (num_nrns)
            return preprocess.pattern2psps(stimulus, self.dt, self.duration,
                                           **self.cell_params)
        else:
            # Already an array of evoked psps.
            return stimulus

    def times2steps(self, times):
        """
        Convert times to time_steps.
        """
        return np.round(times / self.dt).astype(int)

    def clip_weights(self):
        """
        Clip out-of-bound weights in each layer.
        """
        [np.clip(w, self.w_bounds[0], self.w_bounds[1], w) for w in self.w]
