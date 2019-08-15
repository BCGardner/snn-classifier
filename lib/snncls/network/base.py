#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mar 2019

@author: BG

Base spiking neural network classes.
"""

from __future__ import division

import numpy as np

from ..neuron import ExpRate
from .. import preprocess


class MultilayerBase(object):
    """
    Base class of spiking neural network with hidden neurons.
    Connectivity is feedforward with all-all connectivity.
    """
    def __init__(self, sizes, param):
        """
        Initialise common network parameters, constraints.
        """
        self.sizes = sizes  # Num. neurons in each layer
        self.num_layers = len(sizes)  # Total num. layers (incl. input)
        self.dt = param.dt  # Sim. time elapsed per iter
        self.duration = param.pattern['duration']  # Default sim. duration
        self.rng = param.rng
        # Parameters
        self.w_h_init = param.net['w_h_init']
        self.w_o_init = param.net['w_o_init']
        self.w_bounds = param.net['w_bounds']

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


class MultilayerSRMBase(MultilayerBase):
    """
    Base class of multilayer SNN. Subthreshold neuron dynamics are governed by
    simplified SRM in all layers, using optimisation procedures.
    """
    def __init__(self, sizes, param, EscapeRate=ExpRate):
        """
        Initialise common network parameters, constraints, cell parameters,
        optimisations.
        """
        super(MultilayerSRMBase, self).__init__(sizes, param)

        # Parameters
        self.cell_params = param.cell
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

    def stimulus_as_psps(self, stimulus):
        """
        Sets a stimulus as an array of evoked psps.
        Stimulus: list / array -> psps: array, shape (num_nrns, num_iter).
        """
        # Check num_inputs matched to stimulus size
        assert self.sizes[0] == len(stimulus)
        if type(stimulus) is list:
            # Presents as a list of spike trains, len (num_nrns)
            return preprocess.pattern2psps(stimulus, self.dt, self.duration,
                                           **self.cell_params)
        else:
            # Already an array of evoked psps.
            return stimulus
