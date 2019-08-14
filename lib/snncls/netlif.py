#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Aug 2019

@author: BG

Abstract network class, optimised for LIF neurons with shared cellular prms.
"""

from __future__ import division

import numpy as np

from snncls import netbase, escape_noise, preprocess


class NetLIF(netbase.NetBase):
    """
    Abstract class of spiking neural network with hidden neurons.
    Connectivity is feedforward with all-all connectivity.
    Subthreshold neuronal dynamics are governed by LIF model in all layers.
    """
    def __init__(self, sizes, param, EscapeRate=escape_noise.ExpRate):
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
        super(NetLIF, self).__init__(sizes, param)

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
