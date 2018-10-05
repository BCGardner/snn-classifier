#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Jun 2018

@author: BG

Parameter containers with default values for a feedforward spiking network.
"""

import numpy as np

from snncls.parameters import ParamSet


class SimParam(object):
    """
    Top-level container for all simulation parameters, including:
        - common params
        - cell / neuron params
        - network / regularisation params
        - pattern statistics
    """
    def __init__(self, dt=0.1, seed=None):
        # === Common parameters ============================================= #
        self.dt = dt         # Time step (ms)
        self.seed = seed     # Seed for initialisation
        self.rng = np.random.RandomState(self.seed)
        self.rng_st0 = self.rng.get_state()  # Initial rng state

        # === Default parameters ============================================ #
        # Cell
        self.cell = ParamSet({'tau_m': 10.,      # Membrane time constant (ms)
                              'tau_s': 5.,       # Synaptic rise time (ms)
                              'theta': 15.,      # Firing threshold (mV)
                              'cm': 2.5,         # Membrane capacitance (nF)
                              'kappa_0': -15.,   # Reset strength (mV)
                              'psp_coeff': 4.})  # psp coeff (mV)
        # Network (based on iris dataset with latency decoding)
        self.net = ParamSet({'w_h_init': (0.0, 2.0),  # Initial hidden weights
                             'w_o_init': (0.0, 4.0),  # Initial output weights
                             'w_bounds': (-15., 15.),  # Weight bounds
                             'l2_pen': 2E-3,        # L2 weight penalty term
                             'syn_scale': 0.1})     # Synaptic scaling
        # Patterns
        self.pattern = ParamSet({'neurons_f': 12,   # num. encoding nrns
                                 'beta': 1.5,       # width receptive fields
                                 'duration': 40.})  # sim. runtime per pattern

    def update(self, **kwargs):
        """
        Update parameter sets for matching keys, update derived parameters.
        """
        self.cell.overwrite(**kwargs)
        self.net.overwrite(**kwargs)
        self.pattern.overwrite(**kwargs)
        # Derived variables
        self.decay_m = np.exp(-self.dt / self.cell['tau_m'])
        self.decay_s = np.exp(-self.dt / self.cell['tau_s'])


class LatencyParam(SimParam):
    """
    Parameters for NetworkSoftmax.
    """
    def __init__(self, dt=0.1, seed=None, **kwargs):
        # Common defaults
        super(LatencyParam, self).__init__(dt, seed)
        # Latency specific defaults
        self.net.update({'eta0': 300.,
                         'tau_max': 1000.,
                         'cpd_scale': 2.})
        # Update defaults
        self.update(**kwargs)


class TemporalParam(SimParam):
    """
    Parameters for NetworkTemporal.
    """
    def __init__(self, dt=0.1, seed=None, spikes_ref=np.array([10.]),
                 **kwargs):
        # Common defaults
        super(TemporalParam, self).__init__(dt, seed)
        # Temporal specific
        self.net['eta0'] = 20.
        self.spikes_ref = spikes_ref
        # Update defaults
        self.update(**kwargs)
