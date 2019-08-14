#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Aug 2019

@author: BG

Specify specific neuron type and its prms.
"""

from __future__ import division

import numpy as np

from snncls.parameters import ParamSet


class LIFNeuron(object):
    """
    Leaky Integrate-and-Fire neuron type.

    Parameters
    ----------
    tau_m : float
        Membrane time constant (ms).
    tau_s : float
        Synaptic rise time (ms).
    theta : float
        Firing threshold (mV).
    cm : float
        Membrane capacitance (nF)
    kappa_0 : float
        Membrane time constant (ms).
    q : float
        Charge transferred due to a presynaptic spike (nA).
    """
    def __init__(self, **kwargs):
        # Cell
        self.prms = ParamSet({'tau_m': 10.,      # Membrane time constant (ms)
                              'tau_s': 5.,       # Synaptic rise time (ms)
                              'theta': 15.,      # Firing threshold (mV)
                              'cm': 2.5,         # Membrane capacitance (nF)
                              'kappa_0': -15.,   # Reset strength (mV)
                              'psp_coeff': 4.})  # psp coeff (mV)
        self.prms.overwrite(**kwargs)