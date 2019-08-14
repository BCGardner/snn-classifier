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


class SRM(object):
    """
    (Simplified) Spike-Response Model (SRM).

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
        Reset strength (mV).
    q : float
        Charge transferred due to a presynaptic spike (nA).
    """
    def __init__(self, **kwargs):
        # Default cell prms
        self.prms = ParamSet({'tau_m': 10.,
                              'tau_s': 5.,
                              'theta': 15.,
                              'cm': 2.5,
                              'kappa_0': -15.,
                              'q': 5.})
        self.update(**kwargs)

    def update(self, **kwargs):
        self.prms.overwrite(**kwargs)
        # Derived values
        # psp coeff (mV)
        self.prms['psp_coeff'] = self.prms['q'] / self.prms['cm'] * \
            self.prms['tau_m'] / (self.prms['tau_m'] - self.prms['tau_s'])
