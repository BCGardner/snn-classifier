#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on May 2017

@author: BG
"""

from __future__ import division

import numpy as np


class PSPWindow(object):
    """
    PSP learning window.
    Causal (pre-to-post) correlation traces only.
    """
    def __init__(self, param):
        self.epsilon_0 = param.cell['psp_coeff']
        self.tau_m = param.cell['tau_m']
        self.tau_s = param.cell['tau_s']

    def causal(self, post_spikes, pre_spikes):
        """
        Causal-correlation traces at each [post-spike] due to each [pre-spike].
        Output size: <# post_spikes> by <# pre_spikes>.
        """
        lags = post_spikes[:, np.newaxis] - pre_spikes[np.newaxis, :]
        u = (lags > 0.).astype(float)
        traces = self.epsilon_0 * (np.exp(-lags / self.tau_m) -
                                   np.exp(-lags / self.tau_s)) * u
        return traces

    def causal_reduce(self, post_spikes, pre_spikes):
        """
        Summed traces at each output time, due to pre_spikes.
        Output size: <# post_spikes>.
        """
        return np.sum(self.causal(post_spikes, pre_spikes), 1)

    def double_conv_psp(self, spikes_o, spikes_h, psp_in):
        """
        Double conv function, evaluated at each spikes_o due to shared spikes_h
        and psp_in. psp_in is of size <num_inputs> by <num_spikes_h>.

        Inputs
        ------
        spikes_o : array
            Vector of output spike times: <num_spikes>.
        spikes_h : array
            Vector of hidden spike times: <num_spikes>.
        psp_in : array
            Vector of psps due to each input spike train at each hidden firing
            time: <# inputs> by <# spikes_h>.

        Output
        ------
        return : array
            Double convolution of input-hidden-output spiking of size:
                <# spikes_o> by <# inputs>.
        """
        # Correlations between each output-hidden spike pair:
        # <# spikes_o> by <# spikes_h>
        corr_oh = self.causal(spikes_o, spikes_h)
        # <# spikes_o> by <1> by <# spikes_h>
        corr_oh = corr_oh[:, np.newaxis, :]
        psp_in = psp_in[np.newaxis, :, :]  # Prep for broadcasting
        return np.sum(corr_oh * psp_in, 2)
