#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Definitions of standard neuron types and their associated parameter sets.

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

from .escape_noise import ExpRate
from ..parameters import ParamSet
from ..preprocess import pattern2psps, refr_reduce


class SRM(object):
    """
    (Simplified) Spike-Response Model (SRM).

    Parameters
    ----------
    stochastic : bool, optional, default False
        Enable stochastic spike generator.
    EscapeRate : class, default ExpRate
        Escape noise model used to distribute spikes, if stochastic enabled.
    rng : object, optional
        Numpy random num. generator, if stochastic enabled.
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
    def __init__(self, stochastic=False, EscapeRate=ExpRate,
                 rng=np.random.RandomState(), **kwargs):
        # Stochastic spike generator
        self.stochastic = stochastic
        if stochastic:
            self.rng = rng
        # Default cell prms
        self.prms = ParamSet({'tau_m': 10.,
                              'tau_s': 5.,
                              'theta': 15.,
                              'cm': 2.5,
                              'kappa_0': -15.,
                              'q': 5.})
        self.update(EscapeRate=EscapeRate, **kwargs)

    def update(self, EscapeRate=ExpRate, **kwargs):
        self.prms.overwrite(**kwargs)
        # Derived values
        # psp coeff (mV)
        self.prms['psp_coeff'] = self.prms['q'] / self.prms['cm'] * \
            self.prms['tau_m'] / (self.prms['tau_m'] - self.prms['tau_s'])
        # Stochastic spike generator
        if self.stochastic:
            self.esc_rate = EscapeRate(theta=self.prms['theta'], **kwargs)

    def response(self, stimulus, w, dt, duration, return_psps=False,
                 debug=False):
        """
        Neuron's response to a driving stimulus, through a weight vector.

        Parameters
        ----------
        stimulus : list, len (num_inputs)
            Set of presynaptic spike trains.
        w : array, shape (num_inputs,)
            Input weight vector.
        dt : float
            Time resolution.
        duration : float
            Cut-off point for determining neuron response.
        return_psps : bool
            Additionally return PSPs evoked at this neuron.
        debug : bool
            Additionally return recorded potential.

        Returns
        -------
        spike_train : array, shape (num_spikes,)
            Output spike train.
        psps : array, optional, shape (num_pre, num_iter)
            Evoked PSPs at this neuron.
        potentials : array, optional, shape (num_iter)
            Recorded voltages.
        """
        # Optimisation
        lut_refr = refr_reduce(np.arange(0., duration, dt), np.array([0.]),
                               **self.prms)
        # Transform stimulus into PSPs evoked at this neuron
        # psps : array, shape (num_pre, num_iter)
        psps = pattern2psps(stimulus, dt, duration, **self.prms)
        num_iter = psps.shape[1]

        # Initialisation
        spike_train = np.array([])
        potentials = np.dot(w, psps)
        if self.stochastic:
            unif_samples = self.rng.uniform(size=len(potentials))
        # Find spike-trigger events
        num_spikes = 0
        while True:
            if self.stochastic:
                rates = self.esc_rate.activation(potentials)
                thr_idxs = np.where(unif_samples < rates * dt)[0]
            else:
                thr_idxs = np.where(potentials > self.prms['theta'])[0]
            if num_spikes < len(thr_idxs):
                        fire_idx = thr_idxs[num_spikes]
                        iters = num_iter - fire_idx
                        potentials[fire_idx:] += lut_refr[:iters]
                        spike_train = np.append(spike_train, fire_idx * dt)
                        num_spikes += 1
            else:
                break
        # Optional return values
        ret = (spike_train,)
        if return_psps:
            ret += (psps,)
        if debug:
            ret += (potentials,)
        if len(ret) > 1:
            return ret
        else:
            return ret[0]
