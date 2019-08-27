#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Definitions of multilayer spiking networks, optimised for Spike
Response Model (SRM) neuronal dynamics.

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

from .base import MultilayerSRMBase
from ..neuron import ExpRate


class MultilayerSRM(MultilayerSRMBase):
    """
    Spiking neural network, optimised specifically for SRM neurons.

    Parameters
    ----------
    sizes : list, len (num_layers)
        Num. neurons in [input, hidden, output layers].
    param : container
        Using cell, pattern, net and common parameters.
    weights : list, optional, len (num_layers - 1)
        Sets initial network weights. Each element is an array of shape
        (num_post, num_pre).
    EscapeRate : class, default EXPRate
        Stochastic spike generator for hidden layer neurons.
    """
    def simulate(self, stimulus, latency=False, return_psps=False,
                 debug=False):
        """
        Network activity in response to an input stimulus driving the
        network. Based on iterative procedure to determine spike times.
        Profiled at t = 0.34 s (iris, defaults (29/08/18), 20 hidden nrns).
        Speedup of 9x compared with non-iterative method.
        Scales less well as number of neurons increases (loops used).

        Parameters
        ----------
        stimulus : array or list
            Input drive to the network, as one of two types:
                - predetermined psps: array, shape (num_inputs, num_iter)
                - set of spike trains: list, len (num_inputs)
        latency : bool, optional
            Restrict to just finding first output spikes.
        return_psps : bool
            Return PSPs evoked due to each layer, excluding output layer, each
            with shape (num_nrns, num_iter).
        debug : bool, optional
            Record network dynamics for debugging.

        Returns
        -------
        spike_trains_l : list
            List of neuron spike trains, for layers l > 0.
        psps : list, optional
            List of PSPs evoked, for layers l < L.
        rec : dict, optional
            Debug recordings containing
            {'psp' as list of evoked PSPs, 'u' as list of potentials}.
        """
        # === Initialise ==================================================== #

        # Cast stimulus to PSPs evoked by input neurons
        psp_inputs = self.stimulus_as_psps(stimulus)
        num_iter = psp_inputs.shape[1]
        # Ensure pattern duration is compatible with look up tables
        assert num_iter == len(self.lut['psp'])

        # Record
        rec = {}
        if return_psps:
            rec['psp'] = [np.empty((i, num_iter))
                          for i in self.sizes[:-1]]
            rec['psp'][0] = psp_inputs
        if debug:
            rec['u'] = [np.empty((i, num_iter))
                        for i in self.sizes[1:]]
        # Spike trains for layers: l > 0
        spike_trains_l = [[np.array([]) for j in xrange(self.sizes[i])]
                          for i in xrange(1, self.num_layers)]

        # === Run simulation ================================================ #

        # PSPs evoked by input neurons
        psps = psp_inputs
        # Hidden layer responses: l = [1, L)
        for l in xrange(self.num_layers - 2):
            potentials = np.dot(self.w[l], psps)
            # Stochastic spiking
            unif_samples = self.rng.uniform(size=np.shape(potentials))
            # PSPs evoked by this layer
            psps = np.zeros((self.sizes[l+1], num_iter))
            for i in xrange(self.sizes[l+1]):
                num_spikes = 0
                while True:
                    rates = self.neuron_h.activation(potentials[i])
                    thr_idxs = \
                        np.where(unif_samples[i] < rates * self.dt)[0]
                    if num_spikes < len(thr_idxs):
                        fire_idx = thr_idxs[num_spikes]
                        iters = num_iter - fire_idx
                        potentials[i, fire_idx:] += self.lut['refr'][:iters]
                        psps[i, fire_idx:] += self.lut['psp'][:iters]
                        spike_trains_l[l][i] = \
                            np.append(spike_trains_l[l][i],
                                      fire_idx * self.dt)
                        num_spikes += 1
                    else:
                        break
            # Record
            if return_psps:
                rec['psp'][l+1] = psps
            if debug:
                rec['u'][l] = potentials
        # Output responses
        potentials = np.dot(self.w[-1], psps)
        # PSPs evoked by this layer
        psps = np.zeros((self.sizes[-1], num_iter))
        for i in xrange(self.sizes[-1]):
            num_spikes = 0
            while True:
                thr_idxs = \
                    np.where(potentials[i] > self.cell_params['theta'])[0]
                if num_spikes < len(thr_idxs):
                    fire_idx = thr_idxs[num_spikes]
                    iters = num_iter - fire_idx
                    potentials[i, fire_idx:] += self.lut['refr'][:iters]
                    psps[i, fire_idx:] += self.lut['psp'][:iters]
                    spike_trains_l[-1][i] = np.append(spike_trains_l[-1][i],
                                                      fire_idx * self.dt)
                    # Optimisation: skip output spikes after first
                    if latency:
                        break
                    else:
                        num_spikes += 1
                else:
                    break
        if debug:
            rec['u'][-1] = potentials

        if debug:
            return spike_trains_l, rec
        elif return_psps:
            return spike_trains_l, rec['psp']
        else:
            return spike_trains_l


class MultilayerSRMSub(MultilayerSRMBase):
    """
    Spiking neural network with subconnections. Each connection is split into
    'subconnections' with different conduction delays. The delays are
    linearly-spaced over a specified range.

    Parameters
    ----------
    sizes : list
        Num. neurons in [input, hidden, output layers].
    param : container
        Using cell, pattern, net and common parameters.
    weights : list, optional, len (num_layers-1)
        Sets initial network weights. Each element is an array of shape
        (num_post, num_pre, num_subs).
    num_subs : int, array-like, default 1
        Number of incoming subconnections per neuron for layers l > 0.
    max_delay : int, default 10
        Maximum conduction delay for num_subs > 1.
    conns_fr : float, optional
        Fraction of hidden subconns randomly enabled at init.
    delay_distr : {'lin', 'unif'}, default 'lin'
        Distribution used to set subconnection delay values,
        between 1 and 10 ms.
    EscapeRate : class, default EXPRate
        Stochastic spike generator for hidden layer neurons.
    """
    def __init__(self, sizes, param, weights=None, num_subs=1, max_delay=10.,
                 conns_fr=None, delay_distr='lin', EscapeRate=ExpRate):
        """
        Randomly initialise weights of neurons in each layer. Subconnection
        delays are linearly-spaced. Fractional number of connections are
        approximated by clamping weights to zero on unused subconnections.
        """
        assert max_delay > 1.
        # Set common prms, prototype connectivity and initialise weights
        if np.isscalar(num_subs):
            self.num_subs = [num_subs for i in xrange(len(sizes) - 1)]
        else:
            self.num_subs = np.asarray(num_subs, dtype=int)
        self.conns_fr = conns_fr
        super(MultilayerSRMSub, self).__init__(sizes, param, weights,
                                               EscapeRate)

        # Initialise delay values
        def distr_type(key, low=1., high=10.):
            # Return a desired distribution function, specified over
            # a given range of possible values
            choices = {'lin': lambda num: np.linspace(low, high, num=num),
                       'unif': lambda num: self.rng.uniform(low, high,
                                                            size=num)}
            return choices[key]
        distr = distr_type(delay_distr, high=max_delay)
        self.delays = [self.times2steps(distr(i)) for i in self.num_subs]

    def _build_network(self, weights):
        """
        Prototype network connectivity with subconns. and initialise weight
        values. Clamped hidden weights are determined here.
        """
        self.w = [np.empty((i, j, k))
                  for i, j, k in zip(self.sizes[1:], self.sizes[:-1],
                  self.num_subs)]
        # Hidden weights mask with values clamped to zero
        if self.conns_fr is not None:
            self.clamp_mask = [np.zeros(w.shape, dtype=bool)
                               for w in self.w[:-1]]
            for idx, w in enumerate(self.w[:-1]):
                num_clamps = \
                    np.round((1. - self.conns_fr) *
                             self.num_subs[idx]).astype(int)
                assert 0 <= num_clamps < self.num_subs[idx]
                for coord in np.ndindex(w.shape[:-1]):
                    clamp_idxs = self.rng.choice(self.num_subs[idx],
                                                 num_clamps, replace=False)
                    self.clamp_mask[idx][coord][clamp_idxs] = True
        self.reset(weights=weights)

    def _init_weights(self):
        """
        Randomly intialise weights according to a uniform distribution, and
        normalised w.r.t. num. subconns.
        """
        weights = []
        # Hidden layers
        for i, j, k in zip(self.sizes[1:-1], self.sizes[:-2],
                           self.num_subs[:-1]):
            weights.append(self.rng.uniform(*self.w_h_init,
                                            size=(i, j, k)))
        # Output layer
        weights.append(self.rng.uniform(*self.w_o_init,
                                        size=(self.sizes[-1],
                                              self.sizes[-2],
                                              self.num_subs[-1])))
        # Normalise weights w.r.t. num_subs
        if hasattr(self, 'clamp_mask'):
            # Num. actual hidden weights with clamped values
            for idx, w in enumerate(weights[:-1]):
                num_conns = np.round(self.conns_fr *
                                     self.num_subs[idx]).astype(int)
                w /= num_conns
            weights[-1] /= self.num_subs[-1]
        else:
            weights = [w / w.shape[-1] for w in weights]
        self.set_weights(weights)

    def set_weights(self, weights, assert_bounds=False):
        """
        Set network weights to new values, subject to constraints, including
        missing subconnections (weights clamped to zero).
        """
        super(MultilayerSRMSub, self).set_weights(weights, assert_bounds)
        if hasattr(self, 'clamp_mask'):
            for w, m in zip(self.w[:-1], self.clamp_mask):
                w[m] = 0.

    def simulate(self, stimulus, latency=False, return_psps=False,
                 debug=False):
        """
        Network activity in response to an input stimulus driving the
        network. Based on iterative procedure to determine spike times.
        Expected speedup of ~9x compared with non-iterative method.

        Inputs
        ------
        stimulus : array or list
            Input drive to the network, as one of two types:
                - predetermined psps: array, shape (num_inputs, num_iter)
                - set of spike trains: list, len (num_inputs)
        latency : bool, optional
            Restrict to just finding first output spikes.
        return_psps : bool
            Return PSPs evoked due to each layer, excluding output layer, each
            with shape (num_nrns, num_subconns, num_iter).
        debug : bool, optional
            Record network dynamics for debugging.

        Outputs
        -------
        spike_trains_l : list
            List of neuron spike trains, for layers l > 0.
        psps : list, optional
            List of PSPs evoked, for layers l < L.
        rec : dict, optional
            Debug recordings containing
            {'psp' as list of evoked PSPs, 'u' as list of potentials}.
        """
        # === Initialise ==================================================== #

        # Cast stimulus to PSPs evoked by input neurons
        psp_inputs = self.stimulus_as_psps(stimulus)
        num_iter = psp_inputs.shape[1]
        # Ensure pattern duration is compatible with look up tables
        assert num_iter == len(self.lut['psp'])
        # Expand PSPs: shape (num_inputs, num_iter) ->
        # shape (num_inputs, num_subs, num_iter)
        shape = (len(psp_inputs), self.num_subs[0], num_iter)
        psps = np.zeros(shape)
        for i, delay in enumerate(self.delays[0]):
            psps[:, i, delay:] = psp_inputs[:, :-delay]

        # Record
        rec = {}
        if return_psps:
            rec['psp'] = [np.empty((i, j, num_iter))
                          for i, j in zip(self.sizes[:-1], self.num_subs)]
            rec['psp'][0] = psps
        if debug:
            rec['u'] = [np.empty((i, num_iter))
                        for i in self.sizes[1:]]
        # Spike trains for layers: l > 0
        spike_trains_l = [[np.array([]) for j in xrange(self.sizes[i])]
                          for i in xrange(1, self.num_layers)]

        # === Run simulation ================================================ #

        # Hidden layer responses: l = [1, L)
        for l in xrange(self.num_layers - 2):
            potentials = np.tensordot(self.w[l], psps)
            # Stochastic spiking: the order in which random values are sampled
            # differs from simulate_steps
            unif_samples = self.rng.uniform(size=np.shape(potentials))
            # PSPs evoked by this layer
            psps = np.zeros((self.sizes[l+1], self.num_subs[l+1], num_iter))
            for i in xrange(self.sizes[l+1]):
                num_spikes = 0
                while True:
                    rates = self.neuron_h.activation(potentials[i])
                    thr_idxs = \
                        np.where(unif_samples[i] < rates * self.dt)[0]
                    if num_spikes < len(thr_idxs):
                        fire_idx = thr_idxs[num_spikes]
                        iters = num_iter - fire_idx
                        potentials[i, fire_idx:] += self.lut['refr'][:iters]
                        for j, delay in enumerate(self.delays[l+1]):
                            fire_idx_delay = fire_idx + delay
                            if fire_idx_delay < num_iter:
                                psps[i, j, fire_idx_delay:] += \
                                    self.lut['psp'][:iters-delay]
                        spike_trains_l[l][i] = \
                            np.append(spike_trains_l[l][i],
                                      fire_idx * self.dt)
                        num_spikes += 1
                    else:
                        break
            # Record
            if return_psps:
                rec['psp'][l+1] = psps
            if debug:
                rec['u'][l] = potentials
        # Output responses
        potentials = np.tensordot(self.w[-1], psps)
        for i in xrange(self.sizes[-1]):
            num_spikes = 0
            while True:
                thr_idxs = \
                    np.where(potentials[i] > self.cell_params['theta'])[0]
                if num_spikes < len(thr_idxs):
                    fire_idx = thr_idxs[num_spikes]
                    iters = num_iter - fire_idx
                    potentials[i, fire_idx:] += self.lut['refr'][:iters]
                    spike_trains_l[-1][i] = np.append(spike_trains_l[-1][i],
                                                      fire_idx * self.dt)
                    # Optimisation: skip output spikes after first
                    if latency:
                        break
                    else:
                        num_spikes += 1
                else:
                    break
        if debug:
            rec['u'][-1] = potentials

        if debug:
            return spike_trains_l, rec
        elif return_psps:
            return spike_trains_l, rec['psp']
        else:
            return spike_trains_l
