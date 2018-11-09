#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mar 2017

@author: BG

Define multilayer network object, used for network training.

TODO: Implement two network types and have them contained rather than inherited
by a training class:
    1) LIF-type (optimised for large datasets)
    2) general-type (works with any provided neuron type))
"""

from __future__ import division

import numpy as np

from snncls import escape_noise, preprocess


class Network(object):
    """
    Feed-forward spiking neural network with no conduction delays. Subthreshold
    neuronal dynamics are governed by LIF model in all layers.
    """
    def __init__(self, sizes, param, weights=None,
                 EscapeRate=escape_noise.ExpRate):
        """
        Randomly initialise weights of neurons in each layer.

        Inputs
        ------
        sizes : list
            Num. neurons in [input, hidden, output layers].
        param : container
            Using cell, pattern, net and common parameters.
        weights : list, optional
            List of initial weight values of network.
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
        # Initialise network state
        self.reset(weights=weights)

    def reset(self, rng_st=None, weights=None):
        """
        (Re)set network parameters. Optionally set rng to a given state, set
        weights to a given value.
        """
        # Set rng to a given state, otherwise store last state
        if rng_st is not None:
            self.rng.set_state(rng_st)
        else:
            self.rng_st = self.rng.get_state()
        # Set weights to a given value, otherwise randomly intialise according
        # to a uniform distribution
        if weights is not None:
            self.w = [w.copy() for w in weights]
            # Weight bound constraints
            for w in self.w:
                assert np.all(w >= self.w_bounds[0])
                assert np.all(w <= self.w_bounds[1])
        else:
            self.w = []
            # Hidden layers
            for i, j in zip(self.sizes[1:-1], self.sizes[:-2]):
                self.w.append(self.rng.uniform(*self.w_h_init, size=(i, j)))
            # Output layer
            self.w.append(self.rng.uniform(*self.w_o_init,
                                           size=self.sizes[-1:-3:-1]))
            # Clip out-of-bound weights
            self.clip_weights()

    def simulate(self, stimulus, latency=False, debug=False):
        """
        Network activity in response to an input stimulus driving the
        network. Based on iterative procedure to determine spike times.
        Profiled at t = 0.34 s (iris, defaults (29/08/18), 20 hidden nrns).
        Speedup of 9x compared with non-iterative method.
        Scales less well as number of neurons increases (loops used).

        Inputs
        ------
        stimulus : array or list
            Input drive to the network, as one of two types:
                - predetermined psps: array, shape (num_inputs, num_iter)
                - set of spike trains: list, len (num_inputs)
        latency : bool, optional
            Restrict to just finding first output spikes.
        debug : bool, optional
            Record network dynamics for debugging.

        Outputs
        -------
        spike_trains_l : list
            List of neuron spike trains, for layers l > 0.
        rec : dict, optional
            Debug recordings containing
            {psps evoked from hidden neurons 'psp', potentials 'u',
             bool spike trains 'S'}.
        """
        # === Initialise ==================================================== #

        # Cast stimulus to PSPs evoked by input neurons
        psp_inputs = self.stimulus_as_psps(stimulus)
        num_iter = psp_inputs.shape[1]
        # Ensure pattern duration is compatible with look up tables
        assert num_iter == len(self.lut['psp'])

        # Debug
        if debug:
            rec = {'psp': [np.empty((i, num_iter))
                           for i in self.sizes[1:]],
                   'u': [np.empty((i, num_iter))
                         for i in self.sizes[1:]],
                   'S': [np.empty((i, num_iter), dtype=int)
                         for i in self.sizes[1:]],
                   'smpls': []}
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
            if debug:
                rec['u'][l] = potentials
                rec['psp'][l] = psps
                rec['smpls'].append(unif_samples)
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
            rec['psp'][-1] = psps

        if debug:
            return spike_trains_l, rec
        else:
            return spike_trains_l

    def simulate_steps(self, psp_inputs, debug=False):
        """
        Network activity in response to an input pattern presented to the
        network.
        Profiled at t = 3.1 s (iris, defaults (29/08/18), 20 hidden nrns).

        Inputs
        ------
        psp_inputs : array
            PSPs from input layer neurons of size: <num_inputs> by <num_iter>.
        debug : bool
            Record network dynamics for debugging.

        Outputs
        -------
        spiked_l : list
            List of boolean spike trains in layers l > 0.
        rec : dict, optional
            Debug recordings containing
            {psps evoked from hidden neurons 'psp', potentials 'u',
             bool spike trains 'S'}.
        """
        # === Initialise ==================================================== #

        # Pattern stats
        num_iter = psp_inputs.shape[1]  # Num. time steps per duration
        # Debug
        if debug:
            rec = {'psp': [np.empty((i, num_iter))
                           for i in self.sizes[1:-1]],
                   'u': [np.empty((i, num_iter))
                         for i in self.sizes[1:]],
                   'S': [np.empty((i, num_iter), dtype=int)
                         for i in self.sizes[1:]]}
        # Bool spike trains in each layer: l > 1
        spiked_l = [np.zeros((i, num_iter), dtype=bool)
                    for i in self.sizes[1:]]
        # PSPs from each layer: 1 <= l < L, reset kernels in each layer: l >= 1
        # Membrane, synaptic exponential decays
        psp_trace_l = [np.zeros((i, 2)) for i in self.sizes[1:-1]]
        refr_l = [np.zeros(i) for i in self.sizes[1:]]

        # === Run simulation ================================================ #
        unif_samples = [self.rng.uniform(size=(i, num_iter))
                        for i in self.sizes[1:-1]]
        if debug:
            rec['smpls'] = unif_samples
        for t_step in xrange(num_iter):
            # Update kernels: l >= 1
            psp_trace_l = [i * [self.decay_m, self.decay_s]
                           for i in psp_trace_l]
            psp_l = [i[:, 0] - i[:, 1] for i in psp_trace_l]
            refr_l = [i * self.decay_m for i in refr_l]

            psps = psp_inputs[:, t_step]
            # Hidden layer responses: 1 <= l < L
            for l in xrange(self.num_layers - 2):
                potentials = np.dot(self.w[l], psps) + refr_l[l]
                rates = self.neuron_h.activation(potentials)
                spiked_l[l][:, t_step] = unif_samples[l][:, t_step] < \
                    self.dt * rates
                psps = psp_l[l]
                # Debugging
                if debug:
                    rec['psp'][l][:, t_step] = psps
                    rec['u'][l][:, t_step] = potentials
            # Output layer response
            potentials = np.dot(self.w[-1], psps) + refr_l[-1]
            spiked_l[-1][:, t_step] = potentials > self.cell_params['theta']
            # Update kernels
            for l in xrange(self.num_layers - 2):
                psp_trace_l[l][spiked_l[l][:, t_step], :] += \
                    self.cell_params['psp_coeff']
            for l in xrange(self.num_layers - 1):
                refr_l[l][spiked_l[l][:, t_step]] += \
                    self.cell_params['kappa_0']
            # Debugging
            if debug:
                rec['u'][-1][:, t_step] = potentials
                for l in xrange(self.num_layers - 1):
                    rec['S'][l][:, t_step] = spiked_l[l][:, t_step]

        # === Gather output spike trains ==================================== #

        # Spike trains: l > 0
        spike_trains_l = []
        for spiked in spiked_l:
            spike_trains_l.append([np.where(isspike)[0] * self.dt
                                   for isspike in spiked])
        if debug:
            return spike_trains_l, rec
        else:
            return spike_trains_l

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
