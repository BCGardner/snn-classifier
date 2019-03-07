#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Feb 2019

@author: BG

Network with multiple subconnections and conduction delays.
"""

from __future__ import division

import numpy as np

from snncls import escape_noise, preprocess


class Network(object):
    """
    Feed-forward spiking neural network with conduction delays. Subthreshold
    neuronal dynamics are governed by LIF model in all layers.
    """
    def __init__(self, sizes, param, num_subs=1, max_delay=10, weights=None,
                 EscapeRate=escape_noise.ExpRate):
        """
        Randomly initialise weights of neurons in each layer.

        Inputs
        ------
        sizes : list
            Num. neurons in [input, hidden, output layers].
        param : container
            Using cell, pattern, net and common parameters.
        num_subs : int
            Number of subconnections per neuron in each layer.
        max_delay : int
            Maximum conduction delay for num_subs > 1.
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
        # Initialise subconnections
        self.num_subs = num_subs
        self.delays = self.times2steps(np.linspace(1, max_delay, num=num_subs,
                                                   dtype=int))
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
                self.w.append(self.rng.uniform(*self.w_h_init,
                                               size=(i, j, self.num_subs)))
            # Output layer
            self.w.append(self.rng.uniform(*self.w_o_init,
                                           size=(self.sizes[-1],
                                                 self.sizes[-2],
                                                 self.num_subs)))
            # Normalise weights w.r.t. num_subs
            self.w = [w / self.num_subs for w in self.w]
            # Clip out-of-bound weights
            self.clip_weights()

    def simulate_iter(self, stimulus, latency=False, return_psps=False,
                      debug=False):
        """
        Network activity in response to an input stimulus driving the
        network. Based on iterative procedure to determine spike times.
        Speedup of ~9x compared with non-iterative method.
        Scales less well as number of neurons increases (loops used).

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
            {psps evoked from hidden neurons 'psp', potentials 'u',
             bool spike trains 'S'}.
        """
        # === Initialise ==================================================== #

        # Cast stimulus to PSPs evoked by input neurons
        psp_inputs = self.stimulus_as_psps(stimulus)
        num_iter = psp_inputs.shape[1]
        # Ensure pattern duration is compatible with look up tables
        assert num_iter == len(self.lut['psp'])

        # Expand PSPs: shape (num_inputs, num_iter) ->
        # shape (num_inputs, num_subs, num_iter)
        shape = (len(psp_inputs), self.num_subs, num_iter)
        psps = np.zeros(shape)
        for i, delay in enumerate(self.delays):
            psps[:, i, delay:] = psp_inputs[:, :-delay]

        # Record
        rec = {}
        if return_psps:
            rec['psp'] = [np.empty((i, self.num_subs, num_iter))
                          for i in self.sizes[:-1]]
            rec['psp'][0] = psps
        if debug:
            rec.update({'u': [np.empty((i, num_iter))
                              for i in self.sizes[1:]],
                        'S': [np.empty((i, num_iter), dtype=int)
                              for i in self.sizes[1:]],
                        'smpls': []})
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
            psps = np.zeros((self.sizes[l+1], self.num_subs, num_iter))
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
                        for j, delay in enumerate(self.delays):
                            psps[i, j, fire_idx+delay:] += \
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
                rec['smpls'].append(unif_samples)
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
        else:
            if return_psps:
                return spike_trains_l, rec['psp']
            else:
                return spike_trains_l

    def simulate_steps(self, stimulus, return_psps=False, debug=False):
        """
        Network activity in response to an input pattern presented to the
        network. Conduction delays are included.

        Inputs
        ------
        stimulus : array or list
            Input drive to the network, as one of two types:
                - predetermined (pre-shifted) psps: array,
                  shape (num_inputs, num_iter)
                - set of spike trains: list, len (num_inputs)
        return_psps : bool
            Return PSPs evoked due to each layer, excluding output layer, each
            with shape (num_nrns, num_subconns, num_iter).
        debug : bool
            Record network dynamics for debugging.

        Outputs
        -------
        spiked_l : list
            List of boolean spike trains in layers l > 1.
        psps : list, optional
            List of PSPs evoked, for layers l < L.
        rec : dict, optional
            Debug recordings containing
            {hidden layer psps 'psp', potential 'u', bool spike trains 'S'}.
        """
        # === Initialise ==================================================== #

        # Cast stimulus to PSPs evoked by input neurons
        psp_inputs = self.stimulus_as_psps(stimulus)
        num_iter = psp_inputs.shape[1]

        # Record
        rec = {}
        if return_psps:
            rec['psp'] = [np.empty((i, self.num_subs, num_iter))
                          for i in self.sizes[:-1]]
        # Debug
        if debug:
            rec.update({'u': [np.empty((i, num_iter))
                              for i in self.sizes[1:]],
                        'S': [np.empty((i, num_iter), dtype=int)
                              for i in self.sizes[1:]]})
        # Bool spike trains in each layer: l > 1
        spiked_l = [np.zeros((i, num_iter), dtype=bool)
                    for i in self.sizes[1:]]
        # PSPs are padded to account for maximum conduction delay
        # PSPs : array, shape -> (num_inputs, num_iter + max_delay_iter)
        psp_inputs = np.concatenate((psp_inputs, np.zeros((self.sizes[0],
                                                           self.delays[-1]))),
                                    axis=1)
        # PSPs from each layer: 1 <= l < L, reset kernels in each layer: l >= 1
        # Membrane, synaptic exponential decays with memory up to max delay
        psp_trace_l = [np.zeros((2, i, self.delays[-1] + 1))
                       for i in self.sizes[1:-1]]
        reset_l = [np.zeros(i) for i in self.sizes[1:]]

        # === Run simulation ================================================ #

        for t_step in xrange(num_iter):
            # Update kernels: l >= 1
            # psp_trace_l is used to store historic PSP values evoked by hidden
            # layer neurons - values are stored up to the maximum conduction
            # delay, such that oldest PSP values are removed to make space for
            # updated PSP values
            for traces in psp_trace_l:
                traces[:, :, 1:] = traces[:, :, :-1]  # Shift trace memory
                traces[0, :, 0] *= self.decay_m
                traces[1, :, 0] *= self.decay_s
            psp_l = [traces[0] - traces[1] for traces in psp_trace_l]
            reset_l = [traces * self.decay_m for traces in reset_l]
            # PSPs contributed by inputs at each delay
            psps = psp_inputs[:, t_step - self.delays]
            # Hidden layer responses: 1 <= l < L
            for l in xrange(self.num_layers - 2):
                potentials = np.tensordot(self.w[l], psps) + reset_l[l]
                rates = self.neuron_h.activation(potentials)
                spiked_l[l][:, t_step] = self.rng.rand(len(rates)) < \
                    self.dt * rates
                # Record
                if return_psps:
                    rec['psp'][l][:, :, t_step] = psps
                if debug:
                    rec['u'][l][:, t_step] = potentials
                # Propagated, evoked PSPs at next layer
                psps = psp_l[l][:, self.delays]
            # Output layer response
            potentials = np.tensordot(self.w[-1], psps) + reset_l[-1]
            spiked_l[-1][:, t_step] = potentials > self.cell_params['theta']
            # Update kernels
            for l in xrange(self.num_layers - 2):
                psp_trace_l[l][:, spiked_l[l][:, t_step], 0] += \
                    self.cell_params['psp_coeff']
            for l in xrange(self.num_layers - 1):
                reset_l[l][spiked_l[l][:, t_step]] += \
                    self.cell_params['kappa_0']
            # Record
            if return_psps:
                rec['psp'][-1][:, :, t_step] = psps
            if debug:
                rec['u'][-1][:, t_step] = potentials
                for l in xrange(self.num_layers - 1):
                    rec['S'][l][:, t_step] = spiked_l[l][:, t_step]

        # === Gather output spike trains ==================================== #

        # Spike trains: l > 1
        spike_trains_l = []
        for spiked in spiked_l:
            spike_trains_l.append([np.where(isspike)[0] * self.dt
                                   for isspike in spiked])
        if debug:
            return spike_trains_l, rec
        else:
            if return_psps:
                return spike_trains_l, rec['psp']
            else:
                return spike_trains_l

    def simulate_steps_nondelay(self, psp_inputs, debug=False):
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
