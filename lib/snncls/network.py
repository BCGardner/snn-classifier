#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mar 2017

@author: BG

Define multilayer network object, which minimises cross-entropy loss.
"""

from __future__ import division

import numpy as np

from snncls import escape_noise


class Network(object):
    """
    Feed-forward spiking neural network.
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
            Using:
                cell_params : dict of neuron parameters.
                w_h_init : 2-tuple containing initial hidden weight range.
                w_o_init : 2-tuple containing initial output weight range.
        weights : list, optional
            List of initial weight values of network.
        EscapeRate : class
            Hidden layer neuron firing density.
        """
        self.sizes = sizes  # No. neurons in each layer
        self.num_layers = len(sizes)  # Total num. layers (incl. input)
        self.dt = param.dt  # Sim. time elapsed per iter
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

    def simulate(self, psp_inputs, debug=False):
        """
        Network activity in response to an input pattern presented to the
        network.

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
            {hidden layer psps 'psp', potential 'u', bool spike trains 'S'}.
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
                spiked_l[l][:, t_step] = self.rng.rand(len(rates)) < \
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

    def times2steps(self, spike_times):
        """
        Convert a sequence of spike times to their time_steps.
        """
        return np.round(spike_times / self.dt).astype(int)

    def clip_weights(self):
        """
        Clip out-of-bound weights in each layer.
        """
        [np.clip(w, self.w_bounds[0], self.w_bounds[1], w) for w in self.w]


class NetworkDelay(object):
    """
    Feed-forward spiking neural network with conduction delays.
    """
    def __init__(self, sizes, num_subs, param, weights=None,
                 EscapeRate=escape_noise.ExpRate):
        """
        Randomly initialise weights of neurons in each layer.
        Each neuron receives multiple connections (subconnections) from every
        neuron in the previous layer. Each subconnection has a delay ranging
        from [1, N] ms for num_sub subconnections per neuron.
        Weights in a layer are of size: <n^l> by <n^l-1> by <num_subs>.

        Inputs
        ------
        sizes : list
            Num. neurons in [input, hidden, output layers].
        num_subs : int
            Num. subconnections per neuron in each layer.
        param : container
            Using:
                cell_params : dict of neuron parameters.
                w_h_init : 2-tuple containing initial hidden weight range.
                w_o_init : 2-tuple containing initial output weight range.
        weights : list, optional
            List of initial weight values of network.
        EscapeRate : class
            Hidden layer neuron firing density.
        """
        self.sizes = sizes            # Num. neurons in each layer
        self.num_layers = len(sizes)  # Total num. layers (incl. input)
        self.num_subs = num_subs      # Num. subconnections per neuron
        self.dt = param.dt            # Sim. time elapsed per iter
        self.rng = param.rng
        # Parameters
        self.cell_params = param.cell
        self.w_h_init = param.net['w_h_init']
        self.w_o_init = param.net['w_o_init']
        # Hidden neuron model
        self.neuron_h = EscapeRate(param.cell['theta'])
        # Optimisations
        self.decay_m = param.decay_m
        self.decay_s = param.decay_s
        # Initialise network parameters and state
        self.delays = self.times2steps(np.arange(1., self.num_subs + 1))
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
        # Each weight array is of size: <num_post> by <num_pre> by <num_subs>
        if weights is not None:
            self.w = weights
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

    def simulate(self, psp_inputs, debug=False):
        """
        Network activity in response to an input pattern presented to the
        network. Conduction delays are included.

        Inputs
        ------
        psp_inputs : array
            PSPs from input layer neurons of size: <num_inputs> by <num_iter>.
        debug : bool
            Record network dynamics for debugging.

        Outputs
        -------
        spiked_l : list
            List of boolean spike trains in layers l > 1.
        rec : dict, optional
            Debug recordings containing
            {hidden layer psps 'psp', potential 'u', bool spike trains 'S'}.
        """
        # === Initialise ==================================================== #

        # Pattern stats
        num_iter = psp_inputs.shape[1]  # Num. time steps per duration
        # Debug
        if debug:
            rec = {'psp': [np.empty((i, self.num_subs, num_iter))
                           for i in self.sizes[:-1]],
                   'u': [np.empty((i, num_iter))
                         for i in self.sizes[1:]],
                   'S': [np.empty((i, num_iter), dtype=int)
                         for i in self.sizes[1:]]}
        # Bool spike trains in each layer: l > 1
        spiked_l = [np.zeros((i, num_iter), dtype=bool)
                    for i in self.sizes[1:]]
        # PSPs inputs adjusted to account for delays in time steps
        # PSPs of size: <num_inputs> by <num_iter + max_delay_iter>
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
                # Debugging
                if debug:
                    rec['psp'][l][:, :, t_step] = psps
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
            # Debugging
            if debug:
                rec['psp'][-1][:, :, t_step] = psps
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
            return spike_trains_l

    def times2steps(self, spike_times):
        """
        Convert a sequence of spike times to their time_steps.
        """
        return np.round(spike_times / self.dt).astype(int)
