"""
Network training and classification with softmax outputs, based on
time-to-first-spike decoding.

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

import numpy as np

from .base import NetworkTraining


class SoftmaxClf(NetworkTraining):
    """
    Training class for feed-forward spiking neural networks with softmax
    output activations. Class predictions are decided based on first output
    spikes.

    Parameters
    ----------
    sizes : list
        Num. neurons in [input, hidden, output layers].
    param : container
        Using:
            l2_pen : L2 penalty term.
            rate_pow : Exponent of rate penalties.
            syn_scale : synaptic scaling coefficient.
            cpd_scale : scaling factor of conditional probability distr.
            tau_max : default first spike time (numerical stability).
    """
    def __init__(self, sizes, param, **kwargs):
        """
        Initialise feedforward multilayer network and set training parameters.
        """
        super(SoftmaxClf, self).__init__(sizes, param, **kwargs)
        self.l2_pen = param.net['l2_pen']
        self.rate_pow = param.net['rate_pow']
        self.syn_scale = param.net['syn_scale']
        self.cpd_scale = param.net['cpd_scale']
        self.tau_max = param.net['tau_max']

    def backprop(self, X, y):
        """
        Forwards pass, and then backwards pass, given training case with
        target output vector y. Currently computes weight gradients for output
        and last hidden layer weights only.

        Inputs
        ------
        X : array or list
            Input stimulus driving the network, as one of two types:
                - predetermined psps: array, shape (num_inputs, num_iter)
                - set of spike trains: list, len (num_inputs)
        y : array, shape (num_outputs,)
            One-hot encoded class label, PMF for cross-entropy.

        Output
        ------
        dW : list
            Gradient of model parameters as list of arrays for each layer,
            shapes (n^l+1, n^l) for l in [0, L).
        """
        # === Initialise ==================================================== #

        # Gather weights
        weights = self.net.get_weights()
        # Gradients of cost function w.r.t. weights
        grad_w = [np.zeros(w.shape) for w in weights]

        # == Forwards pass ================================================== #

        # Gather network response, output scores and categorical probability
        # distribution (cpd)
        (spike_trains_l, psps), taus, cpd = self.feedforward(X, latency=True,
                                                             return_psps=True)

        # == Backwards pass ================================================= #

        # Output weight updates
        # Error signals array (dC/du): array, shape (num_outputs,)
        delta = self.cost_derivative(cpd, y)
        # Gradient of output layer weights for valid taus
        isspiked = taus < self.tau_max
        # Eligible PSPs, set to shape (isspiked, num_pre, ...)
        tau_steps = self.net.times2steps(taus[isspiked])
        psps_pre = np.moveaxis(psps[-1][..., tau_steps], -1, 0)
        # Reshape output errors at firing times to broadcast with presyn
        # PSPs of arbitrary shape, deltas as shape (num_outputs, 1, 1, ...)
        delta = delta.reshape(delta.shape + (1,) *
                              (psps_pre.ndim - 1))
        grad_w[-1][isspiked] = delta[isspiked] * psps_pre
        if self.net.num_layers > 2:
            # Hidden weight updates (last hidden layer)
            # List of each hidden neuron's firing times (as time steps)
            t_steps_h = [self.net.times2steps(spikes)
                         for spikes in spike_trains_l[-2]]
            # Weighted errors: array, shape
            # (num_taus, num_hidden[, num_subsO])
            delta_h = delta[isspiked] * weights[-1][isspiked]
            if hasattr(self.net, 'delays'):
                # Output delays : time_steps -> times
                delay_times = self.net.delays[-1] * self.net.dt
            for i, spikes_h in enumerate(spike_trains_l[-2]):
                # PSPs evoked at the i-th hidden neuron: array, shape
                # (num_inputs[, num_subsH], num_hidden_spikes)
                psps_pre = psps[-2][..., t_steps_h[i]]
                if hasattr(self.net, 'delays'):
                    # Subconnections
                    # Individual hidden spike contributions to output responses
                    # as array, shape (num_taus, num_subsO, num_spikesH)
                    corrs_oh = self.corr.causal(taus[isspiked], spikes_h,
                                                delay_times)
                    # Input PSPs weighted by hidden-output correlations, giving
                    # array, shape (num_taus, num_subsO, num_inputs, num_subsH)
                    double_convs = np.dot(corrs_oh, psps_pre.swapaxes(-1, -2))
                    grad_w[-2][i] = \
                        np.sum(delta_h[:, i, :, np.newaxis, np.newaxis] *
                               double_convs, (0, 1))
                else:
                    # No subconnections
                    grad_w[-2][i] = \
                        np.sum(delta_h[:, i, np.newaxis] *
                               self.corr.double_conv_psp(taus[isspiked],
                                                         spikes_h,
                                                         psps_pre), 0)

        # == Regularisation ================================================= #

        if self.rate_pow == 0:
            # Apply L2 penalty to all layers
            for l in range(self.net.num_layers - 1):
                # Compatability with existing results
                grad_w[l] += self.l2_pen * weights[l]
        else:
            # Apply L2 penalty with rate-dependence to all layers
            for l in range(self.net.num_layers - 1):
                for i, spike_train in enumerate(spike_trains_l[l]):
                    spike_count = len(spike_train)
                    if spike_count > 0:
                        grad_w[l][i] += self.l2_pen * weights[l][i] * \
                            spike_count**self.rate_pow

        # Synaptic scaling in all layers to sustain activity
        for l in range(self.net.num_layers - 1):
            fired = np.array([len(spikes) > 0 for spikes in spike_trains_l[l]])
            grad_w[l][~fired] -= self.syn_scale * \
                np.abs(weights[l][~fired])
        return grad_w

    def feedforward(self, X, latency=True, return_psps=False):
        """
        Forwards pass of network on an input sample.

        Inputs
        ------
        X : array or list
            Input stimulus driving the network.
        latency : bool, optional
            Optimisation - run simulation only up to first output spikes.
        return_psps : bool, optional
            Return list of PSPs evoked for layers l < L.

        Outputs
        -------
        spike_trains_l[, PSPs] : list or 2-tuple
            List of spike trains for layers l > 0. Optionally returns a
            2-tuple, containing the list of spike trains and list of evoked
            PSPs with shapes (num_nrns[, num_subconns], num_iter).
        taus : array, shape (num_outputs,)
            Set of first output spike times.
        cpd : array, shape (num_outputs, num_samples)
            Conditional probability distr. of first output spikes.
        """
        # Simulate network and gather spike trains[, PSPs] w.r.t. layers
        responses = self.net.simulate(X, latency=latency,
                                      return_psps=return_psps)
        # First output spike times
        if return_psps:
            taus = self.first_spikes(responses[0][-1])
        else:
            taus = self.first_spikes(responses[-1])
        # Categorical probability distribution at output layer
        cpd = softmax(-taus, self.cpd_scale)
        return (responses, taus, cpd)

    def first_spikes(self, spike_trains):
        """
        Find first spike time for each spike train. If a spike train is empty,
        set first spike to some default value (>>T).
        """
        taus = np.full(len(spike_trains), self.tau_max)
        for idx, spike_train in enumerate(spike_trains):
            if len(spike_train) > 0:
                taus[idx] = spike_train[0]
        return taus

    def cost_derivative(self, cpd, y):
        """
        Vector of cost derivatives w.r.t. 1st spikes: dC / du.
        dtau / du is taken to be -1.
        """
        return cpd - y

    def predict(self, X):
        """
        Model prediction on an input sample, returning class index.
        """
        taus = self.feedforward(X)[1]
        p = np.argmin(taus)  # Predicted class index
        # Ensure unique class prediction
        if np.sum(taus == taus[p]) == 1:
            return p
        else:
            return np.nan

    def loss(self, data):
        """
        Average cross entropy of network over a batch of data, consisting of a
        list of 2-tuples [(X_1, y_1), ...].
        """
        H = 0.  # Cross entropy
        num_samples = len(data)
        for sample in data:
            X, y = sample
            cpd = self.feedforward(X)[2] + 1e-16  # Model distribution
            H -= np.sum(y * np.log(cpd), 0)
        return H / num_samples


def softmax(z, coeff=1.):
    """
    Softmax activation for weighted inputs z:
        array, shape (num_units, num_samples).
    Returns scores for each sample:
        array, shape (num_units, num_samples).
    """
    z_ = z - np.max(z, 0)
    exp_z = np.exp(coeff * z_)
    return exp_z / np.sum(exp_z, 0)
