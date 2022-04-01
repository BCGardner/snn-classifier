"""
Learning schedules for network training via SGD.

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

from argparse import Namespace
import numpy as np

from ..parameters import ParamSet


class Solver():
    """
    Abstract solver class. Computes weight changes based on learning schedule.
    """
    def __init__(self, weights=[], **kwargs):
        """
        Set default parameters, store list of weight shapes, optionally specify
        warmstart if supported.
        """
        self._w_shapes = [w.shape for w in weights]
        self.prms = ParamSet(kwargs)

    def update(self, **kwargs):
        """
        Update default prms with user-specified values.
        """
        self.prms.overwrite(**kwargs)

    def warmup(self, grad_w):
        """
        Initialise solver state using initial weight gradients.
        """
        pass

    def weight_changes(self, grad_w, **kwargs):
        """
        Returns dW, as determined from grad_w per layer, idxs: [0, L-1].
        """
        raise NotImplementedError


class ConstLR(Solver):
    """
    Constant learning rate.

    Parameters
    ----------
    eta : float
        Const. LRate.
    """
    def __init__(self, weights=[], **kwargs):
        # Defaults
        super(ConstLR, self).__init__(weights, eta=1.)
        # Update with user-specified
        self.update(**kwargs)

    def weight_changes(self, grad_w, num_samples=1, **kwargs):
        delta_ws = [None] * len(grad_w)
        for idx, dC in enumerate(grad_w):
            delta_ws[idx] = -self.prms['eta'] / num_samples * dC
        return delta_ws


class RMSProp(Solver):
    """
    Adaptive learning rate at each synapse, based on a moving average of the
    RMS of weight gradients.

    Parameters
    ----------
    decay : float
        Decay rate controlling EMA of squared weight grads.
    alpha : float
        LRate scaling factor.
    epsilon : float
        Regularisation prm for numerical stability.
    """
    def __init__(self, weights=[], **kwargs):
        # Defaults
        super(RMSProp, self).__init__(weights, decay=0.9, alpha=0.1,
                                      epsilon=1e-8)
        # Update with user-specified
        self.update(**kwargs)
        # EMA of squared weight gradients up to current iteration.
        self.grad_w_av_sq = [np.zeros(w_shape) for w_shape in self._w_shapes]

    def warmup(self, grad_w):
        self.grad_w_av_sq = [dC**2 for dC in grad_w]

    def weight_changes(self, grad_w, **kwargs):
        prms = Namespace(**self.prms)
        delta_ws = [None] * len(grad_w)
        for idx, dC in enumerate(grad_w):
            self.grad_w_av_sq[idx] = prms.decay * \
                self.grad_w_av_sq[idx] + (1. - prms.decay) * dC**2
            delta_ws[idx] = -prms.alpha / np.sqrt(self.grad_w_av_sq[idx] +
                                                  prms.epsilon) * dC
        return delta_ws


class Adam(Solver):
    """
    Adaptive learning rate at each synapse, based on RMSProp with momentum.

    Parameters
    ----------
    betas : tuple
        Two decay rates controlling EMA of first and second grad_w moments.
    alpha : float
        LRate scaling factor.
    epsilon : float
        Regularisation prm for numerical stability.
    """
    def __init__(self, weights=[], **kwargs):
        # Defaults
        super(Adam, self).__init__(weights, betas=(0.9, 0.999), alpha=0.1,
                                   epsilon=1e-8)
        # Update with user-specified
        self.update(**kwargs)
        # EMA of 1st and 2nd grad_w moments up to current iteration.
        # m : 1st moments (mean / EMA of gradients)
        # v : 2nd moments (uncentered variance / EMA of squared gradients)
        self.m = [np.zeros(w_shape) for w_shape in self._w_shapes]
        self.v = [np.zeros(w_shape) for w_shape in self._w_shapes]
        # Keep track of num. weight changes
        self._iters = 0

    def warmup(self, grad_w):
        self.m = [dC for dC in grad_w]
        self.v = [dC**2 for dC in grad_w]

    def weight_changes(self, grad_w, **kwargs):
        prms = Namespace(**self.prms)
        delta_ws = [None] * len(grad_w)
        self._iters += 1  # Update for next iteration
        for idx, dC in enumerate(grad_w):
            self.m[idx] = prms.betas[0] * self.m[idx] + \
                (1. - prms.betas[0]) * dC
            self.v[idx] = prms.betas[1] * self.v[idx] + \
                (1. - prms.betas[1]) * dC**2
            m_unbias = self.m[idx] / (1. - prms.betas[0]**self._iters)
            v_unbias = self.v[idx] / (1. - prms.betas[1]**self._iters)
            delta_ws[idx] = -(prms.alpha * m_unbias) / \
                (np.sqrt(v_unbias) + prms.epsilon)
        return delta_ws
