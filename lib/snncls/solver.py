#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Jul 2019

@author: BG

Learning schedules for network training.
"""

from __future__ import division

from argparse import Namespace
import numpy as np

from snncls.parameters import ParamSet


class Solver(object):
    """
    Abstract solver class. Computes weight changes based on learning schedule.
    """
    def __init__(self, **kwargs):
        """
        Default parameters.
        """
        self.prms = ParamSet(kwargs)

    def update(self, **kwargs):
        """
        Update default prms with user-specified, cast to Namespace for
        convenience.
        """
        self.prms.overwrite(**kwargs)

    def weight_change(self, grad_w):
        """
        Returns dW.

        Inputs
        ------
        grad_w : list
            List of weight gradient arrays per layer, idxs: [0, L-1]

        Output
        ------
        Return : list
            List of weight changes per layer, idxs: [0, L-1].
        """
        raise NotImplementedError


class ConstLR(Solver):
    """
    Constant learning rate.
    """
    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        eta : float
            Const. LRate.
        """
        # Defaults
        super(ConstLR, self).__init__(eta=1.)
        # Update with user-specified
        self.update(**kwargs)

    def weight_change(self, grad_w, num_samples=1):
        return [-self.prms['eta'] / num_samples * dC for dC in grad_w]


class RMSProp(Solver):
    """
    Adaptive learning rate at each synapse, based on a moving average of the
    RMS of weight gradients.
    """
    def __init__(self, weights=[],  **kwargs):
        """
        Inputs
        ------
        weights : list
            List of net. weights, idxs: [0, L-1].

        Parameters
        ----------
        decay : float
            Decay rate controlling EMA of squared weight grads.
        alpha : float
            LRate scaling factor.
        epsilon : float
            Regularisation prm for numerical stability.
        """
        # Defaults
        super(RMSProp, self).__init__(decay=0.9, alpha=0.1, epsilon=1e-8)
        # Update with user-specified
        self.update(**kwargs)
        # EMA of squared weight gradients up to current iteration.
        self.grad_w_av_sq = [np.zeros(w.shape) for w in weights]

    def weight_change(self, grad_w):
        prms = Namespace(**self.prms)  # Convenience
        delta_ws = [None] * len(self.grad_w_av_sq)
        for idx, dC in enumerate(grad_w):
            self.grad_w_av_sq[idx] = prms.decay * \
                self.grad_w_av_sq[idx] + (1. - prms.decay) * dC**2
            delta_ws[idx] = -prms.alpha / np.sqrt(self.grad_w_av_sq[idx] +
                                                  prms.epsilon) * dC


class Adam(Solver):
    """
    Adaptive learning rate at each synapse, based on RMSProp with momentum.
    """
    def __init__(self, weights=[],  **kwargs):
        """
        Inputs
        ------
        weights : list
            List of net. weights, idxs: [0, L-1].

        Parameters
        ----------
        betas : tuple
            Two decay rates controlling EMA of first and second grad_w moments.
        alpha : float
            LRate scaling factor.
        epsilon : float
            Regularisation prm for numerical stability.
        """
        # Defaults
        super(Adam, self).__init__(betas=(0.9, 0.999), alpha=0.1,
                                   epsilon=1e-8)
        # Update with user-specified
        self.update(**kwargs)
        # EMA of 1st and 2nd grad_w moments up to current iteration.
        # m : 1st moments (mean / EMA of gradients)
        # v : 2nd moments (uncentered variance / EMA of squared gradients)
        self.m = [np.zeros(w.shape) for w in weights]
        self.v = [np.zeros(w.shape) for w in weights]
