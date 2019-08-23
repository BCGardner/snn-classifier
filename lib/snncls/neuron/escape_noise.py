#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Apr 2018

@author: BG

Escape noise neuron model.
"""

import numpy as np


class LinearRate(object):
    """Linear escape rate"""
    def __init__(self, theta, scale=1.0):
        self.theta = theta
        self.scale = scale

    def activation(self, u):
        return self.scale * (u - self.theta)

    def grad_activation(self, u):
        return self.scale


class ExpRate(object):
    """Exponential escape rate"""
    def __init__(self, theta, rho_0=0.01, thr_width=1.0):
        self.theta = theta
        self.rho_0 = rho_0
        self.thr_width = thr_width

    def activation(self, u):
        return self.rho_0 * np.exp((u - self.theta) / self.thr_width)

    def grad_activation(self, u):
        return self.activation(u) / self.thr_width
