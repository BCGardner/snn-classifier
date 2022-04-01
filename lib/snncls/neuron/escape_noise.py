"""
Escape noise neuron model.

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


class LinearRate():
    """Linear escape rate"""
    def __init__(self, theta, scale=1.0):
        self.theta = theta
        self.scale = scale

    def activation(self, u):
        return self.scale * (u - self.theta)

    def grad_activation(self, u):
        return self.scale


class ExpRate():
    """Exponential escape rate"""
    def __init__(self, theta, rho_0=0.01, thr_width=1.0):
        self.theta = theta
        self.rho_0 = rho_0
        self.thr_width = thr_width

    def activation(self, u):
        return self.rho_0 * np.exp((u - self.theta) / self.thr_width)

    def grad_activation(self, u):
        return self.activation(u) / self.thr_width
