"""
Scanline encoder models used to convert features into spike trains.
Encoders are based on spiking neuron models.

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


class Neuron():
    """
    Abstract encoding neuron.
    """
    def __init__(self, dt=0.1):
        """
        Default setup.
        """
        self.dt = dt  # Integration time step
        self._params = dict()  # Default Parameters
        self._v = 0.  # Voltage (mV)

    def reset(self):
        """
        Reset neuron's state.
        """
        raise NotImplementedError

    def update(self, current=0.):
        """
        Advances neuron's state by dt[, integrate current], return bool output.
        """
        raise NotImplementedError

    def get_v(self):
        return self._v

    def set_params(self, **kwargs):
        """
        Change neuron's default parameters.
        """
        for key, val in kwargs.items():
            if key in self._params:
                self._params[key] = val


class LIF(Neuron):
    """
    Leaky-Integrate-and-Fire type neuron.
    """
    def __init__(self, dt=0.1, **kwargs):
        """
        Update default LIF neuron parameters, reset dynamics.
        """
        super(LIF, self).__init__(dt)
        self._params = {'R': 10.,      # Membrane resistance
                        'tau_m': 3.,  # Membrane time constant
                        't_abs': 1.,  # Absolute refractory time
                        'v_thr': 1.}  # Threshold voltage
        self.set_params(**kwargs)
        assert(self._params['v_thr'] > 0.)
        # Constants
        self._coeff_0 = np.exp(-self.dt / self._params['tau_m'])
        self._coeff_1 = 1. - np.exp(-self.dt / self._params['tau_m'])
        # Setup
        self.reset()

    def reset(self):
        self._v = 0.  # Voltage (mV)
        self._t_refr = self._params['t_abs']  # Refractory time

    def update(self, current=0.):
        fired = False
        if self._t_refr >= self._params['t_abs']:
            # Integrate and advance by dt
            self._v = self._coeff_0 * self._v \
                + self._coeff_1 * self._params['R'] * current
            # Check firing condition
            if self._v >= self._params['v_thr']:
                fired = True
                self._v = 0.
                self._t_refr = self.dt
        else:
            self._t_refr += self.dt
        return fired


class Izhikevich(Neuron):
    """
    Non-linear neuron type.
    """
    def __init__(self, dt=0.1, **kwargs):
        """
        Default params: phasic spiking type (signal step-current onset).
        """
        super(Izhikevich, self).__init__(dt)
        self._params = {'a': 0.4,   # Recovery variable (u) time scale
                        'b': 0.25,  # u subthreshold sensitivity to v
                        'c': -65.,  # After-spike reset of v
                        'd': 12.}   # After-spike reset of u
        self.set_params(**kwargs)
        # Setup
        self.reset()

    def reset(self):
        self._v = -64.
        self._u = self._params['b'] * self._v

    def update(self, current=0.):
        fired = False
        # Scale up the current
        current *= 40.
        # Integrate and advance by dt
        self._v += self.dt * (0.04 * self._v**2 + 5 * self._v
                              + 140. - self._u + current)
        self._u += self.dt * self._params['a'] \
            * (self._params['b'] * self._v - self._u)
        # Check firing condition
        if self._v >= 30.:
            fired = True
            self._v = self._params['c']
            self._u += self._params['d']
        return fired
