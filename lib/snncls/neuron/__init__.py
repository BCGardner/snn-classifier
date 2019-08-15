"""
Standard neuron models are defined here for spiking neural networks.

This module includes:
    - SRM (simplified)
"""

from .models import SRM
from .escape_noise import LinearRate, ExpRate

__all__ = ['SRM',
           'LinearRate',
           'ExpRate']
