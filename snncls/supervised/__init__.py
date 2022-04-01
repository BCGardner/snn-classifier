"""
Training models are defined here for spiking neural networks.

This module includes:
    - SoftmaxClf : First output spikes are passed through a softmax function.
"""

from .softmax_clf import SoftmaxClf

__all__ = ['SoftmaxClf']
