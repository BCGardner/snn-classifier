"""
Spiking neural networks (SNNs) are defined here.

This module includes SNNs optimised specifically for the simplified SRM model,
and a more generalised multilayer SNN supporting mixed cell parameters.
"""

from .multilayer_srm import MultilayerSRM, MultilayerSRMSub

__all__ = ['MultilayerSRM',
           'MultilayerSRMSub']
