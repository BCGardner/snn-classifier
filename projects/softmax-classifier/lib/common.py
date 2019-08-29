#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Shared routines between main scripts.

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
import json
import argparse


def default_opt(data_id):
    """
    Default parameters for each data set, based on latency encoding.
    """
    opt = dict()
    # Set initial firing rates to one spike / pattern
    if data_id == 'iris':
        # 12 encoding nrns per feature (total 48): on average, ~ 24 % of the
        # input nrns transmit a spike per sample.
        num_epochs = 60  # 100 epochs / weight updates (1 iteration / epoch)
        opt.update({'w_h_init': (0., 4.),
                    'w_o_init': (0., 2.),
                    'neurons_f': 12})
    elif data_id == 'wisc':
        # 7 encoding nrns per feature (total 63): on average, ~ 35 % of the
        # input nrns transmit a spike per sample.
        num_epochs = 90  # 375 epochs: 1500 iterations (5 iterations / epoch)
        opt.update({'w_h_init': (0., 2.2),
                    'w_o_init': (0., 2.),
                    'neurons_f': 7})  # Aim for almost 66 % silent input nrns
    elif data_id == 'wisc_nmf':
        num_epochs = 20  # 300 epochs: 1500 iterations (5 iterations / epoch)
        opt.update({'neurons_f': 6})
    elif data_id == 'mnist':
        # 1 encoding nrns per feature (total 784): on average, ~ 15 % of the
        # input nrns transmit a spike per sample.
        num_epochs = 20
        opt.update({'w_h_init': (0., 0.4),
                    'w_o_init': (0., 1.6),
                    'neurons_f': 1})  # One nrn per input pixel
    else:
        raise KeyError(data_id)
    return opt, num_epochs


def save_data(data, args, prm_labels=None, basename=None):
    """
    Write gathered results to out/, optionally with basename prefix.

    data : array_like[, shape (num_prms0, num_prms1, ...)]
        Gathered results.
    args : NameSpace or dict
        A namespace or dict containing sim. arguments.
    prm_labels : list, optional
        Prm labels sweeped over.
    basename : string, optional
        Simulation identifier.
    """
    if isinstance(args, argparse.Namespace):
        args = vars(args)
    if basename is not None:
        fname = '{}_{}'.format(basename, args['fname'])
    else:
        fname = args['fname']
    np.save('out/{}'.format(fname), data)
    # Config
    if prm_labels is not None:
        args['prm_labels'] = prm_labels
    with open('out/{}_cfg.json'.format(fname),
              'w') as h:
        json.dump(args, h, indent=4)
