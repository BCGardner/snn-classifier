#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on June 2018

@author: BG

Helper functions.
"""

import gzip
import cPickle as pkl


def save_data(data, filename):
    """
    Save data as gzipped pickle file.
    """
    with gzip.open(filename, 'wb') as f:
        pkl.dump(data, f)


def load_data(path):
    """
    Load data saved as gzipped pickle file.
    """
    with gzip.open(path, 'rb') as f:
        data = pkl.load(f)
    return data
