#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created Dec 2018

@author: BG

Multiprocessing routines.
"""

import multiprocessing as mp
import time
import itertools
import copy

import numpy as np


def param_sweep(worker_func, prm_vals, prm_labels, args_com, seed=None,
                num_runs=1, report=True):
    """
    Maps a pool of workers to the provided function, sweeping over a list
    of parameter sets as an argument. The grid of parameter coords are run
    repeated times, outputting an ndarray of gathered results.

    Inputs
    ------
    worker_func : function
                  Handle on the worker function, with a dict as its argument.
    prm_vals : list, len (num_prms)
               Contains lists of prm choices, identified by one-one association
               with prm_labels.
    prm_labels : list, len (num_prms)
                 Contains labels of parameters sweeped over.
    args_com : dict
               Common arguments assigned to the worker func on each run.
    seed : int, optional
           Reference seed value for reproducable results.
    num_runs : int, optional
               Number of repeated runs per grid coord.
    report : bool, optional
             Report status and runtime.

    Output
    ------
    return : array, shape (num_prms0, num_prms1, ..., num_prmsN)
             Gathered results, containing outputs of len (num_runs) per grid
             coord.
    """
    # Task parameters
    grid_shape = tuple([len(i) for i in prm_vals])
    grid_ranges = [xrange(i) for i in grid_shape]

    # Setup pool of workers
    pool = mp.Pool(processes=None, maxtasksperchild=None)
    if report:
        print 'Num workers: {}'.format(pool._processes)
        t_start = time.time()

    # Initialise seeds
    dims = grid_shape + (num_runs,)
    if seed is not None:
        seeds = (np.arange(np.prod(dims)) + seed).reshape(dims)
    else:
        seeds = np.full(dims, None)

    # Assign tasks
    results = np.empty(grid_shape, dtype=object)
    for coord in itertools.product(*grid_ranges):
        arg_dict = dict(args_com)
        # Set shared parameter values over repeated runs
        for idx, label in enumerate(prm_labels):
            arg_dict[label] = prm_vals[idx][coord[idx]]
        arg_set = [copy.deepcopy(arg_dict) for i in xrange(num_runs)]
        # Set (unique) seeds
        for d, s in zip(arg_set, seeds[coord]):
            d['seed'] = s
        # Assign parallel jobs
        results[coord] = pool.map_async(worker_func, arg_set)
        if report:
            print('{}: {}'.format(coord,
                                  [arg_dict[label] for label in prm_labels]))
    # Wait for results
    for coord in itertools.product(*grid_ranges):
        results[coord].wait()
        if report:
            print('Completed: {}'.format(coord))
    if report:
        t_elapsed = time.time() - t_start
        print '{:.2f} s'.format(t_elapsed)

    # Gather results as array of lists
    results_gtr = np.empty(grid_shape, dtype=object)
    for coord, result in np.ndenumerate(results):
        results_gtr[coord] = result.get()
    return results_gtr
