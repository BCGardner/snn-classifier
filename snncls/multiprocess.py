"""
Multiprocessing routines.

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

import multiprocessing as mp
import time
import itertools
import copy
import psutil
from argparse import Namespace

import numpy as np

from snncls.decorators import deprecated


def args_map(worker_func, args, prm_labels=None, report=True):
    """
    Maps parsed arguments to a pool of workers, using one list of arguments.
    """
    # Sweeped parameters
    if prm_labels is not None:
        prm_vals = [vars(args)[k] for k in prm_labels]
        for vals in prm_vals:
            if len(vals) == 0:
                raise ValueError('No sweeped prms.')
    else:
        prm_vals = [[None]]
    grid_shape = tuple([len(i) for i in prm_vals])
    num_args = np.prod(grid_shape + (args.num_runs,))

    # Setup pool of workers
    if args.num_proc is not None:
        pool = mp.Pool(processes=args.num_proc, maxtasksperchild=None)
    else:
        num_phys_cores = psutil.cpu_count(logical=False)
        pool = mp.Pool(processes=num_phys_cores, maxtasksperchild=None)
    if report:
        print('Num workers: {}'.format(pool._processes))
        t_start = time.time()

    # Initialise seeds
    if args.seed is not None:
        seeds = range(args.seed, num_args + args.seed)
    else:
        seeds = list(itertools.repeat(None, num_args))

    # Create arguments list, len (num_prms0 x num_prms1 x ... x num_runs)
    args_dict = vars(args)
    wkr_args = []
    for idx, vals in enumerate(itertools.product(*prm_vals)):
        prmset_dict = dict(args_dict)
        if prm_labels is not None:
            for k, v in zip(prm_labels, vals):
                prmset_dict[k] = v
        # Repeated runs per parameter point
        args_runs = []
        for i in range(args.num_runs):
            d = copy.deepcopy(prmset_dict)
            args_runs.append(Namespace(**d))
        # Assign (unique) seeds
        offset = idx * args.num_runs
        for args_run, seed in zip(args_runs,
                                  seeds[offset:offset+args.num_runs]):
            args_run.seed = seed
        # Assign repeated args
        wkr_args += args_runs
        if prm_labels is not None and report:
            coord = np.unravel_index(idx, grid_shape)
            print('{}: {}'.format(coord, [prmset_dict[label]
                                          for label in prm_labels]))

    # Assign parallel jobs
    results = pool.map(worker_func, wkr_args)

    # Cleanup
    pool.close()
    pool.join()
    if report:
        t_elapsed = time.time() - t_start
        print('{:.2f} s'.format(t_elapsed))

    # Gather results as array of lists
    results = [results[i:i+args.num_runs] for i in
               range(0, len(results), args.num_runs)]
    results_gtr = np.empty(grid_shape, dtype=object)
    for idx, result in enumerate(results):
        coord = np.unravel_index(idx, grid_shape)
        results_gtr[coord] = result
    return results_gtr


@deprecated
def prms_map(worker_func, prm_vals, prm_labels, args_com, seed=None,
             num_runs=1, report=True, num_proc=None):
    """
    Maps a pool of workers to worker_func, using one list of arguments.
    """
    # Task parameters
    grid_shape = tuple([len(i) for i in prm_vals])
    num_args = np.prod(grid_shape + (num_runs,))

    # Setup pool of workers
    if num_proc is not None:
        pool = mp.Pool(processes=num_proc, maxtasksperchild=None)
    else:
        num_phys_cores = psutil.cpu_count(logical=False)
        pool = mp.Pool(processes=num_phys_cores, maxtasksperchild=None)
    if report:
        print('Num workers: {}'.format(pool._processes))
        t_start = time.time()

    # Initialise seeds
    if seed is not None:
        seeds = range(seed, num_args + seed)
    else:
        seeds = list(itertools.repeat(None, num_args))

    # Create arguments list, len (num_prms0 x num_prms1 x ... x num_runs)
    args = []
    for idx, vals in enumerate(itertools.product(*prm_vals)):
        arg_dict = dict(args_com)
        for k, v in zip(prm_labels, vals):
            arg_dict[k] = v
        # Repeated runs per parameter point
        arg_dicts = [copy.deepcopy(arg_dict) for i in range(num_runs)]
        # Assign (unique) seeds
        offset = idx * num_runs
        for d, s in zip(arg_dicts, seeds[offset:offset+num_runs]):
            d['seed'] = s
        # Assign repeated args
        args += arg_dicts
        if report:
            coord = np.unravel_index(idx, grid_shape)
            print('{}: {}'.format(coord,
                                  [arg_dict[label] for label in prm_labels]))

    # Assign parallel jobs
    results = pool.map(worker_func, args)

    # Cleanup
    pool.close()
    pool.join()
    if report:
        t_elapsed = time.time() - t_start
        print('{:.2f} s'.format(t_elapsed))

    # Gather results as array of lists
    results = \
        [results[i:i+num_runs] for i in range(0, len(results), num_runs)]
    results_gtr = np.empty(grid_shape, dtype=object)
    for idx, result in enumerate(results):
        coord = np.unravel_index(idx, grid_shape)
        results_gtr[coord] = result
    return results_gtr


@deprecated
def param_sweep(worker_func, prm_vals, prm_labels, args_com, seed=None,
                num_runs=1, report=True, num_proc=None):
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
    num_proc : int, optional
               Maximum number of processes to run concurrently.

    Output
    ------
    return : array, shape (num_prms0, num_prms1, ..., num_prmsN)
             Gathered results, containing outputs of len (num_runs) per grid
             coord.
    """
    # Task parameters
    grid_shape = tuple([len(i) for i in prm_vals])
    grid_ranges = [range(i) for i in grid_shape]

    # Setup pool of workers
    if num_proc is not None:
        pool = mp.Pool(processes=num_proc, maxtasksperchild=None)
    else:
        num_phys_cores = psutil.cpu_count(logical=False)
        pool = mp.Pool(processes=num_phys_cores, maxtasksperchild=None)
    if report:
        print('Num workers: {}'.format(pool._processes))
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
        arg_set = [copy.deepcopy(arg_dict) for i in range(num_runs)]
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
        print('{:.2f} s'.format(t_elapsed))

    # Gather results as array of lists
    results_gtr = np.empty(grid_shape, dtype=object)
    for coord, result in np.ndenumerate(results):
        results_gtr[coord] = result.get()
    return results_gtr
