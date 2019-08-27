#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
This script transforms pixel images into sets of encoding spike trains. The
default loaded dataset is MNIST. The scanline encoding process can optionally
be viewed as a dynamic plot.

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
import matplotlib.pyplot as plt
import argparse
import json

from snncls import dataset_loader, helpers
from snncls.scanline import scanner, neuron
from lib import plotter, common

# Matched to Intel's Loihi encoder for 28x28
line_eqs_loihi = [(0.844, 6.27), (1.00, -4.62), (-250., 5733.),
                  (-1.654, 23.156), (-0.59, 30.509)]


def main(opt):
    # Setup
    rng = np.random.RandomState(opt.seed)
    # Data
    X, y = dataset_loader.load_data_file(opt.data_id)
    bounds = (int(np.sqrt(len(X[0]))),) * 2
    idxs = np.arange(len(X))
    # Normalisation
    data_min = np.min(X)
    data_range = np.max(X) - data_min
    X = (X - data_min) / data_range
    # [Randomise data selection]
    if opt.randomise:
        rng.shuffle(idxs)
    # Data selection
    X, y = X[idxs][:opt.num_cases], y[idxs][:opt.num_cases]
    # Scanners
    duration = opt.duration
    dt = 0.1
    times = np.arange(0., duration, dt)
    num_steps = len(times)

    # Line equations
    if opt.distr == 'loihi':
        line_eqs = line_eqs_loihi
    elif opt.distr == 'edges':
        line_eqs = common.generate_eqs(opt.scans, scale=opt.scale, rng=rng)
    elif opt.distr == 'ctr':
        line_eqs = common.generate_eqs_ctr(opt.scans, scale=opt.scale, rng=rng)
    # Setup scanners
    scanners = []
    for eq in line_eqs:
        scanners.append(scanner.Scanner(eq, bounds, duration))
    num_scanners = len(scanners)
    # Neurons
    nrns = []
    for i in xrange(num_scanners):
        if opt.nrn == 'lif':
            nrns.append(neuron.LIF(dt, R=opt.resistance, tau_m=opt.tau_m,
                                   t_abs=opt.t_abs, v_thr=opt.v_thr))
        elif opt.nrn == 'izh':
            # a=0.2: 5 ms recovery time
            # d=12: suppress late spiking
            nrns.append(neuron.Izhikevich(dt, a=.4, d=12.))
        else:
            raise ValueError('Invalid neuron type')
    # Recorder (one sample)
    rec = dict()
    if opt.plot:
        rec['r'] = [np.full((num_steps, 2), np.nan)
                    for i in xrange(num_scanners)]  # Scanner positions
        rec['i'] = [np.full(num_steps, np.nan)
                    for i in xrange(num_scanners)]  # Pixel intensities
        rec['addr'] = [[] for i in xrange(num_scanners)]  # Scanned pixel addrs
        rec['v'] = np.full((num_steps, num_scanners), np.nan)  # Nrn voltages
    # Record spike trains for each sample
    spike_trains = [[np.array([]) for i in xrange(num_scanners)]
                    for j in xrange(opt.num_cases)]

    # Scan each sample
    for idx_im, x in enumerate(X):
        img_scan = x.reshape(bounds)
        # Reset scanners and associated neurons
        for scan, nrn in zip(scanners, nrns):
            scan.reset()
            nrn.reset()
        for step, t in enumerate(times):
            for idx_sc, scan in enumerate(scanners):
                # Read
                stimulus = scan.read(img_scan)
                # Record
                if opt.plot:
                    rec['r'][idx_sc][step, :] = scan.get_pos()
                    rec['i'][idx_sc][step], addr = scan.read(img_scan, True)
                    rec['addr'][idx_sc].append(addr)
                    rec['v'][step, idx_sc] = nrns[idx_sc].get_v()
                # Update
                fired = nrns[idx_sc].update(stimulus)
                scan.translate(dt)
                # Record nrn
                if fired:
                    spike_time = np.around(t, decimals=1)
                    spike_trains[idx_im][idx_sc] = \
                        np.append(spike_trains[idx_im][idx_sc], spike_time)
    if opt.plot:
        rec['spikes'] = spike_trains[-1]
    # [Plot first sample]
    if opt.plot:
        playbk = plotter.Playback(bounds, times, duration, opt.wait)
        playbk.play_nrns(rec, img_scan)
#        f, ax = plt.subplots()
#        for nrn, v in enumerate(rec['v'].T):
#            thr_idxs = np.round(rec['spikes'][nrn] / dt).astype(int)
#            v[thr_idxs] = 1.
#            ax.plot(times, v + nrn, label=str(nrn), linewidth=2)
#            ax.set_ylabel('Encoder #')
#            ax.set_ylim(0, num_scanners)
#            ax.set_xlabel('Time (ms)')
#            ax.set_xlim(0., opt.duration)
        if opt.fname is not None:
            plt.savefig('out/{}.pdf'.format(opt.fname), bbox_inches='tight',
                        dpi=300)
        else:
            plt.show()
    return spike_trains, y


if __name__ == "__main__":
    parser = argparse.ArgumentParser("scanline encoder")
    # Input data
    parser.add_argument("--data_id", type=str, default='mnist')
    # Images
    parser.add_argument("-r", "--randomise", action="store_true",
                        help="optional, randomise data selection")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("-n", "--num_cases", type=int, default=10,
                        help="size of samples selected")
    # Scanners
    parser.add_argument("-d", "--duration", type=float, default=9.,
                        help="scan duration")
    # Random distr.
    parser.add_argument("--distr", type=str, default='ctr',
                        help="line distributions: 'loihi', 'edges', 'ctr'")
    parser.add_argument("-s", "--scans", type=int, default=6,
                        help="number of randomly-oriented scanlines")
    parser.add_argument("--scale", type=float, default=0.25,
                        help="scale parameter of normal distribution")
    # Neurons
    parser.add_argument("--nrn", type=str, default='lif',
                        help="neuron types: 'lif', 'izh'")
    parser.add_argument("-R", "--resistance", type=float, default=10.,
                        help="LIF prm")
    parser.add_argument("--tau_m", type=float, default=3.,
                        help="LIF prm")
    parser.add_argument("--t_abs", type=float, default=1.,
                        help="LIF prm")
    parser.add_argument("--v_thr", type=float, default=1.,
                        help="LIF prm")
    # Plotting
    parser.add_argument("-p", "--plot", action="store_true",
                        help="plot last sample")
#    parser.add_argument("-p", "--plot", type=bool, default=True)
    parser.add_argument("-w", "--wait", type=float, default=0.02,
                        help="plot speed")
    # Save preprocessed data
    parser.add_argument("--fname", type=str, default=None)
    args = parser.parse_args()

    # Type safety
    assert args.distr in ['loihi', 'edges', 'ctr']

    data = main(args)
    # Class frequencies
    # print 'class counts:' + \
    #     str([np.sum(data[1] == i) for i in np.unique(data[1])])

    # Save results
    if args.fname is not None:
        helpers.save_data(data, 'out/{}.pkl.gz'.format(args.fname))
        # Config
        with open('out/{}_cfg.json'.format(args.fname), 'w') as h:
            json.dump(vars(args), h, indent=4)
