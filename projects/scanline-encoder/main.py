#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Demonstration of scanline-encoding as a dynamic plot. Low resolution digits
are used in this example, obtained from:
https://archive.ics.uci.edu/ml/machine-learning-databases/semeion/

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

from lib import loader, scanner, neuron, plotter
import lib.helpers as hp


def main(opt):
    # Data
    fname = 'data/digits.data'
    X, Y = loader.load_data(fname)
    bounds = X[0].shape
    img_scan = X[opt.index]  # Image to scan

    # Scanners
    duration = opt.duration
    dt = 0.1
    times = np.arange(0., duration, dt)
    num_steps = len(times)
    #line_eqs = [(-2, 3), (2, 2)]

    # Matched to Intel's Loihi encoder
    line_eqs = [(0.857, 4.00), (1.00, -3.00), (-80.0, 1060.),
                (-0.615, 17.8), (-1.63, 13.0)]
    scanners = []
    for eq in line_eqs:
        scanners.append(scanner.Scanner(eq, bounds, duration))
    num_scanners = len(scanners)
    # Neurons
    nrns = []
    for i in xrange(num_scanners):
        if opt.nrn == 'lif':
            nrns.append(neuron.LIF(dt, R=10.))
        elif opt.nrn == 'izh':
            # a=0.2: 5 ms recovery time
            # d=12: suppress late spiking
            nrns.append(neuron.Izhikevich(dt, a=.4, d=12.))
        else:
            raise ValueError('Invalid neuron type')
    # Recorder
    rec = dict()
    rec['r'] = [np.full((num_steps, 2), np.nan) for i in xrange(num_scanners)]
    rec['i'] = [np.full(num_steps, np.nan) for i in xrange(num_scanners)]
    rec['addr'] = [[] for i in xrange(num_scanners)]  # Stores [(row, col), ...]
    rec['v'] = np.full((num_steps, num_scanners), np.nan)  # Nrn voltages
    rec['spikes'] = [np.array([]) for i in xrange(num_scanners)]

    # Scan data
    for step, t in enumerate(times):
        for idx, scan in enumerate(scanners):
            # Read
            stimulus = scan.read(img_scan)
            # Record
            rec['r'][idx][step, :] = scan.get_pos()
            rec['i'][idx][step], addr = scan.read(img_scan, True)
            rec['addr'][idx].append(addr)
            rec['v'][step, idx] = nrns[idx].get_v()
            # Update
            fired = nrns[idx].update(stimulus)
            scan.translate(dt)
            # Record nrn
            if fired:
                rec['spikes'][idx] = np.append(rec['spikes'][idx], t)
    # Plot data
    playbk = plotter.Playback(bounds, times, duration, opt.wait)
    #playbk.play(rec, img_scan, opt.prompt)
    playbk.play_nrns(rec, img_scan, opt.prompt)
#    plt.plot(times, rec['v'][:, 2])
    # plt.show()
    return rec


if __name__ == "__main__":
    parser = argparse.ArgumentParser("scanline encoder")
    # Image
    parser.add_argument("-i", "--index", type=int, default=1116,
                        help="image index")
    # Scanners
    parser.add_argument("-d", "--duration", type=float, default=10.,
                        help="scan duration")
    # Neurons
    parser.add_argument("-n", "--nrn", type=str, default='lif',
                        help="neuron types: 'lif', 'izh'")
    # Plotting
    parser.add_argument("-p", "--prompt", action="store_true",
                        help="enable user prompts")
    parser.add_argument("-w", "--wait", type=float, default=0.05,
                        help="plot speed")
    parser.add_argument("--fname", type=str, default=None)
    args = parser.parse_args()

    rec = main(args)

    if args.fname is not None:
        fig = plt.gcf()
        fig.savefig('out/{}.pdf'.format(args.fname), bbox_inches='tight',
                    dpi=300)
