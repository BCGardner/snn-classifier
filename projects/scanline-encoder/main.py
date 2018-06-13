#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 10 15:45:24 2018

@author: BG
"""

import numpy as np
import matplotlib.pyplot as plt

from lib import loader, scanner, plotter
import lib.helpers as hp


# Options
prompt = False
wait = 0.05

# Data
fname = 'data/digits.data'
X, Y = loader.load_data(fname)
bounds = X[0].shape
img_scan = X[1116]

# Scanner
duration = 10.
dt = 0.1
times = np.arange(0., duration, dt)
#line_eqs = [(-2, 3), (2, 2)]
# Matched to Intel's Loihi encoder
line_eqs = [(0.857, 4.00), (1.00, -3.00), (-80.0, 1060.),
            (-0.615, 17.8), (-1.63, 13.0)]

scanners = []
for eq in line_eqs:
    scanners.append(scanner.Scanner(eq, bounds, duration))

# Recorder
rec = dict()
rec['r'] = [np.full((len(times), 2), np.nan) for i in xrange(len(scanners))]
rec['i'] = [np.full(len(times), np.nan) for i in xrange(len(scanners))]
rec['addr'] = [[] for i in xrange(len(scanners))]  # Stores [(row, col), ...]

# Scan data
for step, t in enumerate(times):
    for idx, scan in enumerate(scanners):
        rec['r'][idx][step, :] = scan.get_pos()
        rec['i'][idx][step], addr = scan.read(img_scan, True)
        rec['addr'][idx].append(addr)
        scan.translate(dt)

# Plot
playbk = plotter.Playback(bounds, times, wait)
playbk.play(rec, scanners, img_scan, prompt)
