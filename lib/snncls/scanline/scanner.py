#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 10 19:42:01 2018

@author: BG
@licence: GNU v3
@file One sentence describing the purpouse of this file.
"""

from __future__ import division

import numpy as np


class Scanner(object):
    """
    Scanning element - scans along a given line
    """
    def __init__(self, line_eq, bounds, duration):
        """
        Initialise position of scanner given a line equation.
        Default starting point is highest intercept with boundary.
        Velocity points towards lower intercept.

        Inputs
        ------
        line_eq : 2-tuple
            Format (m, c), where m is gradient, c intercept.
        bounds : int
            Format (num_row, num_col) of image dimensions in pixels.
            rows, cols := y, x.
        """
        self.line_eq = line_eq
        self.x_lim = np.array([0., bounds[1]])
        self.y_lim = np.array([0., bounds[0]])
        self.duration = duration
        self._r = np.full(2, np.nan)  # Current position (x, y)
        self._v = np.full(2, np.nan)  # Velocity (dx/dt, dy/dt)
        # Assert and find intercepts with image boundary
        self.find_intercepts()
        # Find velocity vector towards lower intercept
        self.reset()
        self._v = (self.points[1] - self.points[0]) / duration

    def read(self, img, addr=False):
        """
        Read image value at scanner's position.
        """
        # Find image column (scanner's x position)
        if self._r[0] == self.x_lim[1] and self._v[0] < 0:
            # Special case: starting at max x_limit and decreasing
            col = (self.x_lim[1] - 1).astype(int)
        else:
            col = np.floor(self._r[0]).astype(int)
        # Find image row (scanner's y position)
        if self._r[1] == self.y_lim[1] and self._v[1] < 0:
            # Special case: starting at max y_limit and decreasing
            row = (self.y_lim[1] - 1).astype(int)
        else:
            row = np.floor(self._r[1]).astype(int)
        if addr:
            return img[row, col], (row, col)
        else:
            return img[row, col]

    def get_pos(self):
        """
        Get scanner's current position.
        """
        return self._r.copy()

    def translate(self, dt):
        """
        Translate scanner along line according to a time shift of dt.
        """
        self._r += self._v * dt

    def reset(self):
        """
        Reset scanner to its initial position.
        """
        # Highest intercept
        self._r = self.points[0].copy()

    def find_intercepts(self):
        """
        Find unique intercepts of line_eq with image boundary.
        """
        # x intercepts at each y_limit
        xs = (np.array(self.y_lim) - self.line_eq[1]) / self.line_eq[0]
        # y intercepts at each x_limit
        ys = self.line_eq[0] * np.array(self.x_lim) + self.line_eq[1]
        # Coordinates of intercepts at x and y limits
        self.intercepts = zip(xs, self.y_lim) + zip(self.x_lim, ys)
        # Assert unique intercepts through two boundary points
        points = np.array(self.intercepts)
        assert(len(np.unique(points, axis=0)) == 4)
        mask = np.zeros(4, dtype=bool)
        x_mask = (points[:, 0] >= self.x_lim[0]) \
            & (points[:, 0] <= self.x_lim[1])
        y_mask = (points[:, 1] >= self.y_lim[0]) \
            & (points[:, 1] <= self.y_lim[1])
        mask = x_mask & y_mask
        assert(np.sum(mask) == 2)
        points = points[mask, :]
        # Sort vectors from highest y intercept to lowest
        self.points = points[np.argsort(points[:, 1])[::-1], :]
