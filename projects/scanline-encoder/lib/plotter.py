#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 10 22:46:08 2018

@author: BG
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time


class Playback(object):
    """
    Plays back recordings from scanners.
    """
    def __init__(self, bounds, times, wait=0.05):
        self.x_lim = np.array([0, bounds[1]])
        self.y_lim = np.array([0, bounds[0]])
        self.times = times
        self.wait = wait

    def setup_plot(self, ax, img=None):
        """
        Default fig setup on an axis handle.
        Outline plotted if no img specified.
        """
        if img is None:
            # Plot image outline
            ax.plot([self.x_lim[0], self.x_lim[0]],
                    [self.y_lim[0], self.y_lim[1]], 'k-')
            ax.plot([self.x_lim[0], self.x_lim[1]],
                    [self.y_lim[1], self.y_lim[1]], 'k-')
            ax.plot([self.x_lim[1], self.x_lim[1]],
                    [self.y_lim[1], self.y_lim[0]], 'k-')
            ax.plot([self.x_lim[1], self.x_lim[0]],
                    [self.y_lim[0], self.y_lim[0]], 'k-')
        else:
            ax.imshow(img, cmap='gray', interpolation='none',
                      extent=[0, self.x_lim[1], self.y_lim[1], 0])
#        # Limits
#        ax.set_xlim(self.x_lim + [-.2, .2])
#        ax.set_ylim(self.y_lim + [-.2, .2])
#        # Ticks
#        ax.set_xticks(np.arange(0, self.x_lim[1] + 2, 2))
#        ax.set_yticks(np.arange(0, self.y_lim[1] + 2, 2))
        # Labels
        ax.set_title('{}'.format(self.times[0]))

    def play(self, recs, scanners, img=None, prompt=False):
        """
        Playback scanner recordings.
        """
        num_scanners = len(scanners)
        # Figure setup
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # Setup
        self.setup_plot(ax, img)
        # Plot each scanline and dynamic point
        points = []
        colors = []
        for rec, scan in zip(recs['r'], scanners):
            l, = ax.plot(scan.points[:, 0], scan.points[:, 1], '-')
            colors.append(l.get_color())
            points.append(ax.plot(rec[0, 0], rec[0, 1], 'o')[0])
            points[-1].set_markerfacecolor(colors[-1])
        # Plot dynamic activations
        acts = []
        for idx, rec in enumerate(recs['addr']):
            # y := rows, x := cols
            y, x = rec[0]
            acts.append(ax.add_patch(patches.Rectangle((x, y),
                                                       1., 1.,
                                                       facecolor=colors[idx],
                                                       alpha=.3)))
        # Run
        for step, t in enumerate(self.times):
            for idx in range(num_scanners):
                # Update points
                x, y = recs['r'][idx][step, :]
                points[idx].set_xdata(x)
                points[idx].set_ydata(y)
                # Update activations
                y, x = recs['addr'][idx][step]
                acts[idx].set_xy((x, y))
#            for idx, rec in enumerate(recs['r']):
#                # Update points
#                x, y = rec[step, :]
#                points[idx].set_xdata(x)
#                points[idx].set_ydata(y)
#                # Update activations
#                acts[idx].set_xy()
            ax.set_title('{}'.format(t))
            fig.canvas.draw()
            if prompt:
                raw_input()
            else:
                time.sleep(self.wait)
            fig.canvas.flush_events()


def crosses(scan):
    points = np.array(scan.intercepts)
    x_min, y_min = np.min(points, 0) - 1.
    x_max, y_max = np.max(points, 0) + 1.

    # Line eq
    x = np.arange(x_min, x_max, 0.1)
    y = scan.line_eq[0] * x + scan.line_eq[1]
    # Figure
    plt.ioff()
#    plt.hold()
    # Line
    plt.plot(x, y, 'r-')
    # Scan position
    r = scan.get_pos()
    plt.plot(r[0], r[1], 'ko')
    # Boundaries
    plt.plot([scan.x_lim[0], scan.x_lim[0]], [scan.y_lim[0], scan.y_lim[1]], 'k:')
    plt.plot([scan.x_lim[0], scan.x_lim[1]], [scan.y_lim[1], scan.y_lim[1]], 'k:')
    plt.plot([scan.x_lim[1], scan.x_lim[1]], [scan.y_lim[1], scan.y_lim[0]], 'k:')
    plt.plot([scan.x_lim[1], scan.x_lim[0]], [scan.y_lim[0], scan.y_lim[0]], 'k:')
    # Axes
    plt.xlim([x_min, x_max])
    plt.ylim([y_min, y_max])
    plt.ion()
    plt.show()


def player(recs, times, scanners, wait=0.05):
    """
    Playback of program recordings.

    Inputs
    ------
    recs : list
        Recordings per scanner.
    times : array
        Time at each iteration.
    scanners : list
        Contains scanner objects.
    wait : float
        Time between fig updates.
    """
    # Extract common parameters
    x_lim = scanners[0].x_lim
    y_lim = scanners[0].y_lim
    # Figure setup
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(211)
    # Plot image boundaries
    ax.plot([x_lim[0], x_lim[0]], [y_lim[0], y_lim[1]], 'k-')
    ax.plot([x_lim[0], x_lim[1]], [y_lim[1], y_lim[1]], 'k-')
    ax.plot([x_lim[1], x_lim[1]], [y_lim[1], y_lim[0]], 'k-')
    ax.plot([x_lim[1], x_lim[0]], [y_lim[0], y_lim[0]], 'k-')
    # Plot each scanline and dynamic point
    points = []
    for rec, scan in zip(recs, scanners):
        ax.plot(scan.points[:, 0], scan.points[:, 1], '-')
        points.append(ax.plot(rec[0, 0], rec[0, 1], 'ko')[0])
#    line1, = ax.plot(recs[0][0, 0], recs[0][0, 1], 'ko')
#    points.append(line1)
    ax.set_title('{}'.format(times[0]))
    # Limits
    ax.set_xlim(x_lim + [-.2, .2])
    ax.set_ylim(y_lim + [-.2, .2])
    # Plot
    for step, t in enumerate(times):
        for idx, rec in enumerate(recs):
            x, y = rec[step, :]
            points[idx].set_xdata(x)
            points[idx].set_ydata(y)
#        x, y = recs[0][step, :]
#        points[0].set_xdata(x)
#        points[0].set_ydata(y)
        ax.set_title('{}'.format(t))
        fig.canvas.draw()
        time.sleep(wait)
        fig.canvas.flush_events()
