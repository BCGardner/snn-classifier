"""
Class definition used to playback scanline encoding for visualisation
purposes.

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
import matplotlib.patches as patches
import time


class Playback(object):
    """
    Plays back recordings from scanners.
    """
    def __init__(self, bounds, times, duration, wait=0.05):
        self.x_lim = np.array([0, bounds[1]])
        self.y_lim = np.array([0, bounds[0]])
        self.times = times
        self.duration = duration
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
        # Ticks
        ax.set_xticks(np.arange(0, self.x_lim[1] + 4, 4))
        ax.set_yticks(np.arange(0, self.y_lim[1] + 4, 4))
        # Labels
        ax.set_title(f'{self.times[0]:.1f}')

    def setup_scanners(self, ax, recs):
        """
        Initial scanner plots, return handle on dynamic objects.
        """
        # Plot each scanline and dynamic point
        points = []
        colors = []
        for rec in recs['r']:
            (x1, y1), (x2, y2) = rec[0], rec[-1]
            l, = ax.plot([x1, x2], [y1, y2], '-')
            colors.append(l.get_color())
            points.append(ax.plot(rec[0, 0], rec[0, 1], 'ko')[0])
            points[-1].set_markerfacecolor(colors[-1])
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
        return points, acts, colors

    def play(self, recs, img=None, prompt=False):
        """
        Playback scanner recordings.
        """
        num_scanners = len(recs['r'])
        # Figure setup
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # Setup image and get dynamic objects
        self.setup_plot(ax, img)
        points, acts, _ = self.setup_scanners(ax, recs)
        # Run
        for step, t in enumerate(self.times):
            for idx in range(num_scanners):
                # Update points
                x, y = recs['r'][idx][step, :]
                points[idx].set_data(x, y)
                # Update activations
                y, x = recs['addr'][idx][step]
                acts[idx].set_xy((x, y))
            ax.set_title(f'{t:.1f}')
            fig.canvas.draw()
            if prompt:
                input()
            else:
                time.sleep(self.wait)
            fig.canvas.flush_events()

    def play_nrns(self, recs, img=None, prompt=False):
        """
        Playback scanner and nrn recordings.
        """
        num_scanners = len(recs['r'])
        # Figure setup
        plt.ion()
        fig, axes = plt.subplots(1, 2)
        # Setup image, dynamics
        self.setup_plot(axes[0], img)  # Image
        points, acts, colors = self.setup_scanners(axes[0], recs)  # Scanners
        # Spike raster
        axes[1].set_xlim([0., self.duration])
        axes[1].set_ylim([0, num_scanners])
        axes[1].set_xlabel('Time (ms)')
        axes[1].set_ylabel('Encoder')
        asp = np.diff(axes[1].get_xlim())[0] / np.diff(axes[1].get_ylim())[0]
        axes[1].set_aspect(asp)  # Set aspect to match image
        l_spikes = []  # Handle on spike vlines
        for idx in range(num_scanners):
            l_spikes.append(axes[1].vlines([], 0., 1., colors=colors[idx]))
        # Sweeping line
        l_sweep, = axes[1].plot([0., 0.], [0., num_scanners],
                                'k:', linewidth=1)
        # Run
        for step, t in enumerate(self.times):
            for idx in range(num_scanners):
                # Update points
                x, y = recs['r'][idx][step, :]
                points[idx].set_data(x, y)
                # Update activations
                y, x = recs['addr'][idx][step]
                acts[idx].set_xy((x, y))
                # Plot spikes up to t
                spikes = np.array(recs['spikes'][idx])
                spikes = spikes[spikes <= t]
                segs = []
                for spike in spikes:
                    segs.append(np.stack([2 * [spike], [idx, idx + 1]], 1))
                l_spikes[idx].set_segments(segs)
                # Shift sweeping line
                l_sweep.set_xdata((t, t))
            axes[0].set_title(f'{t:.1f}')
            fig.canvas.draw()
            if prompt:
                input()
            else:
                time.sleep(self.wait)
            fig.canvas.flush_events()
        input("press any key ")


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
    ax.set_title(f'{times[0]:.1f}')
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
        ax.set_title(f'{t:.1f}')
        fig.canvas.draw()
        time.sleep(wait)
        fig.canvas.flush_events()
