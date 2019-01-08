#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Jan 2018

@author: BG

Tex textwidth (multilayer-classifier): ~ 5.75 in
Poster fighwidth: ~ 9 in
Default fig scale: fig_height = fig_width / 1.5
"""

import itertools
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import ticker

# Use common plotting scheme
import matplotlib as mpl
from cycler import cycler
# Line colours
# mpl.rcParams['axes.prop_cycle'] = cycler(color='bgrcmyk')
mpl.rcParams['axes.prop_cycle'] = \
    cycler('color', ['b', 'g', 'r', 'c', 'm', 'y', 'k'])
# Grid
mpl.rcParams['grid.color'] = 'k'
mpl.rcParams['grid.linestyle'] = ':'
mpl.rcParams['grid.linewidth'] = 0.5

# Default pattern duration
dt = 0.1
duration = 40.
times = np.arange(0., duration, dt)
# Text
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
# Default fig_width
fig_width = 5.5


def weight_distr(w_h, w_o, fig_width=fig_width, bins=80, num_yticks=4,
                 normed=False, fname=None):
    """
    Plot output and hidden weight distributions, side by side.
    """
    figsize = (fig_width, fig_width / 1.5)
    f, axarr = plt.subplots(2, figsize=figsize, sharex=True)
    # Plot hidden weights
    axarr[0].hist(w_h.ravel(), bins, normed=normed)
    y_max = np.ceil(axarr[0].get_ybound()[-1])
    dy = y_max / num_yticks
    axarr[0].set_ylim([0., y_max])
    axarr[0].set_yticks(np.arange(0, y_max + dy, dy))
    if normed:
        axarr[0].set_ylabel(r'$f(w)$')
    else:
        axarr[0].set_ylabel('Count')
    axarr[0].text(0.9, 0.9, 'Hidden', transform=axarr[0].transAxes,
                  fontsize=14, verticalalignment='top',
                  horizontalalignment='right', bbox=props)
    # Plot output weights
    axarr[1].hist(w_o.ravel(), bins, normed=normed)
    y_max = np.ceil(axarr[1].get_ybound()[-1])
    dy = y_max / num_yticks
    axarr[1].set_ylim([0., y_max])
    axarr[1].set_yticks(np.arange(0, y_max + dy, dy))
    axarr[1].set_xlabel('Weight')
    if normed:
        axarr[1].set_ylabel(r'$f(w)$')
    else:
        axarr[1].set_ylabel('Count')
    axarr[1].text(0.9, 0.9, 'Output', transform=axarr[1].transAxes,
                  fontsize=14, verticalalignment='top',
                  horizontalalignment='right', bbox=props)
    # Save
    print_plot(f, fname)


def confusion_matrix(mat, fig_width=fig_width, fontsize=None, fname=None,
                     dtick=20, skip_lteq=0.):
    """
    Plot square confusion matrix of size <true classes> by <predicted classes>,
    assuming accuracies as percentages normalised by each row.
    Optional width: 8.
    """
    figsize = (fig_width, fig_width / 1.5)
    f, ax = plt.subplots(figsize=figsize)
    num_classes = len(mat)
    # Plot
    im = ax.imshow(mat, cmap='Blues', interpolation='none', vmin=0, vmax=100)
    cb = f.colorbar(im, ticks=np.arange(0, 120, dtick))
    cb.ax.tick_params(labelsize=fontsize)
    # Add values
    thresh = 100. / 2.
    for i, j in itertools.product(range(mat.shape[0]), range(mat.shape[1])):
        if mat[i, j] <= skip_lteq:
            continue
        ax.text(j, i, format(mat[i, j], '.1f'),
                horizontalalignment="center",
                color="white" if mat[i, j] > thresh else "black",
                fontsize=fontsize)
    # Labels
    ax.set_xticks(np.arange(0, num_classes + 1, 1))
    xticks = [clabel for clabel in range(num_classes)] + [r'$\emptyset$']
    ax.set_xticklabels(xticks)
    ax.set_xlim(np.array([0, num_classes + 1]) - 0.5)  # Final col is null
    ax.set_yticks(np.arange(0, num_classes + 1, 1))
    ax.set_ylim(np.array([num_classes, 0]) - 0.5)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    # Reverse x ticks
    ax.xaxis.tick_top()
    # Set custom font sizes
    if fontsize is not None:
        set_fontsize(ax, fontsize)
    # Save
    print_plot(f, fname)


def spike_raster(st, figsize=None):
    if figsize is not None:
        plt.figure(figsize=figsize)
    else:
        plt.figure()
    for nrn, spikes in enumerate(st[-1]):
        plt.vlines(spikes, nrn, nrn + .5)
    plt.xlabel("Time (ms)")
    plt.ylabel("Neuron #")
    plt.xlim([0., 40.])
    plt.xticks(np.arange(0, 50, 10))
    plt.ylim([0, nrn+0.8])
    plt.yticks(np.arange(nrn+1))
#    plt.title('output layer')
    plt.savefig('out/spike_raster')


def spike_rasters(spikes, voltages=None, text=None, textsize=None,
                  fontsize=None, linewidth=1,
                  figsize=(fig_width, 0.75 * fig_width), fname=None):
    """
    Plots input, hidden, output spike trains, optionally output voltages.
    """
    # Net / pattern stats
    num_inputs = len(spikes[0])
    num_hidden = len(spikes[1])
    num_outputs = len(spikes[-1])
#    if voltages is not None:
#        f, axarr = plt.subplots(4, figsize=figsize, sharex=True)
#    else:
#        f, axarr = plt.subplots(3, figsize=figsize, sharex=True)
    f, axarr = plt.subplots(3, figsize=figsize, sharex=True)

    def st_panel(spike_trains, ax, vline_size=0.5, ymax=None):
        for nrn, spikes in enumerate(spike_trains):
            ax.vlines(spikes, nrn, nrn + vline_size)
        if ymax is not None:
            ax.set_ylim([0, ymax])

    # Input spike trains
    st_panel(spikes[0], axarr[0], vline_size=2.)
    axarr[0].set_yticks([0, num_inputs])
    axarr[0].set_ylabel("Input")
    # Sample class label
    if text is not None:
        axarr[0].text(0.95, 0.85, text, transform=axarr[0].transAxes,
                      fontsize=textsize, verticalalignment='top',
                      horizontalalignment='right', bbox=props)
#    set_fontsize(axarr[0], fontsize)
    # Hidden spike trains
    st_panel(spikes[1], axarr[1], vline_size=1.)
    axarr[1].set_yticks([0, num_hidden])
    axarr[1].set_ylabel("Hidden")
    # Output [voltage traces,] spike trains
    if voltages is not None:
        for nrn, v in enumerate(voltages):
            axarr[2].plot(times, v / 15. + nrn, label=str(nrn), linewidth=2)
#        axarr[2].set_ylim([-15., 30])
#        axarr[2].set_yticks([0])
#        axarr[2].set_ylabel("Voltage")
#        axarr[2].legend(loc="lower right", fontsize='x-small')
    st_panel(spikes[2], axarr[2], vline_size=0.5, ymax=num_outputs)
    axarr[2].set_yticks([0, num_outputs])
    axarr[2].set_ylabel("Output")
    # [Output voltage traces]
#    if voltages is not None:
#        for nrn, v in enumerate(voltages):
#            axarr[3].plot(times, v, label=str(nrn), linewidth=1)
#        axarr[3].set_ylim([-15., 30])
#        axarr[3].set_yticks([0])
#        axarr[3].set_ylabel("Voltage")
#        axarr[3].legend(loc="lower right", fontsize='x-small')
    # Axis labels
    plt.xlim([0., 40.])
    plt.xlabel('Time (ms)')
    plt.xticks(np.arange(0, 50, 10))
    # Set font size
    if fontsize is not None:
        for arr in axarr:
            set_fontsize(arr, fontsize)
    # Spacing
#    f.subplots_adjust(hspace=hspace)
    print_plot(f, fname)


def weight_heatmaps(weights, figsize=(8.0, 6.0), fname=None):
    """
    Plot heatmaps of weight matrices, [save to fname].

    Inputs
    ------
    weights: list
        List of numpy arrays by layer, each of size:
            <# epochs> by <# post> by <# pre>.
    """
    # Params
    num_outputs = weights[-1].shape[1]
    num_hidden = weights[-2].shape[1]
    # Figure setup
    fig, axes = plt.subplots(2, 1, figsize=figsize)

    # Change in output / hidden weights with epochs
    dw_o = [weights[-1][:, i, :] / weights[-1][0, i, :]
            for i in xrange(num_outputs)]
    dw_h = [weights[-2][:, i, :] / weights[-2][0, i, :]
            for i in xrange(num_hidden)]

    # Plotting
    def im_panel(fig, ax, arr, clim=(None, None), nbins=None,
                 extend='neither'):
        cmap = plt.cm.get_cmap('Blues')
#        cmap.set_over('black')
        im = ax.imshow(arr, cmap=cmap, interpolation=None, clim=clim)
        ax.set_aspect('equal')
        divider = make_axes_locatable(ax)
        ax_cb = divider.new_horizontal(size="5%", pad=0.2, pack_start=False)
        fig.add_axes(ax_cb)
        cb = fig.colorbar(im, cax=ax_cb, orientation='vertical', extend=extend)
        if nbins is not None:
            tick_locator = ticker.MaxNLocator(nbins=nbins)
            cb.locator = tick_locator
            cb.update_ticks()
    # Plot final weight heatmaps for outputs
    ax = axes[0]
    im_panel(fig, ax, weights[-1][-1], nbins=3)
    ax.set_xticks(np.arange(0, 20, 4))
    ax.set_yticks(np.arange(0, 3, 1))
    ax.set_xlabel("Hidden #")
    ax.set_ylabel("Output #")
    ax = axes[1]
    im_panel(fig, ax, weights[-2][-1], clim=(None, 20), extend='max')
    ax.set_xticks(np.arange(0, 50, 10))
    ax.set_yticks(np.arange(0, 20, 4))
    ax.set_xlabel("Input #")
    ax.set_ylabel("Hidden #")
    fig.set_tight_layout(True)

    # Print data
    if fname is not None:
        plt.savefig('out/{}.pdf'.format(fname), dpi=300)
    else:
        plt.show()


def errbars(*losses, **kwargs):
    """
    Plot set of loss curves, sharing same length. Each set of loss curves has
    shape (num_epochs, num_runs).
    """
    args = {'y_max': 0.2,
            'dx': None,
            'dy': None,
            'fig_width': fig_width,
            'labels': None,
            'text': None,
            'textsize': 14,
            'fontsize': None,
            'linewidth': 1.,
            'fname': None}
    for k, v in kwargs.items():
        if k in args:
            args[k] = v
    # Plot error as a function of epochs
    figsize = (args['fig_width'], args['fig_width'] / 1.5)
    f, ax = plt.subplots(figsize=figsize)
    # Epochs, mean, std
    num_epochs = len(losses[0])
    epochs = np.arange(num_epochs) + 1

    def plot_errbar(loss, label=None):
        loss_av = np.mean(loss, 1)
        loss_std = np.std(loss, 1)
        # Evolution of loss
        ax.plot(epochs, loss_av, '-', label=label, linewidth=args['linewidth'])
        line_h = ax.get_lines()[-1]
        color = line_h.get_color()
        ax.fill_between(epochs, loss_av - loss_std, loss_av + loss_std,
                        alpha=0.2, color=color)
    # Plot
    if args['labels'] is not None:
        for loss, label in zip(losses, args['labels']):
            plot_errbar(loss, label)
        ax.legend(fontsize=args['fontsize'])
    else:
        for loss in losses:
            plot_errbar(loss)
    # Labels
    plt.xlim([0, num_epochs])
    plt.ylim([0, args['y_max']])
    if args['dx'] is not None:
        dx = args['dx']
    else:
        dx = 0.2 * num_epochs
    plt.xticks(np.arange(0, num_epochs + dx, dx))
    if args['dy'] is not None:
        plt.yticks(np.arange(0, args['y_max'] + args['dy'], args['dy']))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid()
    if args['text'] is not None:
        ax.text(0.5, 0.9, args['text'], transform=ax.transAxes,
                fontsize=args['textsize'], verticalalignment='top',
                horizontalalignment='right', bbox=props)
    # Set custom font sizes
    if args['fontsize'] is not None:
        set_fontsize(ax, args['fontsize'])
    # Save
    print_plot(f, args['fname'])


def heatmap(data, figsize=None, fname=None, vmin=None, vmax=None,
            cmap='Blues', orientation='horizontal'):
    # Plot heatmap of 2d data array (e.g. weights)
    if figsize is not None:
        f, ax = plt.subplots(figsize=figsize)
    else:
        f, ax = plt.subplots()
#    cmap = plt.cm.Blues
#    cmap = plt.cm.jet
    heatmap = ax.imshow(data, cmap=cmap, interpolation='nearest',
                        vmin=vmin, vmax=vmax)
    # Labels
#    plt.xticks(np.arange(0, 20, 2))
#    plt.xlabel('Hidden #')
#    plt.ylabel('Output #')
#    bounds = np.linspace(0, 20, 50)
    if vmax is not None and vmin is not None:
        f.colorbar(heatmap, orientation=orientation, extend='both')
    elif vmax is not None:
        f.colorbar(heatmap, orientation=orientation, extend='max')
    elif vmin is not None:
        f.colorbar(heatmap, orientation=orientation, extend='min')
    else:
        f.colorbar(heatmap, orientation=orientation)
#    cb = mpl.colorbar.Colorbar(ax, heatmap, norm=norm)
#    cb = f.colorbar(heatmap, orientation='horizontal')
#    plt.xticks(np.arange(0, 50, 10))
    if fname is not None:
        plt.savefig('out/{}.pdf'.format(fname), dpi=300)
    else:
        plt.show()


def set_fontsize(ax, fontsize):
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(fontsize)


def print_plot(fig, fname=None):
    if fname is not None:
        fig.savefig('out/{}.pdf'.format(fname), bbox_inches='tight', dpi=300)
    else:
        fig.show()
