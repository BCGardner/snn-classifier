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

from snncls.parameters import ParamSet

# Use common plotting scheme
import matplotlib as mpl
from cycler import cycler
# Line colours
# mpl.rcParams['axes.prop_cycle'] = cycler(color='bgrcmyk')
mpl.rcParams['axes.prop_cycle'] = \
    cycler('color', ['k', 'g', 'r', 'c', 'm', 'y', 'k'])
# Grid
mpl.rcParams['grid.color'] = 'k'
mpl.rcParams['grid.linestyle'] = ':'
mpl.rcParams['grid.linewidth'] = 0.5


class Plotter(object):
    """
    Top level plotting class, with adjustable parameters.
    """
    def __init__(self, **kwargs):
        """
        Overwrite common, default plotting parameters.
        """
        self.props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        self.prms = ParamSet({'dt': 0.1,
                              'duration': 40.,
                              'figwidth': 5.5,
                              'fontsize': None,
                              'cmap': 'Blues'})
        self.update(**kwargs)

    def update(self, **kwargs):
        """
        Update internal plotter parameters.
        """
        self.prms.overwrite(**kwargs)
        self.times = np.arange(0., self.prms['duration'], self.prms['dt'])

    def setup_fig(self, fig_shape=None, ratio=2./3, figsize=None, **kwargs):
        """
        Default setup of figure and axes. fig_shape specifies subplot panels.
        Returns handle to figure pane and axis as 2-tuple.
        """
        if figsize is None:
            figwidth = self.prms['figwidth']
            figsize = (figwidth, ratio * figwidth)
        if fig_shape is not None:
            return plt.subplots(*fig_shape, figsize=figsize, **kwargs)
        else:
            return plt.subplots(figsize=figsize, **kwargs)

    def heatmap_prms(self, mat, prms0, prms1, labels=None, vmin=None,
                     vmax=None, num_ticks=None, figsize=None, fname=None):
        """
        Plot heatmap of 2D array of values, w.r.t. parameter values.

        Inputs
        ------
        mat : array, shape (num_prms0, num_prms1)
            Matrix of final accuracies at each prm coord.
        prms0 : list, len (num_prms0)
            Parameter values corresponding to axis 0 (y-axis).
        prms1 : list, len (num_prms1)
            Parameter values corresponding to axis 1 (x-axis).
        labels : list, len (num_dims)
            Parameter labels: [prms0, prms1].
        """
        # Setup
        fontsize = self.prms['fontsize']
        f, ax = self.setup_fig(figsize=figsize)
        # Plot
        im = ax.imshow(mat, cmap='Blues', interpolation='none',
                       vmin=vmin, vmax=vmax)
        # Colorbar
    #    if vmin is not None:
    #        cb = f.colorbar(im, extend='max')
    #    cb.cmap.set_over('white')
    #    else:
        cb = f.colorbar(im)
        tb = cb.get_ticks()[[0, -1]]
        self.set_ticks(cb, num_ticks, tick_bounds=tb)
        cb.ax.tick_params(labelsize=fontsize)
        # Add values
        thresh = np.mean(cb.get_ticks()[[0, -1]])
        for coord, v in np.ndenumerate(mat):
            i, j = coord
            ax.text(j, i, format(mat[i, j], '.0f'),
                    horizontalalignment="center",
                    color="white" if mat[i, j] > thresh else "black",
                    fontsize=fontsize)
        # Set labels on x-axis (prms1)
        ax.set_xticks(np.arange(len(prms1)))
        ax.set_xticklabels(prms1)
        ax.set_xlim(np.array([0, len(prms1)]) - 0.5)
        # Set labels on y-axis (prms0)
        ax.set_yticks(np.arange(len(prms0)))
        ax.set_yticklabels(prms0)
        ax.set_ylim(np.array([0, len(prms0)]) - 0.5)
        if labels is not None:
            ax.set_xlabel(labels[1])
            ax.set_ylabel(labels[0])
        # Reverse x ticks
    #    ax.xaxis.tick_top()
        # Set custom font sizes
        self.set_fontsize(ax)
        # Save
        self.print_plot(f, fname)

    def data_violin(self, arr, xlim=None, ylim=None, dy=None, xlabel=None,
                    ylabel=None, figsize=None, fname=None):
        """
        Violin plots for each column of a data array. Array may be 1D or 2D.

        Inputs
        ------
        arr : array, shape (num_samples[, num_cols])
            Array values.
        lim : tuple, len (2)
            Lower and upper plotted values: (v_min, v_max)
        """
        # Setup
        f, ax = self.setup_fig(figsize=figsize)
        # Plot
        parts = ax.violinplot(arr)
        for pc in parts['bodies']:
            pc.set_facecolor('blue')
        parts['cbars'].set_color('black')
        parts['cmaxes'].set_color('black')
        parts['cmins'].set_color('black')
        # Limits and ticks
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        if dy is not None:
            lims = ax.get_ylim()
            ticks = np.arange(lims[0], lims[1] + dy, dy)
            ax.set_yticks(ticks)
        # Labels
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        # Set custom font sizes
        self.set_fontsize(ax)
        # Save
        self.print_plot(f, fname)

    def weight_hist2d(self, weights, w_lims, bins=20, thr_max=None,
                      normed=False, num_ticks=None, dx=None, debug=False,
                      figsize=None, fname=None):
        """
        Plot large array of weights, gathered over multiple runs, as 2d hist
        per epoch.

        Inputs
        ------
        weights : array, shape ([num_runs], num_epochs, num_post, num_pre)
            Weights of a layer.
        w_lims : tuple, len (2)
            Lower and upper weight bounds: (w_min, w_max)
        bins : int
            Number of intervals the weights are counted over.
        thr_max : int, optional
            Threshold for maximum count.
        normed : bool, optional
            Normalise the 2D histrogram.
        num_ticks : int, optional
            Number of ticks on colorbar (excluding thr_max).
        """
        # Setup
        f, ax = self.setup_fig(figsize=figsize)
        # Cast weights array to shape (num_epochs, pooled_connections)
        if np.ndim(weights) == 4:
            weights = weights.swapaxes(0, 1)
        elif np.ndim(weights) != 3:
            raise ValueError
        num_epochs = len(weights)
        ws = weights.reshape((num_epochs, -1))
        num_conn = ws.shape[1]
        # Assign epoch indices, flatten points
        epoch_idxs = np.concatenate([np.full(num_conn, idx) for idx in
                                     xrange(num_epochs)])
        ws = ws.ravel()
        # Define bin edges
        w_edges = np.linspace(*w_lims, num=bins)
        idx_edges = np.linspace(0, num_epochs, num=num_epochs+1)
        # Plot 2D histogram of weights
        counts, xedges, yedges, im = \
            ax.hist2d(ws, epoch_idxs, bins=(w_edges, idx_edges), vmax=thr_max,
                      normed=normed, cmap='Blues')
        # Colorbar
        if thr_max is not None:
            if num_ticks is not None:
                ticks = np.linspace(0, thr_max, num_ticks + 1)
            else:
                ticks = None
            cb = f.colorbar(im, ticks=ticks, extend='max')
            cb.cmap.set_over('black')
        else:
            cb = f.colorbar(im)
        cb.set_label('Count')
        # Ticks and labels
        if dx is not None:
            xticks = np.arange(w_lims[0], w_lims[1] + dx, dx)
            ax.set_xticks(xticks)
        ax.set_xlabel('Weight')
        ax.set_ylabel('Epochs')
        # Set custom font sizes
        self.set_fontsize(ax)
        # Save
        self.print_plot(f, fname)
        # Debug
        if debug:
            return counts, xedges, yedges

    def weight_distr(self, w_h, w_o, bins=80, num_yticks=4, y_max=None,
                     normed=False, figsize=None, fname=None):
        """
        Plot output and hidden weight distributions, side by side.
        """
        f, axarr = self.setup_fig((2, 1), figsize=figsize, sharex=True)
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
                      horizontalalignment='right', bbox=self.props)
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
                      horizontalalignment='right', bbox=self.props)
        # Save
        self.print_plot(f, fname)

    def confusion_matrix(self, mat, dtick=20, skip_lteq=0., figsize=None,
                         fname=None):
        """
        Plot square confusion matrix with shape
        (true classes, predicted classes), assuming accuracies as percentages
        normalised by each row. Optional width: 8.
        """
        # Setup
        fontsize = self.prms['fontsize']
        f, ax = self.setup_fig(figsize=figsize)
        num_classes = len(mat)
        # Plot
        im = ax.imshow(mat, cmap='Blues', interpolation='none', vmin=0,
                       vmax=100)
        cb = f.colorbar(im, ticks=np.arange(0, 120, dtick))
        cb.ax.tick_params(labelsize=fontsize)
        # Add values
        thresh = 100. / 2.
        for i, j in itertools.product(range(mat.shape[0]),
                                      range(mat.shape[1])):
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
        ax.set_xlabel('Predicted label')
        ax.set_ylabel('True label')
        # Reverse x ticks
        ax.xaxis.tick_top()
        # Set custom font sizes
        self.set_fontsize(ax)
        # Save
        self.print_plot(f, fname)

    def spike_raster(self, spike_trains, num_xticks=None, figsize=None,
                     fname=None):
        """
        Plot a list of spike trains.
        """
        # Setup
        duration = self.prms['duration']
        f, ax = self.setup_fig(figsize=figsize)
        for nrn, spikes in enumerate(spike_trains):
            ax.vlines(spikes, nrn, nrn + .5)
        # Limits
        ax.set_xlim([0., duration])
        ax.set_ylim([0, nrn+0.8])
        # Ticks
        self.set_ticks(ax.get_xaxis(), num_xticks)
        ax.set_yticks(np.arange(nrn+1))
        # Labels
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Neuron #")
        # Print fig
        self.print_plot(f, fname)

    def spike_rasters(self, spikes, voltages=None, text=None, textsize=None,
                      linewidth=None, num_xticks=None, figsize=None,
                      fname=None):
        """
        TODO: Plots input, hidden, output spike trains, optionally output
        voltages.
        """
        # Setup
        duration = self.prms['duration']
        num_inputs = len(spikes[0])
        num_hidden = len(spikes[1])
        num_outputs = len(spikes[-1])
    #    if voltages is not None:
    #        f, axarr = plt.subplots(4, figsize=figsize, sharex=True)
    #    else:
    #        f, axarr = plt.subplots(3, figsize=figsize, sharex=True)
        f, axarr = self.setup_fig((3, 1), ratio=3/4., figsize=figsize,
                                  sharex=True)

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
                          horizontalalignment='right', bbox=self.props)
    #    set_fontsize(axarr[0], fontsize)
        # Hidden spike trains
        st_panel(spikes[1], axarr[1], vline_size=1.)
        axarr[1].set_yticks([0, num_hidden])
        axarr[1].set_ylabel("Hidden")
        # Output [voltage traces,] spike trains
        if voltages is not None:
            for nrn, v in enumerate(voltages):
                axarr[2].plot(self.times, v / 15. + nrn, label=str(nrn),
                              linewidth=linewidth)
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
        axarr[-1].set_xlim([0., duration])
#        self.set_ticks(axrr[-2])
        axarr[-1].set_xlabel('Time (ms)')
#        plt.xticks(np.arange(0, 50, 10))
        # Set font size
        for ax in axarr:
            self.set_fontsize(ax)
        # Spacing
    #    f.subplots_adjust(hspace=hspace)
        self.print_plot(f, fname)

    def weight_heatmaps(self, weights, figsize=(8.0, 6.0), fname=None):
        """
        TODO: Plot heatmaps of weight matrices, [save to fname].

        Inputs
        ------
        weights: list, len (num_layers)
            List of numpy arrays by layer, each with shape
            (num_epochs, num_post, num_pre).
        """
        # Setup
#        num_outputs = weights[-1].shape[1]
#        num_hidden = weights[-2].shape[1]
        cmap = self.prms['cmap']
        f, axs = self.setup_fig((2, 1), figsize=figsize)

#        # Change in output / hidden weights with epochs
#        dw_o = [weights[-1][:, i, :] / weights[-1][0, i, :]
#                for i in xrange(num_outputs)]
#        dw_h = [weights[-2][:, i, :] / weights[-2][0, i, :]
#                for i in xrange(num_hidden)]

        # Plotting
        def im_panel(f, ax, arr, clim=(None, None), nbins=None,
                     extend='neither'):
            # cmap.set_over('black')
            im = ax.imshow(arr, cmap=cmap, interpolation=None, clim=clim)
            ax.set_aspect('equal')
            divider = make_axes_locatable(ax)
            ax_cb = divider.new_horizontal(size="5%", pad=0.2,
                                           pack_start=False)
            f.add_axes(ax_cb)
            cb = f.colorbar(im, cax=ax_cb, orientation='vertical',
                            extend=extend)
            if nbins is not None:
                tick_locator = ticker.MaxNLocator(nbins=nbins)
                cb.locator = tick_locator
                cb.update_ticks()
        # Plot final weight heatmaps for outputs
        ax = axs[0]
        im_panel(f, ax, weights[-1][-1], nbins=3)
        ax.set_xticks(np.arange(0, 20, 4))
        ax.set_yticks(np.arange(0, 3, 1))
        ax.set_xlabel("Hidden #")
        ax.set_ylabel("Output #")
        ax = axs[1]
        im_panel(f, ax, weights[-2][-1], clim=(None, 20), extend='max')
        ax.set_xticks(np.arange(0, 50, 10))
        ax.set_yticks(np.arange(0, 20, 4))
        ax.set_xlabel("Input #")
        ax.set_ylabel("Hidden #")
        # Save
        self.print_plot(f, fname)

    def errbars(self, *losses, **kwargs):
        """
        Plot set of loss curves, sharing same length.
        Each set of loss curves has shape (num_epochs, num_runs).
        """
        args = {'y_max': None,
                'num_yticks': None,
                'num_xticks': None,
                'labels': None,
                'text': None,
                'textsize': 14,
                'fontsize': None,
                'linewidth': 1.,
                'figsize': None,
                'fname': None}
        for k, v in kwargs.items():
            if k in args:
                args[k] = v
        # Setup
        f, ax = self.setup_fig(figsize=args['figsize'])
        num_epochs = len(losses[0])
        epochs = np.arange(num_epochs) + 1

        def plot_errbar(loss, label=None):
            loss_av = np.mean(loss, 1)
            loss_std = np.std(loss, 1)
            # Evolution of loss
            ax.plot(epochs, loss_av, '-', label=label,
                    linewidth=args['linewidth'])
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
        # Set axes
        ax.set_xlim([0, num_epochs])
        ax.set_ylim([0, args['y_max']])
        self.set_ticks(ax.get_xaxis(), args['num_xticks'])
        self.set_ticks(ax.get_yaxis(), args['num_yticks'])
        # Labels
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.grid(b=True)
        if args['text'] is not None:
            ax.text(0.5, 0.9, args['text'], transform=ax.transAxes,
                    fontsize=args['textsize'], verticalalignment='top',
                    horizontalalignment='right', bbox=self.props)
        # Set custom font sizes
        self.set_fontsize(ax)
        # Save
        self.print_plot(f, args['fname'])

    def heatmap(self, data, vmin=None, vmax=None, orientation='horizontal',
                figsize=None, fname=None):
        """
        Plot heatmap of 2d data array (e.g. weights).
        """
        # Setup
        cmap = self.prms['cmap']
        f, ax = self.setup_fig(figsize=figsize)
        # Plot
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
        self.print_plot(f, fname)

    def set_fontsize(self, ax):
        fontsize = self.prms['fontsize']
        if fontsize is not None:
            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                         ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(fontsize)

    def set_ticks(self, h, num_ticks=None, tick_bounds=None):
        """
        Set number of ticks, with equal spacing, on a given object handle.
        """
        if num_ticks is not None:
            if tick_bounds is None:
                tick_bounds = h.get_ticklocs()[[0, -1]]
            ticks = np.linspace(*tick_bounds, num=num_ticks)
            h.set_ticks(ticks)

    def print_plot(self, fig, fname=None):
        """
        Print provided figure.
        """
        if fname is not None:
            fig.savefig('out/{}.pdf'.format(fname), bbox_inches='tight',
                        dpi=300)
        else:
            fig.show()
