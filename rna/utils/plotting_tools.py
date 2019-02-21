# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 17:48:35 2018

@author: yanm
"""
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from custom_losses import *
from utils import num2date, date2num, time_units
from matplotlib.collections import LineCollection

#
params = {
    'axes.labelsize': 9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'text.usetex': False,
    'axes.titlesize': 10,
    'font.family': 'serif',
    'font.weight': 'light',
    'font.style': 'normal'
}

plt.rcParams.update(params)

def transform_dates(dates):
    # from matlab unicode dates to python datetime object
    return [str(num2date(date, time_units)).split(' ')[0] for date in dates]

def plot_dates(x, data, labels):

    num_x = x
    x = transform_dates(num_x)

    fig = plt.figure(figsize=(9, 3), facecolor='white')
    ax = fig.add_subplot(111)
    ax.autoscale()

    colors = ['.k', '.r', '.b']
    for y, l, c in zip(data, labels, colors):

        y_std = np.std(y)
        ax.plot(num_x, y, c, linewidth=0.8, markersize=1.5, label=l)
        ax.fill_between(num_x, y-y_std, y+y_std, color='gray', alpha=0.3)

    num_ticks = slice(None, None, num_x.size / 15)
    plt.xticks(num_x[num_ticks], x[num_ticks], rotation=20)

    y_mean, y_min, y_max, y_ptp = np.mean(data), np.min(data), np.max(
        data), np.ptp(data)
    ax.set_xlim(num_x.min() - 2.5, num_x.max() + 2.5)
    ax.set_ylim(y_min - 0.01 * y_ptp, y_max + 0.01 * y_ptp)

    plt.legend(loc="best", scatterpoints=1, prop={'size': 8})
    ax.set_ylabel('(mm/3h)')

    return fig

def scatter_plot(y_true, y_pred, y_frcs, name_str=''):

    str_format = ["Forecast rmse:  {:2.4f}, Neural-net rmse:  {:2.4f}",
                  "Forecast score: {:.2%}, Neural-net score: {:.2%}"]

    slope, intcp, r2_value, _, _ = linregress(y_true, y_frcs)
    y_linr = y_frcs / slope - intcp

    raw_rmse = np.sqrt(mean_squared_error(y_true, y_frcs))
    ada_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    lin_rmse = np.sqrt(mean_squared_error(y_true, y_linr))

    raw_corr = explained_variance_score(y_true, y_frcs)
    ada_corr = explained_variance_score(y_true, y_pred)
    lin_corr = explained_variance_score(y_true, y_linr)

    print(str_format[0].format(raw_rmse, ada_rmse))
    print(str_format[1].format(raw_corr, ada_corr), '\n')

    # look at the results
    str_linc = 'Linear Regression ' + r'$r_{2}$ ' + 'coefficient: %.2f' % (1e2*r2_value)

    str_linr = 'Linear baseline (rmse: %.3f, corr: %.2f)' % (lin_rmse, 1e2*lin_corr)
    str_fcst = 'SPNOA forecast (rmse: %.3f, corr: %.2f)' % (raw_rmse, 1e2*raw_corr)
    str_pred = 'MLP Regression (rmse: %.3f, corr: %.2f)' % (ada_rmse, 1e2*ada_corr)

    fig = plt.figure()
    plt.title(name_str)
    plt.scatter(y_true, y_frcs, s=2.5, c='k', linewidth=1.2,
                edgecolors='k', label=str_fcst, zorder=1, alpha=0.6)
    plt.scatter(y_true, y_pred, s=2.5, c='r', linewidth=1.2,
                edgecolors='r', label=str_pred, zorder=1, alpha=0.6)
    plt.plot(y_true, y_true * slope + intcp, '-k', label=str_linr, linewidth=0.5)
    plt.plot(y_true, y_true, '--k', linewidth=0.25)

    plt.xlabel('observations')
    plt.ylabel('predictions')
    plt.legend()

    return fig

def plot_lines(stages, data_set, varname, stn_id, domain, metric):

    figure = plt.figure(figsize=(10, 4))
    plt.grid(True)

    # adecuate concatenation of multi-type strings:
    title = ' '.join([u'Análisis Anual de', varname.capitalize()+'.',
        u'Estación:', stn_id.split('_')[-1],
        u'\nPronósticos SPNOA (WRF) vs MOS'
    ])

    plt.title(title)
    plt.xlabel(u'Plazos de pronóstico (Horas)')
    plt.ylabel(metric.upper())

    labels = ['WRF '+domain+' 0000 UTC', 'WRF '+domain+' 1200 UTC',
              'MOS '+domain+' 0000 UTC', 'MOS '+domain+' 1200 UTC']
    lines = 2 * ["--o", ] + 2 * ["--*"]
    colors = ["blue", "orange", "green", "cyan"]

    # get ticks
    x_range = []
    for stage in stages:
        init = int(stage[0].split(':')[0])
        x_range.append(init + 3*np.arange(len(stage)))

    for stage, x, data, l, c, label in zip(stages, x_range, data_set, lines, colors, labels):

        plt.plot(x, data, l, color=c, markersize=5, linewidth=1., label=label)

    ticks_range = 3 * np.arange(len(stage)+4)
    ticks_label = stages[0] + stages[1][-4:]
    plt.xticks(ticks_range[::3], ticks_label[::3], rotation=0)

    plt.legend(loc='best', fontsize=8)

    return figure


def plot_embedding(partial_correlations, embedding, names, labels):

    figure = plt.figure(1, facecolor='w', figsize=(10, 8))
    plt.clf()
    ax = plt.axes([0., 0., 1., 1.])
    plt.axis('off')

    # Display a graph of the partial correlations
    d = 1 / np.sqrt(np.diag(partial_correlations))
    partial_correlations *= d
    partial_correlations *= d[:, np.newaxis]

    non_zero = (np.abs(np.triu(partial_correlations, k=1)) > 0.02)

    # Plot the nodes using the coordinates of our embedding
    plt.scatter(embedding[0], embedding[1], s=100 * d ** 2, c=labels,
                cmap=plt.cm.spectral)

    # Plot the edges
    start_idx, end_idx = np.where(non_zero)
    # a sequence of (*line0*, *line1*, *line2*), where::
    #            linen = (x0, y0), (x1, y1), ... (xm, ym)
    segments = [[embedding[:, start], embedding[:, stop]]
                for start, stop in zip(start_idx, end_idx)]
    values = np.abs(partial_correlations[non_zero])
    lc = LineCollection(segments,
                        zorder=0, cmap=plt.cm.hot_r,
                        norm=plt.Normalize(0, .7 * values.max()))
    lc.set_array(values)
    lc.set_linewidths(15 * values)
    ax.add_collection(lc)

    # Add a label to each node. The challenge here is that we want to
    # position the labels to avoid overlap with other labels
    n_labels = labels.max()
    for index, (name, label, (x, y)) in enumerate(
            zip(names, labels, embedding.T)):

        dx = x - embedding[0]
        dx[index] = 1
        dy = y - embedding[1]
        dy[index] = 1
        this_dx = dx[np.argmin(np.abs(dy))]
        this_dy = dy[np.argmin(np.abs(dx))]
        if this_dx > 0:
            horizontalalignment = 'left'
            x = x + .002
        else:
            horizontalalignment = 'right'
            x = x - .002
        if this_dy > 0:
            verticalalignment = 'bottom'
            y = y + .002
        else:
            verticalalignment = 'top'
            y = y - .002
        plt.text(x, y, name, size=10,
                 horizontalalignment=horizontalalignment,
                 verticalalignment=verticalalignment,
                 bbox=dict(facecolor='w',
                           edgecolor=plt.cm.spectral(label / float(n_labels)),
                           alpha=.6))

    plt.xlim(embedding[0].min() - .21 * embedding[0].ptp(),
             embedding[0].max() + .21 * embedding[0].ptp(),)
    plt.ylim(embedding[1].min() - .05 * embedding[1].ptp(),
             embedding[1].max() + .05 * embedding[1].ptp())

    return figure
