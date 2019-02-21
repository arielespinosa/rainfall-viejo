#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
=============================================================
Compare the effect of different scalers on data with outliers
=============================================================
"""

# Modified from Sklearn Examples by:
#
#          Raghav RV <rvraghav93@gmail.com>
#          Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Thomas Unterthiner
#
# License: BSD 3 clause

from __future__ import print_function

import numpy as np

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import cm

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing.data import QuantileTransformer

from utils.utils import create_path, loadmat
from sklearn.datasets import fetch_california_housing

print(__doc__)

class LinearScaler(object):
    '''
       Perform no operation on the data
       (just here for convenience)
    '''
    def fit(self, x, y):
        pass
    def fit_transform(self, x):
        return x
    def inverse_transform(self, x):
        return x


Scalers = {
    'Unscaled data':
    LinearScaler(),
    'Standard scaling':
    StandardScaler(),
    'Min-max scaling':
    MinMaxScaler(),
    'Max-abs scaling':
    MaxAbsScaler(),
    'Robust scaling':
    RobustScaler(quantile_range=(25, 75)),
    'Quantile transformation (uniform pdf)':
    QuantileTransformer(output_distribution='uniform'),
    'Quantile transformation (gaussian pdf)':
    QuantileTransformer(output_distribution='normal'),
    'Sample-wise L2 normalizing':
    Normalizer(norm='l2'),
    'Sample-wise L1 normalizing':
    Normalizer(norm='l1'),
}


def create_axes(title, figsize=(16, 6)):
    fig = plt.figure(figsize=figsize)
    fig.suptitle(title)

    # define the axis for the first plot
    left, width = 0.1, 0.22
    bottom, height = 0.1, 0.7
    bottom_h = height + 0.15
    left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.1]
    rect_histy = [left_h, bottom, 0.05, height]

    ax_scatter = plt.axes(rect_scatter)
    ax_histx = plt.axes(rect_histx)
    ax_histy = plt.axes(rect_histy)

    # define the axis for the zoomed-in plot
    left = width + left + 0.2
    left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.1]
    rect_histy = [left_h, bottom, 0.05, height]

    ax_scatter_zoom = plt.axes(rect_scatter)
    ax_histx_zoom = plt.axes(rect_histx)
    ax_histy_zoom = plt.axes(rect_histy)

    # define the axis for the colorbar
    left, width = width + left + 0.13, 0.01

    rect_colorbar = [left, bottom, width, height]
    ax_colorbar = plt.axes(rect_colorbar)

    return ((ax_scatter, ax_histy, ax_histx),
            (ax_scatter_zoom, ax_histy_zoom, ax_histx_zoom),
            ax_colorbar)


def plot_distribution(axes, X, y, hist_nbins=50, title="",
                      x0_label="", x1_label=""):
    ax, hist_X1, hist_X0 = axes

    ax.set_title(title)
    ax.set_xlabel(x0_label)
    ax.set_ylabel(x1_label)

    # The scatter plot
    colors = cm.plasma_r(y)
    ax.scatter(X[:, 0], X[:, 1], alpha=0.5, marker='o', s=5, lw=0, c=colors)

    # Removing the top and the right spine for aesthetics
    # make nice axis layout
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))

    # Histogram for axis X1 (feature 5)
    hist_X1.set_ylim(ax.get_ylim())
    hist_X1.hist(X[:, 1], bins=hist_nbins, orientation='horizontal',
                 color='grey', ec='grey')
    hist_X1.axis('off')

    # Histogram for axis X0 (feature 0)
    hist_X0.set_xlim(ax.get_xlim())
    hist_X0.hist(X[:, 0], bins=hist_nbins, orientation='vertical',
                 color='grey', ec='grey')
    hist_X0.axis('off')

    return ax

def make_plot(X, y, title='', xlabels=['x1', 'x2', 'z'], norm=None):

    ax_zoom_out, ax_zoom_in, ax_colorbar = create_axes(title)
    axarr = (ax_zoom_out, ax_zoom_in)
    _ = plot_distribution(axarr[0], X, y, hist_nbins=200,
                      x0_label=xlabels[0],
                      x1_label=xlabels[1],
                      title="Full data")
    # zoom-in
    zoom_in_percentile_range = (0, 99)
    cutoffs_X0 = np.percentile(X[:, 0], zoom_in_percentile_range)
    cutoffs_X1 = np.percentile(X[:, 1], zoom_in_percentile_range)

    non_outliers = (
        np.all(X > [cutoffs_X0[0], cutoffs_X1[0]], axis=1) &
        np.all(X < [cutoffs_X0[1], cutoffs_X1[1]], axis=1)
        )
    ax = plot_distribution(axarr[1], X[non_outliers], y[non_outliers],
                      hist_nbins=50,
                      x0_label=xlabels[0],
                      x1_label=xlabels[1],
                      title="Zoom-in")

    mpl.colorbar.ColorbarBase(ax_colorbar, cmap=cm.plasma_r,
                              norm=norm, orientation='vertical',
                              label='Color mapping for values of ' + xlabels[-1])

    return ax.get_figure()

if __name__ == '__main__':

    # Create the dataset
    ref_var = ['rain', 'mslp', 't2m', 'rh2m']

    path = 'data/train_data/'
    path_output = create_path('outputs/test_scalers/')

    var_key = 't2m'
    init_time = '00'
    domain = 'd01'

    ref_var.remove(var_key)
    file_name = '_'.join( [var_key] + ref_var + [domain, init_time]) + '.mat'

    print('Loading data...')
    data = loadmat(path + file_name, squeeze_me=True)
    #
    for key in ['__header__', '__version__', '__globals__']:
        _ = data.pop(key)

    # join all data
    dataset = []
    for stn, values in data.iteritems():
        dataset.append(values)
    if len(dataset) > 0:
        values = np.concatenate(dataset, axis=0)
    else:
        exit()

    # Take only 2 features to make visualization easier
    # also take a random sample of the data of length `num_samples`
    num_samples = 50000
    samples = np.random.randint(0, high=len(values), size=num_samples)
    samples.sort()

    X = values[:, [2, -1]][samples]
    y_full = values[:, 1][samples]

    # scale the output between 0 and 1 for the colorbar
    y = minmax_scale(y_full)
    norm = mpl.colors.Normalize(y_full.min(), y_full.max())

    for name, scaler in Scalers.iteritems():

        fig = make_plot(
            scaler.fit_transform(X), y,
            title=name,
            xlabels=['forecast temp2m', 'forecast relh2m', 'observed ' + var_key],
            norm=norm)

        fig.savefig(path_output + '_'.join(name.split(' ')) + '.png')
