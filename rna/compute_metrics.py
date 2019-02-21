#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Friday July 27 00:28:35 2018

@author: yanmichel morfa
"""
from __future__ import print_function

import os
import numpy as np
import pandas as pd
from glob import glob
from utils.custom_losses import *
from utils.utils import create_path, timedelta, combine
from utils.plotting_tools import plt, plot_lines
from multiprocessing import Pool, cpu_count
from functools import partial

# global variables:
init_times = ["0000", "1200"]
domains = ["d01", "d02"]

combined_list = combine([init_times, domains])

# dictionary with variables names and aliases
variables = {
    'Temperatura': 't2m',
    'Precipitacion': 'rain',
    'Velocidad del Viento': 'wind',
    'Presion a nivel del mar': 'mslp',
    'Humedad Relativa': 'rh2m'
}

#
path_join = os.path.join

from_dirs = ['stn_vs_raw', 'stn_vs_mos']
input_dir = 'data'
output_dir = 'data/comparison/computed_metrics'

# loss functions to messure performance:
# each element in this dictionary must be a callable accepting
# two arguments: (y_true, y_pred)

# wrapper for root mean squared error from sklearn (mse):
rmse = lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred))

# wrapper for roc_score as if it were a clasiffication problem:

metrics = {
    # Classical metrics:
    'pearson': pearson_correlation,
    'mse': mean_squared_error,
    'rmse': rmse,
    'mae': mean_absolute_error,
    # Explained variance regression score function best possible score is 1.0.
    'evscore': explained_variance_score,
    # Best possible score is 1.0 and it can be negative (because the
    # model can be arbitrarily worse). A constant model that always
    # predicts the expected value of y, disregarding the input features,
    # would get a R^2 score of 0.0.
    'r2score': r2_score
}


def process_files(from_dir, this_file):

    stn = this_file.split('/')[-1]
    print("Prossecing station: ", stn)
    path = create_path(path_join(output_dir, from_dir, stn))

    for init_time, domain in combined_list:

        var_metrics = []
        for var in variables.itervalues():

            file_names = '_'.join([var, domain, '*']) + ".txt"
            stages = glob(path_join(this_file, init_time, file_names))

            # sorting stages
            stages_idx = [s.split('_')[-1].split('.')[0] for s in stages]
            stages_idx = np.argsort(np.array(stages_idx, dtype='int32'))
            stages = [stages[idx] for idx in stages_idx]

            # computing all metrics for each variable at the same stage
            s_metrics = [
                compute_metrics(stg, metrics.values()) for stg in stages
            ]
            var_metrics.append(s_metrics)

        var_metrics = np.transpose(var_metrics, axes=(2, 1, 0))
        for name, arr in zip(metrics.keys(), var_metrics):

            sf = path_join(path, '_'.join([name, init_time, domain]) + '.txt')
            dframe = pd.DataFrame(data=arr, columns=variables.values())
            dframe.to_csv(sf, index=True, header=True, sep=' ', na_rep='nan')


def plot_files(combined_list):

    # get values from iterable:
    metric, stn_id, domain = combined_list

    print("Creating '{}' plots for station: {}, domain: {}".format(
        metric, stn_id, domain))

    # get files from model and mos at the same time:
    sorted_files = []
    for from_dir in from_dirs:
        path = path_join(output_dir, from_dir, stn_id,
                         '_'.join([metric, '*', domain]) + '*')
        sorted_files.extend(sorted(glob(path)))

    plot_stages = []
    plot_values = []
    for this_file in sorted_files:

        dframe = pd.read_csv(this_file, delimiter=' ', index_col=0)

        # create a nice string representation of forecast hours
        # (adding stages to initialization hour)
        init = this_file.split('.')[0].split('_')[-2][:2]
        stages = [
            str(timedelta(hours=int(init) + 3.0 * off_set)).split(', ')[-1]
            for off_set in range(dframe.shape[0])
        ]
        plot_stages.append(stages)
        plot_values.append(dframe)

    for name, var in variables.iteritems():

        # list containning values for all initializations of the same
        # variable (var) for the same domain...
        data = [values[var].values for values in plot_values]

        # finally doing some actual plotting
        fig = plot_lines(plot_stages, data, name, stn_id, domain, metric)

        fig_path = create_path(path_join(output_dir, 'plots', stn_id))
        fig_name = path_join(fig_path, '_'.join([metric, var, domain]))

        fig.savefig(fig_name + '.png', dpi=200)
        plt.close(fig)


def compute_metrics(file, metrics):
    '''
        Read single file and compute metrics
    '''
    _, obs, fcs = np.loadtxt(file).T

    # filtering nan values
    not_nans = np.logical_and(np.isfinite(obs), np.isfinite(fcs))

    return [metric(obs[not_nans], fcs[not_nans]) for metric in metrics]


#=================== main functions for running in parallel ===============

def create_files(n_process=1):

    for from_dir in from_dirs:

        print('Creating files from directory: {}'.format(from_dir))

        pool = Pool(n_process)

        # partialize arguments for using pool.map()
        funct = partial(process_files, from_dir)
        stn_files = glob(path_join(input_dir, from_dir, "stn_*"))

        pool.map(funct, stn_files)
        pool.close()
        pool.join()


def make_plots(plot_metrics=[], n_process=1):

    stn_files = os.listdir(path_join(output_dir, from_dirs[0]))
    combined_list = combine([plot_metrics, stn_files, domains])

    pool = Pool(n_process)

    pool.map(plot_files, combined_list)
    pool.close()
    pool.join()


if __name__ == "__main__":

    n = cpu_count()
    print('Running with {:n} workers.'.format(n))

    # save validation data to ascii files ???
    #create_files(n_process=n)

    # plot computed metrics from files
    make_plots(plot_metrics=['r2score', 'evscore', 'rmse', 'pearson'], n_process=n)
