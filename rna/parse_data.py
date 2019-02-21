#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import numpy as np
import pandas as pd
from glob import glob
from scipy.io import loadmat, savemat
from functools import partial
from multiprocessing import Pool, cpu_count
from utils.utils import create_path, str2stamp, combine

# globals
names = ['dates', 'observed', 'predicted']
time_cycle = 3  # hours between consecutive dates
print_message = 'dropped {} stations: {}.'
dims = [slice(None), -1, np.newaxis]


def get_fields(path, init_time='00', wildcards='*', chunks=60, verbose=False):

    fields_dict = {}

    stn_files = os.listdir(path)

    init_time += '00'
    for stn_id in stn_files:

        if verbose: print('Creating validation data for station: ', stn_id)

        files = sorted(glob(os.path.join(path, stn_id, init_time, wildcards)))

        dates = []
        observations = []
        predictions = []
        for this_file in files:

            # in-place date convertion (adding stages to initialization hour)
            stage = time_cycle * int(this_file.split('.')[0].split('_')[-1])
            date_converter = {
                # 'dates': lambda x: str2stamp(x, off_set=stage, to_julian=False)
                'dates': lambda x: int(x) + stage
            }

            data = pd.read_table(
                this_file,
                names=names,
                delimiter=' ',
                converters=date_converter,
                # for low level operations `C` engine is recomended
                engine='c')

            # remove nan values from model and observations
            not_nans = np.logical_and(
                np.isfinite(data['observed'].values),
                np.isfinite(data['predicted'].values))

            # append member to existing ones
            dates.extend(data['dates'].values[not_nans])
            observations.extend(data['observed'].values[not_nans])
            predictions.extend(data['predicted'].values[not_nans])

        reordered = np.argsort(dates)
        fields_dict[stn_id] = np.stack(
            [[dates[ind] for ind in reordered], observations, predictions],
            axis=1)[reordered]

    return fields_dict

# function for processing a single combination:
def process_item(path_input, path_output, unique, items_list):

    init_time, domain, var_key = items_list

    # file_name patterns to search
    sufix = [var_key, domain]
    wildcards = '*'.join(sufix + [''])

    print('find files with pattern: ', wildcards, 'from initialization: ',
          init_time)

    data_dict = get_fields(
        path_input, init_time=init_time, wildcards=wildcards, verbose=True)

    sufix = '_'.join(sufix + [init_time])
    # find unique dates for all stations: (clustering the data data)
    flag = True
    if unique:
        sufix += '_unique'
        # find intersections:
        start_stn = data_dict.keys()[-1]
        unique_set = set(data_dict[start_stn][:, 0])
        for stn, values in data_dict.iteritems():
            unique_set.intersection_update( set(values[:, 0]) )

        flag = len(unique_set) > 0
        if flag:
            for stn, values in data_dict.items():

                dates = values[:, 0]
                index = [date in unique_set for date in dates]
                data_dict[stn] = values[index]
        else:
            print('Empty set found at file: {}'.format(sufix))

    if flag:
        file_name = os.path.join(path_output, sufix)
        savemat(file_name, data_dict, do_compression=True)

def combined_items(path_input, path_output, var):

    print_message = 'dropped {} stations: {}.'

    dim = [slice(None), -1, np.newaxis]
    #
    for init_time in ['00', '12']:

        for domain in ['d01', 'd02']:

            # file_name patterns to search
            var_keys = ['rain', 'mslp', 't2m', 'rh2m']
            wildcard = [
                '*'.join([var_key, domain] + ['']) for var_key in var_keys
            ]

            print('find files with patterns: ', wildcard,
                  'from initialization: ', init_time, '\n')

            predictors = [
                get_fields(path_input, init_time=init_time, wildcards=pattern)
                for pattern in wildcard
            ]

            # isolate precipitation data
            index = var_keys.index(var)
            var_dict = predictors.pop(index)
            var_keys.remove(var)

            stacked_data = {}
            dropped_stations = []
            # iterate for predictors
            for var_key, predictor in zip(var_keys, predictors):

                # get variables per station:
                for stn, values in predictor.iteritems():

                    if not stacked_data.has_key(stn):
                        stacked_data[stn] = []

                    dates = values[:, 0]
                    now_len = len(dates)
                    ref_len = len(var_dict[stn])

                    if now_len != ref_len:

                        ref_dates = var_dict[stn][:, 0]

                        indexs = [date in ref_dates for date in dates]
                        values = values[indexs]

                        if len(values) == ref_len:
                            stacked_data[stn].append(values[dim])
                        else:
                            if stn not in dropped_stations:
                                dropped_stations.append(stn)

                    else:
                        stacked_data[stn].append(values[dim])

            combined_dict = {}
            for stn in var_dict.keys():

                if stn not in dropped_stations:

                    combined_dict[stn] = np.concatenate(
                        [var_dict[stn]] + stacked_data[stn], axis=-1)

            dropped_stations = [stn.split('_')[-1] for stn in dropped_stations]
            print(print_message.format(
                len(dropped_stations), ', '.join(dropped_stations)), '\n')

            file_name = os.path.join(path_output, '_'.join([
                var,
            ] + var_keys + [domain, init_time]))
            savemat(file_name, combined_dict, do_compression=True)


def assingle_items(path_input, path_output, unique=False, n_process=None):

    # compute iterable for parallel processing:
    combined_list = combine([['00', '12'], ['d01', 'd02'],
                             ['rain', 'mslp', 't2m', 'rh2m', 'wind']])

    # create pool of workers:
    if np.isscalar(n_process):
        n_process = min(1, int(n_process))
    else:
        n_process = cpu_count()

    print('Running with {:n} workers.'.format(n_process))
    pool = Pool(n_process)

    # running in parallel
    function = partial(process_item, path_input, path_output, unique)

    pool.map(function, combined_list)
    pool.close()
    pool.join()


if __name__ == '__main__':

    # create training dataset
    path_input = 'data/stn_vs_raw'
    path_output = create_path('data/train_data')

    # assingle_items(path_input, path_output, unique=True)

    var_key = 'rain'
    combined_items(path_input,path_output, var_key)
