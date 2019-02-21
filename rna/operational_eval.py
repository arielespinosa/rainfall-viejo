#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import numpy as np
import pandas as pd
from glob import glob
from utils.custom_losses import pearson_correlation
from utils.utils import str2stamp, create_path, combine

# globals
var_keys = ['rain', 'mslp', 't2m', 'rh2m']
names = ['dates', 'observed', 'forecast']

time_cycle = 3  # hours
time_units = 'seconds since 2000-01-01 00:00:00'
time_format = "%Y%m%d%H"

date_converter = {'dates': lambda x: np.int32(x)}


class Dummy_Regressor(object):
    def __init__(self, x):
        self.x = x

    def predict(self):
        #
        return self.x

def apply_physics(relh, temp, press):

    # get relative humidity: range (0, 1)
    relh *= 1e-2

    # get virtual temperature increment: range (0, 1)
    temp = temp2kelvin(temp)

    # get scaled pressure : range (0, 1)

    return relh, calc_moist(temp, press, relh), calc_exner(press)


def apply_regression(path_input,
                     path_output,
                     var,
                     init_times=[],
                     domains=[],
                     regressors=None,
                     verbose=False):

    ind = var_keys.index(var)
    #
    for init_time, domain in combine([init_times, domains]):

        # file_name patterns to search
        wildcards = [
            '*'.join([var_key, domain] + ['']) for var_key in var_keys
        ]

        print('find files with patterns: ', wildcards,
              'from initialization: ', init_time, '\n')

        stn_files = os.listdir(path_input)

        init_time += '00'
        stacked_correlations = {}
        for stn_id in stn_files:
            stn = int(stn_id.split('_')[-1])
            if verbose:
                print('Creating bias-corrected data for station: ', stn)

            path = create_path(
                os.path.join(path_output, stn_id, init_time))

            var_files = [
                sorted(
                    glob(
                        os.path.join(path_input, stn_id, init_time,
                                     wildcard))) for wildcard in wildcards
            ]

            # crate dictionary for computed correlations (one for each stage)
            stage_correlations = {s: None for s in range(25)}
            for same_stages in zip(*var_files):

                stage = same_stages[0].split('.')[0].split('_')[-1]

                dates = []
                observations = []
                predictions = []
                for this_file in same_stages:

                    data = pd.read_table(
                        this_file,
                        names=names,
                        sep=' ',
                        converters=date_converter,
                        engine='c')

                    # remove nan model values
                    find_nans = np.isnan(data['forecast'].values)

                    indexs = [not nans for nans in find_nans]

                    # append member to existing ones
                    dates.append(data['dates'].values[indexs])
                    observations.append(data['observed'].values[indexs])
                    predictions.append(data['forecast'].values[indexs])
                dates = np.array(dates)
                if dates.shape[0] == len(var_files) and dates.ndim != 1:

                    # only if all predictors share the same dates
                    if not np.diff(dates, axis=0).all():

                        # creating predictors (some physics here): mslp, temp, relh
                        if var is 'rain':
                            x = np.stack(predictions, axis=0)
                            x[-1], x[-2], x[-3] = apply_physics(
                                x[-1], x[-2], x[-3])
                            # normalization factor for target variable
                            scale = 1.0  # mm / 3h
                        else:
                            # (temp, press, wind?) no need for extra
                            # predictions (for now) we will only use
                            # model forecasts: only simple normalization
                            x = np.expand_dims(predictions[ind], axis=0)
                            # normalization factor for target variable
                            scale = 36.0  # degrees celcius
                            x /= scale

                        x_pred = x.T
                        if hasattr(regressors, 'predict'):

                            corrected = regressors.predict(x_pred) * scale

                        elif regressors.has_key(stn_id):

                            corrected = regressors[stn_id].predict(
                                x_pred) * scale

                        # just use forecast values
                        else:
                            print(
                                'MOS is not available for this station at stage: ',
                                str2stamp(
                                    str(dates[0]),
                                    off_set=3 * int(stage),
                                    to_julian=False,
                                    as_int=False),
                                ' using forecast instead')
                            corrected = predictions[ind]

                        stacked_array = np.stack(
                            [dates[0], observations[ind], corrected],
                            axis=-1)

                        # write stn predictions to disk
                        file_name = '_'.join([var, domain, stage])
                        file_name = os.path.join(path, file_name)
                        np.savetxt(file_name + '.txt', stacked_array)

                        stage_correlations[int(
                            stage)] = pearson_correlation(
                                stacked_array[:, 1], stacked_array[:, 2])

                else:
                    # if station is no suitable for bias-correction
                    # algorithm then use model forecast only:
                    stage_correlations[int(stage)] = pearson_correlation(
                        observations[ind], predictions[ind])

            # write pearson correlations for gridded mos
            stacked_correlations[stn] = stage_correlations.values()

        # write scv file for make plots:
        stacked_correlations = pd.DataFrame(data=stacked_correlations)

        file_name = '_'.join(['pearson', init_time, domain]) + '.csv'
        path = create_path(
            os.path.join(
                'test_data/processed/CorrelacionesPuntuales/PearsonT/MOS/pearson',
                var))

        print('Save file: ', file_name, 'with data shaped: ',
              stacked_correlations.values.shape, '\n')

        stacked_correlations.to_csv(
            os.path.join(path, file_name),
            sep=',',
            na_rep='nan',
            header=list(stacked_correlations.columns),
            index=False,
            mode='w')

        print('Post-processing finished. Congrats!')


if __name__ == '__main__':
    '''
        This script generates the data for each station after appling the
        regression equation found for that station and then save the file
        in the same fashion as the inputs: /stn_***/init/var_d0*_*.txt

        The bias-corrected series does not cover the same dates because
        simultaneous ocurrences of the predictors is not granted and
        of course the nans filtering process
        (this should be seriously revised for better regression models)
    '''

    # create training dataset
    path_input = 'data/stn_vs_raw/'
    path_output = create_path('data/stn_vs_mos/')

    var_key = 'rain'
    init_times = ['00']  #, '12']
    domains = ['d01']  #, 'd02']:

    regressors = {'stn_308': Dummy_Regressor, 'stn_320': Dummy_Regressor}

    apply_regression(
        path_input,
        path_output,
        var_key,
        init_times=init_times,
        domains=domains,
        regressors=regressors)
