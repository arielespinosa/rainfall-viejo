from __future__ import print_function

import os
import numpy as np
from glob import glob
import pandas as pd
from scipy.io import loadmat, savemat
from netCDF4 import num2date, date2num, datetime, timedelta

# globals
skip_parameters = ['']
time_units = 'seconds since 2010-01-01 00:00:00'
time_format = "%Y-%m-%d %H:%M:%S"
obs_names = [
    'stn_id', 'stn_name', 'year', 'month', 'day', 'hours', 'temp2m', 'rh2m',
    'dd', 'ff', 'pp', 'P', 'rain'
]


def get_fields(path, var_key, wildcards='*', num_members=4, chunks=60):

    fields_dict = {}

    # create coordinates
    domain = path.split('/')[-1]
    coordn = loadmat(os.path.join(path, 'coords_' + domain + '.mat'))

    grid_shape = coordn['lon'].shape
    fields_dict['lon'] = coordn['lon']
    fields_dict['lat'] = coordn['lat']

    files = sorted(glob(os.path.join(path, wildcards)))

    # Find unique dates for members ...
    print('Creating model data: ', len(files), ' files found')
    members = {}
    for file in files:

        data = loadmat(file)
        for time, var in zip(data['dates'], data[var_key]):

            if not members.has_key(time):
                # create empty lists for unseen data..
                members[time] = []

            # append member to existing ones
            members[time].append(var)

    # create output arrays:
    valid = 0
    times = []
    for time, values in members.items():

        # excluding last value (is an initialization)
        if len(values) > num_members:

            check_values = not np.isnan(values).any() and np.ptp(values) < 1e3
            if check_values:

                members[time] = np.array(values[-num_members - 1:-1])

                print('processed: ', time, 'shape: ', members[time].shape)

                # create times:
                year, month, days_hours = time.split('-')
                day, hour = days_hours.split('_')

                time_args = (int(year), int(month), int(day),
                             int(hour.split('.')[0]))

                times.append(datetime(*time_args))
            else:
                print('discarded: ', time, 'var contains nans: ',
                      np.max(members.pop(time)))

        else:
            print('discarded: ', time, 'shape: ', np.shape(members.pop(time)))

    # store dates
    dates = date2num(times, time_units)

    # index for ordered dates
    ind = np.argsort(dates)

    # stack arrays and send members to last dimension (as channels)
    dates = dates[ind]
    values = np.stack(members.values()).transpose([0, 2, 3, 1])[ind]

    if var_key is 'rain':
        # create actual 3h acumm variables
        time_s = dates[0]
        value_s = values[0]
        fields_dict['time'] = []
        fields_dict['values'] = []
        for time, value in zip( dates[1:], values[1:] ):

            if time-time_s == 10800: # 3 h
                print('calculate acummulated rain from: ', time_s, 'to: ', time)
                fields_dict['time'].append( time )
                fields_dict['values'].append( (value-value_s).clip(min=0.0) )
            time_s = time
            value_s = value

        fields_dict['time'] = np.array( fields_dict['time'] )
        fields_dict['values'] = np.stack( fields_dict['values'] )

    else:
        fields_dict['time'] = dates
        fields_dict['values'] = values

    return fields_dict


def parse_obs(path, grd_dict, var_key, patience=0.8):

    obs_dict = {}

    use_names = ['stn_id', 'year', 'month', 'day', 'hours', var_key]

    # in-place convertions:
    converters = {
        'stn_id': lambda x: int(x[1:4]),  # station id to int
        var_key:
        lambda x: np.nan if 'NULL' in x else np.float(x)  # data to np.float
    }

    print('Parsing csv file: ')

    # create pandas dataframe for date related operations
    dataframe = pd.read_csv(
        path,
        header=None,
        names=obs_names,
        usecols=use_names,
        parse_dates=[['year', 'month', 'day']],
        converters=converters,
        engine='c',
        memory_map=True)

    # create requested dates as datetime objects
    # datetime.strptime('2015-06-30 21:00:00', time_format)
    req_dates = grd_dict['time']

    obs_dates = date2num(
        [  # adding 5 hours to convert to UTC
            date + timedelta(hours=hour + 5) for date, hour in zip(
                dataframe.pop('year_month_day'), dataframe.pop('hours'))
        ],
        time_units)

    counted = dataframe['stn_id'].value_counts(sort=False, dropna=False)

    # now we can extract data based on dates
    ind_s = 0
    ref_len = 1e5
    min_len = int( patience * len(req_dates) ) + 1
    obs_dict['stn_id'] = {}
    obs_dict['time'] = []

    print('Creating observation data: minimum of ', min_len, ' observations are required')

    for stn, num in counted.items():

        ind_e = ind_s + num

        ndates = obs_dates[ind_s:ind_e]
        indexs = [date in req_dates for date in ndates]

        values = dataframe[var_key][ind_s:ind_e][indexs]

        # skip stations with less than 80 % of observations
        len_values = len(values)
        if len_values > min_len:

            ndates = ndates[indexs]
            if len_values < ref_len:
                ref_len = len_values
                ref_dates = ndates

            obs_dict['time'].append(ndates)
            obs_dict['stn_id'][stn] = values

        else:
            print('Dropping station: ', stn, 'with: ', len_values,
                  'observations')
        ind_s = ind_e

    # last check for uncomplete dates ...
    for dates, (stn, values) in zip(obs_dict['time'],
                                    obs_dict['stn_id'].items()):

        indexs = [date in ref_dates for date in dates]
        ref_values = values[ indexs ]

        # just in case ...
        if len(ref_values) < ref_len:
            # when a station with more than 80 % of observed data is taken,
            # we need to drop dates that aren't present in all stations.
            print('dropping station: ', stn, 'with: ',
                  len(obs_dict['stn_id'].pop(stn)[indexs]), 'observations')
        else:

            obs_dict['stn_id'][stn] = ref_values

    # remove dropped dates from model data
    indexs = [date in ref_dates for date in grd_dict['time'] ]

    grd_dict['time'] = grd_dict['time'][indexs]
    grd_dict['values'] = grd_dict['values'][indexs]

    # create actual dictionary to be saved
    obs_dict['time'] = grd_dict['time']
    obs_dict['values'] = np.stack(obs_dict['stn_id'].values()).T
    obs_dict['stn_id'] = obs_dict['stn_id'].keys()

    print('Observation parsing finished, data dimensions: ', obs_dict['values'].shape)

    # get station coordinates from file
    loc = np.loadtxt('test_data/estaciones.dt').T
    loc_dict = {stn:[lon, lat] for stn, lon, lat in zip(*loc)}

    # check which stations were removed:
    obs_dict['lon'] = []
    obs_dict['lat'] = []
    for stn in obs_dict['stn_id']:

        lon, lat = loc_dict.pop(stn)
        obs_dict['lon'].append(lon)
        obs_dict['lat'].append(lat)

    return obs_dict


if __name__ == '__main__':

    domain = 'd01'
    var_key = 'temp2m'
    wildcards = '201*.mat'

    path_output = os.path.join('./data/train_data', domain, var_key)

    try:
        os.makedirs(path_output)
    except:
        pass

    # create training dataset
    path_input = os.path.join('data/model_data', domain)
    grd_fields = get_fields(path_input, var_key, wildcards=wildcards, num_members=3)

    # dates for extracting observations
    path_input = 'data/observed_data/Horarias_2015-16.csv'
    obs_fields = parse_obs(path_input, grd_fields, var_key, patience=0.7)

    savemat(path_output + '/grd_data.mat', grd_fields, do_compression=True)
    del grd_fields

    savemat(path_output + '/stn_data.mat', obs_fields, do_compression=True)
    del obs_fields
