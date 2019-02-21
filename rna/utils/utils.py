#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 18:25:43 2018

@author: yanm
"""
import numpy as np
import pandas as pd
from os import makedirs
from numpy import array
import numpy.core.numeric as _nx
from sklearn.externals import joblib
from scipy.io import loadmat, savemat
from keras.utils import to_categorical
from sklearn.preprocessing import binarize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from netCDF4 import num2date, date2num, datetime, timedelta

time_units = 'seconds since 2000-01-01 00:00:00'
time_format = "%Y%m%d%H"

def str2stamp(date, off_set=0, fmt=time_format, to_julian=True, as_int=True):
    '''
    Wrapper of 'strptime' and strftime' functions to convert datetime
    objects to string after adding some offset based on time_format.
    '''
    if not isinstance(date, str):
        raise ValueError("date must be a string")

    date = datetime.strptime(date, time_format) + timedelta(hours=off_set)

    if to_julian:
        # return julian date from time_units
        return date2num(date, time_units)
    else:
        if as_int:
            # return a long integer in the form: 'yyyymmddhh'
            return int(date.strftime(fmt))
        else:
            # return a nice string representation of a date string
            return date

def create_path(path):
    try:
        makedirs(path)
    except:
        pass
    return path


def save_model(model, filename='model'):
    '''
        Function for saving a python object
    '''
    if isinstance(filename, str):

        with open(filename, mode='wb') as to_file:
            joblib.dump(model, to_file)
    else:
        raise ValueError('Argument `filename` must be a string')

def load_model(filename):
    #
    if os.path.exists(filename):
        with open(filename, mode='rb') as from_file:
            return joblib.load(from_file)
    else:
        raise ValueError('File: ' + filename + ' does not exist')



# Arrange temporal data
def load_data(file_paths,
              normalize_data=False,
              split_test=0.1,
              shuffle=True,
              add_features=True):

    X_data, y_data = [
        loadmat(path, squeeze_me=True)['values'] for path in file_paths
    ]

    # add a new dimension for channels:
    if add_features:
        X_data = np.expand_dims(X_data, axis=-1)

    fitted_scaler = None
    if normalize_data:
        # data normalization (0.0-1.0)
        print 'Normalizing data between range: (0.0-1.0)...'
        X_data, fitted_scaler = normalize(
            X_data, MinMaxScaler(), inverse=False)

        print 'Data values extend from: ', X_data.min(), ' to ', X_data.max()

    # split factor:
    split_test = np.clip(split_test, 0.1, 0.9)
    data = train_test_split(
        X_data, y_data, test_size=split_test, shuffle=shuffle)

    # data = (training X, testing X, training y, testing y)
    return data, fitted_scaler


def load_series(filename, seq_len, split_test=0.1):

    data = pd.read_csv('./data/' + filename)

    #dates = data.values[:,0]
    data = data.values[:, 1:]

    seq_len += 1

    result = np.array(
        [data[i:i + seq_len] for i in range(len(data) - seq_len)])

    factor = np.clip(1. - split_test, 0., 1.)
    row = int(factor * result.shape[0])

    train = result[:row]
    x_test = result[row:, :-1]
    y_test = result[row:, -1]

    return (train[:, :-1], train[:, -1]), (x_test, y_test)

def series_generator(data, lookback, delay, batch_size=32, step=1, shuffle=False):

    min_index = 0
    max_index = len(data) - delay - 1
    i = min_index + lookback
    while True:
        if shuffle:
            rows = np.random.randint(
                min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)

        row_len = len(rows)
        samples = np.zeros((row_len,
                           lookback // step,
                           data.shape[-1]))
        targets = np.zeros((row_len,))
        for j, row in enumerate(rows):
            indices = range(row - lookback, row, step)
            samples[j] = data[indices]
            targets[j] = data[rows + delay][1]
        yield samples, targets


class TrainDataSet(object):
    """docstring for TrainDataSet"""

    def __init__(self, x):
        super(TrainDataSet, self).__init__()

        self.values = np.asanyarray(x)
        self.dtype = self.values.dtype
        self.shape = self.values.shape
        self.ndims = self.values.ndim
        self.values = self._transform()

        self.n_samples = self.shape[0]
        self.n_features = self.shape[1]

    def _transform(self):
        #
        if self.ndims < 2:
            return np.expand_dims(self.values, axis=-1)
        elif self.ndims > 2:
            return np.reshape(self.values, (self.shape[0], -1))
        else:
            return self.values

    def get_binary(self, threshold=0.0, as_bool=False):
        # doc this later

        x = binarize(self.values, threshold=threshold).reshape(self.shape)

        if as_bool:
            return np.where(x >= 1, True, False)
        else:
            return x.astype(self.dtype)

    def get_categories(self, classes=None):
        # doc this later ...
        if classes is None:
            nbins = 'auto'
        elif np.isscalar(classes) or np.iterable(classes):
            nbins = classes
        else:
            raise ValueError('type of argument `classes` not understood')

        # compute histogram
        _, thresholds = np.histogram(self.values, nbins)

        data = self.values
        min_bin = thresholds[0]
        for label, max_bin in enumerate(thresholds[1:]):

            cond = np.logical_and(data >= min_bin, data <= max_bin)
            data = np.where(cond, label, data)
            min_bin = max_bin

        return thresholds, data.astype('int32')

    def get_categorical(self):
        # doc this later ...

        if 'int' not in str(self.dtype):

            self.thresholds, x = self.get_categories()
            num_classes = len(self.thresholds) - 1

        else:
            x = self.values
            num_classes = self.values.max() + 1

        # convert to categorical
        categories = to_categorical(x, num_classes).astype(self.dtype)

        if hasattr(self, 'thresholds'):
            return self.thresholds, categories
        else:
            return categories

    def set_train_pairs(self, train_dict={}):
        #
        n_features = range(self.n_features)
        if columns is None:
            columns = n_features
        else:
            columns = _nx.normalize_axis_tuple(columns, self.n_features)

        data = np.stack(
            [
                self.values[:, n_features.pop(n_features.index(col))]
                for col in columns
            ],
            axis=-1)

        if name is 'predictors':
            self.predictors = data
        elif name is 'predictand':
            self.predictors = data
        elif name is 'both':

            if len(n_features) != 0:
                left_data = self.predictand = np.stack(
                    [self.values[:, f] for f in n_features], axis=-1)
                self.predictand = left_data
            self.predictors = data

    def get_predictors(self, columns=None):
        #
        if not hasattr(self, 'predictors'):
            self.predictors = self.set_predictors(self, columns=columns)

        return self.predictors

    def split_dataset(self, test_size=0.1, shuffle=True):
        #
        if hasattr(self, 'predictand'):
            return train_test_split(
                self.predictors,
                self.predictand,
                test_size=test_size,
                shuffle=shuffle)
        else:
            return train_test_split(
                self.values, test_size=test_size, shuffle=shuffle)

    def _normalization(self, data=None, inverse=False):
        # doc this later ...

        if data is None:
            x = self.values
        elif hasattr(self, data):
            if data is 'predictors':
                x = self.predictors
            else:
                x = self.predictand
        else:
            raise ValueError('Method `set_predictors`, must be called first')

        if inverse:
            return self.scaler.inverse_transform(x).astype(self.dtype)
        else:
            self.scaler = StandardScaler().fit(x)
            return self.scaler.transform(x)

    def cofiguration():
        # implement this to print the status of the dataset !!
        pass


def combine(elm_set):
    ''' Permute all to all elements in a set
    this is to avoid nesting all loops together

    example:
        >>> for elm in combine( [[0,1], [2, 3, 4]] ):
        >>>     x, y = elm

        is equivalent to:

        >>> for i in [0, 1]:
        >>>     for j in [2, 3, 4]:
        >>>         x, y = i, j

        but more elegant, not necessary faster as long as you pre-compute the iterator
    '''
    queue = np.meshgrid(*elm_set)

    return [[e.tolist() for e in elms] for elms in np.nditer(queue)]


def categorize(x, nbins):
    # compute histogram
    _, thresholds = np.histogram(x, nbins)

    min_bin = thresholds[0]
    for label, max_bin in enumerate(thresholds[1:]):

        cond = np.logical_and(x >= min_bin, x <= max_bin)
        x = np.where(cond, label, x)
        min_bin = max_bin

    return thresholds, x.astype('int32')

def transform_categorical(y, classes=None):
    # doc this later ...
    if classes is None:
        nbins = 'auto'
    elif np.isscalar(classes) or np.iterable(classes):
        nbins = classes
    else:
        raise ValueError('type of argument `classes` not understood')
    flag = 'int' not in str(y.dtype)

    if flag:
        thresholds, x = categorize(y, nbins)
        num_classes = len(thresholds) - 1
    else:
        x = y.copy()
        num_classes = y.max() + 1

    # convert to categorical:
    categories = to_categorical(x, num_classes).astype(y.dtype)

    if flag:
        return thresholds, categories
    else:
        return categories
