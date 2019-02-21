#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
======================================
Decision Tree Regression with AdaBoost
======================================

A decision tree is boosted using the AdaBoost.R2 [1]_ algorithm.

.. [1] H. Drucker, "Improving Regressors using Boosting Techniques", 1997.

"""
from __future__ import print_function

print(__doc__)

# Author: Yanmichel Morfa <morfayanmichel@gmail.com>

import os
import numpy as np
import matplotlib.pyplot as plt
from utils.custom_losses import *
from utils.utils import TrainDataSet
from plotting_tools import scatter_plot, plot_dates
from apply_regression import apply_regression, create_path
from utils.utils import save_model, load_model, loadmat, savemat
from utils.utils import StandardScaler, train_test_split, to_categorical
from sklearn_models import models, rng, BaggingRegressor, AdaBoostRegressor
from keras_models import RBF_Regressor, run_experiment

parameters = {
    'batch_size': 132,
    'epochs': 100,
    'test_split': 0.1,
    'val_split': 0.1,
    'test_shuffle': False,
    'train_shuffle': True
}

if __name__ == '__main__':

    # Create the dataset
    path = 'data/train_data/'

    var_key = 't2m'
    init_time = '00'
    domain = 'd01'

    mod_path = create_path(os.path.join('models', var_key))

    file_name = 't2m_rain_mslp_rh2m_d01_00.mat'

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

    # create dataset
    # data = TrainDataSet(values[:, 1:])
    # data.set_predictors(columns=[2, -1])

    # hyper-parameters
    split_test = 0.1
    shuffle = False
    n_estimators = 3
    nodes = 4  # number of threats

    # use_model = models['mlp']

    # Apply AdaBoosting to estimator
    # regressor = AdaBoostRegressor(use_model, n_estimators=n_estimators, random_state=rng)

    # regressor = BaggingRegressor(base_estimator=use_model, n_estimators=3,
    #                              max_samples=1.0, max_features=1.0, bootstrap=True,
    #                              oob_score=True, n_jobs=-1, random_state=rng, verbose=1)

    regressor = RBF_Regressor(
        units=50,
        kernel_activation='cauchy',
        output_activation='linear',
        num_inputs=2,
        num_classes=1)

    scaler = StandardScaler()

    # # scale the data
    # predictors = np.stack( [values[:, 2], values[:, -1] ], axis=-1 )
    # x_data, y_data = scaler.fit_transform(predictors), values[:, 1]

    predictors = np.stack( [values[:, 2], values[:, -1] ], axis=-1 )
    x_data, y_data = scaler.fit_transform(predictors), values[:, 1]

    # save scaler for later use on implementation
    str_mssg = "Standard scaler parameters: mean = {}, variance = {}"
    print(str_mssg.format(scaler.mean_, scaler.var_))
    save_model(scaler, os.path.join(mod_path, 'standard_scaler.mod'))

    # split data for validation and testing
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=split_test, shuffle=shuffle)

    # free some memory ...
    del x_data, y_data

    # fitting model
    print('----------------------------------------------------------------')
    print('inputs: integer tensor of shape (samples, predictors)')
    print('inputs_train shape:', x_train.shape)
    print('inputs_test shape:', x_test.shape)
    print('----------------------------------------------------------------')
    print('outputs: float, tensor of shape (samples, 1)')
    print('outputs_train shape:', y_train.shape)
    print('outputs_test shape:', y_test.shape)
    print('----------------------------------------------------------------')
    print('Compiling...')

    history_model = run_experiment(
        regressor, x_train, y_train, parameters, save_model=True)

    score_model = regressor.evaluate(
        x_test, y_test, batch_size=len(x_test), verbose=1)

    print('\nNetwork results')
    print('Test score:', score_model[0])
    print('Test accuracy:', score_model[1])

    # # fitting model
    # regressor.fit(x_train, y_train)
    # print("Test score: {0:.2f} %".format(regressor.score(x_test, y_test)))
    # save_model(regressor, os.path.join(mod_path, 'regression_model.mod'))

    # Predict
    y_pred = regressor.predict(x_test)

    print("Test score: {0:.2f} %".format(r2_score(y_test, y_pred)))

    #==========================================================================
    x_test = scaler.inverse_transform(x_test)
    x_plot = x_test[:, 0]

    fig = scatter_plot(y_test, y_pred, x_plot)
    fig.savefig('outputs/'+file_name+'_sklearn_plot.png', dpi=300)
    plt.show()
    #==========================================================================
