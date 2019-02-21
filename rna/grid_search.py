#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
==================================
Grid Search for Optimal Parameters
==================================

"""
from __future__ import print_function

print(__doc__)
#
# Author: Yanmichel Morfa <morfayanmichel@gmail.com>
#

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

from sklearn_models import models, rng, param_grid
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from multiprocessing import cpu_count
from utils.custom_losses import *
from utils.utils import save_model

if __name__ == '__main__':

    # Create the dataset
    print('Loading data...')
    path = 'data/train_data/t2m_rain_mslp_rh2m_d01_00'
    data = loadmat(path, squeeze_me=True)

    # remove dummy keys first
    for key in ['__header__', '__version__', '__globals__']:
        _ = data.pop(key)

    # join all data
    dataset = []
    for stn, values in data.iteritems():
        dataset.append(values)

    # scale the data
    scaler = StandardScaler()

    predictors = np.stack( [values[:, 2], values[:, -1] ], axis=-1 )
    x_data, y_data = scaler.fit_transform(predictors), values[:, 1]

    # save scaler for later use on implementation
    str_mssg = "Standard scaler parameters: mean = {}, variance = {}"
    print(str_mssg.format(scaler.mean_, scaler.var_))

    # Grid search for estimating best hyper-parameters:
    nb_nodes = cpu_count()
    # cv = StratifiedShuffleSplit(n_splits=5, test_size=0.1, random_state=rng)
    cv = 5

    grid = GridSearchCV( models['mlp_r'], cv=cv,
        param_grid=param_grid,
        n_jobs=nb_nodes,
        pre_dispatch=2 * nb_nodes,
        verbose=1)

    # fitting model
    grid.fit(x_data, y_data)

    print("The best parameters are %s with a score of %0.2f" % (grid.best_params_, grid.best_score_))

    save_model(grid.best_estimator_, 'models/GS_model.mod')
