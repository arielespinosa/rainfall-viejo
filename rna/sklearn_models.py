#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.tree import DecisionTreeRegressor as DTR
from sklearn.linear_model import PassiveAggressiveRegressor as PAR
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

rng = np.random.RandomState(1)

models = {
    'lin':
    LinearRegression(fit_intercept=True, normalize=True, n_jobs=-1),
    'eln':
    ElasticNet(
        alpha=1.0,
        l1_ratio=0.5,
        fit_intercept=True,
        normalize=False,
        tol=1e-8,
        random_state=rng,
        selection='random'),
    'par':
    PAR(C=1.0,
        fit_intercept=False,
        tol=None,
        shuffle=True,
        verbose=1,
        loss='epsilon_insensitive',
        epsilon=0.01,
        random_state=rng),
    'svr_rbf':
    SVR(kernel='rbf', C=1e3, shrinking=True, verbose=True),
    'svr_ply':
    SVR(kernel='poly', C=1e3, degree=3, shrinking=True, verbose=True),
    'gpr':
    GPR(kernel=None, alpha=1e-10, optimizer='fmin_l_bfgs_b', random_state=rng),
    'dtr':
    DTR(max_depth=10),
    'kr_rbf':
    KernelRidge(kernel='rbf', gamma=0.1, alpha=1e-2),
    'kr_ply':
    KernelRidge(kernel='poly', gamma=10.1, alpha=1e-2, degree=3),
    'mlp_r':
    MLPRegressor(
        hidden_layer_sizes=(
            10,
            8,
            5,
        ),
        activation='tanh',
        solver='adam', #'lbfgs',  #
        alpha=0.0001,
        batch_size=32,
        power_t=0.5,
        max_iter=1000,
        shuffle=True,
        random_state=rng,
        tol=1e-8,
        learning_rate='adaptive',
        learning_rate_init=0.02,
        verbose=False,
        warm_start=True,
        momentum=0.9,
        nesterovs_momentum=True,
        early_stopping=False,
        validation_fraction=0.2,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-08),
    'mlp_c':
    MLPClassifier(
        hidden_layer_sizes=(10, 8, 4),
        activation='tanh',
        solver='lbfgs',
        alpha=0.0001,
        batch_size=32,
        max_iter=500,
        shuffle=True,
        random_state=rng,
        tol=1e-8,
        learning_rate='adaptive',
        learning_rate_init=0.01,
        verbose=True,
        momentum=0.9,
        early_stopping=False,
        validation_fraction=0.2)
}

# careful with adding other configuration parameters
# this search grow in a conbinatorial way ...
param_grid = {
    # testing several arquitectures
    'hidden_layer_sizes': [(10, 3), (3, ), (5, 1,), (6, 3, 2),],
    # identity activation was excluded
    'activation': ['logistic', 'tanh', 'relu'],
    'alpha': [0.01, 0.001, 0.0001, 0.00001],
    'learning_rate_init': [0.1, 0.05, 1e-2, 1e-3, 1e-4],
}
''' Top best configurations:
   - parameters: {'alpha': 0.0001, 'activation': 'tanh', 'learning_rate_init': 0.001, 'hidden_layer_sizes': (5, 5, 5)} with a score of 0.76
   - parameters: {'alpha': 0.001, 'activation': 'tanh', 'learning_rate_init': 0.1, 'hidden_layer_sizes': (3,)} with a score of 0.80
'''
