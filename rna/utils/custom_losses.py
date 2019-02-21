#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
from keras import backend as K
from scipy.stats import linregress
from custom_layers import margin_loss
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import explained_variance_score, r2_score, f1_score
# categorical scores
from sklearn.metrics import recall_score, roc_auc_score
from sklearn.metrics import brier_score_loss, precision_score

epsilon = 1.e-10


def abse(x, y):
    return abs(x - y)


def sqre(x, y):
    return np.square(x - y)


def mae(x, y):
    '''
        Compute mean absolute differences between samples x and y
    '''
    return np.nanmean(abse(x, y))


def mse(x, y):
    '''
        Compute mean squared differences between samples x and y
    '''
    return np.nanmean(sqre(x, y))


def msle(x, y):
    '''
        Compute mean squared logarithmic error
    '''
    return mean_squared_log_error(x, y)

def rmse(x, y):
    '''
    Compute root mean squared differences between samples x and y
    '''
    return np.sqrt(mse(x, y))


def nrmse(x, y):
    '''
    Compute normalized root mean squared differences between samples x and y
    '''
    xmin = np.nanmin(x)
    xmax = np.nanmax(x)

    den = xmax - xmin if xmax != xmin else xmax

    return rmse(x, y) / den.clip(min=epsilon)  # clip to avoid overflow


# Perceptual metrics:
def psnr(x, y):
    '''
    Compute peak signal-to-noise ratio between samples x and y
    '''
    den = rmse(x, y)

    return 20.0 * np.log10(np.nanmax(x) / den.clip(min=epsilon))


def spatial_correlation(x, y):
    '''
    Compute correlation between two samples x and y
    '''
    den = np.nansum(x * x) * np.nansum(y * y)

    return 1.0 - np.nansum(x * y) / np.sqrt(den).clip(min=epsilon)


def pearson_correlation(x, y):
    '''
    Compute correlation between two samples x and y
    '''
    num_x = (x - np.nanmean(x))
    num_y = (y - np.nanmean(y))

    den = np.sqrt(np.nansum(num_x * num_x) * np.nansum(num_y * num_y))

    return np.nansum(num_x * num_y) / den.clip(min=epsilon)

def cross_entropy(x, y):
    '''
     The cross-entropy loss of the predicted frame x
     and the ground-truth frame y:
    '''
    log_arg = x * np.log(y.clip(min=epsilon)) + (1.0 - x) * np.log(
        (1.0 - y).clip(min=epsilon))

    return -np.sum(log_arg)

def apercb(x, y):

    return mae(x, y) / np.nanmean(x).clip(min=epsilon)

def kge(x, y):
    '''
    Kling-Gupta Efficiency (KGE)
    The acceptable range for KGE is considered to be above 0.6
    '''
    _, _, r_value, _, _ = linregress(x, y)

    alpha = np.nanvar(y) / np.nanvar(x).clip(min=epsilon)
    beta = apercb(x, y)

    ed = np.square(r_value-1) - np.square(alpha-1) - np.square(beta-1)

    return 1.0 - np.sqrt(ed)


def correlation_loss(y_true, y_pred):
    #
    den = K.sum(y_true * y_true) * K.sum(y_pred * y_pred)

    return 1.0 - K.sum(y_true * y_pred) / (K.sqrt(den) + K.epsilon())
