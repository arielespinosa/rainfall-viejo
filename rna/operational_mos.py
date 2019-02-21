#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

"""
from __future__ import print_function

print(__doc__)
#
# Author: Yanmichel Morfa <morfayanmichel@gmail.com>
#

# importing necessary libraries
from utils.utils import create_path, load_model




if __name__ == '__main__':

    path_input = 'data/stn_vs_raw'
    path_output = create_path('data/stn_vs_mos')

    regressor = load_model('regression_model')

    #
    apply_regression(
        path_input,
        path_output,
        var_key,
        init_times=init_times,
        domains=domains,
        regressors=regressor,
        verbose=True)
