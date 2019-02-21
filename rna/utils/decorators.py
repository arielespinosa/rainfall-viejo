#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 12:03:02 2017

@author: yanm
"""

import timeit

def profile(function):
    start_time = timeit.time.time()

    def set_time(*args):
        return function(*args)

    print "Excecution time of function '{}': {} seconds".format(
        function.__name__,
        timeit.time.time() - start_time)
    return set_time
