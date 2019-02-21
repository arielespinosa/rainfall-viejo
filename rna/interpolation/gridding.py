#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 16:18:33 2017

@author: yanm
"""
import numpy as np
import interp_functions as intp
from scipy.interpolate import griddata, Rbf
from scipy.spatial.distance import cdist


def interpolate(x,
                y,
                data,
                grid_xy=None,
                interp_type='linear',
                hres=50000,
                minimum_neighbors=3,
                num_pass=1,
                gamma=0.25,
                kappa_star=5.052,
                search_radius=None,
                rbf_func='linear',
                rbf_smooth=0,
                op_param=None,
                comp_err=0):
    r"""Interpolate given (x,y), observation (z) pairs to a grid based on given parameters.

    Parameters
    ----------
    x: array_like
        x coordinate
    y: array_like
        y coordinate
    data: array_like
        observation value
    grid_xy: (grid_x, grid_y) tuple of (N, 2) ndarrays
        Meshgrid for the resulting interpolation in the x and y dimension respectively
        Default None (calculated from x and y)
    interp_type: str
        What type of interpolation to use. Available options include:
        1) "linear", "nearest", "cubic", or "rbf" from Scipy.interpolate.
        2) "natural_neighbor", "barnes", or "cressman" from Metpy.mapping .
        Default "linear".
    hres: float
        The horizontal resolution of the generated grid. Default 50000 meters.
    min_neighbors: int
        Minimum number of neighbors needed to perform barnes or cressman interpolation for a
        point. Default is 3.
    gamma: float
        Adjustable smoothing parameter for the barnes interpolation. Default 0.25.
    kappa_star: float
        Response parameter for barnes interpolation, specified nondimensionally
        in terms of the Nyquist. Default 5.052
    search_radius: float
        A search radius to use for the barnes and cressman interpolation schemes.
        If search_radius is not specified, it will default to the average spacing of
        observations.
    rbf_func: str
        Specifies which function to use for Rbf interpolation.
        Options include: 'multiquadric', 'inverse', 'gaussian', 'linear', 'cubic',
        'quintic', and 'thin_plate'. Defualt 'linear'. See scipy.interpolate.Rbf for more
        information.
    rbf_smooth: float
        Smoothing value applied to rbf interpolation.  Higher values result in more smoothing.

    Returns
    -------
    grid_xy: tuple of two (N, 2) ndarrays
        Meshgrid for the resulting interpolation in the x and y dimensions respectively
    img: (M, N) ndarray
        2-dimensional array representing the interpolated values for each grid.
    """

    # set up resulting grid
    if grid_xy is None:

        grid_x, grid_y = intp.points.generate_grid(
            hres, intp.points.get_boundary_coords(x, y))
    else:
        if len(grid_xy) != 2:
            raise ValueError("when specified, grid_xy must have 2 elements")
        else:
            grid_x, grid_y = grid_xy

    if grid_x.shape != grid_y.shape:
        raise ValueError("grid_x and grid_y must have the same shape")

    # interpolate
    if interp_type in ['linear', 'nearest', 'cubic']:
        points_zip = np.array(list(zip(x, y)))
        img = griddata(points_zip, data, (grid_x, grid_y), method=interp_type)

    elif interp_type == 'natural_neighbor':
        img = intp.natural_neighbor(x, y, data, grid_x, grid_y)

    elif interp_type in ['cressman', 'barnes', 'bratseth']:

        if interp_type == 'cressman':

            img = intp.inverse_distance(x, y, data, grid_x, grid_y,
                                        search_radius, minimum_neighbors,
                                        kind=interp_type)

        elif interp_type == 'barnes':

            ave_spacing = np.mean((cdist(list(zip(x, y)), list(zip(x, y)))))

            kappa = intp.calc_kappa(ave_spacing, kappa_star)

            img = intp.inverse_distance(x, y, data, grid_x, grid_y,
                                        search_radius, minimum_neighbors,
                                        gamma=gamma, kappa=kappa,
                                        num_pass=num_pass, kind=interp_type)
    elif interp_type == 'rbf':

        # 3-dimensional support not yet included.
        # Assign a zero to each z dimension for observations.
        h = np.zeros((len(x)))

        rbfi = Rbf(x, y, h, data, function=rbf_func, smooth=rbf_smooth)

        # 3-dimensional support not yet included.
        # Assign a zero to each z dimension grid cell position.
        hi = np.zeros(grid_x.shape)
        img = rbfi(grid_x, grid_y, hi)

    elif interp_type == 'optimal':

        grid_points = intp.points.generate_grid_coords(grid_x, grid_y)

        if op_param == None:
            raise ValueError(
                'Parameters must be set as input with argument op_param=:')
        elif np.size(op_param) != 4:
            raise ValueError('Argument op_param most be length 4')
        else:
            parameters = np.concatenate((op_param, [data.max()], [data.min()]))

        #else: Interpolate with user specified parameters ...
        args, spect_err = intp.optimal_parameters(
            x, y, data, grid_points, parameters, comp_err=comp_err)

        img = intp.optimal_interp(args).reshape(grid_x.shape)
        if comp_err == 1:
            spect_err = spect_err.reshape(grid_x.shape)

    else:
        raise ValueError('Interpolation option not available. '
                         'Try: linear, nearest, cubic, natural_neighbor, '
                         'optimal', 'barnes, cressman, rbf')

    if grid_xy == None:
        return (grid_x, grid_y), img
    else:
        if comp_err:
            return img, spect_err
        else:
            return img
