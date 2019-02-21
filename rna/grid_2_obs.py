#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import numpy as np
import utils.maps_utils as mu
from scipy.io import loadmat, savemat
from datetime import datetime
from netCDF4 import num2date
from interpolation import points, polygons, triangles
from interpolation.interp_functions import cKDTree, Delaunay
from utils.cmaps import make_cmap, rain, rain1, temp
from scipy.interpolate import griddata

def grid_search(obs_points, grid_points, r, min_neighbors):
    """
    """

    obs_tree = cKDTree(obs_points)
    grid_indexs = obs_tree.query_ball_point(grid_points, r=r)

    # create grid2obs info first (speed up stuff)...
    print('Working on grid search, this may take a while ...')
    gridl = []
    obsnl = []
    dists = []
    match = []
    for idx, (matches, grid) in enumerate(zip(grid_indexs, grid_points)):
        if len(matches) >= min_neighbors:
            x1, y1 = obs_tree.data[matches].T
            dists.append(triangles.dist_2(grid[0], grid[1], x1, y1))
            obsnl.append(np.array(zip(y1, x1)))
            gridl.append(idx)
            match.append(matches)
    print('done!')

    if len(match) == 0:
        raise ValueError("Could not find at least {} points "
                         "within search radius {}".format(min_neighbors, r))

    return gridl, obsnl, dists, match


if __name__ == '__main__':

    domain = 'd01'
    var_key = 'temp2m'

    path_io = 'data/train_data/'

    # get actual data
    path = os.path.join(path_io, domain, var_key, 'stn_data.mat')
    data_dict = loadmat(path, squeeze_me=True)
    stn, lon_p, lat_p, obs_data = data_dict['stn_id'], data_dict['lon'], data_dict[
        'lat'], data_dict['values']

    path = os.path.join(path_io, domain, var_key, 'grd_data.mat')
    data_dict = loadmat(path, squeeze_me=True)
    lon_g, lat_g, grd_data = data_dict['lon'], data_dict['lat'], data_dict[
        'values']

    # get subdomain
    lon_lat = [[-85.60, -73.20], [ 19.42, 23.60]]
    mask = mu.get_mask(lon_g, lat_g, lon_lat )

    lon_g, lat_g, grd_data = mu.subdomain( [lon_g, lat_g, grd_data], mask, axes=[1, 2])

    # string representation of dates based on file names
    dates = data_dict['time']

    # get mapa projection object (SPNOA standard)
    proj_dict = {
        'rsphere': 6370997.0, #6371229,
        'lat_0': 22.0,
        'lon_0':-80.0,
        'lat_1': 15.0,
        'lat_2': 35.0,
        'proj': 'lcc',
        'res': 'h'
    }
    mapa = mu.get_mapa(lon_lat[0], lon_lat[1], proj=proj_dict)

    (grd_x, grd_y) = mu.get_mapa(lon_g, lat_g, mapa=mapa)
    (obs_x, obs_y) = mu.get_mapa(lon_p, lat_p, mapa=mapa)

    # transform to kilometers
    print(obs_x.min(), grd_x.min())
    print(obs_x.max(), grd_x.max())

    #====================== testing =======================================
    search_radius = 50902.  # m
    sqr_neighbors = (4, 4)  # number of grid points around observation
    min_neighbors = np.prod(sqr_neighbors)

    grid_points = points.generate_grid_coords(grd_x, grd_y)
    obs_points = np.array(zip(obs_x, obs_y))

    indexs, grd_loc, distances, matches = grid_search(
        grid_points, obs_points, search_radius, min_neighbors)

    obs_dict = {
        'obs_x': obs_x,
        'obs_y': obs_y,
        'grd_x': [],
        'grd_y': [],
    }

        # get grid points near to observations
        ind = np.sort( np.argsort(dist)[:min_neighbors] )

    x_values = {}
    y_values = {}
    z_values = {}
    grd_shape = grd_data.shape
    for stn, x, y, z, grid, dist in zip(stn, obs_x, obs_y, obs_data.T, grd_loc, distances):

        stn_id = 'stn_' + str(stn)

        # get grid points near to observations
        ind = np.sort( np.argsort(dist)[:min_neighbors] )

        # print idx, ind, match, grid[ind].shape
        obs_dict['grd_x'].append(grid[ind][:, 1])
        obs_dict['grd_y'].append(grid[ind][:, 0])

        values = grd_data.reshape((grd_shape[0], -1, grd_shape[-1]))[:,ind]

        new_shape = np.shape(values)

        value = values.transpose( [0, 2, 1] ).reshape(-1, new_shape[1])

        value = [ griddata(grid[ind], grd, (x, y), method='nearest') for grd in value]
        value = np.reshape(value, new_shape[::2])

        print( 'predicted mae: ', abs(value.mean(axis=0) - z.mean()) )

        x_values[stn_id] = values
        y_values[stn_id] = np.stack([dates, z], axis=-1)
        z_values[stn_id] = np.concatenate([dates.reshape([-1, 1]), z.reshape([-1, 1]), value], axis=1)

        print('Processing station: ', stn, ind, x_values[stn_id].shape, y_values[stn_id].shape)

    savemat(path_io + '_'.join([var_key, 'x', 'samples']), x_values)
    savemat(path_io + '_'.join([var_key, 'y', 'samples']), y_values)
    savemat(path_io + '_'.join([var_key, 'z', 'samples']), z_values)
    #======================================================================
    # plotting stuff

    asp = (8.6, 3.2)
    cblabel = 'temperature 2 m [C]' #'3h acumm precipitation [mm]' #
    time_units = 'seconds since 2010-01-01 00:00:00'

    # map data
    members = 3
    treshhold = 0.01   # mm / h
    # grd_data = np.where(grd_data < treshhold, np.nan, grd_data)

    clevs, cmaps = make_cmap(temp()) #[0], precip1()


    for date, grid, obs in zip(dates, grd_data, obs_data):

        obs_dict['colors'] = obs

        title = num2date(date, time_units)
        print('Plotting date: {}'.format(title) )

        print('--> observed:  minimum value: {:.2f}, maximum value: {:.2f}'.format(obs.min(), obs.max()))
        for m in range(members):

            grid_m = grid[...,m]
            print('--> member: {}, minimum value: {:.2f}, maximum value: {:.2f}'.format(m+1, grid_m.min(), grid_m.max()))
            mapa = mu.get_mapa(lon_lat[0], lon_lat[1], proj=proj_dict)

            fig = mu.graphgtmap(grd_x, grd_y, grid_m, mapa, cblabel, title,
                                clevs, cmaps, asp,  calpha=0.9, stn_dict=obs_dict,
                                fig_dpi=300)

            fname = 'map_' + str(title).replace(' ', '_') + '_member:'+str(m+1)
            fig.savefig(os.path.join('outputs/plots', domain, var_key, fname + '.png'))
            mu.plt.close(fig)
