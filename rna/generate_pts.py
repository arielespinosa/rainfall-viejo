# -*- coding: utf-8 -*-

import os
import numpy as np
import utils.maps_utils as mu
from scipy.io import savemat, loadmat
from interpolation import points, polygons, triangles
from interpolation.interp_functions import cKDTree, Delaunay

from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt


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
        raise ValueError("Hey!, could not find at least {} points "
                         "within search radius {}".format(min_neighbors, r))

    return gridl, obsnl, dists, match


# get mapa projection object (SPNOA standard)
# getting coordinate transformations right
proj_dict = {
    'rsphere': 6370997.0,  #6371229,
    'lat_0': 22.0,
    'lon_0': -80.0,
    'lat_1': 15.0,
    'lat_2': 35.0,
    'proj': 'lcc',
    'res': 'h'
}

if __name__ == '__main__':

    domain = 'd02'
    # get stations corrdinates

    # create data points
    stn_id, stn_lon, stn_lat = np.loadtxt('/data/estaciones.dt').T
    coords = loadmat('data/' + '_'.join(['coords', domain]) + '.mat')

    # Create map object
    grd_lon, grd_lat = coords['lon'], coords['lat']

    # cropping the domain a little bit:   western, easter, southern northern points
    mask = mu.get_mask(grd_lon, grd_lat, [[-85.60, -73.20], [ 19.38, 23.60]] )

    # create map (this should be done just once to speed up things)
    grd_lon, grd_lat = mu.subdomain( [grd_lon, grd_lat], mask, axes=[1, 2] )
    mapa = mu.get_mapa(grd_lon, grd_lat, proj=proj_dict, buff=0.1)

    # get physical coordinates in [meters] for computing distances
    (grd_x, grd_y) = mu.get_mapa(grd_lon, grd_lat, mapa=mapa)
    (obs_x, obs_y) = mu.get_mapa(stn_lon, stn_lat, mapa=mapa)

    #====================== testing grid search ===========================
    search_radius = 50902.  # m (this radius is not arbitrary but depends on the variable)
    min_neighbors = 16  # minimun number of grid points allowed around stations

    grid_points = points.generate_grid_coords(grd_x, grd_y)
    obs_points = np.array(zip(obs_x, obs_y))

    indexs, grd_loc, distances, matches = grid_search(
        grid_points, obs_points, search_radius, min_neighbors)

    # search for reduced min_neighbors
    indices = []
    max_neighbors = 1e5
    for distance in distances:
        # old school searching of minimum number of neighbors
        ref_neighbors = len(distance)
        if ref_neighbors < max_neighbors:
            max_neighbors = ref_neighbors
        # store all indeces
        indices.append(np.argsort(distance))

    # create dictionary for grid indices around stations for later recall
    # and its coordinates:
    stn_dict = {}
    grd_dict = {}
    for stn, index, grid in zip(stn_id.astype('int32'), indices, grd_loc):

        stn = str(stn)
        ind = index[:max_neighbors]

        print('Processing station: {}, with {} grid points'.format(
            stn, len(ind)))

        stn_dict[stn] = ind
        grd_dict[stn] = np.stack([grid[ind][:, 1], grid[ind][:, 0]], axis=-1)

    savemat('train_data/grid_points_' + domain, stn_dict)
    savemat('train_data/grid_coords_' + domain, grd_dict)

    #=================================================================
    figure = plt.figure(figsize=(8., 3.2), dpi=300)

    mapa.drawcoastlines(linewidth=0.3)
    mapa.scatter(obs_x, obs_y, marker='o', s=5, edgecolors='k',
                 linewidth=0.5, c='r', latlon=False)

    for stn, points in grd_dict.iteritems():

        x, y = points.T
        mapa.scatter(x, y, marker='.', s=1., c='gray', zorder=1, latlon=False)

    figure.savefig('outputs/plots/grid_points_'+ domain+'.png')
    mu.plt.close(figure)

    #======================================================================
