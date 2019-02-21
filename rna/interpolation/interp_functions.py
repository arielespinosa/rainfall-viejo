#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 12:28:01 2017

@author: yanm
"""
import numpy as np
import logging
import points, polygons, triangles
from scipy.spatial import cKDTree, ConvexHull, Delaunay, qhull
from scipy import matmul, dot
from scipy.linalg import inv, solve
from scipy.interpolate import RegularGridInterpolator as rgi

log = logging.getLogger(__name__)
log.addHandler(logging.StreamHandler())
log.setLevel(logging.WARNING)

def calc_kappa(spacing, kappa_star=5.052):
    r"""Calculate the kappa parameter for barnes interpolation.

    Parameters
    ----------
    spacing: float
        Average spacing between observations
    kappa_star: float
        Non-dimensional response parameter. Default 5.052.

    Returns
    -------
        kappa: float
    """
    return kappa_star * (2.0 * spacing / np.pi)**2

def barnes_weights(sq_dist, kappa, gamma):
    r"""Calculate the Barnes weights from squared distance values.

    Parameters
    ----------
    sq_dist: (N, ) ndarray
        Squared distances from interpolation point
        associated with each observation in meters.
    kappa: float
        Response parameter for barnes interpolation. Default None.
    gamma: float
        Adjustable smoothing parameter for the barnes interpolation. Default None.

    Returns
    -------
    weights: (N, ) ndarray
        Calculated weights for the given observations determined by their distance
        to the interpolation point.
    """
    return np.exp(-1.0 * sq_dist / (kappa * gamma))

def cressman_weights(sq_dist, r):
    r"""Calculate the Cressman weights from squared distance values.

    Parameters
    ----------
    sq_dist: (N, ) ndarray
        Squared distances from interpolation point
        associated with each observation in meters.
    r: float
        Maximum distance an observation can be from an
        interpolation point to be considered in the inter-
        polation calculation.

    Returns
    -------
    weights: (N, ) ndarray
        Calculated weights for the given observations determined by their distance
        to the interpolation point.
    """
    return (r * r - sq_dist) / (r * r + sq_dist)


def grid_search(xp, yp, grid_x, grid_y, r, min_neighbors):
    """
    """
    obs_tree = cKDTree(list(zip(xp, yp)))

    grid_points = points.generate_grid_coords(grid_x, grid_y)

    grid_indexs = obs_tree.query_ball_point(grid_points, r=r)

    # create grid2obs info first (speed up stuff)...
    print 'Working on grid search, this may take a while ...'
    gridl = []; obsnl = []; dists = []; match = []
    for idx, (matches, grid) in enumerate(zip(grid_indexs, grid_points)):
        if len(matches) >= min_neighbors:
            # unnecesary repeated caomputations !!!
            x1, y1 = obs_tree.data[matches].T
            dists.append( triangles.dist_2(grid[0], grid[1], x1, y1) )
            obsnl.append( np.array( zip(y1, x1) ) )
            gridl.append( idx )
            match.append( matches )
    print 'done!'

    if len(match)==0:
        raise ValueError("Could not find at least {} points "
                         "within search radius {}".
                         format(min_neighbors, r))

    return gridl, obsnl, dists, match

def inverse_distance(xp, yp, variable, grid_x, grid_y, r,
                     min_neighbors, gamma=None, kappa=None,
                     num_pass=None, kind='cressman'):

    r"""Generate an inverse distance weighting interpolation of the given points.

    Values are assigned to the given grid based on either [Cressman1959]_ or [Barnes1964]_.
    The Barnes implementation used here based on [Koch1983]_.

    Parameters
    ----------
    grid_search: (5, ) tuple containing
    variable: (N, ) ndarray
        observation values associated with (xp, yp) pairs.
        IE, variable[i] is a unique observation at (xp[i], yp[i]).
    grid_x: (M, 2) ndarray
        Meshgrid associated with x dimension.
    grid_y: (M, 2) ndarray
        Meshgrid associated with y dimension.
    gamma: float
        Adjustable smoothing parameter for the barnes interpolation. Default None.
    kappa: float
        Response parameter for barnes interpolation. Default None.
    min_neighbors: int
        Minimum number of neighbors needed to perform barnes or cressman interpolation
        for a point. Default is 3.
    num_pass: int
        Number of passes of barnes filter. Default is None.
    Returns
    -------
    img: (M, N) ndarray
        Interpolated values on a 2-dimensional grid
    """
    if num_pass is None:
        num_pass = 0

    indexs, obsloc, distances, matches = grid_search(
                        xp, yp, grid_x, grid_y, r, min_neighbors)

    img = np.empty(shape=(grid_x.size), dtype=variable.dtype)
    img.fill(np.nan)

    # first guess
    for idx, dist, match in zip(indexs, distances, matches):

        if kind == 'cressman':
            img[idx] = cressman_point(dist, variable[match], r)
        elif kind == 'barnes':
            img[idx] = barnes_point(dist, variable[match], kappa)

    # multiple passes correction
    if kind == 'barnes' and num_pass:

        img_p = img.reshape(grid_x.shape)
        for ipass in xrange(num_pass):
            print 'Correction pass number: ', ipass+1
            # define analized data at observation points
            sim_obs = rgi((grid_y[:,0], grid_x[0]), img_p, method='linear')

            for idx, obs, dist, match in zip(indexs, obsloc, distances, matches):
                img[idx] += barnes_point(dist, variable[match] - sim_obs(obs), kappa, gamma)

    return img.reshape(grid_x.shape)

def cressman_point(sq_dist, values, radius):
    r"""Generate a Cressman interpolation value for a point.

    The calculated value is based on the given distances and search radius.

    Parameters
    ----------
    sq_dist: (N, ) ndarray
        Squared distance between observations and grid point
    values: (N, ) ndarray
        Observation values in same order as sq_dist
    radius: float
        Maximum distance to search for observations to use for
        interpolation.

    Returns
    -------
    value: float
        Interpolation value for grid point.
    """
    weights = cressman_weights(sq_dist, radius)
    total_weights = np.sum(weights)

    return sum(v * (w / total_weights) for (w, v) in zip(weights, values))


def barnes_point(sq_dist, values, kappa, gamma=None):
    r"""Generate a single pass barnes interpolation value for a point.

    The calculated value is based on the given distances, kappa and gamma values.

    Parameters
    ----------
    sq_dist: (N, ) ndarray
        Squared distance between observations and grid point
    values: (N, ) ndarray
        Observation values in same order as sq_dist
    kappa: float
        Response parameter for barnes interpolation.
    gamma: float
        Adjustable smoothing parameter for the barnes interpolation. Default 1.

    Returns
    -------
    value: float
        Interpolation value for grid point.
    """
    if gamma is None:
        gamma = 1
    weights = barnes_weights(sq_dist, kappa, gamma)
    total_weights = np.sum(weights)

    return sum(v * (w / total_weights) for (w, v) in zip(weights, values))

def natural_neighbor(xp, yp, variable, grid_x, grid_y):
    r"""Generate a natural neighbor interpolation of the given points.

    This assigns values to the given grid using the Liang and Hale [Liang2010]_.
    approach.

    Parameters
    ----------
    xp: (N, ) ndarray
        x-coordinates of observations
    yp: (N, ) ndarray
        y-coordinates of observations
    variable: (N, ) ndarray
        observation values associated with (xp, yp) pairs.
        IE, variable[i] is a unique observation at (xp[i], yp[i])
    grid_x: (M, 2) ndarray
        Meshgrid associated with x dimension
    grid_y: (M, 2) ndarray
        Meshgrid associated with y dimension

    Returns
    -------
    img: (M, N) ndarray
        Interpolated values on a 2-dimensional grid
    """
    tri = Delaunay(list(zip(xp, yp)))

    grid_points = points.generate_grid_coords(grid_x, grid_y)

    members, triangle_info = triangles.find_natural_neighbors(tri, grid_points)

    img = np.empty(shape=(grid_points.shape[0]), dtype=variable.dtype)
    img.fill(np.nan)

    for ind, (grid, neighbors) in enumerate(members.items()):

        if len(neighbors) > 0:

            img[ind] = nn_point(xp, yp, variable, grid_points[grid],
                                tri, neighbors, triangle_info)

    img = img.reshape(grid_x.shape)
    return img


def nn_point(xp, yp, variable, grid_loc, tri, neighbors, triangle_info):
    r"""Generate a natural neighbor interpolation of the observations to the given point.

    This uses the Liang and Hale approach [Liang2010]_. The interpolation will fail if
    the grid point has no natural neighbors.

    Parameters
    ----------
    xp: (N, ) ndarray
        x-coordinates of observations
    yp: (N, ) ndarray
        y-coordinates of observations
    variable: (N, ) ndarray
        observation values associated with (xp, yp) pairs.
        IE, variable[i] is a unique observation at (xp[i], yp[i])
    grid_loc: (float, float)
        Coordinates of the grid point at which to calculate the
        interpolation.
    tri: object
        Delaunay triangulation of the observations.
    neighbors: (N, ) ndarray
        Simplex codes of the grid point's natural neighbors. The codes
        will correspond to codes in the triangulation.
    triangle_info: dictionary
        Pre-calculated triangle attributes for quick look ups. Requires
        items 'cc' (circumcenters) and 'r' (radii) to be associated with
        each simplex code key from the delaunay triangulation.

    Returns
    -------
    value: float
       Interpolated value for the grid location
    """
    edges = triangles.find_local_boundary(tri, neighbors)
    edge_vertices = [segment[0] for segment in polygons.order_edges(edges)]
    num_vertices = len(edge_vertices)

    p1 = edge_vertices[0]
    p2 = edge_vertices[1]

    polygon = list()
    c1 = triangles.circumcenter(grid_loc, tri.points[p1], tri.points[p2])
    polygon.append(c1)

    area_list = []
    total_area = 0.0

    for i in range(num_vertices):

        p3 = edge_vertices[(i + 2) % num_vertices]

        try:

            c2 = triangles.circumcenter(grid_loc, tri.points[p3], tri.points[p2])
            polygon.append(c2)

            for check_tri in neighbors:
                if p2 in tri.simplices[check_tri]:
                    polygon.append(triangle_info[check_tri]['cc'])

            pts = [polygon[i] for i in ConvexHull(polygon).vertices]
            value = variable[(tri.points[p2][0] == xp) & (tri.points[p2][1] == yp)]

            cur_area = polygons.area(pts)

            total_area += cur_area

            area_list.append(cur_area * value[0])

        except (ZeroDivisionError, qhull.QhullError) as e:
            message = ('Error during processing of a grid. '
                       'Interpolation will continue but be mindful '
                       'of errors in output. ') + str(e)

            log.warning(message)
            return np.nan

        polygon = list()
        polygon.append(c2)

        p2 = p3

    return sum(x / total_area for x in area_list)

def optimal_parameters(xp, yp, variable, grid_loc, parameters, comp_err=0):
    r"""Generate an optimal interpolation of 2D observations to the given points.

    This uses Optimal Interpolation Scheme by [Roemmich1983]_.

    Parameters
    ----------
    xp: (N, ) ndarray
        x-coordinates of observations
    yp: (N, ) ndarray
        y-coordinates of observations
    variable: (N, ) ndarray
        observation values associated with (xp, yp, zp) coordinates.
        IE, variable[i] is a unique observation at (xp[i], yp[i], zp[i])
    grid_loc: (float, float)
        Coordinates of the grid point at which the interpolation is performed.

    parameters: (largex, largey, largez, smallx, smally, smallz, largevar, smallvar)
        Large-Small scales correlation parameters.

    Returns
    -------
    value: float
       Interpolated value for the grid location
    """
    # fmodule contains fortran subroutines
    #import fmodule

    n = (variable.size, 1)
    m = (grid_loc.shape[0], 1)

    iparams = 1.0 / (parameters * parameters).clip(min=1e-15)

    xg, yg = grid_loc[:, 0], grid_loc[:, 1]

    # compute square distances between obs-obs points...
    x = np.square( np.tile(xp, n).T - xp ).T
    y = np.square( np.tile(yp, n).T - yp ).T

    # compute square distances between obs-grid points...
    dx = np.square( np.tile(xp, m).T - xg ).T
    dy = np.square( np.tile(yp, m).T - yg ).T

    alasru = np.exp(-np.sqrt(x * iparams[0] + y * iparams[1]))
    asmsru = np.exp(-np.sqrt(x * iparams[2] + y * iparams[3]))

    ala = alasru + np.diag(n[0] * [parameters[4]])
    asm = asmsru + np.diag(n[0] * [parameters[5]])

    expdl = np.exp(-np.sqrt(dx * iparams[0] + dy * iparams[1]))
    expds = np.exp(-np.sqrt(dx * iparams[2] + dy * iparams[3]))

    # compute residuals & weights (using python just for linealg) ...
    alainv = inv(ala, check_finite=False)
    wlarge = solve(ala, variable, assume_a='pos')

    vmean = np.nansum(wlarge) / np.nansum(alainv)
    residl = (variable - vmean) - matmul(alasru, wlarge)
    wsmall = solve(asm, residl, assume_a='pos')

    args = vmean, expds, expdl, wsmall, wlarge

    # Compute expected interpolation error...
    spected_err = None
    if comp_err:
        asminv = inv(asm, check_finite=False)
        expds = matmul(expds, asminv) * expds
        spected_err = 1.0 - np.nansum(expds, axis=1)

    return args, spected_err

def optimal_interp(args):
    r"""Generate an optimal interpolation based on small-large scale values and weights.

    This uses Optimal Interpolation Scheme by [Roemmich1983]_.

    Returns
    -------
    value: float
       Interpolated value for the grid location
    """
    vmean, expds, expdl, wsmall, wlarge = args

    return vmean + dot(expds,wsmall) + dot(expdl, wlarge)
