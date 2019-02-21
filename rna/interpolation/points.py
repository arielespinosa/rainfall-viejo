# Copyright (c) 2008-2016 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Tools for working with points."""

from __future__ import division

import logging

import numpy as np
from scipy.spatial import cKDTree

log = logging.getLogger(__name__)
log.addHandler(logging.StreamHandler())  # Python 2.7 needs a handler set
log.setLevel(logging.WARNING)


def get_points_within_r(center_points, target_points, r):
    r"""Get all target_points within a specified radius of a center point.

    All data must be in same coordinate system, or you will get undetermined results.

    Parameters
    ----------
    center_points: (X, Y) ndarray
        location from which to grab surrounding points within r
    target_points: (X, Y) ndarray
        points from which to return if they are within r of center_points
    r: integer
        search radius around center_points to grab target_points

    Returns
    -------
    matches: (X, Y) ndarray
        A list of points within r distance of, and in the same
        order as, center_points
    """
    tree = cKDTree(target_points)
    indices = tree.query_ball_point(center_points, r)
    return tree.data[indices].T


def get_point_count_within_r(center_points, target_points, r):
    r"""Get count of target points within a specified radius from center points.

    All data must be in same coordinate system, or you will get undetermined results.

    Parameters
    ----------
    center_points: (X, Y) ndarray
        locations from which to grab surrounding points within r
    target_points: (X, Y) ndarray
        points from which to return if they are within r of center_points
    r: integer
        search radius around center_points to grab target_points

    Returns
    -------
    matches: (N, ) ndarray
        A list of point counts within r distance of, and in the same
        order as, center_points
    """
    tree = cKDTree(target_points)
    indices = tree.query_ball_point(center_points, r)
    return np.array([len(x) for x in indices])


def generate_grid(horiz_dims, bbox):
    r"""Generate a meshgrid based on bounding box and x & y resolution.

    Parameters
    ----------
    horiz_dim: integer
        Horizontal resolution in meters
    bbox: dictionary
        Dictionary containing coordinates for corners of study area.

    Returns
    -------
    grid_x: (X, Y) ndarray
        X dimension meshgrid defined by given bounding box
    grid_y: (X, Y) ndarray
        Y dimension meshgrid defined by given bounding box
    """
    if np.min(horiz_dims) < 10000:
        log.warning('Grids less than 10km may be slow to load at synoptic scale.')

    x_steps, y_steps = get_xy_steps(bbox, horiz_dims)

    grid_x = np.linspace(bbox['west'], bbox['east'], x_steps)
    grid_y = np.linspace(bbox['south'], bbox['north'], y_steps)

    return np.meshgrid(grid_x, grid_y)


def generate_grid_coords(gx, gy):
    r"""Calculate x,y coordinates of each grid cell.

    Parameters
    ----------
    gx: numeric
        x coordinates in meshgrid
    gy: numeric
        y coordinates in meshgrid

    Returns
    -------
    (X, Y) ndarray
        List of coordinates in meshgrid
    """
    return np.vstack([gx.ravel(), gy.ravel()]).T


def get_xy_range(bbox):
    r"""Return x and y ranges in meters based on bounding box.

    bbox: dictionary
        dictionary containing coordinates for corners of study area

    Returns
    -------
    x_range: float
        Range in meters in x dimension.
    y_range: float
        Range in meters in y dimension.
    """
    x_range = bbox['east'] - bbox['west']
    y_range = bbox['north'] - bbox['south']

    return x_range, y_range


def get_xy_steps(bbox, h_dims):
    r"""Return meshgrid spacing based on bounding box.

    bbox: dictionary
        Dictionary containing coordinates for corners of study area.
    h_dim: integer
        Horizontal resolution in meters.

    Returns
    -------
    x_steps, (X, ) ndarray
        Number of grids in x dimension.
    y_steps: (Y, ) ndarray
        Number of grids in y dimension.
    """
    x_range, y_range = get_xy_range(bbox)

    x_steps = np.ceil(x_range / h_dims[0])
    y_steps = np.ceil(y_range / h_dims[1])

    return int(x_steps), int(y_steps)


def get_boundary_coords(x, y, spatial_pad=0):
    r"""Return bounding box based on given x and y coordinates assuming northern hemisphere.

    x: numeric
        x coordinates.
    y: numeric
        y coordinates.
    spatial_pad: numeric
        Number of meters to add to the x and y dimensions to reduce
        edge effects.

    Returns
    -------
    bbox: dictionary
        dictionary containing coordinates for corners of study area
    """
    west = np.min(x) - spatial_pad
    east = np.max(x) + spatial_pad
    north = np.max(y) + spatial_pad
    south = np.min(y) - spatial_pad

    return {'west': west, 'south': south, 'east': east, 'north': north}

def triangle_area(pt1, pt2, pt3):
    r"""Return the area of a triangle.

    Parameters
    ----------
    pt1: (X,Y) ndarray
        Starting vertex of a triangle
    pt2: (X,Y) ndarray
        Second vertex of a triangle
    pt3: (X,Y) ndarray
        Ending vertex of a triangle

    Returns
    --------
    area: float
        Area of the given triangle.
    """
    a = 0.0

    a += pt1[0] * pt2[1] - pt2[0] * pt1[1]
    a += pt2[0] * pt3[1] - pt3[0] * pt2[1]
    a += pt3[0] * pt1[1] - pt1[0] * pt3[1]

    return abs(a) / 2


def dist_2(x0, y0, x1, y1):
    r"""Return the squared distance between two points.

    This is faster than calculating distance but should
    only be used with comparable ratios.

    Parameters
    ----------
    x0: float
        Starting x coordinate
    y0: float
        Starting y coordinate
    x1: float
        Ending x coordinate
    y1: float
        Ending y coordinate

    Returns
    --------
    d2: float
        squared distance

    See Also
    --------
    distance
    """
    d0 = x1 - x0
    d1 = y1 - y0
    return d0 * d0 + d1 * d1


def distance(p0, p1):
    r"""Return the distance between two points.

    Parameters
    ----------
    p0: (X,Y) ndarray
        Starting coordinate
    p1: (X,Y) ndarray
        Ending coordinate

    Returns
    --------
    d: float
        distance

    See Also
    --------
    dist_2
    """
    return np.sqrt(dist_2(p0[0], p0[1], p1[0], p1[1]))


def circumcircle_radius_2(pt0, pt1, pt2):
    r"""Calculate and return the squared radius of a given triangle's circumcircle.

    This is faster than calculating radius but should only be used with comparable ratios.

    Parameters
    ----------
    pt0: (x, y)
        Starting vertex of triangle
    pt1: (x, y)
        Second vertex of triangle
    pt2: (x, y)
        Final vertex of a triangle

    Returns
    --------
    r: float
        circumcircle radius

    See Also
    --------
    circumcenter
    """
    a = distance(pt0, pt1)
    b = distance(pt1, pt2)
    c = distance(pt2, pt0)

    t_area = triangle_area(pt0, pt1, pt2)
    prod2 = a * b * c

    if t_area > 0:
        radius = prod2 * prod2 / (16 * t_area * t_area)
    else:
        radius = np.nan

    return radius


def circumcircle_radius(pt0, pt1, pt2):
    r"""Calculate and return the radius of a given triangle's circumcircle.

    Parameters
    ----------
    pt0: (x, y)
        Starting vertex of triangle
    pt1: (x, y)
        Second vertex of triangle
    pt2: (x, y)
        Final vertex of a triangle

    Returns
    --------
    r: float
        circumcircle radius

    See Also
    --------
    circumcenter
    """
    a = distance(pt0, pt1)
    b = distance(pt1, pt2)
    c = distance(pt2, pt0)

    t_area = triangle_area(pt0, pt1, pt2)

    if t_area > 0:
        radius = (a * b * c) / (4 * t_area)
    else:
        radius = np.nan

    return radius


def circumcenter(pt0, pt1, pt2):
    r"""Calculate and return the circumcenter of a circumcircle generated by a given triangle.

    All three points must be unique or a division by zero error will be raised.

    Parameters
    ----------
    pt0: (x, y)
        Starting vertex of triangle
    pt1: (x, y)
        Second vertex of triangle
    pt2: (x, y)
        Final vertex of a triangle

    Returns
    --------
    cc: (x, y)
        circumcenter coordinates

    See Also
    --------
    circumcenter
    """
    a_x = pt0[0]
    a_y = pt0[1]
    b_x = pt1[0]
    b_y = pt1[1]
    c_x = pt2[0]
    c_y = pt2[1]

    bc_y_diff = b_y - c_y
    ca_y_diff = c_y - a_y
    ab_y_diff = a_y - b_y
    cb_x_diff = c_x - b_x
    ac_x_diff = a_x - c_x
    ba_x_diff = b_x - a_x

    d_div = (a_x * bc_y_diff + b_x * ca_y_diff + c_x * ab_y_diff)

    if d_div == 0:
        raise ZeroDivisionError

    d_inv = 0.5 / d_div

    a_mag = a_x * a_x + a_y * a_y
    b_mag = b_x * b_x + b_y * b_y
    c_mag = c_x * c_x + c_y * c_y

    cx = (a_mag * bc_y_diff + b_mag * ca_y_diff + c_mag * ab_y_diff) * d_inv
    cy = (a_mag * cb_x_diff + b_mag * ac_x_diff + c_mag * ba_x_diff) * d_inv

    return cx, cy


def find_natural_neighbors(tri, grid_points):
    r"""Return the natural neighbor triangles for each given grid cell.

    These are determined by the properties of the given delaunay triangulation.
    A triangle is a natural neighbor of a grid cell if that triangles circumcenter
    is within the circumradius of the grid cell center.

    Parameters
    ----------
    tri: Object
        A Delaunay Triangulation.
    grid_points: (X, Y) ndarray
        Locations of grids.

    Returns
    --------
    members: dictionary
        List of simplex codes for natural neighbor
        triangles in 'tri' for each grid cell.
    triangle_info: dictionary
        Circumcenter and radius information for each
        triangle in 'tri'.
    """
    tree = cKDTree(grid_points)

    in_triangulation = tri.find_simplex(tree.data) >= 0

    triangle_info = dict()

    members = {key: [] for key in range(len(tree.data))}

    for i, simplices in enumerate(tri.simplices):

        ps = tri.points[simplices]

        cc = circumcenter(*ps)
        r = circumcircle_radius(*ps)

        triangle_info[i] = {'cc': cc, 'r': r}

        qualifiers = tree.query_ball_point(cc, r)

        for qualifier in qualifiers:
            if in_triangulation[qualifier]:
                members[qualifier].append(i)

    return members, triangle_info


def find_nn_triangles_point(tri, cur_tri, point):
    r"""Return the natural neighbors of a triangle containing a point.

    This is based on the provided Delaunay Triangulation.

    Parameters
    ----------
    tri: Object
        A Delaunay Triangulation
    cur_tri: int
        Simplex code for Delaunay Triangulation lookup of
        a given triangle that contains 'position'.
    point: (x, y)
        Coordinates used to calculate distances to
        simplexes in 'tri'.
    Returns
    --------
    nn: (N, ) array
        List of simplex codes for natural neighbor
        triangles in 'tri'.
    """
    nn = []

    candidates = set(tri.neighbors[cur_tri])

    # find the union of the two sets
    candidates |= set(tri.neighbors[tri.neighbors[cur_tri]].flat)

    # remove instances of the "no neighbors" code
    candidates.discard(-1)

    for neighbor in candidates:

        triangle = tri.points[tri.simplices[neighbor]]
        cur_x, cur_y = circumcenter(triangle[0], triangle[1], triangle[2])
        r = circumcircle_radius_2(triangle[0], triangle[1], triangle[2])

        if dist_2(point[0], point[1], cur_x, cur_y) < r:

            nn.append(neighbor)

    return nn


def find_local_boundary(tri, triangles):
    r"""Find and return the outside edges of a collection of natural neighbor triangles.

    There is no guarantee that this boundary is convex, so ConvexHull is not
    sufficient in some situations.

    Parameters
    ----------
    tri: Object
        A Delaunay Triangulation
    triangles: (N, ) array
        List of natural neighbor triangles.

    Returns
    --------
    edges: (2, N) ndarray
        List of vertex codes that form outer edges of
        a group of natural neighbor triangles.
    """
    edges = []

    for triangle in triangles:

        for i in range(3):

            pt1 = tri.simplices[triangle][i]
            pt2 = tri.simplices[triangle][(i + 1) % 3]

            if (pt1, pt2) in edges:
                edges.remove((pt1, pt2))

            elif (pt2, pt1) in edges:
                edges.remove((pt2, pt1))

            else:
                edges.append((pt1, pt2))

    return edges
