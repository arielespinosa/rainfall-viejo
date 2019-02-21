# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 13:12:14 2017

@author: yanm
"""

import numpy as np

eg  = 9.8
er  = 6.3712e6

def g(lat, z=0):

    earth_radius = er
    
    lat, z = list(map(np.asanyarray, (lat, z)))

    # Eqn p27.  UNESCO 1983.
    lat = np.abs(lat)
    sin2 = np.sin( np.deg2rad(lat) ) ** 2
    grav = 9.780318 * (1.0 + (5.2788e-3 + 2.36e-5 * sin2) * sin2)
    return grav / ((1 + z / earth_radius) ** 2)  # From A.E.Gill p.597.

def calc_comp(speed, dd):
    #
    wdir = np.deg2rad(dd)
    return - speed * np.sin( wdir ), - speed * np.cos( wdir )

def calc_derv_ugrid(var, ds, order=6):
    
    if order not in [2,4,6]: return []
    
    ads = np.zeros_like(var)
    flag56 = False
    
    def moment(var, n, m):
        
        if m-n==0:
            return var[n+m:] - var[:-n-m]
        else:
            return var[n+m:-n+m] - var[n-m:-n-m]
    
    # All centered formulas (2nd 4th and 6th orderers)...
    #
    # Compute fluxes ...
    if   order == 2:
        
        ads[1:-1] = moment(var, 1, 1) / 2.0
    
    elif order == 4:
        
        ads[2:-2] = ( 8.0 * moment(var, 2, 1) - moment(var, 2, 2) ) / 12.0
    
    elif order == 6:        
        flag56 = True

        ads[3:-3] = ( 45.0 * moment(var, 3, 1) - 9.0 * moment(var, 3, 2) + moment(var, 3, 3) ) / 60.0
    
    # boundary fluxes: one(two) grid-point(s) into the grid from the boundary ...
    if order != 2:

        ads[ 1] = ( var[ 2] - var[ 0] ) / 2.0
        ads[-2] = ( var[-1] - var[-3] ) / 2.0 
    
    if flag56:  # default: 3rd/4th orderer

        m1 = moment(var, 2, 1)
        m2 = moment(var, 2, 2)
        
        ads[ 2] = ( 8.0 * m1[ 0] - m2[ 0] ) / 12.0
        ads[-3] = ( 8.0 * m1[-1] - m2[-1] ) / 12.0

    # BOUNDARIES ...
    ads[ 0] = (-3.0 * var[ 0] + 4.0 * var[ 1] - var[ 2] ) / 2.0
    ads[-1] = ( 3.0 * var[-1] - 4.0 * var[-2] + var[-3] ) / 2.0
    
    return ads / ds