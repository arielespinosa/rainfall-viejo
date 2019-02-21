# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 13:48:47 2017

@author: yanm
"""

import numpy as np
import numpy.ma as ma
from scipy.integrate import simps
#from metpy.calc import tools as mpt

def ann_model(datain):
    #
    # Feedfoward Neural Network (2 hidden neurons)
    #
  
    # Input 1
    x_x = np.array( [199.739, 0.0, 57.9] )
    x_g = np.array( [0.018528812,2.659574468,0.002118419] )
    x_y = -1.0
    
    # Layer 1
    b1 = np.array( [-0.080779711,-0.134816909] )
   
    w1 = np.array([[-0.902012104 ,
                     0.234481949 ,
                     0.709999947],
                   [-1.027753297 ,
                     0.203303267 ,
                     0.623903401] ])
    
      # Layer 2
    b2 = -0.403873644
    w2 = np.array( [4.415464159,-5.061089079] )
      
    # Output 1
    y_g = 0.022869991
    y_x = 212.142471047
    
    # ===== SIMULATION ========
    n = datain.shape[1]
    # Input 1
    xp = mapminmax_diapply( datain, x_g, x_x, x_y )
    
    # Layer 1
    out = tansig_apply( repmat(b1,n) + np.matmul(w1, xp) )
    
    # Layer 2
    out = b2 + np.matmul(w2, out)
    
    return mapminmax_reverse( out, y_g, y_x, x_y )


def tansig_apply(n):
    # Sigmoid Symmetric Transfer Function
    return 2.0 / ( 1.0 + np.exp(-2.0 * n) ) - 1.0

def mapminmax_diapply( x, set_g, set_x, set_y ):
    # Map Minimum and Maximum Input Processing Function ...
    x = (x.T - set_x) * set_g + set_y
    return x.T

def mapminmax_reverse( y, set_g, set_x, set_y ):
    # Map Minimum and Maximum Output Reverse-Processing Function ...
    y = (y.T - set_y) / set_g + set_x
    return y.T

def repmat(a, ncopies, order='c'):
    
    if ncopies < 1: return np.array([])
    
    new_l = [ (a) for _ in range(ncopies) ]
    
    if order == 'c':
        arr = np.column_stack(new_l)
    else:
        arr = np.row_stack(new_l)
        
    return arr.squeeze()

def almost_equal(a, b, c=0, p=7):
    '''
     Function to compare small number's equalty
     c =: max difference alowed between a and b (c=0 default for equalty)
     p =: precision is p places after floating point.
    '''
    return round(abs(a-b), p) <= c

def ezip(a, b):
    #
    # name: ezip (extended zip)
    #
    # description:
    # ( Return a list of tuples, where each tuple contains the i-th element
    #   from each of the sequences. The returned list is extended
    #   in length to the length of the largest argument sequence.
    #   this is the opposite of function zip, arguments could be scalars )
    #
    if np.iterable(a):
        ac = np.array(a)
    else:
        ac = np.array([a])

    if np.iterable(b): 
        bc = np.array(b)
    else:
        bc = np.array([b])

    
    la, lb = len( ac ), len( bc )
        
    if la > lb:
        bc = repmat(bc, la)
    elif la < lb:
        ac = repmat(ac, lb)
            
    return zip(ac,bc)
    

def intsec(a,b,s=0,e=-1):

    s, e = max(1, s), min(len(a), e)
    
    if np.isscalar(b):
        diff = abs( a[s:e] - b )
    else:
        diff = abs( a[s:e] - b[s:e] )

    return s + list(diff).index( diff.min() )
    
def integ(x,y):
    
    not_nan = ~np.isnan(y)
    return simps(y[not_nan], x[not_nan])
    
def lin_adj(Hr,p,adj):

    hr = 0.01 * Hr
    if adj:
        
        i_p85 = intsec(p,850.,0,-1)
        hro =0.51
        
        mh = (hr[i_p85] - hro) / (p[i_p85] - p[0])
        hr[:i_p85] = hro + mh*( p[:i_p85] - p[0] )
    
    return np.maximum(1.0e-5, hr)

def newton_adap(f, xo, imax=100, tol=1.0e-12):

    dx = 1.0e-1
    eps= 1.0e-30
    while imax:

        ermax = np.nanmax( np.abs(dx) )
        if ermax <= tol: break
    
        # Central diff deriv
        dx = - dx * f (xo) / ( f( xo + dx / 2 ) - f( xo - dx / 2 ) + eps)

        xo += dx
        print(100-imax)
        print(ermax)
        imax -= 1


    return xo

def trid_thomas0(a,b,c,d):
    #    
    # COEFICIENTES MATRICIALES ...
    #
    # a  (diagonal debajo de la principal)
    # b  (diagonal principal             )
    # c  (diagonal encima de la principal)
    # d  ... (miembro derecho del sistema de ecuaciones)

    bk = b.copy()
    dk = d.copy()
    
    nn = d.size
    # foward substitution ...
    for k in range( nn-1 ):
    
        r = a[k] / bk[k]
        
        bk[k+1] -= r *  c[k]
        dk[k+1] -= r * dk[k]

    # backward substitution...
    dk /= bk
    vk = dk.copy()
    for k in range( nn-2, -1, -1):
        vk[k] -= c[k] * vk[k+1] / bk[k]
        
    return vk

def trid_thomas1(a,b,c,d):
    #    
    # COEFICIENTES MATRICIALES ...
    #
    # a  (diagonal debajo de la principal)
    # b  (diagonal principal             )
    # c  (diagonal encima de la principal)
    # d  ... (miembro derecho del sistema de ecuaciones)

    nn = d.size
    gam, v = np.zeros((nn)), np.zeros((nn))
    bet = b[0]
    
    if almost_equal(bet, 0.0, p=10): return []
    
    v[0] = d[0] / bet
    # foward substitution ...
    for j in range( 1, nn ):
        
        gam[j] = c[j-1] / bet
        bet = b[j] - a[j-1] * gam[j]
        v[j] = ( d[j] - a[j-1] * v[j-1] ) / bet
    
    # backward substitution...
    for j in range( nn-2, -1, -1):
        v[j] -= gam[j+1] * v[j+1]
        
    return v

def low_pass(var, ipass=1):
    #
    # Right hand of system of equations...

    wt = np.array( [0.50,-0.52] )
    w = 0.5 * wt

    vv = var.copy()
    for i in range(ipass):
        
        for k in range(2):
            
            vf = vv[1:-1] + w[k] * ( vv[2:] - 2.0 * vv[1:-1] + vv[:-2] )
            vv[1:-1] = vf
    
    return np.concatenate(([var[0]], vv[1:-1], [var[-1]]))

def calc_derv_ngrid(var, xs, order=2):

    dx = np.diff(xs,axis=0)
    
    stp_ratio = dx[1:] / dx[:-1]
    
    # central differences ...
    adc  = (var[ 2:]-var[1:-1]) * stp_ratio + (var[1:-1]-var[:-2]) / stp_ratio
    adc /= ( dx[1:] + dx[:-1] )
    
    # foward-backward differences for boundaries...
    n = 0
    l = (dx[n] + dx[n+1]) / dx[n]
    adf = ( (var[n+1]-var[n]) * l - (var[n+2]-var[n]) / l ) / dx[n+1]
    
    n = -1
    l = (dx[n] + dx[n-1]) / dx[n]
    adb = ( (var[n-2]-var[n]) / l - (var[n-1]-var[n]) * l ) / dx[n-1]
    
    return np.concatenate( ([adf],adc,[adb]) )

def intercep(a,b,z,d='all',i0=1, i1=-1,item=0):
    #
    if np.isscalar(b):
        b0 = repmat(b, z.size)[i0:i1]
    else:
        b0 = b[i0:i1]

    if np.isscalar(a):
        a0 = repmat(a, z.size)[i0:i1]
    else:
        a0 = a[i0:i1]

    x, y = find_intersections( z[i0:i1], a0[i0:i1], b0[i0:i1], direction=d)
    if len(x) == 0:
        return np.nan, np.nan
    else:
        return x[item], y[item]

def nearest_intersection_idx(a, b):
    """Determine the index of the point just before two lines with common x values.

    Parameters
    ----------
    a : array-like
        1-dimensional array of y-values for line 1
    b : array-like
        1-dimensional array of y-values for line 2

    Returns
    -------
        An array of indexes representing the index of the values
        just before the intersection(s) of the two lines.
    """
    # Difference in the two y-value sets
    difference = a - b

    # Determine the point just before the intersection of the lines
    # Will return multiple points for multiple intersections
    sign_change_idx, = np.nonzero(np.diff(np.sign(difference)))

    return sign_change_idx

def find_intersections(x, a, b, direction='all'):
    """Calculate the best estimate of intersection.

    Calculates the best estimates of the intersection of two y-value
    data sets that share a common x-value set.

    Parameters
    ----------
    x : array-like
        1-dimensional array of numeric x-values
    a : array-like
        1-dimensional array of y-values for line 1
    b : array-like
        1-dimensional array of y-values for line 2
    direction : string
        specifies direction of crossing. 'all', 'increasing' (a becoming greater than b),
        or 'decreasing' (b becoming greater than a).

    Returns
    -------
        A tuple (x, y) of array-like with the x and y coordinates of the
        intersections of the lines.
    """
    # Find the index of the points just before the intersection(s)
    nearest_idx = nearest_intersection_idx(a, b)
    next_idx = nearest_idx + 1

    # Determine the sign of the change
    sign_change = np.sign(a[next_idx] - b[next_idx])

    # x-values around each intersection
    _, x0 = _next_non_masked_element(x, nearest_idx)
    _, x1 = _next_non_masked_element(x, next_idx)

    # y-values around each intersection for the first line
    _, a0 = _next_non_masked_element(a, nearest_idx)
    _, a1 = _next_non_masked_element(a, next_idx)

    # y-values around each intersection for the second line
    _, b0 = _next_non_masked_element(b, nearest_idx)
    _, b1 = _next_non_masked_element(b, next_idx)

    # Calculate the x-intersection. This comes from finding the equations of the two lines,
    # one through (x0, a0) and (x1, a1) and the other through (x0, b0) and (x1, b1),
    # finding their intersection, and reducing with a bunch of algebra.
    delta_y0 = a0 - b0
    delta_y1 = a1 - b1
    intersect_x = (delta_y1 * x0 - delta_y0 * x1) / (delta_y1 - delta_y0)

    # Calculate the y-intersection of the lines. Just plug the x above into the equation
    # for the line through the a points. One could solve for y like x above, but this
    # causes weirder unit behavior and seems a little less good numerically.
    intersect_y = ((intersect_x - x0) / (x1 - x0)) * (a1 - a0) + a0

    # Make a mask based on the direction of sign change desired
    if direction == 'increasing':
        mask = sign_change > 0
    elif direction == 'decreasing':
        mask = sign_change < 0
    elif direction == 'all':
        return intersect_x, intersect_y
    else:
        raise ValueError('Unknown option for direction: {0}'.format(str(direction)))
    return intersect_x[mask], intersect_y[mask]

def _next_non_masked_element(a, idx):
    """Return the next non masked element of a masked array.

    If an array is masked, return the next non-masked element (if the given index is masked).
    If no other unmasked points are after the given masked point, returns none.

    Parameters
    ----------
    a : array-like
        1-dimensional array of numeric values
    idx : integer
        index of requested element

    Returns
    -------
        Index of next non-masked element and next non-masked element
    """
    try:
        next_idx = idx + a[idx:].mask.argmin()
        if ma.is_masked(a[next_idx]):
            return None, None
        else:
            return next_idx, a[next_idx]
    except (AttributeError, TypeError, IndexError):
        return idx, a[idx]