#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 08:24:50 2017

@author: yanm
"""
import numpy as np
import numpy.core.numeric as _nx
from matplotlib.patches import Polygon, PathPatch
from matplotlib.path import Path
from mpl_toolkits.basemap import Basemap
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy.interpolate as spin


def get_mapa(lon, lat, proj=None, mapa=None, buff=0.01):

    if mapa is None:

        llon=np.min(lon); llat=np.min(lat); urlon=np.max(lon); urlat=np.max(lat)

        mapa = Basemap(llcrnrlon=llon-buff, llcrnrlat=llat-buff,
                       urcrnrlon=urlon+buff, urcrnrlat=urlat+buff,
                       projection=proj['proj'],
                       resolution=proj['res'],
                       rsphere = proj['rsphere'],
                       lat_0=proj['lat_0'], lon_0=proj['lon_0'],
                       lat_1=proj['lat_1'], lat_2=proj['lat_2'])
        return mapa

    else:

        if not isinstance(mapa, Basemap):
            raise ValueError("mapa must be a basemap object")

        if lon.shape != lat.shape:
            lon, lat = np.meshgrid(lon, lat)

        return mapa(lon, lat)

def get_mask(lon, lat, lon_lat):
    '''
    Get slices for extracting subdomain based on lon_lat limits...
    '''
    flat_lon = lon.flatten()
    flat_lat = lat.flatten()

    # compute sorted indeces
    lon_idx = np.argsort(flat_lon)
    lat_idx = np.argsort(flat_lat)

    # find insertion indeces with binary search:
    se_j = np.searchsorted(flat_lon, lon_lat[0], side='left', sorter=lon_idx)
    se_i = np.searchsorted(flat_lat, lon_lat[1], side='left', sorter=lat_idx)

    # get actual indeces of original arrays:
    se_j = [ lon_idx[j] for j in se_j ]
    se_i = [ lat_idx[i] for i in se_i ]

    if lon.shape != lat.shape:
        if lon.ndim == 1 and lat.ndim == 1:
            s_i, e_i = se_i
            s_j, e_j = se_j
        else:
            raise TypeError('lon and lat must be 1d arrays')
    else:
        size = lon.shape[1]
        s_i, e_i = np.divide(se_i, size)
        s_j, e_j = np.mod(se_j, size)

    return [slice(s_i, e_i, None), slice(s_j, e_j, None)]


def subdomain(datalist, mask, axes=[-2,-1]):
    '''
    mask: list of slice objects
    '''
    if len(mask) != len(axes):
        raise ValueError("lengths of mask and axes must match")

    dataout = []
    for data in datalist:

        n = data.ndim
        l = len(axes)
        if n == l:
            expanded_mask = mask
        elif n > l:
            expanded_mask = n * [slice(None)]
            for axis, m in zip(axes, mask):
                axis = _nx.normalize_axis_index(axis, n)
                expanded_mask[axis] = m
        else:
            raise ValueError("data dimensions must have at least the length of axes")

        dataout.append( data[expanded_mask] )

    return dataout


def fill_shp(sph_info, sph_data, pro='', ax=None):
    """
    rellenando poligonos del shapefile
    """
    import numpy as np
    from matplotlib.collections import PatchCollection

    patch = [Polygon(np.array(shape), True) for info, shape in zip(sph_info, sph_data)]

    ax.add_collection(PatchCollection(patch, facecolor='#7f7f7f', alpha=0.3,
                                      edgecolor='0.2', linewidths=0.1, zorder=1))
    #old color set to #adadad

def graphgtmap(x, y, data, mapa, cblabel, titstr, lvs, cmps, asp, stn_dict=None, fig_dpi=120, calpha=1.0):
    '''
    Funcion para plotear mapa 2D

    inputs: todos los argumentos son posicionales ...
    - lon (matriz con las longitudes de la rejilla)
    - lat (matriz con las latitudes de la rejilla)
    - data (matriz de valores a plotear)
    - units (string, unidades de la variable ej: units='(m/s)')
    - cmps (objeto colormaps, ver function: make_cmap('name'))
    - extd (argumento para colorbar: ej, 'min', 'max', 'bouth')
    '''

    params = {'axes.labelsize': 5,
              'text.usetex': False, 'font.size': 5,
              'font.family': 'serif', 'font.weight': 'normal'}

    plt.rcParams.update(params)

    figura = plt.figure(figsize=asp, facecolor='white', dpi=fig_dpi)

    ax = figura.add_subplot(111)

    mapa.set_axes_limits(ax=ax)

    figura.subplots_adjust(wspace=1., hspace=0.05, top=0.94, bottom=0.05, left=0.06, right=0.96)

    ax.set_title(titstr)

    # draw meridians and parallels
    llon=mapa.llcrnrlon; llat=mapa.llcrnrlat
    rlon=mapa.urcrnrlon; rlat=mapa.urcrnrlat

    nny = 6
    nnx = nny * ((rlon - llon) / (rlat - llat))

    dlon = 0.1 * (rlon - llon)
    dlat = 0.1 * (rlat - llat)

    nxrange = np.linspace(llon + dlon, rlon - dlon, nnx)
    nyrange = np.linspace(llat + dlat, rlat - dlat, nny)

    mapa.drawmeridians(nxrange, labels=[1, 0, 0, 1], dashes=[1, 5], fontsize=5, fmt='%2.1f', linewidth=0.2, color='k')
    mapa.drawparallels(nyrange, labels=[1, 0, 0, 1], dashes=[1, 5], fontsize=5, fmt='%2.1f', linewidth=0.2, color='k')
    mapa.drawmapboundary(fill_color='#ffffff')
    mapa.drawcoastlines(linewidth=0.3)

    # plot shapefiles
    # mapa.readshapefile('/home/yanm/git_projects/met_grid/shapefiles/Nueva_DPA',
    #                     'prov', ax=ax, color='k', zorder=1, linewidth=0.3)
    # fill_shp(mapa.prov_info, mapa.prov, ax = ax)

    # create contours ...
    # for ind in np.ndindex(data.shape):

    #     data[ind] = data[ind] if mapa.is_land(x[ind], y[ind]) else np.nan

    cs = mapa.contourf(x, y, data, levels=lvs, cmap=cmps, ax=ax, alpha=calpha)
    cs.cmap.set_under((0.88, 0.88, 0.92))
    cs.cmap.set_over( (0.18, 0.00, 0.18))

    if stn_dict is not None:

        mapa.scatter(stn_dict['obs_x'], stn_dict['obs_y'], marker='o', s=5,
                     edgecolors='k', linewidth=0.5, c=stn_dict['colors'],
                     cmap=cmps, latlon=False)

        for xg, yg in zip( stn_dict['grd_x'], stn_dict['grd_y'] ):

            mapa.scatter(xg, yg, marker='.',color='gray', s=1., zorder=1, latlon=False)

    lvs = lvs[::lvs.size/9]
    ax0 = figura.add_axes([0.12, 0.16, 0.40, 0.016])
    cb = mpl.colorbar.ColorbarBase(ax0, cmap=cmps, spacing='uniform', ticks=lvs,
                                   boundaries=lvs, format='%3.f',
                                   orientation='horizontal', extendrect=True)

    cb.set_label(cblabel, labelpad=-25, fontsize=6)

    return figura

def plot_performance(epochs, history_model):

    fig = plt.figure(figsize=(9, 3), facecolor='white')
    ax = fig.add_subplot(111)

    ax.plot(
        range(epochs),
        history_model.history['loss'],
        'g--',
        label='Network Train Loss')
    ax.plot(
        range(epochs),
        history_model.history['val_loss'],
        'g-',
        label='Network Val Loss')

    ax.set_xlabel('Epochs')
    ax.set_ylabel('Losses')
    plt.legend()

    return fig


def get_terrain(filename, lon_e, lat_e, pool=(1, 1)):

    # LECTURA DE LOS DATOS DE ALTIMETRIA DE CUBA CON 3'' DE RESOLUCION ESPACIAL ...
    cubaTH =  np.load(filename,'r')

    #PARAMETROS DEL FICHERO 'cuba_3s'

    nn = 0.000832639    # paso de rejilla de coordenadas = (1/1201)°

    southcrn =  19.0    #  latitud  inferior del fichero
    northcrn =  24.0    #  latitud  superior del fichero
    eastncrn = -74.0    #  longitud inferior del fichero
    westncrn = -85.0    #  longitud superior del fichero

    isouthcrn =     0   #
    inorthcrn =  6004   # dimensiones del
    jeastncrn = 13211   # fichero original...
    jwestncrn =     0   #

    # GEOREFERENCIANDO LOS DATOS DE ALTIMETRIA ...

    idcrn = np.int((lat_e[0]-southcrn)/nn) - 1
    iucrn = np.int((lat_e[1]-southcrn)/nn) + 1

    jlcrn = np.int((lon_e[0]-westncrn)/nn) - 1
    jrcrn = np.int((lon_e[1]-westncrn)/nn) + 1

    nhy  = iucrn-idcrn + 1
    nhx  = jrcrn-jlcrn + 1

    lathgt = np.linspace( idcrn*nn , iucrn*nn , nhy )
    lonhgt = np.linspace( jlcrn*nn , jrcrn*nn , nhx )

    lathgt = southcrn + lathgt
    lonhgt = westncrn + lonhgt

    # OBTENCION DE LOS DATOS DE ALTIMETRIA DEL DOMINIO ...

    # nota: (si las dimensiones del dominio pedido por el usuario superan los
    #        limites geograficos de los datos de altimetria, se asumira para
    #        estos puntos exteriores altitud 0.)

    hgt = np.zeros((nhy,nhx))

    idc = np.maximum(idcrn,isouthcrn)    #
    iuc = np.minimum(iucrn,inorthcrn)+1  # limites de la matriz de datos
    jlc = np.maximum(jlcrn,jwestncrn)    # a extraer del fichero 'cuba_3s'
    jrc = np.minimum(jrcrn,jeastncrn)+1  #

    idh = 0 ;iuh = nhy
    jlh = 0 ;jrh = nhx

    if lathgt.min() < southcrn : idh = np.abs(idcrn)
    if lathgt.max() > northcrn : iuh = (inorthcrn-idcrn)

    if lonhgt.min() < westncrn : jlh = np.abs(jlcrn)
    if lonhgt.max() > eastncrn : jrh = (jeastncrn-jlcrn)

    hgt[idh:iuh,jlh:jrh] = np.maximum(hgt[idh:iuh,jlh:jrh], cubaTH[idc:iuc,jlc:jrc])

    return lonhgt[::pool[0]], lathgt[::pool[1]], hgt[::pool[0], ::pool[1]]

def julday(cyear, cmonth, cday, ref_date=None):

    cmonth = np.clip(cmonth, 1, 12)
    cday = np.clip(cday, 1, 31)

    days = [
        31, (28 + (1 - min(1, cyear % 4))),
        31, 30, 31, 30, 31, 31, 30, 31, 30, 31
    ]
    cum_days = [
        min(1, max(0, cmonth - m - 1)) * d for m, d in enumerate(days)
    ]

    return cday + np.sum(cum_days)
