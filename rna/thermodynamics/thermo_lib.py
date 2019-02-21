# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 13:12:14 2017

@author: yanm
"""

import numpy as np
import numrc_lib as nl
import scipy.integrate as sint
import scipy.optimize as so
from collections import Counter as cnt

# define constants:

rg  = 8.3145
mw  = 18.01528
md  = 28.97

rd  = 1000.0 * rg / md
rv  = 1000.0 * rg / mw
cp  = 1004.832
cv  = cp - rd
kp  = rd / cp

t0  = 273.159
p0  = 1000.0

eo  = 6.105
eps = rd / rv
ee  = eps * eo
fct = (1.0 - eps) / eps
lo  = 2500780.0
g   = 9.7882953113149824 #(La Habana)
Ric = 0.55

class Error_Message(Exception): pass

def temp2kelvin(t):

    return t + t0

def temp2celcius(t):

    return (t - t0).clip(min=-t0)

def calc_rho(t,p):

    return 100. * p / (rd * t)

def calc_w_from_e(e, p):
    return eps * e / ( p - e )

def calc_e_from_w(w, p):
    return p * w / (eps + w)

def calc_q_from_w(w):
    return w / (1.0 + w)

def calc_w_from_q(q):
    return q / (1.0 - q)

def calc_w_from_t(t, tv):

    t_ratio = tv / t
    return (1.0 - t_ratio) / (t_ratio - rv/rd)
def calc_f_from_q(t,q,p):

    es = calc_es(t, p, 'fbuck')
    return calc_w_from_q(q) / calc_w_from_e(es, p)

def calc_eo(p):

    satfwa = 1.0007
    satfwb = 3.46e-6
    satewa = 6.1121

    return ( satfwa + satfwb*p ) * satewa

def calc_es(t,p,opt):

    tc = t-t0
    if opt == 'teten':

        ind = tc >= 0.0
        aw = np.where( ind, 17.270, 21.875)
        bw = np.where( ind, 35.500,  7.500)

        es = eo * np.exp ( aw * tc / (t-bw) )

    elif opt == 'polyg':
        coeffs = [ 6.38780966e-11, 2.03886313e-08,
                   3.02246994e-06, 0.000265027242,
                   0.014305330100, 0.443986062000, eo]

        es = np.poly1d(coeffs)(tc)
    elif opt == 'fbuck':

        satewb = 17.502
        satewc = 32.18

        es = calc_eo(p) * np.exp(satewb*(tc)/(t-satewc))

    return es

def calc_exner(p):

    # Calcular la funcion de Exner
    # constants variations...

    return ( p / p0 ) ** kp

def calc_t(tp, p):
    #
    # Calculate temperature from potential temperature
    #
    return tp * calc_exner(p)

def calc_tp(t, p):
    #
    # Calculate potential temperature from temperature
    #
    return t / calc_exner(p)

def calc_moist(t, p, f):

    es = calc_es(t, p,'fbuck')
    wv = f * calc_w_from_e(es, p)

    return (1.0 + wv / eps) / ( 1.0 + wv )

def calc_tv_from_f(t, p, f):
    #
    return t * calc_moist(t, p, f)

def calc_tv_from_w(t, w):

    return t * (1.0 + w / eps) / ( 1.0 + w )

def calc_lh(t):
    '''calculate latent heat of vaporization'''
    return lo * (t0 / t) ** ( 0.167 + 0.000367 * t )

def calc_td_from_e( e, p ):
    #
    # Funcion para el calculo de la temperatura del punto de rocio ...
    #
    satewb = 17.502
    satewc = 32.18

    e1 = np.where(e==0.0, np.NaN,e)

    num = np.log(e1 / calc_eo(p) ) / satewb

    # Cuando e=0.0, (no hay humedad); se indefine el concepto de
    # Temperatura del punto de rocio: Td = -inf
    # En este caso se utiliza Td = NaN ....

    return (satewc * num - t0) / ( num - 1.0 )

def calc_td_from_f( t, p, f ):
    #
    # Funcion para el calculo de la temperatura del punto de rocio ...
    #
    ws = calc_w_from_e(calc_es(t, p,'fbuck'), p)
    e = calc_e_from_w(f * ws, p)

    return calc_td_from_e( e, p )

def calc_trm(t_sfc, p, f_sfc):

    p_sfc = p[0]

    ws = calc_w_from_e( calc_es(t_sfc, p_sfc,'fbuck'), p_sfc )
    wv = f_sfc * ws
    e = calc_e_from_w(wv, p_sfc)

    esd = ( wv / eps ) * (p - e)

    return calc_td_from_e( esd, p )

def calc_tpe(t, p):

    lv = calc_lh(t)
    ws = calc_w_from_e( calc_es(t, p,'fbuck'), p)

    tp = calc_tp( t, p)

    return tp * np.exp( (lv * ws) / (cp * t) )

def calc_tw(t,p,f,method='iterat'):

    if   method == 'annfit':

        datain = np.array( [t, f, p] )
        tw = nl.ann_model( datain )

    elif method == 'iterat':

        pnca, tnca = lcl_iteratively(t, p, f)
        tw = root_find(t,calc_tpe(tnca, pnca), p, 1e-7, 100)

    return np.where(tw>t,t,tw)

def root_find(m,tpe,p,tol,maxi):
    #
    #
    #
    def gc(t):

        # mixing ratio variations...
        ws = calc_w_from_e( calc_es(t, p, 'fbuck'), p)
        # latent heat variations...
        lw = calc_lh( t )
        # calc wet potential temp ...
        tpw = tpe * calc_exner(p)
        # function ...
        return t - tpw * np.exp(-(lw * ws) / (cp * t) )

    return so.fsolve(gc,m,xtol=tol)

def calc_moist_lapse_it( t, p, f_sfc,tol=1e-7,imax=100):

    lcl_tmp = t[0]
    lcl_pre = p[0]

    return root_find(t, calc_tpe(lcl_tmp, lcl_pre), p, tol, imax)


def calc_moist_lapse_ni(tmp, pre):

    def dtdp(t, p):

        lv = calc_lh(t)
        lvs = lv * calc_w_from_e( calc_es(t, p, 'fbuck'), p )
        rdt = rd * t

        frac = (rdt + lvs) / (cp + (lv * lvs * eps) / (rdt * t) )

        return frac / p

    return sint.odeint( dtdp, tmp, pre).squeeze()

def lcl_iteratively(t, p, f, maxi=100, tol=1e-12):


    td = calc_td_from_f( t, p, f )

    kd = 1.0 / kp
    atmp = kd * rv / calc_lh(t)

    def gc (tmp):
        return atmp * np.log( tmp / t ) + 1.0 / tmp - 1.0 / td

    t_nca = so.fsolve(gc,td,xtol=tol)

    return p * ( t_nca / t ) ** kd, t_nca

def calc_lcl(t_sfc, p, f_sfc, method='normand'):

    p_sfc = p[0]

    if method == 'normand':
        tdry = calc_t( calc_tp(t_sfc, p_sfc), p)
        tirm = calc_trm(t_sfc, p, f_sfc)

        return nl.intercep(tirm, tdry, p, i0=1, item=0)

    elif method == 'iterate':

        lcl_pre, lcl_tmp = lcl_iteratively( t_sfc, p_sfc, f_sfc)

        return lcl_pre[0], lcl_tmp[0]

def calc_ccl(t, p, f_sfc):
    #
    # Find Convertive Level for heated parcel
    #
    t_sfc = t[0]
    return nl.intercep(calc_trm(t_sfc, p, f_sfc), t, p, d='all')

def calc_lfc(t, parcel, p):
    #
    # Find Level of free conversion
    #
    return nl.intercep(t,parcel,p, d='all', i0=1, item=0)

def calc_eql(t,parcel,p):
    #
    # Find Level of Equilibrium
    #
    return nl.intercep(t, parcel, p, item=-1)

def calc_hail_prob(t, p, f, z, mask):

    sze = p[0].size
    shp = p.shape

    temp = t.reshape( (shp[0], sze) ).T
    geop = z.reshape( (shp[0], sze) ).T
    pres = p.reshape( (shp[0], sze) ).T
    relh = f.reshape( (shp[0], sze) ).T

    i = 0
    prob = np.zeros(sze)
    for tm, gp, pr, rh in zip(temp, geop, pres, relh):

        wb = calc_tw(tm, pr, rh, method='annfit')

        wbz = nl.intercep(wb, 273.159, gp)[0]
        hgr = -1.67e-4 * wbz + 1.0737

        k3 = 4.98957134574e-06 * 64.29275512704638
        dr1 = (hgr - 0.0001 * wbz)
        var = np.var(dr1)

        prob[i] = 100. * np.exp( - dr1**2 / k3 )#2.5e-4

        if i%100 == 0: print '%.1f %s' % (100. * i / sze, '%')
        i += 1

    return np.where( mask, prob.reshape(shp[1:]), np.nan ).clip(min=25.)

def calc_profile( t, p, f, method = 'normand', kind='ni'):

    t_sfc = t[0]
    p_sfc = p[0]
    f_sfc = f[0]

    lcl_pre, lcl_tmp = calc_lcl(t_sfc, p, f_sfc, method=method)
    ccl_pre, ccl_tmp = calc_ccl(t, p, f_sfc)

    cl_pre = np.maximum(lcl_pre, ccl_pre)

    pre_low = np.concatenate( ( p[p >= cl_pre], [cl_pre] ) )
    pre_upp = np.concatenate( ( [cl_pre], p[p <  cl_pre] ) )

    if cl_pre == ccl_pre:

        print('Heated Surface-Parcel')
        tdry = calc_t( calc_tp(ccl_tmp, ccl_pre), pre_low )

    else:
        print('Lifted Surface-Parcel')
        tdry = calc_t( calc_tp(t_sfc, p_sfc), pre_low )

    tmp_upp = np.concatenate( ( [tdry[-1]], t[p <= cl_pre] ) )

    if kind == 'it':

        tsat = calc_moist_lapse_it( tmp_upp, pre_upp, f_sfc )

    elif kind == 'ni':

        tsat = calc_moist_lapse_ni( tdry[-1], pre_upp )

    return np.concatenate( (tdry[:-1], tsat[1:]) )

def pbl_height_from_profiles(t,tv,p,rh,z, z0 = 0., z1 = 4000.0):


    # user defined height limit (default = 4000.)...
    ind_lim = np.logical_and( z >= z0, z <= z1)

    # using variables just below zlim:
    zlevls = z[ind_lim]
    pressn = p[ind_lim]
    thetad = t[ind_lim]
    thetav = tv[ind_lim]
    relhum = rh[ind_lim]

    tempsn = calc_t(thetad, pressn)

    # Calculate Specific humidity :
    wvapor = calc_w_from_t(thetad, thetav)
    evapor = calc_e_from_w(wvapor, pressn)
    qvapor = calc_q_from_w( wvapor )

    # Calculate refractivity:
    nrefrc = calc_refr(pressn, tempsn, evapor)

    # Store gradients:
    # dq_dz: vertical gradient of (Qv) specific humidity,
    # dh_dz: vertical gradient of (Hr) relative humidity,
    # dt_dz: vertical gradient of (Thetav) virtual potential temperature,
    # dn_dz: vertical gradient of (Nr) atmospheric refractivity.

    dt_dz = nl.calc_derv_ngrid(thetav, zlevls)
    dh_dz = nl.calc_derv_ngrid(relhum, zlevls)
    dq_dz = nl.calc_derv_ngrid(qvapor, zlevls)
    dn_dz = nl.calc_derv_ngrid(nrefrc, zlevls)

    # passing a low-pass filter:
    ips = 10
    dt_dz = nl.low_pass(dt_dz, ipass=ips)
    dh_dz = nl.low_pass(dh_dz, ipass=ips)
    dq_dz = nl.low_pass(dq_dz, ipass=ips)
    dn_dz = nl.low_pass(dn_dz, ipass=ips)

    # Taking the first (max-min) gradients values with length: sample
    sample = 20

    # Sorting gradients
    tp_h = zlevls[ np.argsort(dt_dz)[:-sample-1:-1] ]
    rh_h = zlevls[ np.argsort(dh_dz)[:sample] ]
    qv_h = zlevls[ np.argsort(dq_dz)[:sample] ]
    nr_h = zlevls[ np.argsort(dn_dz)[:sample] ]
    #
    heights = tp_h, rh_h, qv_h, nr_h

    pair = cnt( np.concatenate((heights), axis=0) ).most_common()

    repv = pair[0][1]
    if repv < 3:
        raise Error_Message('Not found')
        return []
    else:
        return [pair[0][0]]



def pbl_layer_height(t,p,f,u,v,z,zlim=4000.):

    # user defined height limit (default = 4000.)...
    ind_lim = z < zlim

    # using variables just below zlim:
    zlevls = z[ind_lim]
    ucompn = u[ind_lim]
    vcompn = v[ind_lim]
    tempsn = t[ind_lim]
    pressn = p[ind_lim]
    relhum = f[ind_lim]

    # Calculate virtual potential temperature:
    esatrn = calc_es(tempsn, pressn, 'fbuck')
    wvapor = relhum * calc_w_from_e(esatrn, pressn)

    thetad = calc_tp( tempsn , pressn)
    thetav = calc_tv_from_w(thetad, wvapor)

    # calculate Gradient Richardson number
    RiG = nl.low_pass( calc_gRi(thetav, ucompn, vcompn, zlevls), ipass = 10)

    Rleng = zlevls[ np.argsort(RiG)[:-10:-1] ]
    depht = Rleng.max() - Rleng.min()

    # Truncating

    h0 = pbl_height_from_profiles(thetad, thetav, pressn, relhum, zlevls,
                                  z0 = Rleng.min(), z1 = Rleng.max() )

    if len(h0) == 0:
        return []
    else:

        # Calculate Cloud base and Cloud top:
        lcl_p, _ = calc_lcl(t[0], p, f[0], method='iterate')

        cloud_base = nl.intercep(p, lcl_p, z)[0]

        if h0 > cloud_base:
            # If h0 is higher than the cloud base, we find the level at wish
            # the first stable layer is located within the cluod.

            # calculate equivalent temperature
            ind_lim = zlevls > cloud_base

            zlevls = zlevls[ ind_lim ]
            pressn = pressn[ ind_lim ]
            tempsn = tempsn[ ind_lim ]

            thetae = calc_tpe(tempsn, pressn)
            dthe_dz = nl.calc_derv_ngrid(thetae, zlevls)
            pbl_height = zlevls[ dthe_dz > 0.0 ][0]

        else:
            pbl_height = h0[0]

    return depht, pbl_height

def pbl_stull(tempsn, pressn, relhum, zlevls):

    # Stull method ...
    parcel = calc_profile(tempsn, pressn, relhum,method='iterate', kind='it')

    sfc_parcel_virtual = calc_tv_from_f(parcel, pressn, relhum)
    sfc_soundg_virtual = calc_tv_from_f(tempsn, pressn, relhum)

    return nl.intercep( sfc_soundg_virtual, sfc_parcel_virtual, zlevls)

def calc_refr(t, p, e):
    return 77.6 * (p / t) + 3.73e5 * ( e / t**2 )

def calc_gRi(t, u, v, z):

    # vertical gradients ...
    dt_dz = nl.calc_derv_ngrid(t, z)
    du_dz = nl.calc_derv_ngrid(u, z)
    dv_dz = nl.calc_derv_ngrid(v, z)

    N2 = g * ( dt_dz / t + g / cp)
    shear = du_dz * du_dz + dv_dz * dv_dz

    return N2 / shear

def calc_cape(p,parcel,airtmp,wettmp,f):

    # find intersections between t_ap and t_a
    sfc_pre = p[0]
    lfc_pre, _ = calc_lfc(airtmp, parcel, p)
    eql_pre, _ = calc_eql(airtmp, parcel, p)

    prep = [ np.logical_and(p > eql_pre, p < lfc_pre) ]
    pren = [ np.logical_and(p > lfc_pre, p < sfc_pre) ]
    prel = [ np.logical_and(p > eql_pre, p < sfc_pre) ]

    # converting to virtual temperatures ...
    vparcel = calc_tv_from_f(parcel, p, f)
    vairtmp = calc_tv_from_f(airtmp, p, f)

    dtv, dtw = (vairtmp - vparcel), (wettmp - vairtmp)

    cape = rd * nl.integ( np.log( p[prep] ), dtv[prep] )
    cin  = rd * nl.integ( np.log( p[pren] ), dtv[pren] )

    dtwm = 0.5 * ( dtw[prel][:-1] + dtw[prel][1:] )

    dape = rd * np.diff( np.log(p[prel]) ) * dtwm

    return np.maximum(0.0,cape), cin, dape
