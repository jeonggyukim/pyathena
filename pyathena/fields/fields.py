from functools import wraps

import matplotlib as mpl
import numpy as np
import xarray as xr
import astropy.constants as ac
import astropy.units as au

from ..plt_tools.cmap_shift import cmap_shift

def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate

def set_derived_fields(par, x0):
    """
    Function to define derived fields info, for example,
    functions to calculate derived fields, dependency, label, colormap, etc.

    Parameters
    ----------
    par: dict
       Dictionary containing simulation parameter information
    x0: sequence of floats
       Coordinate of the center with respect to which distance is measured

    Returns
    -------
    Tuple of dictionaries containing derived fields info
    """
    
    func = dict()
    field_dep = dict()
    label = dict()
    cmap = dict()

    # rho [g cm^-3]
    f = 'rho'
    field_dep[f] = ['density']
    def _rho(d, u):
        return d['density']*(u.muH*u.mH).cgs.value
    func[f] = _rho
    label[f] = r'$\rho\;[{\rm g\,cm^{-3}}]$'
    cmap[f] = 'Spectral_r'
    
    # nH [cm^-3] (assume d=nH)
    f = 'nH'
    field_dep[f] = ['density']
    def _nH(d, u):
        return d['density']
    func[f] = _nH
    label[f] = r'$n_{\rm H}\;[{\rm cm^{-3}}]$'
    cmap[f] = 'Spectral_r'

    # nH2 [cm^-3] (assume d=nH)
    f = 'nH2'
    field_dep[f] = set(['density', 'xH2'])
    def _nH2(d, u):
        return d['density']*d['xH2']
    func[f] = _nH2
    label[f] = r'$n_{\rm H_2}\;[{\rm cm^{-3}}]$'
    cmap[f] = 'Spectral_r'

    # xH2 [cm^-3] (assume d=nH)
    f = 'xH2'
    field_dep[f] = set(['xH2'])
    def _xH2(d, u):
        return d['xH2']
    func[f] = _xH2
    label[f] = r'$x_{\rm H_2}$'
    cmap[f] = 'viridis'

    # P/kB [K cm^-3]
    f = 'pok'
    field_dep[f] = set(['pressure'])
    def _pok(d, u):
        return d['pressure']*(u.energy_density/ac.k_B).cgs.value
    func[f] = _pok
    label[f] = r'$P/k_{\rm B}\;[{\rm cm^{-3}\,K}]$'
    cmap[f] = 'inferno'

    # distance from x0 [pc]
    f = 'r'
    field_dep[f] = set(['density'])
    @static_vars(x0=x0)
    def _r(d, u):
        z, y, x = np.meshgrid(d['z'], d['y'], d['x'], indexing='ij')
        return xr.DataArray(np.sqrt((x - _r.x0[0])**2 + (y - _r.x0[1])**2 + (z - _r.x0[2])**2),
                            dims=('z','y','x'), name='r')
    func[f] = _r
    label[f] = r'$r\;{\rm pc}$'
    cmap[f] = 'viridis'

    # radial velocity w.r.t. x0 [km/s]
    f = 'vr'
    field_dep[f] = set(['velocity'])
    @static_vars(x0=x0)
    def _vr(d, u):
        z, y, x = np.meshgrid(d['z'], d['y'], d['x'], indexing='ij')
        r = xr.DataArray(np.sqrt((x - _r.x0[0])**2 + (y - _r.x0[1])**2 + (z - _r.x0[2])**2),
                            dims=('z','y','x'), name='r')
        return (x*d['velocity1'] + y*d['velocity2'] + z*d['velocity3'])/r
    func[f] = _vr
    label[f] = r'$v_r\;[{\rm km\,s^{-1}}]$'
    cmap[f] = 'RdBu'

    # vx [km/s]
    f = 'vz'
    field_dep[f] = set(['velocity'])
    def _vx(d, u):
        return d['velocity1']*u.kms
    func[f] = _vx
    label[f] = r'$v_x\;[{\rm km\,s^{-1}}]$'
    cmap[f] = 'RdBu'

    # vy [km/s]
    f = 'vy'
    field_dep[f] = set(['velocity'])
    def _vy(d, u):
        return d['velocity2']*u.kms
    func[f] = _vy
    label[f] = r'$v_y\;[{\rm km\,s^{-1}}]$'
    cmap[f] = 'RdBu'

    # vz [km/s]
    f = 'vz'
    field_dep[f] = set(['velocity'])
    def _vz(d, u):
        return d['velocity3']*u.kms
    func[f] = _vz
    label[f] = r'$v_z\;[{\rm km\,s^{-1}}]$'
    cmap[f] = 'RdBu'
    
    # T [K]
    f = 'T'
    field_dep[f] = set(['temperature'])
    def _T(d, u):
        return d['temperature']
    func[f] = _T
    label[f] = r'$T\;[{\rm K}]$'
    cmap[f] = cmap_shift(mpl.cm.RdYlBu_r, midpoint=3./7.)

    # heat_ratio (G0)
    f = 'heat_ratio'
    field_dep[f] = set(['heat_ratio'])
    def _heat_ratio(d, u):
        return d['heat_ratio']
    func[f] = _heat_ratio
    label[f] = r'heat_ratio'
    cmap[f] = 'viridis'
    
    return func, field_dep, label, cmap

class DerivedFields(object):

    def __init__(self, par, x0=np.array([0.0,0.0,0.0])):

        # Create a dictionary containing all information about derived fields
        self.dfi = dict()
        self.func, self.field_dep, \
            self.label, self.cmap = set_derived_fields(par, x0)
        self.derived_field_list = self.func
        
        for f in self.derived_field_list:
            self.dfi[f] = dict(field_dep=self.field_dep[f],
                               func=self.func[f],
                               label=self.label[f],
                               cmap=self.cmap[f])
