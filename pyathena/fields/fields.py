from functools import wraps

import matplotlib as mpl
import numpy as np
import xarray as xr
import astropy.constants as ac
import astropy.units as au

from matplotlib.colors import Normalize, LogNorm

from ..plt_tools.cmap_shift import cmap_shift
from ..plt_tools.cmap_custom import get_my_cmap

from .xray_emissivity import get_xray_emissivity

def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate

def set_derived_fields_def(par, x0):
    """
    Function to define derived fields info, for example,
    functions to calculate derived fields, dependency, label, colormap, etc.

    May not work perfectly for problems using different unit system.
    Assume that density = nH, length unit = pc, etc.

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
    vminmax = dict()
    take_log = dict()
    
    # rho [g cm^-3]
    f = 'rho'
    field_dep[f] = ['density']
    def _rho(d, u):
        return d['density']*(u.muH*u.mH).cgs.value
    func[f] = _rho
    label[f] = r'$\rho\;[{\rm g\,cm^{-3}}]$'
    cmap[f] = 'Spectral_r'
    vminmax[f] = (1e-27,1e-20)
    take_log[f] = True
    
    # nH [cm^-3] (assume d=nH)
    f = 'nH'
    field_dep[f] = ['density']
    def _nH(d, u):
        return d['density']
    func[f] = _nH
    label[f] = r'$n_{\rm H}\;[{\rm cm^{-3}}]$'
    cmap[f] = 'Spectral_r'
    vminmax[f] = (1e-3,1e4)
    take_log[f] = True

    # P/kB [K cm^-3]
    f = 'pok'
    field_dep[f] = set(['pressure'])
    def _pok(d, u):
        return d['pressure']*(u.energy_density/ac.k_B).cgs.value
    func[f] = _pok
    label[f] = r'$P/k_{\rm B}\;[{\rm cm^{-3}\,K}]$'
    cmap[f] = 'inferno'
    vminmax[f] = (1e2,1e7)
    take_log[f] = True
    
    # Distance from x0 [pc]    
    f = 'r'
    field_dep[f] = set(['density'])
    @static_vars(x0=x0)
    def _r(d, u):
        z, y, x = np.meshgrid(d['z'], d['y'], d['x'], indexing='ij')
        return xr.DataArray(np.sqrt((x - _r.x0[0])**2 + \
                                    (y - _r.x0[1])**2 + \
                                    (z - _r.x0[2])**2),
                            dims=('z','y','x'), name='r')
    func[f] = _r
    label[f] = r'$r\;[{\rm pc}]$'
    cmap[f] = 'viridis'
    # Set vmax to box length
    Lx = par['domain1']['x1max']-par['domain1']['x1min']
    Ly = par['domain1']['x2max']-par['domain1']['x2min']
    Lz = par['domain1']['x3max']-par['domain1']['x3min']
    Lmax = max(max(Lx,Ly),Lz)
    vminmax[f] = (0,Lmax)
    take_log[f] = False

    # Velocity magnitude [km/s]
    f = 'vmag'
    field_dep[f] = set(['velocity'])
    def _vmag(d, u):
        return (d['velocity1']**2 + d['velocity2']**2 + d['velocity3']**2)**0.5*u.kms
    func[f] = _vmag
    label[f] = r'$|\mathbf{v}|\;[{\rm km\,s^{-1}}]$'
    vminmax[f] = (0.1, 1000.0)
    cmap[f] = 'cividis'
    take_log[f] = True

    # Radial velocity w.r.t. x0 [km/s]
    f = 'vr'
    field_dep[f] = set(['velocity'])
    @static_vars(x0=x0)
    def _vr(d, u):
        z, y, x = np.meshgrid(d['z'], d['y'], d['x'], indexing='ij')
        r = xr.DataArray(np.sqrt((x - _r.x0[0])**2 + (y - _r.x0[1])**2 + (z - _r.x0[2])**2),
                            dims=('z','y','x'), name='r')
        return (x*d['velocity1'] + y*d['velocity2'] + z*d['velocity3'])/r*u.kms
    func[f] = _vr
    label[f] = r'$v_r\;[{\rm km\,s^{-1}}]$'
    vminmax[f] = (-20, 100)
    # Set cmap midpoint accordingly (midpoint=abs(vmin)/(abs(vmin)+abs(vmax))
    cmap[f] = cmap_shift(mpl.cm.BrBG,
                         midpoint=abs(vminmax[f][0]) / \
                                  (abs(vminmax[f][0]) + abs(vminmax[f][1])),
                         name='cmap_vr')
    take_log[f] = False
    
    # vx [km/s]
    f = 'vx'
    field_dep[f] = set(['velocity'])
    def _vx(d, u):
        return d['velocity1']*u.kms
    func[f] = _vx
    label[f] = r'$v_x\;[{\rm km\,s^{-1}}]$'
    cmap[f] = 'RdBu'
    vminmax[f] = (-100.0,100.0)
    take_log[f] = False

    # vy [km/s]
    f = 'vy'
    field_dep[f] = set(['velocity'])
    def _vy(d, u):
        return d['velocity2']*u.kms
    func[f] = _vy
    label[f] = r'$v_y\;[{\rm km\,s^{-1}}]$'
    cmap[f] = 'RdBu'
    vminmax[f] = (-100.0,100.0)
    take_log[f] = False

    # vz [km/s]
    f = 'vz'
    field_dep[f] = set(['velocity'])
    def _vz(d, u):
        return d['velocity3']*u.kms
    func[f] = _vz
    label[f] = r'$v_z\;[{\rm km\,s^{-1}}]$'
    cmap[f] = 'RdBu'
    vminmax[f] = (-100.0,100.0)
    take_log[f] = False
    
    # Radial momentum w.r.t. x0 [km/s cm^-3]
    f = 'pr'
    field_dep[f] = set(['density','velocity'])
    @static_vars(x0=x0)
    def _pr(d, u):
        z, y, x = np.meshgrid(d['z'], d['y'], d['x'], indexing='ij')
        r = xr.DataArray(np.sqrt((x - _r.x0[0])**2 + (y - _r.x0[1])**2 + (z - _r.x0[2])**2),
                            dims=('z','y','x'), name='r')
        return d['density']*(x*d['velocity1'] + y*d['velocity2'] + z*d['velocity3'])/r*u.kms
    func[f] = _pr
    label[f] = r'$p_r\;[{\rm cm^{-3}\,km\,s^{-1}}]$'
    vminmax[f] = (-1e5, 1e5)
    # Set cmap midpoint accordingly (midpoint=abs(vmin)/(abs(vmin)+abs(vmax))
    cmap[f] = cmap_shift(mpl.cm.BrBG,
                         midpoint=abs(vminmax[f][0]) / \
                                  (abs(vminmax[f][0]) + abs(vminmax[f][1])),
                         name='cmap_vr')
    take_log[f] = False

    # Absolute value of radial momentum w.r.t. x0 [km/s cm^-3]
    f = 'pr_abs'
    field_dep[f] = set(['density','velocity'])
    @static_vars(x0=x0)
    def _pr_abs(d, u):
        z, y, x = np.meshgrid(d['z'], d['y'], d['x'], indexing='ij')
        r = xr.DataArray(np.sqrt((x - _r.x0[0])**2 + (y - _r.x0[1])**2 + (z - _r.x0[2])**2),
                            dims=('z','y','x'), name='r')
        return np.abs(d['density']*(x*d['velocity1'] + y*d['velocity2'] + z*d['velocity3'])/r)*u.kms
    func[f] = _pr_abs
    label[f] = r'$|p_r|\;[{\rm cm^{-3}\,km\,s^{-1}}]$'
    vminmax[f] = (1e-2, 1e4)
    # Set cmap midpoint accordingly (midpoint=abs(vmin)/(abs(vmin)+abs(vmax))
    cmap[f] = cmap_shift(mpl.cm.BrBG,
                         midpoint=abs(vminmax[f][0]) / \
                                  (abs(vminmax[f][0]) + abs(vminmax[f][1])),
                         name='cmap_vr')
    take_log[f] = True
    
    # Cooling
    if par['configure']['cooling'] == 'ON':
        # T [K]
        f = 'T'
        if par['configure']['new_cooling'] == 'ON':
            field_dep[f] = set(['density','pressure','xe','xH2'])
            def _T(d, u):
                return d['pressure']/(d['density']*(1.1 + d['xe'] - d['xH2']))/\
                    (ac.k_B/u.energy_density).cgs.value
        else:
            field_dep[f] = set(['temperature'])
            def _T(d, u):
                return d['temperature']
        func[f] = _T
        label[f] = r'$T\;[{\rm K}]$'
        cmap[f] = cmap_shift(mpl.cm.RdYlBu_r, midpoint=3./7., name='cmap_T')
        vminmax[f] = (1e1,1e7)
        take_log[f] = True

        f = 'cool_rate'
        field_dep[f] = set(['cool_rate'])
        def _cool_rate(d, u):
            return d['cool_rate']
        func[f] = _cool_rate
        label[f] = r'$\mathcal{L}\;[{\rm erg}\,{\rm cm^{-3}}\,{\rm s}^{-1}]$'
        cmap[f] = 'cubehelix_r'
        vminmax[f] = (1e-26,1e-18)
        take_log[f] = True

        f = 'heat_rate'
        field_dep[f] = set(['heat_rate'])
        def _heat_rate(d, u):
            return d['heat_rate']
        func[f] = _heat_rate
        label[f] = r'$\mathcal{G}\;[{\rm erg}\,{\rm cm^{-3}}\,{\rm s}^{-1}]$'
        cmap[f] = 'cubehelix_r'
        vminmax[f] = (1e-28,1e-20)
        take_log[f] = True

        f = 'net_cool_rate'
        field_dep[f] = set(['cool_rate','heat_rate'])
        def _net_cool_rate(d, u):
            return d['cool_rate'] - d['heat_rate']
        func[f] = _net_cool_rate
        label[f] = r'$\mathcal{L}\;[{\rm erg}\,{\rm cm^{-3}}\,{\rm s}^{-1}]$'
        cmap[f] = 'bwr_r'
        vminmax[f] = (-1e-20,1e-20)
        take_log[f] = False

        f = 'Lambda_cool'
        field_dep[f] = set(['density','cool_rate'])
        def _Lambda_cool(d, u):
            return d['cool_rate']/d['density']**2
        func[f] = _Lambda_cool
        label[f] = r'$\Lambda\;[{\rm erg}\,{\rm cm^{3}}\,{\rm s}^{-1}]$'
        cmap[f] = 'cubehelix_r'
        vminmax[f] = (1e-30,1e-20)
        take_log[f] = True

    return func, field_dep, label, cmap, vminmax, take_log    

def set_derived_fields_mag(par, x0):

    func = dict()
    field_dep = dict()
    label = dict()
    cmap = dict()
    vminmax = dict()
    take_log = dict()

    # Bx [G]
    f = 'Bx'
    field_dep[f] = set(['cell_centered_B'])
    def _Bx(d, u):
        return d['cell_centered_B1']*np.sqrt(u.energy_density.cgs.value)\
            *np.sqrt(4.0*np.pi)
    func[f] = _Bx
    label[f] = r'$B_{x}\;[{\rm G}]$'
    cmap[f] = 'RdBu'
    vminmax[f] = (-1e-4,-1e-4)
    take_log[f] = False

    # By [G]
    f = 'By'
    field_dep[f] = set(['cell_centered_B'])
    def _By(d, u):
        return d['cell_centered_B2']*np.sqrt(u.energy_density.cgs.value)\
            *np.sqrt(4.0*np.pi)
    func[f] = _By
    label[f] = r'$B_{y}\;[{\rm G}]$'
    cmap[f] = 'RdBu'
    vminmax[f] = (-1e-4,-1e-4)
    take_log[f] = False

    # Bz [G]
    f = 'Bz'
    field_dep[f] = set(['cell_centered_B'])
    def _Bz(d, u):
        return d['cell_centered_B3']*np.sqrt(u.energy_density.cgs.value)\
            *np.sqrt(4.0*np.pi)
    func[f] = _Bz
    label[f] = r'$B_{z}\;[{\rm G}]$'
    cmap[f] = 'RdBu'
    vminmax[f] = (-1e-4,-1e-4)
    take_log[f] = False

    # Magnetic fields magnitude [G]
    f = 'Bmag'
    field_dep[f] = set(['cell_centered_B'])
    def _Bmag(d, u):
        return (d['cell_centered_B1']**2 +
                d['cell_centered_B2']**2 +
                d['cell_centered_B3']**2)**0.5*np.sqrt(u.energy_density.cgs.value)\
            *np.sqrt(4.0*np.pi)
    func[f] = _Bmag
    label[f] = r'$|\mathbf{B}|\;[{\rm G}]$'
    vminmax[f] = (1e-7, 1e-4)
    cmap[f] = 'cividis'
    take_log[f] = True
    
    return func, field_dep, label, cmap, vminmax, take_log

def set_derived_fields_newcool(par, x0):
    
    func = dict()
    field_dep = dict()
    label = dict()
    cmap = dict()
    vminmax = dict()
    take_log = dict()
    
    # nH2 [cm^-3] (assume d=nH)
    f = 'nH2'
    field_dep[f] = set(['density', 'xH2'])
    def _nH2(d, u):
        return d['density']*d['xH2']
    func[f] = _nH2
    label[f] = r'$n_{\rm H_2}\;[{\rm cm^{-3}}]$'
    cmap[f] = get_my_cmap('Greens')
    vminmax[f] = (1e0,1e4)
    take_log[f] = True

    # 2nH2 [cm^-3] (assume d=nH)
    f = '2nH2'
    field_dep[f] = set(['density', 'xH2'])
    def _2nH2(d, u):
        return 2.0*d['density']*d['xH2']
    func[f] = _2nH2
    label[f] = r'$2n_{\rm H_2}\;[{\rm cm^{-3}}]$'
    cmap[f] = get_my_cmap('Greens')
    vminmax[f] = (1e0,1e4)
    take_log[f] = True

    # xH2 [cm^-3] (assume d=nH)
    f = 'xH2'
    field_dep[f] = set(['xH2'])
    def _xH2(d, u):
        return d['xH2']
    func[f] = _xH2
    label[f] = r'$x_{\rm H_2}$'
    cmap[f] = 'viridis'
    vminmax[f] = (0,0.5)
    take_log[f] = False

    # 2xH2 [cm^-3] (assume d=nH)
    f = '2xH2'
    field_dep[f] = set(['xH2'])
    def _2xH2(d, u):
        return 2*d['xH2']
    func[f] = _2xH2
    label[f] = r'$2x_{\rm H_2}$'
    cmap[f] = 'viridis'
    vminmax[f] = (0,1.0)
    take_log[f] = False

    # nHI [cm^-3]
    f = 'nHI'
    field_dep[f] = set(['density', 'xHI'])
    def _nHI(d, u):
        return d['density']*d['xHI']
    func[f] = _nHI
    label[f] = r'$n_{\rm H^0}\;[{\rm cm^{-3}}]$'
    cmap[f] = get_my_cmap('Blues')
    vminmax[f] = (1e-3,1e4)
    take_log[f] = True

    # xHI
    f = 'xHI'
    field_dep[f] = set(['xHI'])
    def _xHI(d, u):
        return d['xHI']
    func[f] = _xHI
    label[f] = r'$x_{\rm H^0}$'
    cmap[f] = 'viridis'
    vminmax[f] = (0.0, 1.0)
    take_log[f] = False

    # nHII [cm^-3]
    f = 'nHII'
    field_dep[f] = set(['density', 'xHI', 'xH2'])
    def _nHII(d, u):
        return d['density']*(1.0 - d['xHI'] - 2.0*d['xH2'])
    func[f] = _nHII
    label[f] = r'$n_{\rm H^+}\;[{\rm cm^{-3}}]$'
    cmap[f] = get_my_cmap('Oranges')
    vminmax[f] = (1e-3,1e4)
    take_log[f] = True

    # xHII
    f = 'xHII'
    field_dep[f] = set(['xH2','xHI'])
    def _xHII(d, u):
        return 1.0 - d['xHI'] - 2.0*d['xH2']
    func[f] = _xHII
    label[f] = r'$x_{\rm H^+}$'
    cmap[f] = 'viridis'
    vminmax[f] = (0.0, 1.0)
    take_log[f] = False

    # nHn [cm^-3]
    f = 'nHn'
    field_dep[f] = set(['density', 'xHI', 'xH2'])
    def _nHn(d, u):
        return d['density']*(d['xHI'] + 2.0*d['xH2'])
    func[f] = _nHn
    label[f] = r'$n_{\rm H^0} + 2n_{\rm H_2}\;[{\rm cm^{-3}}]$'
    cmap[f] = get_my_cmap('Blues')
    vminmax[f] = (1e-3,1e4)
    take_log[f] = True

    # xn [cm^-3]
    f = 'xn'
    field_dep[f] = set(['xHI', 'xH2'])
    def _xn(d, u):
        return d['xHI'] + 2.0*d['xH2']
    func[f] = _xn
    label[f] = r'$x_{\rm n}$'
    cmap[f] = get_my_cmap('YlGn')
    vminmax[f] = (0,1)
    take_log[f] = False
        
    # ne
    f = 'ne'
    field_dep[f] = set(['density','xe'])
    def _ne(d, u):
        return d['density']*d['xe']
    func[f] = _ne
    label[f] = r'$n_{\rm e}$'
    cmap[f] = 'viridis'
    vminmax[f] = (1e-4, 1e3)
    take_log[f] = True

    # nesq
    f = 'nesq'
    field_dep[f] = set(['density','xe'])
    def _nesq(d, u):
        return (d['density']*d['xe'])**2
    func[f] = _nesq
    label[f] = r'$n_{\rm e}^2$'
    cmap[f] = 'viridis'
    vminmax[f] = (1e-8, 1e6)
    take_log[f] = True
    
    # xe
    f = 'xe'
    field_dep[f] = set(['xe'])
    def _xe(d, u):
        return d['xe']
    func[f] = _xe
    label[f] = r'$x_{\rm e}$'
    cmap[f] = 'viridis'
    vminmax[f] = (1e-6, 1.208)
    take_log[f] = True

    # xCI - atomic neutral carbon
    f = 'xCI'
    try:
        xCtot = par['problem']['Z_gas']*par['cooling']['xCstd']
    except KeyError:
        # print('xCtot not found. Use 1.6e-4.')
        xCtot = 1.6e-4
    field_dep[f] = set(['xCI_over_xCtot'])
    def _xCI(d, u):
        # Apply floor and ceiling
        return np.maximum(0.0,np.minimum(xCtot,d['xCI_over_xCtot']*xCtot))
    func[f] = _xCI
    label[f] = r'$x_{\rm CI}$'
    cmap[f] = 'viridis'
    vminmax[f] = (0,xCtot)
    take_log[f] = False

    # nCI
    f = 'nCI'
    field_dep[f] = set(['density','xCI'])
    def _nCI(d, u):
        return d['density']*np.maximum(0.0,np.minimum(xCtot,d['xCI']))
    func[f] = _nCI
    label[f] = r'$x_{\rm CI}$'
    cmap[f] = 'viridis'
    vminmax[f] = (1e2*xCtot,1e4*xCtot)
    take_log[f] = True
    
    # xCII - single ionized carbon
    # Use with caution!
    # (Do not apply to hot gas and depend on cooling implementation)
    f = 'xCII'
    try:
        xCtot = par['problem']['Z_gas']*par['cooling']['xCstd']
    except KeyError:
        # print('xCtot not found. Use 1.6e-4.')
        xCtot = 1.6e-4
    field_dep[f] = set(['xe','xH2','xHI'])
    def _xCII(d, u):
        # Apply floor and ceiling
        return np.maximum(0.0,np.minimum(xCtot,d['xe'] - (1.0 - d['xHI'] - 2.0*d['xH2'])))
    func[f] = _xCII
    label[f] = r'$x_{\rm CII}$'
    cmap[f] = 'viridis'
    vminmax[f] = (0,xCtot)
    take_log[f] = False

    # xi_CR
    f = 'xi_CR'
    field_dep[f] = set(['CR_ionization_rate'])
    def _xi_CR(d, u):
        return d['CR_ionization_rate']
    func[f] = _xi_CR
    label[f] = r'$\xi_{\rm CR}\;[{\rm s}^{-1}]$'
    cmap[f] = 'viridis'
    vminmax[f] = (1e-15,1e-17)
    take_log[f] = True

    # T_alt [K]
    f = 'T_alt'
    field_dep[f] = set(['pressure','density','xe','xH2'])
    def _T_alt(d, u):
        return d['pressure']*(u.energy_density/ac.k_B).cgs.value/(d['density']*(1.1 + d['xe'] - d['xH2']))
    func[f] = _T_alt
    label[f] = r'$T_{\rm alt}\;[{\rm K}]$'
    cmap[f] = cmap_shift(mpl.cm.RdYlBu_r, midpoint=3./7., name='cmap_T')
    vminmax[f] = (1e1,1e7)
    take_log[f] = True
    
    return func, field_dep, label, cmap, vminmax, take_log


def set_derived_fields_sixray(par, x0):

    func = dict()
    field_dep = dict()
    label = dict()
    cmap = dict()
    vminmax = dict()
    take_log = dict()

    try:
        Erad_PE0 = par['cooling']['Erad_PE0']
        Erad_LW0 = par['cooling']['Erad_LW0']
    except KeyError:
        Erad_PE0 = 7.613e-14
        Erad_LW0 = 1.335e-14
    
    # Normalized FUV radiation field strength (Draine ISRF)
    f = 'chi_PE_ext'
    field_dep[f] = set(['rad_energy_density_PE_ext'])
    def _chi_PE_ext(d, u):
        return d['rad_energy_density_PE_ext']*(u.energy_density.cgs.value/Erad_PE0)
    func[f] = _chi_PE_ext
    label[f] = r'$\chi_{\rm PE,ext}$'
    cmap[f] = 'viridis'
    vminmax[f] = (1e-4,1e4)
    take_log[f] = True

    # Normalized LW radiation field
    f = 'chi_LW_ext'
    field_dep[f] = set(['rad_energy_density_LW_ext'])
    def _chi_LW_ext(d, u):
        return d['rad_energy_density_LW_ext']*(u.energy_density.cgs.value/Erad_LW0)
    func[f] = _chi_LW_ext
    label[f] = r'$\chi_{\rm LW,ext}$'
    cmap[f] = 'viridis'
    vminmax[f] = (1e-4,1e4)
    take_log[f] = True


    # Normalized LW radiation field
    f = 'chi_CI_ext'
    field_dep[f] = set(['rad_energy_density_CI_ext'])
    def _chi_CI_ext(d, u):
        return d['rad_energy_density_CI_ext']*(u.energy_density.cgs.value/Erad_LW0)
    func[f] = _chi_CI_ext
    label[f] = r'$\chi_{\rm CI,ext}$'
    cmap[f] = 'viridis'
    vminmax[f] = (1e-4,1e4)
    take_log[f] = True

    
    # Normalized LW radiation field strength (Draine ISRF)
    f = 'chi_H2_ext'
    field_dep[f] = set(['rad_energy_density_LW_diss_ext'])
    def _chi_H2_ext(d, u):
        return d['rad_energy_density_LW_diss_ext']*(u.energy_density.cgs.value/Erad_LW0)
    func[f] = _chi_H2_ext
    label[f] = r'$\chi_{\rm H2,ext}$'
    cmap[f] = 'viridis'
    vminmax[f] = (1e-8,1e2)
    take_log[f] = True

    return func, field_dep, label, cmap, vminmax, take_log

def set_derived_fields_rad(par, x0):

    func = dict()
    field_dep = dict()
    label = dict()
    cmap = dict()
    vminmax = dict()
    take_log = dict()
    
    # Dust PE opacity for Z'=1
    kappa_dust_PE_def = 418.7

    try:
        Erad_PE0 = par['cooling']['Erad_PE0']
        Erad_LW0 = par['cooling']['Erad_LW0']
    except KeyError:
        Erad_PE0 = 7.613e-14
        Erad_LW0 = 1.335e-14
    
    # Normalized FUV radiation field strength (Draine field unit)
    f = 'chi_PE'
    field_dep[f] = set(['rad_energy_density_PE'])
    def _chi_PE(d, u):
        return d['rad_energy_density_PE']*(u.energy_density.cgs.value/Erad_PE0)
    func[f] = _chi_PE
    label[f] = r'$\chi_{\rm PE}$'
    cmap[f] = 'viridis'
    vminmax[f] = (1e-4,1e4)
    take_log[f] = True

    # Normalized FUV radiation field strength (Draine field unit)
    f = 'chi_FUV'
    field_dep[f] = set(['rad_energy_density_PE','rad_energy_density_LW'])
    def _chi_FUV(d, u):
        return (d['rad_energy_density_PE'] + d['rad_energy_density_LW'])*(u.energy_density.cgs.value/(Erad_PE0 + Erad_LW0))
    func[f] = _chi_FUV
    label[f] = r'$\chi_{\rm FUV}$'
    cmap[f] = 'viridis'
    vminmax[f] = (1e-4,1e4)
    take_log[f] = True
    
    # Normalized FUV radiation field strength (Draine field unit)
    f = 'Erad_LyC'
    field_dep[f] = set(['rad_energy_density_PH'])
    def _Erad_LyC(d, u):
        return d['rad_energy_density_PH']*u.energy_density.cgs.value
    func[f] = _Erad_LyC
    label[f] = r'$\mathcal{E}_{\rm LyC}\;[{\rm erg\,{\rm cm}^{-3}}]$'
    cmap[f] = 'viridis'
    vminmax[f] = (5e-16,5e-11)
    take_log[f] = True

    # Normalized FUV radiation field strength (Draine field unit)
    f = 'Erad_FUV'
    field_dep[f] = set(['rad_energy_density_PE','rad_energy_density_LW'])
    def _Erad_FUV(d, u):
        return (d['rad_energy_density_PE'] + d['rad_energy_density_LW'])*u.energy_density.cgs.value
    func[f] = _Erad_FUV
    label[f] = r'$\mathcal{E}_{\rm FUV}$'
    cmap[f] = 'viridis'
    vminmax[f] = (5e-16,5e-11)
    take_log[f] = True
    
    # heat_ratio (G0) ; temporary output
    f = 'heat_ratio'
    field_dep[f] = set(['heat_ratio'])
    def _heat_ratio(d, u):
        return d['heat_ratio']
    func[f] = _heat_ratio
    label[f] = r'heat_ratio'
    cmap[f] = 'viridis'
    vminmax[f] = (1e-1,1e4)
    take_log[f] = True

    # NHeff (based on dust attenuation of photoelectric band)
    f = 'NHeff'
    field_dep[f] = set(['rad_energy_density_PE','rad_energy_density_PE_unatt'])
    def _NHeff(d, u):
        return 1e21/par['radps']['kappa_dust_PE']*kappa_dust_PE_def*\
            np.log(d['rad_energy_density_PE_unatt']/d['rad_energy_density_PE'])
    func[f] = _NHeff
    label[f] = r'$N_{\rm H,eff}\;[{\rm cm^{-2}}]$'
    cmap[f] = 'gist_earth'
    vminmax[f] = (0,1e22)
    take_log[f] = False

    try:
        if par['configure']['lwrad'] == 'ON':
            # Normalized LW intensity (attenuated by dust and H2 self-shielding)
            # From Sternberg+14,
            # For Draine ISRF and 912 < lambda/angstrom < 1108,
            # The total photon density is 6.9e-4/cm^3, or flux 2.07e7/cm^2/s
            # Energy density is u_LW = 1.33e-14 erg/cm^3 and
            # the mean energy of photons is ~12eV
            # (cf. Mathis ISRF gives u_LW = 9.56e-15 erg/cm^3; see also Heays+17)
            f = 'chi_LW'
            field_dep[f] = set(['rad_energy_density_LW'])
            def _chi_LW(d, u):
                return d['rad_energy_density_LW']*(u.energy_density.cgs.value/Erad_LW0)
            func[f] = _chi_LW
            label[f] = r'$\chi_{\rm LW}$'
            cmap[f] = 'viridis'
            vminmax[f] = (1e-4,1e4)
            take_log[f] = True

            f = 'chi_H2'
            field_dep[f] = set(['rad_energy_density_LW_diss'])
            def _chi_H2(d, u):
                return d['rad_energy_density_LW_diss']*(u.energy_density.cgs.value/Erad_LW0)
            func[f] = _chi_H2
            label[f] = r'$\chi_{\rm H_2}$'
            cmap[f] = 'viridis'
            vminmax[f] = (1e-8,1e4)
            take_log[f] = True

            f = 'chi_CI'
            field_dep[f] = set(['rad_energy_density_CI'])
            def _chi_CI(d, u):
                return d['rad_energy_density_CI']*(u.energy_density.cgs.value/Erad_LW0)
            func[f] = _chi_CI
            label[f] = r'$\chi_{\rm C^0}$'
            cmap[f] = 'viridis'
            vminmax[f] = (1e-8,1e4)
            take_log[f] = True
            
            # fshld_H2 (effective)
            f = 'fshld_H2'
            field_dep[f] = set(['rad_energy_density_LW','rad_energy_density_LW_diss'])
            def _fshld_H2(d, u):
                return d['rad_energy_density_LW_diss']/d['rad_energy_density_LW']
            func[f] = _fshld_H2
            label[f] = r'$f_{\rm shld,H2}$'
            cmap[f] = 'tab20c'
            vminmax[f] = (1e-6,1e0)
            take_log[f] = True

    except KeyError:
        pass
    
    return func, field_dep, label, cmap, vminmax, take_log

def set_derived_fields_xray(par, x0):
    
    func = dict()
    field_dep = dict()
    label = dict()
    cmap = dict()
    vminmax = dict()
    take_log = dict()

    # TODO-JKIM: Need Z_gas parameter in the problem block
    # Or metallicity field
    Z_gas = par['problem']['Z_gas']
    emin = 0.5 # keV
    emax = 7.0 # keV
    energy = True # If set to False, returns photon emissivity [#/s/cm^3]

    # Normalized FUV radiation field strength (Draine field unit)
    f = 'j_X'
    field_dep[f] = set(['density','temperature'])
    # Frequency integrated volume emissivity
    def _j_Xray(d, u):
        em = get_xray_emissivity(d['temperature'].data, Z_gas,
                                 emin, emax, energy=energy)
        return d['density']**2*em
    func[f] = _j_Xray
    label[f] = r'$j_{\rm X}$ (0.5-7.0 keV) $[{\rm erg}\,{\rm s}^{-1}\,{\rm cm}^{-3}]$'
    cmap[f] = 'plasma'
    vminmax[f] = (1e-34,1e-25)
    take_log[f] = True

    return func, field_dep, label, cmap, vminmax, take_log


class DerivedFields(object):

    def __init__(self, par, x0=np.array([0.0,0.0,0.0])):

        # Create a dictionary containing all information about derived fields
        self.dfi = dict()

        self.func, self.field_dep, \
            self.label, self.cmap, \
            self.vminmax, self.take_log = set_derived_fields_def(par, x0)

        dicts = (self.func, self.field_dep, self.label, self.cmap, \
                 self.vminmax, self.take_log)

        if par['configure']['gas'] == 'mhd':
            dicts_ = set_derived_fields_mag(par, x0)
            for d, d_ in zip(dicts, dicts_):
                d = d.update(d_)

        try:
            if par['configure']['new_cooling'] == 'ON':
                dicts_ = set_derived_fields_newcool(par, x0)
                for d, d_ in zip(dicts, dicts_):
                    d = d.update(d_)
        except KeyError:
            pass

        try:
            if par['configure']['radps'] == 'ON':
                dicts_ = set_derived_fields_rad(par, x0)
                for d, d_ in zip(dicts, dicts_):
                    d = d.update(d_)
        except KeyError:
            pass                    

        try:
            if par['configure']['sixray'] == 'ON':
                dicts_ = set_derived_fields_sixray(par, x0)
                for d, d_ in zip(dicts, dicts_):
                    d = d.update(d_)
        except KeyError:
            pass
                
        # Add X-ray emissivity if Wind or SN is turned on
        try:
            if par['feedback']['iSN'] > 0 or par['feedback']['iWind'] > 0 or \
               par['feedback']['iEarly'] > 0:
                dicts_ = set_derived_fields_xray(par, x0)
                for d, d_ in zip(dicts, dicts_):
                    d = d.update(d_)
        except KeyError:
            pass
                    
        self.derived_field_list = self.func

        # Set colormap normalization and scale
        self.norm = dict()
        self.scale = dict()
        self.imshow_args = dict()
        for f in self.derived_field_list:
            if self.take_log[f]:
                self.norm[f] = LogNorm(*self.vminmax[f])
                self.scale[f] = 'log'
            else:
                self.norm[f] = Normalize(*self.vminmax[f])
                self.scale[f] = 'linear'

            self.imshow_args[f] = dict(norm=self.norm[f], cmap=self.cmap[f],
                                       cbar_kwargs=dict(label=self.label[f]))
            
        for f in self.derived_field_list:
            self.dfi[f] = dict(field_dep=self.field_dep[f],
                               func=self.func[f],
                               label=self.label[f],
                               norm=self.norm[f],
                               vminmax=self.vminmax[f],
                               cmap=self.cmap[f],
                               scale=self.scale[f],
                               take_log=self.take_log[f],
                               imshow_args=self.imshow_args[f])
                               
