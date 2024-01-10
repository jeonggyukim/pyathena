from functools import wraps

import matplotlib as mpl
import numpy as np
import xarray as xr
import astropy.constants as ac
import astropy.units as au

from matplotlib.colors import Normalize, LogNorm

from ..plt_tools.cmap import cmap_apply_alpha,cmap_shift
from ..microphysics.cool import get_xe_mol
from .xray_emissivity import get_xray_emissivity

def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate

def set_derived_fields_def(par, x0, newcool):
    """
    Function to define derived fields info, for example,
    functions to calculate derived fields, dependency, label, colormap, etc.

    May not work correctly for problems using different unit system.
    Assume that density = nH, length unit = pc, etc.

    Parameters
    ----------
    par: dict
       Dictionary containing simulation parameter information
    x0: sequence of floats
       Coordinate of the center with respect to which distance is measured
    newcool: bool
       Is new cooling turned on?

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

    # rho [g cm^-3] - gas density
    f = 'rho'
    field_dep[f] = ['density']
    def _rho(d, u):
        return d['density']*(u.muH*u.mH).cgs.value
    func[f] = _rho
    label[f] = r'$\rho\;[{\rm g\,cm^{-3}}]$'
    cmap[f] = 'Spectral_r'
    vminmax[f] = (1e-27,1e-20)
    take_log[f] = True

    # nH [cm^-3] (assume that density = nH)
    f = 'nH'
    field_dep[f] = ['density']
    def _nH(d, u):
        return d['density']
    func[f] = _nH
    label[f] = r'$n_{\rm H}\;[{\rm cm^{-3}}]$'
    cmap[f] = 'Spectral_r'
    vminmax[f] = (1e-3,1e4)
    take_log[f] = True

    # P/kB [K cm^-3] - thermal pressure
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

    # cs [km/s]
    f = 'cs'
    field_dep[f] = set(['pressure','density'])
    def _cs(d, u):
        return np.sqrt(d['pressure']/d['density'])
    func[f] = _cs
    label[f] = r'$c_s\;[{\rm km}\,{\rm s}^{-1}]$'
    cmap[f] = 'magma'
    vminmax[f] = (0.1,1e3)
    take_log[f] = True

    # cs [km/s] - sound speed
    f = 'csound'
    field_dep[f] = set(['pressure','density'])
    def _csound(d, u):
        return np.sqrt(par['problem']['gamma']*d['pressure']/d['density'])
    func[f] = _csound
    label[f] = r'$c_s\;[{\rm km}\,{\rm s}^{-1}]$'
    cmap[f] = 'magma'
    vminmax[f] = (0.1,1e3)
    take_log[f] = True

    # Radial momentum w.r.t. x0 [km/s cm^-3]
    f = 'Mr'
    field_dep[f] = set(['density','velocity'])
    @static_vars(x0=x0)
    def _Mr(d, u):
        z, y, x = np.meshgrid(d['z'], d['y'], d['x'], indexing='ij')
        r = xr.DataArray(np.sqrt((x - _r.x0[0])**2 + (y - _r.x0[1])**2 + (z - _r.x0[2])**2),
                            dims=('z','y','x'), name='r')
        return d['density']*(x*d['velocity1'] + y*d['velocity2'] + z*d['velocity3'])/r*u.kms
    func[f] = _Mr
    label[f] = r'$p_r\;[{\rm cm^{-3}\,km\,s^{-1}}]$'
    vminmax[f] = (-1e5, 1e5)
    # Set cmap midpoint accordingly (midpoint=abs(vmin)/(abs(vmin)+abs(vmax))
    cmap[f] = cmap_shift(mpl.cm.BrBG,
                         midpoint=abs(vminmax[f][0]) / \
                                  (abs(vminmax[f][0]) + abs(vminmax[f][1])),
                         name='cmap_pyathena_Mr')
    take_log[f] = False

    # Absolute value of radial momentum w.r.t. x0 [km/s cm^-3]
    f = 'Mr_abs'
    field_dep[f] = set(['density','velocity'])
    @static_vars(x0=x0)
    def _Mr_abs(d, u):
        z, y, x = np.meshgrid(d['z'], d['y'], d['x'], indexing='ij')
        r = xr.DataArray(np.sqrt((x - _r.x0[0])**2 + (y - _r.x0[1])**2 + (z - _r.x0[2])**2),
                            dims=('z','y','x'), name='r')
        return np.abs(d['density']*(x*d['velocity1'] + y*d['velocity2'] + z*d['velocity3'])/r)*u.kms
    func[f] = _Mr_abs
    label[f] = r'$|p_r|\;[{\rm cm^{-3}\,km\,s^{-1}}]$'
    vminmax[f] = (1e-2, 1e4)
    # Set cmap midpoint accordingly (midpoint=abs(vmin)/(abs(vmin)+abs(vmax))
    cmap[f] = cmap_shift(mpl.cm.BrBG,
                         midpoint=abs(vminmax[f][0]) / \
                                  (abs(vminmax[f][0]) + abs(vminmax[f][1])),
                         name='cmap_pyathena_Mr_abs')
    take_log[f] = True

    # rho vr^2 / kB [cm^-3 K]
    f = 'rhovr2ok'
    field_dep[f] = set(['density','velocity'])
    @static_vars(x0=x0)
    def _rhovr2ok(d, u):
        z, y, x = np.meshgrid(d['z'], d['y'], d['x'], indexing='ij')
        r = xr.DataArray(np.sqrt((x - _r.x0[0])**2 + (y - _r.x0[1])**2 + (z - _r.x0[2])**2),
                            dims=('z','y','x'), name='r')
        return (d['density']*u.density.cgs.value)*\
            ((x*d['velocity1'] + y*d['velocity2'] + z*d['velocity3'])/r*u.velocity.cgs.value)**2/ac.k_B.cgs.value
    func[f] = _rhovr2ok
    label[f] = r'$\rho v_r^2\;[{\rm cm}^{-3}\,{\rm K}]$'
    vminmax[f] = (1e2, 1e7)
    # Set cmap midpoint accordingly (midpoint=abs(vmin)/(abs(vmin)+abs(vmax))
    cmap[f] = 'inferno'
    take_log[f] = True

    # Cooling related fields
    if par['configure']['cooling'] == 'ON':
        # T [K] - gas temperature
        f = 'T'
        if newcool:
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
        cmap[f] = cmap_shift(mpl.cm.RdYlBu_r, midpoint=3./7., name='cmap_pyathena_T')
        vminmax[f] = (1e1,1e7)
        take_log[f] = True

        # Td [K] - dust temperature
        if newcool:
            f = 'Td'
            field_dep[f] = set(['temperature_dust'])
            def _Td(d, u):
                return d['temperature_dust']

            func[f] = _Td
            label[f] = r'$T_{\rm d}\;[{\rm K}]$'
            cmap[f] = cmap_shift(mpl.cm.RdYlBu_r, midpoint=3./7., name='cmap_pyathena_Td')
            vminmax[f] = (1e0,1e2)
            take_log[f] = True

        # Cooling rate per volume [erg/s/cm^3] - nH^2*Lambda
        f = 'cool_rate'
        field_dep[f] = set(['cool_rate'])
        def _cool_rate(d, u):
            return d['cool_rate']
        func[f] = _cool_rate
        label[f] = r'$\mathcal{L}\;[{\rm erg}\,{\rm cm^{-3}}\,{\rm s}^{-1}]$'
        cmap[f] = 'cubehelix_r'
        vminmax[f] = (1e-26,1e-18)
        take_log[f] = True

        # Heating rate per volume [erg/s/cm^3] - nH*Gamma
        f = 'heat_rate'
        field_dep[f] = set(['heat_rate'])
        def _heat_rate(d, u):
            return d['heat_rate']
        func[f] = _heat_rate
        label[f] = r'$\mathcal{G}\;[{\rm erg}\,{\rm cm^{-3}}\,{\rm s}^{-1}]$'
        cmap[f] = 'cubehelix_r'
        vminmax[f] = (1e-28,1e-20)
        take_log[f] = True

        # Net cooling rate per volume [erg/s/cm^3] - nH^2*Lambda - nH*Gamma
        # (averaged over dt_mhd)
        f = 'net_cool_rate'
        field_dep[f] = set(['net_cool_rate'])
        def _net_cool_rate(d, u):
            return d['net_cool_rate']
        func[f] = _net_cool_rate
        label[f] = r'$\mathcal{L}\;[{\rm erg}\,{\rm cm^{-3}}\,{\rm s}^{-1}]$'
        cmap[f] = 'bwr_r'
        vminmax[f] = (-1e-20,1e-20)
        take_log[f] = False

        # Cooling efficiency [erg*cm^3/s]
        f = 'Lambda_cool'
        field_dep[f] = set(['density','cool_rate'])
        def _Lambda_cool(d, u):
            return d['cool_rate']/d['density']**2
        func[f] = _Lambda_cool
        label[f] = r'$\Lambda\;[{\rm erg}\,{\rm cm^{3}}\,{\rm s}^{-1}]$'
        cmap[f] = 'cubehelix_r'
        vminmax[f] = (1e-30,1e-20)
        take_log[f] = True

        # Specific cooling rate [erg/s/H]
        f = 'nHLambda_cool'
        field_dep[f] = set(['density','cool_rate'])
        def _nHLambda_cool(d, u):
            return d['cool_rate']/d['density']
        func[f] = _nHLambda_cool
        label[f] = r'$n_{\rm H}\Lambda\;[{\rm erg}\,{\rm cm^{3}}\,{\rm s}^{-1}]$'
        cmap[f] = 'cubehelix_r'
        vminmax[f] = (1e-30,1e-20)
        take_log[f] = True

        # Specific net cooling rate [erg/s/H]
        f = 'nHLambda_cool_net'
        field_dep[f] = set(['density','cool_rate','heat_rate'])
        def _nHLambda_cool_net(d, u):
            return (d['cool_rate'] - d['heat_cool'])/d['density']
        func[f] = _nHLambda_cool_net
        label[f] = r'$n_{\rm H}\Lambda_{\rm net}\;[{\rm erg}\,{\rm cm^{3}}\,{\rm s}^{-1}]$'
        cmap[f] = 'cubehelix_r'
        vminmax[f] = (1e-30,1e-20)
        take_log[f] = True

        # Heating efficiency [erg/s/H]
        f = 'Gamma_heat'
        field_dep[f] = set(['density','heat_rate'])
        def _Gamma_heat(d, u):
            return d['heat_rate']/d['density']
        func[f] = _Gamma_heat
        label[f] = r'$\Gamma_{\rm heat}\;[{\rm erg}\,{\rm s}^{-1}]$'
        cmap[f] = 'cubehelix_r'
        vminmax[f] = (1e-30,1e-20)
        take_log[f] = True

        # Cooling time [Myr]
        f = 't_cool'
        field_dep[f] = set(['pressure', 'cool_rate'])
        def _t_cool(d, u):
            return d['pressure']/d['cool_rate']*u.Myr
        func[f] = _t_cool
        label[f] = r'$t_{\rm cool}\;[{\rm yr}]$'
        cmap[f] = 'cubehelix_r'
        vminmax[f] = (1e-4,1e2)
        take_log[f] = True

    return func, field_dep, label, cmap, vminmax, take_log

def set_derived_fields_mag(par, x0):

    func = dict()
    field_dep = dict()
    label = dict()
    cmap = dict()
    vminmax = dict()
    take_log = dict()

    # Magnitude of Alfven velocity [km/s]
    f = 'vAmag'
    field_dep[f] = set(['density', 'cell_centered_B'])
    def _vAmag(d, u):
        return (d['cell_centered_B1']**2 +
                d['cell_centered_B2']**2 +
                d['cell_centered_B3']**2)**0.5*np.sqrt(u.energy_density.cgs.value)\
            /np.sqrt(d['density']*u.density.cgs.value)/1e5

    func[f] = _vAmag
    label[f] = r'$v_A\;[{\rm km}\,{\rm s}^{-1}]$'
    vminmax[f] = (0.1, 1000.0)
    cmap[f] = 'cividis'
    take_log[f] = True

    # vAx [km/s]
    f = 'vAx'
    field_dep[f] = set(['density','cell_centered_B'])
    def _vAx(d, u):
        return d['cell_centered_B1']*np.sqrt(d['density'])*u.kms
    func[f] = _vAx
    label[f] = r'$v_{A,x}\;[{\rm km/s}]$'
    cmap[f] = 'RdBu'
    vminmax[f] = (-1e2,1e2)
    take_log[f] = False

    # vAy [km/s]
    f = 'vAy'
    field_dep[f] = set(['density','cell_centered_B'])
    def _vAy(d, u):
        return d['cell_centered_B2']*np.sqrt(d['density'])*u.kms
    func[f] = _vAy
    label[f] = r'$v_{A,y}\;[{\rm km/s}]$'
    cmap[f] = 'RdBu'
    vminmax[f] = (-1e2,1e2)
    take_log[f] = False

    # vAz [km/s]
    f = 'vAz'
    field_dep[f] = set(['density','cell_centered_B'])
    def _vAz(d, u):
        return d['cell_centered_B3']*np.sqrt(d['density'])*u.kms
    func[f] = _vAz
    label[f] = r'$v_{A,z}\;[{\rm km/s}]$'
    cmap[f] = 'RdBu'
    vminmax[f] = (-1e2,1e2)
    take_log[f] = False

    # Bx [G]
    f = 'Bx'
    field_dep[f] = set(['cell_centered_B'])
    def _Bx(d, u):
        return d['cell_centered_B1']*np.sqrt(u.energy_density.cgs.value)\
            *np.sqrt(4.0*np.pi)*1e6
    func[f] = _Bx
    label[f] = r'$B_{x}\;[\mu{\rm G}]$'
    cmap[f] = 'RdBu'
    vminmax[f] = (-1e2,1e2)
    take_log[f] = False

    # By [G]
    f = 'By'
    field_dep[f] = set(['cell_centered_B'])
    def _By(d, u):
        return d['cell_centered_B2']*np.sqrt(u.energy_density.cgs.value)\
            *np.sqrt(4.0*np.pi)*1e6
    func[f] = _By
    label[f] = r'$B_{y}\;[\mu{\rm G}]$'
    cmap[f] = 'RdBu'
    vminmax[f] = (-1e2,1e2)
    take_log[f] = False

    # Bz [G]
    f = 'Bz'
    field_dep[f] = set(['cell_centered_B'])
    def _Bz(d, u):
        return d['cell_centered_B3']*np.sqrt(u.energy_density.cgs.value)\
            *np.sqrt(4.0*np.pi)*1e6
    func[f] = _Bz
    label[f] = r'$B_{z}\;[\mu{\rm G}]$'
    cmap[f] = 'RdBu'
    vminmax[f] = (-1e2,1e2)
    take_log[f] = False

    # Magnetic fields magnitude [G]
    f = 'Bmag'
    field_dep[f] = set(['cell_centered_B'])
    def _Bmag(d, u):
        return (d['cell_centered_B1']**2 +
                d['cell_centered_B2']**2 +
                d['cell_centered_B3']**2)**0.5*np.sqrt(u.energy_density.cgs.value)\
            *np.sqrt(4.0*np.pi)*1e6
    func[f] = _Bmag
    label[f] = r'$|\mathbf{B}|\;[\mu{\rm G}]$'
    vminmax[f] = (1e-1, 1e2)
    cmap[f] = 'cividis'
    take_log[f] = True

    return func, field_dep, label, cmap, vminmax, take_log

def set_derived_fields_newcool(par, x0):

    try:
        Erad_PE0 = par['cooling']['Erad_PE0']
        Erad_LW0 = par['cooling']['Erad_LW0']
    except KeyError:
        Erad_PE0 = 7.613e-14
        Erad_LW0 = 1.335e-14

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
    cmap[f] = cmap_apply_alpha('Greens')
    vminmax[f] = (1e0,1e4)
    take_log[f] = True

    # 2nH2 [cm^-3] (assume d=nH)
    f = '2nH2'
    field_dep[f] = set(['density', 'xH2'])
    def _2nH2(d, u):
        return 2.0*d['density']*d['xH2']
    func[f] = _2nH2
    label[f] = r'$2n_{\rm H_2}\;[{\rm cm^{-3}}]$'
    cmap[f] = cmap_apply_alpha('Greens')
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
    cmap[f] = cmap_apply_alpha('Blues')
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
    cmap[f] = cmap_apply_alpha('Oranges')
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
    cmap[f] = cmap_apply_alpha('Blues')
    vminmax[f] = (1e-3,1e4)
    take_log[f] = True

    # xn [cm^-3]
    f = 'xn'
    field_dep[f] = set(['xHI', 'xH2'])
    def _xn(d, u):
        return d['xHI'] + 2.0*d['xH2']
    func[f] = _xn
    label[f] = r'$x_{\rm n}$'
    cmap[f] = cmap_apply_alpha('YlGn')
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
    cmap[f] = 'plasma'
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
    try:
        xCtot = par['problem']['Z_gas']*par['cooling']['xCstd']
    except KeyError:
        xCtot = 1.6e-4
        print('xCtot not found. Use {:.1f}.'.format(xCtot))
    f = 'xCI'
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
    field_dep[f] = set(['density','xCI_over_xCtot'])
    def _nCI(d, u):
        return d['density']*xCtot*np.maximum(0.0,np.minimum(1.0,d['xCI_over_xCtot']))
    func[f] = _nCI
    label[f] = r'$x_{\rm CI}$'
    cmap[f] = 'viridis'
    vminmax[f] = (1e2*xCtot,1e4*xCtot)
    take_log[f] = True

    # xOII - single ionized oxygen
    f = 'xOII'
    try:
        xOtot = par['problem']['Z_gas']*par['cooling']['xOstd']
    except KeyError:
        xOtot = 3.2e-4*par['problem']['Z_gas']
    field_dep[f] = set(['xH2','xHI'])
    def _xOII(d, u):
        return xOtot*(1.0 - d['xHI'] - 2.0*d['xH2'])
    func[f] = _xOII
    label[f] = r'$x_{\rm OII}$'
    cmap[f] = 'viridis'
    vminmax[f] = (0,xOtot)
    take_log[f] = False

    # xCII - singly ionized carbon
    # Use with caution.
    # (Do not apply to hot gas and depend on cooling implementation)
    f = 'xCII'
    try:
        xCtot = par['problem']['Z_gas']*par['cooling']['xCstd']
        xOtot = par['problem']['Z_gas']*par['cooling']['xOstd']
    except KeyError:
        xCtot = 1.6e-4*par['problem']['Z_gas']
        xOtot = 3.2e-4*par['problem']['Z_gas']

    field_dep[f] = set(['xe','xH2','xHI','pressure','density','CR_ionization_rate'])
    def _xCII(d, u):
        d['T'] = d['pressure']/(d['density']*(1.1 + d['xe'] - d['xH2']))/\
            (ac.k_B/u.energy_density).cgs.value
        xe_mol = get_xe_mol(d['density'],d['xH2'],d['xe'],d['T'],d['CR_ionization_rate'],
                            par['problem']['Z_gas'],par['problem']['Z_dust'])
        # Apply floor and ceiling
        return np.maximum(0.0,np.minimum(xCtot,
                    d['xe'] - (1.0 - d['xHI'] - 2.0*d['xH2'])*(1.0 + xOtot) - xe_mol))
    func[f] = _xCII
    label[f] = r'$x_{\rm CII}$'
    cmap[f] = 'viridis'
    vminmax[f] = (0,xCtot)
    take_log[f] = False

    # xCII - singlly ionized carbon (use when CR ionization is uniform everywhere)
    # Use with caution.
    # (Do not apply to hot gas and depend on cooling implementation)
    f = 'xCII_alt'
    try:
        xCtot = par['problem']['Z_gas']*par['cooling']['xCstd']
        xOtot = par['problem']['Z_gas']*par['cooling']['xOstd']
    except KeyError:
        xCtot = 1.6e-4*par['problem']['Z_gas']
        xOtot = 3.2e-4*par['problem']['Z_gas']

    field_dep[f] = set(['xe','xH2','xHI','pressure','density'])
    def _xCII_alt(d, u):
        d['T'] = d['pressure']/(d['density']*(1.1 + d['xe'] - d['xH2']))/\
            (ac.k_B/u.energy_density).cgs.value
        xe_mol = get_xe_mol(d['density'],d['xH2'],d['xe'],d['T'],par['problem']['xi_CR0'],
                            par['problem']['Z_gas'],par['problem']['Z_dust'])
        # Apply floor and ceiling
        return np.maximum(0.0,np.minimum(xCtot,
                    d['xe'] - (1.0 - d['xHI'] - 2.0*d['xH2'])*(1.0 + xOtot) - xe_mol))
    func[f] = _xCII_alt
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
    vminmax[f] = (1e-17,1e-15)
    take_log[f] = True

    # T_alt [K]
    f = 'T_alt'
    field_dep[f] = set(['pressure','density','xe','xH2'])
    def _T_alt(d, u):
        return d['pressure']*(u.energy_density/ac.k_B).cgs.value/(d['density']*(1.1 + d['xe'] - d['xH2']))
    func[f] = _T_alt
    label[f] = r'$T_{\rm alt}\;[{\rm K}]$'
    cmap[f] = cmap_shift(mpl.cm.RdYlBu_r, midpoint=3./7., name='cmap_pyathena_T')
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

    # Normalized FUV radiation field strength (Draine field unit)
    f = 'chi_FUV_ext'
    field_dep[f] = set(['rad_energy_density_PE_ext','rad_energy_density_LW_ext'])
    def _chi_FUV_ext(d, u):
        return (d['rad_energy_density_PE_ext'] + d['rad_energy_density_LW_ext'])*\
            (u.energy_density.cgs.value/(Erad_PE0 + Erad_LW0))
    func[f] = _chi_FUV_ext
    label[f] = r'$\chi_{\rm FUV,ext}$'
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

    # Normalized PE radiation field strength (Draine field unit)
    f = 'chi_PE'
    field_dep[f] = set(['rad_energy_density_PE'])
    def _chi_PE(d, u):
        return d['rad_energy_density_PE']*(u.energy_density.cgs.value/Erad_PE0)
    func[f] = _chi_PE
    label[f] = r'$\chi_{\rm PE}$'
    cmap[f] = 'viridis'
    vminmax[f] = (1e-4,1e4)
    take_log[f] = True

    # Normalized LW radiation field strength (Draine field unit)
    f = 'chi_LW'
    field_dep[f] = set(['rad_energy_density_LW'])
    def _chi_LW(d, u):
        return d['rad_energy_density_LW']*(u.energy_density.cgs.value/Erad_LW0)
    func[f] = _chi_LW
    label[f] = r'$\chi_{\rm LW}$'
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

    try:
        if (par['radps']['iPhotIon'] == 1):
            iPhot = True
        else:
            iPhot = False
    except KeyError:
        iPhot = True

    if iPhot:
        # Radiation energy density of ionizing radiation in cgs units
        f = 'Erad_LyC'
        field_dep[f] = set(['rad_energy_density_PH'])
        def _Erad_LyC(d, u):
            return d['rad_energy_density_PH']*u.energy_density.cgs.value
        func[f] = _Erad_LyC
        label[f] = r'$\mathcal{E}_{\rm LyC}\;[{\rm erg\,{\rm cm}^{-3}}]$'
        cmap[f] = 'viridis'
        vminmax[f] = (5e-16,5e-11)
        take_log[f] = True

        # Ionizing photon number flux in cgs units [number / cm^2 / s]
        f = 'Jphot_LyC'
        field_dep[f] = set(['rad_energy_density_PH'])
        def _Jphot_LyC(d, u):
            hnu_LyC = (par['radps']['hnu_PH']*au.eV).cgs.value
            return d['rad_energy_density_PH']*u.energy_density.cgs.value\
                /hnu_LyC*ac.c.cgs.value/(4.0*np.pi)
        func[f] = _Jphot_LyC
        label[f] = r'$J_{\rm LyC}^{*}\;[{\rm phot}\,{\rm cm}^{-2}\,{\rm s}^{-1}\,{\rm sr}^{-1}]$'
        cmap[f] = 'viridis'
        vminmax[f] = (1e2,1e8)
        take_log[f] = True

        # Dimensionless ionization parameter Uion = Erad_LyC/(hnu_LyC*nH)
        f = 'Uion'
        field_dep[f] = set(['density','rad_energy_density_PH'])
        def _Uion(d, u):
            return d['rad_energy_density_PH']*u.energy_density.cgs.value/ \
                    ((par['radps']['hnu_PH']*au.eV).cgs.value*d['density'])
        func[f] = _Uion
        label[f] = r'$\mathcal{U}_{\rm ion}$'
        cmap[f] = 'cubehelix'
        vminmax[f] = (1e-5,1e2)
        take_log[f] = True

    # Halpha emissivity [erg/s/cm^-3/sr]
    # Caution: Draine (2011)'s alpha_eff_Halpha valid for ~1000 K < T < ~30000 K
    # Better to use this for warm gas only
    f = 'j_Halpha'
    field_dep[f] = set(['density', 'pressure', 'xe', 'xHI', 'xH2'])
    def _j_Halpha(d, u):
        hnu_Halpha = (ac.h*ac.c/(6562.8*au.angstrom)).to('erg')
        alpha_eff_Halpha = lambda T: 1.17e-13*(T*1e-4)**(-0.942-0.031*np.log(T*1e-4))
        # j_Halpha = nHII*ne*alpha_eff_Halpha*hnu_Halpha/(4pi)
        return d['density']**2*(1.0 - d['xHI'] - d['xH2'])*d['xe']*\
            alpha_eff_Halpha(d['pressure']/(d['density']*(1.1 + d['xe'] - d['xH2']))/\
                (ac.k_B/u.energy_density).cgs.value)*hnu_Halpha/(4.0*np.pi)
    func[f] = _j_Halpha
    label[f] = r'$\mathcal{j}_{\rm H\alpha}\;[{\rm erg}\,{\rm cm}^{-3}\,{\rm sr}^{-1}]$'
    cmap[f] = 'plasma'
    vminmax[f] = (1e-22,1e-30)
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

    # Heating rate by H photoionization
    f = 'heat_rate_HI_phot'
    field_dep[f] = set(['density','rad_energy_density_PH','xHI'])
    def _heat_rate_HI_phot(d, u):
        if 'dhnu_HI_PH' in par['radps']:
            dhnu_HI_PH = par['radps']['dhnu_HI_PH']*(1.0*au.eV).cgs.value
        else:
            dhnu_HI_PH = (par['radps']['hnu_PH'] - 13.6)*(1.0*au.eV).cgs.value

        sigma_HI_PH = par['opacity']['sigma_HI_PH']
        hnu_PH = par['radps']['hnu_PH']*(1.0*au.eV).cgs.value
        xi_ph_HI = d['rad_energy_density_PH']*u.energy_density.cgs.value*ac.c.cgs.value/hnu_PH*sigma_HI_PH
        return d['density']*d['xHI']*xi_ph_HI*dhnu_HI_PH

    func[f] = _heat_rate_HI_phot
    label[f] = r'$\mathcal{H}_{\rm pi,H}\;[{\rm erg}\,{\rm cm}^{-3}\,{\rm s}^{-1}]$'
    cmap[f] = 'copper'
    vminmax[f] = (1e-28,1e-19)
    take_log[f] = True

    # Heating rate by H photoionization
    f = 'heat_rate_H2_phot'
    field_dep[f] = set(['density','rad_energy_density_PH','xH2'])
    def _heat_rate_H2_phot(d, u):
        if 'dhnu_H2_PH' in par['radps']:
            dhnu_H2_PH = par['radps']['dhnu_H2_PH']*(1.0*au.eV).cgs.value
        else:
            dhnu_H2_PH = (par['radps']['hnu_PH'] - 15.4)*(1.0*au.eV).cgs.value

        sigma_H2_PH = par['opacity']['sigma_H2_PH']
        hnu_PH = par['radps']['hnu_PH']*(1.0*au.eV).cgs.value
        xi_ph_H2 = d['rad_energy_density_PH']*u.energy_density.cgs.value*ac.c.cgs.value/hnu_PH*sigma_H2_PH
        return d['density']*d['xH2']*xi_ph_H2*dhnu_H2_PH

    func[f] = _heat_rate_H2_phot
    label[f] = r'$\mathcal{H}_{\rm pi,H_2}\;[{\rm erg}\,{\rm cm}^{-3}\,{\rm s}^{-1}]$'
    cmap[f] = 'copper'
    vminmax[f] = (1e-28,1e-19)
    take_log[f] = True

    # Volumetric absorption rate of LyC radiation by dust (erg/cm^3/s)
    f = 'heat_rate_dust_LyC'
    field_dep[f] = set(['density','rad_energy_density_PH'])
    def _heat_rate_dust_LyC(d, u):
        conv = u.energy_density.cgs.value*ac.c.cgs.value
        return d['density']*(d['rad_energy_density_PH']*par['opacity']['sigma_dust_PH0']*par['problem']['Z_dust'])*conv

    func[f] = _heat_rate_dust_LyC
    label[f] = r'$\mathcal{H}_{\rm d,LyC}\;[{\rm erg}\,{\rm cm}^{-3}\,{\rm s}^{-1}]$'
    cmap[f] = 'copper'
    vminmax[f] = (1e-28,1e-19)
    take_log[f] = True

    # Volumetric absorption rate of FUV radiation by dust (erg/cm^3/s)
    f = 'heat_rate_dust_FUV'
    field_dep[f] = set(['density','rad_energy_density_PE','rad_energy_density_LW'])
    def _heat_rate_dust_FUV(d, u):
        conv = u.energy_density.cgs.value*ac.c.cgs.value
        return d['density']*(d['rad_energy_density_PE']*par['opacity']['sigma_dust_PE0']*par['problem']['Z_dust'] +
                             d['rad_energy_density_LW']*par['opacity']['sigma_dust_LW0']*par['problem']['Z_dust'])*conv

    func[f] = _heat_rate_dust_FUV
    label[f] = r'$\mathcal{H}_{\rm d,FUV}\;[{\rm erg}\,{\rm cm}^{-3}\,{\rm s}^{-1}]$'
    cmap[f] = 'copper'
    vminmax[f] = (1e-28,1e-19)
    take_log[f] = True

    # Volumetric absorption rate of UV radiation by dust (erg/cm^3/s)
    f = 'heat_rate_dust_UV'
    field_dep[f] = set(['density','rad_energy_density_PE','rad_energy_density_LW','rad_energy_density_PH'])
    def _heat_rate_dust_UV(d, u):
        conv = u.energy_density.cgs.value*ac.c.cgs.value
        return d['density']*(d['rad_energy_density_PE']*par['opacity']['sigma_dust_PE0']*par['problem']['Z_dust'] +
                             d['rad_energy_density_LW']*par['opacity']['sigma_dust_LW0']*par['problem']['Z_dust'] +
                             d['rad_energy_density_PH']*par['opacity']['sigma_dust_PH0']*par['problem']['Z_dust'])*conv

    func[f] = _heat_rate_dust_UV
    label[f] = r'$\mathcal{H}_{\rm d,UV}\;[{\rm erg}\,{\rm cm}^{-3}\,{\rm s}^{-1}]$'
    cmap[f] = 'copper'
    vminmax[f] = (1e-28,1e-19)
    take_log[f] = True

    # Grain charge parameter
    f = 'psi_gr'
    field_dep[f] = set(['density','pressure','xe','xH2',
                        'rad_energy_density_LW','rad_energy_density_PE'])
    def _psi_gr(d, u):
        G0 = (d['rad_energy_density_PE']*u.energy_density.cgs.value/Erad_PE0 +
              d['rad_energy_density_LW']*u.energy_density.cgs.value/Erad_LW0)/1.7
        T = d['pressure']/(d['density']*(1.1 + d['xe'] - d['xH2']))/\
                    (ac.k_B/u.energy_density).cgs.value
        return G0*T**0.5/(d['density']*d['xe']) + 50.0 # add a floor

    func[f] = _psi_gr
    label[f] = r'$\psi_{\rm gr}\;[{\rm cm}^{3}\,{\rm K}^{1/2}]$'
    cmap[f] = 'viridis'
    vminmax[f] = (1,1000)
    take_log[f] = True

    # PE heating efficiency
    f = 'eps_pe'
    field_dep[f] = set(['density','pressure','xe','xH2',
                        'rad_energy_density_LW','rad_energy_density_PE'])
    def _eps_pe(d, u):
        CPE_ = np.array([5.22, 2.25, 0.04996, 0.00430, 0.147, 0.431, 0.692])
        conv = u.energy_density.cgs.value*ac.c.cgs.value
        T = d['pressure']/(d['density']*(1.1 + d['xe'] - d['xH2']))/\
            (ac.k_B/u.energy_density).cgs.value
        chi_FUV = (d['rad_energy_density_PE']*u.energy_density.cgs.value/Erad_PE0 +
                   d['rad_energy_density_LW']*u.energy_density.cgs.value/Erad_LW0)
        G0 = chi_FUV*1.7 # Habing field
        # Grain charging
        x = G0*T**0.5/(d['density']*d['xe']) + 50.0 # add a floor
        eps = (CPE_[0] + CPE_[1]*np.power(T, CPE_[4]))/ \
            (1. + CPE_[2]*np.power(x, CPE_[5])*(1. + CPE_[3]*np.power(x, CPE_[6])))
        # PE heating rate
        Gamma_pe = 1.7e-26*chi_FUV*par['problem']['Z_dust']*eps
        # Dust heating rate
        Gamma_dust_FUV = (d['rad_energy_density_PE']*par['opacity']['sigma_dust_PE0']*par['problem']['Z_dust'] +
                          d['rad_energy_density_LW']*par['opacity']['sigma_dust_LW0']*par['problem']['Z_dust'])*conv
        return Gamma_pe/Gamma_dust_FUV

    func[f] = _eps_pe
    label[f] = r'$\epsilon_{\rm pe}$'
    cmap[f] = 'viridis'
    vminmax[f] = (1.0e-3,1)
    take_log[f] = True

    # PE heating rate
    f = 'Gamma_pe'
    field_dep[f] = set(['density','pressure','xe','xH2',
                        'rad_energy_density_LW','rad_energy_density_PE'])
    def _eps_PE(d, u):
        CPE_ = np.array([5.22, 2.25, 0.04996, 0.00430, 0.147, 0.431, 0.692])
        G0 = (d['rad_energy_density_PE']*u.energy_density.cgs.value/Erad_PE0 +
              d['rad_energy_density_LW']*u.energy_density.cgs.value/Erad_LW0)*1.7
        T = d['pressure']/(d['density']*(1.1 + d['xe'] - d['xH2']))/\
                    (ac.k_B/u.energy_density).cgs.value
        # Grain charging
        x = G0*T**0.5/(d['density']*d['xe']) + 50.0 # add a floor
        return (CPE_[0] + CPE_[1]*np.power(T, CPE_[4]))/ \
            (1. + CPE_[2]*np.power(x, CPE_[5])*(1. + CPE_[3]*np.power(x, CPE_[6])))

    func[f] = _eps_PE
    label[f] = r'$\epsilon_{\rm PE}$'
    cmap[f] = 'viridis'
    vminmax[f] = (1.0e-3,1)
    take_log[f] = True

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

def set_derived_fields_wind(par, x0):

    func = dict()
    field_dep = dict()
    label = dict()
    cmap = dict()
    vminmax = dict()
    take_log = dict()

    # Wind mass fraction
    f = 'fwind'
    field_dep[f] = set(['specific_scalar[1]'])
    def _fwind(d, u):
        return d['specific_scalar[1]']
    func[f] = _fwind
    label[f] = r'$f_{\rm wind}$'
    cmap[f] = cmap_apply_alpha('Greens')
    vminmax[f] = (1e-6,1)
    take_log[f] = True

    # Wind mass density
    f = 'swind'
    field_dep[f] = set(['density', 'specific_scalar[1]'])
    def _swind(d, u):
        return d['density']*d['specific_scalar[1]']
    func[f] = _swind
    label[f] = r'$\rho_{\rm wind}$'
    cmap[f] = cmap_apply_alpha('Greens')
    vminmax[f] = (1e-4,1e3)
    take_log[f] = True

    return func, field_dep, label, cmap, vminmax, take_log

def set_derived_fields_xray(par, x0, newcool):

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
    if newcool:
        field_dep[f] = set(['density','pressure','xe','xH2'])
    else:
        field_dep[f] = set(['density','temperature'])
    # Frequency integrated volume emissivity
    def _j_Xray(d, u):
        if newcool:
            d['temperature'] = d['pressure']/(d['density']*(1.1 + d['xe'] - d['xH2']))/\
                               (ac.k_B/u.energy_density).cgs.value
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

        try:
            if par['configure']['new_cooling'] == 'ON':
                newcool = True
            else:
                newcool = False
        except KeyError:
            newcool = False

        self.func, self.field_dep, \
            self.label, self.cmap, \
            self.vminmax, self.take_log = set_derived_fields_def(par, x0, newcool)

        dicts = (self.func, self.field_dep, self.label, self.cmap, \
                 self.vminmax, self.take_log)

        if par['configure']['gas'] == 'mhd':
            dicts_ = set_derived_fields_mag(par, x0)
            for d, d_ in zip(dicts, dicts_):
                d = d.update(d_)

        if newcool:
            dicts_ = set_derived_fields_newcool(par, x0)
            for d, d_ in zip(dicts, dicts_):
                d = d.update(d_)

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

        try:
            if par['feedback']['iWind'] != 0:
                dicts_ = set_derived_fields_wind(par, x0)
                for d, d_ in zip(dicts, dicts_):
                    d = d.update(d_)
        except KeyError:
            pass

        # Add X-ray emissivity if Wind or SN is turned on
        try:
            if par['feedback']['iSN'] > 0 or par['feedback']['iWind'] > 0 or \
               par['feedback']['iEarly'] > 0:
                dicts_ = set_derived_fields_xray(par, x0, newcool)
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
