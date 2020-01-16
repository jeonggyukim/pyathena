from functools import wraps

import matplotlib as mpl
import numpy as np
import xarray as xr
import astropy.constants as ac
import astropy.units as au

from matplotlib.colors import Normalize, LogNorm

from ..plt_tools.cmap_shift import cmap_shift
from ..plt_tools.cmap_custom import get_my_cmap

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
        return (d['velocity1']**2 + d['velocity2']**2 + d['velocity3']**2)**0.5
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
        return (x*d['velocity1'] + y*d['velocity2'] + z*d['velocity3'])/r
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
    f = 'vz'
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
    
    # Cooling
    if par['configure']['cooling'] == 'ON':
        # T [K]
        f = 'T'
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

    # New cooling
    if par['configure']['new_cooling'] == 'ON':

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

        # ne
        f = 'ne'
        field_dep[f] = set(['density','xe'])
        def _ne(d, u):
            return d['density']*d['xe']
        func[f] = _ne
        label[f] = r'$n_{\rm e}$'
        cmap[f] = 'viridis'
        vminmax[f] = (1e-6, 1e3)
        take_log[f] = False

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


    if par['configure']['radps'] == 'ON':
        # Dust PE opacity for Z'=1
        # kappa_dust_PE_def = 418.7

        # Normalized FUV radiation field strength (Draine field unit)
        f = 'chi_PE'
        field_dep[f] = set(['rad_energy_density_PE'])
        def _chi_PE(d, u):
            Erad_PE_ISRF = 8.9401e-14
            return d['rad_energy_density_PE']*u.energy_density.cgs.value/Erad_PE_ISRF
        func[f] = _chi_PE
        label[f] = r'$\chi_{\rm PE}$'
        cmap[f] = 'viridis'
        vminmax[f] = (1e-4,1e4)
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
            return 1e21/par['radps']['kappa_dust_PE']*418.7*np.log(d['rad_energy_density_PE_unatt']/d['rad_energy_density_PE'])
        func[f] = _NHeff
        label[f] = r'$N_{\rm H,eff}\;[{\rm cm^{-2}}]$'
        cmap[f] = 'gist_earth'
        vminmax[f] = (0,1e22)
        take_log[f] = False

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
                # Erad_LW_ISRF = ((6.9e-4/au.cm**3)*(par['radps']['hnu_LW']*au.eV)).to('erg/cm**3').value
                Erad_LW_ISRF = 1.33e-14
                return d['rad_energy_density_LW']*u.energy_density.cgs.value/Erad_LW_ISRF
            func[f] = _chi_LW
            label[f] = r'$\chi_{\rm LW}$'
            cmap[f] = 'viridis'
            vminmax[f] = (1e-4,1e4)
            take_log[f] = True

            # Normalized LW intensity (attenuated by dust only)
            f = 'chi_LW_dust'
            field_dep[f] = set(['rad_energy_density_LW_dust'])
            def _chi_LW_dust(d, u):
                return d['rad_energy_density_LW_dust']*u.energy_density.cgs.value/1.3266e-14
            func[f] = _chi_LW_dust
            label[f] = r'$\chi_{\rm LW,d}$'
            cmap[f] = 'viridis'
            vminmax[f] = (1e-8,1e4)
            take_log[f] = True

            # fshld_H2 (effective)
            f = 'fshld_H2'
            field_dep[f] = set(['rad_energy_density_LW','rad_energy_density_LW_dust'])
            def _fshld_H2(d, u):
                return d['rad_energy_density_LW']/d['rad_energy_density_LW_dust']
            func[f] = _fshld_H2
            label[f] = r'$f_{\rm shld,H2}$'
            cmap[f] = 'tab20c'
            vminmax[f] = (1e-6,1e0)
            take_log[f] = True

    return func, field_dep, label, cmap, vminmax, take_log

class DerivedFields(object):

    def __init__(self, par, x0=np.array([0.0,0.0,0.0])):

        # Create a dictionary containing all information about derived fields
        self.dfi = dict()
        self.func, self.field_dep, \
            self.label, self.cmap, \
            self.vminmax, self.take_log = set_derived_fields(par, x0)
        
        self.derived_field_list = self.func

        # Set colormap normalization and scale
        self.norm = dict()
        self.scale = dict()
        for f in self.derived_field_list:
            if self.take_log[f]:
                self.norm[f] = LogNorm(*self.vminmax[f])
                self.scale[f] = 'log'
            else:
                self.norm[f] = Normalize(*self.vminmax[f])
                self.scale[f] = 'linear'
        
        for f in self.derived_field_list:
            self.dfi[f] = dict(field_dep=self.field_dep[f],
                               func=self.func[f],
                               label=self.label[f],
                               norm=self.norm[f],
                               vminmax=self.vminmax[f],
                               cmap=self.cmap[f],
                               scale=self.scale[f],
                               take_log=self.take_log[f])
