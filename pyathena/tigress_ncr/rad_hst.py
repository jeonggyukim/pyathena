import pandas as pd
import numpy as np
import astropy.units as au
import astropy.constants as ac
from scipy.integrate import cumtrapz

from ..io.read_hst import read_hst
from ..decorators.decorators import check_pickle_hst

class ReadHstRadiation:

    @check_pickle_hst
    def read_hst_rad(self, prefix='hst_rad', savdir=None, force_override=False):

        par = self.par
        u = self.u
        domain = self.domain

        # Area of domain [pc^2]
        LxLy = self.domain['Lx'][0]*self.domain['Lx'][1]*u.pc**2
        # Volume of domain [code]
        vol = domain['Lx'].prod()

        hnu_LyC = (self.par['radps']['hnu_PH']*au.eV).cgs.value
        hst = read_hst(self.files['hst'])

        h = pd.DataFrame()
        # Time in code unit
        h['time'] = hst['time']
        # Time in Myr
        h['tMyr'] = h['time']*u.Myr
        # Time step
        h['dt_code'] = hst['dt']
        h['dtMyr'] = hst['dt']*u.Myr

        # Total gas mass [Msun]
        h['mass'] = hst['mass']*vol*u.Msun
        h['mass_sp'] = hst['msp']*vol*u.Msun

        # Mass surface density in Msun/pc^2
        h['Sigma_gas'] = h['mass']/LxLy
        h['Sigma_sp'] = h['mass_sp']/LxLy

        h['Ltot_LyC'] = hst['Ltot0']*u.Lsun
        h['Lesc_LyC'] = (hst['Lpp0'] + hst['Lesc0'] + hst['Lxymax0'])*u.Lsun

        h['Ltot_FUV'] = hst['Ltot1']*u.Lsun
        h['Lesc_FUV'] = (hst['Lpp1'] + hst['Lesc1'] + hst['Lxymax1'] + hst['Leps1'] +\
                         hst['Lpp2'] + hst['Lesc2'] + hst['Lxymax2'] + hst['Leps2'])\
                         *u.Lsun

        h['fesc_LyC'] = (hst['Lpp0'] + hst['Lesc0'] + hst['Lxymax0'])/hst['Ltot0']
        h['fesc_FUV'] = (hst['Lpp1'] + hst['Lesc1'] + hst['Lxymax1'] + hst['Leps1'] +\
                         hst['Lpp2'] + hst['Lesc2'] + hst['Lxymax2'] + hst['Leps2'])/\
                         (hst['Ltot1'] + hst['Ltot2'])

        # Luminosity per area [Lsun/pc^2]
        h['Sigma_LyC'] = hst['Ltot0']*u.Lsun/LxLy
        h['Sigma_FUV'] = (hst['Ltot1'] + hst['Ltot2'])*u.Lsun/LxLy

        # Ionizing photon rate per area [#/s/kpc^2]
        h['Phi_LyC'] = h['Sigma_LyC']*(1.0*au.Lsun).cgs.value/hnu_LyC/u.kpc**2

        # Cumulative escape fraction
        h['fesc_LyC_cumul'] = cumtrapz(h['Lesc_LyC'], h['tMyr'], initial=0.0)/\
            cumtrapz(h['Ltot_LyC'], h['tMyr'], initial=0.0)
        # Cumulative escape fraction
        h['fesc_FUV_cumul'] = cumtrapz(h['Lesc_FUV'], h['tMyr'], initial=0.0)/\
            cumtrapz(h['Ltot_FUV'], h['tMyr'], initial=0.0)

        h = h.replace([-np.inf, np.inf], 0.0)

        return h
