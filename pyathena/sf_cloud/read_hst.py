# read_hst.py

import os
import numpy as np
import pandas as pd
import astropy.constants as ac
import astropy.units as au
from scipy import integrate

from ..io.read_hst import read_hst
from ..load_sim import LoadSim
from ..util.derivative import deriv_convolve

class ReadHst:

    @LoadSim.Decorators.check_pickle_hst
    def read_hst(self, savdir=None, force_override=False):
        """Function to read hst and convert quantities to convenient units
        """

        hst = read_hst(self.files['hst'], force_override=force_override)

        par = self.par
        u = self.u
        domain = self.domain
        # volume of resolution element (code unit)
        dvol = domain['dx'].prod()
        # total volume of domain (code unit)
        vol = domain['Lx'].prod()        

        # Rename column names
        hst = hst.rename(columns={"mass": "Mgas",     # total gas mass
                                  "mass_sp": "Mstar", # star particle mass in the box
                                  "mass_sp_esc": "Mstar_esc"})

        # Time in code unit
        hst['time_code'] = hst['time']
        # Time in Myr
        hst['time'] *= u.Myr
        # Time step
        hst['dt_code'] = hst['dt']
        hst['dt'] *= u.Myr
        if par['configure']['new_cooling'] == 'ON' and \
           (par['configure']['radps'] == 'ON' or par['configure']['sixray'] == 'ON'):
            for c in ('dt_cool_min','dt_xH2_min','dt_xHII_min'):
                hst[c] *= u.Myr*vol
        
        # Mass of (gas, gas, starpar, cold/intermediate/warm/hot temperature gas,
        #          molecular,atomic,ionized) in Msun
        for c in ('Mgas','Mcold','Minter','Mwarm','Mhot',
                  'MH2','MHI','MH2_cl','MHI_cl','M_cl',
                  'Mstar','Mstar_esc'):
            try:
                hst[c] *= vol*u.Msun
            except KeyError:
                self.logger.warning('Column {0:s} not found'.format(c))
                continue

        hst = self._calc_SFR(hst)
        hst = self._calc_outflow(hst)
            
        # Mstar: total
        # Mstar_in: mass of sp currently in the domain
        # Mstar_esc: mass of sp escaped the domain
        if 'Mstar' in hst.columns and 'Mstar_esc' in hst.columns:
            hst['Mstar_in'] = hst['Mstar']
            hst['Mstar'] += hst['Mstar_esc']

        hst['MHII'] = hst['Mgas'] - hst['MHI'] - hst['MH2']
        hst['MHII_cl'] = hst['Mgas'] - hst['MHI_cl'] - hst['MH2_cl']

        # Volume
        hst['VHII'] = 1.0 - hst['VHI'] - hst['VH2']
        hst['VHII_cl'] = hst['V_cl'] - hst['VHI_cl'] - hst['VH2_cl']
        
        # Radiation feedback turned on
        if par['configure']['radps'] == 'ON':
            from scipy.integrate import cumtrapz

            # Total/escaping luminosity in Lsun
            ifreq = dict()
            for f in ('PH','LW','PE','PE_unatt'):
                try:
                    ifreq[f] = par['radps']['ifreq_{0:s}'.format(f)]
                except KeyError:
                    pass

            for i in range(par['radps']['nfreq']):
                for k, v in ifreq.items():
                    if i == v:
                        try:
                            hst[f'Ltot_{k}'] = hst[f'Ltot{i}']*vol*u.Lsun
                            hst[f'Lesc_{k}'] = hst[f'Lesc{i}']*vol*u.Lsun
                            hnu = (par['radps'][f'hnu_{k}']*au.eV).cgs.value
                            hst[f'Qtot_{k}'] = hst[f'Ltot_{k}'].values * \
                                               (ac.L_sun.cgs.value)/hnu
                            hst[f'fesc_{k}'] = hst[f'Lesc_{k}']/hst[f'Ltot_{k}']
                        except KeyError:
                            pass
            
        #hst.index = hst['time_code']
        
        self.hst = hst
        
        return hst

    def _calc_outflow(self, hst):

        
        u = self.u
        domain = self.domain
        vol = domain['Lx'].prod()

        # Obtain ejected mass by integrating mass flux
        if 'rho_out' in hst.columns:
            hst.rho_out *= vol*u.Msun
            hst['Mof_dot'] = hst.rho_out
            hst['Mof'] = integrate.cumtrapz(
                hst['rho_out'], hst['time'], initial=0.0)
            
        return hst
    def _calc_SFR(self, hst):
        """Compute instantaneous SFR, SFR averaged over the past 1 Myr, 3Myr, etc.
        """        

        # Instantaneous SFR
        hst['SFR'] = deriv_convolve(hst['Mstar'].values, hst['time'].values,
                                    gauss=True, fft=False, stddev=3.0)
        
        # Set any negative values to zero
        hst['SFR'][hst['SFR'] < 0.0] = 0.0

        if hst.time.max() > 1.0:
            hst_ = hst[hst.time < 1.0]
            winsize_1Myr = hst_.index.size
            winsize_3Myr = 3*winsize_1Myr
            hst['SFR_1Myr'] = hst.SFR.rolling(
                winsize_1Myr, min_periods=1, win_type='boxcar').mean()
            hst['SFR_3Myr'] = hst.SFR.rolling(
                winsize_3Myr, min_periods=1, win_type='boxcar').mean()
        else:
            raise ValueError('Total time interval smaller than 1 Myr')

        return hst

