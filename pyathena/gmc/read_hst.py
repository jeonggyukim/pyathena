# read_hst.py

import os
import numpy as np
import pandas as pd
import astropy.constants as ac
import astropy.units as au

from ..io.read_hst import read_hst
from ..load_sim import LoadSim

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

        # Time in code unit
        hst['time_code'] = hst['time']
        # Time in Myr
        hst['time'] *= u.Myr
        # Time step
        hst['dt_code'] = hst['dt']
        hst['dt'] *= u.Myr
        if par['configure']['new_cooling'] == 'ON' and par['configure']['radps'] == 'ON':
            for c in ('dt_cool_min','dt_xH2_min','dt_xHII_min'):
                hst[c] *= u.Myr*vol
        
        # Mass of (gas, gas, starpar, cold/intermediate/warm/hot temperature gas,
        #          molecular,atomic,ionized) in Msun
        for c in ('mass','mass_gas','mass_sp','Mcold','Minter','Mwarm','Mhot','mass_sp_esc'):
            try:
                hst[c] *= vol*u.Msun
            except KeyError:
                continue

        # mass_sp_in: mass of sp currently in the domain
        # mass_sp_esc: mass of sp escaped the domain
        # mass_sp: total
        if 'mass_sp_esc' in hst.columns and 'mass_sp' in hst.columns:
            hst['mass_sp_in'] = hst['mass_sp']
            hst['mass_sp'] += hst['mass_sp_esc']
 
        if par['configure']['new_cooling'] == 'ON':
            for c in ('MH2','MHI','MHII'):
                hst[c] *= vol*u.Msun

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
                        except KeyError:
                            pass
            
        #hst.index = hst['time_code']
        
        self.hst = hst
        
        return hst
