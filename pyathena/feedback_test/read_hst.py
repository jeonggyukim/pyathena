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
            for f in ('dt_cool_min','dt_xH2_min','dt_xHII_min'):
                hst[f] *= u.Myr*vol
        
        # Shell formation time for a single SN in Myr
        # (Eq.7 in Kim & Ostriker 2015)
        tsf = 4.4e-2*par['problem']['n0']**-0.55
        hst['time_tsf'] = hst['time']/tsf
        hst['tsf'] = tsf

        # Shell formation time for wind goes here..
        
        # Total gas mass in Msun
        hst['mass'] *= vol*u.Msun

        # Mass weighted SNR position in pc
        hst['Rsh'] = hst['Rsh_den']/hst['Msh']*u.pc
        # shell mass in Msun
        hst['Msh'] *= u.Msun*vol
        # hot gas mass in Msun
        hst['Mhot'] *= u.Msun*vol
        # warm gas mass in Msun
        hst['Mwarm'] *= u.Msun*vol
        # intermediate temperature gas in Msun
        hst['Minter'] *= u.Msun*vol
        # cold gas mass in Msun
        hst['Mcold'] *= u.Msun*vol

        # Total/hot gas/shell momentum in Msun*km/s
        hst['pr'] *= vol*(u.mass*u.velocity).value
        hst['pr_hot'] *= vol*(u.mass*u.velocity).value
        hst['pr_sh'] *= vol*(u.mass*u.velocity).value

        hst['vr_sh'] = hst['pr_sh']/hst['Msh']
        
        # Radiation feedback turned on
        if par['configure']['radps'] == 'ON':
            from scipy.integrate import cumtrapz

            # Expected radial momentum injection
            if par['radps']['apply_force'] == 1 and par['configure']['ionrad']:
                # Radiation pressure only (in the optically-thick limit)
                hst['pr_inject'] = cumtrapz(hst['Ltot0']/ac.c.to(u.velocity).value,
                                            hst['time_code'], initial=0.0)
                hst['pr_inject'] *= vol*(u.mass*u.velocity).value

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
            
        hst.index = hst['time_code']
        
        self.hst = hst
        
        return hst
