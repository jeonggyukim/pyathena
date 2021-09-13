# hst.py

import os
import numpy as np
import pandas as pd
import astropy.constants as ac
import astropy.units as au

from ..io.read_hst import read_hst
from ..load_sim import LoadSim

class Hst:

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
        try:
            # hot gas mass in Msun
            hst['Mh'] *= u.Msun*vol
        except KeyError:
            hst['Mhot'] *= u.Msun*vol

        try:
            # warm gas mass in Msun
            hst['Mw'] *= u.Msun*vol
        except KeyError:
            hst['Mwarm'] *= u.Msun*vol

        try:
            # intermediate temperature gas in Msun
            hst['Mi'] *= u.Msun*vol
        except KeyError:
            hst['Minter'] *= u.Msun*vol
            
        try:
            # cold gas mass in Msun
            hst['Mc'] *= u.Msun*vol
        except KeyError:
            hst['Mcold'] *= u.Msun*vol

        try:
            # Hot and ionized
            hst['Mhi'] = hst['Mh'] + hst['Mi']
        except KeyError:
            hst['Mhi'] = hst['Mhot'] + hst['Minter']
            
        # Total/hot gas/shell momentum in Msun*km/s
        pr_conv = vol*(u.mass*u.velocity).to('Msun km s-1').value
        hst['pr'] *= pr_conv
        try:
            hst['pr_h'] *= pr_conv
        except KeyError:
            hst['pr_hot'] *= pr_conv
            
        hst['prsh'] *= pr_conv

        # energy in ergs
        E_conv = vol*(u.energy).cgs.value
        hst['Ethm'] *= E_conv
        hst['Ekin'] *= E_conv
        for ph in ('c','u','w','i','h'):
            hst['Ethm_'+ph] *= E_conv
            hst['Ekin_'+ph] *= E_conv

        # Mean cool/heat rates
        hst['cool_rate'] *= vol
        hst['heat_rate'] *= vol
        hst['net_cr'] *= vol

        # SNR velocity in km/s
        # hst['vsnr'] = hst['pr']/(hst['Msh'] + hst['Mi'] + hst['Mh'])
        # SNR radius
        # hst['Rsnr'] = hst['pr']/(hst['Msh'] + hst['Mi'] + hst['Mh'])
        
        # Dimensionless deceleration parameter
        # hst['eta'] = hst['vsnr']*hst['time_code']/hst['Rsnr']

        vol_cgs = vol*u.length.cgs.value**3
        hst['pok_bub'] = (hst['Ethm_i'] + hst['Ethm_h'])/\
            ((hst['Vi'] + hst['Vh'])*vol_cgs)/ac.k_B.cgs.value

        hst['Vbub'] *= vol
        hst['Rbub'] = (3.0*hst['Vbub']/(4.0*np.pi))**(1.0/3.0)
        hst['vrbub'] = np.gradient(hst['Rbub'], hst['time_code'])
        hst['etabub'] = hst['vrbub']*hst['time']/hst['Rbub']

        hst['vrsh'] = np.gradient(hst['Rsh'], hst['time_code'])
        hst['etash'] = hst['vrsh']*hst['time']/hst['Rsh']
        
        # Radiation feedback turned on
        if par['configure']['radps'] == 'ON':
            hst = self._calc_radiation(hst)
        #hst.index = hst['time_code']
        
        self.hst = hst
        
        return hst

    def _calc_radiation(self, hst):
        
        par = self.par
        u = self.u
        domain = self.domain
        # total volume of domain (code unit)
        vol = domain['Lx'].prod()        
        from scipy.integrate import cumtrapz

        # Ionization/dissociation fronts
        hst['RIF'] = hst['RIF']/hst['RIF_vol']*u.pc
        hst['RDF'] = hst['RDF']/hst['RDF_vol']*u.pc
        
        # Total/escaping luminosity in Lsun
        ifreq = dict()
        for f in ('PH','LW','PE','PE_unatt'):
            try:
                ifreq[f] = par['radps']['ifreq_{0:s}'.format(f)]
            except KeyError:
                pass
            
        # Total luminosity in Lsun
        hst['Ltot'] = 0.0
        for k,v in ifreq.items():
            if v >= 0:
                hst['Ltot'] += hst[f'Ltot{v}']

        # Expected radial momentum injection
        if par['radps']['apply_force'] == 1 and par['configure']['ionrad']:
            # Radiation pressure only (in the optically-thick limit)
            hst['pr_inject'] = cumtrapz(hst['Ltot']/ac.c.to(u.velocity).value,
                                        hst['time_code'], initial=0.0)
            hst['pr_inject'] *= vol*(u.mass*u.velocity).value

        hst['Ltot'] *= vol*u.Lsun
        
        # Other luminosity
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
                


                    
        return hst


