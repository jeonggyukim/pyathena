# read_hst.py

import os
import numpy as np
import pandas as pd
from scipy import integrate

from ..io.read_hst import read_hst
from ..load_sim import LoadSim

class Hst:

    @LoadSim.Decorators.check_pickle_hst
    def read_hst(self, savdir=None, force_override=False):
        """Function to read hst and convert quantities to convenient units
        """

        par = self.par
        u = self.u
        domain = self.domain
        
        # volume of resolution element (code unit)
        dvol = domain['dx'].prod()
        # total volume of domain (code unit)
        vol = domain['Lx'].prod()
        # Area of domain (code unit)
        LxLy = domain['Lx'][0]*domain['Lx'][1]

        Omega = self.par['problem']['Omega']
        time_orb = 2*np.pi/Omega*u.Myr # Orbital time in Myr
        try:
            if self.par['configure']['new_cooling'] == 'ON':
                newcool = True
            else:
                newcool = False
        except KeyError:
            newcool = False

        hst = read_hst(self.files['hst'], force_override=force_override)

        
        h = pd.DataFrame()

        if 'x1Me' in hst:
            mhd = True
        else:
            mhd = False

        # Time in code unit
        h['time_code'] = hst['time']
        # Time in Myr
        h['time'] = h['time_code']*u.Myr
        h['time_orb'] = h['time']/time_orb
        # Time step
        hst['dt_code'] = hst['dt']
        hst['dt'] *= u.Myr
        # if par['configure']['new_cooling'] == 'ON' and \
        #    (par['configure']['radps'] == 'ON' or par['configure']['sixray'] == 'ON'):
        #     for c in ('dt_cool_min','dt_xH2_min','dt_xHII_min'):
        #         hst[c] *= u.Myr*vol
        
        # Total gas mass in Msun
        h['mass'] = hst['mass']*vol*u.Msun
        h['mass_sp'] = hst['msp']*vol*u.Msun

        # Total outflow mass
        h['mass_out'] = integrate.cumtrapz(hst['F3_upper'] - hst['F3_lower'], hst['time'], initial=0.0)
        h['mass_out'] = h['mass_out']/(domain['Nx'][2]*domain['dx'][2])*vol*u.Msun
                
        # Mass surface density in Msun/pc^2
        h['Sigma_gas'] = h['mass']/(LxLy*u.pc**2)
        h['Sigma_sp'] = h['mass_sp']/(LxLy*u.pc**2)
        h['Sigma_out'] = h['mass_out']/(LxLy*u.pc**2)

        # Calculate (cumulative) SN ejecta mass
        # JKIM: only from clustered type II(?)
        try:
            sn = read_hst(self.files['sn'], force_override=force_override)
            t_ = np.array(hst['time'])
            Nsn, snbin = np.histogram(sn.time, bins=np.concatenate(([t_[0]], t_)))
            h['mass_snej'] = Nsn.cumsum()*self.par['feedback']['MejII'] # Mass of SN ejecta [Msun]
            h['Sigma_snej'] = h['mass_snej']/(LxLy*u.pc**2)
        except:
            pass
        
        # H mass/surface density in Msun
        h['MH'] = h['mass']/u.muH
        h['Sigma_H'] = h['MH']/(LxLy*u.pc**2)

        # Mass, volume fraction, scale height
        h['H'] = np.sqrt(hst['H2'] / hst['mass'])
        for ph in ['c','u','w','h1','h2']:
            h['mf_{}'.format(ph)] = hst['M{}'.format(ph)]/hst['mass']
            h['vf_{}'.format(ph)] = hst['V{}'.format(ph)]
            h['H_{}'.format(ph)] = \
                np.sqrt(hst['H2{}'.format(ph)] / hst['M{}'.format(ph)])
        #print(h['mf_c'])

        # mf, vf, H of thermally bistable (cold + unstable + warm) medium
        h['mf_2p'] = h['mf_c'] + h['mf_u'] + h['mf_w']
        h['vf_2p'] = h['vf_c'] + h['vf_u'] + h['vf_w']
        h['H_2p'] = np.sqrt((hst['H2c'] + hst['H2u'] + hst['H2w']) / \
                            (hst['Mc'] + hst['Mu'] + hst['Mw']))

        # Kinetic and magnetic energy
        h['KE'] = hst['x1KE'] + hst['x2KE'] + hst['x3KE']
        if mhd:
            h['ME'] = hst['x1ME'] + hst['x2ME'] + hst['x3ME']

        hst['x2KE'] = hst['x2dke']
        for ax in ('1','2','3'):
            Ekf = 'x{}KE'.format(ax)
            if ax == '2':
                Ekf = 'x2dke'
            # Mass weighted velocity dispersion??
            h['v{}'.format(ax)] = np.sqrt(2*hst[Ekf]/hst['mass'])
            if mhd:
                h['vA{}'.format(ax)] = \
                    np.sqrt(2*hst['x{}ME'.format(ax)]/hst['mass'])
            h['v{}_2p'.format(ax)] = \
                np.sqrt(2*hst['x{}KE_2p'.format(ax)]/hst['mass']/h['mf_2p'])
            
        h['cs'] = np.sqrt(hst['P']/hst['mass'])
        h['Pth_mid'] = hst['Pth']*u.pok
        h['Pth_mid_2p'] = hst['Pth_2p']*u.pok/hst['Vmid_2p']
        h['Pturb_mid'] = hst['Pturb']*u.pok
        h['Pturb_mid_2p'] = hst['Pturb_2p']*u.pok/hst['Vmid_2p']

        # Midplane number density
        h['nmid'] = hst['nmid']
        h['nmid_2p'] = hst['nmid_2p']/hst['Vmid_2p']

        # Star formation rate per area [Msun/kpc^2/yr]
        h['sfr10'] = hst['sfr10']
        h['sfr40'] = hst['sfr40']
        h['sfr100'] = hst['sfr100']

        # Cosmic ray ionization rate
        if newcool:
            tdecay = self.par['problem']['CR_tdecay']
            xi_CR_amp = self.par['problem']['xi_CR_amp']
            try:
                Sigma_gas0 = self.par['problem']['Sigma_gas0']
            except KeyError:
                Sigma_gas0 = 13.0
            Sigma_SFR0 = self.par['problem']['Sigma_SFR0']
            Sigma_SFR = self.par['problem']['Sigma_SFR']
            h['tfact'] = np.exp(-8.0*((hst['time'] - 0.5*tdecay)/hst['time'])**2)
            idx = hst['time'] < 0.5*tdecay
            h.loc[idx, 'tfact'] = 1.0
            h['xi_CR0_init'] = xi_CR_amp*2e-16*(Sigma_SFR/Sigma_SFR0)*(Sigma_gas0/h['Sigma_gas'])*h['tfact']
            h['xi_CR0'] = h['xi_CR0_init'] + xi_CR_amp*2e-16*(h['sfr40']/Sigma_SFR0)*(Sigma_gas0/h['Sigma_gas'])

        h.index = h['time_code']
        
        self.hst = h
        
        return h
        
        # # Ionized gas mass in Msun
        # hst['Mion'] *= vol*u.Msun
        # # Collisionally ionized gas (before ray tracing) in Msun
        # hst['Mion_coll'] *= vol*u.Msun
        # # Total photoionization rate [#/sec]
        # hst['Qiphot'] *= vol*(u.length**3).cgs
        # # Total collisional ionization rate [#/sec]
        # hst['Qicoll'] *= vol*(u.length**3).cgs
        # # Total dust absorption rate [#/sec]
        # hst['Qidust'] *= vol*(u.length**3).cgs

    #     # Mass fraction ionized gas
    #     hst['mf_ion'] = hst['Mion']/hst['mass']
    #     hst['mf_ion_coll'] = hst['Mion_coll']/hst['mass']

        # for f in range(self.par['radps']['nfreq']):
        #     # Total luminosity [Lsun]
        #     hst['Ltot_cl{:d}'.format(f)] *= vol*u.Lsun
        #     hst['Ltot_ru{:d}'.format(f)] *= vol*u.Lsun
        #     hst['Ltot{:d}'.format(f)] = \
        #         hst['Ltot_cl{:d}'.format(f)] + hst['Ltot_ru{:d}'.format(f)]
        #     # Total luminosity included in simulation
        #     hst['L_cl{:d}'.format(f)] *= vol*u.Lsun
        #     hst['L_ru{:d}'.format(f)] *= vol*u.Lsun
        #     hst['L{:d}'.format(f)] = \
        #         hst['L_cl{:d}'.format(f)] + hst['L_ru{:d}'.format(f)]
        #     # Luminosity that escaped boundary
        #     hst['Lesc{:d}'.format(f)] *= vol*u.Lsun
        #     # Luminosity lost due to dmax
        #     hst['Llost{:d}'.format(f)] *= vol*u.Lsun
        #     # Escape fraction, lost fraction
        #     # Estimation of true escape fraction estimation (upper bound)
        #     hst['fesc{:d}'.format(f)] = hst['Lesc{:d}'.format(f)] / \
        #                                 hst['L{:d}'.format(f)]
        #     hst['flost{:d}'.format(f)] = hst['Llost{:d}'.format(f)] / \
        #                                  hst['L{:d}'.format(f)]
        #     hst['fesc{:d}_est'.format(f)] = hst['fesc{:d}'.format(f)] + \
        #                                     hst['flost{:d}'.format(f)]
        #     hst['fesc{:d}_cum_est'.format(f)] = \
        #         (hst['Lesc{:d}'.format(f)] + hst['Llost{:d}'.format(f)]).cumsum() / \
        #          hst['L{:d}'.format(f)].cumsum()

        # return hst

    #     # Scale heights of [warm] ionized gas, nesq
    #     # Check if columns exist

    #     # nesq
    #     if 'H2nesq' in hst.columns and 'nesq' in hst.columns:
    #         hst['H_nesq'] = np.sqrt(hst['H2nesq'] / hst['nesq'])
    #         hst.drop(columns=['H2nesq', 'nesq'], inplace=True)

    #     # Warm nesq
    #     if 'H2wnesq' in hst.columns and 'wnesq' in hst.columns:
    #         hst['H_wnesq'] = np.sqrt(hst['H2wnesq'] / hst['wnesq'])
    #         hst.drop(columns=['H2wnesq', 'wnesq'], inplace=True)

    #     # For warm medium, 
    #     # append _ to distinguish from mhd history variable
    #     if 'H2w' in hst.columns and 'massw' in hst.columns:
    #         hst['H_w_'] = np.sqrt(hst['H2w'] / hst['massw'])
    #         hst['Mw_'] = hst['massw']*vol*u.Msun
    #         hst['mf_w_'] = hst['Mw_']/hst['mass']
    #         hst.drop(columns=['H2w', 'massw'], inplace=True)

    #     # Warm ionized
    #     if 'H2wi' in hst.columns and 'masswi' in hst.columns:
    #         hst['H_wi'] = np.sqrt(hst['H2wi'] / hst['masswi'])
    #         hst['Mwion'] = hst['masswi']*vol*u.Msun
    #         hst['mf_wion'] = hst['Mwion']/hst['mass']
    #         hst.drop(columns=['H2wi', 'masswi'], inplace=True)
            
    #     ##########################
    #     # With ionizing radiation
    #     ##########################
    #     if self.par['radps']['nfreq'] == 2 and \
    #        self.par['radps']['nfreq_ion'] == 1:
    #         hnu0 = self.par['radps']['hnu[0]']/u.eV
    #         hnu1 = self.par['radps']['hnu[1]']/u.eV
    #         # Total luminosity
    #         hst['Qitot_cl'] = hst['Ltot_cl0']/u.Lsun/hnu0/u.s
    #         hst['Qitot_ru'] = hst['Ltot_ru0']/u.Lsun/hnu0/u.s
    #         hst['Qitot'] = hst['Qitot_ru'] + hst['Qitot_cl']
    #         # Total Q included as source
    #         hst['Qi_cl'] = hst['L_cl0']/u.Lsun/hnu0/u.s
    #         hst['Qi_ru'] = hst['L_ru0']/u.Lsun/hnu0/u.s
    #         hst['Qi'] = hst['Qi_ru'] + hst['Qi_cl']
    #         hst['Qiesc'] = hst['Lesc0']/u.Lsun/hnu0/u.s
    #         hst['Qilost'] = hst['Llost0']/u.Lsun/hnu0/u.s
    #         hst['Qiesc_est'] = hst['Qilost'] + hst['Qiesc']

    #     else:
    #         self.logger.error('Unrecognized option nfreq={0:d}, nfreq_ion={1:d}'.\
    #                           format(self.par['radps']['nfreq'],
    #                                  self.par['radps']['nfreq_ion']))

    #     # midplane radiation energy density in cgs units
    #     hst['Erad0_mid'] *= u.energy_density
    #     hst['Erad1_mid'] *= u.energy_density


    #     try:
    #         hst.to_pickle(fpkl)
    #     except IOError:
    #         self.logger.warning('[read_hst]: Could not pickle hst to {0:s}.'.format(fpkl))

    #     self.hst = hst
    #     return self.hst

    def read_hst_mhd(self):

        # Read original mhd history dump from /tigress/changgoo
        hst = read_hst('/tigress/changgoo/{0:s}/hst/{0:s}.hst'.\
                       format(self.problem_id))

        u = self.u
        domain = self.par['domain1']
        Lx = domain['x1max'] - domain['x1min']
        Ly = domain['x2max'] - domain['x2min']
        Lz = domain['x3max'] - domain['x3min']
        Nx = domain['Nx1']
        Ny = domain['Nx2']
        Nz = domain['Nx3']
        Ntot = Nx*Ny*Nz
        vol = Lx*Ly*Lz
        LxLy = Lx*Ly
        dz = Lz/Nz
        Omega = self.par['problem']['Omega']
        time_orb = 2*np.pi/Omega*u.Myr # Orbital time in Myr

        if 'x1Me' in hst:
            mhd = True
        else:
            mhd = False

        h = pd.DataFrame()
        h['time_code'] = hst['time']
        h['time'] = h['time_code']*u.Myr # time in Myr
        h['time_orb'] = h['time']/time_orb

        h['mass'] = hst['mass']*u.Msun*vol
        h['Sigma'] = h['mass']/LxLy
        h['mass_sp'] = hst['msp']*u.Msun*vol
        h['Sigma_sp'] = h['mass_sp']/LxLy

        # Mass, volume fraction, scale height
        h['H'] = np.sqrt(hst['H2'] / hst['mass'])
        for ph in ['c','u','w','h1','h2']:
            h['mf_{}'.format(ph)] = hst['M{}'.format(ph)]/hst['mass']
            h['vf_{}'.format(ph)] = hst['V{}'.format(ph)]
            h['H_{}'.format(ph)] = \
                np.sqrt(hst['H2{}'.format(ph)] / hst['M{}'.format(ph)])

        # mf, vf, H of thermally bistable (cold + unstable + warm) medium
        h['mf_2p'] = h['mf_c'] + h['mf_u'] + h['mf_w']
        h['vf_2p'] = h['vf_c'] + h['vf_u'] + h['vf_w']
        h['H_2p'] = np.sqrt((hst['H2c'] + hst['H2u'] + hst['H2w']) / \
                            (hst['Mc'] + hst['Mu'] + hst['Mw']))

        # Kinetic and magnetic energy
        h['KE'] = hst['x1KE'] + hst['x2KE'] + hst['x3KE']
        if mhd:
            h['ME'] = hst['x1ME'] + hst['x2ME'] + hst['x3ME']

        hst['x2KE'] = hst['x2dke']
        for ax in ('1','2','3'):
            Ekf = 'x{}KE'.format(ax)
            if ax == '2':
                Ekf = 'x2dke'
            # Mass weighted velocity dispersion??
            h['v{}'.format(ax)] = np.sqrt(2*hst[Ekf]/hst['mass'])
            if mhd:
                h['vA{}'.format(ax)] = \
                    np.sqrt(2*hst['x{}ME'.format(ax)]/hst['mass'])
            h['v{}_2p'.format(ax)] = \
                np.sqrt(2*hst['x{}KE_2p'.format(ax)]/hst['mass']/h['mf_2p'])
            
        h['cs'] = np.sqrt(hst['P']/hst['mass'])
        h['Pth_mid'] = hst['Pth']*u.pok
        h['Pth_mid_2p'] = hst['Pth_2p']*u.pok/hst['Vmid_2p']
        h['Pturb_mid'] = hst['Pturb']*u.pok
        h['Pturb_mid_2p'] = hst['Pturb_2p']*u.pok/hst['Vmid_2p']

        # Midplane number density
        h['nmid'] = hst['nmid']
        h['nmid_2p'] = hst['nmid_2p']/hst['Vmid_2p']

        # Star formation rate per unit area [Msun/kpc^2/yr]
        h['sfr10']=hst['sfr10']
        h['sfr40']=hst['sfr40']
        h['sfr100']=hst['sfr100']

        h.index = h['time_code']
        #h.index.name = 'index'
        
        self.hst_mhd = h

        return self.hst_mhd
    
        # return pd.read_pickle(
        #     '/tigress/changgoo/{0:s}/hst/{0:s}.hst_cal.p'.format(self.problem_id))
