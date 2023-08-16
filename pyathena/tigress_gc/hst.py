# read_hst.py

import os
import os.path as osp
import glob
import xarray as xr
import numpy as np
import pandas as pd
from scipy import integrate
import matplotlib.pyplot as plt
import astropy.units as au
import astropy.constants as ac

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
        # Lz
        Lz = domain['Lx'][2]

        hst = read_hst(self.files['hst'], force_override=force_override)

        h = pd.DataFrame()

        if self.par['configure']['gas'] == 'mhd':
            mhd = True
        else:
            mhd = False

        # Time in code unit
        h['time_code'] = hst['time']
        # Time in Myr
        h['time'] = h['time_code']*u.Myr
        # Time step
        h['dt_code'] = hst['dt']
        h['dt'] = hst['dt']*u.Myr

        # Total gas mass in Msun
        h['mass'] = hst['mass']*vol*u.Msun
        h['mass_sp'] = hst['msp']*vol*u.Msun

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
        h['KE'] = (hst['x1KE'] + hst['x2KE'] + hst['x3KE'])*vol*u.erg
        if mhd:
            h['ME'] = (hst['x1ME'] + hst['x2ME'] + hst['x3ME'])*vol*u.erg

        for ax in ('1','2','3'):
            Ekf = 'x{}KE'.format(ax)
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

        # Star formation rate [Msun/yr]
        h['sfr10'] = hst['sfr10']*LxLy/1e6
        h['sfr40'] = hst['sfr40']*LxLy/1e6
        h['sfr100'] = hst['sfr100']*LxLy/1e6

        h.index = h['time_code']

        self.hst = h

        return h
