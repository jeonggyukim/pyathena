# read_hst.py

import os
import numpy as np
import pandas as pd

from ..io.read_hst import read_hst
from ..load_sim import LoadSim

class ReadHst:

    @LoadSim.Decorators.check_pickle_hst
    def read_hst(self, savdir=None, force_override=False):
        """Function to read hst and convert quantities to convenient units
        """

        hst = read_hst(self.files['hst'], force_override=force_override)
    
        u = self.u
        domain = self.domain

        # volume of resolution element (code unit)
        dvol = domain['dx'].prod()
        # total volume of domain (code unit)
        vol = domain['Lx'].prod()        
        # Area of domain (code unit)
        Lx = domain['Lx'][0]
        Ly = domain['Lx'][1]
        Lz = domain['Lx'][2]

        hst['time_code'] = hst['time']
        hst['time'] *= u.Myr
        hst['dt'] *= u.Myr
        hst['mass'] *= vol*u.Msun
        hst['Mh2'] *= vol*u.Msun
        hst['Mh1'] *= vol*u.Msun
        hst['Mw'] *= vol*u.Msun
        hst['Mu'] *= vol*u.Msun
        hst['Mc'] *= vol*u.Msun
        hst['msp'] *= vol*u.Msun
        hst['msp_left'] *= vol*u.Msun
        hst['F1_lower'] *= (-1.*Ly*Lz*u.mass_flux*u.length**2).to('Msun/yr').value
        hst['F2_lower'] *= (-1.*Lx*Lz*u.mass_flux*u.length**2).to('Msun/yr').value
        hst['F3_lower'] *= (-1.*Lx*Ly*u.mass_flux*u.length**2).to('Msun/yr').value
        hst['F1_upper'] *= (Ly*Lz*u.mass_flux*u.length**2).to('Msun/yr').value
        hst['F2_upper'] *= (Lx*Lz*u.mass_flux*u.length**2).to('Msun/yr').value
        hst['F3_upper'] *= (Lx*Ly*u.mass_flux*u.length**2).to('Msun/yr').value
        # star formation rate in Msun/yr
        hst['sfr10'] *= (Lx*Ly/1e6)
        hst['sfr40'] *= (Lx*Ly/1e6)
        hst['sfr100'] *= (Lx*Ly/1e6)
        # Gas surface density in Msun/pc^2
        hst['Sigma_gas'] = hst['mass']/(Lx*Ly*u.pc**2)

        if 'x1Me' in hst:
            mhd = True
        else:
            mhd = False
        
        hst.index = hst['time_code']
        
        self.hst = hst
        
        return hst
