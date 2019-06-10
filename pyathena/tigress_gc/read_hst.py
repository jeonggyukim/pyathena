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
        LxLy = domain['Lx'][0]*domain['Lx'][1]

        # Time in code unit
        hst['time_code'] = hst['time']
        # Time in Myr
        hst['time'] *= u.Myr
        # Total gas mass in Msun
        hst['mass'] *= vol*u.Msun
        # Gas surface density in Msun/pc^2
        hst['Sigma_gas'] = hst['mass']/(LxLy*u.pc**2)

        if 'x1Me' in hst:
            mhd = True
        else:
            mhd = False
        
        hst.index = hst['time_code']
        
        self.hst = hst
        
        return hst
