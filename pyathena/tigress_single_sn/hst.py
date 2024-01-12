# hst.py

import os
import numpy as np
import pandas as pd

from ..io.read_hst import read_hst
from ..load_sim import LoadSim

class Hst:

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

        # Time in code unit
        hst['time_code'] = hst['time']
        # Time in Myr
        hst['time'] *= u.Myr
        # Total gas mass in Msun
        hst['mass'] *= vol*u.Msun

        # Shell formation time in Myr (Eq.7 in Kim & Ostriker 2015)
        tsf = 4.4e-2*self.par['problem']['n0']**-0.55

        hst['time_code'] = hst['time']
        hst['time'] = hst['time_code']*u.Myr # time in Myr
        hst['time_tsf'] = hst['time']/tsf
        hst['tsf'] = tsf

        hst['Rsh'] = hst['Rsh']/hst['Msh'] # Mass weighted SNR position in pc

        hst['Msh'] *= u.Msun*vol # shell mass in Msun
        hst['Mhot'] *= u.Msun*vol # shell mass in Msun
        hst['Mwarm'] *= u.Msun*vol # shell mass in Msun
        hst['Minter'] *= u.Msun*vol # shell mass in Msun
        hst['Mcold'] *= u.Msun*vol # shell mass in Msun

        hst['rmom_bub'] *= vol*(u.mass*u.velocity).value
        hst['rmom_hot'] *= vol*(u.mass*u.velocity).value
        hst['rmom_shell'] *= vol*(u.mass*u.velocity).value

        hst.index = hst['time_code']

        self.hst = hst

        return hst
