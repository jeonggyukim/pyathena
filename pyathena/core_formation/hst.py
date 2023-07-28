# read_hst.py

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
        h = pd.DataFrame()

        # Time in code unit
        h['time'] = hst['time']
        # Time in unit of free-fall time
        h['time_in_tff'] = h['time']/self.tff0
        # Timestep
        h['dt'] = hst['dt']
        h['dt_in_tff'] = h['dt']/self.tff0

        # Total gas mass
        h['mass'] = hst['mass']

        # Kinetic and gravitational energies
        h['KE'] = (hst['1KE'] + hst['2KE'] + hst['3KE'])
        h['gravE'] = hst['gravE']

        # Mass weighted velocity dispersions
        for name, ax in zip(('x', 'y', 'z'), ('1', '2', '3')):
            KE = hst['{}KE'.format(ax)]
            h['v{}'.format(name)] = np.sqrt(2*KE/hst['mass'])

        # 3D Mach number
        h['Mach'] = np.sqrt(h['vx']**2 + h['vy']**2 + h['vz']**2) / self.cs

        h.set_index('time', inplace=True)
        self.hst = h
        return h
