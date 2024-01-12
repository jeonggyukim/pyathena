# starpar.py

import numpy as np
import pandas as pd

from ..load_sim import LoadSim
from ..util.mass_to_lum import mass_to_lum

class StarPar():

    @LoadSim.Decorators.check_pickle
    def read_starpar_all(self, prefix='starpar_all',
                         savdir=None, force_override=False):
        rr = dict()
        for i in self.nums_starpar:
            print(i, end=' ')
            r = self.read_starpar(num=i, force_override=False)
            if i == 0:
                for k in r.keys():
                    rr[k] = []

            for k in r.keys():
                try:
                    rr[k].append(r[k].value.item())
                except:
                    rr[k].append(r[k])

        rr = pd.DataFrame(rr)
        return rr

    @LoadSim.Decorators.check_pickle
    def read_starpar(self, num, savdir=None, force_override=False):

        sp = self.load_starpar_vtk(num)
        if sp.empty:
            return None

        u = self.u
        domain = self.domain
        par = self.par

        try:
            agemax = par['radps']['agemax_rad']
        except KeyError:
            self.logger.warning('agemax_rad was not found and set to 40 Myr.')
            agemax = 20.0

        sp['age'] *= u.Myr
        sp['mage'] *= u.Myr
        sp['mass'] *= u.Msun

        # Select non-runaway starpar particles with mass-weighted age < agemax_rad
        isrc = np.logical_and(sp['mage'] < agemax,
                              sp['mass'] != 0.0)

        if np.sum(isrc) == 0:
            return None

        # Center of mass, luminosity, standard deviation of z-position
        # Summary
        r = dict()
        r['sp'] = sp
        r['time'] = sp.time
        r['nstars'] = sp.nstars
        r['isrc'] = isrc
        r['nsrc'] = np.sum(isrc)

        return r
