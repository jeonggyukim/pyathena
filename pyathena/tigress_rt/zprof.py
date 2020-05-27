# zprof.py

import os
import os.path as osp

import xarray as xr
import numpy as np

from ..load_sim import LoadSim
from ..io.read_zprof import read_zprof_all, ReadZprofBase

class Zprof(ReadZprofBase):
    
    @LoadSim.Decorators.check_netcdf_zprof
    def _read_zprof(self, phase='whole', savdir=None, force_override=False):
        """Function to read zprof and convert quantities to convenient units.
        """

        ds = read_zprof_all(osp.dirname(self.files['zprof'][0]),
                            self.problem_id, phase=phase,
                            force_override=False)

        # Divide all variables by total area Lx*Ly
        domain = self.domain
        dxdy = domain['dx'][0]*domain['dx'][1]
        Atot = domain['Lx'][0]*domain['Lx'][1]

        ds = ds/Atot


        # For the moment s3 is assumed to be nH2
        ds['nH2'] = ds.s3

        return ds
