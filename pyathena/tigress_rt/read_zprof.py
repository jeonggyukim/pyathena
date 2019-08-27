# read_zprof.py

import os
import os.path as osp

import xarray as xr
import numpy as np

from ..load_sim import LoadSim
from ..io.read_zprof import read_zprof_all, ReadZprofBase

class ReadZprof(ReadZprofBase):
    
    @LoadSim.Decorators.check_netcdf_zprof
    def _read_zprof(self, phase='whole', savdir=None, force_override=False):
        """Function to read zprof and convert quantities to convenient units.
        """

        ds = read_zprof_all(osp.dirname(self.files['zprof'][0]),
                            self.problem_id, phase=phase,
                            force_override=False)

        return ds
