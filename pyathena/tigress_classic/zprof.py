import os.path as osp

from ..load_sim import LoadSim
from ..io.read_zprof import read_zprof_all, ReadZprofBase

class Zprof(ReadZprofBase):
    @LoadSim.Decorators.check_netcdf_zprof
    def _read_zprof(self, phase='whole', savdir=None, force_override=False):
        """Function to read zprof and convert quantities to convenient units.
        """

        ds = read_zprof_all(osp.dirname(self.files['zprof'][0]),
                            self.problem_id, phase=phase, savdir=savdir,
                            force_override=force_override)
        u = self.u
        # Divide all variables by total area Lx*Ly
        domain = self.domain
        dxdy = domain['dx'][0]*domain['dx'][1]
        LxLy = domain['Lx'][0]*domain['Lx'][1]

        ds = ds/LxLy

        # Rename time to time_code and use physical time in Myr as dimension
        ds = ds.rename(dict(time='time_code'))
        ds = ds.assign_coords(time=ds.time_code*self.u.Myr)
        ds = ds.assign_coords(z_kpc=ds.z*self.u.kpc)
        ds = ds.swap_dims(dict(time_code='time'))

        return ds