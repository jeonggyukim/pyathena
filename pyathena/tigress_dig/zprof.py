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
                            self.problem_id,
                            phase=phase, force_override=False)

        # Delete zeroth (t=0.0) snapshot.
        # Since coordinates cannot be changed,
        # we have to create a new dataset from raw data.
        if ds.time[0] == 0.0:
            da_ = ds.to_array()
            dat = np.delete(da_.data, 0, axis=-1)
            data_vars = dict()
            i = 0
            for v in ds.variables:
                if v == 'z' or v == 'time':
                    continue
                else:
                    #print(dat[i,...].shape)
                    data_vars[v] = (('z', 'time'), dat[i, ...])
                    i += 1

            ds = xr.Dataset(data_vars, coords=dict(z=ds.z, time=ds.time[1:]))

        # Divide all variables by total area Lx*Ly
        domain = self.domain
        dxdy = domain['dx'][0]*domain['dx'][1]
        Atot = domain['Lx'][0]*domain['Lx'][1]

        ds = ds/Atot

        if self.par['configure']['species_HI'] == 'ON':
            # For the moment s1 is assumed to be nH0
            ds['nH0'] = ds.s1
            # Volume filling factor of ionized gas
            if 'xn' in ds:
                ds['xi'] = ds.A - ds.xn
            else:
                ds['xi'] = ds.A - ds.xHI
            # Electron number density averaged over Atot
            ds['ne'] = ds.d - ds.s1
            # Electron number density averaged over Atot
            ds['nebar'] = ds.ne/ds.xi

        # Rename time to time_code and use physical time in Myr as dimension
        ds = ds.rename(dict(time='time_code'))
        ds = ds.assign_coords(time=ds.time_code*self.u.Myr)
        ds = ds.assign_coords(z_kpc=ds.z*self.u.kpc)
        ds = ds.swap_dims(dict(time_code='time'))
        
        # self._set_attrs(ds.domain)
        
        return ds

    # def _set_attrs(self, zp):

    #     d1 = dict()
    #     d2 = dict()

    #     # Should be defined somewhere else, e.g., tigress..? # JGKIM
    #     d1['A'] = 'Area filling fraction'
    #     d2['A'] = '\int \Theta_{\rm ph} dxdy'
    #     d1['xn'] = 'Neutral fraction'
    #     d2['xn'] = '\int n_{\mathrm H^0}/n_{\mathrm H} dxdy'
        
    #     for v in zp.variables:
    #         if v in d1.keys() and v in d2.keys():
    #         #zp.attrs[v] = r'\texttt{{{0:s}}}:'.format(v) 
    #             zp.attrs[v] = r'\text{{{0:s}}}:'.format(d1[v]) + d2[v]

    # def print_zprof_attrs(self, zp):
        
    #     # This function should also be defined in somewhere else # JGKIM
    #     from IPython.display import display, Math #, Latex

    #     display(Math(zp.attrs['xn']))
    #     display(Math(zp.attrs['A']))
